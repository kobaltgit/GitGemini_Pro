# --- Файл: summarizer.py ---

import logging
from typing import Dict, Optional, List, Any

from PySide6.QtCore import QObject, Signal, QThread, Slot, QLocale

import google.generativeai as genai
from google.api_core import exceptions as google_exceptions

from github_manager import GitHubManager
from github.Repository import Repository

# Настраиваем логгер для этого модуля
logger = logging.getLogger(__name__)

# Промпт для создания саммари файла. Теперь в двух версиях.
SUMMARIZATION_PROMPT_RU = """
Проанализируй содержимое этого файла:

--- НАЧАЛО ФАЙЛА: {file_path} ---
{file_content}
--- КОНЕЦ ФАЙЛА ---

Создай для него краткое, но емкое саммари (2-4 предложения).
В саммари обязательно отрази:
1. Основное назначение файла (что он делает, за что отвечает).
2. Ключевые классы, функции или компоненты, которые в нем определены.
3. Его основные зависимости от других частей проекта, если они очевидны из кода.

Ответ должен быть только текстом саммари, без лишних фраз и вступлений.
"""

SUMMARIZATION_PROMPT_EN = """
Analyze the contents of this file:

--- START OF FILE: {file_path} ---
{file_content}
--- END OF FILE ---

Create a brief but comprehensive summary (2-4 sentences) for it.
In the summary, be sure to reflect:
1. The main purpose of the file (what it does, what it is responsible for).
2. Key classes, functions, or components defined in it.
3. Its main dependencies on other parts of the project, if they are evident from the code.

The response should be only the summary text, without any extra phrases or introductions.
"""


# --- НОВЫЙ КЛАСС ДЛЯ РАЗБИЕНИЯ ТЕКСТА ---
class SimpleTextSplitter:
    """
    Реализация рекурсивного сплиттера текста на фрагменты (чанки).
    Предназначен для разбиения как обычного текста, так и кода,
    пытаясь сохранить структурные единицы.
    """
    def __init__(self, chunk_size: int = 700, chunk_overlap: int = 150): # Немного уменьшены размеры для лучшей фокусировки кода
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # Порядок важен: сначала пытаемся разбивать по крупным, осмысленным разделителям
        # Для кода приоритет отдается определениям функций и классов.
        self._separators = [
            "\ndef ",    # Python: Определение функции
            "\nclass ",   # Python: Определение класса
            "\n\n",      # Пустая строка (часто отделяет логические блоки/абзацы)
            "\n",        # Одиночный перевод строки (строка за строкой)
            " ",         # Пробел (граница слова)
            "",          # Пустая строка (последний запасной вариант: символ за символом)
        ]
        self._length_function = len # Используем длину символов

    def split_text(self, text: str) -> List[str]:
        """
        Разбивает текст на чанки, стараясь сохранить структурные единицы.
        Использует рекурсивный подход с учетом перекрытия.
        """
        final_chunks: List[str] = []
        self._recursive_split(text, self._separators, final_chunks)
        return final_chunks

    def _recursive_split(self, text_to_split: str, separators: List[str], final_chunks: List[str]):
        """
        Внутренняя рекурсивная функция для разбиения текста.
        """
        # Базовый случай 1: Нет разделителей или текст пуст
        if not separators or not text_to_split:
            if text_to_split:
                # Если текст все еще существует и нет разделителей,
                # добавляем его как один чанк или принудительно разбиваем по символам, если он слишком большой.
                if len(text_to_split) > self.chunk_size:
                    # Принудительно разбиваем на чанки размера chunk_size,
                    # сдвигаясь на (chunk_size - chunk_overlap)
                    for i in range(0, len(text_to_split), self.chunk_size - self.chunk_overlap):
                        chunk = text_to_split[i:i + self.chunk_size]
                        final_chunks.append(chunk)
                else:
                    final_chunks.append(text_to_split)
            return

        current_separator = separators[0]
        remaining_separators = separators[1:]

        # Разбиваем текст по текущему разделителю
        # Используем rstrip(' ') чтобы не было лишних пробелов перед def/class
        if current_separator:
            parts = text_to_split.split(current_separator)
        else: # Запасной вариант: разбиение посимвольно (для пустой строки-разделителя)
            parts = list(text_to_split)

        # Комбинируем части обратно в чанки, учитывая chunk_size и chunk_overlap
        current_chunk_elements: List[str] = []
        current_chunk_length = 0

        for i, part in enumerate(parts):
            # Вычисляем "эффективную" длину этой части (включая разделитель, если она не первая в новом чанке)
            effective_part_length = len(part)
            if current_chunk_elements and current_separator: # Добавляем длину разделителя, если это не первая часть чанка
                effective_part_length += len(current_separator)

            # Если добавление этой части сделает текущий чанк слишком большим
            if current_chunk_length + effective_part_length > self.chunk_size:
                # Если у нас уже есть что-то в текущем чанке, завершаем его
                if current_chunk_elements:
                    chunk = current_separator.join(current_chunk_elements)
                    # Рекурсивно разбиваем этот чанк, если он все еще слишком большой с помощью следующего разделителя
                    if len(chunk) > self.chunk_size:
                        self._recursive_split(chunk, remaining_separators, final_chunks)
                    else:
                        final_chunks.append(chunk)

                # Начинаем новый чанк с перекрытием
                overlap_content = self._get_overlap_content(current_separator.join(current_chunk_elements), self.chunk_overlap)
                current_chunk_elements = [overlap_content] if overlap_content else []
                current_chunk_length = len(overlap_content) if overlap_content else 0
            
            # Добавляем текущую часть к новому или существующему чанку
            current_chunk_elements.append(part)
            current_chunk_length += effective_part_length

        # Обрабатываем последний оставшийся чанк
        if current_chunk_elements:
            chunk = current_separator.join(current_chunk_elements)
            if len(chunk) > self.chunk_size:
                self._recursive_split(chunk, remaining_separators, final_chunks)
            else:
                final_chunks.append(chunk)

    def _get_overlap_content(self, text: str, overlap: int) -> str:
        """Вспомогательная функция для получения содержимого для перекрытия."""
        # Для перекрытия лучше брать с конца, чтобы сохранить контекст
        return text[-overlap:] if len(text) > overlap else text


# --- ПЕРЕРАБОТАННЫЙ WORKER ---
class SummarizerWorker(QThread):
    """
    Рабочий поток, который выполняет анализ файлов репозитория:
    1. Создает саммари с помощью Gemini.
    2. Разбивает содержимое файла на чанки.
    3. Отправляет готовые документы и метаданные для добавления в векторную БД.
    """
    # Сигналы
    progress_updated = Signal(int, int)
    file_summarized = Signal(str, str)
    documents_for_db_ready = Signal(list, list)
    error_occurred = Signal(str)
    finished = Signal()

    def __init__(self,
                 github_manager: GitHubManager,
                 repo: Repository,
                 branch_name: str,
                 files_to_summarize: Dict[str, int],
                 gemini_api_key: str,
                 model_name: str,
                 app_lang: str = 'en', # Добавляем app_lang
                 parent: Optional[QObject] = None):
        super().__init__(parent)
        self.github_manager = github_manager
        self.repo = repo
        self.branch_name = branch_name
        self.files_to_summarize = files_to_summarize
        self.gemini_api_key = gemini_api_key
        self.model_name = model_name
        self._is_cancelled = False
        self.generative_model: Optional[genai.GenerativeModel] = None
        self.text_splitter = SimpleTextSplitter(chunk_size=1000, chunk_overlap=150)

        # Выбираем шаблон промпта в зависимости от переданного app_lang
        if app_lang == 'ru':
            self.summarization_prompt_template = SUMMARIZATION_PROMPT_RU
        else:
            self.summarization_prompt_template = SUMMARIZATION_PROMPT_EN

    def cancel(self):
        """Запрашивает отмену операции."""
        logger.info(self.tr("Получен запрос на отмену анализа."))
        self._is_cancelled = True

    def run(self):
        """Основной метод потока, выполняющий анализ."""
        logger.info(self.tr("Запуск потока анализа для {0} файлов.").format(len(self.files_to_summarize)))
        
        try:
            genai.configure(api_key=self.gemini_api_key)
            self.generative_model = genai.GenerativeModel(self.model_name)
        except Exception as e:
            error_msg = self.tr("Ошибка инициализации модели Gemini в SummarizerWorker: {0}").format(e)
            logger.error(error_msg)
            self.error_occurred.emit(error_msg)
            self.finished.emit()
            return

        processed_count = 0
        total_count = len(self.files_to_summarize)
        
        for file_path in self.files_to_summarize.keys():
            if self._is_cancelled:
                logger.warning(self.tr("Операция анализа была отменена пользователем."))
                break

            logger.debug(f"Анализ файла: {file_path}")
            
            content = self.github_manager.get_file_content(self.repo, file_path, self.branch_name)
            
            if content is None:
                logger.warning(self.tr("Пропуск анализа для файла '{0}', так как не удалось получить его содержимое.").format(file_path))
                processed_count += 1
                self.progress_updated.emit(processed_count, total_count)
                continue

            documents_to_add = []
            metadatas_to_add = []

            summary_text = self.tr("(Файл пуст)")
            if content.strip():
                prompt = self.summarization_prompt_template.format(file_path=file_path, file_content=content)
                try:
                    response = self.generative_model.generate_content(prompt)
                    summary_text = response.text.strip()
                    logger.info(self.tr("Успешно создано саммари для '{0}'.").format(file_path))
                except google_exceptions.ResourceExhausted as e:
                    error_msg = self.tr("Исчерпаны квоты API Gemini при саммаризации '{0}'. Прерывание. Ошибка: {1}").format(file_path, e)
                    logger.error(error_msg)
                    self.error_occurred.emit(error_msg)
                    break 
                except Exception as e:
                    summary_text = self.tr("(Ошибка саммаризации: {0})").format(type(e).__name__)
                    error_msg = self.tr("Ошибка API Gemini при саммаризации файла '{0}': {1} - {2}").format(file_path, type(e).__name__, e)
                    logger.error(error_msg)
                    self.error_occurred.emit(self.tr("Ошибка саммаризации для '{0}', файл пропущен в саммари.").format(file_path))
            
            self.file_summarized.emit(file_path, summary_text)
            documents_to_add.append(summary_text)
            metadatas_to_add.append({'file_path': file_path, 'type': 'summary'})

            if content.strip():
                chunks = self.text_splitter.split_text(content)
                logger.debug(f"Файл '{file_path}' разбит на {len(chunks)} чанков.")
                for i, chunk_text in enumerate(chunks):
                    documents_to_add.append(chunk_text)
                    metadatas_to_add.append({'file_path': file_path, 'type': 'chunk', 'chunk_num': i + 1})

            if documents_to_add:
                self.documents_for_db_ready.emit(documents_to_add, metadatas_to_add)

            processed_count += 1
            self.progress_updated.emit(processed_count, total_count)
            
            self.msleep(500) 

        logger.info(self.tr("Поток анализа завершил свою работу."))
        self.finished.emit()