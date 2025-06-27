# --- Файл: summarizer.py ---

import logging
from typing import Dict, Optional, List, Any

from PySide6.QtCore import QObject, Signal, QThread, Slot

import google.generativeai as genai
from google.api_core import exceptions as google_exceptions

from github_manager import GitHubManager
from github.Repository import Repository

# Настраиваем логгер для этого модуля
logger = logging.getLogger(__name__)

# Промпт для создания саммари файла (без изменений)
SUMMARIZATION_PROMPT_TEMPLATE = """
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

# --- НОВЫЙ КЛАСС ДЛЯ РАЗБИЕНИЯ ТЕКСТА ---
class SimpleTextSplitter:
    """
    Простая реализация рекурсивного сплиттера текста на фрагменты (чанки).
    Предназначен для разбиения как обычного текста, так и кода.
    """
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 150):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # Приоритет разделителей: от более крупных структур к более мелким
        self._separators = ["\n\n", "\n", ". ", " ", ""]

    def split_text(self, text: str) -> List[str]:
        """
        Разбивает большой текст на чанки заданного размера.

        Args:
            text: Исходный текст для разбиения.

        Returns:
            Список текстовых фрагментов (чанков).
        """
        final_chunks = []
        # Начинаем с одного большого фрагмента
        chunks = [text]
        
        for sep in self._separators:
            if not chunks:
                break
                
            new_chunks = []
            for chunk in chunks:
                if len(chunk) > self.chunk_size:
                    # Если разделитель не пустой, используем его
                    if sep:
                        splits = chunk.split(sep)
                    else:
                        # Если разделитель пустой, просто режем по размеру
                        splits = [chunk[i:i + self.chunk_size] for i in range(0, len(chunk), self.chunk_size)]
                    
                    # Объединяем мелкие фрагменты обратно в чанки нужного размера
                    merged_splits = self._merge_splits(splits, sep)
                    new_chunks.extend(merged_splits)
                else:
                    new_chunks.append(chunk)
            chunks = new_chunks
        
        final_chunks.extend(chunks)
        return final_chunks

    def _merge_splits(self, splits: List[str], separator: str) -> List[str]:
        """Вспомогательный метод для объединения мелких сплитов в чанки."""
        docs = []
        current_doc = []
        total = 0
        for s in splits:
            # Добавляем длину сплита и разделителя
            length = len(s) + (len(separator) if separator else 0)
            if total + length > self.chunk_size:
                # Если добавление нового сплита превысит размер чанка
                if total > 0:
                    docs.append(separator.join(current_doc))
                
                # Обработка перекрытия (overlap)
                while total > self.chunk_overlap:
                    total -= len(current_doc[0]) + (len(separator) if separator else 0)
                    current_doc = current_doc[1:]
            
            current_doc.append(s)
            total += length

        if current_doc:
            docs.append(separator.join(current_doc))
        
        return docs


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
    file_summarized = Signal(str, str) # (путь, текст саммари) - для UI
    documents_for_db_ready = Signal(list, list) # (texts, metadatas) - для VectorDB
    error_occurred = Signal(str)
    finished = Signal()

    def __init__(self,
                 github_manager: GitHubManager,
                 repo: Repository,
                 branch_name: str,
                 files_to_summarize: Dict[str, int], # {path: size}
                 gemini_api_key: str,
                 model_name: str,
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

    def cancel(self):
        """Запрашивает отмену операции."""
        logger.info("Получен запрос на отмену анализа.")
        self._is_cancelled = True

    def run(self):
        """Основной метод потока, выполняющий анализ."""
        logger.info(f"Запуск потока анализа для {len(self.files_to_summarize)} файлов.")
        
        try:
            genai.configure(api_key=self.gemini_api_key)
            self.generative_model = genai.GenerativeModel(self.model_name)
        except Exception as e:
            error_msg = f"Ошибка инициализации модели Gemini в SummarizerWorker: {e}"
            logger.error(error_msg)
            self.error_occurred.emit(error_msg)
            self.finished.emit()
            return

        processed_count = 0
        total_count = len(self.files_to_summarize)
        
        for file_path in self.files_to_summarize.keys():
            if self._is_cancelled:
                logger.warning("Операция анализа была отменена пользователем.")
                break

            logger.debug(f"Анализ файла: {file_path}")
            
            # 1. Получаем содержимое файла
            content = self.github_manager.get_file_content(self.repo, file_path, self.branch_name)
            
            if content is None:
                logger.warning(f"Пропуск анализа для файла '{file_path}', так как не удалось получить его содержимое.")
                processed_count += 1
                self.progress_updated.emit(processed_count, total_count)
                continue

            documents_to_add = []
            metadatas_to_add = []

            # 2. Генерируем саммари (если файл не пустой)
            summary_text = "(Файл пуст)"
            if content.strip():
                prompt = SUMMARIZATION_PROMPT_TEMPLATE.format(file_path=file_path, file_content=content)
                try:
                    response = self.generative_model.generate_content(prompt)
                    summary_text = response.text.strip()
                    logger.info(f"Успешно создано саммари для '{file_path}'.")
                except google_exceptions.ResourceExhausted as e:
                    error_msg = f"Исчерпаны квоты API Gemini при саммаризации '{file_path}'. Прерывание. Ошибка: {e}"
                    logger.error(error_msg)
                    self.error_occurred.emit(error_msg)
                    break 
                except Exception as e:
                    summary_text = f"(Ошибка саммаризации: {type(e).__name__})"
                    error_msg = f"Ошибка API Gemini при саммаризации файла '{file_path}': {type(e).__name__} - {e}"
                    logger.error(error_msg)
                    self.error_occurred.emit(f"Ошибка саммаризации для '{file_path}', файл пропущен в саммари.")
            
            # Отправляем саммари в UI и добавляем его в пакет для БД
            self.file_summarized.emit(file_path, summary_text)
            documents_to_add.append(summary_text)
            metadatas_to_add.append({'file_path': file_path, 'type': 'summary'})

            # 3. Разбиваем содержимое на чанки
            if content.strip():
                chunks = self.text_splitter.split_text(content)
                logger.debug(f"Файл '{file_path}' разбит на {len(chunks)} чанков.")
                for i, chunk_text in enumerate(chunks):
                    documents_to_add.append(chunk_text)
                    metadatas_to_add.append({'file_path': file_path, 'type': 'chunk', 'chunk_num': i + 1})

            # 4. Отправляем готовый пакет документов и метаданных в основной поток
            if documents_to_add:
                self.documents_for_db_ready.emit(documents_to_add, metadatas_to_add)

            processed_count += 1
            self.progress_updated.emit(processed_count, total_count)
            
            # Небольшая задержка, чтобы не превысить лимиты API (requests per minute)
            self.msleep(500) 

        logger.info("Поток анализа завершил свою работу.")
        self.finished.emit()