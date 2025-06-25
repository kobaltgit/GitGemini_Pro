# --- Файл: summarizer.py ---

import logging
from typing import Dict, Optional, Any

from PySide6.QtCore import QObject, Signal, QThread, Slot

import google.generativeai as genai
from google.api_core import exceptions as google_exceptions

from github_manager import GitHubManager
from github.Repository import Repository

# Настраиваем логгер для этого модуля
logger = logging.getLogger(__name__)

# Промпт для создания саммари файла
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

class SummarizerWorker(QThread):
    """
    Рабочий поток, который выполняет саммаризацию файлов репозитория,
    чтобы не блокировать основной поток GUI.
    """
    # Сигнал: (обработанные файлы, всего файлов)
    progress_updated = Signal(int, int)
    # Сигнал: (путь к файлу, текст саммари)
    file_summarized = Signal(str, str)
    # Сигнал: (сообщение об ошибке)
    error_occurred = Signal(str)
    # Сигнал о завершении работы
    finished = Signal()

    def __init__(self,
                 github_manager: GitHubManager,
                 repo: Repository,
                 files_to_summarize: Dict[str, int], # {path: size}
                 gemini_api_key: str,
                 model_name: str,
                 parent: Optional[QObject] = None):
        super().__init__(parent)
        self.github_manager = github_manager
        self.repo = repo
        self.files_to_summarize = files_to_summarize
        self.gemini_api_key = gemini_api_key
        self.model_name = model_name
        self._is_cancelled = False
        self.generative_model: Optional[genai.GenerativeModel] = None

    def cancel(self):
        """Запрашивает отмену операции саммаризации."""
        logger.info("Получен запрос на отмену саммаризации.")
        self._is_cancelled = True

    def run(self):
        """Основной метод потока, выполняющий саммаризацию."""
        logger.info(f"Запуск потока саммаризации для {len(self.files_to_summarize)} файлов.")
        
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
                logger.warning("Операция саммаризации была отменена пользователем.")
                break

            logger.debug(f"Саммаризация файла: {file_path}")
            
            # 1. Получаем содержимое файла
            content = self.github_manager.get_file_content(self.repo, file_path)
            
            if content is None:
                logger.warning(f"Пропуск саммаризации для файла '{file_path}', так как не удалось получить его содержимое.")
                processed_count += 1
                self.progress_updated.emit(processed_count, total_count)
                continue

            if not content.strip():
                 logger.info(f"Файл '{file_path}' пуст, пропускаем саммаризацию.")
                 self.file_summarized.emit(file_path, "(Файл пуст)")
                 processed_count += 1
                 self.progress_updated.emit(processed_count, total_count)
                 continue

            # 2. Формируем промпт и отправляем запрос к Gemini
            prompt = SUMMARIZATION_PROMPT_TEMPLATE.format(file_path=file_path, file_content=content)
            
            try:
                response = self.generative_model.generate_content(prompt)
                summary_text = response.text.strip()
                self.file_summarized.emit(file_path, summary_text)
                logger.info(f"Успешно создано саммари для '{file_path}'.")

            except google_exceptions.ResourceExhausted as e:
                error_msg = f"Исчерпаны квоты API Gemini при саммаризации '{file_path}'. Прерывание. Ошибка: {e}"
                logger.error(error_msg)
                self.error_occurred.emit(error_msg)
                break # Прерываем цикл при исчерпании квот
            except Exception as e:
                error_msg = f"Ошибка API Gemini при саммаризации файла '{file_path}': {type(e).__name__} - {e}"
                logger.error(error_msg)
                # Не прерываемся, просто пропускаем этот файл
                self.error_occurred.emit(f"Ошибка саммаризации для '{file_path}', файл пропущен.")
            
            processed_count += 1
            self.progress_updated.emit(processed_count, total_count)
            
            # Небольшая задержка, чтобы не превысить лимиты API (requests per minute)
            self.msleep(500) 

        logger.info("Поток саммаризации завершил свою работу.")
        self.finished.emit()