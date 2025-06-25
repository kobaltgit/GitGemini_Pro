# --- Файл: chat_model.py ---

import os
import re
import logging
from typing import Optional, List, Dict, Any, Tuple

from PySide6.QtCore import QObject, Signal, Slot, QThread

import db_manager
from dotenv import load_dotenv, set_key, find_dotenv
import google.generativeai as genai
import google.generativeai.types as genai_types
from google.api_core import exceptions as google_exceptions

# Наши новые модули
from github_manager import GitHubManager
from summarizer import SummarizerWorker

logger = logging.getLogger(__name__)

CONTEXT_WINDOW_LIMIT = 1048576  # Глобальная константа

# --- Класс GeminiWorker (без изменений, отвечает за один запрос к API) ---
class GeminiWorker(QObject):
    response_received = Signal(str)
    error_occurred = Signal(str)
    finished_work = Signal()

    def __init__(self, model: genai.GenerativeModel, prompt_parts: List[Dict[str, Any]], max_output_tokens: int):
        super().__init__()
        self.model = model
        self.prompt_parts = prompt_parts
        self.max_output_tokens_config = max_output_tokens
        self._is_cancelled = False

    def cancel(self):
        self._is_cancelled = True
        logger.info("GeminiWorker: Получен запрос на отмену.")

    @Slot()
    def run(self):
        try:
            if self._is_cancelled:
                logger.info("GeminiWorker: Запрос отменен до старта.")
                self.finished_work.emit()
                return

            generation_config = genai_types.GenerationConfig(max_output_tokens=self.max_output_tokens_config)
            logger.info(f"GeminiWorker: Отправка запроса (max_tokens_response={self.max_output_tokens_config}).")
            
            response = self.model.generate_content(
                self.prompt_parts,
                generation_config=generation_config,
                request_options={"timeout": 180},
            )

            if self._is_cancelled:
                logger.info("GeminiWorker: Запрос отменен после получения ответа.")
                self.finished_work.emit()
                return

            if hasattr(response, "text") and response.text:
                self.response_received.emit(response.text)
            else:
                # Обработка блокировок и пустых ответов (упрощенная)
                reason = "Неизвестно"
                if response.prompt_feedback and response.prompt_feedback.block_reason:
                    reason = response.prompt_feedback.block_reason.name
                self.error_occurred.emit(f"Генерация прервана. Причина: {reason}")

        except Exception as e:
            if not self._is_cancelled:
                err_msg = f"Ошибка API Gemini: {type(e).__name__} - {e}"
                logger.error(err_msg)
                self.error_occurred.emit(err_msg)
        finally:
            logger.info("GeminiWorker: Завершение работы run().")
            self.finished_work.emit()


# --- Основной класс Модели (полностью переработан) ---
class ChatModel(QObject):
    # --- Сигналы ---
    # Статус ключей и токенов
    geminiApiKeyStatusChanged = Signal(bool, str)
    githubTokenStatusChanged = Signal(bool, str)
    
    # Модели Gemini
    availableModelsChanged = Signal(list)

    # Анализ репозитория (Саммаризация)
    analysisStarted = Signal()
    analysisProgressUpdated = Signal(int, int, str) # (обработано, всего, текущий файл)
    analysisFinished = Signal()
    analysisError = Signal(str)

    # История чата и сессия
    historyChanged = Signal(list)
    sessionStateChanged = Signal(str, bool) # (filepath, is_dirty)
    sessionLoaded = Signal()
    sessionError = Signal(str)

    # Взаимодействие с API
    apiRequestStarted = Signal()
    apiResponseReceived = Signal(str) # Финальный ответ для пользователя
    apiIntermediateStep = Signal(str) # Сообщение о промежуточном этапе (RAG)
    apiErrorOccurred = Signal(str)
    apiRequestFinished = Signal()

    # Общие
    statusMessage = Signal(str, int)
    tokenCountUpdated = Signal(int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        # --- Состояние аутентификации ---
        self._dotenv_path: Optional[str] = find_dotenv()
        self._gemini_api_key: Optional[str] = None
        self._github_token: Optional[str] = None
        self._gemini_api_key_loaded: bool = False
        self._github_token_loaded: bool = False

        # --- Состояние сессии ---
        self._repo_url: Optional[str] = None
        self._chat_history: List[Dict[str, Any]] = []
        self._file_summaries: Dict[str, str] = {} # {file_path: summary}
        self._current_session_filepath: Optional[str] = None
        self._is_dirty: bool = False

        # --- Настройки ---
        self._model_name: str = "gemini-1.5-flash-latest"
        self._available_models: List[str] = [self._model_name]
        self._max_output_tokens: int = 65536
        self._extensions: Tuple[str, ...] = ( ".py", ".txt", ".md", ".json", ".html", ".css", ".js", ".yaml", ".yml", ".pdf", ".docx")
        self._instructions: str = ""

        # --- Состояние токенов ---
        self._current_prompt_tokens: int = 0
        self._token_limit_for_display: int = CONTEXT_WINDOW_LIMIT

        # --- Воркеры и менеджеры ---
        self._gemini_model: Optional[genai.GenerativeModel] = None
        self._gemini_worker: Optional[GeminiWorker] = None
        self._summarizer_worker: Optional[SummarizerWorker] = None
        self._github_manager: Optional[GitHubManager] = None
        self._current_request_thread: Optional[QThread] = None

        # --- Инициализация ---
        self._load_credentials()

    # --- Управление Ключами и Токенами ---
    def _load_credentials(self):
        load_dotenv(dotenv_path=self._dotenv_path, override=True)
        # Gemini API Key
        self._gemini_api_key = os.getenv("GEMINI_API_KEY")
        self._gemini_api_key_loaded = bool(self._gemini_api_key)
        gemini_status = "Загружен" if self._gemini_api_key_loaded else "Не найден!"
        logger.info(f"Ключ Gemini API: {gemini_status.lower()}.")
        self.geminiApiKeyStatusChanged.emit(self._gemini_api_key_loaded, f"Ключ API: {gemini_status}")
        if self._gemini_api_key_loaded:
            self._initialize_gemini()

        # GitHub Token
        self._github_token = os.getenv("GITHUB_TOKEN")
        self._github_token_loaded = bool(self._github_token)
        github_status = "Загружен" if self._github_token_loaded else "Не найден!"
        logger.info(f"Токен GitHub: {github_status.lower()}.")
        self.githubTokenStatusChanged.emit(self._github_token_loaded, f"Токен GitHub: {github_status}")
        if self._github_token_loaded:
            self._initialize_github_manager()

    def _save_credential(self, key_name: str, value: str) -> bool:
        if not value:
            self.statusMessage.emit(f"{key_name} не может быть пустым.", 3000)
            return False
        try:
            path = self._dotenv_path or os.path.join(os.getcwd(), ".env")
            if not os.path.exists(path):
                logger.info(f"Файл .env не найден, будет создан: {path}")
            if set_key(path, key_name, value, quote_mode="always"):
                self._dotenv_path = path
                self._load_credentials()
                self.statusMessage.emit(f"{key_name} успешно сохранен и загружен.", 5000)
                return True
            else:
                self.statusMessage.emit(f"Ошибка сохранения {key_name}.", 5000)
                return False
        except Exception as e:
            logger.error(f"Ошибка при сохранении {key_name}: {e}", exc_info=True)
            self.statusMessage.emit(f"Ошибка сохранения: {e}", 0)
            return False

    def save_gemini_api_key(self, key: str) -> bool:
        return self._save_credential("GEMINI_API_KEY", key)

    def save_github_token(self, token: str) -> bool:
        return self._save_credential("GITHUB_TOKEN", token)

    # --- Инициализация сервисов ---
    def _initialize_gemini(self):
        if not self._gemini_api_key:
            logger.warning("Попытка инициализации Gemini без ключа API.")
            return
        try:
            genai.configure(api_key=self._gemini_api_key)
            self._gemini_model = genai.GenerativeModel(self._model_name)
            logger.info(f"Gemini успешно инициализирован с моделью {self._model_name}.")
            self._fetch_available_models()
        except Exception as e:
            logger.error(f"Ошибка инициализации Gemini: {e}")
            self.statusMessage.emit(f"Ошибка Gemini: {e}", 0)
            self._gemini_model = None

    def _initialize_github_manager(self):
        if not self._github_token:
            logger.warning("Попытка инициализации GitHub Manager без токена.")
            return
        self._github_manager = GitHubManager(self._github_token)
        if not self._github_manager.is_authenticated():
            self.githubTokenStatusChanged.emit(False, "Токен GitHub: Ошибка!")
            self.statusMessage.emit("Неверный токен GitHub.", 5000)
        else:
            self.githubTokenStatusChanged.emit(True, f"Токен GitHub: Загружен ({self._github_manager.rate_limit_info})")

    @Slot()
    def _fetch_available_models(self):
        logger.info("Запрос списка доступных моделей Gemini...")
        try:
            models = genai.list_models()
            # Фильтруем модели, оставляя те, что поддерживают генерацию контента
            self._available_models = sorted([
                m.name.replace("models/", "") for m in models 
                if 'generateContent' in m.supported_generation_methods
            ])
            logger.info(f"Получено {len(self._available_models)} моделей.")
            self.availableModelsChanged.emit(self._available_models)
        except Exception as e:
            logger.error(f"Не удалось получить список моделей Gemini: {e}")
            self.statusMessage.emit("Не удалось загрузить список моделей.", 5000)

    # --- Анализ репозитория (Саммаризация) ---
    def start_repository_analysis(self):
        if not self._github_manager or not self._repo_url:
            self.analysisError.emit("Токен GitHub или URL репозитория не заданы.")
            return
        if not self._gemini_model:
            self.analysisError.emit("Модель Gemini не инициализирована. Проверьте ключ API.")
            return
        if self._summarizer_worker and self._summarizer_worker.isRunning():
            self.statusMessage.emit("Анализ уже запущен.", 3000)
            return

        repo = self._github_manager.get_repo(self._repo_url)
        if not repo:
            self.analysisError.emit(f"Не удалось получить доступ к репозиторию: {self._repo_url}")
            return

        files_to_process, skipped = self._github_manager.get_repo_file_tree(repo, self._extensions)
        if not files_to_process:
            self.analysisError.emit("В репозитории не найдено файлов с указанными расширениями.")
            return

        self._file_summaries = {} # Очищаем старые саммари перед новым анализом
        self._mark_dirty()
        self.analysisStarted.emit()
        self.statusMessage.emit(f"Начат анализ {len(files_to_process)} файлов...", 0)

        self._summarizer_worker = SummarizerWorker(
            github_manager=self._github_manager,
            repo=repo,
            files_to_summarize=files_to_process,
            gemini_api_key=self._gemini_api_key,
            model_name=self._model_name
        )
        self._summarizer_worker.file_summarized.connect(self._on_file_summarized)
        self._summarizer_worker.progress_updated.connect(self._on_analysis_progress)
        self._summarizer_worker.error_occurred.connect(self.analysisError)
        self._summarizer_worker.finished.connect(self._on_analysis_finished)
        self._summarizer_worker.start()

    def cancel_analysis(self):
        if self._summarizer_worker and self._summarizer_worker.isRunning():
            self._summarizer_worker.cancel()
            self.statusMessage.emit("Отмена анализа...", 3000)
            
    @Slot(str, str)
    def _on_file_summarized(self, file_path: str, summary: str):
        self._file_summaries[file_path] = summary
        self._mark_dirty()

    @Slot(int, int)
    def _on_analysis_progress(self, processed: int, total: int):
        # file_path может быть пустым, если мы не хотим его показывать
        self.analysisProgressUpdated.emit(processed, total, "") 
        self.statusMessage.emit(f"Анализ... {processed}/{total}", 0)

    @Slot()
    def _on_analysis_finished(self):
        self.analysisFinished.emit()
        self.statusMessage.emit("Анализ репозитория завершен.", 5000)
        self._summarizer_worker = None

    # --- RAG и основной запрос к API ---
    def send_request_to_api(self, user_input: str):
        # 1. Проверки
        if not self._is_ready_for_request():
            return
        
        user_input_stripped = user_input.strip()
        if not user_input_stripped:
            self.statusMessage.emit("Введите ваш запрос.", 3000); return
            
        self.apiRequestStarted.emit()
        self.add_user_message(user_input_stripped)

        # 2. Этап 1: Retrieval (Поиск релевантных файлов)
        self.apiIntermediateStep.emit("Этап 1: Поиск релевантных файлов по саммари...")
        
        summaries_str = "\n".join([f"- `{path}`: {summary}" for path, summary in self._file_summaries.items()])
        retrieval_prompt = (
            f"Проанализируй вопрос пользователя и список файлов с их описаниями.\n"
            f"Выдай список только тех путей к файлам (каждый с новой строки), которые наиболее релевантны для ответа на вопрос. "
            f"Если ни один файл не релевантен, верни пустой ответ.\n\n"
            f"ВОПРОС ПОЛЬЗОВАТЕЛЯ:\n{user_input_stripped}\n\n"
            f"ФАЙЛЫ И ИХ ОПИСАНИЯ:\n{summaries_str}"
        )

        try:
            retrieval_response = self._gemini_model.generate_content(retrieval_prompt)
            retrieval_text = ""
            try:
                # Безопасно пытаемся получить текст
                retrieval_text = retrieval_response.text
            except ValueError:
                # Это происходит, когда модель возвращает пустой ответ (e.g., finish_reason=STOP)
                logger.warning("Этап 1: Модель вернула пустой ответ. Считаем, что релевантных файлов не найдено.")
                # retrieval_text остается пустой строкой, что корректно

            # Улучшенный парсинг и строгая валидация
            unvalidated_files = [line.strip().replace("`", "") for line in retrieval_text.splitlines() if line.strip()]
            
            relevant_files = []
            for file_path in unvalidated_files:
                if file_path in self._file_summaries:
                    relevant_files.append(file_path)
                    logger.debug(f"Этап 1: Файл '{file_path}' прошел валидацию.")
                else:
                    logger.warning(f"Этап 1: Модель вернула невалидный путь, который будет проигнорирован: '{file_path}'")

            if not relevant_files:
                self.apiIntermediateStep.emit("Релевантных файлов не найдено. Ответ будет основан только на истории чата.")
                final_context = "Контекст из файлов не был найден."
            else:
                self.apiIntermediateStep.emit(f"Этап 2: Загрузка содержимого {len(relevant_files)} релевантных файлов...")
                final_context = self._build_final_context(relevant_files)

        except Exception as e:
            err_msg = f"Ошибка на этапе 1 (Retrieval): {e}"
            logger.error(err_msg, exc_info=True)
            self.apiErrorOccurred.emit(err_msg)
            self.apiRequestFinished.emit()
            return
            
        # 3. Этап 2: Augmentation & Generation (Обогащение и Генерация)
        final_prompt_parts = self._build_final_prompt(final_context)
        
        logger.info("Отправка финального запроса к API...")

        # Запускаем воркер для генерации финального ответа
        self._gemini_worker = GeminiWorker(
            model=self._gemini_model,
            prompt_parts=final_prompt_parts,
            max_output_tokens=self._max_output_tokens
        )
        
        thread = QThread()
        self._gemini_worker.moveToThread(thread)
        
        # Подключаем сигналы от воркера к слотам модели
        self._gemini_worker.response_received.connect(self._handle_final_api_response)
        self._gemini_worker.error_occurred.connect(self._handle_final_api_error)
        
        # Жизненный цикл потока и воркера
        thread.started.connect(self._gemini_worker.run)
        self._gemini_worker.finished_work.connect(thread.quit)
        self._gemini_worker.finished_work.connect(self._handle_worker_finished)
        thread.finished.connect(self._gemini_worker.deleteLater)
        thread.finished.connect(self._cleanup_request_thread)
        
        thread.start()
        
        # Сохраняем ссылку на поток, чтобы он не был удален сборщиком мусора
        self._current_request_thread = thread
        
    def _is_ready_for_request(self) -> bool:
        # Правильная проверка: проверяем, запущен ли ПОТОК, а не воркер
        if self._current_request_thread and self._current_request_thread.isRunning():
            self.statusMessage.emit("Дождитесь завершения предыдущего запроса.", 3000); return False
        if not self._gemini_api_key_loaded:
            self.apiErrorOccurred.emit("Ключ Gemini API не загружен."); return False
        if not self._github_token_loaded:
            self.apiErrorOccurred.emit("Токен GitHub не загружен."); return False
        if not self._repo_url:
            self.apiErrorOccurred.emit("URL репозитория не указан."); return False
        if not self._file_summaries:
            self.apiErrorOccurred.emit("Репозиторий не проанализирован. Нажмите 'Анализировать'."); return False
        return True

    def _build_final_context(self, file_paths: List[str]) -> str:
        """Собирает полное содержимое релевантных файлов в единый контекст."""
        context_parts = []
        repo = self._github_manager.get_repo(self._repo_url)
        if not repo: return "Ошибка: не удалось получить доступ к репозиторию для сборки контекста."

        for path in file_paths:
            content = self._github_manager.get_file_content(repo, path)
            if content is not None:
                context_parts.append(f"--- Файл: {path} ---\n{content}\n" + "-" * 20 + "\n")
            else:
                context_parts.append(f"--- Файл: {path} (Ошибка чтения) ---\n\n")
        return "".join(context_parts)

    def _build_final_prompt(self, context_str: str) -> List[Dict[str, Any]]:
        """Собирает финальный промпт для генерации ответа."""
        prompt_parts: List[Dict[str, Any]] = []
        
        # 1. Системные инструкции
        if self._instructions:
            prompt_parts.extend([
                {"role": "user", "parts": [f"**Системные инструкции:**\n{self._instructions}"]},
                {"role": "model", "parts": ["OK. Инструкции приняты."]}
            ])

        # 2. Контекст из файлов
        prompt_parts.extend([
            {"role": "user", "parts": [f"**Контекст из релевантных файлов проекта:**\n{context_str}"]},
            {"role": "model", "parts": ["OK. Контекст проекта получен."]}
        ])
        
        # 3. История чата (упрощенный вариант, без отбора по токенам)
        for message in self._chat_history:
            if not message.get("excluded", False):
                prompt_parts.append({"role": message["role"], "parts": message["parts"]})
        
        return prompt_parts
    
    @Slot(str)
    def _handle_final_api_response(self, response_text: str):
        """Обрабатывает успешный финальный ответ от API."""
        logger.info("Получен финальный ответ от API.")
        self.add_model_response(response_text)
        self.apiResponseReceived.emit(response_text)

    @Slot(str)
    def _handle_final_api_error(self, error_message: str):
        """Обрабатывает ошибку от воркера."""
        logger.error(f"Получена ошибка от GeminiWorker: {error_message}")
        self.apiErrorOccurred.emit(error_message)

    @Slot()
    def _handle_worker_finished(self):
        """Срабатывает по завершению работы воркера (успешно или с ошибкой)."""
        logger.info("GeminiWorker завершил работу, отправка apiRequestFinished.")
        self.apiRequestFinished.emit()

    @Slot()
    def _cleanup_request_thread(self):
        """Безопасно очищает ссылки на воркер и поток ПОСЛЕ их завершения."""
        logger.info("Очистка ресурсов потока GeminiWorker.")
        self._gemini_worker = None
        self._current_request_thread = None

    def _update_token_count(self):
        """Подсчитывает токены в текущем промпте и испускает сигнал."""
        if not self._gemini_model:
            self.tokenCountUpdated.emit(0, self._token_limit_for_display)
            return

        # Собираем части промпта, которые идут в модель (без RAG-контекста)
        prompt_parts_for_counting: List[Dict[str, Any]] = []
        if self._instructions:
            prompt_parts_for_counting.extend([
                {"role": "user", "parts": [f"**Системные инструкции:**\n{self._instructions}"]},
                {"role": "model", "parts": ["OK. Инструкции приняты."]}
            ])
        
        for message in self._chat_history:
            if not message.get("excluded", False):
                prompt_parts_for_counting.append({"role": message["role"], "parts": message["parts"]})

        # Выполняем подсчет
        try:
            if prompt_parts_for_counting:
                result = self._gemini_model.count_tokens(prompt_parts_for_counting)
                self._current_prompt_tokens = result.total_tokens
            else:
                self._current_prompt_tokens = 0
            
            logger.debug(f"Подсчет токенов: {self._current_prompt_tokens}")
            self.tokenCountUpdated.emit(self._current_prompt_tokens, self._token_limit_for_display)

        except Exception as e:
            logger.error(f"Ошибка при подсчете токенов: {e}")
            self.tokenCountUpdated.emit(0, self._token_limit_for_display)

    # --- Управление состоянием (геттеры/сеттеры) ---
    def set_repo_url(self, url: str):
        if url != self._repo_url:
            self._repo_url = url
            self._file_summaries = {} # Сброс саммари при смене репозитория
            self._mark_dirty()
    def get_repo_url(self) -> Optional[str]: return self._repo_url
    def get_available_models(self) -> List[str]: return self._available_models
    def set_model_name(self, name: str):
        # Убедимся, что имя не пустое и действительно изменилось
        if name and name != self._model_name:
            logger.info(f"Смена модели с '{self._model_name}' на '{name}'")
            self._model_name = name
            self._mark_dirty()
            # Самое главное: пересоздаем объект модели с новым именем
            self._initialize_gemini()
            self._update_token_count()
    def get_model_name(self) -> str: return self._model_name
    def set_max_tokens(self, tokens: int):
        if tokens != self._max_output_tokens: self._max_output_tokens = tokens; self._mark_dirty()
    def get_max_tokens(self) -> int: return self._max_output_tokens
    def set_extensions(self, ext_tuple: Tuple[str, ...]):
        if ext_tuple != self._extensions: self._extensions = ext_tuple; self._mark_dirty()
    def get_extensions(self) -> Tuple[str, ...]: return self._extensions
    def set_instructions(self, text: str):
        if text != self._instructions: self._instructions = text; self._mark_dirty()
        self._update_token_count()
    def get_instructions(self) -> str: return self._instructions
    
    # --- Управление историей чата (аналогично старой версии) ---
    def get_chat_history(self) -> List[Dict[str, Any]]: return self._chat_history[:]
    def add_user_message(self, text: str):
        if not text: return
        self._chat_history.append({"role": "user", "parts": [text], "excluded": False})
        self._mark_dirty()
        self.historyChanged.emit(self.get_chat_history())
        self._update_token_count()
    def add_model_response(self, text: str):
        self._chat_history.append({"role": "model", "parts": [text or ""], "excluded": False})
        self._mark_dirty()
        self.historyChanged.emit(self.get_chat_history())
        self._update_token_count()
    def toggle_api_exclusion(self, index: int):
        if 0 <= index < len(self._chat_history):
            self._chat_history[index]["excluded"] = not self._chat_history[index].get("excluded", False)
            self._mark_dirty()
            self.historyChanged.emit(self.get_chat_history())
            self._update_token_count()

    # --- Управление Сессиями (обновлено) ---
    def new_session(self):
        self._repo_url = None
        self._chat_history = []
        self._file_summaries = {}
        self._current_session_filepath = None
        self._extensions = (".py", ".txt", ".md", ".json", ".html", ".css", ".js", ".yaml", ".yml")
        self._model_name = self._available_models[0] if self._available_models else "gemini-1.5-flash-latest"
        self._max_output_tokens = 65536
        self._instructions = ""
        self._is_dirty = False
        self.sessionLoaded.emit() # Чтобы UI обновился
        self.statusMessage.emit("Новая сессия создана.", 3000)
        self._update_token_count()

    def load_session(self, filepath: str):
        logger.info(f"Загрузка сессии: {filepath}")
        loaded_data = db_manager.load_session_data(filepath)
        if loaded_data:
            meta, msgs, summaries = loaded_data
            self._repo_url = meta.get("repo_url")
            self._chat_history = msgs
            self._file_summaries = summaries
            self._model_name = meta.get("model_name", self._available_models[0] if self._available_models else "gemini-1.5-flash-latest")
            self._max_output_tokens = meta.get("max_output_tokens", 65536)
            ext_str = meta.get("extensions", ".py .txt")
            self._extensions = tuple(p.strip() for p in ext_str.split())
            self._instructions = meta.get("instructions", "")
            self._current_session_filepath = filepath
            self._is_dirty = False
            self.sessionLoaded.emit()
            self.statusMessage.emit(f"Сессия '{os.path.basename(filepath)}' загружена.", 5000)
            self._update_token_count()
        else:
            self.sessionError.emit(f"Не удалось загрузить сессию: {filepath}")

    def save_session(self, filepath: Optional[str] = None) -> Tuple[bool, Optional[str]]:
        save_path = filepath or self._current_session_filepath
        if not save_path: return False, None
        
        metadata = {
            "repo_url": self._repo_url, 
            "model_name": self._model_name, 
            "max_output_tokens": self._max_output_tokens,
            "extensions": " ".join(self._extensions), 
            "instructions": self._instructions
        }
        
        if db_manager.save_session_data(save_path, metadata, self._chat_history, self._file_summaries):
            self._current_session_filepath = save_path
            self._is_dirty = False
            self.sessionStateChanged.emit(save_path, False)
            self.statusMessage.emit(f"Сессия сохранена.", 5000)
            return True, save_path
        else:
            self.sessionError.emit(f"Не удалось сохранить сессию: {save_path}")
            return False, None

    # --- Состояние "грязи" ---
    def _mark_dirty(self):
        if not self._is_dirty:
            self._is_dirty = True
            self.sessionStateChanged.emit(self._current_session_filepath, True)
    def is_dirty(self) -> bool: return self._is_dirty
    def get_current_session_filepath(self) -> Optional[str]: return self._current_session_filepath