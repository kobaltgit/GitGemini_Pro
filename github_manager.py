# --- Файл: github_manager.py ---

import logging
import base64
import re
from typing import Optional, Dict, Tuple, List, Set

from github import Github, UnknownObjectException, BadCredentialsException, RateLimitExceededException
from github.Repository import Repository
from github.ContentFile import ContentFile
from github.GitTreeElement import GitTreeElement

# Настраиваем логгер для этого модуля
logger = logging.getLogger(__name__)

# Папки, которые стандартно игнорируются при анализе репозитория
DEFAULT_IGNORED_DIRS = {
    "venv", ".venv", "__pycache__", ".git", ".vscode", ".idea", 
    "node_modules", "build", "dist", "target", "out", "bin", "obj",
    "docs", "examples", "tests", "test", "samples"
}

class GitHubManager:
    """
    Класс для инкапсуляции логики взаимодействия с GitHub API.
    Отвечает за аутентификацию, получение информации о репозитории и чтение файлов.
    """

    def __init__(self, token: Optional[str]):
        """
        Инициализирует менеджер с помощью Personal Access Token (PAT).

        Args:
            token: Персональный токен доступа GitHub.
        """
        self.token = token
        self.gh: Optional[Github] = None
        self.rate_limit_info: str = "Неизвестно"
        
        if self.token:
            try:
                self.gh = Github(self.token)
                # Проверим соединение и получим информацию о лимитах
                rate_limit = self.gh.get_rate_limit()
                self.rate_limit_info = f"Осталось {rate_limit.core.remaining}/{rate_limit.core.limit}"
                logger.info(f"GitHubManager инициализирован успешно. Лимит запросов: {self.rate_limit_info}")
            except BadCredentialsException:
                logger.error("Ошибка аутентификации GitHub: Неверный токен (BadCredentialsException).")
                self.gh = None
            except Exception as e:
                logger.error(f"Неожиданная ошибка при инициализации GitHub-клиента: {e}")
                self.gh = None
        else:
            logger.warning("GitHubManager инициализирован без токена. Доступ будет только к публичным репозиториям.")
            self.gh = Github() # Инициализация без токена для публичного доступа

    def is_authenticated(self) -> bool:
        """Проверяет, аутентифицирован ли клиент."""
        return self.gh is not None and self.token is not None

    @staticmethod
    def _parse_repo_url(repo_url: str) -> Optional[Tuple[str, str]]:
        """
        Извлекает 'владелец/имя_репозитория' из полного URL GitHub.
        
        Args:
            repo_url: URL репозитория (например, https://github.com/user/repo).
        
        Returns:
            Кортеж (owner, repo_name) или None, если URL некорректен.
        """
        # Паттерн для извлечения 'owner/repo' из различных форматов URL
        pattern = r"(?:https?://)?(?:www\.)?github\.com/([\w\.\-]+)/([\w\.\-]+)"
        match = re.search(pattern, repo_url)
        if match:
            owner, repo_name = match.groups()
            logger.debug(f"URL '{repo_url}' успешно распарсен как '{owner}/{repo_name}'")
            return owner, repo_name
        logger.warning(f"Не удалось распарсить URL репозитория: '{repo_url}'")
        return None

    def get_repo(self, repo_url: str) -> Optional[Repository]:
        """
        Получает объект репозитория по его URL.

        Args:
            repo_url: Полный URL репозитория.

        Returns:
            Объект `Repository` из PyGithub или None в случае ошибки.
        """
        if not self.gh:
            logger.error("Невозможно получить репозиторий: клиент GitHub не инициализирован.")
            return None

        repo_identifier = self._parse_repo_url(repo_url)
        if not repo_identifier:
            return None

        try:
            repo = self.gh.get_repo(f"{repo_identifier[0]}/{repo_identifier[1]}")
            logger.info(f"Успешно получен доступ к репозиторию: {repo.full_name}")
            return repo
        except UnknownObjectException:
            logger.error(f"Репозиторий '{repo_identifier[0]}/{repo_identifier[1]}' не найден или является приватным без доступа.")
            return None
        except BadCredentialsException:
            logger.error("Ошибка аутентификации при доступе к репозиторию. Проверьте ваш токен.")
            return None
        except RateLimitExceededException:
            logger.error("Превышен лимит запросов к GitHub API. Попробуйте позже.")
            return None
        except Exception as e:
            logger.error(f"Неожиданная ошибка при получении репозитория: {e}")
            return None

    def get_repo_file_tree(
        self, 
        repo: Repository, 
        extensions: Tuple[str, ...],
        ignored_dirs: Set[str] = DEFAULT_IGNORED_DIRS,
        max_file_size_kb: int = 512
    ) -> Tuple[Dict[str, int], List[str]]:
        """
        Получает плоский список путей к файлам в репозитории, отфильтрованный по расширениям и размеру.

        Args:
            repo: Объект репозитория.
            extensions: Кортеж разрешенных расширений (например, ('.py', '.md')).
            ignored_dirs: Множество имен папок для игнорирования.
            max_file_size_kb: Максимальный размер файла в килобайтах.

        Returns:
            Кортеж, где:
            - Первый элемент: Словарь {путь_к_файлу: размер_в_байтах}.
            - Второй элемент: Список строк с информацией о пропущенных файлах.
        """
        logger.info(f"Начинается анализ дерева файлов для репозитория '{repo.full_name}'...")
        try:
            default_branch = repo.get_branch(repo.default_branch)
            tree = repo.get_git_tree(default_branch.commit.sha, recursive=True)
            logger.info(f"Получено дерево файлов, {len(tree.tree)} элементов.")
        except Exception as e:
            logger.error(f"Не удалось получить дерево файлов для репозитория '{repo.full_name}': {e}")
            return {}, [f"Ошибка: Не удалось получить дерево файлов: {e}"]

        filtered_files: Dict[str, int] = {}
        skipped_info: List[str] = []
        max_size_bytes = max_file_size_kb * 1024

        for element in tree.tree:
            if element.type == "blob":  # 'blob' означает файл
                path_parts = element.path.split('/')
                
                # Пропускаем игнорируемые директории
                if any(part in ignored_dirs for part in path_parts):
                    continue

                # Пропускаем файлы без нужного расширения
                if not element.path.endswith(extensions):
                    continue
                
                # Пропускаем слишком большие файлы
                if element.size > max_size_bytes:
                    skipped_info.append(f"Пропущен (размер > {max_file_size_kb}KB): {element.path}")
                    continue

                filtered_files[element.path] = element.size

        logger.info(f"Анализ дерева завершен. Найдено подходящих файлов: {len(filtered_files)}. Пропущено: {len(skipped_info)}.")
        return filtered_files, skipped_info

    def get_file_content(self, repo: Repository, file_path: str) -> Optional[str]:
        """
        Получает содержимое одного файла из репозитория.

        Args:
            repo: Объект репозитория.
            file_path: Путь к файлу внутри репозитория.

        Returns:
            Содержимое файла в виде строки или None в случае ошибки.
        """
        if not file_path:
             logger.warning("get_file_content вызван с пустым путем. Пропуск.")
             return None
             
        logger.debug(f"Запрос содержимого файла: {file_path}")
        try:
            content_file = repo.get_contents(file_path)

            # Проверка, что мы получили файл, а не список (директорию)
            if isinstance(content_file, list):
                logger.warning(f"Путь '{file_path}' указывает на директорию, а не на файл. Пропуск.")
                return None

            if content_file.encoding == "base64" and content_file.content:
                decoded_content = base64.b64decode(content_file.content).decode('utf-8', errors='ignore')
                return decoded_content
            else:
                logger.warning(f"Файл '{file_path}' пуст или имеет неизвестную кодировку: {content_file.encoding}")
                return "" # Возвращаем пустую строку для пустых файлов
        except UnknownObjectException:
            logger.error(f"Файл '{file_path}' не найден в репозитории.")
            return None
        except Exception as e:
            # GitHub API может вернуть ошибку, если файл слишком большой
            logger.error(f"Не удалось получить содержимое файла '{file_path}': {e}")
            return None