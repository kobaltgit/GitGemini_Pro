# --- Файл: db_manager.py ---

import sqlite3
import os
import datetime
import logging
from typing import Optional, Dict, List, Tuple, Any

logger = logging.getLogger(__name__)

SESSION_EXTENSION = ".gpcs"

# --- ОБНОВЛЕННАЯ СХЕМА БАЗЫ ДАННЫХ ---
DATABASE_SCHEMA = """
CREATE TABLE IF NOT EXISTS metadata (
    id INTEGER PRIMARY KEY DEFAULT 1,
    repo_url TEXT,
    repo_branch TEXT,
    -- НОВОЕ ПОЛЕ для хранения пути к векторной базе данных
    vector_db_path TEXT,
    model_name TEXT,
    max_output_tokens INTEGER,
    extensions TEXT,
    instructions TEXT,
    created_at TIMESTAMP,
    last_saved_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    role TEXT NOT NULL CHECK(role IN ('user', 'model')),
    content TEXT NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    order_index INTEGER NOT NULL,
    excluded_from_api BOOLEAN NOT NULL DEFAULT 0
);

-- Таблица для саммари остается для быстрого отображения в UI
CREATE TABLE IF NOT EXISTS file_summaries (
    file_path TEXT PRIMARY KEY NOT NULL,
    summary TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_messages_order ON messages (order_index);
"""


def dict_factory(cursor, row):
    """Фабрика для преобразования строк sqlite в словари."""
    fields = [column[0] for column in cursor.description]
    return {key: value for key, value in zip(fields, row)}


def _get_connection(filepath: str) -> Optional[sqlite3.Connection]:
    """Устанавливает соединение с БД сессии и настраивает его."""
    try:
        conn = sqlite3.connect(filepath, timeout=10)
        conn.row_factory = dict_factory
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn
    except sqlite3.Error as e:
        logger.error(f"Не удалось установить соединение с базой данных '{filepath}': {e}")
        return None


def init_session_db(filepath: str) -> bool:
    """
    Создает или обновляет файл БД сессии и инициализирует таблицы.
    Возвращает True в случае успеха, False при ошибке.
    """
    if not filepath.endswith(SESSION_EXTENSION):
        logger.error(f"Ошибка: Файл должен иметь расширение {SESSION_EXTENSION}, а не '{filepath}'")
        return False
    try:
        logger.info(f"Инициализация/проверка БД сессии: {filepath}")
        dir_name = os.path.dirname(filepath)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
            
        with _get_connection(filepath) as conn:
            if conn:
                conn.executescript(DATABASE_SCHEMA)
                _update_db_schema(conn)
            else:
                return False
        logger.debug(f"БД сессии успешно инициализирована/проверена.")
        return True
    except sqlite3.Error as e:
        logger.error(f"Ошибка SQLite при инициализации БД {filepath}: {e}")
        return False
    except OSError as e:
        logger.error(f"Ошибка файловой системы при создании/доступе к {filepath}: {e}")
        return False

def _update_db_schema(conn: sqlite3.Connection):
    """
    Пытается обновить схему старой базы данных, добавляя недостающие таблицы/колонки.
    """
    columns_to_add = [
        ("metadata", "repo_branch", "TEXT"),
        ("metadata", "vector_db_path", "TEXT"),
        ("messages", "excluded_from_api", "BOOLEAN NOT NULL DEFAULT 0")
    ]

    for table, column, col_type in columns_to_add:
        try:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type};")
            logger.info(f"Схема обновлена: добавлена колонка {table}.{column}")
        except sqlite3.OperationalError:
            pass # Колонка уже существует

    try:
        # Для обратной совместимости, если старое поле project_path существует
        conn.execute("ALTER TABLE metadata RENAME COLUMN project_path TO repo_url;")
        logger.info("Схема обновлена: metadata.project_path -> repo_url")
    except sqlite3.OperationalError:
        pass # Колонка уже переименована или ее не было

    try:
        conn.execute("CREATE TABLE IF NOT EXISTS file_summaries (file_path TEXT PRIMARY KEY NOT NULL, summary TEXT NOT NULL);")
    except sqlite3.OperationalError:
        pass


def load_session_data(
    filepath: str,
) -> Optional[Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, str]]]:
    """
    Загружает метаданные, сообщения и саммари из файла сессии.
    Возвращает кортеж (metadata, messages, summaries) или None при ошибке.
    """
    if not os.path.exists(filepath):
        logger.error(f"Файл сессии не найден: {filepath}")
        return None
        
    if not init_session_db(filepath):
        logger.error(f"Не удалось инициализировать/обновить файл сессии '{filepath}' перед загрузкой.")
        return None

    try:
        logger.info(f"Загрузка данных сессии из: {filepath}")
        with _get_connection(filepath) as conn:
            if not conn: return None
            
            metadata_cursor = conn.execute("SELECT * FROM metadata WHERE id = 1")
            metadata = metadata_cursor.fetchone() or {}

            messages_cursor = conn.execute("SELECT role, content, excluded_from_api FROM messages ORDER BY order_index ASC")
            messages_list = [
                {
                    "role": row["role"],
                    "parts": [row["content"]],
                    "excluded": bool(row.get("excluded_from_api", False))
                }
                for row in messages_cursor.fetchall()
            ]

            summaries_cursor = conn.execute("SELECT file_path, summary FROM file_summaries")
            summaries_dict = {row["file_path"]: row["summary"] for row in summaries_cursor.fetchall()}

        logger.info(f"Сессия загружена. Метаданные: {len(metadata)} полей, Сообщений: {len(messages_list)}, Саммари: {len(summaries_dict)}")
        return metadata, messages_list, summaries_dict

    except sqlite3.Error as e:
        logger.error(f"Ошибка SQLite при загрузке сессии {filepath}: {e}")
        return None
    except Exception as e:
        logger.error(f"Неожиданная ошибка при загрузке сессии {filepath}: {e}")
        return None


def save_session_data(
    filepath: str, 
    metadata_dict: Dict[str, Any], 
    messages_list: List[Dict[str, Any]],
    summaries_dict: Dict[str, str]
) -> bool:
    """
    Сохраняет (перезаписывает) все данные сессии в файл.
    """
    if not init_session_db(filepath):
        return False

    try:
        logger.info(f"Сохранение данных сессии в: {filepath}")
        current_time = datetime.datetime.now()
        metadata_dict["last_saved_at"] = current_time
        if not metadata_dict.get("created_at"):
            metadata_dict["created_at"] = current_time

        with _get_connection(filepath) as conn:
            if not conn: return False
            cursor = conn.cursor()
            cursor.execute("BEGIN TRANSACTION;")

            try:
                # Сохранение метаданных
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO metadata (
                        id, repo_url, repo_branch, vector_db_path, model_name, 
                        max_output_tokens, extensions, instructions, created_at, last_saved_at
                    )
                    VALUES (
                        1, :repo_url, :repo_branch, :vector_db_path, :model_name, 
                        :max_output_tokens, :extensions, :instructions, :created_at, :last_saved_at
                    )
                    """,
                    metadata_dict,
                )

                # Сохранение сообщений
                cursor.execute("DELETE FROM messages;")
                messages_to_insert = [
                    (
                        msg.get("role"),
                        (msg.get("parts", [""])[0] if msg.get("parts") else ""),
                        current_time,
                        index,
                        1 if msg.get("excluded", False) else 0,
                    )
                    for index, msg in enumerate(messages_list)
                ]
                cursor.executemany(
                    "INSERT INTO messages (role, content, timestamp, order_index, excluded_from_api) VALUES (?, ?, ?, ?, ?)",
                    messages_to_insert,
                )

                # Сохранение саммари
                cursor.execute("DELETE FROM file_summaries;")
                summaries_to_insert = list(summaries_dict.items())
                if summaries_to_insert:
                    cursor.executemany(
                        "INSERT INTO file_summaries (file_path, summary) VALUES (?, ?)",
                        summaries_to_insert
                    )

                conn.commit()
                logger.info(f"Сессия успешно сохранена. Сообщений: {len(messages_to_insert)}, Саммари: {len(summaries_to_insert)}")
                return True

            except Exception as e:
                logger.error(f"Ошибка во время транзакции сохранения сессии, откат: {e}", exc_info=True)
                conn.rollback()
                return False

    except sqlite3.Error as e:
        logger.error(f"Ошибка SQLite при сохранении сессии {filepath}: {e}")
        return False
    except Exception as e:
        logger.error(f"Неожиданная ошибка при сохранении сессии {filepath}: {e}", exc_info=True)
        return False