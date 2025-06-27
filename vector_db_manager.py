# --- Файл: vector_db_manager.py ---

import logging
import chromadb
from chromadb.errors import NotFoundError
from typing import List, Dict, Any, Optional
import hashlib
import os
import numpy as np

# ИМПОРТИРУЕМ НАШ НОВЫЙ МОДУЛЬ
from embedding_model import ONNXEmbeddingModel

logger = logging.getLogger(__name__)

class VectorDBManager:
    """
    Класс для управления локальной векторной базой данных ChromaDB.
    Инкапсулирует создание эмбеддингов, хранение и семантический поиск.
    Теперь использует внешнюю ONNX-модель для создания эмбеддингов.
    """

    def __init__(self):
        """
        Инициализирует менеджер векторной БД и модель для создания эмбеддингов.
        """
        self._db_path: Optional[str] = None
        self.client: Optional[chromadb.Client] = None
        # Создаем экземпляр нашей ONNX-модели
        try:
            self.embedding_model = ONNXEmbeddingModel()
        except RuntimeError as e:
            # Если модель не смогла загрузиться, это критическая ошибка.
            # Мы не можем продолжать, поэтому логируем и снова выбрасываем исключение,
            # чтобы вышестоящий код мог его обработать.
            logger.critical(f"Не удалось инициализировать VectorDBManager: {e}", exc_info=True)
            raise e
        logger.info("VectorDBManager успешно создан и модель эмбеддингов загружена.")


    def set_db_path(self, path: str):
        """
        Устанавливает новый путь для базы данных и инициализирует клиента.
        """
        if self._db_path == path and self.client:
            return
        
        logger.info(f"Установка пути к векторной БД: {path}")
        self._db_path = path
        try:
            # Убедимся, что директория существует
            os.makedirs(self._db_path, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self._db_path)
            logger.info(f"Клиент ChromaDB успешно инициализирован для пути: {self._db_path}")
        except Exception as e:
            logger.error(f"Не удалось инициализировать PersistentClient для ChromaDB по пути {path}: {e}", exc_info=True)
            self.client = None


    def create_or_get_collection(self, name: str) -> Optional[chromadb.Collection]:
        """
        Создает новую коллекцию в ChromaDB или получает доступ к существующей.
        Теперь не передает embedding_function.
        """
        if not self.client:
            logger.error("Клиент ChromaDB не инициализирован.")
            return None
        try:
            # Мы больше не передаем embedding_function, так как будем
            # предоставлять эмбеддинги вручную.
            collection = self.client.get_or_create_collection(name=name)
            logger.info(f"Успешный доступ к коллекции: '{name}'")
            return collection
        except Exception as e:
            logger.error(f"Не удалось создать или получить коллекцию '{name}': {e}", exc_info=True)
            return None

    @staticmethod
    def _generate_document_id(text: str, metadata: Dict[str, Any]) -> str:
        """Генерирует уникальный и детерминированный ID для документа."""
        identifier_string = f"{metadata.get('file_path', '')}::{metadata.get('type', '')}::{text}"
        return hashlib.sha256(identifier_string.encode('utf-8')).hexdigest()

    def add_documents_batch(
        self,
        collection: chromadb.Collection,
        documents: List[str],
        metadatas: List[Dict[str, Any]]
    ):
        """
        Добавляет пакет документов в коллекцию. Эмбеддинги генерируются здесь.
        """
        if not documents:
            logger.warning("Попытка добавить пустой список документов. Операция пропущена.")
            return

        ids = [self._generate_document_id(doc, meta) for doc, meta in zip(documents, metadatas)]
        
        try:
            # --- КЛЮЧЕВОЕ ИЗМЕНЕНИЕ ---
            # 1. Генерируем эмбеддинги с помощью нашей ONNX-модели
            logger.debug(f"Генерация {len(documents)} эмбеддингов для добавления в '{collection.name}'...")
            embeddings = self.embedding_model.encode(documents)
            logger.debug("Эмбеддинги успешно сгенерированы.")
            
            # 2. Добавляем в коллекцию документы вместе с готовыми эмбеддингами
            collection.add(
                embeddings=embeddings.tolist(), # ChromaDB ожидает list of lists
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            logger.info(f"В коллекцию '{collection.name}' успешно добавлено {len(documents)} документов.")
        except Exception as e:
            logger.error(f"Ошибка при добавлении документов в коллекцию '{collection.name}': {e}", exc_info=True)

    def query(
        self,
        collection: chromadb.Collection,
        query_text: str,
        n_results: int = 15
    ) -> List[Dict[str, Any]]: # Изменен тип возвращаемого значения, чтобы не возвращать None
        """
        Выполняет семантический поиск по коллекции.
        """
        if not query_text.strip():
            logger.warning("Получен пустой запрос для поиска, возвращаем пустой результат.")
            return []
            
        try:
            # --- КЛЮЧЕВОЕ ИЗМЕНЕНИЕ ---
            # 1. Генерируем эмбеддинг для текста запроса
            logger.debug(f"Генерация эмбеддинга для запроса: '{query_text[:50]}...'")
            query_embedding = self.embedding_model.encode([query_text])
            
            # 2. Выполняем поиск, передавая готовый эмбеддинг
            results = collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=min(n_results, collection.count()) # Убедимся, что не запрашиваем больше, чем есть
            )

            formatted_results = []
            if results and results.get('ids') and results['ids'][0]:
                for i in range(len(results['ids'][0])):
                    formatted_results.append({
                        'id': results['ids'][0][i],
                        'document': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'distance': results['distances'][0][i]
                    })
            logger.info(f"Поиск по запросу '{query_text[:50]}...' вернул {len(formatted_results)} результатов.")
            return formatted_results

        except Exception as e:
            logger.error(f"Ошибка при поиске в коллекции '{collection.name}': {e}", exc_info=True)
            return []

    def delete_collection(self, collection_name: str):
        """ Полностью удаляет коллекцию из базы данных. """
        if not self.client:
            logger.error("Невозможно удалить коллекцию, клиент ChromaDB не инициализирован.")
            return
        try:
            self.client.delete_collection(name=collection_name)
            logger.info(f"Старая коллекция '{collection_name}' успешно удалена перед новым анализом.")
        except NotFoundError:
            logger.warning(f"Коллекция '{collection_name}' не найдена для удаления, что нормально для первого анализа.")
        except Exception as e:
            logger.error(f"Не удалось удалить коллекцию '{collection_name}': {e}", exc_info=True)
            
    def get_collection_doc_count(self, collection: chromadb.Collection) -> int:
        """ Возвращает количество документов в коллекции. """
        if not collection:
            return 0
        try:
            return collection.count()
        except Exception as e:
            logger.error(f"Не удалось получить количество документов для коллекции '{collection.name}': {e}")
            return 0