# --- Файл: vector_db_manager.py ---

import logging
import chromadb
from chromadb.errors import NotFoundError
from typing import List, Dict, Any, Optional
import hashlib
import os
import numpy as np

from PySide6.QtCore import QObject

# ИМПОРТИРУЕМ НАШ НОВЫЙ МОДУЛЬ
from embedding_model import ONNXEmbeddingModel

logger = logging.getLogger(__name__)

class VectorDBManager(QObject):
    """
    Класс для управления локальной векторной базой данных ChromaDB.
    Инкапсулирует создание эмбеддингов, хранение и семантический поиск.
    Теперь использует внешнюю ONNX-модель для создания эмбеддингов.
    """

    def __init__(self):
        """
        Инициализирует менеджер векторной БД и модель для создания эмбеддингов.
        """
        super().__init__()
        self._db_path: Optional[str] = None
        self.client: Optional[chromadb.Client] = None
        # Создаем экземпляр нашей ONNX-модели
        try:
            self.embedding_model = ONNXEmbeddingModel()
        except RuntimeError as e:
            # Если модель не смогла загрузиться, это критическая ошибка.
            # Мы не можем продолжать, поэтому логируем и снова выбрасываем исключение,
            # чтобы вышестоящий код мог его обработать.
            error_msg = self.tr("Не удалось инициализировать VectorDBManager: {0}").format(e)
            logger.critical(error_msg, exc_info=True)
            raise RuntimeError(error_msg)
        logger.info(self.tr("VectorDBManager успешно создан и модель эмбеддингов загружена."))


    def set_db_path(self, path: str):
        """
        Устанавливает новый путь для базы данных и инициализирует клиента.
        """
        if self._db_path == path and self.client:
            return
        
        logger.info(self.tr("Установка пути к векторной БД: {0}").format(path))
        self._db_path = path
        try:
            # Убедимся, что директория существует
            os.makedirs(self._db_path, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self._db_path)
            logger.info(self.tr("Клиент ChromaDB успешно инициализирован для пути: {0}").format(self._db_path))
        except Exception as e:
            logger.error(self.tr("Не удалось инициализировать PersistentClient для ChromaDB по пути {0}: {1}").format(path, e), exc_info=True)
            self.client = None


    def create_or_get_collection(self, name: str) -> Optional[chromadb.Collection]:
        """
        Создает новую коллекцию в ChromaDB или получает доступ к существующей.
        """
        if not self.client:
            logger.error(self.tr("Клиент ChromaDB не инициализирован."))
            return None
        try:
            collection = self.client.get_or_create_collection(name=name)
            logger.info(self.tr("Успешный доступ к коллекции: '{0}'").format(name))
            return collection
        except Exception as e:
            logger.error(self.tr("Не удалось создать или получить коллекцию '{0}': {1}").format(name, e), exc_info=True)
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
            logger.warning(self.tr("Попытка добавить пустой список документов. Операция пропущена."))
            return

        ids = [self._generate_document_id(doc, meta) for doc, meta in zip(documents, metadatas)]
        
        try:
            logger.debug(self.tr("Генерация {0} эмбеддингов для добавления в '{1}'...").format(len(documents), collection.name))
            embeddings = self.embedding_model.encode(documents)
            logger.debug(self.tr("Эмбеддинги успешно сгенерированы."))
            
            collection.add(
                embeddings=embeddings.tolist(),
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            logger.info(self.tr("В коллекцию '{0}' успешно добавлено {1} документов.").format(collection.name, len(documents)))
        except Exception as e:
            logger.error(self.tr("Ошибка при добавлении документов в коллекцию '{0}': {1}").format(collection.name, e), exc_info=True)

    def query(
        self,
        collection: chromadb.Collection,
        query_text: str,
        n_results: int = 15
    ) -> List[Dict[str, Any]]:
        """
        Выполняет семантический поиск по коллекции.
        """
        if not query_text.strip():
            logger.warning(self.tr("Получен пустой запрос для поиска, возвращаем пустой результат."))
            return []
            
        try:
            logger.debug(self.tr("Генерация эмбеддинга для запроса: '{0}...'").format(query_text[:50]))
            query_embedding = self.embedding_model.encode([query_text])
            
            results = collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=min(n_results, collection.count())
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
            logger.info(self.tr("Поиск по запросу '{0}...' вернул {1} результатов.").format(query_text[:50], len(formatted_results)))
            return formatted_results

        except Exception as e:
            logger.error(self.tr("Ошибка при поиске в коллекции '{0}': {1}").format(collection.name, e), exc_info=True)
            return []

    def delete_collection(self, collection_name: str):
        """ Полностью удаляет коллекцию из базы данных. """
        if not self.client:
            logger.error(self.tr("Невозможно удалить коллекцию, клиент ChromaDB не инициализирован."))
            return
        try:
            self.client.delete_collection(name=collection_name)
            logger.info(self.tr("Старая коллекция '{0}' успешно удалена перед новым анализом.").format(collection_name))
        except NotFoundError:
            logger.warning(self.tr("Коллекция '{0}' не найдена для удаления, что нормально для первого анализа.").format(collection_name))
        except Exception as e:
            logger.error(self.tr("Не удалось удалить коллекцию '{0}': {1}").format(collection_name, e), exc_info=True)
            
    def get_collection_doc_count(self, collection: chromadb.Collection) -> int:
        """ Возвращает количество документов в коллекции. """
        if not collection:
            return 0
        try:
            return collection.count()
        except Exception as e:
            logger.error(self.tr("Не удалось получить количество документов для коллекции '{0}': {1}").format(collection.name, e))
            return 0