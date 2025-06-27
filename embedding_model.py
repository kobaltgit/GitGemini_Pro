# --- Файл: embedding_model.py ---

import logging
import numpy as np
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForFeatureExtraction
from typing import List

logger = logging.getLogger(__name__)

# Определяем имя модели один раз
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

class ONNXEmbeddingModel:
    """
    Класс для создания текстовых эмбеддингов с использованием ONNX-модели.
    Обеспечивает совместимость с CPU без инструкций AVX, так как использует
    ONNX Runtime вместо PyTorch для вычислений.
    """
    def __init__(self):
        """
        Инициализирует и загружает токенизатор и ONNX-модель в память.
        """
        logger.info(f"Загрузка ONNX-совместимой модели эмбеддингов: {MODEL_NAME}")
        try:
            # provider="CPUExecutionProvider" явно указывает, что нужно использовать CPU.
            # Это гарантирует работу без GPU и на широком спектре процессоров.
            self.model = ORTModelForFeatureExtraction.from_pretrained(
                MODEL_NAME, 
                provider="CPUExecutionProvider"
            )
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            logger.info("Модель и токенизатор для эмбеддингов успешно загружены.")
        except Exception as e:
            error_message = f"Критическая ошибка при загрузке ONNX-модели: {e}"
            logger.critical(error_message, exc_info=True)
            # Если модель не загрузилась, приложение не сможет работать.
            raise RuntimeError(error_message)

    def _mean_pooling(self, model_output, attention_mask: np.ndarray) -> np.ndarray:
        """
        Выполняет Mean Pooling для получения одного вектора для всего текста.
        Усредняет векторы токенов, игнорируя padding-токены.
        """
        # model_output[0] содержит эмбеддинги для каждого токена
        token_embeddings = model_output[0] 
        
        # Расширяем attention_mask для поэлементного умножения
        input_mask_expanded = np.expand_dims(attention_mask, axis=-1).repeat(token_embeddings.shape[-1], axis=-1)
        
        # Умножаем эмбеддинги на маску, чтобы обнулить padding-токены
        sum_embeddings = np.sum(token_embeddings * input_mask_expanded, axis=1)
        
        # Суммируем маску, чтобы получить количество реальных токенов в каждом предложении
        sum_mask = np.maximum(np.sum(input_mask_expanded, axis=1), 1e-9)
        
        # Делим сумму эмбеддингов на количество токенов
        return sum_embeddings / sum_mask

    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Кодирует список текстов в эмбеддинги.

        Args:
            texts: Список строк для кодирования.

        Returns:
            NumPy-массив с эмбеддингами.
        """
        if not texts:
            # Возвращаем пустой массив правильной формы
            return np.array([]).reshape(0, self.model.config.hidden_size)

        # 1. Токенизация
        # return_tensors="np" - ключевой момент, заставляющий токенизатор
        # возвращать NumPy-массивы вместо PyTorch-тензоров.
        encoded_input = self.tokenizer(
            texts, 
            padding=True, 
            truncation=True, 
            return_tensors="np"
        )

        # 2. Получение эмбеддингов от модели
        model_output = self.model(**encoded_input)

        # 3. Mean Pooling для получения итогового вектора
        sentence_embeddings = self._mean_pooling(model_output, encoded_input['attention_mask'])

        # 4. Нормализация (важно для задач семантического поиска)
        norms = np.linalg.norm(sentence_embeddings, axis=1, keepdims=True)
        normalized_embeddings = sentence_embeddings / norms

        return normalized_embeddings