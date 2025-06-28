# --- Файл: embedding_model.py ---

import logging
import os
import numpy as np
from transformers import AutoTokenizer, AutoConfig
import onnxruntime as ort # Импортируем onnxruntime напрямую
from typing import List

logger = logging.getLogger(__name__)

# Определяем имя модели один раз для токенизатора и конфигурации
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

class ONNXEmbeddingModel:
    """
    Класс для создания текстовых эмбеддингов с использованием ONNX-модели.
    Использует onnxruntime напрямую для максимальной совместимости с CPU
    без инструкций AVX, полностью исключая библиотеку optimum.
    """
    def __init__(self):
        """
        Инициализирует токенизатор и загружает ONNX-модель из локального файла.
        """
        logger.info(f"Загрузка компонентов для ONNX-модели: {MODEL_NAME}")

        # Определяем путь к локальному файлу модели
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(script_dir, "onnx_model", "model.onnx")

            if not os.path.exists(model_path):
                error_message = f"Файл модели не найден по пути: {model_path}. Пожалуйста, скачайте его, как описано в инструкции."
                logger.critical(error_message)
                raise FileNotFoundError(error_message)

            # Загружаем ONNX-модель в сессию для инференса
            self.session = ort.InferenceSession(model_path)

            # Токенизатор и конфигурацию по-прежнему загружаем из Hugging Face
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            self.config = AutoConfig.from_pretrained(MODEL_NAME)
            logger.info("ONNX сессия, токенизатор и конфигурация успешно загружены.")

        except Exception as e:
            error_message = f"Критическая ошибка при загрузке ONNX-компонентов: {e}"
            logger.critical(error_message, exc_info=True)
            raise RuntimeError(error_message)

    def _mean_pooling(self, last_hidden_states: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        """
        Выполняет Mean Pooling для получения одного вектора для всего текста.
        Усредняет векторы токенов, игнорируя padding-токены.
        """
        input_mask_expanded = np.expand_dims(attention_mask, axis=-1).repeat(last_hidden_states.shape[-1], axis=-1)
        sum_embeddings = np.sum(last_hidden_states * input_mask_expanded, axis=1)
        sum_mask = np.maximum(np.sum(input_mask_expanded, axis=1), 1e-9)
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
            # Возвращаем пустой массив правильной формы, используя размер из конфига
            return np.array([]).reshape(0, self.config.hidden_size)

        # 1. Токенизация текста
        encoded_input = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="np"
        )

        # 2. Подготовка входов для ONNX сессии
        # Имена ключей ('input_ids', 'attention_mask', 'token_type_ids') должны
        # точно совпадать с именами входов в графе ONNX-модели.
        # Также явно приводим типы к int64, как этого часто требует ONNX.
        ort_inputs = {
            'input_ids': encoded_input['input_ids'].astype(np.int64),
            'attention_mask': encoded_input['attention_mask'].astype(np.int64),
            'token_type_ids': encoded_input['token_type_ids'].astype(np.int64),
        }

        # 3. Запуск инференса через ONNX Runtime
        # session.run возвращает список выходных тензоров. Для этой модели
        # нас интересует первый выход - last_hidden_state.
        ort_outputs = self.session.run(None, ort_inputs)
        last_hidden_state = ort_outputs[0]

        # 4. Применение Mean Pooling
        sentence_embeddings = self._mean_pooling(last_hidden_state, encoded_input['attention_mask'])

        # 5. Нормализация векторов
        norms = np.linalg.norm(sentence_embeddings, axis=1, keepdims=True)
        normalized_embeddings = sentence_embeddings / norms

        return normalized_embeddings