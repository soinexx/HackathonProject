import numpy as np
from sentence_transformers import SentenceTransformer
import logging
from typing import List, Union
import os
from config import Config


class EmbeddingModel:
    """Класс для работы с моделями эмбеддингов"""

    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.model_name = config.EMBEDDING_MODEL

    def load_model(self):
        """Загрузка модели эмбеддингов"""
        try:
            self.logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            self.logger.info("Model loaded successfully")
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise

    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Генерация эмбеддингов для списка текстов
        """
        if self.model is None:
            self.load_model()

        if not texts:
            return np.array([])

        self.logger.info(f"Encoding {len(texts)} texts with batch size {batch_size}")

        try:
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True  # для косинусной схожести
            )
            self.logger.info(f"Generated embeddings shape: {embeddings.shape}")
            return embeddings
        except Exception as e:
            self.logger.error(f"Error encoding texts: {e}")
            raise

    def encode_single(self, text: str) -> np.ndarray:
        """Генерация эмбеддинга для одного текста"""
        return self.encode([text])[0]

    def get_model_dimension(self) -> int:
        """Возвращает размерность эмбеддингов"""
        if self.model is None:
            self.load_model()
        return self.model.get_sentence_embedding_dimension()

    def save_model(self, path: str):
        """Сохранение модели на диск"""
        if self.model is None:
            self.load_model()
        self.model.save(path)

    @classmethod
    def get_available_models(cls) -> List[str]:
        """Возвращает список доступных моделей"""
        return [
            "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            "intfloat/multilingual-e5-base"
        ]