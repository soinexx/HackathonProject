# retriever_hybrid.py
import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Any, Tuple
from tqdm import tqdm

from config import Config
from retriever_tfidf import TFIDFRetrieval
from embedder import EmbeddingModel


class HybridRetrieval:
    """Гибридный ретривер: TF-IDF для быстрого префильтра, эмбеддинги для реранжирования"""

    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Инициализация компонентов
        self.tfidf_retriever = TFIDFRetrieval(config)
        self.embedder = EmbeddingModel(config)

        # Для хранения текстов документов (нужно для реранжирования)
        self.document_texts = {}

    def build_index(self, documents_df: pd.DataFrame):
        """Построение гибридного индекса"""
        self.logger.info("Building hybrid index...")

        # 1. Строим TF-IDF индекс (быстро)
        self.tfidf_retriever.build_index(documents_df)

        # 2. Сохраняем тексты документов для реранжирования
        self._store_document_texts(documents_df)

        # 3. Загружаем модель эмбеддингов
        self.embedder.load_model()

        self.logger.info("Hybrid index built successfully!")

    def _store_document_texts(self, documents_df: pd.DataFrame):
        """Сохраняет тексты документов для реранжирования"""
        for _, row in documents_df.iterrows():
            doc_id = str(row['web_id'])
            text = str(row['text'])
            # Сохраняем обрезанную версию для эффективности
            self.document_texts[doc_id] = text[:2000]  # первые 2000 символов обычно достаточно

    def search(self, query: str, top_k: int = None,
               tfidf_candidates: int = 50,
               rerank_top_k: int = 10) -> List[str]:
        """
        Гибридный поиск:
        1. TF-IDF для быстрого получения кандидатов
        2. Эмбеддинги для точного реранжирования
        """
        if top_k is None:
            top_k = self.config.FINAL_TOP_K_DOCS

        # Этап 1: Быстрый поиск кандидатов через TF-IDF
        self.logger.debug(f"Stage 1: TF-IDF candidate retrieval ({tfidf_candidates} candidates)")
        candidate_docs = self._get_tfidf_candidates(query, tfidf_candidates)

        if not candidate_docs:
            return []

        # Этап 2: Точное реранжирование через эмбеддинги
        self.logger.debug(f"Stage 2: Embedding reranking ({len(candidate_docs)} -> {rerank_top_k})")
        reranked_docs = self._rerank_with_embeddings(query, candidate_docs, rerank_top_k)

        # Этап 3: Возвращаем топ-K результатов
        final_docs = reranked_docs[:top_k]

        self.logger.debug(f"Hybrid search: '{query}' -> {final_docs}")
        return final_docs

    def _get_tfidf_candidates(self, query: str, n_candidates: int) -> List[Tuple[str, float]]:
        """Получение кандидатов через TF-IDF с возвратом скоров"""
        # Используем внутренние структуры TF-IDF для получения скоров
        tokenized_query = query.split()

        # TF-IDF similarity
        query_vec = self.tfidf_retriever.vectorizer.transform([query])
        tfidf_similarities = self._cosine_similarity(
            query_vec, self.tfidf_retriever.tfidf_matrix
        ).flatten()

        # BM25 scores
        bm25_scores = self.tfidf_retriever.bm25.get_scores(tokenized_query)

        # Комбинируем скоры (как в оригинальном TF-IDF)
        combined_scores = {}
        for i in range(len(tfidf_similarities)):
            tfidf_score = tfidf_similarities[i] / tfidf_similarities.max() if tfidf_similarities.max() > 0 else 0
            bm25_score = bm25_scores[i] / bm25_scores.max() if bm25_scores.max() > 0 else 0
            combined_scores[i] = 0.6 * tfidf_score + 0.4 * bm25_score

        # Сортируем и получаем кандидатов
        sorted_indices = sorted(combined_scores.keys(), key=lambda x: combined_scores[x], reverse=True)
        candidate_indices = sorted_indices[:n_candidates]

        # Возвращаем (doc_id, score) пары
        candidates = []
        for idx in candidate_indices:
            doc_id = self.tfidf_retriever.chunk_to_doc_map[idx]
            score = combined_scores[idx]
            candidates.append((doc_id, score))

        return candidates

    def _rerank_with_embeddings(self, query: str, candidates: List[Tuple[str, float]],
                                top_k: int) -> List[str]:
        """Реранжирование кандидатов с помощью эмбеддингов"""
        if not candidates:
            return []

        # Получаем уникальные ID документов
        candidate_docs = list(set([doc_id for doc_id, score in candidates]))

        # Получаем тексты документов для реранжирования
        doc_texts = {}
        for doc_id in candidate_docs:
            if doc_id in self.document_texts:
                doc_texts[doc_id] = self.document_texts[doc_id]

        if not doc_texts:
            self.logger.warning("No document texts found for reranking")
            return [doc_id for doc_id, score in candidates[:top_k]]

        # Генерируем эмбеддинги для запроса и документов
        try:
            # Эмбеддинг запроса
            query_embedding = self.embedder.encode([query])[0]

            # Эмбеддинги документов (батчинг для эффективности)
            doc_embeddings = self.embedder.encode(list(doc_texts.values()))

            # Вычисляем косинусную схожесть
            similarities = self._cosine_similarity([query_embedding], doc_embeddings)[0]

            # Создаем словарь схожестей
            doc_similarities = {}
            for i, doc_id in enumerate(doc_texts.keys()):
                doc_similarities[doc_id] = similarities[i]

            # Комбинируем TF-IDF и embedding скоры
            final_scores = {}
            for doc_id, tfidf_score in candidates:
                embedding_score = doc_similarities.get(doc_id, 0)
                # Взвешенная комбинация (можно настроить веса)
                final_scores[doc_id] = 0.4 * tfidf_score + 0.6 * embedding_score

            # Сортируем по комбинированному скору
            reranked = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
            return [doc_id for doc_id, score in reranked[:top_k]]

        except Exception as e:
            self.logger.error(f"Error in embedding reranking: {e}")
            # Fallback: возвращаем TF-IDF результаты
            return [doc_id for doc_id, score in candidates[:top_k]]

    def _cosine_similarity(self, vec1, vec2):
        """Вычисление косинусной схожести"""
        from sklearn.metrics.pairwise import cosine_similarity
        return cosine_similarity(vec1, vec2)

    def save_index(self, path: str):
        """Сохранение индекса"""
        # Пока сохраняем только TF-IDF часть
        self.tfidf_retriever.save_index(path)

    def load_index(self, path: str):
        """Загрузка индекса"""
        self.tfidf_retriever.load_index(path)
        self.embedder.load_model()

    def get_index_info(self) -> Dict[str, Any]:
        """Информация об индексе"""
        info = self.tfidf_retriever.get_index_info()
        info["type"] = "hybrid_tfidf_embeddings"
        info["embedding_model"] = self.config.EMBEDDING_MODEL
        return info