# retriever_advanced.py
import pandas as pd
import numpy as np
import logging
import re
from typing import List, Dict, Any, Tuple
from tqdm import tqdm

from config import Config
from retriever_tfidf import TFIDFRetrieval
from embedder import EmbeddingModel

# Добавим простой кросс-энкодер
from sentence_transformers import CrossEncoder


class AdvancedHybridRetrieval:
    """Улучшенный гибридный ретривер с кросс-энкодером"""

    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Инициализация компонентов
        self.tfidf_retriever = TFIDFRetrieval(config)
        self.embedder = EmbeddingModel(config)

        # Кросс-энкодер для реранжирования
        self.cross_encoder = None
        self._init_cross_encoder()

        # Для хранения текстов документов
        self.document_texts = {}
        self.document_titles = {}

        # Паттерны для бустинга из конфига
        self.positive_patterns = config.URL_POSITIVE_PATTERNS
        self.negative_patterns = config.URL_NEGATIVE_PATTERNS

    def _init_cross_encoder(self):
        """Инициализация кросс-энкодера с trust_remote_code"""
        try:
            if self.config.USE_RERANKER:
                self.cross_encoder = CrossEncoder(
                    self.config.RERANKER_MODEL,
                    trust_remote_code=self.config.TRUST_REMOTE_CODE
                )
                self.logger.info("Cross-encoder initialized successfully")
            else:
                self.logger.info("Cross-encoder disabled in config")
        except Exception as e:
            self.logger.warning(f"Failed to initialize cross-encoder: {e}")
            # Fallback на простой кросс-энкодер
            try:
                self.cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
                self.logger.info("Fallback cross-encoder loaded")
            except Exception as e2:
                self.logger.warning(f"Fallback cross-encoder also failed: {e2}")
                self.cross_encoder = None

    def build_index(self, documents_df: pd.DataFrame):
        """Построение улучшенного индекса"""
        self.logger.info("Building advanced hybrid index...")

        # 1. Строим TF-IDF индекс
        self.tfidf_retriever.build_index(documents_df)

        # 2. Сохраняем тексты и заголовки документов
        self._store_document_data(documents_df)

        # 3. Загружаем модель эмбеддингов
        self.embedder.load_model()

        self.logger.info("Advanced hybrid index built successfully!")

    def _store_document_data(self, documents_df: pd.DataFrame):
        """Сохраняет тексты и заголовки документов"""
        for _, row in documents_df.iterrows():
            doc_id = str(row['web_id'])
            text = str(row['text'])
            title = str(row.get('title', ''))

            self.document_texts[doc_id] = text[:2000]  # первые 2000 символов
            self.document_titles[doc_id] = title

    def _calculate_dynamic_weights(self, query: str) -> Tuple[float, float]:
        """Динамическое определение весов на основе характеристик запроса"""
        if not self.config.USE_DYNAMIC_WEIGHTS:
            return 0.3, 0.7  # базовые веса

        base_tfidf_weight = 0.3
        base_embedding_weight = 0.7

        # Анализ запроса
        words = query.split()
        has_digits = any(char.isdigit() for char in query)
        is_short = len(words) <= self.config.SHORT_QUERY_TOKENS
        has_keywords = any(keyword in query.lower() for keyword in self.config.BM25_BONUS_KEYWORDS)

        # Корректировка весов
        tfidf_bonus = 0.0
        if is_short:
            tfidf_bonus += self.config.BM25_BONUS_SHORT
        if has_digits:
            tfidf_bonus += self.config.BM25_BONUS_HAS_DIGITS
        if has_keywords:
            tfidf_bonus += 0.15

        tfidf_weight = min(0.6, base_tfidf_weight + tfidf_bonus)
        embedding_weight = 1.0 - tfidf_weight

        return tfidf_weight, embedding_weight

    def _calculate_url_boost(self, doc_id: str, query: str) -> float:
        """Вычисляет буст на основе URL/заголовка документа"""
        title = self.document_titles.get(doc_id, "").lower()
        text = self.document_texts.get(doc_id, "").lower()
        query_lower = query.lower()

        content = title + " " + text

        # Положительные паттерны
        positive_boost = 0.0
        for pattern in self.positive_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                positive_boost += self.config.URL_POSITIVE_BOOST
                break

        # Отрицательные паттерны (штраф)
        negative_penalty = 0.0
        for pattern in self.negative_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                negative_penalty += self.config.URL_NEGATIVE_PENALTY
                break

        # Буст по заголовку
        title_boost = 0.0
        if title and any(word in title for word in query_lower.split()):
            title_boost = self.config.TITLE_BOOST

        return positive_boost + title_boost - negative_penalty

    def search(self, query: str, top_k: int = None,
               tfidf_candidates: int = 80,
               rerank_candidates: int = 30) -> List[str]:
        """
        Улучшенный гибридный поиск с кросс-энкодером
        """
        if top_k is None:
            top_k = self.config.FINAL_TOP_K_DOCS

        # Динамические веса
        tfidf_weight, embedding_weight = self._calculate_dynamic_weights(query)
        self.logger.debug(f"Dynamic weights - TF-IDF: {tfidf_weight}, Embedding: {embedding_weight}")

        # Этап 1: Быстрый поиск кандидатов через TF-IDF
        candidate_docs = self._get_tfidf_candidates(query, tfidf_candidates)

        if not candidate_docs:
            return []

        # Этап 2: Реранжирование через эмбеддинги
        embedding_reranked = self._rerank_with_embeddings(
            query, candidate_docs, min(rerank_candidates * 2, len(candidate_docs)),
            tfidf_weight, embedding_weight
        )

        # Этап 3: Точное реранжирование через кросс-энкодер
        if self.cross_encoder and len(embedding_reranked) > 1:
            final_docs = self._rerank_with_cross_encoder(query, embedding_reranked[:rerank_candidates], top_k)
        else:
            final_docs = embedding_reranked[:top_k]

        self.logger.debug(f"Advanced search: '{query}' -> {final_docs}")
        return final_docs

    def _get_tfidf_candidates(self, query: str, n_candidates: int) -> List[Tuple[str, float]]:
        """Получение кандидатов через TF-IDF"""
        tokenized_query = query.split()

        # TF-IDF similarity
        query_vec = self.tfidf_retriever.vectorizer.transform([query])
        tfidf_similarities = self._cosine_similarity(
            query_vec, self.tfidf_retriever.tfidf_matrix
        ).flatten()

        # BM25 scores
        bm25_scores = self.tfidf_retriever.bm25.get_scores(tokenized_query)

        # Комбинируем скоры
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
                                top_k: int, tfidf_weight: float, embedding_weight: float) -> List[str]:
        """Реранжирование кандидатов с помощью эмбеддингов и бустов"""
        if not candidates:
            return []

        candidate_docs = list(set([doc_id for doc_id, score in candidates]))
        doc_texts = {}

        for doc_id in candidate_docs:
            if doc_id in self.document_texts:
                doc_texts[doc_id] = self.document_texts[doc_id]

        if not doc_texts:
            return [doc_id for doc_id, score in candidates[:top_k]]

        try:
            # Эмбеддинг запроса и документов
            query_embedding = self.embedder.encode([query])[0]
            doc_embeddings = self.embedder.encode(list(doc_texts.values()))

            # Косинусная схожесть
            similarities = self._cosine_similarity([query_embedding], doc_embeddings)[0]

            # Комбинируем TF-IDF и embedding скоры с бустами
            final_scores = {}
            for doc_id, tfidf_score in candidates:
                if doc_id not in doc_texts:
                    continue

                idx = list(doc_texts.keys()).index(doc_id)
                embedding_score = similarities[idx]

                # Применяем бусты
                url_boost = self._calculate_url_boost(doc_id, query)

                # Комбинированный скор
                combined_score = (tfidf_weight * tfidf_score +
                                  embedding_weight * embedding_score) * (1.0 + url_boost)

                final_scores[doc_id] = combined_score

            # Сортируем по комбинированному скору
            reranked = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
            return [doc_id for doc_id, score in reranked[:top_k]]

        except Exception as e:
            self.logger.error(f"Error in embedding reranking: {e}")
            return [doc_id for doc_id, score in candidates[:top_k]]

    def _rerank_with_cross_encoder(self, query: str, candidates: List[str], top_k: int) -> List[str]:
        """Точное реранжирование с помощью кросс-энкодера"""
        if not candidates or not self.cross_encoder:
            return candidates[:top_k]

        try:
            # Подготавливаем тексты для кросс-энкодера
            candidate_texts = []
            for doc_id in candidates:
                text = self.document_texts.get(doc_id, "")
                title = self.document_titles.get(doc_id, "")
                # Комбинируем заголовок и начало текста
                combined = f"{title}. {text[:500]}"
                candidate_texts.append(combined)

            # Скоринг пар (запрос, документ)
            pairs = [[query, doc_text] for doc_text in candidate_texts]
            ce_scores = self.cross_encoder.predict(pairs)

            # Сортируем по скорам кросс-энкодера
            scored_candidates = list(zip(candidates, ce_scores))
            scored_candidates.sort(key=lambda x: x[1], reverse=True)

            return [doc_id for doc_id, score in scored_candidates[:top_k]]

        except Exception as e:
            self.logger.error(f"Error in cross-encoder reranking: {e}")
            return candidates[:top_k]

    def _cosine_similarity(self, vec1, vec2):
        """Вычисление косинусной схожести"""
        from sklearn.metrics.pairwise import cosine_similarity
        return cosine_similarity(vec1, vec2)

    def get_index_info(self) -> Dict[str, Any]:
        """Информация об индексе"""
        info = self.tfidf_retriever.get_index_info()
        info["type"] = "advanced_hybrid"
        info["embedding_model"] = self.config.EMBEDDING_MODEL
        info["cross_encoder"] = self.config.RERANKER_MODEL if self.cross_encoder else "None"
        info["dynamic_weights"] = self.config.USE_DYNAMIC_WEIGHTS
        return info