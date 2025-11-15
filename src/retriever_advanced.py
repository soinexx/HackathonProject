# retriever_advanced.py
import pandas as pd
import numpy as np
import logging
import re
from typing import List, Dict, Any, Tuple

from config import Config
from retriever_tfidf import TFIDFRetrieval
from embedder import EmbeddingModel

from sentence_transformers import CrossEncoder
from query_utils import normalize_query_text
from query_expander import QueryExpander


class AdvancedHybridRetrieval:
    """Улучшенный гибридный ретривер: TF-IDF/BM25 + dense + cross-encoder"""

    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Базовые компоненты
        self.tfidf_retriever = TFIDFRetrieval(config)
        self.embedder = EmbeddingModel(config)

        # Cross-encoder
        self.cross_encoder: CrossEncoder | None = None
        self._init_cross_encoder()

        # Тексты/заголовки/URL документов
        self.document_texts: Dict[str, str] = {}
        self.document_titles: Dict[str, str] = {}
        self.document_urls: Dict[str, str] = {}

        # Паттерны для буста
        self.positive_patterns = config.URL_POSITIVE_PATTERNS
        self.negative_patterns = config.URL_NEGATIVE_PATTERNS

        # dense-часть
        self.doc_ids: List[str] = []
        self.doc_embeddings: np.ndarray | None = None
        self.doc_id_to_index: Dict[str, int] = {}

    def _init_cross_encoder(self):
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
            try:
                self.cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
                self.logger.info("Fallback cross-encoder loaded")
            except Exception as e2:
                self.logger.warning(f"Fallback cross-encoder also failed: {e2}")
                self.cross_encoder = None

    def build_index(self, documents_df: pd.DataFrame):
        """Построение улучшенного индекса"""
        self.logger.info("Building advanced hybrid index...")

        # 1. TF-IDF/BM25 по чанкам
        self.tfidf_retriever.build_index(documents_df)

        # 2. Док-тексты/тайтлы/URL
        self._store_document_data(documents_df)

        # 3. dense-эмбеддинги по документам
        self.embedder.load_model()
        self.doc_ids = list(self.document_texts.keys())
        texts_for_embedding = [self.document_texts[doc_id] for doc_id in self.doc_ids]
        self.doc_embeddings = self.embedder.encode(texts_for_embedding, batch_size=64)
        self.doc_id_to_index = {doc_id: i for i, doc_id in enumerate(self.doc_ids)}

        self.logger.info(
            f"Advanced hybrid index built: {len(self.doc_ids)} docs, "
            f"{self.tfidf_retriever.get_index_info().get('chunks_count', 0)} chunks"
        )

    def _store_document_data(self, documents_df: pd.DataFrame):
        """Сохраняет тексты, заголовки и URL документов"""
        for _, row in documents_df.iterrows():
            doc_id = str(row['web_id'])
            text = str(row['text'])
            title = str(row.get('title', ''))
            url = str(row.get('url', ''))

            self.document_texts[doc_id] = text[:2000]
            self.document_titles[doc_id] = title
            self.document_urls[doc_id] = url

    def _calculate_dynamic_weights(self, query: str) -> Tuple[float, float]:
        if not self.config.USE_DYNAMIC_WEIGHTS:
            return 0.3, 0.7

        base_tfidf_weight = 0.3
        base_embedding_weight = 0.7

        words = query.split()
        has_digits = any(char.isdigit() for char in query)
        is_short = len(words) <= self.config.SHORT_QUERY_TOKENS
        has_keywords = any(kw in query.lower() for kw in self.config.BM25_BONUS_KEYWORDS)

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
        title = self.document_titles.get(doc_id, "").lower()
        url = self.document_urls.get(doc_id, "").lower()
        query_lower = query.lower()

        content = f"{url} {title}"

        positive_boost = 0.0
        for pattern in self.positive_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                positive_boost += self.config.URL_POSITIVE_BOOST
                break

        negative_penalty = 0.0
        for pattern in self.negative_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                negative_penalty += self.config.URL_NEGATIVE_PENALTY
                break

        title_boost = 0.0
        if title and any(word in title for word in query_lower.split()):
            title_boost = self.config.TITLE_BOOST

        return positive_boost + title_boost - negative_penalty

    def _get_tfidf_candidates(self, query: str, n_candidates: int) -> List[Tuple[str, float]]:
        """Кандидаты через TF-IDF/BM25 (агрегация по документах)"""
        if self.tfidf_retriever.tfidf_matrix is None or self.tfidf_retriever.bm25 is None:
            return []

        expander = QueryExpander()
        expanded_query = expander.expand_query(query)

        tokenized_query = expanded_query.split()
        query_vec = self.tfidf_retriever.vectorizer.transform([expanded_query])
        tfidf_similarities = self._cosine_similarity(
            query_vec, self.tfidf_retriever.tfidf_matrix
        ).flatten()

        bm25_scores = self.tfidf_retriever.bm25.get_scores(tokenized_query)

        if tfidf_similarities.max() > 0:
            tfidf_norm = tfidf_similarities / tfidf_similarities.max()
        else:
            tfidf_norm = tfidf_similarities

        if bm25_scores.max() > 0:
            bm25_norm = bm25_scores / bm25_scores.max()
        else:
            bm25_norm = bm25_scores

        combined_scores = 0.6 * tfidf_norm + 0.4 * bm25_norm

        # агрегация на уровень документа (максимум по чанкам)
        doc_scores: Dict[str, float] = {}
        for chunk_idx, score in enumerate(combined_scores):
            doc_id = self.tfidf_retriever.chunk_to_doc_map.get(chunk_idx)
            if doc_id is None:
                continue
            if doc_id not in doc_scores or score > doc_scores[doc_id]:
                doc_scores[doc_id] = float(score)

        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_docs[:n_candidates]

    def _get_dense_candidates(self, query: str, n_candidates: int) -> List[Tuple[str, float]]:
        """Кандидаты по dense-эмбеддингам (по документам)"""
        if self.doc_embeddings is None or not self.doc_ids:
            return []

        query_embedding = self.embedder.encode([query])[0]
        sims = self._cosine_similarity([query_embedding], self.doc_embeddings)[0]

        top_idx = np.argsort(-sims)[:n_candidates]
        return [(self.doc_ids[i], float(sims[i])) for i in top_idx]

    def _rerank_with_embeddings(
        self,
        query: str,
        candidates: List[Tuple[str, float]],
        top_k: int,
        tfidf_weight: float,
        embedding_weight: float
    ) -> List[str]:
        """Реранжирование кандидатов с помощью эмбеддингов и URL-буста"""
        if not candidates:
            return []

        if self.doc_embeddings is None:
            return [doc_id for doc_id, _ in candidates[:top_k]]

        candidate_doc_ids = [doc_id for doc_id, _ in candidates if doc_id in self.doc_id_to_index]
        if not candidate_doc_ids:
            return [doc_id for doc_id, _ in candidates[:top_k]]

        indices = [self.doc_id_to_index[doc_id] for doc_id in candidate_doc_ids]
        doc_embs = self.doc_embeddings[indices]

        query_embedding = self.embedder.encode([query])[0]
        similarities = self._cosine_similarity([query_embedding], doc_embs)[0]
        emb_scores = {doc_id: float(similarities[i]) for i, doc_id in enumerate(candidate_doc_ids)}

        final_scores: Dict[str, float] = {}
        for doc_id, tfidf_score in candidates:
            if doc_id not in emb_scores:
                continue
            embedding_score = emb_scores[doc_id]
            url_boost = self._calculate_url_boost(doc_id, query)

            combined_score = (
                tfidf_weight * tfidf_score +
                embedding_weight * embedding_score
            ) * (1.0 + url_boost)

            final_scores[doc_id] = combined_score

        reranked = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        return [doc_id for doc_id, _ in reranked[:top_k]]

    def _rerank_with_cross_encoder(self, query: str, candidates: List[str], top_k: int) -> List[str]:
        """Точное реранжирование с помощью cross-encoder'а"""
        if not candidates or not self.cross_encoder:
            return candidates[:top_k]

        try:
            candidate_texts = []
            for doc_id in candidates:
                text = self.document_texts.get(doc_id, "")
                title = self.document_titles.get(doc_id, "")
                combined = f"{title}. {text[:500]}"
                candidate_texts.append(combined)

            pairs = [[query, doc_text] for doc_text in candidate_texts]
            ce_scores = self.cross_encoder.predict(pairs)

            scored_candidates = list(zip(candidates, ce_scores))
            scored_candidates.sort(key=lambda x: x[1], reverse=True)

            return [doc_id for doc_id, _ in scored_candidates[:top_k]]
        except Exception as e:
            self.logger.error(f"Error in cross-encoder reranking: {e}")
            return candidates[:top_k]

    def search(
        self,
        query: str,
        top_k: int | None = None,
        tfidf_candidates: int | None = None,
        dense_candidates: int | None = None,
        rerank_candidates: int | None = None
    ) -> List[str]:
        """Основной метод поиска"""

        norm_query = normalize_query_text(query)
        if norm_query != query:
            self.logger.debug(f"Normalized query: '{query}' -> '{norm_query}'")

        if top_k is None:
            top_k = self.config.FINAL_TOP_K_DOCS
        if tfidf_candidates is None:
            tfidf_candidates = self.config.TFIDF_CANDIDATES
        if dense_candidates is None:
            dense_candidates = self.config.DENSE_CANDIDATES
        if rerank_candidates is None:
            rerank_candidates = self.config.RERANK_CANDIDATES

        tfidf_weight, embedding_weight = self._calculate_dynamic_weights(norm_query)
        self.logger.debug(
            f"Dynamic weights for '{norm_query}': TF-IDF={tfidf_weight:.2f}, Embedding={embedding_weight:.2f}"
        )

        bm25_candidates = self._get_tfidf_candidates(norm_query, tfidf_candidates)
        dense_cands = self._get_dense_candidates(norm_query, dense_candidates)

        combined_scores: Dict[str, float] = {}
        for doc_id, score in bm25_candidates:
            combined_scores[doc_id] = max(combined_scores.get(doc_id, 0.0), score)
        for doc_id, _ in dense_cands:
            combined_scores.setdefault(doc_id, 0.0)

        candidates_list: List[Tuple[str, float]] = list(combined_scores.items())
        if not candidates_list:
            return []

        embedding_reranked_docs = self._rerank_with_embeddings(
            norm_query,
            candidates_list,
            top_k=max(rerank_candidates, top_k),
            tfidf_weight=tfidf_weight,
            embedding_weight=embedding_weight
        )

        if self.cross_encoder and len(embedding_reranked_docs) > 1:
            final_docs = self._rerank_with_cross_encoder(
                norm_query,
                embedding_reranked_docs[:rerank_candidates],
                top_k
            )
        else:
            final_docs = embedding_reranked_docs[:top_k]

        self.logger.debug(f"Advanced search: '{query}' -> {final_docs}")
        return final_docs

    def _cosine_similarity(self, vec1, vec2):
        from sklearn.metrics.pairwise import cosine_similarity
        return cosine_similarity(vec1, vec2)

    def get_index_info(self) -> Dict[str, Any]:
        info = self.tfidf_retriever.get_index_info()
        info["type"] = "advanced_hybrid"
        info["embedding_model"] = self.config.EMBEDDING_MODEL
        info["cross_encoder"] = self.config.RERANKER_MODEL if self.cross_encoder else "None"
        info["dynamic_weights"] = self.config.USE_DYNAMIC_WEIGHTS
        info["n_docs"] = len(self.doc_ids)
        return info
