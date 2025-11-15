#retriever_tfidf.py
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
import logging
import pickle
import os
from typing import List, Dict, Any
from tqdm import tqdm

from config import Config
from preprocessor import DataPreprocessor
from advanced_preprocessor import AdvancedPreprocessor
from query_utils import normalize_query_text
from query_expander import QueryExpander


class TFIDFRetrieval:
    """Ретривер на основе TF-IDF и BM25"""

    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        #self.preprocessor = DataPreprocessor(config) # больше не нужен для TF-IDF

        self.vectorizer = None
        self.tfidf_matrix = None
        self.bm25 = None
        self.chunk_to_doc_map = {}
        self.chunk_texts = []
        self.tokenized_chunks = []

    def build_index(self, documents_df: pd.DataFrame):
        """Построение TF-IDF и BM25 индексов"""
        self.logger.info("Building TF-IDF and BM25 indexes...")

        # 1. Предобработка и чанкование
        adv = AdvancedPreprocessor(self.config)

        # Сохраняем маппинг и тексты
        self.chunk_texts = []
        self.chunk_to_doc_map = {}

        idx = 0
        for _, row in documents_df.iterrows():
            doc_id = str(row['web_id'])
            text = str(row['text'])

            doc_chunks = adv.aggressive_chunking(text)
            if not doc_chunks:
                # fallback — ключевые секции или начало текста
                fallback = adv.extract_key_sections(text)
                if fallback:
                    doc_chunks = [fallback]

            for ch in doc_chunks:
                self.chunk_texts.append(ch)
                self.chunk_to_doc_map[idx] = doc_id
                idx += 1

        self.logger.info(f"Created {len(self.chunk_texts)} chunks for TF-IDF/BM25")

        # 2. Создание TF-IDF матрицы
        self.logger.info("Creating TF-IDF matrix...")
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 3),  # униграммы, биграммы и триграммы
            max_features=25000,  # топ-25к features
            min_df=3,  # игнорировать редкие слова (встречаются в >=2 документах)
            max_df=0.7,  # игнорировать слишком частые (в <=80% документов)
            stop_words=None,  # не использовать стоп-слова (для русского лучше свои)
            lowercase=False, # уже сделали в препроцессинге
            analyzer='word',  # явно указываем анализ по словам
            token_pattern=r'(?u)\b\w+\b'  # паттерн для русских слов
        )

        self.tfidf_matrix = self.vectorizer.fit_transform(self.chunk_texts)
        self.logger.info(f"TF-IDF matrix shape: {self.tfidf_matrix.shape}")

        # 3. Создание BM25 индекса
        self.logger.info("Creating BM25 index...")
        # Токенизация для BM25 (просто по словам)
        self.tokenized_chunks = [doc.split() for doc in self.chunk_texts]
        self.bm25 = BM25Okapi(self.tokenized_chunks)

        self.logger.info("Indexes built successfully!")

    def search(self, query: str, top_k: int = None, method: str = "hybrid") -> List[str]:
        """
        Поиск с использованием гибридного подхода (TF-IDF + BM25)

        Args:
            query: поисковый запрос
            top_k: количество возвращаемых документов
            method: "tfidf", "bm25", или "hybrid"
        """
        if top_k is None:
            top_k = self.config.FINAL_TOP_K_DOCS

        if self.tfidf_matrix is None or self.bm25 is None:
            raise ValueError("Index not built. Call build_index() first.")

        # нормализуем и расширяем запрос
        norm_query = normalize_query_text(query)
        expander = QueryExpander()
        expanded_query = expander.expand_query(norm_query)

        # Токенизация запроса для BM25
        tokenized_query = expanded_query.split()

        # Вычисление скоров для каждого метода
        scores = {}

        if method in ["tfidf", "hybrid"]:
            query_vec = self.vectorizer.transform([expanded_query])
            tfidf_similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

            if tfidf_similarities.max() > 0:
                tfidf_scores = tfidf_similarities / tfidf_similarities.max()
            else:
                tfidf_scores = tfidf_similarities

            for i, s in enumerate(tfidf_scores):
                scores.setdefault(i, 0.0)
                scores[i] += s * 0.6

        if method in ["bm25", "hybrid"]:
            bm25_scores = self.bm25.get_scores(tokenized_query)

            if bm25_scores.max() > 0:
                bm25_scores_norm = bm25_scores / bm25_scores.max()
            else:
                bm25_scores_norm = bm25_scores

            for i, s in enumerate(bm25_scores_norm):
                scores.setdefault(i, 0.0)
                scores[i] += s * 0.4

        # Агрегация на уровень документов (максимальный скор чанка)
        doc_scores = {}
        search_k = min(self.config.SEARCH_TOP_K_CHUNKS, len(scores))

        # Берем топ чанки по комбинированному скору
        top_chunk_indices = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:search_k]

        for chunk_idx in top_chunk_indices:
            if scores[chunk_idx] > 0:  # только ненулевая схожесть
                doc_id = self.chunk_to_doc_map[chunk_idx]
                score = scores[chunk_idx]

                if doc_id not in doc_scores or score > doc_scores[doc_id]:
                    doc_scores[doc_id] = score

        # Сортировка и возврат топ-k
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        top_docs = [doc_id for doc_id, score in sorted_docs[:top_k]]

        self.logger.debug(f"Query: '{query}' -> Top docs: {top_docs}")
        return top_docs

    def batch_search(self, queries: List[str], top_k: int = None) -> List[List[str]]:
        """Пакетный поиск для нескольких запросов"""
        results = []
        for query in tqdm(queries, desc="Processing queries"):
            results.append(self.search(query, top_k))
        return results

    def save_index(self, path: str):
        """Сохранение индексов на диск"""
        index_data = {
            'vectorizer': self.vectorizer,
            'tfidf_matrix': self.tfidf_matrix,
            'bm25': self.bm25,
            'chunk_to_doc_map': self.chunk_to_doc_map,
            'chunk_texts': self.chunk_texts,
            'tokenized_chunks': self.tokenized_chunks
        }

        with open(path, 'wb') as f:
            pickle.dump(index_data, f)

        self.logger.info(f"Index saved to {path}")

    def load_index(self, path: str):
        """Загрузка индексов с диска"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Index file {path} not found")

        with open(path, 'rb') as f:
            index_data = pickle.load(f)

        self.vectorizer = index_data['vectorizer']
        self.tfidf_matrix = index_data['tfidf_matrix']
        self.bm25 = index_data['bm25']
        self.chunk_to_doc_map = index_data['chunk_to_doc_map']
        self.chunk_texts = index_data['chunk_texts']
        self.tokenized_chunks = index_data['tokenized_chunks']

        self.logger.info(f"Index loaded from {path}")

    def get_index_info(self) -> Dict[str, Any]:
        """Информация о индексе"""
        if self.tfidf_matrix is None:
            return {"status": "Not built"}

        return {
            "status": "Built",
            "chunks_count": len(self.chunk_texts),
            "vocabulary_size": len(self.vectorizer.vocabulary_),
            "matrix_shape": self.tfidf_matrix.shape,
            "unique_documents": len(set(self.chunk_to_doc_map.values()))
        }