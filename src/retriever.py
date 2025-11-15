#retriever.py
import chromadb
import pandas as pd
import logging
from typing import List, Dict, Any, Optional
import os
import uuid
from tqdm import tqdm

from config import Config
from preprocessor import DataPreprocessor
from embedder import EmbeddingModel


class AlfabankRetrieval:
    """Класс для поиска релевантных документов с использованием ChromaDB"""

    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Инициализация компонентов
        self.preprocessor = DataPreprocessor(config)
        self.embedder = EmbeddingModel(config)

        # Инициализация ChromaDB
        self.chroma_client = None
        self.collection = None
        self.collection_name = "alfabank_documents"

    def initialize_chroma(self, persist_directory: str = "chroma_db"):
        """Инициализация ChromaDB с проверкой существования коллекции"""
        try:
            from sentence_transformers import SentenceTransformer
            import chromadb.utils.embedding_functions as embedding_functions

            # Создаем embedding function совместимую с новой версией ChromaDB
            sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
            )

            self.chroma_client = chromadb.PersistentClient(path=persist_directory)

            # Пытаемся получить существующую коллекцию
            try:
                self.collection = self.chroma_client.get_collection(
                    name=self.collection_name,
                    embedding_function=sentence_transformer_ef
                )
                self.logger.info(f"Loaded existing collection: {self.collection_name}")
            except Exception:
                # Если коллекции нет - создаем новую
                self.collection = self.chroma_client.create_collection(
                    name=self.collection_name,
                    embedding_function=sentence_transformer_ef,
                    metadata={"description": "Alfabank documents and FAQs"}
                )
                self.logger.info(f"Created new collection: {self.collection_name}")

        except Exception as e:
            self.logger.error(f"Error initializing ChromaDB: {e}")
            raise

    def clear_collection(self):
        """Очищает текущую коллекцию"""
        if self.collection:
            all_ids = self.collection.get()['ids']
            if all_ids:
                self.collection.delete(ids=all_ids)
                self.logger.info(f"Cleared {len(all_ids)} records from collection")

    def build_index(self, documents_df: pd.DataFrame, force_rebuild: bool = False) -> None:
        """Построение индекса документов в ChromaDB"""
        self.logger.info("Starting index building process with ChromaDB...")

        # Инициализируем ChromaDB
        self.initialize_chroma()

        # Проверяем, нужно ли перестраивать индекс
        if not force_rebuild and self.collection.count() > 0:
            self.logger.info("Index already exists. Skipping build. Use force_rebuild=True to rebuild.")
            return

        # Очищаем только если force_rebuild=True
        if force_rebuild:
            self.clear_collection()
        else:
            # Если есть данные и не требуется пересборка - выходим
            if self.collection.count() > 0:
                self.logger.info("Using existing index. Collection already contains data.")
                return

        # 1. Предобработка документов
        self.logger.info("Step 1: Preprocessing documents...")
        processed_docs = self.preprocessor.preprocess_documents(documents_df)

        # 2. Создание чанков
        self.logger.info("Step 2: Creating chunks...")
        chunks = self.preprocessor.create_chunks(processed_docs)

        # 3. Подготовка данных для ChromaDB
        self.logger.info("Step 3: Preparing data for ChromaDB...")

        documents = []
        metadatas = []
        ids = []

        for i, (doc_id, chunk_text) in enumerate(tqdm(chunks, desc="Preparing chunks")):
            # Генерируем уникальный ID для чанка
            chunk_id = str(uuid.uuid4())

            documents.append(chunk_text)
            metadatas.append({
                "document_id": doc_id,
                "chunk_index": i,
                "source": "alfabank_website"
            })
            ids.append(chunk_id)

        # 4. Добавление документов в ChromaDB
        self.logger.info("Step 4: Adding documents to ChromaDB...")

        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )

        self.logger.info(f"Index built successfully. Added {len(documents)} chunks to ChromaDB")

    def search(self, query: str, top_k: int = None, n_results: int = None) -> List[str]:
        """Поиск релевантных документов для запроса"""
        if top_k is None:
            top_k = self.config.FINAL_TOP_K_DOCS
        if n_results is None:
            n_results = self.config.SEARCH_TOP_K_CHUNKS

        if self.collection is None:
            raise ValueError("Collection not initialized. Call build_index() first.")

        # Поиск в ChromaDB
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results
            )

            # Агрегация результатов на уровень документов
            doc_scores = {}

            if results['metadatas'] and len(results['metadatas']) > 0:
                metadatas = results['metadatas'][0]
                distances = results['distances'][0] if results['distances'] else []

                for i, metadata in enumerate(metadatas):
                    if metadata and 'document_id' in metadata:
                        doc_id = metadata['document_id']
                        distance = distances[i] if i < len(distances) else 1.0
                        similarity = 1.0 / (1.0 + distance)

                        if doc_id not in doc_scores or similarity > doc_scores[doc_id]:
                            doc_scores[doc_id] = similarity

            # Сортировка по убыванию схожести и возврат топ-k
            sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
            top_docs = [doc_id for doc_id, score in sorted_docs[:top_k]]

            self.logger.debug(f"Query: '{query}' -> Top docs: {top_docs}")
            return top_docs

        except Exception as e:
            self.logger.error(f"Error during search: {e}")
            return []

    def batch_search(self, queries: List[str], top_k: int = None) -> List[List[str]]:
        """Пакетный поиск для нескольких запросов"""
        results = []
        for query in tqdm(queries, desc="Processing queries"):
            results.append(self.search(query, top_k))
        return results

    def get_collection_info(self) -> Dict[str, Any]:
        """Получение информации о коллекции"""
        if self.collection is None:
            return {}

        return {
            "name": self.collection.name,
            "count": self.collection.count(),
            "metadata": self.collection.metadata
        }

    def delete_collection(self):
        """Удаление коллекции (для очистки)"""
        if self.chroma_client and self.collection:
            self.chroma_client.delete_collection(name=self.collection_name)
            self.collection = None
            self.logger.info(f"Collection {self.collection_name} deleted")