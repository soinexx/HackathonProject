import sys
import os

sys.path.append('src')

from config import Config
from preprocessor import DataPreprocessor
from embedder import EmbeddingModel
from retriever import AlfabankRetrieval


def test_chroma_rag():
    print("🧪 Testing ChromaDB RAG system...")

    # Тест конфигурации
    config = Config()
    print("✅ Config loaded")

    # Тест препроцессинга
    preprocessor = DataPreprocessor(config)
    test_text = "Привет! Это тестовый текст с номером +00000000 😊"
    cleaned = preprocessor.clean_text(test_text)
    print(f"✅ Text cleaning: {cleaned}")

    # Тест эмбеддингов
    embedder = EmbeddingModel(config)
    test_embeddings = embedder.encode(["тестовый текст"])
    print(f"✅ Embeddings: shape {test_embeddings.shape}")

    # Тест инициализации ChromaDB
    retriever = AlfabankRetrieval(config)
    retriever.initialize_chroma("test_chroma_db")
    print("✅ ChromaDB initialized")

    # Тест информации о коллекции
    info = retriever.get_collection_info()
    print(f"✅ Collection info: {info}")

    print("🎉 ChromaDB RAG system working!")


if __name__ == "__main__":
    test_chroma_rag()