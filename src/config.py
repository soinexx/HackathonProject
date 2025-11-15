# config.py - ОБНОВЛЕННАЯ ВЕРСИЯ
import os
from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class Config:
    """Конфигурация параметров RAG-системы с улучшениями"""

    # Параметры чанкования - ОБНОВЛЕНО
    CHUNK_SIZE: int = 320      # В токенах примерно
    CHUNK_OVERLAP: int = 64    # В токенах примерно

    # Фильтрация технических текстов
    MIN_UNIQUE_WORDS_RATIO: float = 0.3

    # Модель эмбеддингов - ОБНОВЛЕНО на более современную
    # Добавляем параметр для моделей требующих trust_remote_code
    TRUST_REMOTE_CODE: bool = True
    EMBEDDING_MODEL: str = "Alibaba-NLP/gte-multilingual-base"

    # Кросс-энкодер для реранжирования - НОВОЕ
    USE_RERANKER: bool = True
    RERANKER_MODEL: str = "BAAI/bge-reranker-v2-m3"

    # Динамические веса - НОВОЕ
    USE_DYNAMIC_WEIGHTS: bool = True
    SHORT_QUERY_TOKENS: int = 5
    BM25_BONUS_SHORT: float = 0.20
    BM25_BONUS_HAS_DIGITS: float = 0.10
    BM25_BONUS_KEYWORDS: List[str] = field(default_factory=lambda: [
        "счет", "карта", "перевод", "платеж", "ипотека", "кредит", "валюта", "комисси", "смс", "уведомление", "оповещение"
    ])

    # Бустинг по URL и заголовкам - НОВОЕ
    URL_POSITIVE_PATTERNS: List[str] = field(default_factory=lambda: [
        r"/a-?club", r"\bpremium\b", r"wealth", r"privilege", r"/investment"
    ])
    URL_NEGATIVE_PATTERNS: List[str] = field(default_factory=lambda: [
        r"/vacanc", r"/news", r"/press", r"/cookies", r"/privacy"
    ])
    URL_POSITIVE_BOOST: float = 0.07
    URL_NEGATIVE_PENALTY: float = 0.10
    TITLE_BOOST: float = 0.10

    # Поисковые параметры
    SEARCH_TOP_K_CHUNKS: int = 20
    FINAL_TOP_K_DOCS: int = 5

    # Обработка текста
    MAX_TEXT_LENGTH: int = 8000
    MIN_CHUNK_LENGTH: int = 50

    # ChromaDB настройки
    CHROMA_PERSIST_DIR: str = "chroma_db"
    CHROMA_COLLECTION_NAME: str = "alfabank_documents"

    # Пути к данным
    PROJECT_ROOT: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR: str = os.path.join(PROJECT_ROOT, "data")
    MODELS_DIR: str = os.path.join(PROJECT_ROOT, "models")

    # Имена файлов
    TFIDF_INDEX_FILE: str = "tfidf_index.pkl"
    QUESTIONS_FILE: str = "questions_clean.csv"
    WEBSITES_FILE: str = "websites_updated.csv"
    SAMPLE_SUBMISSION: str = "sample_submission.csv"
    OUTPUT_FILE: str = "submit.csv"

    def get_paths(self) -> Dict[str, str]:
        """Возвращает полные пути к файлам"""
        return {
            'questions': os.path.join(self.DATA_DIR, self.QUESTIONS_FILE),
            'websites': os.path.join(self.DATA_DIR, self.WEBSITES_FILE),
            'sample_submission': os.path.join(self.DATA_DIR, self.SAMPLE_SUBMISSION),
            'output': os.path.join(self.PROJECT_ROOT, self.OUTPUT_FILE),
            'models_dir': self.MODELS_DIR,
            'chroma_db': os.path.join(self.PROJECT_ROOT, self.CHROMA_PERSIST_DIR)
        }

    def create_directories(self):
        """Создает необходимые директории"""
        for directory in [self.DATA_DIR, self.MODELS_DIR, self.get_paths()['chroma_db']]:
            os.makedirs(directory, exist_ok=True)

    def check_data_files(self) -> bool:
        """Проверяет существование необходимых файлов"""
        paths = self.get_paths()
        missing_files = []

        for file_type, path in paths.items():
            if file_type in ['questions', 'websites', 'sample_submission']:
                if not os.path.exists(path):
                    missing_files.append(f"{file_type}: {path}")

        if missing_files:
            print("❌ Missing files:")
            for missing in missing_files:
                print(f"   - {missing}")
            return False

        print("✅ All data files found!")
        return True