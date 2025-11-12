import os
from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class Config:
    """Конфигурация параметров RAG-системы с ChromaDB"""

    # Параметры чанкования
    CHUNK_SIZE: int = 400
    CHUNK_OVERLAP: int = 40

    # Модель эмбеддингов
    EMBEDDING_MODEL: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

    # Поисковые параметры
    SEARCH_TOP_K_CHUNKS: int = 20
    FINAL_TOP_K_DOCS: int = 5

    # Обработка текста
    MAX_TEXT_LENGTH: int = 10000
    MIN_CHUNK_LENGTH: int = 50

    # ChromaDB настройки
    CHROMA_PERSIST_DIR: str = "chroma_db"
    CHROMA_COLLECTION_NAME: str = "alfabank_documents"

    # Пути к данным - ИСПРАВЛЕНО: используем абсолютные пути
    PROJECT_ROOT: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR: str = os.path.join(PROJECT_ROOT, "data")
    MODELS_DIR: str = os.path.join(PROJECT_ROOT, "models")

    # Имена файлов
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