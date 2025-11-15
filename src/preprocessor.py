#preprocessor.py
import re
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize
from typing import List, Tuple, Dict, Any
import logging
from config import Config

# Загрузка данных для NLTK (только при первом запуске)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


class DataPreprocessor:
    """Класс для предобработки текстовых данных"""

    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Регулярные выражения для очистки текста
        self.clean_patterns = [
            (r'\s+', ' '),  # множественные пробелы
            (r'\n+', ' '),  # множественные переносы
        ]

        # Паттерны для сохранения
        self.preserve_patterns = [
            r'[\U0001F600-\U0001F64F]',  # смайлики
            r'[\U0001F300-\U0001F5FF]',  # символы и пиктограммы
            r'[\U0001F680-\U0001F6FF]',  # транспорт и карты
            r'[\U0001F1E0-\U0001F1FF]',  # флаги
        ]

    def clean_text(self, text: str) -> str:
        """Очистка текста с сохранением смайликов и обезличенных номеров"""
        if not isinstance(text, str) or not text.strip():
            return ""

        # Сохраняем смайлики перед общей очисткой
        preserved = {}
        for i, pattern in enumerate(self.preserve_patterns):
            matches = re.finditer(pattern, text)
            for j, match in enumerate(matches):
                placeholder = f"__PRESERVE_{i}_{j}__"
                preserved[placeholder] = match.group()
                text = text.replace(match.group(), placeholder)

        # Базовая очистка
        text = text.lower().strip()

        # Применяем паттерны очистки
        for pattern, replacement in self.clean_patterns:
            text = re.sub(pattern, replacement, text)

        # Восстанавливаем сохраненные элементы
        for placeholder, original in preserved.items():
            text = text.replace(placeholder, original)

        return text

    def smart_chunking(self, text: str, chunk_size: int = None,
                       chunk_overlap: int = None) -> List[str]:
        """
        Умное чанкование с учетом структуры предложений
        """
        if chunk_size is None:
            chunk_size = self.config.CHUNK_SIZE
        if chunk_overlap is None:
            chunk_overlap = self.config.CHUNK_OVERLAP

        if len(text) < self.config.MIN_CHUNK_LENGTH:
            return [text]

        # Токенизация на предложения
        sentences = sent_tokenize(text, language='russian')

        chunks = []
        current_chunk = ""
        current_length = 0

        for sentence in sentences:
            sentence_length = len(sentence.split())

            # Если предложение слишком длинное, разбиваем его
            if sentence_length > chunk_size:
                words = sentence.split()
                sub_sentences = [' '.join(words[i:i + chunk_size])
                                 for i in range(0, len(words), chunk_size - chunk_overlap)]
                for sub_sentence in sub_sentences:
                    if current_length + len(sub_sentence.split()) > chunk_size and current_chunk:
                        chunks.append(current_chunk.strip())
                        # Создаем перекрывающийся чанк
                        overlap_words = current_chunk.split()[-chunk_overlap:]
                        current_chunk = ' '.join(overlap_words + [sub_sentence])
                        current_length = len(overlap_words) + len(sub_sentence.split())
                    else:
                        current_chunk += " " + sub_sentence if current_chunk else sub_sentence
                        current_length += len(sub_sentence.split())
            else:
                # Обычное предложение
                if current_length + sentence_length > chunk_size and current_chunk:
                    chunks.append(current_chunk.strip())
                    # Создаем перекрывающийся чанк
                    overlap_words = current_chunk.split()[-chunk_overlap:]
                    current_chunk = ' '.join(overlap_words + [sentence])
                    current_length = len(overlap_words) + sentence_length
                else:
                    current_chunk += " " + sentence if current_chunk else sentence
                    current_length += sentence_length

        # Добавляем последний чанк
        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def preprocess_documents(self, documents_df: pd.DataFrame) -> List[Tuple[str, str]]:
        """
        Предобработка всех документов
        Возвращает список кортежей (web_id, processed_text)
        """
        processed_docs = []

        for _, row in documents_df.iterrows():
            web_id = str(row['web_id'])
            text = str(row['text'])

            # Очистка текста
            cleaned_text = self.clean_text(text)

            # Ограничение длины для очень длинных документов
            if len(cleaned_text) > self.config.MAX_TEXT_LENGTH:
                cleaned_text = cleaned_text[:self.config.MAX_TEXT_LENGTH]
                self.logger.warning(f"Document {web_id} truncated to {self.config.MAX_TEXT_LENGTH} characters")

            processed_docs.append((web_id, cleaned_text))

        self.logger.info(f"Processed {len(processed_docs)} documents")
        return processed_docs

    def is_quality_chunk(self, text: str) -> bool:
        """Проверяет качество чанка - отфильтровывает технические тексты"""
        if len(text.strip()) < self.config.MIN_CHUNK_LENGTH:
            return False

        words = text.split()
        unique_words = set(words)

        # Фильтр повторяющихся слов (технические тексты)
        if len(unique_words) / len(words) < self.config.MIN_UNIQUE_WORDS_RATIO:
            return False

        # Фильтр слишком коротких предложений (интерфейсные тексты)
        sentences = text.split('.')
        avg_sentence_length = sum(len(sent.split()) for sent in sentences) / len(sentences)
        if avg_sentence_length < 3:
            return False

        # Фильтр шаблонных текстов
        spam_patterns = [
            r'^\s*(далее|продолжить|ещё|еще|загрузить|скачать)\s*$',
            r'\b(?:номер карты|продолжить|далее)\b.*\b(?:номер карты|продолжить|далее)\b',
        ]

        for pattern in spam_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return False

        return True

    def create_chunks(self, processed_docs: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        """Создает чанки с фильтрацией качества"""
        all_chunks = []
        filtered_count = 0

        for web_id, text in processed_docs:
            chunks = self.smart_chunking(text)
            for chunk in chunks:
                if self.is_quality_chunk(chunk):
                    all_chunks.append((web_id, chunk.strip()))
                else:
                    filtered_count += 1

        self.logger.info(f"Created {len(all_chunks)} chunks (filtered {filtered_count} low-quality chunks)")
        return all_chunks