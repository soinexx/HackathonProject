#advanced_preprocessor.py
import re
from typing import List
import logging


class AdvancedPreprocessor:
    """Агрессивная предобработка для максимизации Hit@5"""

    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Ключевые банковские термины - ПРИОРИТЕТ
        self.priority_terms = [
            # базовые банковские
            'счет', 'счёт', 'номер счета', 'номер карты',
            'карта', 'карты', 'кредит', 'кредитк', 'займ',
            'дебет', 'дебетов', 'вклад', 'депозит', 'ипотек',

            # операции
            'перевод', 'перевести', 'платеж', 'платёж',
            'оплата', 'оплатить', 'списание', 'списали',
            'комисси', 'штраф', 'процент', 'ставк',

            # каналы и сервисы
            'смс', 'sms', 'уведомлен', 'оповещен',
            'мобильный банк', 'мобильное приложение',
            'приложение', 'интернет банк', 'онлайн банк',
            'альфа', 'alfabank', 'альфа-банк',

            # продукты и бренды
            'mir', 'visa', 'mastercard',
            'пин код', 'pin', 'cvv', 'cvc',
        ]

        # СТРОГИЕ фильтры для удаления мусора
        self.spam_patterns = [
            r'©.*', r'© \d{4}.*', r'АО.*Альфа-Банк.*',
            r'Генеральная лицензия.*', r'Официальный сайт.*',
            r'[\d\s]*Кб[\d\s]*',  # размеры файлов
            r'Скачать.*', r'Загрузить.*', r'Подробнее.*',
            r'^\s*$', r'^\W+$',  # пустые или только спецсимволы
        ]

    def _basic_clean(self, text: str) -> str:
        text = str(text).lower()
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def extract_key_sections(self, text: str) -> str:
        """Выделяет ТОЛЬКО ключевые разделы текста"""
        key_sections = []

        text = self._basic_clean(text)
        lines = text.split('\n')
        for line in lines:
            line_clean = line.strip()
            if any(term in line_clean.lower() for term in self.priority_terms):
                if 10 < len(line_clean) < 200:
                    key_sections.append(line_clean)

        if key_sections:
            return ' '.join(key_sections[:5])  # максимум 5 ключевых секций

        # fallback — начало текста
        return text[:800]

    def aggressive_chunking(self, text: str) -> List[str]:
        """Менее агрессивное чанкование"""
        original_length = len(text.split())
        key_text = self.extract_key_sections(text)
        words = key_text.split()
        chunks: List[str] = []

        # УВЕЛИЧИВАЕМ размер чанка и добавляем overlap
        step = self.config.MAX_WORDS_IN_CHUNK - 30  # overlap 30 слов
        for i in range(0, len(words), step):
            chunk_words = words[i:i + self.config.MAX_WORDS_IN_CHUNK]
            if len(chunk_words) < self.config.MIN_WORDS_IN_CHUNK:
                continue

            chunk = ' '.join(chunk_words)
            chunks.append(chunk)

        result_chunks = chunks[:15]  # увеличиваем лимит

        # ЛОГИРОВАНИЕ для отладки
        if len(result_chunks) > 0:
            self.logger.debug(
                f"Created {len(result_chunks)} chunks from {original_length} words (reduced to {len(words)})")

        return result_chunks

    def is_high_quality_chunk(self, chunk: str) -> bool:
        """БОЛЕЕ ЛОЯЛЬНАЯ проверка качества чанка"""
        # Убираем строгие проверки на банковские термины
        if not chunk.strip():
            return False

        # Сохраняем только базовую проверку на спам
        for pattern in self.spam_patterns:
            if re.search(pattern, chunk, re.IGNORECASE):
                return False

        # СНИЖАЕМ порог уникальности слов
        words = chunk.split()
        if len(words) < 5:  # минимальная длина
            return False

        unique_words = set(words)
        if len(unique_words) / len(words) < 0.25:  # было 0.35
            return False

        return True  # принимаем больше чанков
