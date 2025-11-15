# query_utils.py
import re

# фразы-приветствия и вежливости
BOILERPLATE_PHRASES = [
    "здравствуйте",
    "добрый день",
    "добрый вечер",
    "доброе утро",
    "подскажите пожалуйста",
    "скажите пожалуйста",
    "подскажите",
    "скажите",
]

# типичные "хвосты" в начале
BOILERPLATE_PREFIXES = [
    "у меня",
    "я не",
    "я могу",
    "могу ли",
    "можно ли",
    "если я",
]


def normalize_query_text(query: str) -> str:
    """
    Убираем из начала запроса приветствия/вежливые фразы,
    оставляем "суть" вопроса.
    """
    orig = str(query).strip()
    q = re.sub(r'\s+', ' ', orig)

    lower = q.lower()

    # убираем приветствия
    for phrase in BOILERPLATE_PHRASES:
        if lower.startswith(phrase):
            pattern = r'^' + re.escape(phrase) + r'[,!.\s]+'
            q = re.sub(pattern, '', q, flags=re.IGNORECASE).strip()
            lower = q.lower()

    # убираем "у меня / я не / могу ли..." в начале
    for phrase in BOILERPLATE_PREFIXES:
        if lower.startswith(phrase):
            pattern = r'^' + re.escape(phrase) + r'[,!.\s]+'
            q = re.sub(pattern, '', q, flags=re.IGNORECASE).strip()
            lower = q.lower()

    # чтобы не получить пустую строку
    return q if q else orig
