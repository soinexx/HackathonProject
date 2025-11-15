class QueryExpander:
    """Расширитель запросов для улучшения поиска"""

    def __init__(self):
        self.synonyms = {
            "номер счета": ["счет", "расчетный счет", "банковский счет", "р/с"],
            "кредитная карта": ["кредитка", "карта кредитная", "банковская карта"],
            "смс": ["смс-уведомления", "смс оповещения", "смс информирование"],
            "перевод денег": ["перевод средств", "денежный перевод", "перевести деньги",
                              "отправить деньги", "перевести на карту", "сделать перевод", "перевод"],
            "бизнес счет": ["счет для бизнеса", "расчетный счет бизнес", "р/с для ип",
                            "счет для ип", "счет для ооо", "коммерческий счет", "счет предпринимателя"],
            "дебетовая карта": ["дебетовая", "карта дебетовая", "пластиковая карта"],
        }

    def expand_query(self, query: str) -> str:
        """Расширяет запрос синонимами"""
        original_query = query.lower()
        expanded_terms = [original_query]

        for term, synonyms in self.synonyms.items():
            if term in original_query:
                expanded_terms.extend(synonyms)

        # Убираем дубликаты и объединяем
        unique_terms = list(set(expanded_terms))
        return " ".join(unique_terms)


# Интегрируем в retriever_tfidf.py
def search(self, query: str, top_k: int = None, n_results: int = None) -> List[str]:
    # Добавляем расширение запроса
    from query_expander import QueryExpander
    expander = QueryExpander()
    expanded_query = expander.expand_query(query)

    # Логируем для отладки
    if query != expanded_query:
        self.logger.debug(f"Query expanded: '{query}' -> '{expanded_query}'")

