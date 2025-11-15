# test_tfidf_quality.py
import sys
import os
import pandas as pd

sys.path.append('src')

from config import Config
from retriever_tfidf import TFIDFRetrieval


def test_tfidf_quality():
    """Тестирование качества TF-IDF поиска"""

    config = Config()

    print("🧪 Testing TF-IDF Search Quality")
    print("=" * 60)

    # Загружаем данные
    websites_df = pd.read_csv(config.get_paths()['websites'])

    # Создаем словарь для отображения web_id в текст
    website_dict = {str(row['web_id']): str(row['text'])[:200] + "..." for _, row in websites_df.iterrows()}

    # Инициализируем и строим индекс
    retriever = TFIDFRetrieval(config)
    retriever.build_index(websites_df)

    # Тестовые запросы
    test_queries = [
        "номер счета",
        "кредитная карта",
        "смс уведомления",
        "дебетовая карта",
        "ипотека",
        "вклад",
        "перевод денег",
        "бизнес счет"
    ]

    for query in test_queries:
        print(f"\n📝 Query: '{query}'")
        print("-" * 40)

        results = retriever.search(query, top_k=3)

        for i, doc_id in enumerate(results, 1):
            doc_preview = website_dict.get(doc_id, "Document not found")
            print(f"  {i}. doc_{doc_id}: {doc_preview}")

        # Оценка релевантности
        print("  💬 Manual assessment: [GOOD/AVERAGE/POOR]")

    # Информация об индексе
    index_info = retriever.get_index_info()
    print(f"\n📊 Index info: {index_info}")


if __name__ == "__main__":
    test_tfidf_quality()