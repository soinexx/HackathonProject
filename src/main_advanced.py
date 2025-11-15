#main_advanced.py
import pandas as pd
import logging
import sys
import os
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config import Config
from retriever_advanced import AdvancedHybridRetrieval
from query_expander import EnhancedQueryExpander


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )


def main():
    setup_logging()
    logger = logging.getLogger(__name__)

    try:
        config = Config()
        config.create_directories()

        logger.info("🚀 Starting OPTIMIZED RAG Pipeline for Hit@5 ≥ 0.40...")

        # Загрузка данных
        paths = config.get_paths()
        questions_df = pd.read_csv(paths['questions'])
        websites_df = pd.read_csv(paths['websites'])
        logger.info(f"📊 Loaded {len(questions_df)} questions and {len(websites_df)} websites")

        # Инициализация улучшенной системы
        logger.info("🛠 Building optimized hybrid index...")
        retriever = AdvancedHybridRetrieval(config)
        retriever.build_index(websites_df)

        # Инициализация расширителя запросов
        expander = EnhancedQueryExpander()

        # Обработка вопросов с улучшенным поиском
        logger.info("🔍 Processing questions with enhanced search...")
        results = []

        for _, row in tqdm(questions_df.iterrows(), total=len(questions_df), desc="Optimized Search"):
            q_id = row['q_id']
            query = row['query']

            try:
                # Расширяем запрос
                expanded_query = expander.smart_expand(query) if config.USE_ENHANCED_EXPANSION else query

                if expanded_query != query:
                    logger.debug(f"Query expanded: '{query}' -> '{expanded_query}'")

                # Поиск с расширенным запросом
                web_list = retriever.search(expanded_query)
                results.append({'q_id': q_id, 'web_list': web_list})

            except Exception as e:
                logger.warning(f"⚠️ Error with expanded query for {q_id}: {e}, trying without expansion")
                try:
                    # Fallback: поиск без расширения
                    web_list = retriever.search(query)
                    results.append({'q_id': q_id, 'web_list': web_list})
                except Exception as e2:
                    logger.error(f"❌ Complete failure for question {q_id}: {e2}")
                    results.append({'q_id': q_id, 'web_list': []})

        # Сохранение результатов
        submission_df = pd.DataFrame(results)
        submission_df['web_list'] = submission_df['web_list'].apply(
            lambda x: '[' + ','.join(map(str, x)) + ']' if x else '[]'
        )

        output_path = config.get_paths()['output'].replace('.csv', '_optimized.csv')
        submission_df.to_csv(output_path, index=False)

        # Расширенная статистика
        empty_results = submission_df[submission_df['web_list'] == '[]'].shape[0]
        success_rate = (len(results) - empty_results) / len(results) * 100

        avg_results = submission_df['web_list'].apply(
            lambda x: len(x.strip('[]').split(',')) if x != '[]' else 0
        ).mean()

        logger.info(f"🎉 OPTIMIZED Pipeline completed!")
        logger.info(f"📈 Success rate: {success_rate:.1f}%")
        logger.info(f"📊 Average results per query: {avg_results:.1f}")
        logger.info(f"💾 Optimized submission saved to: {output_path}")

    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()