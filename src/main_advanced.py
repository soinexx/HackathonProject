# main_advanced.py
import pandas as pd
import logging
import sys
import os
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config import Config
from retriever_advanced import AdvancedHybridRetrieval


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )


def load_data(config: Config):
    paths = config.get_paths()

    if not config.check_data_files():
        raise FileNotFoundError("Some data files are missing.")

    try:
        questions_df = pd.read_csv(paths['questions'])
        websites_df = pd.read_csv(paths['websites'])
        logging.info(f"Loaded {len(questions_df)} questions and {len(websites_df)} websites")
        return questions_df, websites_df
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise


def main():
    setup_logging()
    logger = logging.getLogger(__name__)

    try:
        config = Config()
        config.create_directories()

        logger.info("🚀 Starting ADVANCED HYBRID RAG Pipeline...")

        # Загрузка данных
        questions_df, websites_df = load_data(config)

        # Инициализация и построение индекса
        logger.info("🛠 Building advanced hybrid index...")
        retriever = AdvancedHybridRetrieval(config)
        retriever.build_index(websites_df)

        # Информация об индексе
        index_info = retriever.get_index_info()
        logger.info(f"📊 Advanced index info: {index_info}")

        # Обработка вопросов
        logger.info("🔍 Processing questions with advanced hybrid search...")
        results = []

        for _, row in tqdm(questions_df.iterrows(), total=len(questions_df), desc="Advanced Search"):
            q_id = row['q_id']
            query = row['query']

            try:
                web_list = retriever.search(query)
                results.append({'q_id': q_id, 'web_list': web_list})
            except Exception as e:
                logger.error(f"Error with question {q_id}: {e}")
                results.append({'q_id': q_id, 'web_list': []})

        # Сохранение результатов
        submission_df = pd.DataFrame(results)
        submission_df['web_list'] = submission_df['web_list'].apply(
            lambda x: '[' + ','.join(map(str, x)) + ']' if x else '[]'
        )

        output_path = config.get_paths()['output'].replace('.csv', '_advanced.csv')
        submission_df.to_csv(output_path, index=False)

        # Статистика
        success_rate = (len(results) - submission_df[submission_df['web_list'] == '[]'].shape[0]) / len(results) * 100
        logger.info(f"🎉 ADVANCED Pipeline completed!")
        logger.info(f"📈 Success rate: {success_rate:.1f}%")
        logger.info(f"💾 Advanced submission saved to: {output_path}")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()