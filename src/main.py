import pandas as pd
import logging
import sys
import os
from tqdm import tqdm

# Добавляем src в путь для импортов
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config import Config
from retriever import AlfabankRetrieval


def setup_logging():
    """Настройка логирования"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )


def load_data(config: Config):
    """Загрузка данных с проверкой"""
    paths = config.get_paths()

    # Проверяем существование файлов
    if not config.check_data_files():
        raise FileNotFoundError("Some data files are missing. Please check the paths above.")

    try:
        questions_df = pd.read_csv(paths['questions'])
        websites_df = pd.read_csv(paths['websites'])

        logging.info(f"Loaded {len(questions_df)} questions from {paths['questions']}")
        logging.info(f"Loaded {len(websites_df)} websites from {paths['websites']}")

        return questions_df, websites_df

    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise


def main():
    """Основная функция пайплайна"""
    setup_logging()
    logger = logging.getLogger(__name__)

    try:
        # Инициализация конфигурации
        config = Config()
        config.create_directories()

        logger.info("Starting RAG Pipeline with ChromaDB...")

        # Загрузка данных
        questions_df, websites_df = load_data(config)

        # Инициализация и построение индекса
        logger.info("Initializing retrieval system with ChromaDB...")
        retriever = AlfabankRetrieval(config)

        logger.info("Building search index...")
        retriever.build_index(websites_df)

        # Информация о коллекции
        collection_info = retriever.get_collection_info()
        logger.info(f"Collection info: {collection_info}")

        # Обработка вопросов
        logger.info("Processing questions...")
        results = []

        for _, row in tqdm(questions_df.iterrows(), total=len(questions_df), desc="Searching"):
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

        output_path = config.get_paths()['output']
        submission_df.to_csv(output_path, index=False)
        logger.info(f"Submission saved to {output_path}")
        logger.info(f"Processed {len(results)} questions successfully!")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()