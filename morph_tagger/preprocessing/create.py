"""Модуль для распаковки датасетов из формата .conllu в .parquet.

Предоставляет класс DatasetCreator, который можно использовать:
  1. Программно:
      creator = DatasetCreator()
      creator.run('syntagrus')

  2. Из CLI:
      python -m morph_tagger.cli.datasets.create --dataset syntagrus
"""

from conllu import parse_incr
import pandas as pd
import os
import sys
from dotenv import load_dotenv
from pathlib import Path
import logging
import argparse

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)


class DatasetCreator:
    """Распаковывает датасеты из формата .conllu в .parquet.

    Args:
        syntagrus_version: Версия Syntagrus — '2.3' или '2.16'.
            По умолчанию берётся из переменной окружения SYNTAGRUS_VERSION.

    Примеры:
        >>> creator = DatasetCreator()
        >>> creator.run('syntagrus')

        >>> creator = DatasetCreator(syntagrus_version='2.3')
        >>> creator.run('merged')
    """

    # Списки файлов для каждого датасета
    DATASETS_FILES = {
        'syntagrus': {
            '2.16': {
                'train_list': ['ru_syntagrus-ud-train-a.conllu',
                               'ru_syntagrus-ud-train-b.conllu',
                               'ru_syntagrus-ud-train-c.conllu'],
                'test_list': ['ru_syntagrus-ud-test.conllu',
                              'ru_syntagrus-ud-dev.conllu'],
            },
            '2.3': {
                'train_list': ['ru_syntagrus-ud-train.conllu'],
                'test_list': ['ru_syntagrus-ud-test.conllu',
                              'ru_syntagrus-ud-dev.conllu'],
            },
        },
        'taiga': {
            'train_list': ['ru_taiga-ud-train-a.conllu',
                           'ru_taiga-ud-train-b.conllu',
                           'ru_taiga-ud-train-c.conllu',
                           'ru_taiga-ud-train-d.conllu',
                           'ru_taiga-ud-train-e.conllu'],
            'test_list': ['ru_taiga-ud-dev.conllu',
                          'ru_taiga-ud-test.conllu'],
        }
    }

    SUPPORTED_DATASETS = ('taiga', 'syntagrus', 'merged')


    def __init__(self, syntagrus_version: str = None):
        load_dotenv(dotenv_path=(Path('.') / '.env'))

        self._syntagrus_version = syntagrus_version or os.getenv('SYNTAGRUS_VERSION', '2.16')
        self._datasets_folder_path = os.getenv('DATASETS_FOLDER_PATH')

        self._syntagrus_path = os.getenv('SYNTAGRUS_PATH')
        self._syntagrus_texts_path = os.getenv('SYNTAGRUS_TEXTS_PATH')
        self._taiga_path = os.getenv('TAIGA_PATH')
        self._taiga_texts_path = os.getenv('TAIGA_TEXTS_PATH')

        # Пути для merged-датасета
        self._merged_path = os.path.join(self._datasets_folder_path, 'sintagrus_taiga_merged')
        Path.mkdir(Path(self._merged_path), exist_ok=True)
        self._merged_texts_path = os.path.join(self._merged_path, 'sintagrus_taiga_merged.txt')


    def _resolve_paths(self, dataset: str) -> tuple[str, str]:
        """Возвращает (dataset_path, texts_path) для заданного датасета."""
        if dataset == 'syntagrus':
            return self._syntagrus_path, self._syntagrus_texts_path
        elif dataset == 'taiga':
            return self._taiga_path, self._taiga_texts_path
        elif dataset == 'merged':
            return self._merged_path, self._merged_texts_path
        else:
            raise ValueError(f"Неверный параметр датасета: {dataset}. Допустимые: {self.SUPPORTED_DATASETS}")


    @staticmethod
    def _unfold_nested_elements(df, feat_column: str = "feats") -> pd.DataFrame:
        """Распаковывает вложенные грамматические атрибуты из столбца feats."""
        nested_features = set()

        for idx, row in df.iterrows():
            df_row = row[feat_column]
            if isinstance(df_row, list):
                for item in df_row:
                    if isinstance(item, dict):
                        for feature in item.keys():
                            nested_features.add(feature)

        for feature in nested_features:
            if feature not in df.columns:
                df[feature] = df[feat_column].apply(lambda x: ['None'] * len(x) if isinstance(x, list) else [])

        for idx, row in df.iterrows():
            df_row = row[feat_column]
            if isinstance(df_row, list):
                for dict_idx, item in enumerate(df_row):
                    if isinstance(item, dict):
                        for feature, value in item.items():
                            current_list = df.at[idx, feature]
                            if isinstance(current_list, list) and dict_idx < len(current_list):
                                current_list[dict_idx] = value
                                df.at[idx, feature] = current_list
        return df


    @staticmethod
    def _unpack_conllu(dataset_path: str,
                       train_list: list[str],
                       test_list: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Распаковывает .conllu файлы и возвращает (train_df, dev_df, test_df)."""
        train_df = pd.DataFrame()
        test_df = pd.DataFrame()
        dev_df = pd.DataFrame()


        def _build_df(file_paths: list[str], accumulator: pd.DataFrame, is_test: bool = False, is_dev: bool = False):
            for datapath in file_paths:
                with open(os.path.join(dataset_path, datapath), 'r', encoding='utf-8') as data_file:
                    parsed_list = list(parse_incr(data_file))

                temp_df = pd.DataFrame({
                    'source_text': [seq.metadata['text'] for seq in parsed_list],
                    'source_words': [[str(token) for token in seq] for seq in parsed_list],
                    'lemmas': [[g['lemma'] for g in seq] for seq in parsed_list],
                    'upos': [[g['upos'] for g in seq] for seq in parsed_list],
                    'xpos': [[g['xpos'] for g in seq] for seq in parsed_list],
                    'feats': [[g['feats'] for g in seq] for seq in parsed_list],
                    'head': [[g['head'] for g in seq] for seq in parsed_list],
                    'deprel': [[g['deprel'] for g in seq] for seq in parsed_list],
                    'deps': [[g['deps'] for g in seq] for seq in parsed_list],
                    'misc': [[g['misc'] for g in seq] for seq in parsed_list],
                }, dtype='object')
                accumulator = pd.concat((accumulator, temp_df))
            return accumulator

        train_df = _build_df(train_list, train_df)

        for datapath in test_list:
            with open(os.path.join(dataset_path, datapath), 'r', encoding='utf-8') as data_file:
                parsed_list = list(parse_incr(data_file))

            temp_df = pd.DataFrame({
                'source_text': [seq.metadata['text'] for seq in parsed_list],
                'source_words': [[str(token) for token in seq] for seq in parsed_list],
                'lemmas': [[g['lemma'] for g in seq] for seq in parsed_list],
                'upos': [[g['upos'] for g in seq] for seq in parsed_list],
                'xpos': [[g['xpos'] for g in seq] for seq in parsed_list],
                'feats': [[g['feats'] for g in seq] for seq in parsed_list],
                'head': [[g['head'] for g in seq] for seq in parsed_list],
                'deprel': [[g['deprel'] for g in seq] for seq in parsed_list],
                'deps': [[g['deps'] for g in seq] for seq in parsed_list],
                'misc': [[g['misc'] for g in seq] for seq in parsed_list],
            }, dtype='object')

            if 'test' in datapath:
                test_df = temp_df
            else:
                dev_df = temp_df

        return train_df, dev_df, test_df


    @staticmethod
    def _check_column_consistency(dfs: dict) -> None:
        """Проверяет, что все датафреймы имеют одинаковые колонки."""
        names = list(dfs.keys())
        for i, name_a in enumerate(names):
            df_a = dfs[name_a]
            for name_b in names[i + 1:]:
                df_b = dfs[name_b]
                for col in df_a.columns:
                    if col not in df_b.columns:
                        logging.warning(
                            f'Столбец {col} из {name_a} отсутствует в {name_b}')
                for col in df_b.columns:
                    if col not in df_a.columns:
                        logging.warning(f'Столбец {col} из {name_b} отсутствует в {name_a}')


    def run(self, dataset: str) -> None:
        """Выполняет полную распаковку датасета.

        Args:
            dataset: 'taiga', 'syntagrus' или 'merged'.
        """
        logging.info(f'Текущий датасет для распаковки: {dataset}')

        dataset_path, texts_path = self._resolve_paths(dataset)

        logging.info('Распаковка conllu и создание датафреймов...')

        if dataset == 'taiga':
            train_df, dev_df, test_df = self._unpack_conllu(dataset_path,
                                                            self.DATASETS_FILES['taiga']['train_list'],
                                                            self.DATASETS_FILES['taiga']['test_list'])
        elif dataset == 'syntagrus':
            train_df, dev_df, test_df = self._unpack_conllu(dataset_path,
                                                            self.DATASETS_FILES['syntagrus'][self._syntagrus_version]['train_list'],
                                                            self.DATASETS_FILES['syntagrus'][self._syntagrus_version]['test_list'])
        elif dataset == 'merged':
            syn_train, syn_dev, syn_test = self._unpack_conllu(self._syntagrus_path,
                                                               self.DATASETS_FILES['syntagrus'][self._syntagrus_version]['train_list'],
                                                               self.DATASETS_FILES['syntagrus'][self._syntagrus_version]['test_list'])
            
            tai_train, tai_dev, tai_test = self._unpack_conllu(self._taiga_path,
                                                               self.DATASETS_FILES['taiga']['train_list'],
                                                               self.DATASETS_FILES['taiga']['test_list'])
            train_df = pd.concat([syn_train, tai_train], axis=0, ignore_index=True)
            dev_df = pd.concat([syn_dev, tai_dev], axis=0, ignore_index=True)
            test_df = pd.concat([syn_test, tai_test], axis=0, ignore_index=True)

        # Обработка вложенных элементов
        train_df = self._unfold_nested_elements(train_df.reset_index(drop=True))
        train_df['head'] = train_df['head'].apply(lambda x: [str(num) for num in x])
        
        test_df = self._unfold_nested_elements(test_df.reset_index(drop=True))
        test_df['head'] = test_df['head'].apply(lambda x: [str(num) for num in x])
        
        dev_df = self._unfold_nested_elements(dev_df.reset_index(drop=True))
        dev_df['head'] = dev_df['head'].apply(lambda x: [str(num) for num in x])

        logging.info(f'Длина тренировочного датасета: {len(train_df)}')
        logging.info(f'Длина тестового датасета: {len(test_df)}')
        logging.info(f'Длина валидационного датасета: {len(dev_df)}')

        # Проверка согласованности колонок
        self._check_column_consistency({'train': train_df, 'test': test_df, 'dev': dev_df})

        # Сохранение текстового файла
        logging.info('Создание текстового файла всех исходных текстов')
        with open(texts_path, 'w', encoding='utf-8') as f:
            for raw_text in train_df['source_text']:
                f.write(raw_text + '\n')
            for raw_text in test_df['source_text']:
                f.write(raw_text + '\n')
            for raw_text in dev_df['source_text']:
                f.write(raw_text + '\n')

        # Сохранение .parquet
        logging.info('Сохранение полученных датасетов в формате .parquet')
        for split_name, split_df in [('train', train_df), ('test', test_df), ('dev', dev_df)]:
            split_df.reset_index().drop(columns=["index"]).to_parquet(os.path.join(dataset_path, f'{dataset}_{split_name}.parquet'), engine='fastparquet', index=False)

        logging.info('Готово!')


# CLI entry point
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Распаковка датасетов из формата conllu')
    parser.add_argument(
        '--dataset',
        type=str,
        default='merged',
        choices=DatasetCreator.SUPPORTED_DATASETS,
        help='Какой датасет подготовить: taiga, syntagrus или merged',
        required=True,
    )
    parser.add_argument(
        '--syntagrus-version',
        type=str,
        choices=['2.3', '2.16'],
        default=None,
        help='Версия Syntagrus. По умолчанию берётся из .env.',
    )
    return parser


def main():
    parser = _build_parser()
    args = parser.parse_args()

    creator = DatasetCreator(syntagrus_version=getattr(args, 'syntagrus_version', None))
    creator.run(args.dataset)


if __name__ == '__main__':
    main()