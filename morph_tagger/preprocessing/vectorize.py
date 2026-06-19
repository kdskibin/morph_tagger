"""Модуль для препроцессинга датасетов (токенизация + векторизация).

Предоставляет класс DatasetPreprocessor, который можно использовать:
  1. Программно:
      preproc = DatasetPreprocessor()
      preproc.run('syntagrus', pretrained=True)

  2. Из CLI:
      python -m morph_tagger.preprocessing.vectorize --dataset syntagrus --pretrained

Целевые грамматические атрибуты, которые обрабатываются:
    upos, head, deprel, Mood, NumType, VerbForm, ExtPos, Reflex, Polarity,
    Typo, NameType, InflClass, Person, Poss, Animacy, Degree, Foreign,
    Variant, Number, Gender, NumForm, Aspect, Case, PronType, Tense, Abbr, Voice
"""

from morph_tagger.core.tokenizer import BPETokenizer
from morph_tagger.core.vectorizer import Vectorizer
from morph_tagger.core.vocabulary import Vocabulary
import pandas as pd
import numpy as np
import json
import os
import sys
import logging
import argparse
from pathlib import Path
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)


def _find_max_words_source_len(df: pd.DataFrame) -> int:
    """Максимальная длина НЕ ТОКЕНИЗИРОВАННОЙ последовательности."""
    return int(df['source_words'].str.len().max())


def _find_max_tokens_source_len(df: pd.DataFrame, tokenizer) -> int:
    """Максимальная длина ТОКЕНИЗИРОВАННОЙ последовательности."""
    return max(len(tokenizer.encode(df.loc[i, 'source_text']).tokens)
               for i in range(len(df)))


def _find_max_subtokens_cnt(df: pd.DataFrame) -> int:
    """Максимальное количество субтокенов в одном слове."""
    return max(
        max(len(tokens) for tokens in row)
        for row in df['tokens']
    )


def _find_max_letters_cnt(df: pd.DataFrame) -> int:
    """Максимальное количество букв в одном слове."""
    return max(len(word)
               for i in range(len(df))
               for word in df.loc[i, 'source_words'])


def _tokenize_dataframe(df: pd.DataFrame,
                        tokenizer,
                        return_length: bool = False) -> pd.DataFrame:
    """Токенизирует все слова в датафрейме (in-place)."""
    df['tokens'] = [[] for _ in range(len(df))]
    if return_length:
        df['length'] = 0
    for row in range(len(df)):
        for word in df.loc[row, 'source_words']:
            tokens = tokenizer.encode(str(word)).tokens
            df.loc[row, 'tokens'].append(tokens)
            if return_length:
                df.loc[row, 'length'] += len(tokens)
    return df


class DatasetPreprocessor:
    """Подготавливает .parquet датасеты для обучения модели.

    Выполняет:
    1. Загрузку сырых .parquet из .conllu
    2. BPE-токенизацию (обучение нового или загрузка предобученного токенизатора)
    3. Заполнение словарей (source, target, letters)
    4. Векторизацию токенов и таргетов
    5. Сохранение конфигурации и подготовленных датасетов

    Args:
        experiment_name: Имя эксперимента (из .env EXPERIMENT_NAME).
        data_save_filepath: Корневая директория для сохранения (из .env).
        syntagrus_version: Версия Syntagrus — '2.3' или '2.16'.
        vocab_size: Размер словаря BPE. default = 10000.
        max_letters_count: Максимальное кол-во букв в слове. default = 12.

    Примеры:
>>> preproc = DatasetPreprocessor()
>>> preproc.run('syntagrus', pretrained=True)

>>> preproc = DatasetPreprocessor(mfp=500)
>>> preproc.run('taiga')  # обучит новый токенизатор
    """

    WORD_REPRESENTATION = 'tokens'
    BOS_TOKEN = '<BOS>'
    EOS_TOKEN = '<EOS>'
    PAD_TOKEN = '<PAD>'
    MASK_TOKEN = '<MASK>'
    UNK_TOKEN = '<UNK>'
    ADD_BOS_EOS_TOKENS = False

    TARGET_NAMES = [
        'upos', 'head', 'deprel', 'Mood', 'NumType', 'VerbForm',
        'ExtPos', 'Reflex', 'Polarity', 'Typo', 'NameType', 'InflClass',
        'Person', 'Poss', 'Animacy', 'Degree', 'Foreign', 'Variant', 'Number',
        'Gender', 'NumForm', 'Aspect', 'Case', 'PronType', 'Tense', 'Abbr',
        'Voice']

    SUPPORTED_DATASETS = ('taiga', 'syntagrus', 'merged')


    def __init__(self,
                 experiment_name: str = None,
                 data_save_filepath: str = None,
                 syntagrus_version: str = None,
                 vocab_size: int = 5000,
                 max_letters_count: int = 6,
                 mfp: int = 800):
        load_dotenv(dotenv_path=(Path('.') / '.env'))

        self._experiment_name = experiment_name or os.getenv('EXPERIMENT_NAME')
        self._data_save_filepath = data_save_filepath or os.getenv('DATA_SAVE_FILEPATH')

        self._syntagrus_version = syntagrus_version or os.getenv('SYNTAGRUS_VERSION', '2.16')

        # Пути
        self._datasets_folder_path = os.getenv('DATASETS_FOLDER_PATH')
        self._syntagrus_path = os.getenv('SYNTAGRUS_PATH')
        self._syntagrus_texts_path = os.getenv('SYNTAGRUS_TEXTS_PATH')
        self._taiga_path = os.getenv('TAIGA_PATH')
        self._taiga_texts_path = os.getenv('TAIGA_TEXTS_PATH')

        self._merged_path = os.path.join(self._datasets_folder_path, 'sintagrus_taiga_merged')
        Path.mkdir(Path(self._merged_path), exist_ok=True)
        self._merged_texts_path = os.path.join(
            self._merged_path, 'sintagrus_taiga_merged.txt')

        self._checkpoints_path = os.path.join(self._data_save_filepath, self._experiment_name, 'checkpoints')
        self._data_info_path = os.path.join(self._data_save_filepath, self._experiment_name, 'data')
        self._dataset_save_path = os.path.join(self._data_save_filepath, self._experiment_name, 'dataset')

        # Параметры токенизатора
        self._vocab_size = vocab_size
        self._mfp = mfp # min frequency pair
        self._max_letters_count = max_letters_count


    def _resolve_paths(self, dataset: str) -> tuple[str, str]:
        if dataset == 'syntagrus':
            return self._syntagrus_path, self._syntagrus_texts_path
        elif dataset == 'taiga':
            return self._taiga_path, self._taiga_texts_path
        elif dataset == 'merged':
            return self._merged_path, self._merged_texts_path
        else:
            raise ValueError(f"Неверный датасет: {dataset}. Допустимые: {self.SUPPORTED_DATASETS}")


    def _load_parquet(self, dataset: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Загружает train/dev/test .parquet для заданного датасета."""
        if dataset == 'merged':
            train_df = pd.concat([pd.read_parquet(os.path.join(self._taiga_path, 'taiga_train.parquet')),
                                  pd.read_parquet(os.path.join(self._syntagrus_path, 'syntagrus_train.parquet'))]).reset_index(drop=True)
            dev_df = pd.concat([pd.read_parquet(os.path.join(self._taiga_path, 'taiga_dev.parquet')),
                                pd.read_parquet(os.path.join(self._syntagrus_path, 'syntagrus_dev.parquet'))]).reset_index(drop=True)
            test_df = pd.concat([pd.read_parquet(os.path.join(self._taiga_path, 'taiga_test.parquet')),
                                 pd.read_parquet(os.path.join(self._syntagrus_path, 'syntagrus_test.parquet'))]).reset_index(drop=True)
        else:
            ds_path, _ = self._resolve_paths(dataset)
            train_df = pd.read_parquet(os.path.join(ds_path, f'{dataset}_train.parquet'))
            dev_df = pd.read_parquet(os.path.join(ds_path, f'{dataset}_dev.parquet'))
            test_df = pd.read_parquet(os.path.join(ds_path, f'{dataset}_test.parquet'))
        return train_df, dev_df, test_df


    def _build_vocabularies(self,
                            train_df: pd.DataFrame,
                            val_df: pd.DataFrame,
                            test_df: pd.DataFrame,
                            use_pretrained: bool) -> tuple[Vocabulary, dict, Vocabulary]:
        """Создает или загружает словари (source, target, letters).

        Returns:
            (source_vocab, target_vocabs_dict, letters_vocab)
        """
        if use_pretrained:
            logging.info('Инициализация словарей из json')
            source_vocab = Vocabulary.from_json(f'{self._data_info_path}/merged_source_vocab.json')
            with open(f'{self._data_info_path}/merged_target_vocabs.json', 'r', encoding='utf-8') as f:
                target_vocabs_dict = json.load(f)
            target_vocabs = {name: Vocabulary.from_serializable(target_vocabs_dict[name]) for name in self.TARGET_NAMES}
            letters_vocab = Vocabulary.from_json(f'{self._data_info_path}/merged_letters_vocab.json')
        else:
            logging.info('Заполнение словарей с нуля')

            source_vocab = Vocabulary(bos_token=self.BOS_TOKEN,
                                      eos_token=self.EOS_TOKEN,
                                      pad_token=self.PAD_TOKEN,
                                      mask_token=self.MASK_TOKEN,
                                      unk_token=self.UNK_TOKEN,
                                      add_bos_eos_tokens=self.ADD_BOS_EOS_TOKENS)

            target_vocabs = {name: Vocabulary(
                bos_token=self.BOS_TOKEN, eos_token=self.EOS_TOKEN,
                pad_token=self.PAD_TOKEN, mask_token=self.MASK_TOKEN,
                unk_token=self.UNK_TOKEN,
                add_bos_eos_tokens=self.ADD_BOS_EOS_TOKENS) for name in self.TARGET_NAMES}

            letters_vocab = Vocabulary(
                bos_token=self.BOS_TOKEN, eos_token=self.EOS_TOKEN,
                pad_token=self.PAD_TOKEN, mask_token=self.MASK_TOKEN,
                unk_token=self.UNK_TOKEN, add_bos_eos_tokens=False)

            # Заполнение
            for df in [train_df, val_df, test_df]:
                for row in range(len(df)):
                    for token_lst in df.loc[row, 'tokens']:
                        source_vocab.add_tokens(token_lst)
                    for name in self.TARGET_NAMES:
                        target_vocabs[name].add_tokens(df.loc[row, name])
                for i in range(len(df)):
                    for word in df.loc[i, 'source_words']:
                        letters_vocab.add_tokens(list(word))

            # Сохранение
            source_vocab.to_json(f'{self._data_info_path}/{self._dataset}_source_vocab.json')
            target_vocabs_dict = {name: target_vocabs[name].to_serializable() for name in self.TARGET_NAMES}
            with open(f'{self._data_info_path}/{self._dataset}_target_vocabs.json', 'w', encoding='utf-8') as f:
                json.dump(target_vocabs_dict, f)
            letters_vocab.to_json(f'{self._data_info_path}/{self._dataset}_letters_vocab.json')

        return source_vocab, target_vocabs, letters_vocab


    def _vectorize_dataframe(self,
                             df: pd.DataFrame,
                             vectorizer: Vectorizer,
                             max_subtokens: int,
                             max_words: int) -> pd.DataFrame:
        """Добавляет столбцы input_ids и *_ids в датафрейм."""
        df['input_ids'] = None
        for name in self.TARGET_NAMES:
            df[f'{name}_ids'] = None
        for i in range(len(df)):
            df.at[i, 'input_ids'] = vectorizer.vectorize_tokens(
                max_subtokens, max_words, df.loc[i, 'tokens'])
            trg_vec = vectorizer.vectorize_targets(
                df.loc[i], self.TARGET_NAMES, max_words, self.ADD_BOS_EOS_TOKENS)
            for name in self.TARGET_NAMES:
                df.at[i, f'{name}_ids'] = trg_vec[name]
        return df


    def run(self,
            dataset: str,
            pretrained: bool = False,
            exclude_unused_grammemes: bool = False,
            quantile: float = 0.98) -> dict:
        """Выполняет полную подготовку датасета.

        Args:
            dataset: 'taiga', 'syntagrus' или 'merged'.
            pretrained: Использовать предобученный токенизатор и словари.
            exclude_unused_grammemes: Заменять None-граммемы на padding.
            quantile: Квантиль для определения MAX_SUBTOKENS_COUNT.

        Returns:
            Словарь с метаданными подготовки:
                {'max_words_count': int, 'max_subtokens_count': int,
                 'max_letters_count': int, 'source_vocab_len': int,
                 'target_vocab_lens': dict, 'letters_vocab_len': int,
                 'pad_idx': int}
        """
        self._dataset = dataset
        logging.info(f"Подготовка датасета: {dataset}, pretrained={pretrained}, mfp={self._mfp}")

        # Пути и директории
        dataset_path, texts_path = self._resolve_paths(dataset)
        Path.mkdir(Path(self._data_save_filepath, self._experiment_name), exist_ok=True)
        Path.mkdir(Path(self._checkpoints_path), exist_ok=True)
        Path.mkdir(Path(self._data_info_path), exist_ok=True)
        Path.mkdir(Path(self._dataset_save_path), exist_ok=True)
        logging.info('Пути для сохранения файлов созданы')

        # Загрузка паркета
        logging.info('Считывание датасета...')
        train_df, val_df, test_df = self._load_parquet(dataset)
        for df in [train_df, test_df, val_df]:
            df = df.reset_index().drop(columns=['index'])

        # Токенизатор
        if pretrained:
            logging.info('Инициализация предобученного токенизатора')
            tokenizer = BPETokenizer.from_pretrained(f'{self._checkpoints_path}/tokenizer.json')
        else:
            logging.info('Обучение нового токенизатора')
            tokenizer = BPETokenizer.train([texts_path], self._vocab_size, self._mfp, unk_token=self.UNK_TOKEN, pad_token=self.PAD_TOKEN)
            tokenizer.save(f'{self._checkpoints_path}/tokenizer.json')

        # Токенизация
        logging.info('Токенизация датасета...')
        train_df = _tokenize_dataframe(train_df, tokenizer)
        test_df = _tokenize_dataframe(test_df, tokenizer)
        val_df = _tokenize_dataframe(val_df, tokenizer)

        # Статистика и определение размеров
        for df_name, df in [('train', train_df), ('test', test_df), ('validation', val_df)]:
            subtokens_cnt = [len(t) for row in df['tokens'] for t in row]
            np_arr = np.array(subtokens_cnt)
            print(f"Характеристики {df_name}:\nмедиана={np.median(np_arr):.1f},\nсреднее={np.mean(np_arr):.1f},\nмакс={np.max(np_arr)}")

        max_words_count = max(_find_max_words_source_len(test_df), _find_max_words_source_len(val_df))
        max_subtokens_count = int(np.quantile([len(t) for row in train_df['tokens'] for t in row], quantile))
        max_letters_count = max(_find_max_letters_cnt(test_df), _find_max_letters_cnt(val_df))

        logging.info(f"MAX_WORDS={max_words_count}, MAX_SUBTOKENS={max_subtokens_count}, MAX_LETTERS={max_letters_count}")

        # Фильтрация обучающей выборки
        train_df = train_df.loc[
            train_df['source_words'].apply(len) <= max_words_count]
        train_df = train_df.reset_index()

        # Словари
        source_vocab, target_vocabs, letters_vocab = self._build_vocabularies(train_df, val_df, test_df, pretrained)

        if exclude_unused_grammemes:
            for name in self.TARGET_NAMES:
                target_vocabs[name].token_to_idx['None'] = target_vocabs[name].pad_idx

        pad_idx = source_vocab.pad_idx
        trg_vocabs_len = {k: len(v) for k, v in target_vocabs.items()}

        print(f'Словарь токенов: {len(source_vocab)}')
        for key in self.TARGET_NAMES:
            print(f'  Словарь {key}: {len(target_vocabs[key])}')

        # Векторизация
        vectorizer = Vectorizer(source_vocab, target_vocabs, letters_vocab, self.WORD_REPRESENTATION, pad_idx)

        logging.info('Векторизация...')
        for df in [train_df, val_df, test_df]:
            self._vectorize_dataframe(df, vectorizer, max_subtokens_count, max_words_count)

        # Удаление избыточных столбцов
        drop_cols = [*[n for n in self.TARGET_NAMES], 'index', 'lemmas', 'xpos', 'feats', 'misc', 'source_text']
        for df in [train_df, val_df, test_df]:
            cols_to_drop = [c for c in drop_cols if c in df.columns]
            df = df.drop(columns=cols_to_drop)

        # Сохранение конфигурации
        config = {
            'MIN_FREQUENCY_PAIR': self._mfp,
            'MAX_WORDS_COUNT': max_words_count,
            'MAX_SUBTOKENS_COUNT': max_subtokens_count,
            'MAX_LETTERS_COUNT': max_letters_count,
            'SOURCE_VOCAB_LEN': len(source_vocab),
            'LETTERS_VOCAB_LEN': len(letters_vocab),
            'TRG_VOCABS_LEN': trg_vocabs_len,
            'PAD_IDX': pad_idx}
        
        with open(f'{self._data_info_path}/{dataset}_vocabs_configuration.json', 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4, ensure_ascii=False)

        # Сохранение датасетов
        logging.info('Сохранение датасетов...')
        for split_name, split_df in [('train', train_df), ('test', test_df), ('validation', val_df)]:
            output_name = (f'{dataset}_prepared_dev' if split_name == 'validation' else f'{dataset}_prepared_{split_name}')
            split_df.to_parquet(os.path.join(self._dataset_save_path, f'{output_name}.parquet'), engine='fastparquet', index=False)

        logging.info('Готово!')
        return config


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Подготовка датасетов Syntagrus, Taiga или их слияния')
    parser.add_argument(
        '--dataset',
        type=str,
        default='merged',
        choices=DatasetPreprocessor.SUPPORTED_DATASETS,
        help='Какой датасет подготовить',
        required=True,
    )
    parser.add_argument(
        '--pretrained',
        action='store_true',
        help='Использовать предобученный токенизатор и словари.',
    )
    parser.add_argument(
        '--mfp',
        type=int,
        default=800,
        help='Мин. частота для слияния символов в токен. default = 1000',
    )
    parser.add_argument(
        '--exclude-unused-grammemes',
        action='store_true',
        help='Исключить не принадлежащие слову граммемы.',
    )
    parser.add_argument(
        '--quantile',
        type=float,
        default=0.98,
        help='Квантиль для MAX_SUBTOKENS_COUNT. default = 0.98',
    )
    return parser


def main():
    parser = _build_parser()
    args = parser.parse_args()

    preproc = DatasetPreprocessor(mfp=args.mfp)
    preproc.run(
        dataset=args.dataset,
        pretrained=args.pretrained,
        exclude_unused_grammemes=args.exclude_unused_grammemes,
        quantile=args.quantile,
    )


if __name__ == '__main__':
    main()