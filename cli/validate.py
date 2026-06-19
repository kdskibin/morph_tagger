"""Скрипт для тестирования метрик модели на различных датасетах (syntagrus, taiga, merged).

Пример запуска:
    python -m cli.validate --dataset syntagrus --split test --batch 64 --device cuda

Аргументы:
    --dataset: ['taiga', 'syntagrus', 'merged'] - выбор датасета.
    --split: ['validation', 'test'] - выбор сплита.
    --batch - размер батча, default = 64.
    --device - устройство, default = cuda.
"""

import torch
import json
import time
import os
import sys
import argparse
import logging
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

from morph_tagger.core.dataset import CustomDataset
from cli.utils import generate_batches, compute_loss, print_metrics

load_dotenv(dotenv_path=(Path('.') / '.env'))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)

DATA_SAVE_FILEPATH = os.getenv('DATA_SAVE_FILEPATH')
EXPERIMENT_NAME = os.getenv('EXPERIMENT_NAME')

CHECKPOINTS_FILEPATH = os.path.join(DATA_SAVE_FILEPATH, EXPERIMENT_NAME, 'checkpoints')
DATA_INFO_FILEPATH = os.path.join(DATA_SAVE_FILEPATH, EXPERIMENT_NAME, 'data')
DATASETS_FOLDER_PATH = os.path.join(DATA_SAVE_FILEPATH, EXPERIMENT_NAME, 'dataset')

parser = argparse.ArgumentParser(description='Тестирование модели на различных датасетах')
parser.add_argument(
    '--dataset',
    type=str,
    default='merged',
    choices=['taiga', 'syntagrus', 'merged'],
    help='Какой датасет использовать: taiga, syntagrus или merged',
    required=True,
)
parser.add_argument(
    '--split',
    type=str,
    default='validation',
    choices=['validation', 'test'],
    help='Выбор сплита датасета для теста',
    required=True,
)
parser.add_argument(
    '--batch',
    type=int,
    default=64,
    help='Размер батча при тестировании.'
)
parser.add_argument(
    '--device',
    choices=['cpu', 'cuda'],
    default='cuda',
    help='Устройство для инференса модели.'
)

args = parser.parse_args()

DATASET_TO_PREPARE = args.dataset
BATCH_SIZE = args.batch
SPLIT = args.split
DEVICE = args.device
DEVICE = DEVICE if torch.cuda.is_available() else 'cpu'

logging.info(f'Текущие параметры тестирования: DATASET_TO_PREPARE={DATASET_TO_PREPARE}, '
             f'BATCH_SIZE={BATCH_SIZE}, SPLIT={SPLIT}, DEVICE={DEVICE}')

SHUFFLE = True
DROP_LAST = False
WORD_REPRESENTATION = 'tokens'
MODEL_SAVE_FILEPATH = f'{CHECKPOINTS_FILEPATH}/final_{WORD_REPRESENTATION}_model_params.pt'
RANDOM_STATE = 42



def compute_metrics(predictions, targets, target_names, pad_idx=0, average='macro'):
    """Вычисляет метрики: accuracy, precision, recall, f1 для каждого признака."""
    metrics_dict = {}
    first_key = target_names[0]
    batch_size, seq_len = targets[first_key].size()
    device = predictions[first_key].device

    correct_words_all = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)
    any_non_pad = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
    sentence_accuracy_global = 1.0

    for key in target_names:
        _, pred_indices = predictions[key].max(dim=-1)
        mask = targets[key] != pad_idx
        correct = (pred_indices == targets[key])
        correct_or_pad = correct | ~mask
        correct_words_all = correct_words_all & correct_or_pad
        any_non_pad = any_non_pad | mask

        errors_per_sentence = ((pred_indices != targets[key]) & mask).sum(dim=1)
        sentence_accuracy = (errors_per_sentence == 0).float().mean().item()

        pred_filtered = pred_indices[mask].cpu().numpy()
        target_filtered = targets[key][mask].cpu().numpy()

        if len(target_filtered) == 0:
            metrics_dict[key] = {'accuracy': 1.0, 'sentence_accuracy': 1.0,
                                 'precision': 1.0, 'recall': 1.0, 'f1': 1.0}
            continue

        precision, recall, f1, _ = precision_recall_fscore_support(
            target_filtered, pred_filtered, average=average, zero_division=0
        )
        accuracy = (pred_filtered == target_filtered).mean()
        if accuracy != 1.0:
            sentence_accuracy_global = 0.0

        metrics_dict[key] = {
            'accuracy': accuracy, 'sentence_accuracy': sentence_accuracy,
            'precision': float(precision), 'recall': float(recall), 'f1': float(f1),
        }

    total_non_pad = any_non_pad.sum().item()
    if total_non_pad == 0:
        word_accuracy = 1.0
    else:
        word_accuracy = (correct_words_all & any_non_pad).sum().item() / total_non_pad

    metrics_dict['word_accuracy'] = word_accuracy
    metrics_dict['sentence_accuracy_global'] = sentence_accuracy_global
    return metrics_dict


# Загрузка датасета
logging.info('Загрузка датасетов...')
if SPLIT == 'validation':
    dataframe = pd.read_parquet(os.path.join(DATASETS_FOLDER_PATH, f'{DATASET_TO_PREPARE}_prepared_dev.parquet'))
else:
    dataframe = pd.read_parquet(os.path.join(DATASETS_FOLDER_PATH, f'{DATASET_TO_PREPARE}_prepared_test.parquet'))

logging.info('Чтение конфигурации словаря...')
with open(f'{DATA_INFO_FILEPATH}/merged_vocabs_configuration.json', 'r', encoding='utf-8') as file:
    vocabs_config = json.load(file)

target_names = ['upos', 'head', 'deprel', 'Mood', 'NumType', 'VerbForm',
                'ExtPos', 'Reflex', 'Polarity', 'Typo', 'NameType', 'InflClass',
                'Person', 'Poss', 'Animacy', 'Degree', 'Foreign', 'Variant', 'Number',
                'Gender', 'NumForm', 'Aspect', 'Case', 'PronType', 'Tense', 'Abbr', 'Voice']

model = torch.load(MODEL_SAVE_FILEPATH, weights_only=False, map_location=torch.device(DEVICE))

logging.info('Инициализация датасета...')
dataset = CustomDataset(None, target_names,
                        vocabs_config['MAX_SUBTOKENS_COUNT'], vocabs_config['MAX_WORDS_COUNT'],
                        vocabs_config['MAX_LETTERS_COUNT'], valid_df=dataframe, test_df=dataframe)

model = model.to(device=DEVICE)

logging.info('Переход к валидации...')
dataset.set_dataframe_split(SPLIT)
batch_generator = generate_batches(dataset, BATCH_SIZE, SHUFFLE, DROP_LAST, DEVICE)

epoch_sum_valid_loss = 0.0
epoch_running_valid_loss = 0.0
mean_generation_time = 0.0
valid_epoch_metrics = {key: {'accuracy': 0.0, 'sentence_accuracy': 0.0, 'precision': 0.0,
                             'recall': 0.0, 'f1': 0.0, 'mean_loss': 0.0} for key in target_names}
valid_epoch_metrics['word_accuracy'] = 0.0
valid_epoch_metrics['sentence_accuracy_global'] = 0.0
validation_states = []

model.eval()
valid_start_time = time.time()

with torch.no_grad():
    for batch_idx, batch_dict in enumerate(batch_generator):
        start_generation_time = time.time()

        if WORD_REPRESENTATION == 'tokens':
            predictions = model(tokens=batch_dict['input_ids'], letters=None)
        elif WORD_REPRESENTATION == 'letters':
            predictions = model(tokens=None, letters=batch_dict['letters'])
        else:
            predictions = model(tokens=batch_dict['input_ids'], letters=batch_dict['letters'])

        end_generation_time = time.time()
        cur_metrics = compute_metrics(predictions, batch_dict, target_names, vocabs_config['PAD_IDX'])

        mean_generation_time += ((end_generation_time - start_generation_time) - mean_generation_time) / (batch_idx + 1)

        for key in target_names:
            for metric, value in cur_metrics[key].items():
                valid_epoch_metrics[key][metric] += (value - valid_epoch_metrics[key][metric]) / (batch_idx + 1)

        valid_epoch_metrics['word_accuracy'] += (cur_metrics['word_accuracy'] - valid_epoch_metrics['word_accuracy']) / (batch_idx + 1)
        valid_epoch_metrics['sentence_accuracy_global'] += (cur_metrics['sentence_accuracy_global'] - valid_epoch_metrics['sentence_accuracy_global']) / (batch_idx + 1)

valid_end_time = time.time()

validation_states.append(valid_epoch_metrics)
validation_states[-1]['summed loss'] = epoch_sum_valid_loss
validation_states[-1]['execution_time'] = valid_end_time - valid_start_time

print_metrics(valid_epoch_metrics, target_names, 'Validation',
              mean_loss=epoch_running_valid_loss,
              extra=[
                  f'Среднее время генерации при размере батча {BATCH_SIZE}: {mean_generation_time}',
                  f'Время выполнения всего цикла тестирования: {valid_end_time - valid_start_time}',
              ],
              epoch_suffix=False)

# Сохранение метрик
with open(os.path.join(DATA_INFO_FILEPATH, "dataset_validation_states.json"),
          "w", encoding="utf-8") as file:
    json.dump(validation_states, file, indent=4, ensure_ascii=False)
    logging.info('Метрики валидации сохранены')
logging.info('Готово!')