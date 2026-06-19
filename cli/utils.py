"""Общие утилиты для скриптов обучения и валидации."""

import torch


def generate_batches(dataset, batch_size: int, shuffle: bool = True, drop_last: bool = True, device: str = 'cpu'):
    """Создаёт батчи из датасета и переносит данные на девайс."""
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size, shuffle, drop_last=drop_last)
    for data_dict in dataloader:
        out_data_dict = {name: data_dict[name].to(device) for name in data_dict}
        yield out_data_dict


def normalize_sizes(predictions: dict, targets: dict, target_names: list):
    for key in target_names:
        if len(predictions[key].size()) == 3:
            predictions[key] = predictions[key].contiguous().view(-1, predictions[key].size(-1))
        if len(targets[key].size()) == 2:
            targets[key] = targets[key].contiguous().view(-1)
    return predictions, targets


def compute_loss(predictions: dict, targets: dict, target_names: list, pad_idx: int = 0):
    predictions, targets = normalize_sizes(predictions, targets, target_names)
    losses = {key: torch.nn.functional.cross_entropy(predictions[key], targets[key], ignore_index=pad_idx)
              for key in target_names}
    return sum(losses.values()), losses


def print_metrics(metrics: dict, target_names: list, prefix: str,
                  mean_loss: float = 0.0, extra: list = None,
                  leading_sep: bool = True, epoch_suffix: bool = True):
    """Форматированный вывод метрик.

    Аргументы:
        metrics: словарь метрик по признакам (train_epoch_metrics / valid_epoch_metrics)
        target_names: список названий признаков
        prefix: префикс строк ('Train' / 'Validation')
        mean_loss: средняя ошибка эпохи/сплита
        extra: список дополнительных строк (время, gen_time и т.д.)
        leading_sep: добавлять ли разделитель '- * 40' в начало блока
        epoch_suffix: добавлять ли слово 'эпохи' после 'Средняя ошибка'
    """
    if leading_sep:
        print('-' * 40)
    suffix = ' эпохи' if epoch_suffix else ''
    print(f'{prefix}: Средняя ошибка{suffix} {mean_loss}')
    for key in target_names:
        print('-' * 20)
        print(f'{prefix}: Ошибка на признаке {key}: {metrics[key]["mean_loss"]}')
        print(f'{prefix}: Точность на признаке {key}: {metrics[key]["accuracy"] * 100}%')
        print(f'{prefix}: precision на признаке {key}: {metrics[key]["precision"] * 100}%')
        print(f'{prefix}: recall на признаке {key}: {metrics[key]["recall"] * 100}%')
        print(f'{prefix}: f1-score на признаке {key}: {metrics[key]["f1"] * 100}%')
    print('-' * 20)
    print(f'{prefix}: Точность предсказания всех морфем в слове: {metrics["word_accuracy"] * 100}%')
    print(f'{prefix}: Точность предсказания предложения целиком: {metrics["sentence_accuracy_global"] * 100}%')
    if extra:
        for line in extra:
            print(line)