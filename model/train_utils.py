import json
import operator
import os
import random

import numpy as np
import pandas as pd
import torch

from model.metrics import CustomMetrics
from utils.wandb_utils import wandb_store_file


def make_new_seed(new_seed_value):
    # Ensure deterministic behavior
    print('Will be used new seed postfix:', "'" + new_seed_value + "'")

    # torch.backends.cudnn.deterministic = True
    random.seed(hash("setting random seeds " + new_seed_value) % 2 ** 32 - 1)
    np.random.seed(hash("improves reproducibility " + new_seed_value) % 2 ** 32 - 1)
    torch.manual_seed(hash("by removing stochasticity " + new_seed_value) % 2 ** 32 - 1)
    torch.cuda.manual_seed_all(hash("so runs are repeatable " + new_seed_value) % 2 ** 32 - 1)


def print_info(progress_bar, progress_name, epoch_metrics, loss, lr, metric_prefix, batch_size, epoch, epochs_count,
               seen_samples, total_samples, fold_index, folds_count):
    epoch_eta = "inf"
    if progress_bar.avg_time:
        epoch_eta = int(progress_bar.avg_time * (total_samples - seen_samples))
        epoch_eta = '{:02}:{:02}:{:02}'.format(epoch_eta // 3600, epoch_eta % 3600 // 60, epoch_eta % 60)

    postfix = {}
    if folds_count > 1:
        postfix['F'] = f'{fold_index + 1}/{folds_count}'
    postfix.update({progress_name: '{}/{}'.format(epoch, epochs_count),
                    'S': '{:04d}/{:04d}'.format(seen_samples, total_samples),
                    'E': epoch_eta, 'bL': torch.mean(loss).item()})

    for key, value in epoch_metrics.items():
        if metric_prefix and key.startswith(metric_prefix):
            key = key[len(metric_prefix):]
        if key == 'loss':
            key = 'L'
        postfix[key] = np.mean(value)

    if lr > 0:
        postfix['lr'] = f'{lr:0.3e}'

    progress_bar.update(batch_size)
    progress_bar.set_postfix(postfix, refresh=True)


def collect_metrics(cfg, epoch_metrics, loss, metrics, criterion_list, metric_prefix):
    loss = torch.mean(loss)
    epoch_metrics[metric_prefix + 'loss'].append(loss.item())
    for i, metric_name in enumerate(cfg['metrics']):
        if isinstance(criterion_list[i + 1], CustomMetrics):
            epoch_metrics[metric_prefix + metric_name] = []

        metric_item = torch.mean(metrics[i]).item() if isinstance(metrics[i], torch.Tensor) else metrics[i]
        epoch_metrics[metric_prefix + metric_name].append(metric_item)


def print_info_coteaching(progress_bar, progress_name, epoch_metrics, loss_1, loss_2, lr_1, lr_2, remember_rate_value,
                          metric_prefix, batch_size, epoch, epochs_count, seen_samples, total_samples, fold_index,
                          folds_count):
    epoch_eta = "inf"
    if progress_bar.avg_time:
        epoch_eta = int(progress_bar.avg_time * (total_samples - seen_samples))
        epoch_eta = '{:02}:{:02}:{:02}'.format(epoch_eta // 3600, epoch_eta % 3600 // 60, epoch_eta % 60)

    postfix = {}
    if folds_count > 1:
        postfix['F'] = f'{fold_index + 1}/{folds_count}'
    postfix.update({progress_name: '{}/{}'.format(epoch, epochs_count),
                    'S': '{:04d}/{:04d}'.format(seen_samples, total_samples),
                    'E': epoch_eta})

    for key, value in epoch_metrics.items():
        if metric_prefix and key.startswith(metric_prefix):
            key = key[len(metric_prefix):]
        if key == 'loss':
            key = 'L'
        if key == 'loss_2':
            key = 'L2'
        if key.endswith('_2'):
            key = key[:-2] + key[-1:]
        if key.startswith('Lwlrap'):
            key = key.replace('Lwlrap', 'LW')
        postfix[key] = np.mean(value)

    if lr_1 > 0:
        postfix['lr'] = f'{lr_1:0.2e}'
    if lr_2 > 0:
        postfix['lr2'] = f'{lr_2:0.2e}'
    if remember_rate_value > 0:
        postfix['R'] = f'{remember_rate_value:0.3f}'

    progress_bar.update(batch_size)
    progress_bar.set_postfix(postfix, refresh=True)


def collect_metrics_coteaching(cfg, epoch_metrics, loss_1, loss_2, metrics_1, metrics_2, criterion_list_1,
                               criterion_list_2, metric_prefix):
    epoch_metrics[metric_prefix + 'loss'].append(loss_1.item())
    epoch_metrics[metric_prefix + 'loss_2'].append(loss_2.item())
    for i, metric_name in enumerate(cfg['metrics']):
        if isinstance(criterion_list_1[i + 1], CustomMetrics):
            epoch_metrics[metric_prefix + metric_name] = []
        if isinstance(criterion_list_2[i + 1], CustomMetrics):
            epoch_metrics[metric_prefix + metric_name + '_2'] = []

        metric_item_1 = metrics_1[i].item() if isinstance(metrics_1[i], torch.Tensor) else metrics_1[i]
        metric_item_2 = metrics_2[i].item() if isinstance(metrics_2[i], torch.Tensor) else metrics_2[i]
        epoch_metrics[metric_prefix + metric_name].append(metric_item_1)
        epoch_metrics[metric_prefix + metric_name + '_2'].append(metric_item_2)


def print_fold_train_results_and_save(results: dict, run_folder):
    for key, value in results.items():
        if key == 'fold':
            value.append('MEAN')
        else:
            value.append(np.mean(value))

    df = pd.DataFrame(results, )
    print(df)

    if run_folder is not None:
        df.to_csv(run_folder / 'folds_result.csv', index=False)

    # remove mean values
    for value in results.values():  # type:list
        value.pop()

    return df


def store_samples_dataset_info(run_folder, classes_labels, train_split, val_split, file_suffix, wandb_run):
    if classes_labels is not None:
        labels_path = run_folder / f'labels.json'
        with open(labels_path, 'w', encoding='utf-8') as f:
            json.dump(classes_labels, f, ensure_ascii=False, indent=2)
        if wandb_run is not None:
            wandb_store_file(wandb_run, labels_path)

    if train_split is not None:
        train_split_path = run_folder / f'train_split{file_suffix}.csv'
        train_split.to_csv(train_split_path)
        if wandb_run is not None:
            wandb_store_file(wandb_run, train_split_path)

    if val_split is not None:
        val_split_path = run_folder / f'val_split{file_suffix}.csv'
        val_split.to_csv(val_split_path)
        if wandb_run is not None:
            wandb_store_file(wandb_run, val_split_path)


def get_best_metrics(cfg, best_metrics, metrics_train, epoch):
    res = {}

    epsilon = cfg['watch_metrics_eps']
    operations = {'min': operator.lt, 'max': operator.gt}

    for watch_item in cfg['watch_metrics']:
        watch_better, watch_name = watch_item.split('|')

        last_value = best_metrics.get(watch_name, None)
        new_value = metrics_train[watch_name][epoch - 1]

        if last_value is None:
            is_best_value = True
        else:
            diff = abs(last_value - new_value)
            is_best_value = diff > epsilon and operations[watch_better](new_value, last_value)

        if is_best_value:
            best_metrics[watch_name] = new_value

            if watch_name in cfg['save_metrics']:
                res['best_epoch'] = epoch
            res['best_' + watch_name] = new_value

    return res, 'best_epoch' in res


def delete_old_saved_models(save_last_n_models, saved_models):
    if save_last_n_models > 0:
        while len(saved_models) > save_last_n_models:
            # delete_model, delete_optimizer = saved_models.pop(0)
            delete_model = saved_models.pop(0)
            os.remove(delete_model)
            # os.remove(delete_optimizer)
