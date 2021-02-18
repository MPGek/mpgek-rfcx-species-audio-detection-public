from typing import List, Dict, Optional, Tuple

import numpy as np
import torch
import wandb
from torch import nn, optim
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau, CosineAnnealingLR, CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.datasets import AudioDataset
from model.forward_passes_coteaching import forward_pass_coteaching, forward_pass_by_crops_coteaching
from model.metrics import CustomMetrics
from model.schedulers import RectifiedWarmupScheduler
from model.train_utils import collect_metrics_coteaching, \
    print_info_coteaching, get_best_metrics, delete_old_saved_models
from model.train_validate_loops import save_model
from utils.utils import write_csv_train_metrics


def validate_epoch_coteaching(epoch, epochs_count, model_1: nn.Module, device_1: torch.device,
                              model_2: nn.Module, device_2: torch.device, remember_rate_value,
                              criterion_list_1: List[nn.Module], criterion_list_2: List[nn.Module],
                              val_dataloader: DataLoader, cfg, train_metrics: Dict, progress_bar: tqdm, limit_samples,
                              fold_index, folds_count):
    model_1.eval()
    model_2.eval()

    sampling_rate = cfg['audio_params']['sampling_rate']
    hop_length = cfg['audio_params']['hop_length']
    validate_by_crops = cfg['validate_params'].get('by_crops', 0)
    validate_crops_offset = cfg['validate_params'].get('crops_offset', validate_by_crops)
    by_labeled_crops = cfg['validate_params'].get('by_labeled_crops', False)
    use_fp = cfg['train_data_loader'].get('use_fp', False)
    aux_weights = cfg['train_params'].get('aux_weights', None)
    high_loss_train = cfg['train_params'].get('high_loss_train', 0)

    if validate_by_crops > 0:
        validate_by_crops = int(validate_by_crops * sampling_rate / hop_length)
        validate_crops_offset = int(validate_crops_offset * sampling_rate / hop_length)

    # reset CustomMetrics
    for metric in criterion_list_1:  # type:CustomMetrics
        if isinstance(metric, CustomMetrics):
            metric.reset()
    for metric in criterion_list_2:  # type:CustomMetrics
        if isinstance(metric, CustomMetrics):
            metric.reset()

    epoch_metrics = {'val_loss': [], 'val_loss_2': []}
    for metric in cfg['metrics']:
        epoch_metrics.update({f'val_{metric}': [], f'val_{metric}_2': []})

    seen_samples = 0
    for iteration_id, data in enumerate(val_dataloader):
        if limit_samples is not None and seen_samples >= limit_samples:  # DEBUG
            break

        # if progress_bar.n == 0:
        #     progress_bar.reset()  # reset start time to remove time while DataLoader was populating processes

        data_img, data_class, data_record_ids = data  # type: torch.Tensor, torch.Tensor, Tuple[str]
        apply_results_filter = None
        if by_labeled_crops:
            val_dataset = val_dataloader.dataset  # type: AudioDataset
            apply_results_filter = val_dataset.apply_results_filter

        with torch.no_grad():
            if validate_by_crops <= 0:
                res = forward_pass_coteaching(data_img, data_class, model_1, device_1, model_2, device_2,
                                              criterion_list_1, criterion_list_2, remember_rate_value, high_loss_train,
                                              use_fp)
            else:
                res = forward_pass_by_crops_coteaching(data_img, data_class, model_1, device_1, model_2, device_2,
                                                       criterion_list_1, criterion_list_2, remember_rate_value,
                                                       high_loss_train, validate_by_crops, validate_crops_offset,
                                                       apply_results_filter, data_record_ids, use_fp)
        outputs_1, outputs_2, loss_1, loss_2, metrics_1, metrics_2 = res

        # Metrics and progress
        seen_samples += len(data_img)

        collect_metrics_coteaching(cfg, epoch_metrics, loss_1, loss_2, metrics_1, metrics_2, criterion_list_1,
                                   criterion_list_2, 'val_')
        print_info_coteaching(progress_bar, 'VALID', epoch_metrics, loss_1, loss_2, 0, 0, 0, 'val_', len(data_img),
                              epoch, epochs_count, seen_samples, len(val_dataloader.dataset), fold_index, folds_count)

    for idx, item in enumerate(epoch_metrics.items()):
        key, value = item
        real_idx = idx // 2
        criterion_list = criterion_list_1 if idx % 2 == 0 else criterion_list_2
        if isinstance(criterion_list[real_idx], CustomMetrics):
            value = [criterion_list[real_idx].compute()]
        train_metrics[key].append(np.mean(value))

    progress_bar.write('')


def train_epoch_coteaching(epoch, epochs_count,
                           model_1: nn.Module, device_1: torch.device, optimizer_1: optim.Optimizer,
                           model_2: nn.Module, device_2: torch.device, optimizer_2: optim.Optimizer,
                           scaler: Optional[GradScaler], remember_rate_value, criterion_list_1: List[nn.Module],
                           criterion_list_2: List[nn.Module], train_dataloader: DataLoader, cfg, train_metrics: Dict,
                           progress_bar: tqdm, run_folder, limit_samples, fold_index, folds_count,
                           scheduler_1: _LRScheduler, scheduler_2: _LRScheduler, schedule_lr_each_step):
    epoch_metrics = {'loss': [], 'loss_2': []}
    for metric in cfg['metrics']:
        epoch_metrics.update({metric: [], f'{metric}_2': []})

    grad_acc_iters = max(cfg['train_params'].get('grad_acc_iters', 1), 1)
    grad_clipping = cfg['train_params'].get('grad_clipping', 0)
    high_loss_train = cfg['train_params'].get('high_loss_train', 0)
    use_fp = cfg['train_data_loader'].get('use_fp', False)

    batch_size = cfg['train_data_loader']['batch_size']
    epoch_samples = len(train_dataloader) * batch_size
    seen_samples = 0

    model_1.train()
    model_2.train()

    # reset CustomMetrics
    for metric in criterion_list_1:  # type:CustomMetrics
        if isinstance(metric, CustomMetrics):
            metric.reset()
    for metric in criterion_list_2:  # type:CustomMetrics
        if isinstance(metric, CustomMetrics):
            metric.reset()

    optimizer_1.zero_grad()
    optimizer_2.zero_grad()
    is_optimizer_update_finished = True

    for iteration_id, data in enumerate(train_dataloader):
        if limit_samples is not None and seen_samples >= limit_samples:  # DEBUG
            break

        data_img, data_class, data_record_ids = data
        batch_size = len(data_img)

        if progress_bar.n == 0:
            progress_bar.reset()  # reset start time to remove time while DataLoader was populating processes

        with autocast():
            res = forward_pass_coteaching(data_img, data_class, model_1, device_1, model_2, device_2, criterion_list_1,
                                          criterion_list_2, remember_rate_value, high_loss_train, use_fp)
            outputs_1, outputs_2, loss_1, loss_2, metrics_1, metrics_2 = res
            if grad_acc_iters > 1:
                loss_1 = loss_1 / grad_acc_iters
                loss_2 = loss_2 / grad_acc_iters
        scaler.scale(loss_1).backward()
        scaler.scale(loss_2).backward()

        is_optimizer_update_finished = False

        if grad_acc_iters <= 1 or (iteration_id + 1) % grad_acc_iters == 0:
            if grad_clipping > 0:
                # Unscales the gradients of optimizer's assigned params in-place
                scaler.unscale_(optimizer_1)
                scaler.unscale_(optimizer_2)
                # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
                torch.nn.utils.clip_grad_norm_(model_1.parameters(), grad_clipping)
                torch.nn.utils.clip_grad_norm_(model_2.parameters(), grad_clipping)

            scaler.step(optimizer_1)
            scaler.step(optimizer_2)
            scaler.update()

            optimizer_1.zero_grad()
            optimizer_2.zero_grad()
            is_optimizer_update_finished = True

        # Metrics and progress
        seen_samples += batch_size

        collect_metrics_coteaching(cfg, epoch_metrics, loss_1 * grad_acc_iters, loss_2 * grad_acc_iters,
                                   metrics_1, metrics_2, criterion_list_1, criterion_list_2, '')
        lr = [group['lr'] for group in optimizer_1.param_groups][0]
        lr_2 = [group['lr'] for group in optimizer_2.param_groups][0]
        print_info_coteaching(progress_bar, 'Train', epoch_metrics, loss_1 * grad_acc_iters, loss_2 * grad_acc_iters,
                              lr, lr_2, remember_rate_value, '', batch_size, epoch, epochs_count, seen_samples,
                              epoch_samples, fold_index, folds_count)

        if (iteration_id + 1) % 5 == 0 and iteration_id + 1 < len(train_dataloader):
            log_dict = dict([(key, value[-1]) for key, value in epoch_metrics.items()])
            if schedule_lr_each_step and scheduler_1 is not None:
                log_dict['lr'] = optimizer_1.param_groups[0]['lr']
            if schedule_lr_each_step and scheduler_2 is not None:
                log_dict['lr_2'] = optimizer_2.param_groups[0]['lr']
            wandb.log(log_dict, (epoch - 1) * epoch_samples + seen_samples, commit=True)

        # Step lr scheduler
        if schedule_lr_each_step and scheduler_1 is not None:
            scheduler_1.step()
        if schedule_lr_each_step and scheduler_2 is not None:
            scheduler_2.step()

    # Finish optimizer step after the not completed gradient accumulation batch
    if not is_optimizer_update_finished:
        if grad_clipping > 0:
            # Unscales the gradients of optimizer's assigned params in-place
            scaler.unscale_(optimizer_1)
            scaler.unscale_(optimizer_2)
            # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
            torch.nn.utils.clip_grad_norm_(model_1.parameters(), grad_clipping)
            torch.nn.utils.clip_grad_norm_(model_2.parameters(), grad_clipping)

        scaler.step(optimizer_1)
        scaler.step(optimizer_2)
        scaler.update()

    for idx, item in enumerate(epoch_metrics.items()):
        key, value = item
        real_idx = idx // 2
        criterion_list = criterion_list_1 if idx % 2 == 0 else criterion_list_2
        if isinstance(criterion_list[real_idx], CustomMetrics):
            value = [criterion_list[real_idx].compute()]
        train_metrics[key].append(np.mean(value))

    progress_bar.write('')


def train_loop_coteaching(cfg, model_1, device_1, optimizer_1, model_2, device_2, optimizer_2, scaler, criterion_list_1,
                          criterion_list_2, train_dataloader, val_dataloader, run_folder, csv_path, snapshot_path_1,
                          snapshot_path_2, orig_stderr, fold_index, folds_count):
    metrics_train = {'lr': [], 'lr_2': [], 'loss': [], 'loss_2': []}
    for metric in cfg['metrics']:
        metrics_train.update({metric: [], f'{metric}_2': []})
    metrics_train.update({'val_loss': [], 'val_loss_2': []})
    for metric in cfg['metrics']:
        metrics_train.update({f'val_{metric}': [], f'val_{metric}_2': []})
    best_metrics = {}

    # complete metrics with the
    watch_metrics = cfg['watch_metrics']  # type:List
    save_metrics = cfg['save_metrics']  # type:List
    for metric in list(watch_metrics):
        watch_metrics.append(metric + '_2')
    for metric in list(save_metrics):
        save_metrics.append(metric + '_2')
    cfg['watch_metrics'] = watch_metrics
    cfg['save_metrics'] = save_metrics

    epochs = cfg['train_params']['epochs']
    limit_samples = cfg['train_params'].get('limit_samples', None)
    batch_size = cfg['train_data_loader']['batch_size']
    total_train_steps = epochs * len(train_dataloader)
    total_samples = epochs * (len(train_dataloader) * batch_size + len(val_dataloader.dataset))

    remember_rate = cfg['train_params']['remember_rate']
    remember_c = cfg['train_params']['remember_c']
    remember_length = cfg['train_params']['remember_length']

    save_last_n_models = cfg['train_params'].get('save_last_n_models', 0)
    save_epochs = cfg['train_params'].get('save_epochs', [])  # type: List
    saved_models = {}

    schedule_lr_each_step = cfg['train_params'].get('schedule_lr_each_step', False)
    scheduler_1 = get_scheduler(cfg, epochs, optimizer_1, schedule_lr_each_step, total_train_steps, train_dataloader)
    scheduler_2 = get_scheduler(cfg, epochs, optimizer_2, schedule_lr_each_step, total_train_steps, train_dataloader)

    early_stop = cfg['train_params'].get('early_stop', 0)
    early_stop_counter = 0

    progress_bar = tqdm(total=total_samples, file=orig_stderr, smoothing=0.3)
    for epoch in range(1, epochs + 1):
        # R(T) = 1−min{τ ·T**c/Tk, τ, 0.95} (T=epoch-1, epoch=[1, epochs]), R(1) = 1 which possible when T is 0
        # remember_rate   - τ  = {0.5, 0.75, 1, 1.25, 1.5}
        # remember_c      - c  = {0.5, 1, 2}
        # remember_length - Tk = {5, 10, 15}
        remember_rate_value = 1 - min(remember_rate * (epoch - 1) ** remember_c / remember_length, remember_rate, 0.95)

        # validate_epoch(epoch, epochs, validate_model, device, criterion_list, val_dataloader, cfg, metrics_train,
        #                progress_bar, limit_samples, fold_index, folds_count)

        train_epoch_coteaching(epoch, epochs, model_1, device_1, optimizer_1, model_2, device_2, optimizer_2, scaler,
                               remember_rate_value, criterion_list_1, criterion_list_2, train_dataloader, cfg,
                               metrics_train, progress_bar, run_folder, limit_samples, fold_index, folds_count,
                               scheduler_1, scheduler_2, schedule_lr_each_step)

        validate_epoch_coteaching(epoch, epochs, model_1, device_1, model_2, device_2, remember_rate_value,
                                  criterion_list_1, criterion_list_2, val_dataloader, cfg, metrics_train, progress_bar,
                                  limit_samples, fold_index, folds_count)

        scheduler_1 = update_scheduler(cfg, epoch, metrics_train, schedule_lr_each_step, optimizer_1, scheduler_1)
        scheduler_2 = update_scheduler(cfg, epoch, metrics_train, schedule_lr_each_step, optimizer_2, scheduler_2, '_2')

        write_csv_train_metrics(csv_path, metrics_train)

        log_dict = dict([('epoch', epoch)] + [(key, value[epoch - 1]) for key, value in metrics_train.items()])
        log_dict['R'] = remember_rate_value

        new_best_metrics, is_best_epoch = get_best_metrics(cfg, best_metrics, metrics_train, epoch)
        if new_best_metrics:
            log_dict.update(new_best_metrics)

        if is_best_epoch:
            save_model_coteaching(model_1, model_2, snapshot_path_1, snapshot_path_2, log_dict, saved_models,
                                  save_last_n_models, new_best_metrics, cfg['save_metrics'])
        if epoch in save_epochs:
            save_name = f'e_id_{save_epochs.index(epoch)}'
            save_model(model_1, snapshot_path_1, log_dict, None, None, ['best_' + save_name], [save_name])
            save_model(model_2, snapshot_path_2, log_dict, None, None, ['best_' + save_name], [save_name])

        wandb.log(log_dict, epoch * len(train_dataloader) * batch_size)

        if not is_best_epoch:
            early_stop_counter += 1
            if 0 < early_stop < early_stop_counter:
                print("Early stopping!")
                break
        else:
            early_stop_counter = 0

    progress_bar.close()

    return best_metrics

def save_model_coteaching(model_1, model_2, snapshot_path_1, snapshot_path_2, log_dict, saved_models: Optional[Dict],
                          save_last_n_models, new_best_metrics, save_metrics):
    save_path_list = []

    # snapshot_path = snapshot_path_1
    # model = model_1

    for save_name in save_metrics:
        best_name = 'best_' + save_name
        if best_name in new_best_metrics:
            new_dict = dict(log_dict)
            new_dict['best_name'] = best_name

            if not best_name.endswith('_2'):
                save_path_list.append((model_1, best_name, str(snapshot_path_1).format(**new_dict)))
            else:
                new_dict['best_name'] = best_name[:-2]
                for key in list(new_dict.keys()):
                    if key.endswith('_2'):
                        new_dict[key[:-2]] = new_dict[key]
                save_path_list.append((model_2, best_name, str(snapshot_path_2).format(**new_dict)))

    for model, best_name, save_path in save_path_list:
        # save_path_optimizer = os.path.splitext(save_path)[0] + '_optimizer.pt'
        torch.save(model.state_dict(), save_path)
        # torch.save(optimizer.state_dict(), save_path_optimizer)

        # wandb_store_file(wandb.run, save_path, f'model_{best_name}.pt')

        if saved_models is None:
            continue
        if best_name not in saved_models:
            saved_models[best_name] = []
        saved_models_list = saved_models[best_name]
        # saved_models_list.append((save_path, save_path_optimizer))
        saved_models_list.append(save_path)
        delete_old_saved_models(save_last_n_models, saved_models_list)


def update_scheduler(cfg, epoch, metrics_train, schedule_lr_each_step, optimizer, scheduler, metric_suffix=''):
    lr_name = 'lr' + metric_suffix
    metric_name = 'val_loss' + metric_suffix
    if scheduler is not None:
        # metrics_train[lr_name].append(scheduler_1.get_last_lr()[0])
        metrics_train[lr_name].append(optimizer.param_groups[0]['lr'])
        if not schedule_lr_each_step:
            if isinstance(scheduler, CosineAnnealingLR):
                scheduler.step()
                if scheduler.last_epoch >= scheduler.T_max:
                    scheduler = CosineAnnealingLR(optimizer, scheduler.T_max, eta_min=scheduler.eta_min)
            else:
                scheduler.step(metrics_train[metric_name][epoch - 1])
    else:
        metrics_train[lr_name].append(cfg['train_params']['lr'])
    return scheduler


def get_scheduler(cfg, epochs, optimizer, schedule_lr_each_step, total_train_steps, train_dataloader):
    scheduler_epochs = epochs if not schedule_lr_each_step else total_train_steps
    scheduler = None
    if 'reduceLR' in cfg['train_params']:
        scheduler = ReduceLROnPlateau(optimizer, **cfg['train_params']['reduceLR'])
    elif 'rectLR' in cfg['train_params']:
        scheduler = RectifiedWarmupScheduler(optimizer, total_epoch=scheduler_epochs, **cfg['train_params']['rectLR'])
    elif 'cosineLR' in cfg['train_params']:
        scheduler_cycle = cfg['train_params']['cosineLR']['cycle']
        scheduler = CosineAnnealingLR(optimizer, scheduler_cycle)
    elif 'cosineLRWarm' in cfg['train_params']:
        lr_t_0 = cfg['train_params']['cosineLRWarm']['T_0']
        lr_t_0 = lr_t_0 if not schedule_lr_each_step else lr_t_0 * len(train_dataloader)
        lr_t_mult = cfg['train_params']['cosineLRWarm']['T_mult']
        scheduler = CosineAnnealingWarmRestarts(optimizer, lr_t_0, lr_t_mult)
    return scheduler
