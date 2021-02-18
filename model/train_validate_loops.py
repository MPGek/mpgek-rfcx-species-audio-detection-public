import csv
import math
from typing import List, Dict, Optional, Tuple

import numpy as np
import torch
import wandb
from torch import nn, optim
from torch.cuda.amp import GradScaler, autocast
from torch.nn import DataParallel
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau, CosineAnnealingLR, CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from tqdm import tqdm
from wandb.sdk.wandb_run import Run

from model.datasets import AudioDataset
from model.forward_passes import forward_pass, forward_pass_by_crops
from model.metrics import CustomMetrics
from model.schedulers import RectifiedWarmupScheduler
from model.train_utils import print_info, collect_metrics, get_best_metrics, delete_old_saved_models
from utils.utils import write_csv_train_metrics
from utils.wandb_utils import wandb_store_file


def validate_epoch(epoch, epochs_count, model: nn.Module, device: torch.device, criterion_list: List[nn.Module],
                   val_dataloader: DataLoader, cfg, train_metrics: Dict, progress_bar: tqdm, limit_samples,
                   fold_index, folds_count):
    model.eval()

    sampling_rate = cfg['audio_params']['sampling_rate']
    hop_length = cfg['audio_params']['hop_length']
    validate_by_crops = cfg['validate_params'].get('by_crops', 0)
    validate_crops_offset = cfg['validate_params'].get('crops_offset', validate_by_crops)
    by_labeled_crops = cfg['validate_params'].get('by_labeled_crops', False)
    use_fp = cfg['train_data_loader'].get('use_fp', False)
    aux_weights = cfg['train_params'].get('aux_weights', None)

    if validate_by_crops > 0:
        validate_by_crops = int(validate_by_crops * sampling_rate / hop_length)
        validate_crops_offset = int(validate_crops_offset * sampling_rate / hop_length)

    # reset CustomMetrics
    for metric in criterion_list:  # type:CustomMetrics
        if isinstance(metric, CustomMetrics):
            metric.reset()

    epoch_metrics = {'val_loss': []}
    epoch_metrics.update({'val_' + metric: [] for metric in cfg['metrics']})

    validate_outputs, validate_targets, validate_record_ids = [], [], []
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
                outputs, loss, metrics = forward_pass(data_img, data_class, model, device, criterion_list,
                                                      aux_weights=aux_weights)
            else:
                outputs, loss, metrics = forward_pass_by_crops(data_img, data_class, model, device, criterion_list,
                                                               validate_by_crops, validate_crops_offset,
                                                               apply_results_filter, data_record_ids, use_fp,
                                                               aux_weights=aux_weights)
        validate_outputs.append(outputs)
        validate_targets.append(data_class)
        validate_record_ids.extend(data_record_ids)

        # Metrics and progress
        seen_samples += len(data_img)

        collect_metrics(cfg, epoch_metrics, loss, metrics, criterion_list, 'val_')
        print_info(progress_bar, 'VALID', epoch_metrics, loss, 0, 'val_', len(data_img), epoch, epochs_count,
                   seen_samples, len(val_dataloader.dataset), fold_index, folds_count)

    for idx, item in enumerate(epoch_metrics.items()):
        key, value = item
        if isinstance(criterion_list[idx], CustomMetrics):
            value = [criterion_list[idx].compute()]
        train_metrics[key].append(np.mean(value))

    progress_bar.write('')

    validate_outputs = torch.vstack(validate_outputs)
    validate_targets = torch.vstack(validate_targets)

    return validate_outputs, validate_targets, validate_record_ids


def train_epoch(epoch, epochs_count, model: nn.Module, device: torch.device, optimizer: optim.Optimizer,
                scaler: Optional[GradScaler], criterion_list: List[nn.Module], train_dataloader: DataLoader, cfg,
                train_metrics: Dict, progress_bar: tqdm, run_folder, limit_samples, fold_index, folds_count,
                scheduler: _LRScheduler, schedule_lr_each_step):
    epoch_metrics = {'loss': []}
    epoch_metrics.update({metric: [] for metric in cfg['metrics']})

    grad_acc_iters = max(cfg['train_params'].get('grad_acc_iters', 1), 1)
    grad_clipping = cfg['train_params'].get('grad_clipping', 0)
    pseudo_label = cfg['train_params'].get('pseudo_label', False)
    use_fp = cfg['train_data_loader'].get('use_fp', False)
    aux_weights = cfg['train_params'].get('aux_weights', None)

    batch_size = cfg['train_data_loader']['batch_size']
    epoch_samples = len(train_dataloader) * batch_size
    seen_samples = 0

    model.train()

    # reset CustomMetrics
    for metric in criterion_list:  # type:CustomMetrics
        if isinstance(metric, CustomMetrics):
            metric.reset()

    optimizer.zero_grad()
    is_optimizer_update_finished = True

    for iteration_id, data in enumerate(train_dataloader):
        if limit_samples is not None and seen_samples >= limit_samples:  # DEBUG
            break

        data_img, data_class, data_record_ids = data
        batch_size = len(data_img)

        if progress_bar.n == 0:
            progress_bar.reset()  # reset start time to remove time while DataLoader was populating processes

        if scaler is None:
            pass
            # outputs, loss, metrics = forward_pass(data_img, data_class, model, device, criterion_list, pseudo_label,
            #                                       use_fp, aux_weights=aux_weights)
            # if grad_acc_iters > 1:
            #     loss = loss / grad_acc_iters
            # loss.backward()
        else:
            with autocast():
                outputs, loss, metrics = forward_pass(data_img, data_class, model, device, criterion_list, pseudo_label,
                                                      use_fp, aux_weights=aux_weights)
                if grad_acc_iters > 1:
                    loss = loss / grad_acc_iters
            scaler.scale(loss).backward()

        is_optimizer_update_finished = False

        if grad_acc_iters <= 1 or (iteration_id + 1) % grad_acc_iters == 0:
            if scaler is None:
                pass
                # optimizer.step()
            else:
                if grad_clipping > 0:
                    # Unscales the gradients of optimizer's assigned params in-place
                    scaler.unscale_(optimizer)
                    # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clipping)

                scaler.step(optimizer)
                scaler.update()

            optimizer.zero_grad()
            is_optimizer_update_finished = True

        # Metrics and progress
        seen_samples += batch_size

        collect_metrics(cfg, epoch_metrics, loss * grad_acc_iters, metrics, criterion_list, '')
        lr = [group['lr'] for group in optimizer.param_groups][0]
        print_info(progress_bar, 'Train', epoch_metrics, loss * grad_acc_iters, lr, '', batch_size, epoch, epochs_count,
                   seen_samples, epoch_samples, fold_index, folds_count)

        if (iteration_id + 1) % 5 == 0 and iteration_id + 1 < len(train_dataloader):
            log_dict = dict([(key, value[-1]) for key, value in epoch_metrics.items()])
            if schedule_lr_each_step and scheduler is not None:
                log_dict['lr'] = optimizer.param_groups[0]['lr']
            wandb.log(log_dict, (epoch - 1) * epoch_samples + seen_samples, commit=True)

        # Step lr scheduler
        if schedule_lr_each_step and scheduler is not None:
            scheduler.step()

    # Finish optimizer step after the not completed gradient accumulation batch
    if not is_optimizer_update_finished:
        if scaler is None:
            optimizer.step()
        else:
            if grad_clipping > 0:
                # Unscales the gradients of optimizer's assigned params in-place
                scaler.unscale_(optimizer)
                # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clipping)

            scaler.step(optimizer)
            scaler.update()

    for idx, item in enumerate(epoch_metrics.items()):
        key, value = item
        if isinstance(criterion_list[idx], CustomMetrics):
            value = [criterion_list[idx].compute()]
        train_metrics[key].append(np.mean(value))

    progress_bar.write('')


def train_loop(cfg, model, device, optimizer, scaler, criterion_list, train_dataloader, val_dataloader, run_folder,
               csv_path, snapshot_path, orig_stderr, fold_index, folds_count):
    metrics_train = {'lr': [], 'loss': []}
    metrics_train.update({metric: [] for metric in cfg['metrics']})
    metrics_train['val_loss'] = []
    metrics_train.update({'val_' + metric: [] for metric in cfg['metrics']})
    best_metrics = {}

    epochs = cfg['train_params']['epochs']
    limit_samples = cfg['train_params'].get('limit_samples', None)
    batch_size = cfg['train_data_loader']['batch_size']
    total_train_steps = epochs * len(train_dataloader)
    total_samples = epochs * (len(train_dataloader) * batch_size + len(val_dataloader.dataset))

    save_last_n_models = cfg['train_params'].get('save_last_n_models', 0)
    save_epochs = cfg['train_params'].get('save_epochs', [])  # type: List
    saved_models = {}

    scheduler = None
    schedule_lr_each_step = cfg['train_params'].get('schedule_lr_each_step', False)
    scheduler_epochs = epochs if not schedule_lr_each_step else total_train_steps

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

    early_stop = cfg['train_params'].get('early_stop', 0)
    early_stop_counter = 0

    validate_model = model
    if isinstance(validate_model, DataParallel):
        validate_model = validate_model.module

    progress_bar = tqdm(total=total_samples, file=orig_stderr, smoothing=0.3)
    for epoch in range(1, epochs + 1):
        # validate_epoch(epoch, epochs, validate_model, device, criterion_list, val_dataloader, cfg, metrics_train,
        #                progress_bar, limit_samples, fold_index, folds_count)

        train_epoch(epoch, epochs, model, device, optimizer, scaler, criterion_list, train_dataloader, cfg,
                    metrics_train, progress_bar, run_folder, limit_samples, fold_index, folds_count,
                    scheduler, schedule_lr_each_step)

        validate_epoch(epoch, epochs, validate_model, device, criterion_list, val_dataloader, cfg, metrics_train,
                       progress_bar, limit_samples, fold_index, folds_count)

        if scheduler is not None:
            # metrics_train['lr'].append(scheduler.get_last_lr()[0])
            metrics_train['lr'].append(optimizer.param_groups[0]['lr'])
            if not schedule_lr_each_step:
                if isinstance(scheduler, CosineAnnealingLR):
                    scheduler.step()
                    if scheduler.last_epoch >= scheduler.T_max:
                        scheduler = CosineAnnealingLR(optimizer, scheduler.T_max, eta_min=scheduler.eta_min)
                else:
                    scheduler.step(metrics_train['val_loss'][epoch - 1])
        else:
            metrics_train['lr'].append(cfg['train_params']['lr'])

        write_csv_train_metrics(csv_path, metrics_train)

        log_dict = dict([('epoch', epoch)] + [(key, value[epoch - 1]) for key, value in metrics_train.items()])

        new_best_metrics, is_best_epoch = get_best_metrics(cfg, best_metrics, metrics_train, epoch)
        if new_best_metrics:
            log_dict.update(new_best_metrics)

        if is_best_epoch:
            save_model(model, snapshot_path, log_dict, saved_models, save_last_n_models, new_best_metrics,
                       cfg['save_metrics'])
        if epoch in save_epochs:
            save_name = f'e_id_{save_epochs.index(epoch)}'
            save_model(model, snapshot_path, log_dict, None, None, ['best_' + save_name], [save_name])

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


def save_model(model, snapshot_path, log_dict, saved_models: Optional[Dict], save_last_n_models, new_best_metrics,
               save_metrics):
    save_path_list = []

    for save_name in save_metrics:
        best_name = 'best_' + save_name
        if best_name in new_best_metrics:
            new_dict = dict(log_dict)
            new_dict['best_name'] = best_name
            save_path_list.append((best_name, str(snapshot_path).format(**new_dict)))

    for best_name, save_path in save_path_list:
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


def find_lr_loop(cfg, model, device, optimizer, scaler, criterion_list, train_dataloader, run_folder, orig_stderr,
                 wandb_run: Run):
    start_lr, end_lr = 1e-5, 1e-0  # Adam
    # start_lr, end_lr = 1e-3, 1e+2  # SGD
    # num_batches = 100
    num_batches = len(train_dataloader)
    cur_lr = start_lr
    lr_multiplier = (end_lr / start_lr) ** (1.0 / num_batches)
    beta = 0.98

    avg_loss = 0.
    best_loss = 0.
    best_loss_lr = 0.
    losses = []
    losses_avg = []
    lrs = []
    lrs_log = []
    steps = []
    batch_size = cfg['train_data_loader']['batch_size']

    use_fp = cfg['train_data_loader'].get('use_fp', False)
    aux_weights = cfg['train_params'].get('aux_weights', None)
    grad_clipping = cfg['train_params'].get('grad_clipping', 0)

    model.train()

    progress_bar = tqdm(desc='Find LR', total=num_batches * batch_size, smoothing=0.02, file=orig_stderr)
    tr_it = iter(train_dataloader)
    for step in range(num_batches + 1):
        for i, param_group in enumerate(optimizer.param_groups):
            param_group['lr'] = cur_lr

        try:
            data = next(tr_it)
        except StopIteration:
            tr_it = iter(train_dataloader)
            data = next(tr_it)

        if progress_bar.n == 0:
            progress_bar.reset()  # reset start time to remove time while DataLoader was populating processes

        data_img, data_class, _ = data

        optimizer.zero_grad()

        if scaler is None:
            pass
            # outputs, loss, metrics = forward_pass(data_img, data_class, model, device, criterion_list,
            #                                       aux_weights=aux_weights)
            # loss.backward()
        else:
            with autocast():
                outputs, loss, metrics = forward_pass(data_img, data_class, model, device, criterion_list,
                                                      use_fp=use_fp, aux_weights=aux_weights)

            scaler.scale(loss).backward()

            if grad_clipping > 0:
                # Unscales the gradients of optimizer's assigned params in-place
                scaler.unscale_(optimizer)
                # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clipping)

            scaler.step(optimizer)
            scaler.update()

        # Compute the smoothed loss
        loss = loss.item()
        avg_loss = beta * avg_loss + (1 - beta) * loss
        smoothed_loss = avg_loss / (1 - beta ** (step + 1))

        # Record the best loss
        if smoothed_loss < best_loss or step == 0:
            best_loss = smoothed_loss
            best_loss_lr = cur_lr

        losses.append(loss)
        losses_avg.append(smoothed_loss)
        lrs.append(cur_lr)
        lrs_log.append(math.log10(cur_lr))
        steps.append(progress_bar.n)

        log_dict = {'lr': cur_lr, 'lr_log': math.log10(cur_lr), 'loss': loss, 'loss_avg': smoothed_loss,
                    'best_loss': best_loss, 'best_loss_lr': best_loss_lr}
        wandb_run.log(log_dict, step=progress_bar.n)

        progress_bar.set_postfix(log_dict, refresh=False)
        progress_bar.update(batch_size)

        # # Stop if the loss is exploding
        # if step > 0 and smoothed_loss > 6 * best_loss:
        #     break

        cur_lr = cur_lr * lr_multiplier  # increase LR

    csv_path = run_folder / 'find_lr.csv'
    with open(csv_path, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['step', 'seen', 'lr', 'lr_log', 'loss', 'loss_avg'])

        for step in range(len(losses)):
            row = [step + 1, steps[step], f"{lrs[step]:.8f}", f"{lrs_log[step]:.8f}",
                   f"{losses[step]:.8f}", f"{losses_avg[step]:.8f}"]
            writer.writerow(row)
