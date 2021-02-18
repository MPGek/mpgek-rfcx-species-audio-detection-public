import json
import multiprocessing
import os
import platform
import sys
from datetime import datetime
from multiprocessing import Queue
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import wandb
from torch.cuda.amp import GradScaler
from wandb import Settings

from model.augementations import init_mixers
from model.train_utils import make_new_seed, print_fold_train_results_and_save, store_samples_dataset_info
from model.train_validate_loops_coteaching import train_loop_coteaching
from run_scripts.train_model_simple import get_fold_group_id, build_model, build_optimizer
from utils import torchsummary
from utils.utils import StreamWrapper
from utils.wandb_utils import wandb_store_file

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from model.datasets import make_dataloader, SampleDataset


def run_fold_train(folds_count):
    train_start_time = datetime.now()
    fold_group_id = get_fold_group_id()

    queue = Queue()
    results = {'fold': []}
    for fold_index in range(folds_count):
        # process = Process(target=main,
        #                   args=(False, fold_index, fold_group_id, train_start_time, queue))
        # process.start()
        # process.join()
        main(fold_index, fold_group_id, train_start_time, queue)

        run_folder, process_results, cfg = queue.get()
        for key, value in process_results.items():
            if key not in results:
                results[key] = []
            results[key].append(value)
        results['fold'].append(fold_index)

        print_fold_train_results_and_save(results, run_folder)

    prepare_submission_files(run_folder, cfg)


def main(fold_index=None, fold_group_id=None, train_start_time=None, queue: Optional[Queue] = None):
    is_debug_mode = sys.gettrace() is not None
    train_workers = min(6, multiprocessing.cpu_count() // 2) if not is_debug_mode else 0
    val_workers = min(4, multiprocessing.cpu_count() // 2) if not is_debug_mode else 0

    multi_gpu = False
    devices_count = torch.cuda.device_count() if multi_gpu else 1

    cfg = {
        'audio_params': {
            'sampling_rate': 48000,
            # 'hop_length': 375 * 2,  # 375 * 2 - 64 pixels per second
            # 'fmin': 20, 'fmax': 20000,
            'fmin': 50, 'fmax': 15000,
            # 'n_mels': 128, 'n_fft_mels_ratio': 20,
            # 'n_mels': 224, 'n_fft_mels_ratio': 20,
            # 'n_mels': 256, 'n_fft': 4096, 'ftt_win_length': 1536, 'hop_length': 256,  # 256 - 187.5 pixels per second
            # 'n_mels': 384, 'n_fft': 4096, 'ftt_win_length': 1536, 'hop_length': 375,  # 375 - 128 pixels per second
            # 'n_mels': 256, 'n_fft': 4096, 'ftt_win_length': 1280, 'hop_length': 500,  # 500 - 96 pixels per second
            # 'n_mels': 384, 'n_fft': 4096, 'ftt_win_length': 1536, 'hop_length': 375*2,  # 375*2 - 64 pixels per second
            'n_mels': 380, 'n_fft': 4096, 'ftt_win_length': 1536, 'hop_length': 400,  # 400 - 120 pixels per second
            # 'n_mels': 380, 'n_fft': 4096, 'ftt_win_length': 1280, 'hop_length': 250,  # 250 - 192 pixels per second
        },
        'train_params': {
            'mixed_precision': True,
            'multi_gpu': multi_gpu,
            # 'epochs': 35 * 4,
            # 'limit_samples': 16,  # DEBUG
            'folds': 5,
            # 'folds': 7, 'kfold_sss': True,
            'input_channels': 1,
            # 'new_seed': 'new_seed_1',
            'save_last_n_models': 1,
            # 'epochs': 35 * 3, 'save_epochs': [35, 70, 105],
            # 'epochs': 225, 'save_epochs': [15, 45, 105],
            'epochs': 150,  # 'save_epochs': [20, 60, 140],
            # 'epochs': 150,  # 'save_epochs': [10, 30, 70],
            # 'epochs': 250,  # 'save_epochs': [20, 60, 140],

            # https://arxiv.org/pdf/1804.06872.pdf
            # R(T) = 1−min{τ ·T**c/Tk, τ, 0.95} (T=epoch-1, epoch=[1, epochs]), R(1) = 1 which possible when T is 0
            # remember_rate   - τ  = {0.5, 0.75, 1, 1.25, 1.5}
            # remember_c      - c  = {0.5, 1, 2}
            # remember_length - Tk = {5, 10, 15}
            # 'remember_rate': 0.4, 'remember_c': 0.5, 'remember_length': 5,  # become 1 on the 26th epoch (from 1)
            'remember_rate': 0.5, 'remember_c': 0.5, 'remember_length': 6,  # become 1 on the 37th epoch (from 1)
            'high_loss_train': 0.1,

            # 'reduceLR': {'patience': 1 * 5, 'factor': 0.5, 'min_lr': 1e-6},
            # 'reduceLR': {'patience': 1 * 6, 'factor': 0.6, 'min_lr': 1e-6},
            # 'reduceLR': {'patience': 1 * 4, 'factor': 0.5, 'min_lr': 1e-6},
            # 'reduceLR': {'patience': 1 * 4, 'factor': 0.6, 'min_lr': 1e-6},
            # 'reduceLR': {'patience': 1 * 4, 'factor': 0.75, 'min_lr': 1e-6},
            'reduceLR': {'patience': 1 * 5, 'factor': 0.75, 'min_lr': 1e-6},
            # 'rectLR': {'warmup_proportion': 0.1, 'min_lr': 1e-7, 'log_scale': True, 'log_weight': 2},
            # 'cosineLR': {'cycle': 35},
            # 'cosineLRWarm': {'T_0': 50, 'T_mult': 2}, 'schedule_lr_each_step': True,
            # 'cosineLRWarm': {'T_0': 30, 'T_mult': 2}, 'schedule_lr_each_step': True,
            # 'cosineLRWarm': {'T_0': 20, 'T_mult': 2}, 'schedule_lr_each_step': True,
            # 'cosineLRWarm': {'T_0': 20, 'T_mult': 2}, 'schedule_lr_each_step': True,
            # 'cosineLRWarm': {'T_0': 10, 'T_mult': 2}, 'schedule_lr_each_step': True,
            # 'cosineLRWarm': {'T_0': 15, 'T_mult': 2}, 'schedule_lr_each_step': True,
            # 'cosineLRWarm': {'T_0': 20, 'T_mult': 2}, 'schedule_lr_each_step': False,
            # 'schedule_lr_each_step': True,
            'early_stop': 45,
            # 'early_stop': 65,
            # 'early_stop': 65,
            # 'loss': 'bce_fp',
            # 'loss': 'bce',
            # 'loss': 'focal',
            # 'loss': 'LSoft0.3',
            'loss': 'LSoft0.7',
            # 'loss': 'SelNP0.3',
            # 'loss': 'LSoft0.7', 'scale_noisy_loss': True,
            'loss_reduction': 'none',
            # 'loss_reduction': 'sum',
            # 'bce_smoothing': 0.24,

            # 'pseudo_label': True,
            'grad_acc_iters': 4,
            # 'grad_clipping': 2,

            # 'model': 'en_b0_ns', 'pretrained': True, 'optimizer': 'Adam', 'lr': 2e-3,
            # 'model': 'en_b2_ns', 'pretrained': True, 'optimizer': 'Adam', 'lr': 1e-3,
            # 'model': 'en_b4_ns', 'pretrained': True, 'optimizer': 'Adam', 'lr': 1e-3,
            # 'model': 'en_b4_ns', 'pretrained': True, 'optimizer': 'Adam', 'lr': 1e-3,
            'model': 'en_b5_ns', 'pretrained': True, 'optimizer': 'Adam', 'lr': 1e-3,
            # 'model': 'en_b6_ns', 'pretrained': True, 'optimizer': 'Adam', 'lr': 1e-3,
            # 'model': 'en_b7_ns', 'pretrained': True, 'optimizer': 'Adam', 'lr': 1e-3,
            # 'grad_acc_iters': 4,
        },
        'metrics': ['bce', 'Lwlrap', 'LS7TN', 'bceTP'],  # , 'Lwlrap_Full'],
        # 'f2_metrics_threshold': 0.8,
        'watch_metrics': ['min|val_loss', 'max|val_Lwlrap', 'min|val_LS7TN', 'min|val_bceTP'],
        'save_metrics': ['val_loss', 'val_bceTP'],
        'watch_metrics_eps': 1e-5,
        'train_data_loader': {
            'key': 'train',
            # 'batch_size': 38 * devices_count,  # en0, m380, hop 400 x 6 sec (720 pixels)
            # 'batch_size': 24 * devices_count,  # en2, m380, hop 400 x 6 sec (720 pixels)
            # 'batch_size': 12 * devices_count,  # en2, m380, hop 400 x 6 sec (720 pixels)
            # 'batch_size': 14 * devices_count,  # en4, m380, hop 400 x 6 sec (720 pixels)
            # 'batch_size': 10 * devices_count,  # en5, m380, hop 400 x 6 sec (720 pixels)
            # 'batch_size': 6 * devices_count,  # en4, m380, hop 400 x 6 sec (720 pixels)
            'batch_size': 4 * devices_count,  # en5, m380, hop 400 x 6 sec (720 pixels)
            # 'batch_size': 5 * devices_count,  # en7, m380, hop 400 x 6 sec (720 pixels)
            # 'batch_size': 7 * devices_count,  # en6, m380, hop 400 x 6 sec (720 pixels)
            # 'batch_size': 20 * devices_count,  # aux, m384, hop 400 x 6 sec (720 pixels)
            'shuffle': True,
            'num_workers': train_workers,
            'wave_aug': 0,
            'mel_aug': 15, 'mixup': 8,
            # 'mel_aug': 9, 'mixup': 0,
            'cut_samples': True, 'cut_size': 10, 'random_crop': 6, 'fix_cut_targets': True,
            # 'cut_samples': True, 'cut_size': 6, 'crops_offset': 2, 'fix_cut_targets': True,
            # 'cut_samples': True, 'cut_size': 8, 'random_crop': 5, 'fix_cut_targets': True,
            # 'use_fp': True,
            # 'sample_val_fp': True,
            # 'tp_fp_sampler': True,
            # 'seed': 163,
            # 'remove_duplicates': True,

            # 'cut_samples': True, 'cut_size': 3, 'random_crop': 2, 'fix_cut_targets': True,
            # 'filter_species': [0, 1, 10, 11, 13, 14, 16, 18, 19, 2, 21, 22, 3, 4, 6],
            # 'filter_species': [12, 15, 17, 20, 23, 5, 7, 8, 9],
        },
        'test_data_loader': {
            'batch_size': 4,  # 'by_labeled_crops' will be faster with batch_size 4 - skips more cuts
            # 'batch_size': 32,
            'shuffle': False,
            'num_workers': val_workers,
            'mel_aug': 9,
        },
        'validate_params': {
            'by_crops': 6, 'crops_offset': 2, 'by_labeled_crops': True,
            # 'by_crops': 2, 'crops_offset': 1, 'by_labeled_crops': True,
            # 'by_crops': 10, 'crops_offset': 5,
            # 'by_crops': 5, 'crops_offset': 2, 'by_labeled_crops': True,
            # 'by_crops': 4, 'crops_offset': 2,
        }
    }
    # is_debug_mode = True  # DEBUG

    folds_count = cfg['train_params'].get('folds', 1)
    if folds_count > 1 and fold_index is None:
        run_fold_train(folds_count)
        return

    # group_name = 'en4 Ag14 Mix5 380x720 RC6+6-2 CoTeach rLR-5-0.75 LR1e-3 kSSS7 bce'
    # group_name = 'en4 Ag14 Mix5 380x720 RC6+6-2 CoTeach-0.4-0.5-5 cLR-35 LR1e-3 kSSS7 bce'
    # group_name = 'en4 Ag14 Mix5 380x720 RC6+6-2 FP-S CoTeach-0.4-0.5-5 cLRw-20-2 LR1e-3 kSSS7 bce'
    # group_name = 'en2 Ag15 Mix8 380x720 RC6+6-2 FP-S CoTeach-0.5-0.5-6-0.1 rLR-5-0.75 LR1e-3 kSSS7 bce'
    # group_name = 'en2 Ag15 Mix8 380x720 RC6+6-2 FP-S fCoTeach-0.5-0.5-6-0.1 rLR-5-0.75 LR1e-3 kf5 bce'
    group_name = 'en5 Ag15 Mix8 380x720 RC6+6-2 fCoTeach-0.5-0.5-6-0.1 rLR-5-0.75 LR1e-3 kf5 LSoft0.7 gAcc4'

    wandb_name = None

    # set env variable for data
    os.environ['DATA_FOLDER'] = '../data'
    is_notebook = platform.node() == 'nb-162'

    # ===== Folders
    fold_group_id = fold_group_id or get_fold_group_id()
    train_start_time = train_start_time or datetime.now()
    run_folder_suffix = 'gf_' + fold_group_id
    run_folder = Path(os.environ['DATA_FOLDER']) / 'runs'
    run_folder = run_folder / (train_start_time.strftime('run_%y-%m-%d_%H-%M-%S') + '_' + run_folder_suffix)
    run_folder.mkdir(parents=True, exist_ok=True)

    # ===== Loggers
    orig_stdout, orig_stderr = sys.stdout, sys.stderr
    sys.stdout = StreamWrapper(run_folder / f'output.log', orig_stdout)
    sys.stderr = StreamWrapper(run_folder / f'error.log', orig_stderr)

    if folds_count > 1:
        def _print_fold_start(f):
            print('=' * 90, file=f)
            print(f'Fold: {fold_index + 1}/{folds_count}', file=f)
            print('=' * 90, file=f)

        _print_fold_start(sys.stdout)
        _print_fold_start(sys.stderr)

    # ===== Initialize hyper_parameter_defaults
    hyper_parameter_defaults = dict(
        train=cfg['train_params'],
        audio=cfg['audio_params'],
        d_train=cfg['train_data_loader'],
        val=cfg['validate_params'],
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES', -1),
    )

    hyper_parameter_defaults['fold_group'] = fold_group_id
    file_suffix = ''
    if fold_index is not None:
        hyper_parameter_defaults['fold'] = fold_index
        file_suffix = f'_f{fold_index}'
    else:
        fold_index = 0  # default value to store models in the folder 'fold_0'

    print('Group:', group_name)
    print('Parameters: {}'.format(json.dumps(hyper_parameter_defaults, ensure_ascii=False, indent=2)))

    # ===== Initialize wandb
    wandb_run = wandb.init(config=hyper_parameter_defaults, group=group_name, settings=Settings(console="off"),
                           name=wandb_name, mode="online" if not is_debug_mode else "offline")

    # ===== Other files

    # snapshot_path = 'model_{epoch:03d}_loss_{val_loss:.4f}_nll_{val_nll:.4f}.pt'
    snapshot_folder = run_folder / f'models_fold_{fold_index}'
    snapshot_folder.mkdir(parents=True, exist_ok=True)
    snapshot_path_1 = snapshot_folder / 'model_{best_name}_{epoch:03d}_loss_{val_loss:.4f}_' \
                                      'bce_{val_bce:.4f}_bceTP_{val_bceTP:.4f}_' \
                                      'Lwlrap_{val_Lwlrap:.4f}.pt'
    snapshot_path_2 = snapshot_folder / 'model2_{best_name}_{epoch:03d}_loss_{val_loss:.4f}_' \
                                        'bce_{val_bce:.4f}_bceTP_{val_bceTP:.4f}_' \
                                        'Lwlrap_{val_Lwlrap:.4f}.pt'

    csv_path = run_folder / f'train{file_suffix}.csv'
    cfg_path = run_folder / f'cfg.json'
    with open(cfg_path, 'w', encoding='utf-8') as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    print('=' * 36, 'Config', '=' * 36, file=sys.stdout.file)
    json.dump(cfg, sys.stdout.file, ensure_ascii=False, indent=2)
    print('', file=sys.stdout.file)
    print('=' * 36, 'Config', '=' * 36, file=sys.stdout.file)

    wandb_store_file(wandb_run, cfg_path)

    # ===== INIT DATASET
    print("Opening dataset...")
    init_mixers(cfg['audio_params']['fmin'], cfg['audio_params']['fmax'], cfg['audio_params']['n_mels'])

    seed = cfg['train_data_loader'].get('seed', 1327)
    kfold_statified_shuffle_splits = cfg['train_params'].get('kfold_sss', False)
    samples_dataset = SampleDataset(cfg['audio_params'], cfg['train_data_loader'], folds_count,
                                    disable_cache=is_notebook,
                                    kfold_statified_shuffle_splits=kfold_statified_shuffle_splits)

    train_dataloader, train_ds = make_dataloader(samples_dataset, samples_dataset.train_split[fold_index], cfg, 'train',
                                                 seed=seed)
    val_dataloader, val_ds = make_dataloader(samples_dataset, samples_dataset.test_split[fold_index], cfg, 'test',
                                             sort_t_min=True)

    store_samples_dataset_info(run_folder, samples_dataset.classes_labels, train_ds.dataset_split,
                               val_ds.dataset_split, file_suffix, wandb_run)

    # ==== INIT MODEL
    print("Init model...")
    # Ensure deterministic behavior
    make_new_seed(cfg['train_params'].get('new_seed', ''))

    device_1 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device_2 = torch.device("cuda:1" if torch.cuda.is_available() and torch.cuda.device_count() > 1 else device_1)
    model_1 = build_model(cfg, samples_dataset.classes_num, fold_index).to(device_1)
    model_2 = build_model(cfg, samples_dataset.classes_num, fold_index).to(device_2)

    print("Models was loaded")

    class_map = {i: v for i, v in enumerate(samples_dataset.classes_labels)}
    optimizer_1, criterion_list_1 = build_optimizer(cfg, model_1, class_map, samples_dataset.classes_num)
    optimizer_2, criterion_list_2 = build_optimizer(cfg, model_2, class_map, samples_dataset.classes_num)

    # Creates a GradScaler once at the beginning of training.
    scaler = GradScaler() if cfg['train_params']['mixed_precision'] else None

    model_info_path = run_folder / f'model_info.log'
    with open(model_info_path, 'w', encoding='utf-8') as f:
        json.dump(cfg['train_params'], f, ensure_ascii=False, indent=2)
        sample_data, _, _ = train_dataloader.dataset[0]  # type: np.ndarray
        input_shape = sample_data.shape
        print('\n' + '=' * 80, file=f)

        def _print(to_file):
            print('Input shape:', input_shape, file=to_file)
            torchsummary.summary(model_1, tuple(input_shape), cfg['train_data_loader']['batch_size'], file=to_file)
            print(model_1, file=to_file)

        _print(f)
        if is_debug_mode:
            _print(None)
    wandb_store_file(wandb_run, model_info_path)

    # wandb_run.watch(model, criterion_list[0], log=None, log_freq=200)

    # ==== TRAIN LOOP
    best_metrics = train_loop_coteaching(cfg, model_1, device_1, optimizer_1, model_2, device_2, optimizer_2, scaler,
                                         criterion_list_1, criterion_list_2, train_dataloader, val_dataloader,
                                         run_folder, csv_path, snapshot_path_1, snapshot_path_2, orig_stderr,
                                         fold_index, folds_count)
    if queue is not None:
        queue.put((run_folder, best_metrics, cfg))

    wandb_run.finish()

    # ===== Close Loggers
    sys.stdout.close()
    sys.stderr.close()
    sys.stdout, sys.stderr = orig_stdout, orig_stderr

    # ===== Submission
    if fold_index is None:
        prepare_submission_files(run_folder, cfg)


def prepare_submission_files(run_folder, cfg):
    from run_scripts.validate_submit_model import do_validate_submit

    save_names = []

    for metric_name in cfg.get('save_metrics', []):
        save_names.append(metric_name)
    for idx, epoch in enumerate(cfg['train_params'].get('save_epochs', [])):
        save_names.append(f'e_id_{idx}')

    if len(save_names) > 0:
        for save_name in save_names:
            do_validate_submit(is_submit_mode=True, run_name=run_folder.name, model_prefix='best_' + save_name)
    else:
        do_validate_submit(is_submit_mode=True, run_name=run_folder.name)


if __name__ == '__main__':
    main()
