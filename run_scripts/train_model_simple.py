import argparse
import json
import multiprocessing
import os
import platform
import sys
import time
from datetime import datetime
from multiprocessing import Queue
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import wandb
from timm.optim import RAdam
from torch import nn, optim
from torch.cuda.amp import GradScaler
from wandb import Settings

from model.augementations import init_mixers
from model.losses import BCELossWithSmoothing, BinaryRecallLoss, FocalLoss, CombinedTPLSoftLoss, CombinedNoisyLoss, \
    LSoftLoss, NPLoss
from model.metrics import Lwlrap, BinaryRecallMetrics, BinaryPrecisionMetrics, BinaryFScoreMetrics, CombinedMetrics
from model.rainforest_models import RainforestModel
from model.train_utils import make_new_seed, print_fold_train_results_and_save, store_samples_dataset_info
from model.train_validate_loops import train_loop, find_lr_loop
from utils import torchsummary
from utils.utils import StreamWrapper
from utils.wandb_utils import wandb_store_file

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from model.datasets import make_dataloader, SampleDataset


def build_model(cfg: Dict, classes_num: int, fold_index=0) -> torch.nn.Module:
    train_params = cfg['train_params']  # type: Dict
    model_name = train_params['model']
    model_top = train_params.get('model_top', '')
    pretrained = train_params['pretrained']
    use_coord_conv = train_params['use_coord_conv'] if 'use_coord_conv' in train_params else False
    efficient_net_hyper_column = train_params.get('efficient_net_hyper_column', False)

    input_channels = train_params.get('input_channels', 1)
    backbone = train_params.get('backbone', None)
    model = RainforestModel(model_name, model_top, pretrained, use_coord_conv, efficient_net_hyper_column,
                            input_channels, classes_num, backbone_name=backbone,
                            input_height=cfg['audio_params']['n_mels'])

    if 'model_weights' in train_params:
        model_weights = train_params['model_weights']
        if isinstance(model_weights, list):
            model_weights = model_weights[fold_index]
        if os.path.exists(model_weights):
            model_state_dict = torch.load(model_weights)
            model.load_state_dict(model_state_dict, strict=not efficient_net_hyper_column)
            print("Was loaded model weights '{}'".format(model_weights))

    return model


def build_optimizer(cfg, model, class_map, classes_num):
    optimizer = None
    if cfg['train_params']['optimizer'] == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=cfg['train_params']['lr'])
    if cfg['train_params']['optimizer'] == 'RAdam':
        optimizer = RAdam(model.parameters(), lr=cfg['train_params']['lr'])
    elif cfg['train_params']['optimizer'] == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=cfg['train_params']['lr'], momentum=0.9)

    class_num_tp = 0
    if cfg['train_data_loader'].get('use_fp', False):
        class_num_tp = classes_num // 2

    def _get_loss(name, is_loss=False):
        reduction = cfg['train_params'].get('loss_reduction', 'mean')
        combined_noisy_loss_reduction = reduction if is_loss else 'mean'
        bce_smoothing = cfg['train_params'].get('bce_smoothing', 0)
        f2_metrics_threshold = cfg.get('f2_metrics_threshold', None)
        scale_noisy_loss = cfg['train_params'].get('scale_noisy_loss', False)

        if name == 'mse':
            return nn.MSELoss(reduction="none")
        elif name == 'mae':
            return nn.L1Loss(reduction="none")
        elif name == 'bce':
            if bce_smoothing <= 0:
                return nn.BCEWithLogitsLoss(reduction=reduction)
            else:
                return BCELossWithSmoothing(bce_smoothing, reduction=reduction)
        elif name == 'bce_fp':
            return CombinedNoisyLoss(nn.BCEWithLogitsLoss(reduction='none'), None, scale_filtered_loss=scale_noisy_loss,
                                     combined_noisy_loss_reduction=combined_noisy_loss_reduction)
        elif name == 'bce_r':
            return CombinedMetrics([nn.BCEWithLogitsLoss(reduction=reduction), BinaryRecallLoss()], [1, 0.5])
        elif name == 'bce_r1':
            return CombinedMetrics([nn.BCEWithLogitsLoss(reduction=reduction), BinaryRecallLoss()], [1, 1])
        elif name == 'bce_r2':
            return CombinedMetrics([nn.BCEWithLogitsLoss(reduction=reduction), BinaryRecallLoss()], [1, 2])
        elif name == 'r':
            return BinaryRecallMetrics(threshold=f2_metrics_threshold)
        elif name == 'rL':
            return BinaryRecallLoss()
        elif name == 'focal':
            return FocalLoss()
        elif name == 'LSoft0.3':
            return CombinedTPLSoftLoss(0.3, scale_filtered_loss=scale_noisy_loss,
                                     combined_noisy_loss_reduction=combined_noisy_loss_reduction)
        elif name == 'LSoft0.7':
            return CombinedTPLSoftLoss(0.7, scale_filtered_loss=scale_noisy_loss,
                                     combined_noisy_loss_reduction=combined_noisy_loss_reduction)
        elif name == 'LS7TN':
            return CombinedNoisyLoss(None, LSoftLoss(0.7, reduction='none'), scale_filtered_loss=scale_noisy_loss,
                                     combined_noisy_loss_reduction=combined_noisy_loss_reduction)
        elif name == 'NP3':
            return NPLoss(0.3, nn.BCEWithLogitsLoss(reduction='none'), None, scale_filtered_loss=scale_noisy_loss)
        elif name == 'SelNP0.3':
            return CombinedNoisyLoss(
                nn.BCEWithLogitsLoss(reduction='none'),
                NPLoss(0.3, nn.BCEWithLogitsLoss(reduction='none'), None, scale_filtered_loss=scale_noisy_loss),
                scale_filtered_loss=scale_noisy_loss,
                                     combined_noisy_loss_reduction=combined_noisy_loss_reduction)
        elif name == 'bceTP':
            return CombinedNoisyLoss(nn.BCEWithLogitsLoss(reduction='none'), None, scale_filtered_loss=scale_noisy_loss,
                                     combined_noisy_loss_reduction=combined_noisy_loss_reduction)
        elif name == 'p':
            return BinaryPrecisionMetrics(threshold=f2_metrics_threshold)
        elif name == 'f1':
            return BinaryFScoreMetrics(1, threshold=f2_metrics_threshold)
        elif name == 'f2':
            return BinaryFScoreMetrics(2, threshold=f2_metrics_threshold)
        elif name == 'Lwlrap':
            return Lwlrap(class_map, class_num_tp)
        elif name == 'Lwlrap_Full':
            return Lwlrap(class_map)
        else:
            raise ValueError(f"Can't find metric with name '{name}'")

    criterion_list = [_get_loss(cfg['train_params']['loss'], True)]
    criterion_list.extend([_get_loss(metric_name) for metric_name in cfg['metrics']])

    return optimizer, criterion_list


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
        main(False, fold_index, fold_group_id, train_start_time, queue)

        run_folder, process_results, cfg = queue.get()
        for key, value in process_results.items():
            if key not in results:
                results[key] = []
            results[key].append(value)
        results['fold'].append(fold_index)

        print_fold_train_results_and_save(results, run_folder)

    prepare_submission_files(run_folder, cfg)


def get_fold_group_id():
    time_with_offset = int(time.time() - 1606780800) // 60  # minutes after the 2020-12-01
    fold_group_id = f'{time_with_offset:05x}'
    return fold_group_id


def main(is_find_lr_mode, fold_index=None, fold_group_id=None, train_start_time=None, queue: Optional[Queue] = None):
    is_debug_mode = sys.gettrace() is not None
    train_workers = min(6, multiprocessing.cpu_count() // 2) if not is_debug_mode else 0
    val_workers = min(4, multiprocessing.cpu_count() // 2) if not is_debug_mode else 0

    multi_gpu = False
    multi_gpu = True if not is_find_lr_mode else False
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
            # 'limit_samples': 100,  # DEBUG
            # 'folds': 5,
            'folds': 7, 'kfold_sss': True,
            'input_channels': 1,
            # 'new_seed': 'new_seed_1',
            'save_last_n_models': 1,
            # 'epochs': 35 * 3, 'save_epochs': [35, 70, 105],
            # 'epochs': 225, 'save_epochs': [15, 45, 105],
            'epochs': 140,  # 'save_epochs': [20, 60, 140],
            # 'epochs': 150,  # 'save_epochs': [10, 30, 70],
            # 'epochs': 250,  # 'save_epochs': [20, 60, 140],

            # 'reduceLR': {'patience': 1 * 5, 'factor': 0.5, 'min_lr': 1e-6},
            # 'reduceLR': {'patience': 1 * 6, 'factor': 0.6, 'min_lr': 1e-6},
            # 'reduceLR': {'patience': 1 * 4, 'factor': 0.5, 'min_lr': 1e-6},
            # 'reduceLR': {'patience': 1 * 4, 'factor': 0.6, 'min_lr': 1e-6},
            # 'reduceLR': {'patience': 1 * 4, 'factor': 0.75, 'min_lr': 1e-6},
            # 'reduceLR': {'patience': 1 * 5, 'factor': 0.75, 'min_lr': 1e-6},
            # 'rectLR': {'warmup_proportion': 0.1, 'min_lr': 1e-7, 'log_scale': True, 'log_weight': 2},
            # 'cosineLR': {'cycle': 35},
            # 'cosineLRWarm': {'T_0': 50, 'T_mult': 2}, 'schedule_lr_each_step': True,
            # 'cosineLRWarm': {'T_0': 30, 'T_mult': 2}, 'schedule_lr_each_step': True,
            'cosineLRWarm': {'T_0': 20, 'T_mult': 2}, 'schedule_lr_each_step': True,
            # 'cosineLRWarm': {'T_0': 10, 'T_mult': 2}, 'schedule_lr_each_step': True,
            # 'cosineLRWarm': {'T_0': 15, 'T_mult': 2}, 'schedule_lr_each_step': True,
            # 'cosineLRWarm': {'T_0': 20, 'T_mult': 2}, 'schedule_lr_each_step': False,
            # 'schedule_lr_each_step': True,
            # 'early_stop': 45,
            # 'early_stop': 50,
            'early_stop': 65,
            # 'loss': 'bce_fp',
            # 'loss': 'bce_r',
            # 'loss': 'focal',
            # 'loss': 'LSoft0.3',
            'loss': 'LSoft0.7',
            # 'loss': 'SelNP0.3',
            # 'loss': 'LSoft0.7', 'scale_noisy_loss': True,
            # 'loss_reduction': 'none',
            # 'loss_reduction': 'sum',
            # 'bce_smoothing': 0.24,

            # 'pseudo_label': True,
            'grad_acc_iters': 8,
            # 'grad_clipping': 2,

            # 'model': 'resnet50', 'pretrained': True, 'optimizer': 'Adam', 'lr': 2e-3,
            # 'model': 'resnet34', 'pretrained': True, 'optimizer': 'Adam', 'lr': 2e-3,
            # 'model': 'resnet18', 'pretrained': True, 'optimizer': 'Adam', 'lr': 2e-3,
            # 'model': 'en_b0_ns', 'pretrained': True, 'optimizer': 'Adam', 'lr': 2e-3,
            # 'model': 'en_b0_ns', 'pretrained': True, 'optimizer': 'Adam', 'lr': 3e-3,
            # 'model': 'en_b2_ns', 'pretrained': True, 'optimizer': 'Adam', 'lr': 1e-3,
            # 'model': 'en_b2_ns', 'pretrained': True, 'optimizer': 'Adam', 'lr': 2e-3,
            # 'model': 'en_b2_ns', 'pretrained': False, 'optimizer': 'Adam', 'lr': 2e-3,
            # 'model': 'en_b4_ns', 'pretrained': True, 'optimizer': 'Adam', 'lr': 1e-3,
            # 'model': 'en_b4_ns', 'pretrained': True, 'optimizer': 'Adam', 'lr': 1e-3,
            # 'model': 'en_b4_ns', 'pretrained': True, 'optimizer': 'RAdam', 'lr': 4e-3,
            # 'model': 'en_b4_ns', 'pretrained': True, 'optimizer': 'SGD', 'lr': 1e-0,
            # 'model': 'en_b5_ns', 'pretrained': True, 'optimizer': 'Adam', 'lr': 0.6e-3,
            # 'model': 'en_b5_ns', 'pretrained': True, 'optimizer': 'RAdam', 'lr': 4e-3,
            # 'model': 'en_b6_ns', 'pretrained': True, 'optimizer': 'Adam', 'lr': 1e-3,
            'model': 'en_b7_ns', 'pretrained': True, 'optimizer': 'Adam', 'lr': 1e-3,
            # 'model': 'en_b7_ns', 'pretrained': True, 'optimizer': 'Adam', 'lr': 0.3e-3,
            # 'model': 'en_b7_ns', 'pretrained': True, 'optimizer': 'RAdam', 'lr': 4e-3,
            # 'model': 'x41', 'pretrained': True, 'optimizer': 'Adam', 'lr': 1e-3,
            # 'model': 'x71', 'pretrained': True, 'optimizer': 'Adam', 'lr': 1e-3,
            # 'model': 'resnest50d', 'pretrained': True, 'optimizer': 'Adam', 'lr': 0.4e-3,
            # 'model': 'seresnet152d_320', 'pretrained': True, 'optimizer': 'Adam', 'lr': 0.3e-3,
            # 'model': 'aux', 'pretrained': False, 'optimizer': 'Adam', 'lr': 9e-2, 'aux_weights': [1, 0.4, 0.2, 0.1],
            # 'model': 'aux', 'pretrained': False, 'optimizer': 'Adam', 'lr': 2e-2, 'aux_weights': [1, 0.4, 0.2, 0.1],
            # 'model': 'aux', 'pretrained': False, 'optimizer': 'SGD', 'lr': 1.5, 'aux_weights': [1, 0.4, 0.2, 0.1],
            # 'grad_acc_iters': 4,
            # 'model': 'aux2', 'backbone': 'tf_efficientnet_b2_ns', 'pretrained': True, 'optimizer': 'Adam',
            # 'model_top': 'fc1', 'lr': 3e-3,
            # 'model_top': 'fc3', 'lr': 1.5e-3,
            # 'model_top': 'fc1-skip3', 'lr': 2e-3,
            # 'model_top': 'fc1-skip2-dc5', 'lr': 2e-3,
            # 'model_top': 'fc1-att', 'lr': 2e-3,
            # 'model': 'aux2', 'backbone': 'tf_efficientnet_b2_ns', 'pretrained': True, 'optimizer': 'Adam', 'lr': 6e-2,
            # 'model_top': 'fc2'
            # 'model_top': 'v3',
            # 'model_weights': r'..\data\runs\_best_w\w_14299_f3_model_best_val_loss_212_loss_0.1217_bce_0.1059_r_0.9307_p_0.5596_f2_0.7933_Lwlrap_0.8973_meanLB0.894.pt',
            # 'model_weights': [
            #     r'..\data\runs\_best_w\14299\f0_model_best_val_loss_058_loss_0.1460_bce_0.1226_r_0.9000_p_0.4815_f2_0.7398_Lwlrap_0.8675.pt',
            #     r'..\data\runs\_best_w\14299\f1_model_best_val_loss_110_loss_0.1338_bce_0.1123_r_0.9053_p_0.5106_f2_0.7610_Lwlrap_0.8738.pt',
            #     r'..\data\runs\_best_w\14299\f2_model_best_val_loss_108_loss_0.1263_bce_0.0967_r_0.9107_p_0.5123_f2_0.7672_Lwlrap_0.8924.pt',
            #     r'..\data\runs\_best_w\14299\f3_model_best_val_loss_212_loss_0.1217_bce_0.1059_r_0.9307_p_0.5596_f2_0.7933_Lwlrap_0.8973.pt',
            #     r'..\data\runs\_best_w\14299\f4_model_best_val_loss_042_loss_0.1517_bce_0.1644_r_0.9171_p_0.3967_f2_0.6966_Lwlrap_0.8716.pt',
            #     r'..\data\runs\_best_w\14299\f5_model_best_val_loss_048_loss_0.1379_bce_0.1432_r_0.9196_p_0.4371_f2_0.7300_Lwlrap_0.8654.pt',
            #     r'..\data\runs\_best_w\14299\f6_model_best_val_loss_091_loss_0.1337_bce_0.1322_r_0.9166_p_0.4830_f2_0.7486_Lwlrap_0.8695.pt',
            # ],
        },
        'metrics': ['bce', 'rL', 'r', 'p', 'f2', 'Lwlrap', 'LS7TN', 'bceTP', 'NP3'],  # , 'Lwlrap_Full'],
        # 'f2_metrics_threshold': 0.8,
        'watch_metrics': ['min|val_loss', 'max|val_Lwlrap', 'max|val_f2', 'min|val_LS7TN', 'min|val_bceTP'],
        'save_metrics': ['val_loss', 'val_Lwlrap', 'val_bceTP'],
        'watch_metrics_eps': 1e-5,
        'train_data_loader': {
            'key': 'train',
            # 'batch_size': 10 * devices_count,  # m224, hop 375 * 2 x 60 sec
            # 'batch_size': 16 * devices_count,  # m128, hop 375 * 2 x 60 sec
            # 'batch_size': 128 * devices_count,  # m128, hop 375 * 2 x 4 sec
            # 'batch_size': 96 * devices_count,  # m128, hop 375 * 2 x 10 sec
            # 'batch_size': 16 * devices_count,  # en0, m256, hop 256 x 10 sec (1875 pixels)
            # 'batch_size': 32 * devices_count,  # en0, m384, hop 375 x 6 sec (768 pixels)
            # 'batch_size': 38 * devices_count,  # en0, m380, hop 400 x 6 sec (720 pixels)
            # 'batch_size': 24 * devices_count,  # ResNeSt50, m384, hop 375 x 6 sec (768 pixels)
            # 'batch_size': 12 * devices_count,  # en4, m384, hop 375 x 6 sec (768 pixels)
            # 'batch_size': 24 * devices_count,  # en2, m380, hop 400 x 6 sec (720 pixels)
            # 'batch_size': 24 * devices_count,  # aux-en2, m380, hop 400 x 6 sec (720 pixels)
            # 'batch_size': 48 * devices_count,  # aux-en2, m380, hop 250 x 2 sec (384 pixels)
            # 'batch_size': 20 * devices_count,  # aux-en2-fc1-skip3, m380, hop 400 x 6 sec (720 pixels)
            # 'batch_size': 18 * devices_count,  # aux-en2-fc1-skip3-pa7, m380, hop 400 x 6 sec (720 pixels)
            # 'batch_size': 20 * devices_count,  # aux-en2-fc1-skip2-pa7, m380, hop 400 x 6 sec (720 pixels)
            # 'batch_size': 24 * devices_count,  # aux-en2-fc3, m380, hop 400 x 6 sec (720 pixels)
            # 'batch_size': 16 * devices_count,  # x41, m380, hop 400 x 6 sec (720 pixels)
            # 'batch_size': 8 * devices_count,  # x71, m380, hop 400 x 6 sec (720 pixels)
            # 'batch_size': 14 * devices_count,  # en4, m380, hop 400 x 6 sec (720 pixels)
            # 'batch_size': 12 * devices_count,  # en4, m380, hop 400 x 7 sec (840 pixels)
            # 'batch_size': 10 * devices_count,  # en5, m380, hop 400 x 6 sec (720 pixels)
            # 'batch_size': 12 * devices_count,  # seRN152d, m380, hop 400 x 6 sec (720 pixels)
            # 'batch_size': 24 * devices_count,  # en4, m256, hop 500 x 6 sec (576 pixels)
            # 'batch_size': 16 * devices_count,  # en4, m380, hop 400 x 5 sec (600 pixels)
            # 'batch_size': 4 * devices_count,  # en7, m384, hop 375 x 6 sec (768 pixels)
            'batch_size': 5 * devices_count,  # en7, m380, hop 400 x 6 sec (720 pixels)
            # 'batch_size': 7 * devices_count,  # en6, m380, hop 400 x 6 sec (720 pixels)
            # 'batch_size': 24 * devices_count,  # en4, m384, hop 750 x 6 sec (384 pixels)
            # 'batch_size': 48 * devices_count,  # rn34, m256, hop 256 x 10 sec (1875 pixels)
            # 'batch_size': 20 * devices_count,  # rn50, m256, hop 256 x 10 sec (1875 pixels)
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
    if not is_find_lr_mode and folds_count > 1 and fold_index is None:
        run_fold_train(folds_count)
        return

    # group_name = 'en4 Ag14 Mix3 F2 384x768 RC6+v6-2woE rLR-6-0.6 kf-SSS-10'
    # group_name = 'en4 Ag14 Mix3 F2 384x768 RC6+v6-2woE rLR-6-0.6 kf5 g0'
    # group_name = 'en4 Ag14 Mix3 F2 380x720 RC6+v6-2woE rLR-6-0.6 kSSS7 g0'
    # group_name = 'en4 Ag14 Mix3 380x720 RC6+6-2 cLR-35 kSSS7 bceR g0'
    # group_name = 'en0 Ag14 Mix3 380x720 RC6+6-2 cLR-35 kSSS7 bceR1 gAcc4 g0'
    # group_name = 'en7 Ag14 Mix5 380x720 RC6+6-2 rLR-0.1-50-l2 FULL kSSS7 bceR gAcc4 g1'
    # group_name = 'en7 Ag14 Mix5 380x720 RC6+6-2 cLR-35 FULL kf4 bceR gAcc4 g0'
    # group_name = 'en4 Ag14 Mix3 380x720 RC6+6-2 FP-Sampling Adam cLRw-20-2 kSSS7 bce g1'
    # group_name = 'en4 Ag14 Mix3 380x720 RC6+6-2 FP-Sampling Adam cLRw-20-2 kSSS7 bce g1'
    # group_name = 'en4 Ag14 Mix5 380x720 RC6+6-2 FP S163 cLRw-10-2 LR1e-3 kf5 LSoft0.7 g0'
    # group_name = 'x41 Ag14 Mix5 380x720 RC6+6-2 cLRw-20-2 LR1e-3 kSSS7 LSoft0.7 gAcc4 g1'
    # group_name = 'x71 Ag14 Mix5 380x720 RC6+6-2 cLRw-20-2 LR1e-3 kSSS7 LSoft0.7 gAcc8 g1'
    # group_name = 'en4 Ag9 Mix0 380x720 RC6+6-2 FP-Sampling Adam cLRw-20-2 kSSS7 bce g0'
    # group_name = 'en7 Ag14 Mix3 380x720 RC6+6-2 FP-Sampling Adam cLRw-20-2 kSSS7 bce g0'
    # group_name = 'en7 Ag14 Mix5 380x720 RC6+6-2 FP-S LR 3e-4 cLRw-30-2 kf5 bce gAcc4 g2'
    group_name = 'en7 Ag15 Mix8 380x720 RC6+6-2 cLRw-20-2 LR1-3 kSSS7 LSoft0.7 gAcc8 g0'
    # group_name = 'en7 Ag14 Mix5 380x720 RC6+6-2 RAdam LR4e-3 cLRw-10-2 kSSS7 LSoft0.7 gAcc10 gClip2 g0'
    # group_name = 'en5 Ag14 Mix5 380x720 RC6+6-2 FP-S RAdam LR4e-3 cLRw-10-2 kSSS7 bce gAcc4 gClip2 g1'
    # group_name = 'en4 Ag14 Mix5 380x720 RC6+6-2 FP-S RAdam LR4e-3 cLRw-10-2 kSSS7 bce gAcc4 gClip2 g1'
    # group_name = 'aux Ag14 Mix3 380x720 RC6+6-2 rLR-6-0.6 LR 9e-2 kf5 bceR g0'
    # group_name = 'aux Ag14 Mix3 380x720 RC6+6-2 FP-S rLR-6-0.6 LR 2e-2 kf5 bce g1'
    # group_name = 'aux Ag14 Mix3 380x720 RC6+6-2 SGD cLRw-20-2 LR1.5 kf5 bceR g0'
    # group_name = 'aux Ag14 Mix3 380x720 RC6+6-2 FP-S SDG cLRw-50-2 LR0.1 kf5 bce g1'
    # group_name = 'aux-en2-fc1 W Ag14 Mix5 380x720 FP-S RC6+6-2 cLRw-20-2 LR0.5e-3 kSS7 LSoft0.7 g1'
    # group_name = 'aux-en2-fc1 W Ag14 Mix5 380x720 FP-S RC6+6-2 rLR-5-0.75 LR1e-4 kSS7 LSoft0.7 g0'
    # group_name = 'aux-en2-fc1 Ag14 Mix5 380x720 RC6+6-2 rLR-5-0.75 LR2e-3 kSS7 LSoft0.7 g1'
    # group_name = 'aux-en2-fc1 Ag14 Mix5 380x720 FP-S S163 RC6+6-2 rLR-5-0.75 LR1e-3 kSS7 bce g0'
    # group_name = 'aux-en2-fc1 Ag14 Mix5 380x720 FP-S S163 RC6+6-2 cLRw-20-2 LR1e-3 kSS7 bce g1'
    # group_name = 'aux-en2-fc1-s2-pa7 Ag14 Mix5 380x720 RC6+6-2 rLR-5-0.75 LR3e-3 kSS7 LSoft0.7 g1'
    # group_name = 'aux-en2-fc1-s2-dc5 Ag14 Mix5 380x720 RC6+6-2 rLR-5-0.75 LR2e-3 kSS7 LSoft0.7 g1'
    # group_name = 'aux-en2-fc1-s3 Ag14 Mix5 380x720 RC6+6-2 rLR-5-0.75 LR2e-3 kSS7 LSoft0.7 g0'
    # group_name = 'aux-en2-fc1 Ag14 Mix0 380x720 RC6+6-2 cLRw-20-2 LR3e-3 kSS7 LSoft0.7 g1'
    # group_name = 'aux-en2-fc1 Ag14 Mix5 380x384 RC3+2-1 dS cLRw-20-2 LR3e-3 kSS7 LSoft0.7 g0'
    # group_name = 'aux-en2-fc1 Ag14 Mix5 380x384 RC3+2-1 dS FP-S S163 cLRw-20-2 LR3e-3 kSS7 bce g1'
    # group_name = 'aux-en2-fc1 Ag14 Mix5 380x720 RC6+6-2 dL cLRw-20-2 LR3e-3 kSS7 LSoft0.7 g0'
    # group_name = 'aux-en2-fc3 Ag14 Mix5 380x720 RC6+6-2 cLRw-20-2 LR1.5e-3 kSS7 LSoft0.7 g1'
    # group_name = 'aux-en7-fc1 Ag14 Mix5 380x720 FP-S S163 RC6+6-2 cLRw-20-2 LR5-4 kSS7 LSoft0.7 gAcc4 g0'
    # group_name = 'aux-en7-fc1 Ag14 Mix5 380x720 FP-S S163 RC6+6-2 cLRw-20-2 LR5-4 kSS7 bce gAcc4 g1'
    # group_name = 'aux-en2-fc1 Ag14 Mix5 380x720 Cut6-2 cLRw-20-2 LR2e-3 kSS7 LSoft0.7 g0'
    # group_name = 'aux-en2-fc1 Ag14 Mix5 380x720 RC6+6-2 cLRw-20-2 LR1.2e-3 kSS7 LSoft0.7 g1'
    # group_name = 'aux-en2-fc1 Ag14 Mix5 380x720 RC6+6-2 cLRw-15-2 LR2e-3 kSS7 bceR g1'
    # group_name = 'seRN152d Ag14 Mix3 380x720 RC6+6-2 FP-Sampling Adam cLRw-20-2 kf5 bce g0'
    # group_name = 'en2 Ag14 Mix5 380x720 RC6+6-2 FP-S Adam cLRw-20-2 kf5 bce g1'
    # group_name = 'en2 Ag14 Mix5 380x720 RC6+6-2 FP-S cLRw-20-2 LR1e-3 kf5 SelNP0.3 g0'
    # group_name = 'en2 Ag14 Mix5 380x720 RC6+6-2 FP-S cLR-35 LR1e-3 kf5 SelNP0.3 g0'
    # group_name = 'en2 Ag14 Mix5 380x720 RC6+6-2 FP-S rLR-5-0.75 LR1e-3 kf5 SelNP0.3 g1'
    # group_name = 'en2 Ag14 Mix5 380x720 RC6+6-2 cLRw-20-2 kf5 bce g0'
    # group_name = 'en2 Ag14 Mix5 380x720 RC6+6-2 cLRw-15-2 LR2e-3 kSS7 bceR g0'
    # group_name = 'en2 Ag14 Mix5 380x720 RC6+6-2 cLRw-15-2 LR2e-3 kSS7 LSoft0.7-Scaled g0'
    # group_name = 'en2 Ag14 Mix5 380x720 RC6+6-2 FP-S cLRw-15-2 LR1e-3 kSS7 LSoft0.7 g1'
    # group_name = 'en4 Ag14 Mix5 380x840 RC7+7-2 cLR-35 kSSS7 bceR gAcc4 g1'
    # group_name = 'en5 Ag14 Mix3 380x720 RC6+6-2 cLR-30 kSSS7 g0'
    # group_name = 'seRN152d Ag14 Mix3 380x720 RC6+6-2 cLR-30 kSSS7 g1'
    wandb_name = None
    if is_find_lr_mode:
        folds_count = cfg['train_params']['folds'] = -1
        # fold_index = 0
        wandb_name = group_name
        group_name = 'Find LR'

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
    snapshot_path = snapshot_folder / 'model_{best_name}_{epoch:03d}_loss_{val_loss:.4f}_' \
                                      'bce_{val_bce:.4f}_' \
                                      'bceTP_{val_bceTP:.4f}_NP3_{val_NP3:.4f}_' \
                                      'r_{val_r:.2f}_p_{val_p:.2f}_f2_{val_f2:.2f}_' \
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
    if not is_find_lr_mode:
        val_dataloader, val_ds = make_dataloader(samples_dataset, samples_dataset.test_split[fold_index], cfg, 'test',
                                                 sort_t_min=True)

        store_samples_dataset_info(run_folder, samples_dataset.classes_labels, train_ds.dataset_split,
                                   val_ds.dataset_split, file_suffix, wandb_run)

    # ==== INIT MODEL
    print("Init model...")
    # Ensure deterministic behavior
    make_new_seed(cfg['train_params'].get('new_seed', ''))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    orig_model = model = build_model(cfg, samples_dataset.classes_num, fold_index)  # .to(device)
    if cfg['train_params'].get('multi_gpu', False) and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)
    print("Model was loaded")

    class_map = {i: v for i, v in enumerate(samples_dataset.classes_labels)}
    optimizer, criterion_list = build_optimizer(cfg, model, class_map, samples_dataset.classes_num)

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
            torchsummary.summary(orig_model, tuple(input_shape), cfg['train_data_loader']['batch_size'], file=to_file)
            print(orig_model, file=to_file)

        _print(f)
        if is_debug_mode:
            _print(None)
    wandb_store_file(wandb_run, model_info_path)

    # wandb_run.watch(model, criterion_list[0], log=None, log_freq=200)

    # ==== TRAIN LOOP
    if not is_find_lr_mode:
        best_metrics = train_loop(cfg, model, device, optimizer, scaler, criterion_list, train_dataloader,
                                  val_dataloader, run_folder, csv_path, snapshot_path, orig_stderr,
                                  fold_index, folds_count)
        if queue is not None:
            queue.put((run_folder, best_metrics, cfg))
    else:
        find_lr_loop(cfg, model, device, optimizer, scaler, criterion_list, train_dataloader, run_folder, orig_stderr,
                     wandb_run)

    wandb_run.finish()

    # ===== Close Loggers
    sys.stdout.close()
    sys.stderr.close()
    sys.stdout, sys.stderr = orig_stdout, orig_stderr

    # ===== Submission
    if not is_find_lr_mode and fold_index is None:
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--find_lr', '-flr', help="Find LR", action='store_true')

    args = parser.parse_args()

    main(args.find_lr)
