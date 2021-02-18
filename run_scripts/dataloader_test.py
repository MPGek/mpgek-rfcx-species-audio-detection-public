import os
import platform
import pandas as pd
from typing import Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.augementations import init_mixers
from model.datasets import AudioDataset, SampleDataset, DatasetSampler

cfg = {
    'audio_params': {
        'sampling_rate': 48000,
        # 'hop_length': 375 * 2,  # 375 * 2 - 64 pixels per second
        # 'hop_length': 256,  # 256 - 187.5 pixels per second
        # 'hop_length': 375,  # 375 - 128 pixels per second
        # 'fmin': 20, 'fmax': 20000,
        'fmin': 50, 'fmax': 15000,
        # 'fmin': 3500, 'fmax': 6000, # species 8
        # 'n_mels': 128, 'n_fft_mels_ratio': 20,
        # 'n_mels': 224, 'n_fft_mels_ratio': 20,
        # 'n_mels': 256, 'n_fft': 4096, 'ftt_win_length': 1536, 'hop_length': 256,  # 256 - 187.5 pixels per second
        # 'n_mels': 384, 'n_fft': 4096, 'ftt_win_length': 1536, 'hop_length': 375,  # 375 - 128 pixels per second
        # 'n_mels': 380, 'n_fft': 4096, 'ftt_win_length': 1536, 'hop_length': 400,  # 400 - 120 pixels per second
        # 'n_mels': 256, 'n_fft': 4096, 'ftt_win_length': 1280, 'hop_length': 400,  # 500 - 96 pixels per second
        # 'n_mels': 512, 'n_fft': 6*1024, 'ftt_win_length': 1536, 'hop_length': 375,  # 375 - 128 pixels per second
        'n_mels': 380, 'n_fft': 4096, 'ftt_win_length': 1536, 'hop_length': 400,  # 400 - 120 pixels per second
        # 'n_mels': 380, 'n_fft': 4096, 'ftt_win_length': 1536, 'hop_length': 300,  # 400 - 120 pixels per second
        # 'n_mels': 380, 'n_fft': 4096, 'ftt_win_length': 1280, 'hop_length': 250,  # 250 - 192 pixels per second
    },
    'train_data_loader': {
        'key': 'train',
        'batch_size': 10,
        'shuffle': False,
        'num_workers': 0,
        # 'num_workers': 8,
        'wave_aug': 0,
        'mel_aug': 15,
        'mixup': 8,
        # 'mel_aug': 9,
        'cut_samples': True, 'cut_size': 10, 'random_crop': 6, 'fix_cut_targets': True,
        # 'cut_samples': True, 'cut_size': 3, 'random_crop': 2, 'fix_cut_targets': True,
        # 'cut_samples': True, 'cut_size': 6, 'crops_offset': 2, 'fix_cut_targets': True,
        # 'cut_samples': True, 'cut_size': 8, 'random_crop': 5, 'fix_cut_targets': True,
        'use_fp': True,
        'sample_val_fp': True,
        # 'remove_duplicates': True,
        # 'filter_species': [17],
        # 'filter_species': [0, 1, 10, 11, 13, 14, 16, 18, 19, 2, 21, 22, 3, 4, 6],
        # 'filter_species': [13, 21],
        # 'filter_species': [23],
        # 'filter_species': [12, 15, 17, 20, 23, 5, 7, 8, 9],
    },
    'test_data_loader': {
        'key': 'test',
        'batch_size': 2,
        'shuffle': False,
        'num_workers': 4,
    }
}


# --------------------------------
# 0 Proc Batch 16
#
# --------------------------------



def main():
    # set env variable for data
    os.environ['DATA_FOLDER'] = "../data"
    # get config

    debug_img = False
    debug_img = True
    is_notebook = platform.node() == 'nb-162'

    # ===== INIT DATASET
    train_cfg = cfg["train_data_loader"]
    init_mixers(cfg['audio_params']['fmin'], cfg['audio_params']['fmax'], cfg['audio_params']['n_mels'])

    samples_dataset = SampleDataset(cfg['audio_params'], train_cfg, folds=1, test_size=0.9, disable_cache=is_notebook)
    split = samples_dataset.train_split[0] # type:pd.DataFrame
    # split = split[(split[SampleDataset.k_recording_id] == 'cbc07e2ec')]
    audio_dataset = AudioDataset(cfg, samples_dataset, split, train_cfg, debug_img=debug_img)

    generator = torch.Generator()
    generator.manual_seed(1327)
    # sampler = DatasetSampler(audio_dataset, generator=generator)
    sampler = None
    train_dataloader = DataLoader(audio_dataset, batch_size=train_cfg['batch_size'], sampler=sampler,
                                  num_workers=train_cfg['num_workers'], persistent_workers=train_cfg['num_workers'] > 0)
    print('train_dataset len:', len(audio_dataset))
    print(audio_dataset)
    # print('sampler len:', len(sampler))
    # print(sampler)
    print('train_dataloader len:', len(train_dataloader))
    print(train_dataloader)

    # ==== TRAIN LOOP
    # tr_it = iter(train_dataloader)
    # dataset_len = cfg["train_params"]["max_num_steps"] * train_cfg["batch_size"]
    dataset_len = len(audio_dataset)
    repeats = 2000
    progress_bar = None  # type:Optional[tqdm]
    losses_train = []
    for i in range(repeats):
        # try:
        #     data = next(tr_it)
        # except StopIteration:
        #     tr_it = iter(train_dataloader)
        #     data = next(tr_it)
        #
        # if debug_img:
        #     pass

        samples = 0
        for data_img, data_target, data_recotd in train_dataloader:
            samples += len(data_img)

            if progress_bar is None:
                progress_bar = tqdm(total=dataset_len * repeats, smoothing=0.05)
            progress_bar.update(len(data_img))
            progress_bar.set_postfix({'Epoch': f'{samples}/{dataset_len}', 'Repeat': f'{i + 1}/{repeats}'})

    progress_bar.close()


if __name__ == '__main__':
    main()
