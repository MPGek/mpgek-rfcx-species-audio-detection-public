import multiprocessing
import os
import shutil
import sys
from multiprocessing import Pool
from pathlib import Path
from typing import List
from typing import Optional

import cv2
import librosa
import numpy as np
import pandas as pd
import torch
import zarr
from sklearn import model_selection
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from model import audio_utils, transforms
from model.audio_utils import read_as_melspectrogram
from model.augementations import wave_augmentations, mel_augmentations, mixers


class SampleDataset:
    k_recording_id = 'recording_id'
    k_species_id = 'species_id'
    k_songtype_id = 'songtype_id'
    k_t_min = 't_min'
    k_f_min = 'f_min'
    k_t_max = 't_max'
    k_f_max = 'f_max'
    k_key = 'key'
    k_duration = 'duration'
    k_species_id_cls = 'species_id_class'
    k_key_cls = 'key_class'
    k_is_tp = 'is_tp'
    k_cut_place = 'cut_place'

    def __init__(self, audio_params, train_config: dict, folds: int = 1, test_size: float = 0.3,
                 disable_cache=False, is_test_mode=False, classes_num=None, kfold_statified_shuffle_splits=False):
        self.folder_path = os.path.join(os.environ['DATA_FOLDER'], train_config['key'])
        self.use_fp = train_config.get('use_fp', False)
        self.sample_val_fp = train_config.get('sample_val_fp', False)
        self.is_test_mode = is_test_mode
        remove_duplicates = train_config.get('remove_duplicates', False)
        filter_species = train_config.get('filter_species', [])
        seed = train_config.get('seed', 1327)

        if not is_test_mode:
            data_tp = self.load_data('tp', remove_duplicates, filter_species)
            data_fp = self.load_data('fp', remove_duplicates, filter_species)
            data_tp[self.k_is_tp] = True
            data_fp[self.k_is_tp] = False

            self.species_id_set = sorted(set(data_tp[self.k_species_id]))
            self.key_set = sorted(set(data_tp[self.k_key]), key=lambda x: x if len(x) > 3 else '0' + x)

            self.classes_labels = self.species_id_set
            self.classes_num = len(self.classes_labels)
            self.classes_column_name = self.k_species_id_cls

            self.set_classes_indexes(data_tp)
            self.set_classes_indexes(data_fp)

            # if self.use_fp:
            #     self.classes_num *= 2

            data_fp[self.k_species_id_cls] = -1 - data_fp[self.k_species_id_cls]
            data_fp[self.k_key_cls] = -1 - data_fp[self.k_key_cls]

            self.data_tp = data_tp
            self.data_fp = data_fp
            self.data_all = pd.concat([data_tp, data_fp]).reset_index()
        else:
            self.data_all = self.collect_test_data()
            self.data_tp = self.data_all
            self.classes_num = classes_num

        splits = self.split_train_test(folds, test_size, seed, remove_duplicates, kfold_statified_shuffle_splits)
        self.train_split, self.test_split = splits  # type: List[pd.DataFrame], List[pd.DataFrame]

        if not disable_cache:
            self.zarr_cache = self.generate_mel_cache(audio_params)
        else:
            self.zarr_cache = None  # type:Optional[zarr.core.Array]

    def collect_test_data(self):
        files = list(Path(self.folder_path).glob('*.flac'))  # type:List[Path]
        file_names = sorted([f.stem for f in files])

        data = pd.DataFrame({self.k_recording_id: file_names})
        return data

    def load_data(self, suffix, remove_duplicates, filter_species: List):
        data = pd.read_csv(self.folder_path + f'_{suffix}.csv')

        if remove_duplicates:
            data = data.drop_duplicates(self.k_recording_id, keep=False)

        data[self.k_key] = data[self.k_species_id].astype(str) + '|' + data[self.k_songtype_id].astype(str)
        data[self.k_duration] = data[self.k_t_max] - data[self.k_t_min]

        if filter_species:
            data = data.loc[data[self.k_species_id].isin(filter_species)]

        return data

    def set_classes_indexes(self, data: pd.DataFrame):
        data[self.k_species_id_cls] = data.apply(lambda x: self.species_id_set.index(x[self.k_species_id]), axis=1)
        data[self.k_key_cls] = data.apply(lambda x: self.key_set.index(x[self.k_key]), axis=1)

    def generate_mel_cache(self, audio_params):
        record_ids_list = list(sorted(set(self.data_all[SampleDataset.k_recording_id])))
        file_path = self.get_file_path(record_ids_list[0])
        single_item = read_as_melspectrogram(audio_params, file_path, None)
        dataset_shape = (len(record_ids_list),) + single_item.shape
        chunks = (1,) + single_item.shape

        zarr_group_name = self.gen_group_name(audio_params)
        zarr_root = zarr.open(self.folder_path + '_cache.zarr', mode='a')
        zarr_mel = zarr.convenience.open(str(Path(zarr_root.store.path).joinpath(zarr_group_name)), mode='a')
        stored_records = zarr_mel.attrs.get('record_ids_list', [])

        if stored_records != record_ids_list:
            shutil.rmtree(Path(zarr_root.store.path).joinpath(zarr_group_name))
            zarr_mel = zarr_root.create_dataset(zarr_group_name, shape=dataset_shape, dtype=np.float16, chunks=chunks)

            record_path_list = list(map(self.get_file_path, record_ids_list))
            map_iterator = zip(record_path_list, (audio_params,) * len(record_path_list))
            with Pool(multiprocessing.cpu_count() // 2) as pool:
                with tqdm(desc=f"Preparing mel cache [{zarr_group_name}]", total=len(record_ids_list)) as t:
                    for record_id, record_mel in pool.imap(get_mel, map_iterator):  # type: str, np.ndarray
                        idx = record_ids_list.index(record_id)
                        zarr_mel[idx, ...] = record_mel.astype(np.float16)
                        t.update()

            zarr_mel.attrs['record_ids_list'] = record_ids_list

        return zarr.convenience.open(str(Path(zarr_root.store.path).joinpath(zarr_group_name)), mode='r')

    @staticmethod
    def gen_group_name(audio_param: dict):
        sampling_rate = audio_param['sampling_rate']
        hop_length = audio_param['hop_length']
        fmin = audio_param['fmin']
        fmax = audio_param['fmax']
        n_mels = audio_param['n_mels']
        n_fft_mels_ratio = audio_param.get('n_fft_mels_ratio', 0)
        n_fft = audio_param.get('n_fft', 0)
        ftt_win_length = audio_param.get('ftt_win_length', None)

        if n_fft_mels_ratio > 0:
            n_fft = n_mels * n_fft_mels_ratio

        res = f'sr_{sampling_rate}_hop_{hop_length}_f_[{fmin}-{fmax}]_mel_{n_mels}_fft_{n_fft}_ftt-win_{ftt_win_length}'
        return res

    @staticmethod
    def split_train_test_tp_or_fp(data: pd.DataFrame, folds, test_size, seed, kfold_statified_shuffle_splits,
                                  classes_column_name):
        grouped = data[[SampleDataset.k_recording_id, classes_column_name]].groupby(SampleDataset.k_recording_id).min()
        classes = grouped[classes_column_name].tolist()

        train = []
        test = []

        if folds < 2:
            res = model_selection.train_test_split(grouped, test_size=test_size, random_state=seed, stratify=classes)
            train.append(res[0].index)
            test.append(res[1].index)
        else:
            if not kfold_statified_shuffle_splits:
                folds_generator = model_selection.StratifiedKFold(folds, shuffle=True, random_state=seed)
            else:
                folds_generator = model_selection.StratifiedShuffleSplit(folds, random_state=seed, test_size=test_size)

            for train_part, test_part in folds_generator.split(grouped, classes):
                train.append(grouped.iloc[train_part, :].index)
                test.append(grouped.iloc[test_part, :].index)
                # train.append(self.data.iloc[train_part, :])
                # test.append(self.data.iloc[test_part, :])

        for split_id in range(len(train)):
            train_part, test_part = train[split_id], test[split_id]
            train_part = data.loc[data[SampleDataset.k_recording_id].isin(train_part)]
            test_part = data.loc[data[SampleDataset.k_recording_id].isin(test_part)]

            train[split_id], test[split_id] = train_part, test_part

        return train, test

    def split_train_test_new(self, folds, test_size, seed, remove_duplicates, kfold_statified_shuffle_splits):
        data = self.data_tp if not self.use_fp else self.data_all

        if folds < 0:
            return [data], [[]]

        train, test = SampleDataset.split_train_test_tp_or_fp(self.data_tp, folds, test_size, seed,
                                                              kfold_statified_shuffle_splits, self.classes_column_name)

        if self.use_fp:
            train_fp, test_fp = SampleDataset.split_train_test_tp_or_fp(self.data_fp, folds, test_size, seed,
                                                                        kfold_statified_shuffle_splits,
                                                                        self.classes_column_name)

            for split_id in range(len(train)):
                train_part, test_part = train[split_id], test[split_id]
                train_fp_part, test_fp_part = train_fp[split_id], test_fp[split_id]

                train_fp_part, test_fp_part = self.fix_fp_train_test_parts(train_part, test_part,
                                                                           train_fp_part, test_fp_part)

                train_part = pd.concat([train_part, train_fp_part])
                test_part = pd.concat([test_part, test_fp_part])

                if self.sample_val_fp:
                    train_part, test_part = self.perform_sampling_for_val_fp(train_part, test_part)

                # if remove_duplicates:
                #     duplicated_records = train_part[self.k_recording_id].duplicated(keep=False)
                #     test_part = pd.concat([test_part, train_part.loc[duplicated_records, :]])
                #     train_part = train_part.drop_duplicates(self.k_recording_id, keep=False)

                train[split_id], test[split_id] = train_part, test_part

        return train, test

    def split_train_test(self, folds, test_size, seed, remove_duplicates, kfold_statified_shuffle_splits):
        data = self.data_tp if not self.use_fp else self.data_all

        if folds < 0:
            return [data], [[]]

        grouped = data[[self.k_recording_id, self.classes_column_name]].groupby(self.k_recording_id).min()
        classes = grouped[self.classes_column_name].tolist()

        train = []
        test = []

        if folds < 2:
            res = model_selection.train_test_split(grouped, test_size=test_size, random_state=seed, stratify=classes)
            train.append(res[0].index)
            test.append(res[1].index)
        else:
            if not kfold_statified_shuffle_splits:
                folds_generator = model_selection.StratifiedKFold(folds, shuffle=True, random_state=seed)
            else:
                folds_generator = model_selection.StratifiedShuffleSplit(folds, random_state=seed, test_size=test_size)

            for train_part, test_part in folds_generator.split(grouped, classes):
                train.append(grouped.iloc[train_part, :].index)
                test.append(grouped.iloc[test_part, :].index)
                # train.append(self.data.iloc[train_part, :])
                # test.append(self.data.iloc[test_part, :])

        for split_id in range(len(train)):
            train_part, test_part = train[split_id], test[split_id]
            train_part = data.loc[data[self.k_recording_id].isin(train_part)]
            test_part = data.loc[data[self.k_recording_id].isin(test_part)]

            if self.sample_val_fp:
                train_part, test_part = self.perform_sampling_for_val_fp(train_part, test_part)

            # if remove_duplicates:
            #     duplicated_records = train_part[self.k_recording_id].duplicated(keep=False)
            #     test_part = pd.concat([test_part, train_part.loc[duplicated_records, :]])
            #     train_part = train_part.drop_duplicates(self.k_recording_id, keep=False)

            train[split_id], test[split_id] = train_part, test_part

        return train, test

    def get_file_path(self, record_id):
        # record_id = row[SampleDataset.k_recording_id]
        file_path = os.path.join(self.folder_path, record_id + '.flac')
        return file_path

    def get_class_one_hot(self, row):
        class_index = row[self.classes_column_name]
        res = np.zeros(self.classes_num, dtype=np.float32)
        res[class_index] = 1
        return res

    def get_classes_vector(self, rows: pd.DataFrame):
        classes_num = self.classes_num * 2 if self.use_fp else self.classes_num
        res = np.zeros(classes_num, dtype=np.float32)

        if self.is_test_mode:
            res[:] = 0.5
            return res

        if rows is None:
            return res

        class_indices = rows[self.classes_column_name]
        for class_index in class_indices:
            if class_index >= 0:
                res[class_index] = 1
            else:
                class_tp = -1 - class_index
                class_fp = class_tp + classes_num // 2
                # res[[class_tp, class_fp]] = 1
                res[class_fp] = 1

        return res

    def perform_sampling_for_val_fp(self, train_part: pd.DataFrame, test_part: pd.DataFrame):
        tp_data = test_part[test_part[self.classes_column_name] >= 0]
        fp_data = test_part[test_part[self.classes_column_name] < 0]

        tp_classes_count = tp_data.groupby(self.classes_column_name).count()
        tp_classes_count = dict(tp_classes_count[SampleDataset.k_recording_id].reset_index().values.tolist())

        fp_train_rows, fp_test_rows = [], []
        for idx, row in fp_data.iterrows():  # type: int, pd.Series
            tp_class = row.get(self.classes_column_name)
            tp_class = -1 - tp_class

            list_to_add = fp_test_rows if tp_classes_count[tp_class] > 0 else fp_train_rows
            list_to_add.append(idx)
            tp_classes_count[tp_class] = tp_classes_count[tp_class] - 1

        train_part = pd.concat([train_part, fp_data.loc[fp_train_rows]])
        test_part = pd.concat([tp_data, fp_data.loc[fp_test_rows]])

        return train_part, test_part

    def fix_fp_train_test_parts(self, train_part: pd.DataFrame, test_part: pd.DataFrame,
                                train_fp_part: pd.DataFrame, test_fp_part: pd.DataFrame):
        train_records = set(train_part[SampleDataset.k_recording_id].values.tolist())
        test_records = set(test_part[SampleDataset.k_recording_id].values.tolist())

        wrong_train_fp = train_fp_part[SampleDataset.k_recording_id].isin(test_records)
        wrong_test_fp = test_fp_part[SampleDataset.k_recording_id].isin(train_records)

        good_train_fp = train_fp_part[-wrong_train_fp]
        wrong_train_fp = train_fp_part[wrong_train_fp]
        good_test_fp = test_fp_part[-wrong_test_fp]
        wrong_test_fp = test_fp_part[wrong_test_fp]

        train_fp_part = pd.concat([good_train_fp, wrong_test_fp])
        test_fp_part = pd.concat([good_test_fp, wrong_train_fp])

        return train_fp_part, test_fp_part


def get_mel(data):
    record_path, audio_params = data
    return Path(record_path).stem, read_as_melspectrogram(audio_params, record_path, None)


class AudioDataset(Dataset):
    def __init__(self, cfg: dict, samples_dataset: SampleDataset, dataset_split: pd.DataFrame, train_cfg: dict,
                 debug_img=False, verbose=1, sort_t_min=False):
        self.cfg = cfg
        self.sampling_rate = cfg['audio_params']['sampling_rate']
        self.hop_length = cfg['audio_params']['hop_length']
        self.fmin = cfg['audio_params']['fmin']
        self.fmax = cfg['audio_params']['fmax']
        self.n_mels = cfg['audio_params']['n_mels']

        self.samples_dataset = samples_dataset
        self.dataset_split = dataset_split.reset_index()
        self.dataset_grouped = dataset_split[[SampleDataset.k_recording_id]].groupby(SampleDataset.k_recording_id).min()

        self.debug_img = debug_img
        self.verbose = verbose
        self.fix_cut_targets = train_cfg.get('fix_cut_targets', False)
        self.cut_samples = train_cfg.get('cut_samples', False)
        self.cut_size = train_cfg.get('cut_size', 0)
        random_crop = train_cfg.get('random_crop', 0)
        self.cut_size = max(self.cut_size, random_crop + 1)  # cut_size should be bigger than random_crop
        if random_crop > 0:
            self.random_crop = transforms.RandomCrop(int(random_crop * self.sampling_rate / self.hop_length))
        else:
            self.random_crop = None

        self.crops_offset = train_cfg.get('crops_offset', 0)
        if self.cut_samples and self.cut_size > 0 and self.crops_offset > 0:
            self.dataset_split = self.split_dataset_by_crops(dataset_split)

        if sort_t_min:
            self.dataset_split = self.dataset_split.sort_values(by=SampleDataset.k_t_min)
            self.dataset_grouped = dataset_split[[SampleDataset.k_recording_id, SampleDataset.k_t_min]].groupby(
                SampleDataset.k_recording_id).min()
            self.dataset_grouped = self.dataset_grouped.sort_values(by=SampleDataset.k_t_min)

        self.wave_aug = None
        self.mel_aug = None
        self.mixer = None

        wave_aug_index = train_cfg.get('wave_aug', None)
        if wave_aug_index is not None:
            self.wave_aug = wave_augmentations[wave_aug_index]

        mel_aug_index = train_cfg.get('mel_aug', None)
        if mel_aug_index is not None:
            self.mel_aug = mel_augmentations[mel_aug_index]

        mixer_index = train_cfg.get('mixup', None)
        if mixer_index is not None:
            self.mixer = mixers[mixer_index]

    def __len__(self):
        if not self.cut_samples:
            return len(self.dataset_grouped)
        else:
            return len(self.dataset_split)

    def get_record_id_with_rows(self, item):
        if not self.cut_samples:
            record_id = self.dataset_grouped.iloc[[item], :].index[0]
            rows = self.dataset_split[self.dataset_split[SampleDataset.k_recording_id] == record_id]
        else:
            rows = self.dataset_split.iloc[[item], :]
            record_id = rows[SampleDataset.k_recording_id].values[0]

        return record_id, rows

    def filter_samples(self, below_freq=None, above_freq=None):
        if below_freq is not None:
            filtered = self.dataset_split[self.dataset_split[SampleDataset.k_f_max] <= below_freq]
        else:
            filtered = self.dataset_split[self.dataset_split[SampleDataset.k_f_min] >= above_freq]

        res = filtered.index.tolist()
        return res

    def filter_samples_by_fp(self, target_indexes, is_fp):
        new_result = []

        for target_idx in target_indexes:
            is_target_fp = not self.dataset_split[SampleDataset.k_is_tp].values[target_idx]
            if is_target_fp == is_fp:
                new_result.append(target_idx)

        return new_result

    def get_sample(self, item):
        record_id, rows = self.get_record_id_with_rows(item)

        class_vector = self.samples_dataset.get_classes_vector(rows)

        if self.samples_dataset.zarr_cache is not None:
            record_ids_list = self.samples_dataset.zarr_cache.attrs['record_ids_list']
            record_index = record_ids_list.index(record_id)
            mel = self.samples_dataset.zarr_cache[record_index, ...].astype(np.float32)  # type:np.ndarray
        else:
            file_path = self.samples_dataset.get_file_path(record_id)
            mel = read_as_melspectrogram(self.cfg['audio_params'], file_path, None)  # self.wave_aug)

        t_min, t_max = 0, mel.shape[-1]
        if self.cut_samples:
            mel, t_min, t_max = self.cut_mel(mel, rows)
        if self.random_crop is not None:
            mel, offset = self.random_crop(mel)
            # print('original cut: [{}, {}], offset {}'.format(
            #     *[x * self.hop_length / self.sampling_rate for x in [t_min, t_max, offset]]))
            t_min = t_min + offset
            t_max = t_min + mel.shape[-1]

        visible_rows = rows
        if self.fix_cut_targets:
            visible_rows, class_vector = self.get_actual_rows_and_targets_vector(record_id, t_min, t_max)

        orig_mel = mel
        if self.mel_aug is not None:
            mel = self.augment_mel(mel)

        return mel, class_vector, orig_mel, record_id, rows, visible_rows, t_min, t_max

    def __getitem__(self, item):
        mel, class_vector, orig_mel, record_id, all_rows, visible_rows, t_min, t_max = self.get_sample(item)

        if self.mixer:
            mel, class_vector, visible_rows = self.mixer(self, mel, class_vector, visible_rows)

        if self.debug_img:
            print('Target:', class_vector, 'record_id:', record_id, 't_min:', t_min, 't_max:', t_max)
            self.show_debug_img(mel, orig_mel, visible_rows, t_min, t_max)

        if len(mel.shape) < 3:
            mel = mel[np.newaxis, :, :]
        return mel, class_vector, record_id

    def get_actual_rows_and_targets_vector(self, record_id, t_min, t_max):
        rows = self.dataset_split.loc[self.dataset_split[SampleDataset.k_recording_id] == record_id]
        if not self.samples_dataset.is_test_mode:
            visible_rows = self.filter_visible_rows(rows, t_min, t_max)
        else:
            visible_rows = rows
        class_vector = self.samples_dataset.get_classes_vector(visible_rows)

        return visible_rows, class_vector

    def apply_results_filter(self, record_ids, start, end):
        result = np.zeros((len(record_ids), 1))

        for idx, record_id in enumerate(record_ids):
            visible_rows, class_vector = self.get_actual_rows_and_targets_vector(record_id, start, end)
            if np.count_nonzero(class_vector) > 0:
                result[idx, :] = 1

        return result

    def split_dataset_by_crops(self, dataset_split: pd.DataFrame):
        rows = []
        rows_indexes_list = []
        rows_set = set()

        dataset_split = dataset_split.sort_values([SampleDataset.k_recording_id, SampleDataset.k_t_min])
        for row_index, row in tqdm(dataset_split.iterrows(), desc='Cutting samples...',
                                   total=len(dataset_split)):
            record_id = row[SampleDataset.k_recording_id]
            row_start = row[SampleDataset.k_t_min]
            row_end = row[SampleDataset.k_t_max]
            row_start = int(max(0, row_start - self.cut_size) // self.crops_offset)
            row_end = int(min(row_end + self.crops_offset, 60 - self.cut_size) // self.crops_offset)

            is_row_added = False
            for i in range(row_start, row_end + 1):
                crop_start = int(i * self.crops_offset * self.sampling_rate / self.hop_length)
                crop_end = crop_start + int(self.cut_size * self.sampling_rate / self.hop_length)
                visible_rows, class_vector = self.get_actual_rows_and_targets_vector(record_id, crop_start, crop_end)

                if np.count_nonzero(class_vector) > 0:
                    row_cut_id = f'{record_id}_{crop_start}'
                    if row_cut_id in rows_set:
                        if is_row_added:
                            continue
                        else:
                            # we need to add every original row at leas one time even if there are already present
                            # row with the same cut_place
                            index = rows_indexes_list.index(row_cut_id)
                            del rows_indexes_list[index]
                            del rows[index]

                    is_row_added = True
                    rows_set.add(row_cut_id)
                    rows_indexes_list.append(row_cut_id)
                    new_row = row.copy(deep=False)
                    new_row[SampleDataset.k_cut_place] = crop_start
                    rows.append(new_row)

        data = pd.concat(rows, axis=1).transpose()
        data = data.sort_values([SampleDataset.k_recording_id, SampleDataset.k_cut_place]).reset_index()
        return data

    def show_debug_img(self, mel, orig_mel, rows: pd.DataFrame, t_min, t_max):
        mel_frequencies = librosa.mel_frequencies(self.n_mels, self.fmin, self.fmax)

        def find_nearest_idx(array, value):
            array = np.asarray(array)
            idx = (np.abs(array - value)).argmin()
            return idx

        def _draw_rect(mel, name, row_t_min, row_t_max, f_min, f_max):
            mel_min = find_nearest_idx(mel_frequencies, f_min)
            mel_max = find_nearest_idx(mel_frequencies, f_max)
            mel_min = self.n_mels - mel_min
            mel_max = self.n_mels - mel_max

            # x1 = row_t_min - t_min
            # x2 = min(row_t_max, t_max) - t_min
            x1 = row_t_min
            x2 = row_t_max

            cv2.rectangle(mel, (x1, mel_max), (x2, mel_min), (0, 255, 0))
            cv2.putText(mel, name, (x1, mel_max + 16 - 20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))

        orig_mel = orig_mel[::-1, :]
        mel_norm = np.zeros_like(orig_mel)
        cv2.normalize(orig_mel, mel_norm, 0, 255, cv2.NORM_MINMAX)

        mel_norm = cv2.applyColorMap(mel_norm.astype(np.uint8), cv2.COLORMAP_MAGMA)
        mel = cv2.applyColorMap((mel[::-1, :] * 255).astype(np.uint8), cv2.COLORMAP_MAGMA)
        # mel = cv2.cvtColor(mel[::-1, :], cv2.COLOR_GRAY2BGR)

        orig_mel = cv2.cvtColor(orig_mel, cv2.COLOR_GRAY2BGR)
        # mel_norm = cv2.cvtColor(mel_norm, cv2.COLOR_GRAY2BGR)
        # mel = cv2.cvtColor(mel[::-1, :], cv2.COLOR_GRAY2BGR)

        draw = False
        first_row_id = None
        for row_id, row in rows.iterrows():  # type:pd.DataFrame
            first_row_id = first_row_id or row[SampleDataset.k_recording_id]

            row_t_min = int(row[SampleDataset.k_t_min] * self.sampling_rate / self.hop_length)
            row_t_max = int(row[SampleDataset.k_t_max] * self.sampling_rate / self.hop_length)
            # if row_t_min <= t_max and row_t_max >= t_min:
            if 1:
                draw = True
                name = ('tp|' if row[SampleDataset.k_is_tp] else 'fp|') + row[SampleDataset.k_key]
                # row_t_min, row_t_max = max(row_t_min, t_min), min(row_t_max, t_max)
                rect_info = name, row_t_min, row_t_max, row[SampleDataset.k_f_min], row[SampleDataset.k_f_max]
                if first_row_id == row[SampleDataset.k_recording_id]:
                    _draw_rect(orig_mel, *rect_info)
                    _draw_rect(mel_norm, *rect_info)
                _draw_rect(mel, *rect_info)

        if not draw:
            print('Missing draw!')

        # cv2.imshow('orig_mel', orig_mel)
        cv2.imshow('mel_norm', mel_norm)
        cv2.imshow('mel', mel)
        # cv2.moveWindow('orig_mel', 0, 0)
        cv2.moveWindow('mel_norm', 0, (orig_mel.shape[0] + 32) * 0)
        cv2.moveWindow('mel', 0, (orig_mel.shape[0] + 32) * 1)
        cv2.waitKey(0)
        # cv2.destroyAllWindows()

    # @perf_timing()
    def augment_mel(self, mel):
        augmented_mel = self.mel_aug(mel)

        is_debug = False
        # is_debug = True
        if is_debug:
            audio_utils.show_melspectrogram(self.cfg['audio_params'], mel)
            if len(augmented_mel.shape) == 2:
                audio_utils.show_melspectrogram(self.cfg['audio_params'], augmented_mel)
            else:
                for item in augmented_mel:
                    audio_utils.show_melspectrogram(self.cfg['audio_params'], item)

        return augmented_mel

    def cut_mel(self, mel, rows: pd.DataFrame):
        cut_place = None
        if self.crops_offset <= 0:
            row_values = rows[[SampleDataset.k_t_min, SampleDataset.k_t_max]].values
            t_min, t_max = row_values[np.random.randint(0, len(row_values))]
        else:
            row_values = rows[[SampleDataset.k_t_min, SampleDataset.k_t_max, SampleDataset.k_cut_place]].values
            t_min, t_max, cut_place = row_values[np.random.randint(0, len(row_values))]

        t_min = int(t_min * self.sampling_rate // self.hop_length)
        t_max = int(t_max * self.sampling_rate // self.hop_length)
        t_duration = t_max - t_min

        cut_size = int(self.cut_size * self.sampling_rate // self.hop_length)
        if cut_place is None:
            cut_padding = (cut_size - t_duration) // 2
            cut_min = max(0, min(t_min - cut_padding, min(t_max + cut_padding, mel.shape[-1]) - cut_size))
        else:
            cut_min = cut_place
        cut_max = cut_min + cut_size

        one_second = int(self.sampling_rate // self.hop_length)
        if self.random_crop:
            max_padding = self.random_crop.size - one_second - 1  # minus 1 sample to avoid rounding error sec->sample
            if t_min - cut_min > max_padding:
                cut_min = min(t_min - max_padding, cut_max - self.random_crop.size)
            elif cut_max - t_max > max_padding:
                cut_max = max(t_max + max_padding, cut_min + self.random_crop.size)

        return mel[:, cut_min:cut_max], cut_min, cut_max

    def filter_visible_rows(self, rows, t_min_samples, t_max_sample):
        t_min_seconds = t_min_samples * self.hop_length / self.sampling_rate
        t_max_seconds = t_max_sample * self.hop_length / self.sampling_rate

        visible_rows = []
        possible_rows = []

        for idx, row in rows.iterrows():
            row_t_min = row[SampleDataset.k_t_min]
            row_t_max = row[SampleDataset.k_t_max]
            row_t_duration = row_t_max - row_t_min
            intersection_threshold = min(1, row_t_duration * 0.5)

            intersect_t_min = max(row_t_min, t_min_seconds)
            intersect_t_max = min(row_t_max, t_max_seconds)
            intersect_t_duration = intersect_t_max - intersect_t_min

            if intersect_t_duration > 0:
                row[SampleDataset.k_t_min] = max(0, row[SampleDataset.k_t_min] - t_min_seconds)
                row[SampleDataset.k_t_max] = row[SampleDataset.k_t_max] - t_min_seconds

                if intersect_t_duration > intersection_threshold:
                    visible_rows.append(row)
                else:
                    possible_rows.append((intersect_t_duration, intersection_threshold, row))

        if len(visible_rows) == 0:
            if self.verbose > 1:
                for intersect_t_duration, intersection_threshold, row in possible_rows:
                    print(f'The intersect in file "{row[SampleDataset.k_recording_id]}" '
                          f'[{row[SampleDataset.k_t_min] + t_min_seconds}, '
                          f'{row[SampleDataset.k_t_max] + t_min_seconds}] '
                          f'too low - {intersect_t_duration} < {intersection_threshold}, '
                          f'for the cut [{t_min_seconds}, {t_max_seconds}]', file=sys.stderr)

            # if len(possible_rows) > 0:
            #     possible_rows = sorted(possible_rows, key=lambda x: x[0], reverse=True)
            #     visible_rows.append(possible_rows[0][2])  # add the row with the biggest intersection
            # else:
            #     return None

            return None

        return pd.concat(visible_rows, axis=1).transpose()


def make_dataloader(samples_dataset: SampleDataset, split_data: pd.DataFrame, cfg, data_name, sort_t_min=False,
                    seed=1327):
    verbose = 0 if data_name in ('test', 'submit') else 1

    data_cfg = cfg[data_name + '_data_loader']

    audio_dataset = AudioDataset(cfg, samples_dataset, split_data, data_cfg, verbose=verbose, sort_t_min=sort_t_min)
    print(data_name, 'dataset len:', len(audio_dataset))

    sampler = None
    shuffle = data_cfg['shuffle']
    tp_fp_sampler = data_cfg.get('tp_fp_sampler', False)
    drop_last = data_name in ('train',)

    if tp_fp_sampler:
        generator = torch.Generator()
        generator.manual_seed(seed)
        sampler = DatasetSampler(audio_dataset, generator=generator)
        shuffle = False

    dataloader = DataLoader(audio_dataset, shuffle=shuffle, sampler=sampler, batch_size=data_cfg['batch_size'],
                            num_workers=data_cfg['num_workers'], persistent_workers=data_cfg['num_workers'] > 0,
                            drop_last=drop_last)

    print(data_name, 'dataloader len:', len(dataloader))

    return dataloader, audio_dataset


class DatasetSampler(torch.utils.data.Sampler[int]):
    def __init__(self, data_source: AudioDataset, generator: Optional[torch.Generator]) -> None:
        super(DatasetSampler, self).__init__(data_source)

        self.generator = generator

        self.classes_column_name = data_source.samples_dataset.classes_column_name
        self.tp_data = data_source.dataset_split[data_source.dataset_split[self.classes_column_name] >= 0]
        self.fp_data = data_source.dataset_split[data_source.dataset_split[self.classes_column_name] < 0]
        self.grouped_data = data_source.dataset_split.groupby(self.classes_column_name)
        self.grouped_data_count = dict(self.grouped_data.count()[SampleDataset.k_key].reset_index().values.tolist())

        classes_num = max(self.tp_data[self.classes_column_name]) + 1
        self.samples_num = len(self.tp_data) * 2

        self.fp_data = [self.fp_data[self.fp_data[self.classes_column_name] == (-1 - class_id)]
                        for class_id in range(classes_num)]
        self.fp_data_iterators = [iter([]) for _ in range(classes_num)]

    def __iter__(self):
        tp_indexes = torch.randperm(len(self.tp_data), generator=self.generator)

        for tp_index in tp_indexes:
            tp_row = self.tp_data.iloc[[int(tp_index)], :]  # type:pd.Series
            yield tp_row.index.values[0]

            tp_class_id = tp_row.get(self.classes_column_name).values[0]

            fp_row = self.get_fp_row(tp_class_id)  # type:Optional[pd.Series]
            if fp_row is not None:
                yield fp_row.index.values[0]

    def __len__(self):
        return self.samples_num

    def get_fp_row(self, tp_class_id):
        try:
            value = next(self.fp_data_iterators[tp_class_id])
        except StopIteration:
            new_iterator = torch.randperm(len(self.fp_data[tp_class_id]), generator=self.generator)
            self.fp_data_iterators[tp_class_id] = iter(new_iterator)
            value = next(self.fp_data_iterators[tp_class_id])

        row = self.fp_data[tp_class_id].iloc[[int(value)], :]
        return row
