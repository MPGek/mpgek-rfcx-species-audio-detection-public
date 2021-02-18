import random

import librosa
import numpy as np
import pandas as pd
import torch


def get_random_sample(dataset, target:np.ndarray, below_freq=None, above_freq=None):
    use_fp = dataset.samples_dataset.use_fp
    is_fp = False
    if use_fp:
        is_fp = np.argmax(target) > target.shape[0] // 2

    if below_freq is None and above_freq is None:
        filtered_samples_indexes = list(range(len(dataset)))
    else:
        filtered_samples_indexes = dataset.filter_samples(below_freq=below_freq, above_freq=above_freq)

    if use_fp:
        filtered_samples_indexes = dataset.filter_samples_by_fp(filtered_samples_indexes, is_fp)

    if not filtered_samples_indexes:
        return None, None, None

    rnd_idx = random.randint(0, len(filtered_samples_indexes) - 1)
    rnd_idx = filtered_samples_indexes[rnd_idx]

    mel, class_vector, orig_mel, record_id, all_rows, visible_rows, t_min, t_max = dataset.get_sample(rnd_idx)
    return mel, class_vector, visible_rows


class AddMixer:
    def __init__(self, alpha_dist='uniform'):
        assert alpha_dist in ['uniform', 'beta']
        self.alpha_dist = alpha_dist

    def sample_alpha(self):
        if self.alpha_dist == 'uniform':
            return random.uniform(0, 0.5)
        elif self.alpha_dist == 'beta':
            return np.random.beta(0.4, 0.4)

    def __call__(self, dataset, image, target, rows):
        rnd_image, rnd_target, rnd_rows = get_random_sample(dataset, target)

        alpha = self.sample_alpha()
        image = (1 - alpha) * image + alpha * rnd_image
        target = (1 - alpha) * target + alpha * rnd_target
        return image, target, rnd_rows


class SigmoidConcatMixer:
    def __init__(self, sigmoid_range=(3, 12)):
        self.sigmoid_range = sigmoid_range

    def sample_mask(self, size):
        x_radius = random.randint(*self.sigmoid_range)

        step = (x_radius * 2) / size[1]
        x = np.arange(-x_radius, x_radius, step=step)
        y = torch.sigmoid(torch.from_numpy(x)).numpy()
        mix_mask = np.tile(y, (size[0], 1))
        # return torch.from_numpy(mix_mask.astype(np.float32))
        return mix_mask.astype(np.float32)

    def __call__(self, dataset, image, target):
        rnd_image, rnd_target, rnd_rows = get_random_sample(dataset, target)

        mix_mask = self.sample_mask(image.shape[-2:])
        rnd_mix_mask = 1 - mix_mask

        image = mix_mask * image + rnd_mix_mask * rnd_image
        target = target + rnd_target
        target = np.clip(target, 0.0, 1.0)
        return image, target, rnd_rows


class SigmoidVerticalConcatMixer:
    def __init__(self, f_min, f_max, n_mels, sigmoid_range=(3, 12), pad_mels=20, band_attention_mode=False):
        self.sigmoid_range = sigmoid_range
        self.f_min = f_min
        self.f_max = f_max
        self.n_mels = n_mels
        self.pad_mels = pad_mels
        self.band_attention_mode = band_attention_mode

        self.mel_freqs = np.asarray(librosa.mel_frequencies(n_mels, f_min, f_max))

    def sample_mask(self, size, f_min_mel, f_max_mel):
        x_radius = random.randint(*self.sigmoid_range)

        x = np.zeros(size[0], dtype=np.float)
        x[:] = -x_radius
        x[f_min_mel:f_max_mel] = x_radius

        x_transfer = np.linspace(-x_radius, x_radius, self.pad_mels)

        transfer_pos_1 = max(0, f_min_mel - self.pad_mels // 2), f_min_mel + self.pad_mels // 2
        transfer_pos_2 = f_max_mel - self.pad_mels // 2 + 1, min(f_max_mel + self.pad_mels // 2 + 1, x.shape[0])
        transfer_size_1 = transfer_pos_1[1] - transfer_pos_1[0]
        transfer_size_2 = transfer_pos_2[1] - transfer_pos_2[0]

        x[transfer_pos_1[0]: transfer_pos_1[1]] = x_transfer[self.pad_mels - transfer_size_1:]
        x[transfer_pos_2[0]: transfer_pos_2[1]] = x_transfer[::-1][:transfer_size_2]

        y = torch.sigmoid(torch.from_numpy(x)).numpy()
        mix_mask = np.tile(y, (size[1], 1)).transpose()
        # return torch.from_numpy(mix_mask.astype(np.float32))
        return mix_mask.astype(np.float32)

    def __call__(self, dataset, image, target, rows):
        f_min, f_max = self.get_min_max_freq(rows)
        f_min, f_max, f_min_mel, f_max_mel = self.pad_freq(f_min, f_max)
        f_middle_mel = (f_min_mel + f_max_mel) // 2

        mix_mask = self.sample_mask(image.shape[-2:], f_min_mel, f_max_mel)

        if self.band_attention_mode:
            image = mix_mask * image
            return image, target, None

        rnd_below_image, rnd_below_target, rnd_below_rows = get_random_sample(dataset, target, below_freq=f_min)
        rnd_above_image, rnd_above_target, rnd_above_rows = get_random_sample(dataset, target, above_freq=f_max)

        rnd_mix_mask = 1 - mix_mask
        rnd_below_mask = np.array(rnd_mix_mask)
        rnd_above_mask = np.array(rnd_mix_mask)
        rnd_below_mask[f_middle_mel:, :] = 0
        rnd_above_mask[:f_middle_mel, :] = 0

        image = mix_mask * image
        rows = []

        if rnd_below_image is not None:
            mix_below_image = rnd_below_mask * rnd_below_image
            image = image + mix_below_image
            target = target + rnd_below_target
            rows.append(rnd_below_rows)

        if rnd_above_image is not None:
            mix_above_image = rnd_above_mask * rnd_above_image
            image = image + mix_above_image
            target = target + rnd_above_target
            rows.append(rnd_above_rows)

        target = np.clip(target, 0.0, 1.0)

        if rows:
            rows = pd.concat(rows)
        else:
            rows = None
        return image, target, rows

    def get_min_max_freq(self, rows: pd.DataFrame):
        from model.datasets import SampleDataset

        f_min, f_max = None, None

        for row_id, row in rows.iterrows():
            if f_min is None or f_min > row[SampleDataset.k_f_min]:
                f_min = row[SampleDataset.k_f_min]
            if f_max is None or f_max < row[SampleDataset.k_f_max]:
                f_max = row[SampleDataset.k_f_max]

        return f_min, f_max

    def find_nearest_mel(self, value):
        idx = (np.abs(self.mel_freqs - value)).argmin()
        return idx

    def pad_freq(self, f_min, f_max):
        f_min_mel = self.find_nearest_mel(f_min)
        f_max_mel = self.find_nearest_mel(f_max)

        f_min_mel = max(f_min_mel - self.pad_mels, 0)
        f_max_mel = min(f_max_mel + self.pad_mels, len(self.mel_freqs) - 1)

        f_min, f_max = self.mel_freqs[f_min_mel], self.mel_freqs[f_max_mel]
        return f_min, f_max, f_min_mel, f_max_mel


class RandomMixer:
    def __init__(self, mixers, p=None):
        self.mixers = mixers
        self.p = p

    def __call__(self, dataset, image, target, rows):
        mixer = np.random.choice(self.mixers, p=self.p)
        image, target, rows = mixer(dataset, image, target, rows)
        return image, target, rows


class UseMixerWithProb:
    def __init__(self, mixer, prob=.5):
        self.mixer = mixer
        self.prob = prob

    def __call__(self, dataset, image, target, rows):
        if random.random() < self.prob:
            rnd_image, rnd_target, rnd_rows = self.mixer(dataset, image, target, rows)
            if rnd_rows is None:
                rnd_rows = rows
            else:
                rnd_rows = pd.concat([rows, rnd_rows])

            return rnd_image, rnd_target, rnd_rows
        return image, target, rows
