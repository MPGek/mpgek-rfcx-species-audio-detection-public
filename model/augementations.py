import audiomentations
import cv2

from model.mixers import UseMixerWithProb, RandomMixer, SigmoidConcatMixer, AddMixer, SigmoidVerticalConcatMixer
from model.random_resized_crop import RandomResizedCrop, RandomResizedCrop2
from model.transforms import Compose, UseWithProb, SpecAugment, SpectreScale, PreprocessMelImage, GaussNoise, OneOf, \
    PadToSize, RandomCrop, PreprocessSingleChannelMelImage

wave_augmentations = {
    0: None,
    1: audiomentations.Compose([
        audiomentations.AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
        audiomentations.TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
        audiomentations.PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
        audiomentations.Shift(min_fraction=-0.5, max_fraction=0.5, p=0.5),
    ], p=0.96),  # (1-(1-0.5)^4)*0.96==0.9 - In total there will be 90% augmented samples
    2: audiomentations.Compose([
        audiomentations.AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.010, p=0.95),
        audiomentations.Shift(min_fraction=-0.1, max_fraction=0.1, p=0.3),
    ], p=1),
}

size_4_sec_750_hop = 256

_base_mel_post_process = {
    'none': [],
    '3ch_1': [
        # Use librosa.feature.delta with order 1 and 2 for creating 2 additional channels then divide by 100
        PreprocessMelImage(),
    ],
    '1ch_1': [PreprocessSingleChannelMelImage(), ],
    '1ch_2_norm': [PreprocessSingleChannelMelImage(normalize=True), ],
}
_base_mel_aug = {
    'medium_4': [
        UseWithProb(SpectreScale(0.85, 1.15), prob=0.33),
        UseWithProb(GaussNoise(4), prob=0.77),
        UseWithProb(
            # Random resize crop helps a lot, but I can't explain why ¯\_(ツ)_/¯
            RandomResizedCrop(scale=(0.8, 1.0), ratio=(28, 32)),  # wide size
            prob=0.33
        ),
        UseWithProb(SpecAugment(num_mask=3, freq_masking=0.08, time_masking=0.020), 0.5),
    ],
    'medium_5': [
        UseWithProb(GaussNoise(4), prob=0.77),
        UseWithProb(
            # Random resize crop helps a lot, but I can't explain why ¯\_(ツ)_/¯
            RandomResizedCrop(scale=(0.8, 1.0), ratio=(6.5, 7.5)),  # wide size
            prob=0.33
        ),
        UseWithProb(SpecAugment(num_mask=3, freq_masking=0.08, time_masking=0.020), 0.5),
    ],
    'medium_5_ratio_2': [
        UseWithProb(GaussNoise(3), prob=0.77),
        UseWithProb(RandomResizedCrop(scale=(0.8, 1.0), ratio=(1.7, 2.3)), prob=0.33),
        UseWithProb(SpecAugment(num_mask=2, freq_masking=0.08, time_masking=0.020), 0.5),
    ],
    'low_5': [
        UseWithProb(GaussNoise(3), prob=0.77),
        UseWithProb(SpecAugment(num_mask=2, freq_masking=0.08, time_masking=0.020), 0.5),
    ],
    'medium_5_resize': [
        UseWithProb(GaussNoise(3), prob=0.77),
        UseWithProb(RandomResizedCrop2(reduce_size=(40, 20), interpolation=cv2.INTER_NEAREST), prob=0.33),
        UseWithProb(SpecAugment(num_mask=2, freq_masking=0.08, time_masking=0.020), 0.5),
    ],
    'low_6': [
        UseWithProb(GaussNoise(3), prob=0.77),
        UseWithProb(RandomResizedCrop2(reduce_size=(40, 20), interpolation=cv2.INTER_NEAREST), prob=0.33),
    ],
    'medium_6': [
        UseWithProb(GaussNoise(4), prob=0.77),
        UseWithProb(SpecAugment(num_mask=3,
                                freq_masking=0.08,
                                time_masking=0.020), 0.5),
    ],
    'medium_4_RandomCrop': [
        OneOf([
            PadToSize(size_4_sec_750_hop, mode='wrap'),  # Repeat small clips
            PadToSize(size_4_sec_750_hop, mode='constant'),  # Pad with a minimum value
        ], p=[0.5, 0.5]),
        RandomCrop(size_4_sec_750_hop),  # Crop 256 values on time axis
        UseWithProb(SpectreScale(0.85, 1.15), prob=0.33),
        UseWithProb(GaussNoise(4), prob=0.77),
        # Random resize crop helps a lot, but I can't explain why ¯\_(ツ)_/¯
        UseWithProb(RandomResizedCrop(scale=(0.8, 1.0), ratio=(1.7, 2.3)), prob=0.33),  # 128x256 size
        # SpecAugment [1], masking blocks of frequency channels, and masking blocks of time steps
        UseWithProb(SpecAugment(num_mask=2, freq_masking=0.15, time_masking=0.20), 0.5),  # 128x256 size
    ]
}

mel_augmentations = {
    0: None,
    1: Compose([
        UseWithProb(RandomResizedCrop(scale=(0.8, 1.0), ratio=(28, 32)), prob=0.33),
        UseWithProb(SpectreScale(0.85, 1.15), prob=0.33),
        UseWithProb(SpecAugment(num_mask=3, freq_masking=0.08, time_masking=0.020), 0.5),
    ]),
    2: Compose([PreprocessMelImage()]),
    3: Compose([
        UseWithProb(RandomResizedCrop(scale=(0.8, 1.0), ratio=(28, 32)), prob=0.33),
        UseWithProb(SpectreScale(0.85, 1.15), prob=0.33),
        UseWithProb(SpecAugment(num_mask=3, freq_masking=0.08, time_masking=0.020), 0.5),
        PreprocessMelImage(),
    ]),
    4: Compose(_base_mel_aug['medium_4'] + _base_mel_post_process['3ch_1']),
    5: Compose(_base_mel_aug['medium_4_RandomCrop'] + _base_mel_post_process['3ch_1']),
    6: Compose(_base_mel_aug['medium_4'] + _base_mel_post_process['1ch_1']),
    7: Compose(_base_mel_post_process['1ch_1']),
    8: Compose(_base_mel_aug['medium_4'] + _base_mel_post_process['1ch_2_norm']),
    9: Compose(_base_mel_post_process['1ch_2_norm']),
    10: Compose(_base_mel_aug['medium_5'] + _base_mel_post_process['1ch_2_norm']),
    11: Compose(_base_mel_aug['medium_6'] + _base_mel_post_process['1ch_2_norm']),
    12: Compose(_base_mel_aug['medium_5_ratio_2'] + _base_mel_post_process['1ch_2_norm']),
    13: Compose(_base_mel_aug['low_5'] + _base_mel_post_process['1ch_2_norm']),
    14: Compose(_base_mel_aug['medium_5_resize'] + _base_mel_post_process['1ch_2_norm']),
    15: Compose(_base_mel_aug['low_6'] + _base_mel_post_process['1ch_2_norm']),
}

mixers = {
    0: None,
    1: UseMixerWithProb(RandomMixer([
        SigmoidConcatMixer(sigmoid_range=(3, 12)),
        AddMixer(alpha_dist='uniform')
    ], p=[0.6, 0.4]), prob=0.8),
    2: UseMixerWithProb(AddMixer(alpha_dist='uniform'), prob=0.7),
    3: -1,
    4: -1,
    5: -1,
    6: -1,
    7: -1,
    8: -1,
}


def init_mixers(f_min, f_max, n_mels):
    mixers[3] = UseMixerWithProb(RandomMixer([
        SigmoidVerticalConcatMixer(f_min, f_max, n_mels, sigmoid_range=(4, 12), pad_mels=40, band_attention_mode=True),
        AddMixer(alpha_dist='uniform')
    ], p=[0.6, 0.4]), prob=0.8)
    mixers[4] = UseMixerWithProb(RandomMixer([
        SigmoidVerticalConcatMixer(f_min, f_max, n_mels, sigmoid_range=(4, 12), pad_mels=40, band_attention_mode=True),
        SigmoidVerticalConcatMixer(f_min, f_max, n_mels, sigmoid_range=(4, 12), pad_mels=40, band_attention_mode=False),
        AddMixer(alpha_dist='uniform')
    ], p=[0.3, 0.3, 0.4]), prob=0.8)
    mixers[5] = UseMixerWithProb(RandomMixer([
        SigmoidVerticalConcatMixer(f_min, f_max, n_mels, sigmoid_range=(4, 12), pad_mels=40, band_attention_mode=True),
        SigmoidVerticalConcatMixer(f_min, f_max, n_mels, sigmoid_range=(4, 12), pad_mels=40, band_attention_mode=False),
        AddMixer(alpha_dist='uniform')
    ], p=[0.55, 0.15, 0.3]), prob=0.8)
    mixers[6] = UseMixerWithProb(
        SigmoidVerticalConcatMixer(f_min, f_max, n_mels, sigmoid_range=(4, 12), pad_mels=20, band_attention_mode=True),
        prob=0.6)
    mixers[7] = SigmoidVerticalConcatMixer(f_min, f_max, n_mels, sigmoid_range=(4, 12), pad_mels=20,
                                           band_attention_mode=True)
    mixers[8] = UseMixerWithProb(RandomMixer([
        SigmoidVerticalConcatMixer(f_min, f_max, n_mels, sigmoid_range=(4, 12), pad_mels=24, band_attention_mode=True),
        SigmoidVerticalConcatMixer(f_min, f_max, n_mels, sigmoid_range=(4, 12), pad_mels=24, band_attention_mode=False),
    ], p=[0.75, 0.25]), prob=0.8)

