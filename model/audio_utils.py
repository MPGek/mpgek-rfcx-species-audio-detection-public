from typing import Optional

import audiomentations
import librosa
import librosa.display
import numpy as np


# @perf_timing()


def read_audio(audio_config: dict, file_path):
    # min_samples = int(60 * audio_config['sampling_rate'])
    # try:

    y, sr = librosa.load(file_path, sr=audio_config['sampling_rate'])
    return y

    # dirname, file_name = os.path.split(file_path)
    # npz_dir = os.path.join(dirname, 'npz')
    # if not os.path.exists(npz_dir):
    #     os.mkdir(npz_dir)
    #
    # npz_file = os.path.join(npz_dir, os.path.splitext(file_name)[0] + '.npz')
    # if not os.path.exists(npz_file):
    #     y, sr = librosa.load(file_path, sr=audio_config['sampling_rate'])
    #     np.savez(npz_file, wave=y)
    #     # np.savez(npz_file, wave=y.astype(np.float16))
    # else:
    #     y = np.load(npz_file)['wave']  # type: np.ndarray
    #     # y = y.astype(np.float32)

    trim_y = y
    # trim_y, trim_idx = librosa.effects.trim(y)  # trim, top_db=default(60)

    # if len(trim_y) < min_samples:
    #     center = (trim_idx[1] - trim_idx[0]) // 2
    #     left_idx = max(0, center - min_samples // 2)
    #     right_idx = min(len(y), center + min_samples // 2)
    #     trim_y = y[left_idx:right_idx]
    #
    #     if len(trim_y) < min_samples:
    #         padding = min_samples - len(trim_y)
    #         offset = padding // 2
    #         trim_y = np.pad(trim_y, (offset, padding - offset), 'constant')
    return trim_y
    # except BaseException as e:
    #     print(f"Exception while reading file {e}")
    #     return np.zeros(min_samples, dtype=np.float32)


# @perf_timing()
def audio_to_melspectrogram(audio_config: dict, audio):
    sampling_rate = audio_config['sampling_rate']
    hop_length = audio_config['hop_length']
    fmin = audio_config['fmin']
    fmax = audio_config['fmax']
    n_mels = audio_config['n_mels']

    n_fft_mels_ratio = audio_config.get('n_fft_mels_ratio', 0)
    n_fft = audio_config.get('n_fft', 0)
    ftt_win_length = audio_config.get('ftt_win_length', None)

    if n_fft_mels_ratio > 0:
        n_fft = n_mels * n_fft_mels_ratio

    spectrogram = librosa.feature.melspectrogram(audio, sr=sampling_rate, n_mels=n_mels, hop_length=hop_length,
                                                 n_fft=n_fft, fmin=fmin, fmax=fmax, win_length=ftt_win_length)

    spectrogram = librosa.power_to_db(spectrogram)
    spectrogram = spectrogram.astype(np.float32)
    return spectrogram


# @perf_timing()
def show_melspectrogram(audio_config: dict, mels, title='Log-frequency power spectrogram'):
    import matplotlib.pyplot as plt

    plt.rcParams["figure.figsize"] = (20, 4)
    librosa.display.specshow(mels, x_axis='time', y_axis='mel',
                             sr=audio_config['sampling_rate'], hop_length=audio_config['hop_length'],
                             fmin=audio_config['fmin'], fmax=audio_config['fmax'])
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.show()


# @perf_timing()
def read_as_melspectrogram(audio_config: dict, file_path, wave_aug: Optional[audiomentations.Compose],
                           time_stretch=1.0, pitch_shift=0.0, debug_display=False):
    x = read_audio(audio_config, file_path)
    augmented_samples = original_samples = x

    if wave_aug is not None:
        augmented_samples = wave_aug(samples=x, sample_rate=48000)
        x = augmented_samples

    # if time_stretch != 1.0:
    #     x = librosa.effects.time_stretch(x, time_stretch)
    #
    # if pitch_shift != 0.0:
    #     librosa.effects.pitch_shift(x, audio_config['sampling_rate'], n_steps=pitch_shift)

    mels = audio_to_melspectrogram(audio_config, x)
    # mels2 = audio_to_melspectrogram(audio_config, augmented_samples)

    is_debug = False
    # is_debug = True
    if is_debug:
        # DEBUG
        original_mel = audio_to_melspectrogram(audio_config, original_samples)
        augmented_mel = audio_to_melspectrogram(audio_config, augmented_samples)

        diff = augmented_mel - original_mel
        show_melspectrogram(audio_config, diff)

        import sounddevice as sd
        show_melspectrogram(audio_config, original_mel)
        sd.play(original_samples, 48000, blocking=False)
        sd.stop()

        show_melspectrogram(audio_config, augmented_mel)
        sd.play(augmented_samples, 48000, blocking=False)
        sd.stop()

    # import IPython
    # IPython.display.display(IPython.display.Audio(x, rate=audio_config['sampling_rate']))

    if debug_display:
        # import cv2
        # mels2 = np.copy(mels)
        # mels2 = np.array(((mels / 100)  + 1)*255, dtype=np.uint8)
        # mels2 = cv2.applyColorMap(mels2, cv2.COLORMAP_JET)
        # # cv2.normalize(mels, mels2, 0, 1, cv2.NORM_MINMAX)
        # cv2.imshow("Img", mels2)
        # cv2.waitKey(0)
        # return mels
        import IPython
        IPython.display.display(IPython.display.Audio(x, rate=audio_config['sampling_rate']))
        show_melspectrogram(mels)

    return mels
