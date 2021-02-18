# Source: https://www.kaggle.com/daisukelab/creating-fat2019-preprocessed-data
from pathlib import Path

import librosa
import librosa.display
import numpy as np


# from src.config import audio as config


class audio:
    sampling_rate = 48000
    # fmin, fmax, n_mels, n_fft, hop_length, win_length = 50, 15000, 128, 20 * 128, 375 * 2, None
    # fmin, fmax, n_mels, n_fft, hop_length, win_length = 50, 15000, 256, 20 * 256, 375 * 2, None
    # hop_length = 375 * 2 // 2
    # fmin = 20, fmax = 24000,
    # fmin, fmax, n_mels, n_fft, hop_length  = 1500, 6000, 224, 2048, 128
    # fmin, fmax, n_mels, n_fft, hop_length  = 50, 15000, 224, 2048, 256
    # fmin, fmax, n_mels, n_fft, hop_length  = 50, 15000, 256, 2048, 256
    # fmin, fmax, n_mels, n_fft, hop_length, win_length  = 50, 15000, 256, 1024, 256, None
    fmin, fmax, n_mels, n_fft, hop_length, win_length = 50, 15000, 256, 2*2048, 256, 1536
    # fmin, fmax, n_mels, n_fft, hop_length, win_length = 50, 15000, 256, 2*2048, 256, 1024
    min_seconds = 0.5

    @classmethod
    def get_config_dict(cls):
        config_dict = dict()
        for key, value in cls.__dict__.items():
            if key[:1] != '_' and \
                    key not in ['get_config_dict', 'get_hash']:
                config_dict[key] = value
        return config_dict

    @classmethod
    def get_hash(cls, **kwargs):
        config_dict = cls.get_config_dict()
        config_dict = {**config_dict, **kwargs}
        hash_str = json.dumps(config_dict,
                              sort_keys=True,
                              ensure_ascii=False,
                              separators=None)
        hash_str = hash_str.encode('utf-8')
        return sha1(hash_str).hexdigest()[:7]


config = audio


def get_audio_config():
    return config.get_config_dict()


def read_audio(file_path):
    min_samples = int(config.min_seconds * config.sampling_rate)
    try:
        y, sr = librosa.load(file_path, sr=config.sampling_rate)
        trim_y, trim_idx = librosa.effects.trim(y)  # trim, top_db=default(60)

        if len(trim_y) < min_samples:
            center = (trim_idx[1] - trim_idx[0]) // 2
            left_idx = max(0, center - min_samples // 2)
            right_idx = min(len(y), center + min_samples // 2)
            trim_y = y[left_idx:right_idx]

            if len(trim_y) < min_samples:
                padding = min_samples - len(trim_y)
                offset = padding // 2
                trim_y = np.pad(trim_y, (offset, padding - offset), 'constant')
        return trim_y
    except BaseException as e:
        print(f"Exception while reading file {e}")
        return np.zeros(min_samples, dtype=np.float32)


def audio_to_melspectrogram(audio):
    spectrogram = librosa.feature.melspectrogram(audio,
                                                 sr=config.sampling_rate,
                                                 n_mels=config.n_mels,
                                                 hop_length=config.hop_length,
                                                 n_fft=config.n_fft,
                                                 fmin=config.fmin,
                                                 fmax=config.fmax,
                                                 win_length=config.win_length,
                                                 # win_length=config.n_fft,
                                                 )

    # new_audio = librosa.feature.inverse.mel_to_stft(spectrogram, sr=config.sampling_rate, n_fft=config.n_fft)
    # wavfile.write(r'd:\Projects\Kaggle\rfcx-species-audio-detection_data\train\003bec244.wav', config.sampling_rate, new_audio)

    spectrogram = librosa.power_to_db(spectrogram)
    spectrogram = spectrogram.astype(np.float32)
    return spectrogram


def show_melspectrogram(mels, title='Log-frequency power spectrogram'):
    import matplotlib.pyplot as plt

    plt.rcParams["figure.figsize"] = (50, 8)
    librosa.display.specshow(mels, x_axis='time', y_axis='mel',
                             sr=config.sampling_rate, hop_length=config.hop_length,
                             fmin=config.fmin, fmax=config.fmax, )
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.show()


# def _wav_to_spec(x):
#     spectrogram = librosa.feature.melspectrogram(audio,
#                                                  sr=config.sampling_rate,
#                                                  n_mels=config.n_mels,
#                                                  hop_length=config.hop_length,
#                                                  n_fft=config.n_fft,
#                                                  fmin=config.fmin,
#                                                  fmax=config.fmax)
#
#     mel_power = 2
#     stfts = librosa.stft(n_fft=config.n_fft,hop_length=config.hop_length )
#     stfts = tf.signal.stft(x["audio_wav"], frame_length=2048, frame_step=512, fft_length=2048)
#     spectrograms = tf.abs(stfts) ** mel_power
#
#     # Warp the linear scale spectrograms into the mel-scale.
#     num_spectrogram_bins = stfts.shape[-1]
#     lower_edge_hertz, upper_edge_hertz, num_mel_bins = FMIN, FMAX, N_MEL
#
#     linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
#         num_mel_bins, num_spectrogram_bins, SR, lower_edge_hertz,
#         upper_edge_hertz)
#     mel_spectrograms = tf.tensordot(
#         spectrograms, linear_to_mel_weight_matrix, 1)
#     mel_spectrograms.set_shape(spectrograms.shape[:-1].concatenate(
#         linear_to_mel_weight_matrix.shape[-1:]))
#
#     # Compute a stabilized log to get log-magnitude mel-scale spectrograms.
#     log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)
#
#     y = {
#         'audio_spec': tf.transpose(log_mel_spectrograms),  # (num_mel_bins, frames)
#     }
#     y.update(x)
#     return y


def read_as_melspectrogram(file_path, time_stretch=1.0, pitch_shift=0.0,
                           debug_display=False):
    x = read_audio(file_path)
    if time_stretch != 1.0:
        x = librosa.effects.time_stretch(x, time_stretch)

    if pitch_shift != 0.0:
        librosa.effects.pitch_shift(x, config.sampling_rate, n_steps=pitch_shift)

    # mels = audio_to_melspectrogram(x)
    mels = audio_to_melspectrogram(x)
    # start = 5 * config.sampling_rate // config.hop_length
    # end = 7 * config.sampling_rate // config.hop_length
    # start = 50 * config.sampling_rate // config.hop_length
    # end = 54 * config.sampling_rate // config.hop_length
    # mels = mels[:, start:end]

    if debug_display:
        import cv2
        mels2 = np.copy(mels)
        cv2.normalize(mels, mels2, 0, 1, cv2.NORM_MINMAX)
        mels2 = mels2[::-1, :]
        # mels2 = np.array(((mels / 100)  + 1)*255, dtype=np.uint8)
        mels2 = np.array(mels2 * 255, dtype=np.uint8)
        mels2 = cv2.applyColorMap(mels2, cv2.COLORMAP_MAGMA)
        cv2.imshow("Img", mels2)
        # return mels
        import IPython
        IPython.display.display(IPython.display.Audio(x, rate=config.sampling_rate))
        show_melspectrogram(mels)

        cv2.waitKey(0)
    return mels


if __name__ == "__main__":
    train_dir = Path(r'..\data\train')
    # path = train_dir / '003bec244.flac'
    # path = train_dir / '015113cad.flac'  # 015113cad;15;1;50.0533;93.75;53.3973;1125.0
    # path = train_dir / '0295e3234.flac'  # 0295e3234,11,1,5.1606,1808.79,6.2984,5684.77
    path = train_dir / '011f25080.flac'  # 011f25080,18,1,5.6853,3187.5,6.3787,5062.5
    x = read_as_melspectrogram(path, debug_display=True)
    print(x.shape)
