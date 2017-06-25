import numpy as np
from scipy.signal import get_window
from scipy.signal import spectrogram


def get_spectrogram(audio, t1, fs, nperseg, overlap_fraction):
    """Wrapper on scipy.signal.spectrogram with less parameters."""
    window = get_window('hamming', nperseg)
    noverlap = nperseg // overlap_fraction
    f, t, spect = spectrogram(
        audio,
        fs,
        window,
        nperseg,
        noverlap,
        mode='magnitude'
        )
    # Translate times by t1
    t += t1
    return f, t, spect


def aggregate_bark_bands(f, spect):
    """Sum spectrogram values over bark bands."""
    bark_scale = [0, 100, 200, 300, 400, 510, 630, 770,
                  920, 1080, 1270, 1480, 1720, 2000,
                  2320, 2700, 3150, 3700, 4400, 5300]
    max_band_index = len(bark_scale) - 1

    # On our spectrogram frequencies are on the first axis and time is on the second axis
    spectrogram_per_band = np.zeros((max_band_index, spect.shape[1]))
    for band_index in range(max_band_index):
        bark_scale_min = bark_scale[band_index]
        bark_scale_max = bark_scale[band_index + 1]

        f_in_bark_band = (f >= bark_scale_min) & (f < bark_scale_max)
        # Sum over the band
        spectrogram_per_band[band_index, :] = np.sum(spect[f_in_bark_band, :],
                                                     axis=0)

    return bark_scale, spectrogram_per_band
