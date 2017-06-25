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
