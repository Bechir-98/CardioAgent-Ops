"""Mel-spec + MFCC-mean with numpy/scipy only (no librosa)."""
import numpy as np
from scipy.fftpack import dct
from scipy.signal import stft

N_MELS = 64
N_MFCC = 40
N_FFT = 256
TIME_FRAMES = 128


def _hz2mel(hz):
    return 2595.0 * np.log10(1.0 + np.asarray(hz) / 700.0)


def _mel2hz(mel):
    return 700.0 * (10.0 ** (np.asarray(mel) / 2595.0) - 1.0)


def _mel_fb(sr, n_fft, n_mels=N_MELS):
    pts = _mel2hz(np.linspace(_hz2mel(0), _hz2mel(sr / 2), n_mels + 2))
    bins = np.floor((n_fft + 1) * pts / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for m in range(1, n_mels + 1):
        f0, f1, f2 = bins[m - 1], bins[m], bins[m + 1]
        if f1 > f0:
            fb[m - 1, f0:f1] = (np.arange(f0, f1) - f0) / max(f1 - f0, 1)
        if f2 > f1:
            fb[m - 1, f1:f2] = (f2 - np.arange(f1, f2)) / max(f2 - f1, 1)
    return fb


def mel_spectrogram(y, sr=2000, n_mels=N_MELS, n_fft=N_FFT, frames=TIME_FRAMES) -> np.ndarray:
    hop = max(1, (len(y) - n_fft) // max(frames - 1, 1))
    _, _, z = stft(y, fs=sr, window="hann", nperseg=n_fft, noverlap=n_fft - hop,
                   nfft=n_fft, boundary="zeros", padded=True)
    power = np.abs(z) ** 2
    mel = _mel_fb(sr, n_fft, n_mels) @ power
    log_mel = np.log10(mel + 1e-10).astype(np.float32)
    if log_mel.shape[1] >= frames:  # center crop time
        s = (log_mel.shape[1] - frames) // 2
        log_mel = log_mel[:, s: s + frames]
    else:
        log_mel = np.pad(log_mel, ((0, 0), (0, frames - log_mel.shape[1])))
    return log_mel


def mfcc_mean(y, sr=2000, n_mfcc=N_MFCC) -> np.ndarray:
    """1D feature for DT/SVM: mean MFCC over time."""
    log_mel = mel_spectrogram(y, sr)
    mfcc = dct(log_mel, type=2, axis=0, norm="ortho")[:n_mfcc]
    return mfcc.mean(axis=1).astype(np.float32)


if __name__ == "__main__":  # ponytail: runnable self-check
    from preprocess import load_wav
    import sys
    y = load_wav(sys.argv[1] if len(sys.argv) > 1 else "../data/samples/normal_1.wav")
    m = mel_spectrogram(y)
    c = mfcc_mean(y)
    assert m.shape == (N_MELS, TIME_FRAMES), m.shape
    assert c.shape == (N_MFCC,), c.shape
    print("ok mel", m.shape, "mfcc", c.shape)
