"""Canonical PCG preprocessing: raw (4000Hz) -> 2000Hz / 2s / z-norm.

Single function used by train + infer + tests + all 5 models.
Stdlib wave + numpy + scipy only (no librosa).
"""
import contextlib
import wave

import numpy as np
from scipy.signal import resample

SR = 2000
DUR = 2.0
N = int(SR * DUR)  # 4000 samples


def load_wav(path, sr=SR, dur=DUR) -> np.ndarray:
    n_target = int(sr * dur)
    with contextlib.closing(wave.open(str(path), "rb")) as w:
        n_frames, fr, ch, sw = w.getnframes(), w.getframerate(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(n_frames)
    dtype = {1: np.uint8, 2: np.int16, 4: np.int32}[sw]
    y = np.frombuffer(raw, dtype=dtype).astype(np.float32)
    if sw == 1:  # 8-bit PCM is unsigned
        y = (y - 128.0) / 128.0
    elif sw == 2:
        y /= 32768.0
    else:
        y /= 2147483648.0
    if ch > 1:
        y = y.reshape(-1, ch).mean(axis=1)
    if fr == sr:
        pass
    elif fr % sr == 0:  # fast path: 4000 -> 2000 decimate
        y = y[:: fr // sr]
    else:
        y = resample(y, int(len(y) * sr / fr)).astype(np.float32)
    # crop / pad to fixed window
    if len(y) > n_target:
        start = (len(y) - n_target) // 2  # center crop: keeps B1/B2
        y = y[start: start + n_target]
    elif len(y) < n_target:
        y = np.pad(y, (0, n_target - len(y)))
    # z-norm (flat recordings guarded)
    std = y.std()
    y = (y - y.mean()) / (std + 1e-8) if std > 1e-8 else y - y.mean()
    return y.astype(np.float32)


if __name__ == "__main__":  # ponytail: runnable self-check, no framework
    import sys
    y = load_wav(sys.argv[1])
    assert y.shape == (N,), y.shape
    assert abs(float(y.mean())) < 0.05, y.mean()
    print("ok", y.shape, "mean=%.4f std=%.4f" % (y.mean(), y.std()))
