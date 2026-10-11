import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "audio_engine"))

from features import TIME_FRAMES, N_MELS, mel_spectrogram, mfcc_mean
from preprocess import DUR, N, SR, load_wav


def test_wav_window():
    y = load_wav("data/samples/normal_1.wav")
    assert y.shape == (N,) == (SR * int(DUR),)
    assert abs(float(y.mean())) < 0.05


def test_mel_shapes():
    y = load_wav("data/samples/abnormal_1.wav")
    assert mel_spectrogram(y).shape == (N_MELS, TIME_FRAMES)
    assert mfcc_mean(y).shape == (40,)


def test_quality_audit():
    from quality import audit
    s = audit("data/samples")
    assert sum(sum(v.values()) for v in s.values()) == 4
    assert sum(v["corrupt"] + v["flat"] for v in s.values()) == 0


def test_models_forward():
    torch = __import__("torch")
    from models import get
    for n in ("cnn", "cnn_transformer"):
        out = get(n).eval()(torch.randn(1, 1, N_MELS, TIME_FRAMES))
        assert out.shape == (1, 2)
