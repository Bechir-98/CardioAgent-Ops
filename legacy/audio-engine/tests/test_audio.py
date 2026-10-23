import pytest
import numpy as np
from preprocessing import resample_audio, get_windows
from features import extract_features, FeatureConfig

def test_resampling():
    # Create mock audio at 4000Hz
    sr_orig = 4000
    audio = np.random.randn(sr_orig * 2)
    resampled = resample_audio(audio, sr_orig, 2000)
    assert len(resampled) == 2000 * 2

def test_windowing():
    sr = 2000
    audio = np.random.randn(sr * 5) # 5 seconds
    windows = list(get_windows(audio, 2, sr))
    # 5s / 2s = 2.5 -> 3 windows (last one padded)
    assert len(windows) == 3
    assert len(windows[0]) == 2 * sr

def test_feature_extraction():
    sr = 2000
    signal = np.random.randn(sr * 2)
    features = extract_features(signal, sr)
    assert isinstance(features, np.ndarray)
    assert features.ndim == 1
    # MFCC(20) + ZCR(1) + Mel(128) = 149
    assert len(features) == 149
