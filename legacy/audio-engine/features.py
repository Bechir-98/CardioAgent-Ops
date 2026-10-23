import librosa
import numpy as np
from dataclasses import dataclass
from typing import List, Dict

@dataclass
class FeatureConfig:
    n_mfcc: int = 20
    n_mels: int = 128
    hop_length: int = 512
    window_size: int = 2048

def extract_mfcc(signal: np.ndarray, sr: int, config: FeatureConfig):
    """Extracts Mel-frequency cepstral coefficients."""
    mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=config.n_mfcc,
                                 hop_length=config.hop_length, n_fft=config.window_size)
    return np.mean(mfccs, axis=1)

def extract_zcr(signal: np.ndarray):
    """Extracts Zero Crossing Rate."""
    zcr = librosa.feature.zero_crossing_rate(y=signal)
    return np.mean(zcr)

def extract_mel_spectrogram(signal: np.ndarray, sr: int, config: FeatureConfig):
    """Extracts Mel Spectrogram features."""
    mel = librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=config.n_mels,
                                         hop_length=config.hop_length, n_fft=config.window_size)
    return np.mean(librosa.power_to_db(mel), axis=1)

def extract_features(signal: np.ndarray, sr: int = 2000, config: FeatureConfig = FeatureConfig(), mode: str = 'all'):
    """
    Extracts features from the audio signal.
    mode='all': Combines MFCC, ZCR, and Mel Spectrogram into a single vector.
    mode='mfcc': Extracts only MFCC features.
    """
    mfcc = extract_mfcc(signal, sr, config)

    if mode == 'mfcc':
        return mfcc

    zcr = extract_zcr(signal)
    mel = extract_mel_spectrogram(signal, sr, config)

    # Combine features into a single 1D array
    return np.concatenate([mfcc, [zcr], mel])
