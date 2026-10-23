import librosa
import numpy as np

def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int = 2000) -> np.ndarray:
    """Resamples audio to the target sampling rate."""
    if orig_sr == target_sr:
        return audio
    return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)

def get_windows(audio: np.ndarray, window_size_sec: int, sr: int = 2000):
    """
    Slices audio into fixed-size windows.
    Returns a generator of windowed audio segments.
    """
    window_samples = window_size_sec * sr
    audio_len = len(audio)

    for start in range(0, audio_len, window_samples):
        end = start + window_samples
        segment = audio[start:end]

        # Pad with zeros if the final segment is shorter than window_size
        if len(segment) < window_samples:
            segment = np.pad(segment, (0, window_samples - len(segment)), mode='constant')

        yield segment
