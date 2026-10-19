"""ONNX model registry for the v2 Mel 64x128 pipeline."""
import os
from pathlib import Path

import numpy as np
import onnxruntime as ort

from features import N_MELS, TIME_FRAMES

LABELS = {0: "normal", 1: "abnormal"}
SERVICE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = SERVICE_DIR / "artifacts" / "cardio_model.onnx"


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - np.max(logits)
    e = np.exp(z)
    return e / np.sum(e)


class ModelRegistry:
    def __init__(self, model_path: str | None = None):
        self.model_path = Path(
            model_path or os.getenv("MODEL_PATH") or DEFAULT_MODEL_PATH
        )
        self.session = self._load_model()

    def _load_model(self) -> ort.InferenceSession | None:
        if not self.model_path.exists():
            print(
                f"Warning: model file {self.model_path} not found. "
                "Train it with `python train.py` or mount the artifacts volume."
            )
            return None
        try:
            return ort.InferenceSession(
                str(self.model_path), providers=["CPUExecutionProvider"]
            )
        except Exception as exc:  # keep the service importable; /health reports it
            print(f"Warning: could not load {self.model_path}: {exc}")
            return None

    @property
    def loaded(self) -> bool:
        return self.session is not None

    def predict(self, mel: np.ndarray) -> dict:
        """Run inference on a (64, 128) log-Mel spectrogram."""
        if self.session is None:
            raise FileNotFoundError(
                f"Model not loaded ({self.model_path}). Train/export it first."
            )

        x = np.asarray(mel, dtype=np.float32)
        if x.shape != (N_MELS, TIME_FRAMES):
            raise ValueError(
                f"Expected mel shape {(N_MELS, TIME_FRAMES)}, got {x.shape}"
            )

        input_name = self.session.get_inputs()[0].name
        logits = np.asarray(
            self.session.run(None, {input_name: x[None, None, :, :]})[0]
        ).reshape(-1)

        if logits.size == 1:  # single-logit sigmoid export
            prob_abnormal = float(1.0 / (1.0 + np.exp(-logits[0])))
            prediction = int(prob_abnormal >= 0.5)
        else:
            probs = _softmax(logits)
            prediction = int(np.argmax(probs))
            prob_abnormal = float(probs[1]) if probs.size > 1 else float(probs[0])

        return {
            "prediction": prediction,
            "label": LABELS.get(prediction, str(prediction)),
            "probability": prob_abnormal,
        }


registry = ModelRegistry()
