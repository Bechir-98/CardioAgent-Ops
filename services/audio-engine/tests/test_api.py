import sys
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import main  # noqa: E402
from features import N_MELS, TIME_FRAMES  # noqa: E402
from models import get  # noqa: E402

SAMPLES = ROOT / "data" / "samples"
client = TestClient(main.app)


def _upload(path: Path):
    return {"file": (path.name, path.read_bytes(), "audio/wav")}


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] in {"ok", "degraded"}
    assert "model_loaded" in body
    assert "model_path" in body


def test_predict_without_model(monkeypatch):
    monkeypatch.setattr(main.registry, "session", None)
    r = client.post("/predict", files=_upload(SAMPLES / "normal_1.wav"))
    assert r.status_code == 503
    assert "Model not loaded" in r.json()["detail"]


def test_predict_invalid_audio(monkeypatch):
    monkeypatch.setattr(main.registry, "session", object())
    r = client.post("/predict", files={"file": ("bad.wav", b"not-a-wav", "audio/wav")})
    assert r.status_code == 400


def test_predict_with_exported_model(monkeypatch, tmp_path):
    import onnxruntime as ort
    import torch

    onnx_path = tmp_path / "cardio_model.onnx"
    model = get("cnn").eval()
    torch.onnx.export(
        model,
        torch.randn(1, 1, N_MELS, TIME_FRAMES),
        str(onnx_path),
        input_names=["mel"],
        output_names=["logits"],
        dynamo=False,
    )
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    monkeypatch.setattr(main.registry, "session", session)
    monkeypatch.setattr(main.registry, "model_path", onnx_path)

    r = client.post("/predict", files=_upload(SAMPLES / "abnormal_1.wav"))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "success"
    assert body["prediction"] in (0, 1)
    assert body["label"] in {"normal", "abnormal"}
    assert 0.0 <= body["probability"] <= 1.0
    assert body["model"] == "cardio_model.onnx"
