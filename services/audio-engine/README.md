# Audio Engine

Canonical PCG pipeline and inference service for CardioAgent-Ops.

## Pipeline

`preprocess.load_wav` (path or `io.BytesIO`) normalizes every recording to:

- sample rate: 2000 Hz
- duration: 2 s (4000 samples)
- center crop / zero padding
- z-normalization

`features.mel_spectrogram` produces the model input `(64, 128)` and
`features.mfcc_mean` produces the 40-D classical baseline feature.

## Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/` | service and pipeline metadata |
| GET | `/health` | model-loaded status and artifact path |
| POST | `/predict` | multipart wav upload -> prediction |

`POST /predict` returns:

```json
{
  "status": "success",
  "prediction": 1,
  "label": "abnormal",
  "probability": 0.91,
  "model": "cardio_model.onnx"
}
```

If no ONNX artifact is present the endpoint returns `503`; `/health` reports
`status: degraded` and `model_loaded: false`.

## Training

```bash
python train.py --data /path/to/dataset/train --model cnn_transformer \
  --out artifacts/cardio_model.pt --audit
```

Outputs `artifacts/cardio_model.pt` and `artifacts/cardio_model.onnx`.
Environment overrides: `DATA_DIR`, `MODEL_OUT`, `MLFLOW_TRACKING_URI`,
`MODEL_PATH` (serving).

## Tests

```bash
python -m pytest tests -q
```

## Legacy

The previous librosa / 149-D feature implementation is archived at
`legacy/audio-engine/` for reference.
