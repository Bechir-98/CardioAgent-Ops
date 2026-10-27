# CardioAgent-Ops

Micro-services platform for cardio-acoustic analysis (PCG heart sounds): a
canonical audio ML engine, experimentation/tracking, and service scaffolding for
the LLM/RAG/agent layers.

The audio pipeline is the `v2` design, now housed in
[`services/audio-engine`](services/audio-engine):

- raw PCG -> **2000 Hz**, **2 s / 4000 samples**, center crop/pad, z-normalized
- **log-Mel 64x128** + **MFCC-40 mean**, numpy/scipy only (no librosa)
- PyTorch **CNN** and **CNN + Transformer** models
- quality audit for corrupt / short / flat / clipped recordings
- training selected on **abnormal (pathological) recall**
- ONNX export for serving

## Services

| Service | Port | Purpose |
|---|---|---|
| `api-gateway` | 8000 | Upload UI / gateway (mock response for now) |
| `audio-engine` | 8001 | Real inference: wav -> canonical pipeline -> ONNX |
| `agent-orchestrator` | 8002 | Health-check scaffold |
| `llm-server` | 8003 | Health-check scaffold |
| `vector-db` | 8004 | Health-check scaffold |
| `mlflow-server` | 5000 | Experiment tracking |

## Quick start

```bash
docker compose up --build
```

Then:

```bash
curl http://localhost:8001/health
curl -F "file=@services/audio-engine/data/samples/normal_1.wav" \
     http://localhost:8001/predict
```

The audio engine returns `503` on `/predict` until a model artifact exists at
`services/audio-engine/artifacts/cardio_model.onnx`.

## Train and export a model

The fused PASCAL + PhysioNet dataset is local-only (not in git). Expected layout:

```text
dataset/train/
  normal/*.wav
  abnormal/*.wav
```

```bash
pip install -r services/audio-engine/requirements.txt

python services/audio-engine/train.py \
  --data /path/to/dataset/train \
  --model cnn_transformer \
  --epochs 3 \
  --out services/audio-engine/artifacts/cardio_model.pt \
  --tracking-uri http://localhost:5000 \
  --audit
```

This writes `cardio_model.pt` and `cardio_model.onnx` into
`services/audio-engine/artifacts/` (git-ignored). Docker Compose mounts that
directory into the audio-engine container.

## Tests

```bash
pip install -r services/audio-engine/requirements.txt
python -m pytest services/audio-engine/tests -q
```

Four small wavs under `services/audio-engine/data/samples/` keep the pipeline and
API tests dataset-free.

## Notebook

[`notebooks/01_eda_data_wrangling.ipynb`](notebooks/01_eda_data_wrangling.ipynb)
covers EDA, signal/quality audits, feature mining, the 80/10/10 split, training,
and confusion-matrix evaluation.

### Google Colab

1. Upload the repo folder or a repo `.zip` to Drive (or `/content`). The
   notebook can auto-detect it and will unzip a matching archive if needed.
   Alternatively set `REPO_DIR` to the repo path before running.
2. Put the dataset on Drive at
   `/content/drive/MyDrive/CardioAgentOps/dataset/train/{normal,abnormal}`.
   Alternatively set `DATA_DIR` to the dataset folder before running.
3. Run the notebook; it installs `onnx` / `onnxruntime` automatically when
   missing. If export still prints `onnx skipped`, run
   `!pip install -q onnx onnxruntime` and re-run the training cell.
4. Download `cardio_model.pt` / `cardio_model.onnx` from the final cell and copy
   the ONNX file to `services/audio-engine/artifacts/`.

The setup cell removes any pip-installed `preprocess` module from `sys.modules`
and checks that the imported module really comes from the repo, so uploading only
the notebook is no longer enough on its own — the repo folder/zip must be
available in Colab.

The notebook automatically detects the repo and dataset under the common Colab
paths, so only step 2 usually needs adjusting.

## Repository layout

```text
services/
  audio-engine/       canonical pipeline, API, training, tests, samples
  api-gateway/        FastAPI upload UI
  agent-orchestrator/ health scaffold
  llm-server/         health scaffold
  vector-db/          health scaffold
  mlflow-server/      tracking server
notebooks/            EDA + benchmark notebook
legacy/audio-engine/  previous librosa-based implementation (reference only)
```
