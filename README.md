# CardioAgent-Ops

Micro-services platform for cardio-acoustic analysis (PCG heart sounds): audio ML engine, fine-tuned LLM, RAG, and multi-agent orchestration. Mini-project @ ENSIT.

## Repo status

- [x] **audio_engine/** — canonical preprocessing (4000Hz→2000Hz, 2s/4000-sample window, z-norm), Mel 64×128 + MFCC-40 features (numpy/scipy only), CNN & CNN+Transformer models, benchmark training (recall-on-patho selection), quality audit
- [ ] rag_engine/, agents/, api/, docker/, llm_finetuning/ — next steps

## Run tests

```bash
pip install torch numpy scipy scikit-learn pytest
python -m pytest tests/ -q
```

## Data

PASCAL + PhysioNet (fused, 3784 wavs) lives locally in `dataset/train/{normal,abnormal}` — **not in git** (362MB). Mount it from Drive on Colab or set your own path; `data/samples/` ships 4 small wavs for tests.

## Colab

Open `notebooks/01_eda_data_wrangling.ipynb` — it auto-detects `/content/drive/MyDrive/dataset/train` and `audio_engine/`, audits + mines the data, trains, and exports `cardio_model.onnx`.
