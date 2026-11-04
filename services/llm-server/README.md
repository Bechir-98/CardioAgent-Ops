# llm-server — local LLM + RAG

This service hosts the sovereign LLM and the retrieval-augmented generation (RAG)
layer over the ESC guideline corpus.

```text
                 ┌──────────────────────────────────────────────┐
                 │ llm-server container                          │
                 │                                               │
  ESC corpus ───►│ ingest → chunk → MiniLM(ONNX) → ChromaDB (vector-db) │
                 │                                    │           │
  question ─────►│ /query  ───────────────────────────┘           │
                 │ /answer → prompt + top-k chunks ──┐            │
                 │                                    ▼           │
                 │                     llama-server (localhost:8081)
                 │                            GGUF, OpenAI API   │
                 └──────────────────────────────────────────────┘
```

Two models, one LLM:

| Component | Role | Model |
|---|---|---|
| Embedder | text → 384-d vector, retrieval only | `all-MiniLM-L6-v2` (ONNX, frozen) |
| LLM | writes the answer from retrieved context | GGUF served by llama.cpp |

The embedder is a bi-encoder, not a language model, and it is never fine-tuned.
The LLM is the fine-tuning target: swapping `LLM_MODEL_PATH` from the development
GGUF to the QLoRA-exported one changes nothing else in this service.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | llama.cpp readiness + collection stats |
| `POST` | `/ingest` | chunk, embed and upsert the corpus (or an upload) |
| `POST` | `/query` | retrieval only — ranked chunks, no generation |
| `POST` | `/answer` | retrieval + LLM answer with citations |
| `DELETE` | `/collection` | clear the vector collection |

`/query` is separated from `/answer` so retrieval quality can be evaluated
independently of generation quality.

### Quickest way to try it

`scripts/ask.py` is a thin client over the two endpoints — no dependencies, it
prints the answer, its citations and the excerpts that were cited:

```bash
cd services/llm-server
python scripts/ask.py          # interactive; type a question, Ctrl-D to quit
python scripts/ask.py "How often should heart failure treatment be uptitrated?"
```

Add `-k 6` for more excerpts or `--base-url` to point at another host.

### `/ingest`

```bash
# everything under CORPUS_DIR
curl -s -X POST http://localhost:8003/ingest \
  -H 'Content-Type: application/json' -d '{}'

# one uploaded file, without committing it
curl -s -X POST http://localhost:8003/ingest -F file=@guideline.md

# wipe and rebuild
curl -s -X POST http://localhost:8003/ingest \
  -H 'Content-Type: application/json' -d '{"reset": true}'
```

Re-ingesting a source replaces its chunks; other sources are untouched.

### `/query`

```bash
curl -s -X POST http://localhost:8003/query \
  -H 'Content-Type: application/json' \
  -d '{"question": "follow-up interval for moderate aortic stenosis", "k": 4}'
```

### `/answer`

```bash
curl -s -X POST http://localhost:8003/answer \
  -H 'Content-Type: application/json' \
  -d '{"question": "how often should moderate aortic stenosis be followed up?"}'
```

```json
{
  "question": "...",
  "answer": "Echocardiography is recommended every 6 to 12 months [1].",
  "grounded": true,
  "best_score": 0.71,
  "sources": [{"rank": 1, "source": "esc.md", "section": "Follow-up", "score": 0.71}]
}
```

When the best retrieved chunk scores below `RAG_MIN_SCORE`, the service returns
`"I don't know — this question is not covered by the guideline corpus."` **without calling the model** — a cheap, testable
guardrail that prevents answering from parametric memory. The floor is compared
against **cosine similarity**, so it stays interpretable even though ranking is
hybrid (see below).

## Retrieval

`/query` and `/answer` use two-stage retrieval:

1. **Dense** — MiniLM embeds the query; Chroma returns the nearest chunks. Each
   chunk is embedded together with its section title, because slide text alone is
   often too terse to match a natural-language question.
2. **Lexical** — BM25 over the whole corpus. Terms are compared on a short
   prefix so `uptitrate`/`Uptitration` and `dose`/`dosing` match.
3. **Fusion** — reciprocal-rank fusion of the two rankings.

Lexical search runs over the full corpus, not just the dense candidates: a chunk
that MiniLM misses can never be recovered by re-ranking a pool that already
excluded it. That is what fixed "how often should treatment be uptitrated",
which sat at dense rank 40 but lexical rank 1.

Measured on 16 questions against the ESC slide corpus (228 chunks):

| Metric | Dense only* | + title in embedding* | + lexical fusion |
|---|---|---|---|
| hit@1 | 0.33 | 0.42 | **0.75** |
| hit@3 | 0.50 | 0.67 | **0.875** |
| hit@5 | — | — | **1.00** |
| MRR | 0.46 | 0.55 | **0.841** |

\* measured on the 12-question development set used while tuning; the last
column is the larger, harder evaluation set. Full methodology, answer-level
results and failure analysis: [`EVALUATION.md`](EVALUATION.md).

The reported `score` is the cosine similarity; `hybrid_score` is the RRF value
that determines ordering. The relevance floor is compared against the best
cosine score in the retrieved set.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `LLM_BASE_URL` | `http://127.0.0.1:8081/v1` | llama.cpp OpenAI-compatible root |
| `LLM_MODEL_PATH` | `/models/qwen2.5-1.5b-instruct-q4_k_m.gguf` | GGUF loaded by `llama-server` |
| `LLM_CTX` | `4096` | context window |
| `LLM_THREADS` | `nproc` | CPU threads for inference |
| `CORPUS_DIR` | `/app/corpus` | default ingest source |
| `CHROMA_HOST` | empty | if set (e.g. `vector-db`) use Chroma HTTP server; otherwise use local `/data/chroma` |
| `CHROMA_PORT` | `8000` | port for Chroma server (used by `CHROMA_HOST`) |
| `CHROMA_DIR` | `/data/chroma` | fallback persistence dir when `CHROMA_HOST` is unset |
| `CHROMA_COLLECTION` | `esc_guidelines` | collection name |
| `EMBED_BACKEND` | `onnx` | `onnx` (production) or `hash` (offline/tests) |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `800` / `120` | chunking window |
| `RAG_TOP_K` | `4` | default number of retrieved chunks |
| `RAG_MIN_SCORE` | `0.40` | cosine-similarity floor for answering |

## Model files

GGUFs are git-ignored (~1 GB for the development model):

```bash
services/llm-server/scripts/fetch_model.sh
```

Any GGUF works — point `LLM_MODEL_PATH` at it. The fine-tuned QLoRA export drops
into the same directory and is selected by changing that one variable.

## Running

`llm-server` needs `vector-db` because Chroma is now served as a separate
HTTP service, persisted at `./services/vector-db/data/`. Compose starts
`vector-db` first (health-checked). If `CHROMA_HOST` is unset, `llm-server`
falls back to an embedded Chroma at `/data/chroma`.

```bash
docker compose up --build llm-server
```

`entrypoint.sh` starts `llama-server` on `127.0.0.1:8081` and gunicorn on `:8000`
inside the same container, so the LLM stays a hosted service without adding a
seventh container. If the model process dies, the container exits and Compose
restarts it.

### CUDA offload

The image is built on `ghcr.io/ggml-org/llama.cpp:server` (CPU). On a machine with
`nvidia-container-toolkit`:

1. change the `FROM` tag to `...:server-cuda`,
2. add `runtime: nvidia` (or `gpus: all`) to the `llm-server` service.

CPU inference for the 1.5B development model is already interactive
(a few seconds per answer), so this is optional.

## Tests

```bash
cd services/llm-server
python -m pytest tests -q
```

The suite uses `HashEmbedder` and a stub LLM client — no model download, no GPU,
no network. It covers chunking, idempotent re-ingestion, retrieval ranking, the
`/answer` citation contract, the insufficient-evidence short circuit, and the
503 path when llama.cpp is down.

Retrieval and answer quality are measured separately by
`scripts/evaluate_rag.py` against a running service; see
[`EVALUATION.md`](EVALUATION.md).

## Corpus

See [`corpus/README.md`](corpus/README.md) and [`corpus/SOURCES.md`](corpus/SOURCES.md)
for provenance and licensing. Nothing proprietary is committed: drop your
ESC-derived notes in `corpus/` (or upload them through `/ingest`).
