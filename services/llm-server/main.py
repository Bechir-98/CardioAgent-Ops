"""CardioAgent-Ops — llm-server.

Hosts the local GGUF model (llama.cpp, OpenAI-compatible API) and exposes the
retrieval-augmented generation layer over the ESC guideline corpus.
"""
from __future__ import annotations

import logging
import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from flask import Flask, jsonify, request
from pydantic import BaseModel, Field, ValidationError

from rag import (
    CorpusError,
    INSUFFICIENT_EVIDENCE,
    LlamaCppClient,
    LlmError,
    LlmUnavailable,
    VectorStore,
    build_chunks,
    build_embedder,
    build_messages,
    load_corpus,
)

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("llm-server")


@dataclass
class Config:
    corpus_dir: Path = field(default_factory=lambda: Path(os.getenv("CORPUS_DIR", "/app/corpus")))
    chroma_dir: str = os.getenv("CHROMA_DIR", "/data/chroma")
    collection: str = os.getenv("CHROMA_COLLECTION", "esc_guidelines")
    embed_backend: str = os.getenv("EMBED_BACKEND", "onnx")
    embed_model: str = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "800"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "120"))
    top_k: int = int(os.getenv("RAG_TOP_K", "4"))
    min_score: float = float(os.getenv("RAG_MIN_SCORE", "0.15"))


# -- request schemas ------------------------------------------------------

class QuestionRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    k: int = Field(default=4, ge=1, le=20)


class AnswerRequest(QuestionRequest):
    temperature: float = Field(default=0.2, ge=0.0, le=1.5)
    max_tokens: int = Field(default=512, ge=16, le=2048)


class IngestRequest(BaseModel):
    path: str | None = None
    reset: bool = False
    chunk_size: int | None = Field(default=None, ge=120, le=4000)
    overlap: int | None = Field(default=None, ge=0, le=1000)


_CITATION = re.compile(r"\[(\d{1,2})\]")


def extract_citations(answer: str, n_sources: int) -> list[int]:
    """Return the source ranks the model actually cited, in order, deduplicated."""
    seen: list[int] = []
    for match in _CITATION.finditer(answer or ""):
        rank = int(match.group(1))
        if 1 <= rank <= n_sources and rank not in seen:
            seen.append(rank)
    return seen


# -- lazy singletons ------------------------------------------------------

class Services:
    """Holds the (expensive) embedder/store/client, created on first use."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self._store: VectorStore | None = None
        self._llm: LlamaCppClient | None = None

    @property
    def llm(self) -> LlamaCppClient:
        if self._llm is None:
            self._llm = LlamaCppClient()
        return self._llm

    @property
    def store(self) -> VectorStore:
        if self._store is None:
            self._store = VectorStore(
                persist_dir=self.config.chroma_dir,
                collection=self.config.collection,
                embedder=build_embedder(self.config.embed_backend, self.config.embed_model),
            )
        return self._store


def create_app(config: Config | None = None, services: Services | None = None) -> Flask:
    config = config or Config()
    services = services or Services(config)
    app = Flask(__name__)

    # -- helpers -----------------------------------------------------------

    def store_snapshot() -> dict:
        try:
            return {"collection": services.store.stats()}
        except Exception as exc:  # pragma: no cover - defensive
            return {"collection": {"error": str(exc)}}

    def parse(model_cls, payload):
        try:
            return model_cls.model_validate(payload), None
        except ValidationError as exc:
            return None, (jsonify(error="invalid request", detail=exc.errors()), 400)

    # -- health ------------------------------------------------------------

    @app.get("/")
    @app.get("/health")
    def health():
        llm = services.llm.health()
        try:
            store = services.store.stats()
        except Exception as exc:  # pragma: no cover - defensive
            store = {"error": str(exc)}
        status = "ok" if llm.get("ready") else "degraded"
        return jsonify(
            service="llm-server",
            status=status,
            llm=llm,
            rag=store,
        )

    # -- ingestion ---------------------------------------------------------

    def _ingest_chunks(chunks, reset: bool):
        store = services.store
        if reset:
            store.reset()
        return store.ingest(chunks)

    @app.post("/ingest")
    def ingest():
        upload = request.files.get("file")

        if upload:
            try:
                payload = IngestRequest.model_validate(request.form.to_dict() or {})
            except ValidationError as exc:
                return jsonify(error="invalid request", detail=exc.errors()), 400
            suffix = Path(upload.filename or "").suffix.lower()
            with tempfile.TemporaryDirectory() as tmp:
                target = Path(tmp) / (upload.filename or f"upload{suffix}")
                upload.save(target)
                try:
                    chunks = build_chunks(
                        target,
                        root=Path(tmp),
                        size=payload.chunk_size or config.chunk_size,
                        overlap=payload.overlap if payload.overlap is not None else config.chunk_overlap,
                    )
                except CorpusError as exc:
                    return jsonify(error=str(exc)), 400
                result = _ingest_chunks(chunks, payload.reset)
            result["files"] = [upload.filename]
            result.update(store_snapshot())
            return jsonify(result)

        req, error = parse(IngestRequest, request.get_json(silent=True) or {})
        if error:
            return error

        root = Path(req.path) if req.path else config.corpus_dir
        try:
            chunks = load_corpus(
                root,
                size=req.chunk_size or config.chunk_size,
                overlap=req.overlap if req.overlap is not None else config.chunk_overlap,
            )
        except CorpusError as exc:
            return jsonify(error=str(exc), hint="place .md/.txt/.pdf files in the corpus dir"), 400

        result = _ingest_chunks(chunks, req.reset)
        result["files"] = sorted({c.source for c in chunks})
        result.update(store_snapshot())
        return jsonify(result)

    # -- retrieval ---------------------------------------------------------

    @app.post("/query")
    def query():
        req, error = parse(QuestionRequest, request.get_json(silent=True) or {})
        if error:
            return error
        hits = services.store.query(req.question, k=req.k)
        return jsonify(
            question=req.question,
            k=req.k,
            hits=[hit.as_dict(rank) for rank, hit in enumerate(hits, start=1)],
        )

    # -- retrieval + generation -------------------------------------------

    @app.post("/answer")
    def answer():
        req, error = parse(AnswerRequest, request.get_json(silent=True) or {})
        if error:
            return error

        hits = services.store.query(req.question, k=req.k)
        best = hits[0].score if hits else 0.0

        if not hits or best < config.min_score:
            return jsonify(
                question=req.question,
                answer=INSUFFICIENT_EVIDENCE,
                reason="no retrieved excerpt passed the relevance threshold",
                best_score=round(best, 4),
                min_score=config.min_score,
                sources=[],
            )

        messages = build_messages(req.question, hits)
        try:
            text = services.llm.chat(
                messages,
                temperature=req.temperature,
                max_tokens=req.max_tokens,
            )
        except LlmUnavailable as exc:
            return jsonify(error="llm backend unavailable", detail=str(exc)), 503
        except LlmError as exc:
            return jsonify(error="llm backend error", detail=str(exc)), 502

        insufficient = INSUFFICIENT_EVIDENCE in text
        return jsonify(
            question=req.question,
            answer=text.strip(),
            grounded=not insufficient,
            citations=extract_citations(text, len(hits)),
            best_score=round(best, 4),
            sources=[hit.as_dict(rank) for rank, hit in enumerate(hits, start=1)],
        )

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8000")), threaded=True)
