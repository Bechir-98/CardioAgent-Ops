"""ChromaDB-backed vector store.

Chroma can run embedded (``PersistentClient``) on the local disk, or as a
separate service (``HttpClient``) when ``CHROMA_HOST`` is set. The embedder is
injected at construction time, so the store never depends on Chroma's own
embedding machinery regardless of which mode is selected.
"""
from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from pathlib import Path

from .corpus import Chunk
from .embedder import Embedder

DEFAULT_COLLECTION = "esc_guidelines"
DEFAULT_PERSIST_DIR = "/data/chroma"
RRF_K = 60


@dataclass(frozen=True)
class Retrieved:
    """A chunk returned by a similarity search."""

    text: str
    source: str
    section: str
    chunk_index: int
    score: float  # cosine similarity (used by the relevance floor)
    dense_score: float = 0.0  # alias kept for older payloads
    hybrid_score: float = 0.0  # reciprocal-rank fusion score (ordering only)

    def as_dict(self, rank: int) -> dict:
        return {
            "rank": rank,
            "text": self.text,
            "source": self.source,
            "section": self.section,
            "chunk_index": self.chunk_index,
            "score": round(self.score, 4),
            "hybrid_score": round(self.hybrid_score, 6),
        }


class VectorStore:
    """Thin wrapper around a persistent Chroma collection."""

    def __init__(
        self,
        persist_dir: str | Path | None = None,
        collection: str = DEFAULT_COLLECTION,
        embedder: Embedder | None = None,
        chroma_host: str | None = None,
        chroma_port: int | None = None,
    ) -> None:
        if embedder is None:
            raise ValueError("VectorStore requires an embedder")

        import chromadb
        from chromadb.config import Settings

        host = chroma_host or os.getenv("CHROMA_HOST")
        port = chroma_port if chroma_port is not None else int(os.getenv("CHROMA_PORT", "8000"))

        self.embedder = embedder
        self.collection_name = collection
        self._corpus: tuple[list[str], list[str], list[dict]] | None = None

        if host:
            self.persist_dir = f"http://{host}:{port}"
            self._client = chromadb.HttpClient(
                host=host,
                port=port,
                settings=Settings(anonymized_telemetry=False, allow_reset=True),
            )
        else:
            self.persist_dir = str(persist_dir or DEFAULT_PERSIST_DIR)
            Path(self.persist_dir).mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(
                path=self.persist_dir,
                settings=Settings(anonymized_telemetry=False, allow_reset=True),
            )
        self._collection = self._client.get_or_create_collection(
            name=collection,
            metadata={"hnsw:space": "cosine", "embedder": embedder.name},
        )

    # -- introspection -----------------------------------------------------

    def count(self) -> int:
        return self._collection.count()

    def sources(self) -> list[str]:
        result = self._collection.get(include=["metadatas"])
        return sorted({m.get("source", "?") for m in result.get("metadatas") or []})

    def stats(self) -> dict:
        return {
            "collection": self.collection_name,
            "persist_dir": self.persist_dir,
            "embedder": self.embedder.name,
            "chunks": self.count(),
            "sources": self.sources(),
        }

    # -- mutation ----------------------------------------------------------

    def ingest(self, chunks: list[Chunk], batch_size: int = 64) -> dict:
        """Upsert ``chunks``, replacing all previous chunks from the same source.

        Deleting by source first keeps re-ingestion idempotent even when a
        document shrinks (which would otherwise leave orphaned tail chunks).
        """
        if not chunks:
            return {"chunks": 0, "sources": []}

        by_source: dict[str, list[Chunk]] = {}
        for chunk in chunks:
            by_source.setdefault(chunk.source, []).append(chunk)

        for source in by_source:
            self.delete_source(source)

        added = 0
        for batch in _batched(chunks, batch_size):
            # Embed the section title with the body: slide/heading text alone is
            # often too terse ("Vmax >= 4.0 m/s, mean gradient >= 40 mmHg") to
            # match a natural-language question. Stored documents stay raw so
            # citations quote the guideline and not the heading.
            embeddings = self.embedder.encode([c.embed_text() for c in batch])
            self._collection.upsert(
                ids=[c.chunk_id for c in batch],
                documents=[c.text for c in batch],
                metadatas=[c.as_metadata() for c in batch],
                embeddings=embeddings,
            )
            added += len(batch)
        self._corpus = None
        return {"chunks": added, "sources": sorted(by_source)}

    def delete_source(self, source: str) -> None:
        self._collection.delete(where={"source": source})
        self._corpus = None

    def reset(self) -> None:
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine", "embedder": self.embedder.name},
        )
        self._corpus = None

    # -- retrieval ---------------------------------------------------------

    def _documents(self) -> tuple[list[str], list[str], list[dict]]:
        """All (ids, documents, metadatas), cached until the collection changes.

        The lexical half of retrieval needs the whole corpus, not just the dense
        candidates: a chunk that MiniLM misses cannot be recovered by re-ranking
        a pool that already excluded it. Fine for the corpus sizes used here
        (hundreds to a few thousand chunks); swap for a real BM25 index if this
        ever grows past ~50k chunks.
        """
        if self._corpus is None:
            result = self._collection.get(include=["documents", "metadatas"])
            self._corpus = (
                list(result.get("ids") or []),
                list(result.get("documents") or []),
                list(result.get("metadatas") or []),
            )
        return self._corpus

    def query(self, question: str, k: int = 4) -> list[Retrieved]:
        """Hybrid retrieval: dense candidates fused with lexical candidates.

        Pure cosine similarity dilutes long recommendation tables (one chunk can
        hold five unrelated class-I recommendations) and misses morphology
        ("uptitrate" vs "Uptitration"). Lexical search over the full corpus fixes
        both; reciprocal-rank fusion combines the two rankings without needing
        their scores to share a scale.
        """
        if self.count() == 0:
            return []
        k = max(1, min(k, self.count()))
        ids, documents, metadatas = self._documents()

        dense_result = self._collection.query(
            query_embeddings=self.embedder.encode([question]),
            n_results=min(max(k * 6, 30), self.count()),
            include=["documents", "metadatas", "distances"],
        )
        dense_ids = (dense_result.get("ids") or [[]])[0]
        dense_scores = [1.0 - float(d) for d in (dense_result.get("distances") or [[]])[0]]

        lexical = bm25_scores(question, documents)
        lexical_order = sorted(range(len(lexical)), key=lambda i: lexical[i], reverse=True)
        lexical_ids = [ids[i] for i in lexical_order[: max(k * 6, 30)] if lexical[i] > 0]

        rrf: dict[str, float] = {}
        for rank, doc_id in enumerate(dense_ids, start=1):
            rrf[doc_id] = rrf.get(doc_id, 0.0) + 1.0 / (RRF_K + rank)
        for rank, doc_id in enumerate(lexical_ids, start=1):
            rrf[doc_id] = rrf.get(doc_id, 0.0) + 1.0 / (RRF_K + rank)

        by_id = {doc_id: (doc, meta) for doc_id, doc, meta in zip(ids, documents, metadatas)}
        dense_by_id = dict(zip(dense_ids, dense_scores))

        hits: list[Retrieved] = []
        for doc_id in sorted(rrf, key=lambda i: rrf[i], reverse=True)[:k]:
            doc, meta = by_id[doc_id]
            hits.append(
                Retrieved(
                    text=doc,
                    source=meta.get("source", "?"),
                    section=meta.get("section", "?"),
                    chunk_index=int(meta.get("chunk_index", -1)),
                    # Reported score stays cosine so the relevance floor keeps a
                    # stable meaning. A purely lexical hit has no cosine value in
                    # this pool; it is reported as 0 and gated separately.
                    score=dense_by_id.get(doc_id, 0.0),
                    hybrid_score=rrf[doc_id],
                )
            )
        return hits


_STOPWORDS = {
    "the", "a", "an", "of", "for", "in", "on", "to", "and", "or", "is", "are",
    "be", "with", "when", "what", "which", "how", "should", "can", "may", "it",
    "this", "that", "at", "by", "as", "from", "patients", "patient",
}

_TOKEN = re.compile(r"[a-z0-9]+")


def _stem(token: str) -> str:
    """Crude suffix stripping so uptitrate/uptitration and dosing/dose match."""
    for suffix in ("ations", "ation", "ments", "ment", "ings", "ing", "ies", "ed", "es", "s"):
        if len(token) > len(suffix) + 3 and token.endswith(suffix):
            token = token[: -len(suffix)]
            break
    # uptitrate -> uptitrat, dose -> dos: matches the stripped noun form.
    if len(token) > 4 and token.endswith("e"):
        token = token[:-1]
    return token


# Matching key length. Suffix stripping cannot align every pair
# (uptitration -> "uptitr" vs uptitrate -> "uptitrat"), so terms are compared on
# a short prefix. Five characters keeps treat/treatment and uptitrate/uptitration
# together at the cost of occasional false friends, which only affect re-ranking.
_PREFIX = 5


def _key(token: str) -> str:
    stemmed = _stem(token)
    return stemmed[:_PREFIX] if len(stemmed) > _PREFIX else stemmed


def _terms(text: str) -> list[str]:
    return [_key(t) for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS and len(t) > 2]


def bm25_scores(question: str, documents: list[str], k1: float = 1.2, b: float = 0.75) -> list[float]:
    """BM25 over the candidate pool, normalised to [0, 1]."""
    query_terms = set(_terms(question))
    if not query_terms or not documents:
        return [0.0] * len(documents)

    docs = [_terms(doc) for doc in documents]
    n = len(docs)
    avg_len = sum(len(d) for d in docs) / n or 1.0

    df: dict[str, int] = {}
    for doc in docs:
        for term in set(doc):
            df[term] = df.get(term, 0) + 1

    scores: list[float] = []
    for doc in docs:
        tf: dict[str, int] = {}
        for term in doc:
            tf[term] = tf.get(term, 0) + 1
        score = 0.0
        for term in query_terms:
            if term not in tf:
                continue
            idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
            freq = tf[term]
            score += idf * (freq * (k1 + 1)) / (freq + k1 * (1 - b + b * len(doc) / avg_len))
        scores.append(score)

    peak = max(scores) or 1.0
    return [s / peak for s in scores]


def _batched(items: list, size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]
