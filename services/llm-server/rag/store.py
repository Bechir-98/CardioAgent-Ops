"""ChromaDB-backed vector store.

Chroma runs embedded (``PersistentClient``) inside the service container: one
process, one on-disk directory, no extra network hop. Embeddings are computed by
the injected :class:`rag.embedder.Embedder` and passed in explicitly, so the
store never depends on Chroma's own embedding machinery.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .corpus import Chunk
from .embedder import Embedder

DEFAULT_COLLECTION = "esc_guidelines"
DEFAULT_PERSIST_DIR = "/data/chroma"


@dataclass(frozen=True)
class Retrieved:
    """A chunk returned by a similarity search."""

    text: str
    source: str
    section: str
    chunk_index: int
    score: float  # 1 - cosine distance, so higher is better

    def as_dict(self, rank: int) -> dict:
        return {
            "rank": rank,
            "text": self.text,
            "source": self.source,
            "section": self.section,
            "chunk_index": self.chunk_index,
            "score": round(self.score, 4),
        }


class VectorStore:
    """Thin wrapper around a persistent Chroma collection."""

    def __init__(
        self,
        persist_dir: str | Path | None = None,
        collection: str = DEFAULT_COLLECTION,
        embedder: Embedder | None = None,
    ) -> None:
        if embedder is None:
            raise ValueError("VectorStore requires an embedder")

        import chromadb
        from chromadb.config import Settings

        self.embedder = embedder
        self.persist_dir = str(persist_dir or DEFAULT_PERSIST_DIR)
        self.collection_name = collection

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
            embeddings = self.embedder.encode([c.text for c in batch])
            self._collection.upsert(
                ids=[c.chunk_id for c in batch],
                documents=[c.text for c in batch],
                metadatas=[c.as_metadata() for c in batch],
                embeddings=embeddings,
            )
            added += len(batch)
        return {"chunks": added, "sources": sorted(by_source)}

    def delete_source(self, source: str) -> None:
        self._collection.delete(where={"source": source})

    def reset(self) -> None:
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine", "embedder": self.embedder.name},
        )

    # -- retrieval ---------------------------------------------------------

    def query(self, question: str, k: int = 4) -> list[Retrieved]:
        if self.count() == 0:
            return []
        k = max(1, min(k, self.count()))
        result = self._collection.query(
            query_embeddings=self.embedder.encode([question]),
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]

        hits: list[Retrieved] = []
        for text, meta, distance in zip(documents, metadatas, distances):
            hits.append(
                Retrieved(
                    text=text,
                    source=meta.get("source", "?"),
                    section=meta.get("section", "?"),
                    chunk_index=int(meta.get("chunk_index", -1)),
                    score=1.0 - float(distance),
                )
            )
        return hits


def _batched(items: list, size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]
