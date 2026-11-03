"""Vector store behaviour against a temporary Chroma instance."""
from __future__ import annotations

from pathlib import Path

import pytest

from rag.corpus import Chunk
from rag.embedder import HashEmbedder
from rag.store import VectorStore

pytest.importorskip("chromadb")


def make_chunk(text: str, source: str, section: str, index: int) -> Chunk:
    return Chunk(text=text, source=source, section=section, chunk_index=index, doc_hash="d")


@pytest.fixture()
def store(tmp_path: Path) -> VectorStore:
    return VectorStore(
        persist_dir=tmp_path / "chroma",
        collection="test_esc",
        embedder=HashEmbedder(),
    )


def test_ingest_and_count(store: VectorStore):
    chunks = [
        make_chunk("aortic stenosis peak velocity", "esc.md", "AS", 0),
        make_chunk("mitral regurgitation surgery", "esc.md", "MR", 1),
    ]
    result = store.ingest(chunks)
    assert result["chunks"] == 2
    assert store.count() == 2
    assert store.sources() == ["esc.md"]


def test_reingest_same_source_is_idempotent(store: VectorStore):
    first = [make_chunk("aortic stenosis", "esc.md", "AS", 0)]
    store.ingest(first)
    store.ingest(first)
    assert store.count() == 1


def test_reingest_shrinking_source_removes_orphans(store: VectorStore):
    store.ingest(
        [
            make_chunk("chunk zero about stenosis", "esc.md", "AS", 0),
            make_chunk("chunk one about regurgitation", "esc.md", "MR", 1),
        ]
    )
    store.ingest([make_chunk("only chunk left about stenosis", "esc.md", "AS", 0)])
    assert store.count() == 1


def test_query_returns_ranked_hits(store: VectorStore):
    store.ingest(
        [
            make_chunk("aortic stenosis echocardiography follow up", "esc.md", "AS", 0),
            make_chunk("atrial fibrillation anticoagulation", "esc.md", "AF", 1),
        ]
    )
    hits = store.query("aortic stenosis follow up", k=2)
    assert len(hits) == 2
    assert "stenosis" in hits[0].text
    assert hits[0].score >= hits[1].score


def test_query_on_empty_store_returns_nothing(store: VectorStore):
    assert store.query("anything", k=3) == []


def test_reset_clears_collection(store: VectorStore):
    store.ingest([make_chunk("text", "esc.md", "S", 0)])
    store.reset()
    assert store.count() == 0


def test_stats_reports_metadata(store: VectorStore):
    store.ingest([make_chunk("text", "esc.md", "S", 0)])
    stats = store.stats()
    assert stats["collection"] == "test_esc"
    assert stats["embedder"] == "hash"
    assert stats["chunks"] == 1
