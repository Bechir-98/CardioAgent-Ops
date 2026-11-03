"""Unit tests for the corpus loader and chunker (no external services needed)."""
from __future__ import annotations

from pathlib import Path

import pytest

from rag.corpus import (
    Chunk,
    CorpusError,
    build_chunks,
    chunk_text,
    load_corpus,
    split_sections,
)

SAMPLE = """# Aortic stenosis

Severe aortic stenosis is defined by a peak velocity of 4.0 m/s or more.

## Follow-up

Echocardiography is recommended every 6 to 12 months in moderate disease.

# Mitral regurgitation

Primary mitral regurgitation requires surgical evaluation when symptomatic.
"""


def test_split_sections_uses_headings():
    sections = split_sections(SAMPLE)
    titles = [title for title, _ in sections]
    assert titles == ["Aortic stenosis", "Follow-up", "Mitral regurgitation"]
    assert "4.0 m/s" in sections[0][1]


def test_split_sections_without_headings_uses_preamble():
    sections = split_sections("just a paragraph\n\nand another one")
    assert len(sections) == 1
    assert sections[0][0] == "preamble"


def test_chunk_text_respects_size_and_overlap():
    text = "\n\n".join(f"Paragraph number {i}." for i in range(60))
    chunks = chunk_text(text, size=120, overlap=20)
    assert len(chunks) > 1
    assert all(len(c) <= 120 + 20 for c in chunks)
    assert all(chunk.strip() for chunk in chunks)


def test_chunk_text_rejects_bad_parameters():
    with pytest.raises(CorpusError):
        chunk_text("some text", size=0)
    with pytest.raises(CorpusError):
        chunk_text("some text", size=100, overlap=100)


def test_chunk_ids_are_deterministic():
    a = Chunk(text="x", source="esc.md", section="Intro", chunk_index=0, doc_hash="d")
    b = Chunk(text="y", source="esc.md", section="Intro", chunk_index=0, doc_hash="d")
    assert a.chunk_id == b.chunk_id
    c = Chunk(text="x", source="esc.md", section="Intro", chunk_index=1, doc_hash="d")
    assert a.chunk_id != c.chunk_id


def test_build_chunks_carries_metadata(tmp_path: Path):
    doc = tmp_path / "esc.md"
    doc.write_text(SAMPLE, encoding="utf-8")
    chunks = build_chunks(doc, root=tmp_path)
    assert chunks
    assert {c.source for c in chunks} == {"esc.md"}
    assert {"Aortic stenosis", "Follow-up"} <= {c.section for c in chunks}
    assert all(c.doc_hash for c in chunks)


def test_load_corpus_reads_nested_files(tmp_path: Path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "a.md").write_text(SAMPLE, encoding="utf-8")
    (tmp_path / "sub" / "b.txt").write_text("tricuspid regurgitation notes", encoding="utf-8")
    chunks = load_corpus(tmp_path)
    assert {c.source for c in chunks} == {"a.md", "sub/b.txt"}


def test_load_corpus_skips_readme(tmp_path: Path):
    (tmp_path / "README.md").write_text("# how to add files\n\ninstructions", encoding="utf-8")
    (tmp_path / "esc.md").write_text(SAMPLE, encoding="utf-8")
    chunks = load_corpus(tmp_path)
    assert {c.source for c in chunks} == {"esc.md"}


def test_load_corpus_errors_on_empty_dir(tmp_path: Path):
    with pytest.raises(CorpusError):
        load_corpus(tmp_path)


def test_load_corpus_errors_on_missing_dir(tmp_path: Path):
    with pytest.raises(CorpusError):
        load_corpus(tmp_path / "nope")
