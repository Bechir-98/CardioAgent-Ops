"""Corpus loading and chunking for the ESC guideline RAG pipeline.

The corpus is plain text (Markdown, TXT or PDF). Markdown headings become the
``section`` metadata so retrieved chunks can be cited by section rather than by
raw character offset.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

SUPPORTED_SUFFIXES = {".md", ".markdown", ".txt", ".pdf"}

DEFAULT_CHUNK_SIZE = 800
DEFAULT_OVERLAP = 120

_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")


class CorpusError(RuntimeError):
    """Raised when the corpus cannot be read or parsed."""


@dataclass(frozen=True)
class Chunk:
    """A retrievable unit of text."""

    text: str
    source: str
    section: str
    chunk_index: int
    doc_hash: str
    chunk_id: str = field(default="")

    def __post_init__(self) -> None:
        if not self.chunk_id:
            object.__setattr__(self, "chunk_id", self.make_id())

    def make_id(self) -> str:
        """Stable id: same source/section/index always maps to the same id."""
        raw = f"{self.source}|{self.section}|{self.chunk_index}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    def embed_text(self) -> str:
        """Text used for the embedding (title carries retrieval signal)."""
        return f"{self.section}. {self.text}".strip()

    def as_metadata(self) -> dict:
        return {
            "source": self.source,
            "section": self.section,
            "chunk_index": self.chunk_index,
            "doc_hash": self.doc_hash,
            "char_count": len(self.text),
        }


def read_document(path: Path) -> str:
    """Return the plain text of ``path``, converting PDFs when needed."""
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise CorpusError(f"unsupported file type: {path.name}")
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise CorpusError("pypdf is required to ingest PDF files") from exc
        reader = PdfReader(str(path))
        return "\n\n".join((page.extract_text() or "") for page in reader.pages)
    return path.read_text(encoding="utf-8", errors="replace")


def split_sections(text: str) -> list[tuple[str, str]]:
    """Split a document on Markdown headings.

    Returns ``[(section_title, body), ...]``. Text before the first heading is
    attributed to ``"preamble"``.
    """
    sections: list[tuple[str, str]] = []
    title = "preamble"
    buffer: list[str] = []

    def flush() -> None:
        body = "\n".join(buffer).strip()
        if body:
            sections.append((title, body))
        buffer.clear()

    for line in text.splitlines():
        match = _HEADING.match(line.strip())
        if match:
            flush()
            title = match.group(2).strip() or "untitled"
        else:
            buffer.append(line)
    flush()
    return sections


def chunk_text(text: str, size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_OVERLAP) -> list[str]:
    """Split ``text`` into overlapping windows that prefer paragraph breaks."""
    text = text.strip()
    if not text:
        return []
    if size <= 0:
        raise CorpusError("chunk size must be positive")
    if not 0 <= overlap < size:
        raise CorpusError("overlap must be in [0, size)")

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    current = ""

    for paragraph in paragraphs:
        candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
        if len(candidate) <= size:
            current = candidate
            continue
        if current:
            chunks.append(current)
        # A single paragraph larger than the window is split, preferring the
        # last sentence boundary so a recommendation is not cut in half.
        if len(paragraph) > size:
            start = 0
            while start < len(paragraph):
                end = min(start + size, len(paragraph))
                if end < len(paragraph):
                    window = paragraph[start:end]
                    boundary = max(window.rfind(". "), window.rfind("; "), window.rfind("\n"))
                    if boundary > size // 2:
                        end = start + boundary + 1
                chunks.append(paragraph[start:end].strip())
                next_start = end - overlap
                start = next_start if next_start > start else end
            current = ""
        else:
            tail = current[-overlap:] if overlap and current else ""
            current = f"{tail}\n\n{paragraph}".strip() if tail else paragraph

    if current:
        chunks.append(current)

    # Merge a trailing stub into its predecessor so we never emit crumbs.
    if len(chunks) > 1 and len(chunks[-1]) < size // 4:
        chunks[-2] = f"{chunks[-2]}\n\n{chunks[-1]}".strip()
        chunks.pop()
    return [c for c in chunks if c]


def build_chunks(
    path: Path,
    root: Path | None = None,
    size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[Chunk]:
    """Read one document and return its chunks."""
    text = read_document(path)
    doc_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    source = str(path.relative_to(root)) if root else path.name

    chunks: list[Chunk] = []
    for section, body in split_sections(text):
        for piece in chunk_text(body, size=size, overlap=overlap):
            chunks.append(
                Chunk(
                    text=piece,
                    source=source,
                    section=section,
                    chunk_index=len(chunks),
                    doc_hash=doc_hash,
                )
            )
    return chunks


def load_corpus(
    directory: Path,
    size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[Chunk]:
    """Load and chunk every supported document under ``directory``.

    ``README*`` and ``SOURCES*`` files are treated as documentation, not corpus
    text.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise CorpusError(f"corpus directory not found: {directory}")

    files = sorted(
        p
        for p in directory.rglob("*")
        if p.is_file()
        and p.suffix.lower() in SUPPORTED_SUFFIXES
        and not p.name.lower().startswith(("readme", "sources"))
    )
    if not files:
        raise CorpusError(f"no .md/.txt/.pdf documents found in {directory}")

    chunks: list[Chunk] = []
    for path in files:
        chunks.extend(build_chunks(path, root=directory, size=size, overlap=overlap))
    return chunks
