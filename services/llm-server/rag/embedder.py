"""Embedding backends.

``OnnxEmbedder`` is the production default: all-MiniLM-L6-v2 executed through
ONNX Runtime (shipped with ChromaDB), so no PyTorch/torchvision download is
needed in the image. The model files are baked into the image at build time.

``HashEmbedder`` is a dependency-free deterministic fallback used by the test
suite and by ``EMBED_BACKEND=hash`` when running offline.

The interface is intentionally tiny: ``store.py`` only calls ``encode`` and reads
``name``/``dimension``.
"""
from __future__ import annotations

import hashlib
import os
import re
from typing import Protocol, runtime_checkable

_TOKEN = re.compile(r"[a-z0-9]+")

DEFAULT_MODEL = "all-MiniLM-L6-v2"


@runtime_checkable
class Embedder(Protocol):
    """Minimal contract shared by every embedding backend."""

    name: str

    @property
    def dimension(self) -> int: ...

    def encode(self, texts: list[str]) -> list[list[float]]: ...


class HashEmbedder:
    """Hashing vectoriser: fast, deterministic, no downloads.

    Not semantically strong, but enough to exercise the retrieval wiring in
    tests and as an offline fallback.
    """

    name = "hash"

    def __init__(self, dimension: int = 384) -> None:
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def encode(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            vec = [0.0] * self._dimension
            for token in _TOKEN.findall(text.lower()):
                digest = hashlib.sha256(token.encode("utf-8")).digest()
                idx = int.from_bytes(digest[:4], "big") % self._dimension
                vec[idx] += 1.0
            norm = sum(v * v for v in vec) ** 0.5 or 1.0
            vectors.append([v / norm for v in vec])
        return vectors


class OnnxEmbedder:
    """all-MiniLM-L6-v2 via ChromaDB's bundled ONNX runtime."""

    name = DEFAULT_MODEL

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        if model_name != DEFAULT_MODEL:
            raise ValueError(f"only {DEFAULT_MODEL} is bundled with the ONNX backend")
        self._fn = None
        self._dimension = 384

    def _load(self):
        if self._fn is None:
            from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

            self._fn = ONNXMiniLM_L6_V2()
        return self._fn

    @property
    def dimension(self) -> int:
        return self._dimension

    def encode(self, texts: list[str]) -> list[list[float]]:
        vectors = self._load()(list(texts))
        return [[float(x) for x in v] for v in vectors]


def build_embedder(backend: str | None = None, model_name: str | None = None) -> Embedder:
    """Factory driven by ``EMBED_BACKEND`` / ``EMBED_MODEL`` environment vars."""
    backend = (backend or os.getenv("EMBED_BACKEND", "onnx")).lower()
    model_name = model_name or os.getenv("EMBED_MODEL", DEFAULT_MODEL)

    if backend in {"onnx", "onnx-minilm", "minilm"}:
        return OnnxEmbedder(model_name)
    if backend in {"hash", "offline"}:
        return HashEmbedder()
    raise ValueError(f"unknown embedding backend: {backend}")
