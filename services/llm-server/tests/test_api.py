"""HTTP contract tests for /health, /ingest, /query and /answer."""
from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pytest

from main import Config, Services, create_app
from rag.embedder import HashEmbedder
from rag.llm import LlmUnavailable
from rag.prompts import INSUFFICIENT_EVIDENCE, DONT_KNOW
from rag.store import VectorStore

pytest.importorskip("chromadb")

CORPUS = """# Aortic stenosis

Severe aortic stenosis is defined by a peak velocity of 4.0 m/s or more.

## Follow-up

Echocardiography is recommended every 6 to 12 months in moderate aortic stenosis.
"""


class FakeLlm:
    """Stands in for llama.cpp; records the prompts it was given."""

    def __init__(self, reply: str = "Echocardiography every 6-12 months [1].", fail: bool = False) -> None:
        self.reply = reply
        self.fail = fail
        self.calls: list[list[dict]] = []

    def health(self) -> dict:
        return {"ready": not self.fail, "state": "ok" if not self.fail else "unreachable"}

    def chat(self, messages, temperature=0.2, max_tokens=512, stop=None) -> str:
        if self.fail:
            raise LlmUnavailable("connection refused")
        self.calls.append(messages)
        return self.reply


def build_client(tmp_path: Path, name: str = "t", min_score: float = 0.0, fake: FakeLlm | None = None):
    corpus = tmp_path / "corpus"
    corpus.mkdir(exist_ok=True)
    (corpus / "esc.md").write_text(CORPUS, encoding="utf-8")

    config = Config(
        corpus_dir=corpus,
        chroma_dir=str(tmp_path / f"chroma-{name}"),
        collection=f"api_{name}",
        embed_backend="hash",
        min_score=min_score,
    )
    services = Services(config)
    services._store = VectorStore(
        persist_dir=tmp_path / f"chroma-{name}",
        collection=f"api_{name}",
        embedder=HashEmbedder(),
    )
    services._llm = fake or FakeLlm()

    app = create_app(config, services)
    app.config.update(TESTING=True)
    return app.test_client()


@pytest.fixture()
def client(tmp_path: Path):
    with build_client(tmp_path) as test_client:
        yield test_client


def test_health_reports_llm_and_rag(client):
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["service"] == "llm-server"
    assert payload["status"] == "ok"
    assert payload["llm"]["ready"] is True
    assert payload["rag"]["chunks"] == 0


def test_ingest_from_corpus_dir_then_query(client):
    ingest = client.post("/ingest", json={})
    assert ingest.status_code == 200
    body = ingest.get_json()
    assert body["chunks"] > 0
    assert body["files"] == ["esc.md"]

    query = client.post("/query", json={"question": "aortic stenosis follow up", "k": 2})
    assert query.status_code == 200
    hits = query.get_json()["hits"]
    assert hits
    assert hits[0]["rank"] == 1
    assert "stenosis" in hits[0]["text"].lower()


def test_ingest_upload(client):
    data = {"file": (BytesIO(CORPUS.encode()), "uploaded.md")}
    response = client.post("/ingest", data=data, content_type="multipart/form-data")
    assert response.status_code == 200
    assert response.get_json()["files"] == ["uploaded.md"]


def test_ingest_rejects_missing_corpus(client, tmp_path):
    response = client.post("/ingest", json={"path": str(tmp_path / "does-not-exist")})
    assert response.status_code == 400


def test_query_validates_payload(client):
    assert client.post("/query", json={"question": "x"}).status_code == 400
    assert client.post("/query", json={}).status_code == 400


def test_delete_collection_clears_chunks(client):
    client.post("/ingest", json={})
    response = client.delete("/collection")
    assert response.status_code == 200
    assert response.get_json()["chunks"] == 0
    assert client.post("/query", json={"question": "aortic stenosis"}).get_json()["hits"] == []


def test_answer_is_grounded_and_cites_sources(client):
    client.post("/ingest", json={})
    response = client.post("/answer", json={"question": "how often to follow up aortic stenosis?"})
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["grounded"] is True
    assert "[1]" in payload["answer"]
    assert payload["citations"] == [1]
    assert payload["sources"]


def test_extract_citations_ignores_out_of_range_ranks():
    from main import extract_citations

    assert extract_citations("see [1] and [3] and [1] again", 3) == [1, 3]
    assert extract_citations("no citations here", 3) == []
    assert extract_citations("[9] out of range", 3) == []


def test_answer_short_circuits_when_nothing_is_relevant(tmp_path: Path):
    fake = FakeLlm()
    with build_client(tmp_path, name="strict", min_score=0.99, fake=fake) as c:
        c.post("/ingest", json={})
        response = c.post("/answer", json={"question": "zebra migration patterns"})
        assert response.status_code == 200
        payload = response.get_json()
        assert payload["answer"] == DONT_KNOW
        assert payload["grounded"] is False
        assert payload["citations"] == []
        assert payload["sources"] == []
        assert fake.calls == []  # the model must not be called without evidence


def test_answer_without_citation_is_not_grounded(tmp_path: Path):
    uncited = FakeLlm(reply="Echocardiography is recommended every 6-12 months.")
    with build_client(tmp_path, name="uncited", fake=uncited) as c:
        c.post("/ingest", json={})
        payload = c.post("/answer", json={"question": "aortic stenosis follow-up"}).get_json()
        assert payload["citations"] == []
        assert payload["grounded"] is False


def test_answer_is_case_insensitive_to_the_refusal_token(tmp_path: Path):
    chatty = FakeLlm(reply="Insufficient_Evidence: the corpus does not say.")
    with build_client(tmp_path, name="chatty", fake=chatty) as c:
        c.post("/ingest", json={})
        payload = c.post("/answer", json={"question": "aortic stenosis follow-up"}).get_json()
        assert payload["grounded"] is False


def test_answer_returns_503_when_backend_is_down(tmp_path: Path):
    with build_client(tmp_path, name="down", fake=FakeLlm(fail=True)) as c:
        c.post("/ingest", json={})
        response = c.post("/answer", json={"question": "aortic stenosis"})
        assert response.status_code == 503
        assert "unavailable" in response.get_json()["error"]
