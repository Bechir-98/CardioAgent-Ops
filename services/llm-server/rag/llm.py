"""Client for the llama.cpp server's OpenAI-compatible API.

The llama.cpp ``llama-server`` binary exposes ``/v1/chat/completions`` and
``/health``. This client is deliberately small: it owns the HTTP contract,
timeouts and error translation, nothing else.
"""
from __future__ import annotations

import os

import requests

DEFAULT_BASE_URL = "http://127.0.0.1:8081/v1"
DEFAULT_MODEL = "local-gguf"


class LlmError(RuntimeError):
    """Upstream model returned an error."""


class LlmUnavailable(LlmError):
    """llama.cpp is not reachable (still loading, crashed, not started)."""


class LlamaCppClient:
    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
        api_key: str | None = None,
    ) -> None:
        self.base_url = (base_url or os.getenv("LLM_BASE_URL", DEFAULT_BASE_URL)).rstrip("/")
        self.model = model or os.getenv("LLM_MODEL_NAME", DEFAULT_MODEL)
        self.timeout = float(timeout or os.getenv("LLM_TIMEOUT", "120"))
        self.api_key = api_key or os.getenv("LLM_API_KEY", "no-key")

    # -- health ------------------------------------------------------------

    @property
    def _root(self) -> str:
        return self.base_url[: -len("/v1")] if self.base_url.endswith("/v1") else self.base_url

    def health(self) -> dict:
        """Return the backend status without raising."""
        try:
            response = requests.get(f"{self._root}/health", timeout=5)
        except requests.RequestException as exc:
            return {"ready": False, "state": "unreachable", "detail": str(exc)}
        try:
            payload = response.json()
        except ValueError:
            payload = {"raw": response.text[:200]}
        ready = response.status_code == 200 and payload.get("status") in (None, "ok")
        return {"ready": ready, "state": payload.get("status", "ok" if ready else "loading"), **payload}

    def is_ready(self) -> bool:
        return bool(self.health().get("ready"))

    # -- inference ---------------------------------------------------------

    def chat(
        self,
        messages: list[dict],
        temperature: float = 0.2,
        max_tokens: int = 512,
        stop: list[str] | None = None,
    ) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if stop:
            payload["stop"] = stop

        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            raise LlmUnavailable(f"llama.cpp unreachable at {self.base_url}: {exc}") from exc

        if response.status_code >= 400:
            raise LlmError(f"llama.cpp returned {response.status_code}: {response.text[:300]}")

        try:
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError) as exc:
            raise LlmError(f"malformed llama.cpp response: {exc}") from exc
