"""Prompt construction for retrieval-augmented answering."""
from __future__ import annotations

from .store import Retrieved

INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"

SYSTEM_PROMPT = (
    "You are CardioAgent, a cardiology decision-support assistant running fully on-premise.\n"
    "Answer using ONLY the numbered guideline excerpts provided in the user message.\n"
    "Rules:\n"
    "1. Cite every factual claim with the bracket number of the excerpt it came from, e.g. [1].\n"
    "2. If the excerpts do not contain the answer, reply exactly: "
    f"{INSUFFICIENT_EVIDENCE} and name what is missing.\n"
    "3. Never invent thresholds, drug doses, or follow-up intervals.\n"
    "4. Be concise: at most 150 words, plain prose, no markdown tables.\n"
    "5. Answer in the same language as the question.\n"
    "6. This is a research prototype, not a medical device; do not give individual patient advice."
)

MAX_CONTEXT_CHARS = 6000

_EXAMPLE = (
    "Worked example (format only, not a real excerpt):\n"
    "[1] source=example.md | section=Definition\n"
    "Severe aortic stenosis is defined by a peak velocity of 4.0 m/s or more.\n\n"
    "Question: What defines severe aortic stenosis?\n"
    "Answer: Severe aortic stenosis is defined by a peak velocity of 4.0 m/s or more [1].\n"
)


def format_context(chunks: list[Retrieved], max_chars: int = MAX_CONTEXT_CHARS) -> str:
    """Render retrieved chunks as a numbered, citable block."""
    lines: list[str] = []
    used = 0
    for rank, chunk in enumerate(chunks, start=1):
        header = f"[{rank}] source={chunk.source} | section={chunk.section}"
        body = chunk.text.strip()
        block = f"{header}\n{body}\n"
        if used + len(block) > max_chars:
            break
        lines.append(block)
        used += len(block)
    return "\n".join(lines).strip()


def build_messages(question: str, chunks: list[Retrieved]) -> list[dict]:
    """Build the chat payload sent to llama.cpp."""
    context = format_context(chunks) or "(no excerpts retrieved)"
    user = (
        f"{_EXAMPLE}\n"
        f"Guideline excerpts:\n\n{context}\n\n"
        f"Question: {question.strip()}\n\n"
        "Answer using only the excerpts above, with bracket citations."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
