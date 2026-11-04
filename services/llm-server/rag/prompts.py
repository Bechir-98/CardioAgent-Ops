"""Prompt construction for retrieval-augmented answering."""
from __future__ import annotations

from .store import Retrieved

INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
DONT_KNOW = "I don't know — this question is not covered by the guideline corpus."

SYSTEM_PROMPT = (
    "You are CardioAgent, a cardiology decision-support assistant running fully on-premise.\n"
    "You answer ONLY from the numbered guideline excerpts in the user message.\n"
    "\n"
    "Rules:\n"
    f"1. If the excerpts do not contain the answer, output only: {INSUFFICIENT_EVIDENCE}\n"
    "2. Otherwise answer in 1-3 sentences and end with the bracket number of every excerpt\n"
    "   you used, for example [1] or [2][3].\n"
    "3. Quote numbers, thresholds and doses exactly as they appear in the excerpts.\n"
    "4. Never use outside knowledge. Never invent a threshold, dose or interval.\n"
    "5. Do not repeat the question. Do not use markdown. Do not add a disclaimer.\n"
    "6. Answer in the language of the question.\n"
    "7. The class/level markers in the excerpts (I A, IIa B, IIb C) are not citations.\n"
    "   Always finish with the bracket number of the excerpt you used, e.g. [1].\n"
    "This is a research prototype, not a medical device."
)

# Format demonstration only. Deliberately contains no cardiology content, so a
# small model cannot copy it into an answer about a real clinical question.
FORMAT_EXAMPLE = (
    "Format example (placeholder content, ignore its meaning):\n"
    "Question: What is the registry identifier?\n"
    "Answer: The registry identifier is ZX-7 [1].\n"
)

MAX_CONTEXT_CHARS = 6000

# Instructions live in the system message; the user turn ends on a bare
# "Answer:" cue. Appending instructions *after* the question made small models
# emit only the citation numbers.
ANSWER_CUE = "Answer:"


def format_context(chunks: list[Retrieved], max_chars: int = MAX_CONTEXT_CHARS) -> str:
    """Render retrieved chunks as a numbered list.

    Deliberately no ``source=`` / ``section=`` prefix: small models copy any
    header-looking line straight into the answer. The mapping from bracket
    number to file and section is returned to the caller as ``sources``, so the
    model never needs to see it.
    """
    lines: list[str] = []
    used = 0
    for rank, chunk in enumerate(chunks, start=1):
        block = f"Excerpt {rank}:\n{chunk.text.strip()}\n"
        if used + len(block) > max_chars:
            break
        lines.append(block)
        used += len(block)
    return "\n".join(lines).strip()


def build_messages(question: str, chunks: list[Retrieved]) -> list[dict]:
    """Build the chat payload sent to llama.cpp."""
    context = format_context(chunks) or "(no excerpts retrieved)"
    user = (
        f"{FORMAT_EXAMPLE}\n"
        f"Guideline excerpts:\n\n{context}\n\n"
        f"Question: {question.strip()}\n\n"
        f"{ANSWER_CUE}"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
