"""Prompt construction: citation numbering and leakage guards."""
from __future__ import annotations

from rag.prompts import (
    FORMAT_EXAMPLE,
    INSUFFICIENT_EVIDENCE,
    SYSTEM_PROMPT,
    build_messages,
    format_context,
)
from rag.store import Retrieved

CHUNKS = [
    Retrieved(text="Alpha body.", source="a.md", section="Alpha", chunk_index=0, score=0.9),
    Retrieved(text="Beta body.", source="b.md", section="Beta", chunk_index=1, score=0.8),
]


def test_format_context_numbers_chunks_from_one():
    context = format_context(CHUNKS)
    assert "Excerpt 1:\nAlpha body." in context
    assert "Excerpt 2:\nBeta body." in context


def test_format_context_hides_metadata_from_the_model():
    """Regression: header-shaped lines were copied verbatim into answers."""
    context = format_context(CHUNKS)
    assert "source=" not in context
    assert "section=" not in context


def test_format_context_respects_budget():
    # Each block is ~26 chars, so 30 fits the first chunk and not the second.
    context = format_context(CHUNKS, max_chars=30)
    assert "Excerpt 1:" in context
    assert "Excerpt 2:" not in context


def test_build_messages_contains_question_and_context():
    messages = build_messages("what is alpha?", CHUNKS)
    assert [m["role"] for m in messages] == ["system", "user"]
    assert "what is alpha?" in messages[1]["content"]
    assert "Alpha body." in messages[1]["content"]
    assert "Alpha body." not in messages[0]["content"]


def test_system_prompt_states_the_refusal_token():
    assert INSUFFICIENT_EVIDENCE in SYSTEM_PROMPT


def test_format_example_contains_no_clinical_vocabulary():
    """Regression: the exemplar once leaked 'severe aortic stenosis' into
    answers to unrelated questions. It must stay domain-neutral."""
    forbidden = [
        "aortic",
        "stenosis",
        "heart",
        "valve",
        "cardiac",
        "hypertension",
        "mmhg",
        "m/s",
        "mg",
    ]
    lowered = FORMAT_EXAMPLE.lower()
    for term in forbidden:
        assert term not in lowered, f"{term!r} leaked into the format example"


def test_user_message_ends_on_the_answer_cue():
    """Regression: instructions after the question made the model answer with
    only citation numbers."""
    messages = build_messages("what is alpha?", CHUNKS)
    assert messages[1]["content"].rstrip().endswith("Answer:")


def test_format_example_keeps_one_bracket_citation():
    assert "[1]" in FORMAT_EXAMPLE


def test_system_prompt_warns_that_class_markers_are_not_citations():
    """Regression: answers ended with "I B" (an ESC class/level marker) and
    therefore carried no citation, so they were reported as not grounded."""
    lowered = SYSTEM_PROMPT.lower()
    assert "class/level" in lowered
    assert "not citations" in lowered
