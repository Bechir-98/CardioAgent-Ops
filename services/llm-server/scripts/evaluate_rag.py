#!/usr/bin/env python3
"""Evaluate the RAG endpoints end to end.

Two independent measurements:

*Retrieval* -- a question is a hit at rank k when one of the top-k chunks
contains every phrase in ``evidence``. The phrases are answer-bearing strings
("4.0 m/s", "uptitration") rather than section names, because one slide often
holds several unrelated recommendations, so "same section" is too generous.

*Answering* -- whether the service grounded the answer, whether every citation
index is in range, and whether out-of-domain questions were refused. Factual
correctness is not scored automatically: answers are printed for adjudication,
since a lexical check cannot tell "first-line is an ACE-I or ARNI" from
"first-line is ivabradine" (both mention the drug classes somewhere).

Usage
-----
    python scripts/evaluate_rag.py                    # against localhost:8003
    python scripts/evaluate_rag.py --base-url http://host:8003 --json out.json
"""
from __future__ import annotations

import argparse
import json
import re
import urllib.request
from dataclasses import dataclass, field


@dataclass
class Case:
    question: str
    domain: str
    evidence: list[str]  # all must appear in a retrieved chunk for a hit
    answer_terms: list[list[str]] = field(default_factory=list)  # alternatives per group


# In-domain questions whose answers exist in the ESC slide corpus, and
# out-of-domain controls that must be refused.
CASES: list[Case] = [
    Case("What are the echocardiographic criteria for severe aortic stenosis?", "VHD", ["4.0 m/s"], [["4.0"], ["40"]]),
    Case("When is aortic valve replacement indicated in asymptomatic severe aortic stenosis?", "VHD", ["asymptomatic"], [["asymptomatic"]]),
    Case("What antithrombotic therapy is recommended after TAVI?", "VHD", ["asa"], [["asa", "aspirin"]]),
    Case("How should severe primary mitral regurgitation be managed?", "VHD", ["repair"], [["mitral"], ["repair", "surgery"]]),
    Case("When is intervention recommended in tricuspid regurgitation?", "VHD", ["tricuspid"], [["tricuspid"]]),
    Case("How should coronary artery disease be managed in TAVI candidates?", "VHD", ["coronary angiography"], [["coronary"]]),
    Case("When is surgery indicated in severe aortic regurgitation?", "VHD", ["aortic regurgitation"], [["aortic regurgitation", "ar "]]),
    Case("Which procedures must be performed in a Heart Valve Centre?", "VHD", ["heart valve centre"], [["valve centre", "valve center"]]),
    Case("What is the first-line pharmacological treatment for heart failure with reduced ejection fraction?", "HF", ["arni"], [["ace", "arni", "beta-blocker", "beta blocker", "mra", "sglt2"]]),
    Case("When should SGLT2 inhibitors be started in heart failure?", "HF", ["sglt2"], [["sglt2"]]),
    Case("How often should heart failure treatment be uptitrated?", "HF", ["uptitration"], [["week"]]),
    Case("How is cardiac amyloidosis diagnosed?", "HF", ["amyloidosis"], [["amyloid"]]),
    Case("What are the recommendations for exercise-based cardiac rehabilitation?", "HF", ["rehabilitation"], [["rehabilitation"]]),
    Case("When is an implantable cardioverter defibrillator recommended?", "HF", ["icd"], [["icd"]]),
    Case("How should diuretics be used in decompensated heart failure?", "HF", ["diuretic"], [["diuretic"]]),
    Case("How should iron deficiency be investigated and treated in heart failure?", "HF", ["iron"], [["iron"]]),
    Case("What is the insulin dose for diabetic ketoacidosis?", "OOD", [], []),
    Case("What is the weather in Tunis today?", "OOD", [], []),
]


def _normalize(text: str) -> str:
    for char in ("\u00a0", "\u202f", "\u2009", "\u2013", "\u2014"):
        text = text.replace(char, " ")
    return " ".join(text.lower().split())


def _post(base_url: str, path: str, payload: dict, timeout: int) -> dict:
    request = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def _rank_of(hits: list[dict], phrases: list[str]) -> int | None:
    wanted = [_normalize(p) for p in phrases]
    for rank, hit in enumerate(hits, start=1):
        text = _normalize(hit["text"])
        if all(phrase in text for phrase in wanted):
            return rank
    return None


def _answer_is_correct(answer: str, groups: list[list[str]]) -> bool:
    text = _normalize(answer)
    return all(any(_normalize(option) in text for option in group) for group in groups)


def _citation_in_range(citations: list[int], hit_count: int) -> bool:
    return bool(citations) and all(1 <= c <= hit_count for c in citations)


def run(base_url: str, k: int, timeout: int) -> dict:
    rows = []
    for case in CASES:
        hits = _post(base_url, "/query", {"question": case.question, "k": k}, timeout)["hits"]
        answer = _post(base_url, "/answer", {"question": case.question, "k": min(k, 4)}, timeout)
        rank = _rank_of(hits, case.evidence) if case.evidence else None
        rows.append(
            {
                "question": case.question,
                "domain": case.domain,
                "rank": rank,
                "top_section": hits[0]["section"] if hits else None,
                "top_score": hits[0]["score"] if hits else None,
                "answer": answer.get("answer"),
                "grounded": answer.get("grounded"),
                "citations": answer.get("citations") or [],
                "citation_valid": _citation_in_range(answer.get("citations") or [], len(hits)),
                "answer_terms_ok": _answer_is_correct(answer.get("answer") or "", case.answer_terms),
            }
        )
        print(f"\n[{case.domain}] {case.question}")
        print(f"  rank={rank}  top={rows[-1]['top_section']!r}  score={rows[-1]['top_score']}")
        print(f"  answer: {(answer.get('answer') or '')[:260]}")
        print(f"  grounded={rows[-1]['grounded']} citations={rows[-1]['citations']}")

    in_domain = [r for r in rows if r["domain"] != "OOD"]
    out_domain = [r for r in rows if r["domain"] == "OOD"]
    n = len(in_domain)
    ranks = [r["rank"] for r in in_domain]

    summary = {
        "questions": n,
        "hit@1": round(sum(1 for r in ranks if r == 1) / n, 3),
        f"hit@{k}": round(sum(1 for r in ranks if r and r <= k) / n, 3),
        "mrr": round(sum(1.0 / r for r in ranks if r) / n, 3),
        "grounded_rate": round(sum(1 for r in in_domain if r["grounded"]) / n, 3),
        "citation_valid_rate": round(sum(1 for r in in_domain if r["citation_valid"]) / n, 3),
        "answer_terms_rate": round(sum(1 for r in in_domain if r["answer_terms_ok"]) / n, 3),
        "ood_refusal_rate": round(
            sum(1 for r in out_domain if r["grounded"] is False and (r["answer"] or "").startswith("I don't know"))
            / max(len(out_domain), 1),
            3,
        ),
    }
    print("\n" + "=" * 60)
    for key, value in summary.items():
        print(f"{key:>20}: {value}")
    return {"summary": summary, "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8003")
    parser.add_argument("--k", type=int, default=5, help="retrieval depth for hit@k")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--json", type=argparse.FileType("w"), default=None)
    args = parser.parse_args()

    report = run(args.base_url, args.k, args.timeout)
    if args.json:
        json.dump(report, args.json, indent=2, ensure_ascii=False)
        print(f"wrote {args.json.name}")


if __name__ == "__main__":
    main()
