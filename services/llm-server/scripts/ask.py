#!/usr/bin/env python3
"""Ask the RAG service from a terminal.

Examples
--------
    python scripts/ask.py "When is surgery indicated in severe aortic regurgitation?"
    python scripts/ask.py -k 6 "How is cardiac amyloidosis diagnosed?"
    python scripts/ask.py                      # interactive loop

Prints the grounded answer, its citations, and the excerpts that were cited so
you can check that the answer really comes from the corpus.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request


def post(base_url: str, path: str, payload: dict, timeout: int) -> dict:
    request = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def show(answer: dict, sources: list[dict]) -> None:
    text = (answer.get("answer") or "").strip()
    print("\n" + text)

    if not answer.get("grounded"):
        print("\n  grounded: NO")
        if text.startswith("I don't know"):
            print("  the corpus does not cover this question")
        else:
            print("  answer was produced but carries no valid citation")
    else:
        cited = set(answer.get("citations") or [])
        print(f"\n  grounded: yes  citations: {sorted(cited)}")

    if sources:
        print("  excerpts:")
        for source in sources:
            marker = "*" if source["rank"] in set(answer.get("citations") or []) else " "
            print(
                f"   {marker} [{source['rank']}] {source['section']} "
                f"({source['source']}, cosine {source['score']:.3f})"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="*", help="question(s); omit for a loop")
    parser.add_argument("--base-url", default="http://localhost:8003")
    parser.add_argument("-k", type=int, default=4, help="excerpts (default 4)")
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()

    def ask(question: str) -> None:
        print(f"\n> {question}")
        try:
            answer = post(args.base_url, "/answer", {"question": question, "k": args.k}, args.timeout)
            sources = post(args.base_url, "/query", {"question": question, "k": args.k}, args.timeout)["hits"]
        except urllib.error.HTTPError as exc:
            print(f"  HTTP {exc.code}: {exc.read().decode(errors='replace')[:300]}")
            return
        except urllib.error.URLError as exc:
            print(f"  cannot reach {args.base_url}: {exc.reason}")
            return
        show(answer, sources)

    if args.question:
        for question in args.question:
            ask(question)
        return 0

    print(f"CardioAgent — {args.base_url} (Ctrl-D to quit)")
    while True:
        try:
            question = input("\nQuestion: ").strip()
        except EOFError:
            print()
            return 0
        if question:
            ask(question)


if __name__ == "__main__":
    sys.exit(main())
