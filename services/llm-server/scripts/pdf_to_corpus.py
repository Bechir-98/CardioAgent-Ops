#!/usr/bin/env python3
"""Convert a slide-deck or report PDF into heading-structured Markdown.

`/ingest` chunks by Markdown heading, so one heading per slide (or per
section) gives precise citations such as ``section="New recommendations (3)"``
instead of an opaque page range.

Running headers/footers are detected by frequency: any line that repeats on at
least ``--header-ratio`` of the pages is dropped. This removes the ESC running
title and DOI line from all 105 pages without hard-coding them.

Usage
-----
    python scripts/pdf_to_corpus.py deck.pdf corpus/esc_2025_vhd.md \
        --title "2025 ESC/EACTS Guidelines for valvular heart disease"

Requires ``pypdf`` (already a service dependency).
"""
from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

# Deck front matter: titles, author lists, disclaimers. Indexing them adds
# retrieval noise without any clinical content.
SKIP_TITLE_PREFIXES = (
    "official esc",
    "official slide set",
    "slide set history",
    "authors/task force",
    "authors / task force",
    "table of contents",
    "contents",
    "disclosures",
)

BOILERPLATE = {
    "recommendations",
    "class",
    "level",
    "recommendations class level",
    "class level",
}


def page_lines(path: Path) -> list[list[str]]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages: list[list[str]] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        lines = [line.strip() for line in text.splitlines()]
        pages.append([line for line in lines if line])
    return pages


def repeated_lines(pages: list[list[str]], ratio: float) -> set[str]:
    """Lines present on a large fraction of pages: running headers/footers."""
    counter: Counter[str] = Counter()
    for lines in pages:
        counter.update(set(lines))
    threshold = max(2, int(len(pages) * ratio))
    return {line for line, count in counter.items() if count >= threshold}


def clean_title(line: str, limit: int = 110) -> str:
    line = re.sub(r"\s+", " ", line).strip(" .:;-–—")
    if len(line) > limit:
        line = line[:limit].rsplit(" ", 1)[0] + "…"
    return line or "untitled"


def convert(path: Path, title: str | None, header_ratio: float, min_chars: int = 120) -> str:
    pages = page_lines(path)
    if not pages:
        raise SystemExit(f"no pages extracted from {path}")

    drop = repeated_lines(pages, header_ratio)

    doc_title = title
    if not doc_title:
        first = next((l for l in pages[0] if l not in drop), path.stem)
        doc_title = clean_title(first, 160)

    out: list[str] = [f"# {doc_title}", ""]
    seen: Counter[str] = Counter()

    for number, lines in enumerate(pages, start=1):
        body = [l for l in lines if l not in drop]
        if not body:
            continue

        # A slide's title is its first meaningful line; table headers such as
        # "Recommendations Class Level" are not titles.
        heading_index = None
        for idx, line in enumerate(body):
            if line.lower() not in BOILERPLATE:
                heading_index = idx
                break
        if heading_index is None:
            continue

        heading = clean_title(body[heading_index])
        if heading.lower().startswith(SKIP_TITLE_PREFIXES):
            continue
        content = body[:heading_index] + body[heading_index + 1 :]
        content = [l for l in content if l.lower() not in BOILERPLATE]
        # Figure slides carry only a caption; the actual diagram is an image and
        # cannot be retrieved. Such chunks match every topical question but
        # contain no answer, so they are dropped rather than indexed.
        if len(" ".join(content)) < min_chars:
            continue

        seen[heading] += 1
        if seen[heading] > 1:
            heading = f"{heading} ({seen[heading]})"

        out.append(f"## {heading}")
        out.append("")
        out.extend(content)
        out.append("")

    return "\n".join(out).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--title", default=None, help="document title (default: first page)")
    parser.add_argument(
        "--header-ratio",
        type=float,
        default=0.5,
        help="drop lines repeated on at least this fraction of pages (default 0.5)",
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=120,
        help="drop slides whose body is shorter than this (figure captions)",
    )
    args = parser.parse_args()

    markdown = convert(args.pdf, args.title, args.header_ratio, args.min_chars)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(markdown, encoding="utf-8")

    headings = markdown.count("\n## ")
    print(f"{args.pdf.name}: {headings} sections, {len(markdown):,} chars -> {args.output}")


if __name__ == "__main__":
    main()
