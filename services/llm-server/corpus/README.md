# ESC corpus — drop your guideline files here

`/ingest` reads every `.md`, `.txt` and `.pdf` file under this directory
(recursively). Markdown headings become the `section` metadata attached to each
chunk, so **prefer `.md`** if you have a choice.

## Expected layout

```text
corpus/
├── README.md                                 <- this file (skipped by the loader)
├── SOURCES.md                                <- provenance / licensing (skipped)
├── esc_2025_valvular_heart_disease.md
└── esc_2026_heart_failure.md
```

`README*` and `SOURCES*` files are documentation and are **skipped by the
loader**, so they never pollute retrieval.

The two guideline files are generated from the official ESC slide sets by
[`scripts/pdf_to_corpus.py`](../scripts/pdf_to_corpus.py):

```bash
python scripts/pdf_to_corpus.py deck.pdf corpus/esc_2025_valvular_heart_disease.md \
  --title "2025 ESC/EACTS Guidelines for the management of valvular heart disease"
```

The converter detects running headers by frequency, emits one `##` section per
slide and drops figure captions whose body is too short to answer anything.

## Formatting tips

- One `#` heading per guideline, `##` per section. Sections are the citation unit
  reported by `/answer`.
- Keep paragraphs under ~800 characters; longer ones are split at a sentence
  boundary near the configured `CHUNK_SIZE`.
- Plain prose retrieves better than tables. If you must include a table, add a
  sentence above it summarising the content.

## Ingesting

```bash
# straight from the mounted corpus directory
curl -s -X POST http://localhost:8003/ingest -H 'Content-Type: application/json' -d '{}'

# or upload one file without touching the repo
curl -s -X POST http://localhost:8003/ingest -F file=@my_guideline.md
```

Re-ingesting the same file replaces its chunks; other sources are untouched.
Pass `{"reset": true}` to wipe the collection first.

## Licensing

Do not commit the full text of copyrighted ESC documents. The guideline `.md`
files are git-ignored; only this file and [`SOURCES.md`](SOURCES.md) are tracked.
See `SOURCES.md` for the exact provenance and the ESC licence terms that apply.
