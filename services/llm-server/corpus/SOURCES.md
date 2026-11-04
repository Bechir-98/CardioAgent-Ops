# Corpus sources and licensing

The corpus is **not committed to this repository**. Only this file and
`README.md` are tracked; the derived guideline text is generated locally and
ignored by git.

## Documents used

| Local file | Source | Licence |
|---|---|---|
| `esc_2025_valvular_heart_disease.md` | 2025 ESC/EACTS Guidelines for the management of valvular heart disease — official ESC slide set (105 pages), adapted from *European Heart Journal*, doi:10.1093/eurheartj/ehaf194 | © ESC. Personal/educational use only. |
| `esc_2026_heart_failure.md` | 2026 ESC Guidelines for the management of heart failure — official slide set (105 pages), adapted from *European Heart Journal*, doi:10.1093/eurheartj/ehag100 | © ESC. Personal/educational use only. |

Both PDFs are distributed publicly by the ESC for personal and educational use:

- <https://www.escardio.org/Guidelines/Clinical-Practice-Guidelines/All-ESC-Practice-Guidelines/valvular-heart-disease>
- <https://www.escardio.org/Guidelines/Clinical-Practice-Guidelines/All-ESC-Practice-Guidelines/heart-failure>

## Why slide sets rather than the full guidelines

The ESC website states:

> Guidelines and related materials are for use by individuals for personal or
> educational purposes. No commercial use is allowed. […] The ESC reserves all
> rights to license uses of ESC Guidelines to train or develop generative
> artificial intelligence (AI) models […] Any use of ESC Guidelines in software
> tools, generative AI models, computational models, or algorithms in which ESC
> Guideline content is included, cited, or transformed in any way requires a
> formal license agreement.

Consequences for this project:

1. The full guideline text is **not redistributed**. `.md` and `.pdf` files under
   `corpus/` are git-ignored, so cloning the repository does not copy ESC content.
2. Only the short official slide sets are used, locally, to evaluate the RAG
   pipeline in an academic context — not to train a model.
3. The generator is **not fine-tuned on this text** (jalon 3 fine-tuning uses a
   separate, independently licensed dataset). The corpus is only retrieved from
   at inference time and cited.
4. Before any commercial or public deployment, replace this corpus with
   content you have the right to use, or obtain the ESC licence. Open
   alternatives: NICE guidance (Open Government Licence v3.0), WHO
   publications, or MedlinePlus (US public domain).

## Regenerating the corpus

```bash
cd services/llm-server
# 1. download the two slide sets (see links above) as esc_vhd.pdf / esc_hf.pdf
# 2. convert to heading-structured Markdown
python scripts/pdf_to_corpus.py esc_vhd.pdf corpus/esc_2025_valvular_heart_disease.md \
  --title "2025 ESC/EACTS Guidelines for the management of valvular heart disease"
python scripts/pdf_to_corpus.py esc_hf.pdf corpus/esc_2026_heart_failure.md \
  --title "2026 ESC Guidelines for the management of heart failure"
# 3. index
curl -s -X POST localhost:8003/ingest -H 'Content-Type: application/json' -d '{}'
```

`pdf_to_corpus.py` detects repeated running headers/footers by frequency and
turns every slide into one `##` section, which becomes the citation unit.
