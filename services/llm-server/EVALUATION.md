# RAG evaluation

This page is the evidence behind the "évaluation du RAG" section of the project
report. Everything below is reproducible with
`python scripts/evaluate_rag.py`; the raw JSON is written next to it.

## Setup

| Item | Value |
|---|---|
| Corpus | 2 ESC slide sets, 156 sections → **228 chunks** (128k characters) |
| Embedder | `all-MiniLM-L6-v2`, ONNX, frozen (retrieval only) |
| Generator | Qwen2.5-1.5B-Instruct Q4_K_M via `llama.cpp`, greedy decoding |
| Vector store | ChromaDB via `vector-db` service (`CHROMA_HOST=vector-db`), cosine, persisted at `./services/vector-db/data/` |
| Context | 4 excerpts, 6 000-character budget |
| Hardware | Intel i7-12650H, 16 threads, **CPU only** (no CUDA runtime) |
| Relevance floor | cosine ≥ 0.40, checked before the model is called |

The 1.5B model is a development stand-in for the fine-tuned 7B required by the
brief; it is deliberately small enough to run on CPU.

## Method

*Retrieval* — a question counts as a hit at rank *k* when one of the top-*k*
chunks contains every phrase in a hand-written `evidence` list
(`"4.0 m/s"`, `"uptitration"`, `"heart valve centre"`, …). Phrases are
answer-bearing strings rather than section names, because a single slide often
holds several unrelated recommendations.

*Answering* — the service reports `grounded` and `citations`. We measure the
rate of grounded answers, whether every citation index is in range, and whether
two out-of-domain control questions are refused. Factual correctness is **not**
scored automatically: a lexical check cannot separate "first-line is an ACE-I or
ARNI" from "first-line is ivabradine", since both mention the drug classes
somewhere. The 16 in-domain answers were therefore adjudicated by hand
(correct / partial / incorrect).

## Retrieval

16 in-domain questions, 5 chunks retrieved each.

| Metric | Value |
|---|---|
| hit@1 | **0.750** |
| hit@2 | 0.875 |
| hit@3 | 0.875 |
| hit@4 | 0.938 |
| hit@5 | **1.000** |
| MRR | **0.841** |

Per guideline set:

| Domain | n | hit@1 | hit@3 | MRR |
|---|---|---|---|---|
| Valvular heart disease | 8 | 0.62 | 0.88 | 0.775 |
| Heart failure | 8 | 0.88 | 0.88 | 0.906 |

Every question has its evidence somewhere in the top 5, which is the property
that matters for `/answer`: the excerpt only has to be *present* for the
generator to use it.

### Effect of each retrieval change

| Configuration | hit@1 | hit@3 | MRR |
|---|---|---|---|
| Dense only (12-question pilot) | 0.33 | 0.50 | 0.46 |
| + section title in the embedded text | 0.42 | 0.67 | 0.55 |
| + lexical fusion over the full corpus | 0.92 | 1.00 | 0.958 |
| Lexical fusion, **current 16-question set** | 0.75 | 0.875 | 0.841 |

The first three rows are the 12-question development set used while tuning; the
last row is the larger, harder set that includes questions with no dedicated
slide (heart valve centre requirements, coronary disease in TAVI candidates).
The set changed, not the method — the drop from 0.92 to 0.75 is added
difficulty, not a regression.

Two changes account for most of the gain:

1. **Title-aware embeddings.** Slide bodies are terse (`"Vmax ≥4.0 m/s, mean
   gradient ≥40 mmHg"`); embedding the section title with the text gives the
   question something to match.
2. **Lexical ranking over the whole corpus.** Re-ranking only the dense
   candidates cannot recover a chunk the dense model missed. "How often should
   treatment be uptitrated?" sat at dense rank 40 but lexical rank 1; comparing
   terms on a 5-character prefix aligned `uptitration` with `uptitrate`.

## Answering

| Metric | Value |
|---|---|
| Grounded answers | **15 / 16 = 0.938** |
| Valid citations (all indices in range) | **15 / 16 = 0.938** |
| Out-of-domain refusals | **2 / 2** |
| Median latency per answer | **1.7 s** (mean 2.1 s) |

Manual adjudication of the 16 in-domain answers:

| Verdict | Count | Questions |
|---|---|---|
| Correct | 9 | asymptomatic severe AS, antithrombotic after TAVI, primary mitral regurgitation, coronary disease in TAVI, severe aortic regurgitation, SGLT2 inhibitors, uptitration interval, amyloidosis, ICD |
| Partial | 3 | tricuspid regurgitation, cardiac rehabilitation, diuretics |
| Incorrect | 4 | severe AS criteria, Heart Valve Centre, HFrEF first-line, iron deficiency |

### Failure analysis

**1. Definitional confusion (severe AS criteria).** The answer reported the
"very severe AS" thresholds (mean gradient ≥60 mmHg, Vmax >5.0 m/s) that the
guideline uses to select asymptomatic patients for intervention, instead of the
definition of severe AS (Vmax ≥4.0 m/s, mean gradient ≥40 mmHg,
AVA ≤1.0 cm²). Both statements are in the corpus; the definition sits at
retrieval rank 5, inside a recommendation chunk rather than a dedicated
diagnostic slide. The model trusted the wrong excerpt.

**2. Table structure lost in extraction (Heart Valve Centre).** The slide is a
table of procedures and centre levels. `pypdf` flattens it into a line soup, so
the answer merged rows into a sentence that is not in the guideline
("severe pulmonary hypertension (PMR)"). This is an extraction problem, not a
retrieval or prompting one.

**3. Vocabulary mismatch (HFrEF first-line).** The corpus never says
"first-line" for the four pillars; it says each drug class "is recommended".
The retrieved excerpt that mentions "pharmacological management" is the
ivabradine one, which lists ACE-I/ARNI, MRA, SGLT2-I and beta-blockers only as
*background therapy*, so the model answered "ivabradine". A correct answer needs
the model to infer that the recommended classes collectively form first-line
therapy.

**4. Parametric hallucination (iron deficiency).** The top excerpt reads
"Intravenous iron supplementation is recommended in patients with symptomatic
HFrEF and iron deficiency…", yet the model answered with echocardiography, MRI
and right-heart catheterisation — none of which appear in the excerpts. This
violates the grounding instruction and is the most serious failure mode for a
clinical tool.

The three model-side failures (1, 3, 4) are exactly what the jalon-3 fine-tune
of the 7B generator is meant to reduce: they are failures of instruction
following and domain grounding, not of retrieval. Failure 2 calls for a
layout-aware PDF extractor (`pdfplumber`) before the corpus is regenerated.

## Guardrails verified

* Out-of-domain questions ("insulin dose for DKA", "weather in Tunis") return
  `"I don't know — this question is not covered by the guideline corpus."` with `grounded=false`, without calling the generator
  when the cosine floor is not met.
* An answer with no valid bracket citation is reported `grounded=false`, even
  when the prose is correct — the citation is part of the contract, not
  decoration.
* Answers echo `[n]` indices only; the mapping to file and section is returned
  in `sources`, so the model never needs to reproduce metadata.

## Reproducing

```bash
cd services/llm-server
python scripts/evaluate_rag.py --json eval.json      # against localhost:8003
```

## Limitations

* 16 questions is enough to compare configurations, not to claim clinical
  validity. The final report should extend this to ~50 questions with
  clinician-written reference answers.
* Corpus text comes from official ESC slide sets, which ESC licenses for
  personal/educational use only. The text is not committed; see
  `corpus/SOURCES.md`.
* CPU-only inference keeps the demo portable but caps throughput at roughly one
  answer every two seconds. A CUDA build of `llama.cpp` is a one-line change in
  the Dockerfile on a machine with `nvidia-container-toolkit`.
