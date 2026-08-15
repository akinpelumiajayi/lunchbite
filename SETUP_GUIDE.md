# Setup and Run Guide

This project implements and benchmarks four RAG pipelines for allergy-safe children's
school lunch recommendations:

| Pipeline | What it tests |
|----------|--------------|
| **No-LLM baseline** | Rule-based safety only — no LLM at any step |
| **Neural-only RAG** | Retrieval + LLM with constraints as prompt text only |
| **Neuro-symbolic RAG** | Retrieval + deterministic symbolic gates + LLM |
| **No-RAG control** | LLM with profile only, no retrieved context |

> **Scope:** Research prototype only. Does not provide medical or nutritional advice.
> Not intended for use with real children.

---

## Requirements

- **Python 3.9+** (3.10+ recommended)
- **One of:** a Groq API key (cloud, free) OR Ollama running locally (no key)
- **Optional:** LangSmith API key for node-level tracing

---

## Step 1 — Install dependencies

```bash
cd lunch_rag
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

---

## Step 2 — Configure your .env file

Copy the example and fill in your values:

```bash
cp .env.example .env
```

Then open `.env` in a text editor. The key variables are:

```bash
# ── OPTION A: Groq (cloud, free tier) ──────────────────────────────────────
# Get a free key at: https://console.groq.com
GROQ_API_KEY=gsk_...

# Generator model (fast, used for recommendations)
GROQ_MODEL=llama-3.1-8b-instant

# Judge model (larger — MUST differ from generator to avoid self-preferencing)
GROQ_JUDGE_MODEL=llama-3.3-70b-versatile

# ── OPTION B: Ollama (local, no API key needed) ─────────────────────────────
# Leave GROQ_API_KEY empty and Ollama will be used automatically.
# Install Ollama from https://ollama.com, then:
#   ollama pull llama3.2      # generator model
#   ollama pull llama3.1      # judge model
#   ollama serve
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2
OLLAMA_JUDGE_MODEL=llama3.1

# ── OPTIONAL: LangSmith tracing ─────────────────────────────────────────────
# Get a free key at: https://smith.langchain.com
LANGSMITH_API_KEY=lsv2_...
LANGCHAIN_PROJECT=lunch-rag-benchmark
```

> **Provider resolution:** Groq is used if `GROQ_API_KEY` is set. Ollama is used if
> the Groq key is absent and Ollama is reachable at `OLLAMA_BASE_URL`. If neither is
> configured, use `--mock` mode.

---

## Step 3 — Run the full pipeline (Aims 1–5)

All five aims run sequentially from one command:

```bash
python3 run_all.py
```

This executes:
1. **Database setup** — builds ChromaDB vector store (TF-IDF) + BM25 index (~2s)
2. **Benchmark** — runs all 4 pipelines across all 30 cases (~120 LLM calls for a live run)
3. **Evaluation** — computes safety metrics (deterministic) + LLM-as-judge metrics
4. **Report** — writes `report/COMPARATIVE_REPORT_<timestamp>.md`

---

## Run options

### Mock mode — no API key needed

```bash
python3 run_all.py --mock
```

Uses a simulated LLM that deliberately follows adversarial injections to demonstrate
the architectural difference between `neural_rag` (vulnerable) and `neurosymbolic`
(protected). No API calls, no cost, completes in ~2 seconds.

### Force a specific provider

```bash
python3 run_all.py --provider groq    # force Groq even if Ollama is running
python3 run_all.py --provider ollama  # force Ollama even if GROQ_API_KEY is set
```

### Override the generator model

```bash
python3 run_all.py --model llama-3.1-8b-instant   # faster, cheaper
python3 run_all.py --model mixtral-8x7b-32768      # alternative
```

This overrides `GROQ_MODEL` / `OLLAMA_MODEL` for that run only. The judge model
(`GROQ_JUDGE_MODEL`) is always read from `.env` independently.

### With LangSmith tracing

```bash
# In .env: LANGSMITH_API_KEY=lsv2_...
python3 run_all.py
```

Every node call (bm25_retrieve, symbolic_prefilter, generate, symbolic_postfilter, etc.)
is traced as a named span with full input/output state. View at:
https://smith.langchain.com → project `lunch-rag-benchmark`

### Skip database rebuild (after first run)

```bash
python3 run_all.py --skip-setup
```

Safe to use once `vectordb/` exists and `data/*.json` hasn't changed.

### Re-run eval + report on existing results

```bash
python3 run_all.py --results benchmark/results/run_20260812_123456.json
```

Skips Steps 1–2 and re-runs evaluation and report generation on an existing results
file. Useful for changing `.env` judge model and re-evaluating, or for adding
LangSmith tracing after the fact.

---

## Running individual components

### Rebuild the database only

```bash
python3 src/setup_database.py
```

Run this whenever you add recipes to `data/recipes.json` or change other data files.
Rebuilds both the ChromaDB vector store and the BM25 pickle index.

### Run the benchmark only

```bash
python3 benchmark/runner.py                    # auto-detect provider from .env
python3 benchmark/runner.py --provider ollama  # force Ollama
```

Results saved to `benchmark/results/run_<timestamp>.json`.

### Run evaluation only

```bash
python3 benchmark/evaluator.py benchmark/results/run_<timestamp>.json
python3 benchmark/evaluator.py benchmark/results/run_<timestamp>.json --no-judge
```

`--no-judge` skips the LLM-as-judge metrics and computes only the deterministic
safety metrics (no API call needed).

### Generate a report only

```bash
python3 report/generate_report.py benchmark/results/run_<timestamp>.json \
        --eval benchmark/results/run_<timestamp>_eval.json
```

### Run the interactive CLI

```bash
python3 src/cli.py
```

Prompts for a child profile interactively and runs the original (non-LangGraph)
pipeline. Reads `.env` automatically.

### Open the Jupyter notebook

```bash
jupyter notebook notebooks/lunch_rag_pipeline.ipynb
```

A 29-cell walkthrough covering embedding generation, BM25, semantic search, RRF fusion,
all four pipelines, adversarial testing, and every metric formula. Most cells run
without an API key.

---

## Safety tests (no API key needed)

```bash
python3 -m pytest tests/ -q
```

The whole suite runs offline against a mock LLM. The safety-critical parts:

`tests/test_pipeline_integration.py` — end-to-end against the real
neurosymbolic LangGraph with a deliberately misbehaving mock LLM:
1. Well-behaved LLM → recommendation passes through
2. LLM proposes an allergen-containing recipe → caught by the post-filter
3. LLM claims an allergen is absent that the recipe declares → caught
4. LLM hallucinates a recipe_id not in the database → rejected
5. LLM returns a fabricated citation → corrected against the recipe record
6. Zero safe candidates → the LLM is never called at all

`tests/test_guardrails.py` — the deterministic gate in isolation, including the
compound-word traps (`butternut squash` is not milk, `eggplant` is not egg).

`tests/test_judge_metrics.py` — the LLM-as-judge aggregation, including the
invariant that all three metrics describe the same set of menus.

`tests/test_stats.py` — McNemar p-values against hand-computed exact binomials,
bootstrap intervals, and paired score differences.

`tests/test_eval_generation.py` — evaluation plumbing (claim extraction,
contradiction detection, faithfulness sensitivity, relevancy scoring).

---

## Retrieval models

Semantic retrieval uses neural sentence embeddings and a cross-encoder reranker.
There is no on/off switch for embeddings — this is the only backend. The former
offline TF-IDF implementation has been removed.

`python3 src/setup_database.py` downloads both models on first run and caches
them in `~/.cache/huggingface/`:

| Role | Model | Size |
|---|---|---|
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (384-dim) | ~90 MB |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | ~90 MB |

The first run therefore needs network access. If a model cannot be loaded, the
run **fails loudly** rather than falling back to a weaker retriever — a silent
fallback would attribute benchmark results to a retriever that never ran.

Tunables in `.env`:

```bash
HF_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
USE_CROSS_ENCODER_RERANKER=true          # false skips reranking, keeps RRF order
CROSS_ENCODER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
RERANK_TOP_K=9                           # candidates surviving to the gates / LLM
```

**On negation:** the old TF-IDF retriever could not distinguish "milk-free lunch"
from "contains milk", since both share the token "milk". Neural retrieval was
expected to fix that. Measured on this corpus, it largely does not — see
`eval/eval_negation.py`, which scores all four retrievers on explicit
"<allergen>-free" queries. Retrieval remains an unreliable safety signal, which
is exactly why the deterministic guardrail is a separate stage.

---

## Environment variables reference

All variables read from `.env` (overriding is possible via shell environment).
See `.env.example` for the full annotated list.

| Variable | Default | Purpose |
|----------|---------|---------|
| `GROQ_API_KEY` | — | Groq API key. If set, Groq is used as provider. |
| `GROQ_MODEL` | `llama-3.1-8b-instant` | Generator model (fast, for recommendations). |
| `GROQ_JUDGE_MODEL` | `llama-3.3-70b-versatile` | Judge model (must differ from generator). |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL. |
| `OLLAMA_MODEL` | `llama3.2` | Ollama generator model. |
| `OLLAMA_JUDGE_MODEL` | `llama3.1` | Ollama judge model. |
| `LANGSMITH_API_KEY` | — | LangSmith tracing key (optional). |
| `LANGCHAIN_PROJECT` | `lunch-rag-benchmark` | LangSmith project name. |
| `LLM_TEMPERATURE` | `0.1` | Generation temperature (0.0–1.0). |
| `LLM_MAX_TOKENS` | `2000` | Max tokens per generation call. |
| `LLM_TIMEOUT_SECS` | `60` | Per-request timeout. |
| `LLM_MAX_RETRIES` | `4` | Retry budget; prevents a transient 429 shrinking the scored sample. |
| `LUNCH_NUTRITION_FRACTION` | `0.40` | Share of the daily nutrient max allowed at lunch. See `eval/check_data_quality.py` before changing. |
| `HF_EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model. Always used; there is no off switch. |
| `USE_CROSS_ENCODER_RERANKER` | `true` | Cross-encoder reranking. `false` skips it and keeps RRF order. |
| `CROSS_ENCODER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Reranker model. |
| `RERANK_TOP_K` | `9` | Candidates surviving reranking into the gates / LLM. |

---

## Project structure

```
lunch_rag/
├── run_all.py                    Single entry point (Aims 1–5)
├── requirements.txt              All Python dependencies
├── .env.example                  Documented template — copy to .env
├── .env                          Your secrets (gitignored)
├── .gitignore                    Protects .env, vectordb/, benchmark/results/
│
├── data/
│   ├── recipes.json              29 recipes (UK Gov 001-009, PACK-IT 010-029)
│   ├── allergen_rules.json       EU FIC 14 declarable allergens + school rules
│   ├── nutrition_guidelines.json Eatwell Guide + PHE age-banded dietary tables
│   └── data_sources.json         Full citation metadata for all 6 data sources
│
├── src/
│   ├── llm_provider.py           .env loading, Groq/Ollama resolution,
│   │                             get_llm() + get_judge_llm() (separate models)
│   ├── graphs/
│   │   ├── state.py              PipelineState TypedDict (all 4 pipelines)
│   │   ├── nodes.py              All LangGraph node functions
│   │   └── build_graphs.py       4 graph builders
│   ├── guardrails.py             Deterministic allergen + nutrition checks
│   ├── huggingface_upgrade/
│   │   ├── huggingface_embeddings.py   all-MiniLM-L6-v2 (optional)
│   │   └── reranker.py                 ms-marco cross-encoder reranker
│   ├── document_loader.py        data/*.json → Chunk objects with citations
│   ├── console.py                UTF-8 stdout (Windows code-page safety)
│   ├── vector_store.py           ChromaDB wrapper
│   ├── setup_database.py         Builds ChromaDB + BM25 indexes
│   ├── main.py                   Pipeline entry point (drives the LangGraph)
│   └── cli.py                    Interactive CLI
│
├── benchmark/
│   ├── benchmark_cases.py        30 cases: standard/multi/adversarial/edge/cultural
│   ├── runner.py                 Runs all 4 pipelines, saves JSON results
│   └── evaluator.py              Safety + LLM-as-judge metrics (separate judge)
│
├── report/
│   ├── generate_report.py        Writes Aim 5 comparative Markdown report
│   └── COMPARATIVE_REPORT.md     Sample output from mock benchmark run
│
├── notebooks/
│   └── lunch_rag_pipeline.ipynb  Full pipeline walkthrough with metric formulas
│
└── eval/                         Original pre-LangGraph evaluation suite
    ├── eval_dataset.py
    ├── eval_retrieval.py
    ├── eval_generation.py
    ├── run_full_eval.py
    ├── test_eval_generation_with_mock.py
    └── EVALUATION_REPORT.md
```

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'chromadb'`**
The first `python3 run_all.py` call auto-installs missing packages. If it fails, run
`pip install -r requirements.txt` manually inside your venv.

**`No LLM provider is available`**
Either set `GROQ_API_KEY=gsk_...` in `.env`, or start Ollama (`ollama serve`) and
ensure `ollama pull llama3.2` has been run. Use `--mock` to test without any provider.

**`vectordb/` error or stale embeddings**
Delete `vectordb/` and run `python3 src/setup_database.py` to rebuild from scratch.

**Windows path issues**
All paths use `Path(__file__).resolve()` (absolute) rather than relative paths, so
the project works from any working directory. Use `venv\Scripts\Activate.ps1` to
activate the virtual environment on Windows.

**HuggingFace download hangs**
The `huggingface_upgrade/` modules have a 30-second SIGALRM timeout (Unix only; on
Windows the download attempt proceeds without a hard timeout). If it times out, both
modules fall back cleanly to TF-IDF / no reranking and print a message explaining why.
