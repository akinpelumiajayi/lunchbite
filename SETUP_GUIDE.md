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
GROQ_MODEL=qwen/qwen3.6-27b

# Judge model (MUST come from a different family than the generator, to avoid
# self-preferencing — Qwen generates, an OpenAI open-weight model judges)
GROQ_JUDGE_MODEL=openai/gpt-oss-120b

# Both are reasoning models. You do not set the reasoning budget: it is derived
# from the model id, because the accepted values differ per family (qwen3:
# none|default, gpt-oss: low|medium|high) and a wrong one 400s every call.

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
1. **Database setup** — builds the ChromaDB vector store (`all-MiniLM-L6-v2`
   dense embeddings) + the BM25 index (~2s)
2. **Benchmark** — runs all 5 pipelines across all 36 cases. Four of them call
   the generator, so a live run is ~144 generation calls.
3. **Evaluation** — safety metrics (deterministic) + LLM-as-judge metrics
4. **Retrieval evaluation** — IR metrics per retriever against the golden set
5. **Verifiable reward** — scores every menu and verifies the scores. No model
   is called and nothing is re-run, so this step costs no tokens and cannot be
   stopped by a spent quota.
6. **Report** — writes `report/COMPARATIVE_REPORT_<timestamp>.md`

> **Budget note.** Adding the fifth arm raised the per-case cost by a third.
> On Groq's free tier a full 36-case run needs roughly 250k tokens against a
> 200k/day cap, so it will stop partway and print a `--resume` command. That is
> the intended behaviour, not a failure — see *Daily token budget* under
> Troubleshooting.

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
python3 run_all.py --model openai/gpt-oss-20b     # faster, cheaper
python3 run_all.py --model qwen/qwen3.6-27b       # the default
```

This overrides `GROQ_MODEL` / `OLLAMA_MODEL` for that run only. The judge model
(`GROQ_JUDGE_MODEL`) is always read from `.env` independently — so overriding the
generator to `openai/gpt-oss-20b` puts it in the *same family* as the default judge,
and that run's judge scores carry a self-preferencing risk the defaults avoid.

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

### Score the verifiable reward

```bash
# Score a run, recompute every record, and test the weighting
python3 benchmark/score_rewards.py benchmark/results/run_<timestamp>.json         --verify --sensitivity

# Verify a reward file someone else produced, without rescoring it
python3 benchmark/score_rewards.py --verify-only         benchmark/results/run_<timestamp>_reward.json
```

Writes `<run>_reward.json` beside the results file. **No API key, no network and
no token budget** — every component resolves to a fact in `data/recipes.json` or
a hand-labelled benchmark field, so this works offline and on a results file of
any age.

- `--verify` recomputes every record from the raw trace and reports whether the
  numbers hold. Editing a reward, or reweighting the file header without
  rescoring, fails here rather than passing unnoticed.
- `--sensitivity` re-aggregates under six weightings, including one that weights
  correctness at 0.02, and reports whether the ordering of the arms changes.
- `--menus final` scores what each arm returned rather than what the generator
  proposed. Needed to see the `reward_ranked` arm's effect, since under the
  default `proposed` it is scored before its own reranking.

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
all five pipelines, adversarial testing, and every metric formula. Most cells run
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

`tests/test_reward_checks.py`, `test_reward_model.py`, `test_reward_verify.py`,
`test_reward_rank_node.py`, `test_sensitivity.py` — the verifiable reward. The
cases that carry the most weight:

- a false allergen-absence claim floors groundedness even when every nutrition
  figure quoted is correct
- an unsafe menu scores 0.0 however well it cites, and stays 0.0 under a
  weighting that puts correctness at 0.02 — the gate is a veto, not a term
- an edited reward, an edited component score, or a reweighted file header all
  fail verification; the verifier is shown failing, not just passing
- the reward-ranking node reorders but never admits, and an adversarial injection
  in `cultural_context` cannot change its ordering
- the weight-sensitivity analysis detects an ordering that genuinely flips — an
  analysis that always reported "stable" would be worthless as a defence

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
| `GROQ_MODEL` | `qwen/qwen3.6-27b` | Generator model (for recommendations). |
| `GROQ_REASONING_EFFORT` | *derived from model* | Generator reasoning budget. Unset → looked up from the model id (qwen3 → `none`, gpt-oss → `low`, unknown → parameter not sent). Set it empty to suppress. |
| `GROQ_JUDGE_MODEL` | `openai/gpt-oss-120b` | Judge model (must come from a different family than the generator). |
| `GROQ_JUDGE_REASONING_EFFORT` | *derived from model* | Judge reasoning budget. Same lookup; gpt-oss takes `low`/`medium`/`high` and rejects `none`. |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL. |
| `OLLAMA_MODEL` | `llama3.2` | Ollama generator model. |
| `OLLAMA_JUDGE_MODEL` | `llama3.1` | Ollama judge model. |
| `LANGSMITH_API_KEY` | — | LangSmith tracing key (optional). |
| `LANGCHAIN_PROJECT` | `lunch-rag-benchmark` | LangSmith project name. |
| `LLM_TEMPERATURE` | `0.1` | Generation temperature (0.0–1.0). |
| `LLM_MAX_TOKENS` | `1200` | Max tokens per generation call. Groq bills the *reservation*, so this decides how much of a run fits the 200k/day cap. With five arms a full 36-case run needs ~250k and spans two days; do not lower this below ~1000 to force it into one, as observed completions reach 786 tokens and truncation corrupts the run you are trying to cite. |
| `JUDGE_MAX_TOKENS` | `450` | Max tokens per judge call. Groq bills the *reserved* budget, so this sets how many judge calls a day's quota buys (~344). |
| `JUDGE_TEMPERATURE` | `0.0` | Judge sampling temperature. A scorer should not sample. |
| `LLM_TIMEOUT_SECS` | `60` | Per-request timeout. |
| `LLM_MAX_RETRIES` | `4` | Retry budget; prevents a transient 429 shrinking the scored sample. |
| `LUNCH_NUTRITION_FRACTION` | `0.40` | Share of the daily nutrient max allowed at lunch. See `eval/check_data_quality.py` before changing. |
| `NUTRITION_GATE` | `advisory` | What a breach of that ceiling does: `advisory` reports it and keeps the recipe, `hard` rejects it, `off` skips the check. Allergens are gated the same way in every mode — this setting does not touch them. Advisory is the default because `sugars_g` is TOTAL sugars while the guideline is FREE sugars; enforcing it removed 21 of 29 recipes at age 7–10. |
| `DIET_GATE` | `hard` | What a diet-requirement miss does: `hard` rejects, `advisory` warns, `off` skips. Covers vegetarian / vegan / pescatarian / halal / kosher, declared per child via `diet_requirements` (or written into `cultural_context`, which is also read). Defaults to `hard` because the data is sound — all 29 recipes carry `diet_tags` and the exclusions come from the same ingredient list the allergen scan uses. For halal and kosher it checks **ingredient exclusions only** (pork and derivatives; plus shellfish for kosher) and attaches a warning saying certification, slaughter method and preparation separation are not verified. Allergens are unaffected in every mode. |
| `LLM_MAX_RATE_LIMIT_WAIT` | `90` | Seconds of provider-requested backoff worth waiting out inline. Above this the 429 is treated as a spent daily budget and the run stops rather than sleeping through it. |
| `HF_EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model. Always used; there is no off switch. |
| `USE_CROSS_ENCODER_RERANKER` | `true` | Cross-encoder reranking. `false` skips it and keeps RRF order. |
| `CROSS_ENCODER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Reranker model. |
| `RERANK_TOP_K` | `9` | Candidates surviving reranking into the gates / LLM. |
| `REWARD_WEIGHTS` | *see below* | JSON object overriding the verifiable-reward component weights, e.g. `{"correctness": 0.5}`. Unnamed components keep their defaults: `correctness` 0.35, `groundedness` 0.25, `citation_accuracy` 0.15, `retrieval_accuracy` 0.10, `relevance` 0.10, `completeness` 0.05. An unknown component name is refused rather than ignored, so a typo cannot leave an intended reweighting silently inert. Weights are recorded and digested with every score, so changing them without rescoring fails `--verify`. |

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
│   │   ├── state.py              PipelineState TypedDict (every arm)
│   │   ├── nodes.py              All LangGraph node functions
│   │   └── build_graphs.py       5 graph builders + BUILDER_NAMES
│   ├── reward/                   Verifiable reward (RLVR) — no LLM, no network
│   │   ├── checks.py             The six components, each a pure function
│   │   ├── model.py              Weighted aggregation, gated on correctness
│   │   ├── scoring.py            Applies the reward to a results file, offline
│   │   ├── sensitivity.py        Re-aggregates under six weightings
│   │   ├── verify.py             Recomputes a scored file and proves it
│   │   └── corpus.py             Ground-truth lookups over data/recipes.json
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
│   ├── benchmark_cases.py        36 cases: standard/multi/adversarial/edge/cultural
│   ├── runner.py                 Runs every arm, saves JSON results
│   ├── evaluator.py              Safety + LLM-as-judge metrics (separate judge)
│   ├── score_rewards.py          Verifiable reward + --verify + --sensitivity
│   └── stats.py                  McNemar, bootstrap CIs, paired differences
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

**Ollama run produces almost no menus / `Could not parse LLM response`**
The most common local-model failure, and it is not a safety problem — it is a
formatting one. A small model asked for JSON often replies with prose or, in one
observed case, Python source code that *generates* the JSON. Run 20260820_100542
(`llama3.2`, 3B) recorded 26 such failures across 16 cases: every error in the run.

Fixed by constraining Ollama's sampling to valid JSON (`OLLAMA_JSON_MODE=true`,
now the default). If you still see parse errors, a larger local model holds the
schema better — but check the memory note below before pulling one.

**`Segmentation fault` during `--provider ollama`**
Almost always memory, not a code fault. Ollama holds a model resident for ~5 minutes
after its last call, so the generator and the judge can be in RAM simultaneously:
`llama3.2` (2.0 GB) plus `llama3.1` (4.9 GB) needs about 7 GB. When the allocation
fails inside Ollama's native backend the process dies without naming a cause, and
Git Bash reports it as a segmentation fault.

`run_all.py` now evicts the generator before the judging step and prints a warning at
startup when the configured pair looks too large for the machine. If it still dies:

```bash
# smaller judge (same family, but it fits)
OLLAMA_JUDGE_MODEL=llama3.2      # in .env

# or skip judging entirely — safety and retrieval metrics need no LLM
python3 run_all.py --provider ollama --no-judge

# or check the configuration works before committing to a full run
python3 run_all.py --provider ollama --limit 2
```

Note that `llama3.1` and `llama3.2` are the **same family**, so an all-Ollama run does
not satisfy the different-family requirement the judge exists to meet. If you have the
RAM, `mistral` is a different lineage. Otherwise judge on Groq: leave `GROQ_API_KEY`
set and let the judge resolve to it while the generator runs locally.

**LLM-as-judge produced no scores under `--provider ollama`**
Fixed: `--provider` now reaches the judge. It previously applied to the generator only,
because the evaluator called `get_judge_llm()` with no preference and that resolves to
Groq whenever `GROQ_API_KEY` is set — so an explicitly local run still sent every judge
call to the cloud, and returned empty scores if the key was absent or out of quota. The
run now prints which judge it resolved to (`Judge model: ollama/...`).

**A local run takes hours**
Expected, and the judge is the worst of it. Measured on a 7.7 GB machine: an 8B local
judge (`llama3.1`) ran over 40 minutes for **2 cases** and had not finished. Generation
on `llama3.2` is workable; local judging is not.

The configuration that actually works on a modest machine is local generation with a
cloud judge — which also restores the different-model-family property, since `llama3.1`
and `llama3.2` are the same lineage:

```bash
python3 run_all.py --provider ollama --judge-provider groq
```

Measured end to end at 300s for 2 cases (8/8 generations clean, 8/8 judge calls scored)
against 40+ minutes and unfinished for the all-local equivalent. Use `--limit N` to
smoke-test any configuration first; the truncated run is marked as such in its metadata.

**Daily token budget exhausted partway through a run**

Expected on Groq's free tier, and handled rather than fatal. Four of the five arms
call the generator, so a 36-case run bills roughly `36 x 4 x (prompt + reserved
max_tokens)` — about 250k against a 200k/day cap for `qwen/qwen3.6-27b`.

The runner saves after **every** case, latches the quota so it stops instead of
recording the remainder as failures, marks the file `complete: false`, and prints
the command to finish it:

```bash
# day 1 — runs until the budget latches
python3 run_all.py --no-judge

# day 2 — finishes on the SAME model, no re-spend on what is already done
python3 benchmark/runner.py --resume benchmark/results/run_<timestamp>.json
```

Three things worth knowing:

- The provider's "try again in 16m" refers to the one call it refused, not a
  refill. The daily cap is a rolling window — wait for the day to roll, not for
  that interval.
- **Do not lower `LLM_MAX_TOKENS` to force a run into one day.** Observed
  completions reach 786 tokens; truncation silently corrupts the run.
- Finishing on a different provider is possible but mixes generators in one file,
  so it is refused unless you pass `--allow-provider-change`. The mix is recorded
  in `metadata.generators` and disclosed in the report.

The verifiable reward is unaffected by any of this — it calls no model, so
`benchmark/score_rewards.py` works on a partial run and on a spent quota.

**`vectordb/` error or stale embeddings**
Delete `vectordb/` and run `python3 src/setup_database.py` to rebuild from scratch.

**Windows path issues**
All paths use `Path(__file__).resolve()` (absolute) rather than relative paths, so
the project works from any working directory. Use `venv\Scripts\Activate.ps1` to
activate the virtual environment on Windows.

**HuggingFace download hangs**
The `huggingface_upgrade/` modules have a 30-second SIGALRM timeout (Unix only; on
Windows the download attempt proceeds without a hard timeout). Embeddings have no
fallback — `all-MiniLM-L6-v2` is the only backend and an unavailable model raises
`EmbeddingModelUnavailable` rather than silently degrading, because the removed
TF-IDF path could not tell "milk-free lunch" from "contains milk". Reranking does
fall back: it is skipped and the RRF order is kept.
