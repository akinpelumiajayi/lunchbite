# Improvement Plan — Lunch RAG System

## Context

`C:\lunchbites-1\lunch-rag-system` is a research prototype (~3,000 LOC) benchmarking four
pipeline architectures for allergy-safe children's school-lunch recommendation:

| mode | safety mechanism |
|---|---|
| `no_llm` | retrieval + deterministic guardrail, top-1 pick, no LLM |
| `neural_rag` | retrieval + LLM, allergen constraints as **prompt text only** |
| `neurosymbolic` | retrieval + symbolic pre-filter + LLM + symbolic post-filter |
| `no_rag` | LLM with profile only, no retrieval |

The whole artifact exists to support one claim: *a symbolic constraint layer reduces allergen
violations versus prompt-only safety.* The last real run
(`benchmark/results/run_20260812_154009_eval.json`, `groq/llama-3.1-8b-instant`, 30 cases)
reports `neural_rag` 0.30 violation rate vs `neurosymbolic` 0.0, and adversarial bypass
0.167 vs 0.0.

The architecture is genuinely well thought out — defence-in-depth gates, a pure deterministic
guardrail, source citations carried end-to-end, a hand-labelled IR ground truth, and an honest
scope notice in the report. The problems are not in the idea. They are that **several defects
in the harness make the headline numbers unsafe to publish as they stand**, and that
engineering hygiene (no git, no pinned deps, no test framework, live keys on disk) is at
prototype level.

This plan is ordered by what threatens the research conclusions first, then correctness of the
safety logic itself, then retrieval quality, then engineering.

---

## Status — what has been applied

Work is committed on `master`, one commit per theme, tests green (88 passing).
Several items in this plan were **already fixed** in the codebase before this
pass and are marked as such; the plan text predates them.

| § | Item | Status |
|---|---|---|
| 0.1 | Key rotation | **YOURS TO DO** — revoke at console.groq.com / smith.langchain.com |
| 0.1 | `--groq-key` / `--langsmith-key` flags removed | done |
| 0.2 | Symmetric adversarial injection | already fixed |
| 0.3 | Coverage / abstention-insensitive rates | already fixed |
| 0.4 | Real neural embeddings (TF-IDF removed) | already fixed |
| 0.5 | Judge quota root cause + circuit breaker | done (re-judge outstanding) |
| 0.6 | `--repeats`, mean±SD, McNemar exact test | done |
| 0.7 | `git init` + full history | done |
| 0.8a | HitRate hardcoded to 1.0 | already fixed |
| 0.8b | Golden set re-derived for 29 recipes | done |
| 0.8c | `eval_compare_retrievers.py` written | done |
| 0.9 | Synthetic-data banner | already fixed |
| 1.1–1.3 | Word-boundary matching, fail-open warnings, falsy-zero | already fixed |
| 1.5 | Allergen-claim + citation verification in post-filter | done |
| 2.1–2.5 | Retrieval correctness | already fixed (module deleted / metadata passed) |
| 3.3 | No runtime `pip install` | done |
| 3.4 | `requirements.lock.txt` (193 pins) | done |
| 4.2 | IR metrics in the main run and report | done |
| 4.4 | Injection kept out of judge prompts | already fixed (verified: 0 leaks) |
| 4.5 | README injection count corrected | done |

### Second pass — 2026-08-16 (report honesty + judge instrument)

| § | Item | Status |
|---|---|---|
| 5.1 | §3.1 verdict gated on all three metrics, not relevance alone | done |
| 5.2 | Faithfulness rubric v2: allergen fields in SOURCE, absence claims scoreable | done |
| 5.3 | `--repeats` actually used for the published run | done |
| 5.4 | `--no-judge` flag; judge scores repeat 0 only by default | done |
| 5.5 | Arms-not-identical caveat (prompt text differs) in report §1 | done |
| 5.6 | Pre-filter over-blocking reported as a cost against coverage | done |
| 5.7 | CI on every push (`.github/workflows/tests.yml`) | done |

**Second-pass findings:**

- **The report contradicted its own table.** §3.1 printed "the symbolic constraint
  layer does **not** degrade recommendation quality" whenever *relevance* spanned
  zero — while faithfulness (−0.260 [−0.420, −0.100]) and naturalness
  (−0.600 [−1.040, −0.120]) both excluded it. The verdict now reads all three and
  names whichever degraded. Regression-tested against the exact run 20260816_160852
  numbers in `tests/test_report_claims.py`.
- **Faithfulness was never measuring faithfulness.** Every `no_llm` (n=22) and
  `neurosymbolic` (n=25) menu scored exactly 0.000 with a [0.000, 0.000] interval.
  Cause: `recipe_text` passed to the judge omitted `allergens_present`, so
  "free from milk" had nothing to check against — and the corpus never records
  absent allergens, so the commonest claim type in this domain was unverifiable by
  construction. Menus quoting all four nutrition figures correctly to the decimal
  still scored 0.0; the judge's own reasoning said "the source does not mention the
  absence of various allergens". SOURCE now states both allergen lists, the rubric
  says how to score an absence claim, and process descriptions ("selected by
  rule-based scoring") are excluded from the claim count. Faithfulness is not
  comparable across `_judge_rubric_version`.
- **Over-blocking is the price of the safety result.** Pre-filter precision 0.477
  means ~104 of 199 rejections were recipes that were safe for that profile. That
  is what holds neuro-symbolic coverage to 83.3% against neural_rag's 100%, and it
  is now stated in §2 rather than left implicit across two tables.

**New findings produced by the first pass** (all reproducible):

- **The headline claim is now statistically supported.** McNemar exact test:
  neurosymbolic vs neural_rag **p = 0.00195**, vs no_rag **p = 3.05e-05**, both
  significant at α = 0.05. Previously the report stated 0.333 vs 0.000 with no
  uncertainty at all.
- **Each retrieval stage earns its place.** NDCG@5: bm25 0.555, dense 0.569,
  hybrid RRF 0.622, +cross-encoder 0.727. Fusion beats the better of its inputs;
  reranking beats fusion.
- **Retrieval cannot answer absence queries** (NDCG@5 = 0.00 on gluten-free and
  milk-free). This is a representational limit, not a tuning problem, and is the
  strongest empirical argument in the artefact for enforcing allergens
  symbolically. The stale golden set had hidden it by scoring garbage as correct.
- **The old judge numbers were biased optimistic, not merely imprecise:**
  faithfulness reported as 1.000 for neural_rag and neurosymbolic resolved to
  0.787 / 0.809 once the sample recovered from n≈3 to n=20–29.

**Still outstanding:** a clean judge re-run at `JUDGE_MAX_TOKENS=450` on
`gpt-oss-120b` (§4.3), human κ validation of the judge (§4.3), and the
larger engineering refactors in P3 (package structure §3.5, the duplicated
CLI pipeline §3.6, shared JSON parsing §3.2, config module §3.7), none of
which change any number.

---

## P0 — Defects that invalidate the reported results

These must be fixed and the benchmark re-run before any number here is cited.

### 0.1 Rotate the two live API keys — do this first

`.env:17` holds a real `GROQ_API_KEY=gsk_...` and `.env:39` a real `LANGSMITH_API_KEY=lsv2_pt_...`.
`.gitignore:2` correctly excludes `.env`, but the file is plaintext on disk and has been read
into tooling context. Revoke both at console.groq.com and smith.langchain.com, issue new ones,
and put the new values only in the (still-gitignored) `.env`.

Then remove the `--groq-key` / `--langsmith-key` CLI flags, which leak keys into shell history
and the process table: `run_all.py:218-225`, `benchmark/runner.py:162-165`,
`benchmark/evaluator.py:237-240`, `eval/run_full_eval.py:137`. Also fix the docs that teach the
pattern: `src/setup_database.py:71`, `eval/EVALUATION_REPORT.md:184,273`.

### 0.2 The adversarial injection is applied asymmetrically

`benchmark/runner.py:77-82`:

```python
("no_llm",        no_llm_graph,  False),
("neural_rag",    neural_graph,  True),   # injected
("neurosymbolic", neuro_graph,   False),  # NOT injected
("no_rag",        no_rag_graph,  True),   # injected
```

The comment justifies this as "gates outside LLM, injection irrelevant" — but that assumes the
conclusion. The `neurosymbolic` LLM *does* see the profile (and therefore would see the
injection); the post-filter is exactly what should be shown catching it. As run, the
neurosymbolic arm was **never attacked**, so `adversarial_bypass_rate: 0.0` is not evidence of
robustness — it is an artifact of not testing it.

Meanwhile `report/generate_report.py:178-181` hardcodes the prose:

> "…using identical LLM backbone and recipe corpus. Any observed safety difference is therefore
> attributable solely to the presence or absence of the symbolic constraint layer…"

That sentence is printed regardless of the data, and the design contradicts it.

**Fix:** inject into every LLM-using arm (`neural_rag`, `neurosymbolic`, `no_rag`); leave
`no_llm` uninjected and say why (it has no prompt surface). Then let the post-filter earn the
0.0. Replace the hardcoded causal claim in `generate_report.py` with text derived from the
actual measured deltas.

### 0.3 The violation-rate denominator excludes abstentions

`benchmark/evaluator.py:84`:

```python
"allergen_violation_rate": round(violations / max(total_with_menus, 1), 3),
```

`total_with_menus` counts only cases where that pipeline returned menus. A pipeline that
**refuses to answer** is silently removed from its own denominator. A system answering 5/30
cases with zero violations scores an identical `0.000` to one answering 30/30 with zero
violations — and `neurosymbolic` is precisely the arm that filters most aggressively.

**Fix:** report alongside it, per mode:
- `coverage = cases_with_final_menus / n_cases` (the data is already collected at `evaluator.py:86`)
- `violations / n_cases` as a second, coverage-insensitive rate
- a joint "safe **and** useful" rate — cases with ≥1 menu and no violation

A safety win purchased by refusing to answer is a different result from a safety win with
equal coverage, and the report currently cannot tell them apart.

### 0.4 The retriever misreports itself — HF embeddings are dead code

`.env:51` sets `USE_HUGGINGFACE_EMBEDDINGS=true` and `.env:54` `USE_CROSS_ENCODER_RERANKER=true`.
`src/llm_provider.py:225-226` then prints `HF embeddings : ENABLED` at the top of every run
(`run_all.py:227`).

Neither flag does anything. `src/vector_store.py:29-33` unconditionally constructs
`LocalTfidfEmbeddingFunction()`, and `HuggingFaceEmbeddingFunction` / `CrossEncoderReranker`
are never imported by any production code path. So the recorded run printed "ENABLED" while
retrieving with 512-dim hash-bucketed TF-IDF.

Related Windows bug: `src/huggingface_upgrade/huggingface_embeddings.py:82` does
`__file__.replace("huggingface_upgrade/huggingface_embeddings.py", "")` — `__file__` uses
backslashes on Windows, so the replace never matches and a full file path is prepended to
`sys.path`.

**Fix (pick one and be explicit in the report):**
- **(a)** Wire it up: make `get_embedding_function()` consult `use_huggingface_embeddings()`
  and return the HF function; add the reranker as a node after `rrf_fuse`. Requires a rebuild
  of `vectordb/` since vector dimensionality changes (512 → 384).
- **(b)** Delete `src/huggingface_upgrade/`, drop `sentence-transformers` from
  `requirements.txt`, remove both env flags, and state plainly that retrieval is lexical.

Either way, `print_provider_status()` must report what is actually in use, not what the env
var says.

### 0.5 The judge collapsed to n≈3 — root-caused and fixed; a clean re-run is still outstanding

**Diagnosed.** The failures were never parse failures (`parse_error: 0` in every recorded run).
They were HTTP 429s. `_try_groq` was shared by both roles, so the judge inherited
`LLM_MAX_TOKENS=2000` — a budget sized for multi-menu generation — while its prompts ask for one
score and one sentence (~30 tokens). Groq bills the **reserved** `max_tokens`, not what is
produced (`Requested 2081` in the 429 body), so the judge consumed its 100,000 tokens/day in
~48 calls. A 30-case run needs ~321. Run `run_20260814_084449` recorded
`attempted 321 / ok 42 / call_error 279`, and its published means rest on the 42 that landed
before the quota died.

**Applied this session:**
- `src/llm_provider.py` — `_get_max_tokens(role)` / `_get_temperature(role)` split by role and
  threaded through `_try_groq`/`_try_ollama` (defaulted, so `run_all.py:243` still works).
  Judge defaults `JUDGE_MAX_TOKENS=200`, `JUDGE_TEMPERATURE=0.0`. Measured: **48 → 355 judge
  calls/day**.
- `benchmark/evaluator.py` — a TPD 429 says "try again in 18m29s" and will not refill mid-run,
  yet was retried on 1s/2s backoff; with `ChatGroq(max_retries=4)` each `_judge_call` fired 12
  doomed requests. The provider's wait hint is now parsed — short per-minute limits are honoured,
  long ones trip a circuit breaker. Measured: 30 post-exhaustion calls → **1 API call, not 90**.
  `skipped_quota_exhausted` and `_judge_quota_exhausted` make the collapse visible.
- `report/generate_report.py` — the unreliability banner now counts skipped calls (which never
  reach `attempted`), and the relevance conclusion is gated at n≥10. It had been asserting "the
  symbolic constraint layer does not meaningfully degrade recommendation quality" off **3 menus
  per arm**.

**Still open:**
- **A 95%-complete re-judge now exists, but is not yet the final one.** `gpt-oss-120b` scored
  `run_20260814_084449` at `attempted 310 / ok 304 / parse_error 5 / call_error 1 / skipped 11`
  (n=20–29 per metric, against n≈3 before). It fell 11 calls short only because it ran at
  `JUDGE_MAX_TOKENS=600`; at 450 the budget fits a full 321-call run (§4.3). Re-run at 450, then
  promote into `benchmark/results/`. Output currently parked outside the repo at
  `$CLAUDE_JOB_DIR/tmp/rejudge_oss.json` — deliberately not written into `benchmark/results/`,
  since it is not the final run.
- **The corrected numbers change the story, so nothing citing the old ones survives.**

  | mode | relevance | faithfulness | naturalness |
  |---|---|---|---|
  | `no_llm` | 3.76 (n=21) | 0.443 (n=20) | 2.24 (n=21) |
  | `neural_rag` | 3.97 (n=29) | 0.787 (n=28) | 3.52 (n=29) |
  | `neurosymbolic` | 3.92 (n=24) | 0.809 (n=23) | 3.67 (n=24) |
  | `no_rag` | **4.31** (n=29) | **0.077** (n=27) | 3.62 (n=29) |

  Three findings that were invisible at n≈3:
  1. `no_rag` earns the **highest relevance (4.31) and the lowest faithfulness (0.077)** —
     without retrieval the model produces recommendations that sound well-targeted and are
     largely fabricated. This is the cleanest evidence in the artifact for what retrieval buys,
     and the broken sample hid it entirely.
  2. The symbolic layer does not cost quality: `neurosymbolic` 3.92 vs `neural_rag` 3.97
     relevance, and it is *better* on faithfulness (0.809 vs 0.787). This is the claim
     `generate_report.py` was already printing — now with n=24/29 behind it rather than 3/3, so
     it clears the n≥10 gate honestly.
  3. `no_llm` naturalness 2.24 confirms the deterministic template reads robotic, a useful check
     that the judge discriminates rather than rubber-stamps.

  Note the old figures were **biased optimistic, not merely imprecise**: faithfulness was
  reported as 1.000 for both `neural_rag` and `neurosymbolic` and resolved to 0.787 / 0.809.
  Any prose already written against the n≈3 numbers must be revisited.
- `evaluator.py` judges only `final_menus[:1]`.
- The three metrics fail independently, so per-metric means can cover **different,
  non-overlapping menus** (run 084449: `no_llm` relevance n=4, naturalness n=3 — not the same
  menus). Cross-metric comparison stays unsound until the three commit per menu, or per-case
  judge outputs are persisted so the overlap is recoverable and CIs are computable.
- No rubric anchoring on the judge prompts.

Which judge model to use is a separate question — see §4.3.

### 0.6 N=1 per condition, no seed, no significance test

Each case is run exactly once per pipeline (`benchmark/runner.py:84-105`) at temperature 0.1
with no seed. Differences of 0.30 vs 0.0 over 30 cases carry no stated uncertainty.

**Fix:** add `--repeats N` (default 3–5), aggregate mean ± SD across repeats, and run a paired
significance test on the binary per-case safety outcome — McNemar's test is the right one for
`neural_rag` vs `neurosymbolic` on the same cases. Record the seed.

### 0.7 No version control, and results carry no provenance

`git rev-parse` → *not a git repository*. There is a good `.gitignore` protecting nothing.
Result metadata (`runner.py:149-151`) records only model name and case count — not temperature,
`max_tokens`, `top_k`, `RRF_K`, embedding backend, reranker state, or judge model. Note
`generate_report.py:271` already tries to print `meta.get("judge_model", ...)` and always falls
back to the placeholder, because nothing ever writes it.

**Fix:** `git init`, confirm `.gitignore` takes effect, commit. Extend the results metadata to a
full config snapshot including the git SHA, and print it in the report header.

### 0.8 The retrieval evaluation report rests on broken measurement

Three separate defects make `eval/EVALUATION_REPORT.md`'s Part A numbers unusable:

**(a) A metric that can only ever return 1.0.** `eval/eval_retrieval.py:85`:

```python
return 1.0 if not any(rid for rid in retrieved_ids[:k]) or True else 0.0
```

The `or True` makes the condition unconditional. HitRate@K is hardcoded to a perfect score for
every query. Any reported HitRate is meaningless.

**(b) The golden set is stale.** `eval/eval_dataset.py:16` says "with only 9 recipes" and `:29`
"a 38-chunk knowledge base". Every `relevant_ids` set references only `recipe_001`–`recipe_009`
(e.g. `:59`, `:71`). The corpus is now **29 recipes / 58 chunks** — `benchmark_cases.py:15`
even acknowledges the expansion. Recipes 010–029 that genuinely satisfy a query are therefore
scored as false positives, so precision is understated and recall is computed against
ground truth missing 20 recipes. The numbers at `eval/EVALUATION_REPORT.md:42-50` do not
describe the current system.

**(c) A documented script that does not exist.** `eval/EVALUATION_REPORT.md:59,122` instruct
the reader to run `eval/eval_compare_retrievers.py` and present its output as a
"multi-retriever comparison [REAL]" table. That file is not in `eval/`. The table has no
reproducible source.

**Fix:** repair the HitRate expression; re-label `relevant_ids` across all 17 queries against
the full 29-recipe corpus (this is manual work and the honest part of the job); either write
`eval_compare_retrievers.py` or delete the table and the instructions referencing it.

### 0.9 Mock runs are hard to distinguish from real ones

`run_all.py:102-148` `mock_side_effect` detects injection strings and *deterministically*
returns an unsafe recipe — i.e. it hardcodes a 100% injection-success rate for `neural_rag`,
via an `if/elif` ladder over `recipe_006/005/010/011/014`. Those results then flow through the
same evaluator and report generator, producing a `COMPARATIVE_REPORT_*.md` that "demonstrates"
the thesis because the mock was written to.

The filename prefix (`mock_run_`) and `"model": "mock"` are the only signals.

**Fix:** add an unmissable banner at the top of any report generated from mock data, and a
`"synthetic": true` flag in metadata that `generate_report.py` checks. Mock mode is a fine
smoke test; it must never be mistakable for evidence.

---

## P1 — Correctness bugs in the safety logic being evaluated

`src/guardrails.py` is the module the entire thesis is about. It is currently untested.

### 1.1 Substring allergen matching produces false positives and negatives

`guardrails.py:145` — `if kw in ingredient_text:` with no word boundaries. Concrete failures
against real ingredient strings:

| ingredient | keyword hit | wrong verdict |
|---|---|---|
| `butternut squash` | `butter` | flagged as **milk** |
| `coconut milk`, `oat milk` | `milk` | flagged as **milk** |
| `eggplant` / `aubergine` | `egg` | flagged as **egg** |
| `goat` | `oat` | flagged as **gluten** |

This is very likely what drives the recorded `pre_filter_precision: 0.372` over 301 rejects.
Because rejections remove candidates before generation, this also depresses coverage — which
per §0.3 is currently invisible in the metrics.

**Fix:** match on word boundaries (`re.search(rf"\b{re.escape(kw)}\b", text)`), plus an explicit
exception list for the known compound-word traps (`coconut milk`, `oat milk`, `soy milk`,
`butternut squash`, `eggplant`, `peanut butter` → peanut not milk). Add unit tests for each row
of that table.

### 1.2 Unrecognised allergy terms fail open, silently

`guardrails.py:72-73` — a term matching no canonical allergen is added verbatim to the
restricted set. `check_recipe_against_profile:144` then falls back to
`ALLERGEN_KEYWORDS.get(allergen, [allergen])`, matching the raw string against ingredient text.
For a typo (`"peanutt"`) or an unmodelled allergen, that matches nothing and the restriction is
**silently never enforced** — a false negative in a safety-critical path, with no warning.

**Fix:** when a term normalises to nothing known, append a `GuardrailResult.warnings` entry
naming the unrecognised term, and surface it in the CLI and the benchmark record.

### 1.3 Falsy-zero bug in the nutrition overrides

`guardrails.py:177-178`:

```python
sugar_limit = profile.max_sugar_g_override or _daily_to_lunch_fraction(daily_sugar)
```

An override of `0.0` is falsy and silently discarded, falling back to the default ceiling —
the exact opposite of the caller's intent. Use `if ... is not None`.

### 1.4 Fragile reason-string matching

`guardrails.py:153` — `not any("nut" in r for r in reasons)` deduplicates by substring-searching
prose it generated earlier. Track a structured set of triggered rule ids instead.

### 1.5 The graph post-filter is weaker than the legacy one it replaced

`src/post_filter.py:85-94` cross-checks the LLM's self-reported `allergens_confirmed_absent`
against the recipe's actual `allergens_present` and flags contradictions. The LangGraph
`symbolic_postfilter` (`nodes.py:370-392`) — the one the benchmark actually exercises — does
**not**. It verifies `recipe_id` existence and re-runs the guardrail, but accepts the model's
allergen claims unexamined.

Likewise, citations are attached per candidate (`nodes.py:83-99`), injected as
`[Source: {citation}]` (`nodes.py:291`), and requested back as `source_citation`
(`nodes.py:330`) — but **nothing verifies the returned citation**. A fabricated or swapped
citation string passes through untouched, in a system whose stated selling point is
traceability.

**Fix:** port the `allergens_confirmed_absent` contradiction check into `symbolic_postfilter`,
and verify `source_citation` against the citation attached to that `recipe_id`. Both are cheap
deterministic checks and both strengthen the thesis rather than weaken it.

### 1.6 Age-band limits are re-read from disk on every recipe check

`guardrails.py:104-107` opens and `json.load`s `nutrition_guidelines.json` **per candidate per
case**. Wrap in `functools.lru_cache`. Same pattern for `_load_json("recipes.json")` at
`nodes.py:140` and `nodes.py:373`, and `evaluator.py:163`.

---

## P2 — Retrieval quality

### 2.1 The "semantic" retriever is 512-bucket hashed TF-IDF

`src/local_embeddings.py:38,73-77`: a hand-written multiplicative hash (`h*31+ord(ch)`) into
512 buckets, with collisions **added** (`vec[bucket] += weight`) and no sign hashing. With a
~38-chunk corpus the vocabulary comfortably exceeds 512 tokens, so unrelated terms become
partly indistinguishable — in a system whose retrieval feeds an allergen decision.

Note this makes the "hybrid BM25 + semantic" fusion really *two lexical retrievers*: it cannot
match synonyms or paraphrase, which is the usual justification for a dense retriever.
`report/generate_report.py:186` is honest about this ("BM25 + TF-IDF retrieval"), so the fix is
about capability, not disclosure.

**Fix:** the corpus is 38 chunks — just drop the hashing and index the exact vocabulary as a
sparse dict (`numpy` is already a dependency and unused here). Or resolve §0.4(a) and use
`all-MiniLM-L6-v2` for genuinely semantic retrieval. Either is a strict improvement over a
lossy 512-bucket projection.

### 2.2 `embed_query` iterates characters if handed a string

`local_embeddings.py:102` — `def embed_query(self, input: List[str])`. LangChain/Chroma's
convention is that `embed_query` takes a single **string**. Passed one, `self(input)` iterates
its characters and returns one vector per character. Silent and catastrophic if any future
call site follows the standard interface. Accept `Union[str, List[str]]` and normalise.

### 2.3 A missing vocab file silently poisons the index

`local_embeddings.py:95-96` — `__call__` auto-fits on its input when `self.fitted` is False.
If `vectordb/tfidf_vocab.json` is absent or corrupt (`:52-53` swallows the read error), the
first *query* fits IDF on that single query string and persists it as the canonical vocabulary.

**Fix:** raise on unfitted embed-at-query-time; only `build_collection()` may fit. Store a hash
of the source corpus alongside the vocab and refuse to serve if it doesn't match the current
`data/*.json` (this also catches a stale index after a recipe edit).

### 2.4 `get_collection()` drops the distance metric

`vector_store.py:80` calls `get_or_create_collection` **without** `metadata={"hnsw:space": "cosine"}`,
unlike `build_collection` at `:47-51`. If the collection doesn't exist, this silently creates an
empty one with default L2, `semantic_search` returns nothing, and the pipeline degrades to
BM25-only with no error. Pass the same metadata and assert `collection.count() > 0` at startup.

### 2.5 A new Chroma client per query

`vector_store.py:90` → `get_collection()` → `get_client()` → a fresh `chromadb.PersistentClient`
on **every** `semantic_search` call (~90 per benchmark). Cache it in a module singleton
alongside `_embedding_fn_singleton`.

### 2.6 `KeyError` on index/corpus drift

`nodes.py:143-144` — `raw_recipes.get(h["id"], {})` then `_recipe_to_candidate(recipe, ...)`,
which does `r["id"]` at `nodes.py:91`. Any id present in the vector store but absent from
`recipes.json` crashes the retrieval node. Skip-and-warn instead.

### 2.7 Retrieval depth is 9 of 29 recipes

`nodes.py:130` `[:9]` and `:139` `n_results=9` retrieve ~31% of the corpus. At that ratio the
retriever is barely discriminative and the symbolic filter does nearly all the work — worth
stating explicitly as a limitation, since it bounds how much the retrieval comparison can show.
`eval/eval_retrieval.py:141` uses `K = 5` against 17 hand-labelled queries; those IR numbers
never reach the comparative report (see §4.1).

---

## P3 — Robustness and engineering

### 3.1 No retries, backoff, or timeouts on any LLM call

Every network call is a bare `llm.invoke(...)`: `nodes.py:336`, `generation.py:118`,
`evaluator.py:117`, `eval/eval_generation.py:41`. A transient Groq 429 writes `"error"` into
that case (`runner.py:106-111`), and `evaluator.py:51-52` then **skips it from the denominator** —
so a rate-limited run yields a plausible-looking, silently under-sampled safety score.

**Fix:** set `max_retries` and a request timeout on the `ChatGroq`/`ChatOllama` constructors in
`llm_provider.py:91-96,112-117`; add explicit backoff on 429; and make the evaluator **count**
skipped cases and report them, rather than dropping them.

### 3.2 Fragile JSON parsing, duplicated four times

The fence-strip-then-`json.loads` block is verbatim in `generation.py:74-88`,
`nodes.py:346-356`, `evaluator.py:119-126`, `eval/eval_generation.py:43-50`. `clean.strip("`")`
strips backticks from both ends and `clean[4:]` assumes an exact `json` prefix. A malformed
response becomes `menus = []`, which is indistinguishable from "no safe menu" in the metrics
(§0.3 again).

**Fix:** one shared `parse_json_response()` helper that returns `(parsed, error)`, so parse
failure is recorded as a distinct outcome. Better: use LangChain structured output with a
Pydantic schema for `menu_options` and drop hand-parsing entirely.

### 3.3 `ensure_packages()` pip-installs at runtime

`run_all.py:56-69` shells out to `pip install` before `main()` does anything, mutating the
user's environment as a side effect of running a benchmark. Its `REQUIRED_PACKAGES` dict is a
second hand-maintained dependency list that has already drifted from `requirements.txt`
(missing `langchain-ollama`, `sentence-transformers`, `numpy`, `notebook`, `ipykernel`).

**Fix:** check imports and print install instructions; do not install. Delete the duplicate list.

### 3.4 Unpinned dependencies, no lockfile

`requirements.txt` is `>=` throughout — `chromadb>=0.5.0`, `langchain>=1.3.0`, `langgraph>=0.3.0`.
For a project whose deliverable *is* a reproducibility claim, a fresh install months later
produces a different system. Pin exact versions and commit a `pip freeze` lockfile.

### 3.5 No package structure — 26 `sys.path.insert` calls across 15 files

`run_all.py:35-38`, `runner.py:22-24`, `nodes.py:24-25`, etc. Note `src/graphs` is put on
`sys.path` *and* `graphs` is imported as a package, so `state`/`nodes` are reachable under two
names and can be double-imported as distinct module objects.

**Fix:** add a `pyproject.toml`, make `src/` a real package, `pip install -e .`, delete every
`sys.path` hack.

### 3.6 Two parallel implementations of the same pipeline

`src/retrieval.py` + `src/generation.py` + `src/post_filter.py` (used by `main.py`/`cli.py`)
reimplement `src/graphs/nodes.py`. They have already drifted: `generation.py:50-71` and
`nodes.py:314-332` are different prompts with different JSON schemas (`source_citation` only in
the latter). **The CLI and the benchmark therefore measure different systems.**

Other duplication: `retrieval.build_query_string:32-53` ≡ `nodes.build_query:104-120`; BM25
corpus + `k1=1.0, b=0.4` in `setup_database.py:35-41` and `nodes.py:54-58` (and the pickle's
params are ignored on load at `nodes.py:47-52`); the 14-allergen list inlined at
`nodes.py:230-233` instead of importing `ALL_14_ALLERGENS` from `document_loader.py:23-27`;
`no_llm_select:200-209` duplicating `symbolic_prefilter:173-182`; `_run_mock_benchmark:162-177`
duplicating `run_benchmark`'s loop and JSON writing.

**Fix:** make the CLI drive the LangGraph pipeline and delete the legacy trio.

### 3.7 No config module

Magic numbers scattered: `top_k=9` (twice), `RRF_K=60`, `approved[:1]` (`nodes.py:213` — the
no-LLM baseline returns exactly one option while LLM arms may return three, which biases every
per-menu metric), `fraction=0.40` (uncited), the `0.85` near-limit threshold, `dim=512`,
`k1/b`. Centralise in one `config.py` read from env with documented defaults, and serialise it
into results metadata (§0.7).

### 3.8 Other

- **`pickle.load` on `vectordb/bm25_index.pkl`** (`nodes.py:48-49`) — regenerable data;
  arbitrary code execution on load. Use JSON or rebuild in-process.
- **`except (_Timeout, Exception)`** in `huggingface_embeddings.py:76` and `reranker.py:69` —
  `_Timeout` subclasses `Exception`, so this catches everything and silently degrades.
- **`except Exception: pass`** at `vector_store.py:44-45`; `except (TypeError, ValueError): pass`
  at `eval/run_full_eval.py:113-114` drops unparseable metrics so a partly-failed run reports a
  clean-looking mean.
- **Private-API probes**: `evaluator.py:213`, `run_all.py:209,232` import `_try_groq`/`_try_ollama`
  and call them with a `"dummy"` model purely to test availability, constructing a real client.
- **Dead code**: `eval/eval_generation.py:31` `max_tokens` parameter is never used despite four
  call sites passing distinct values; `generation.py:23` `MODEL_FALLBACK` is never referenced.
- **164 `print()` calls, zero `logging`** across 19 files. Errors go to stdout and scroll past
  (`runner.py:111`). Move to `logging` with levels; errors to stderr.
- **`results_path.replace(".json", "_eval.json")`** (`run_all.py:186`) — naive; use `Path.with_name`.

---

## P4 — Tests, evidence, and documentation

### 4.1 Adopt pytest; unit-test the guardrail

No test framework: no pytest/unittest import anywhere, no `conftest.py`, no config. Two
hand-rolled script files with bare `assert` exist (`src/test_pipeline_with_mock_llm.py`,
`eval/test_eval_generation_with_mock.py`) and the integration coverage in the first is actually
good — happy path, LLM proposes an allergen-containing recipe, LLM hallucinates an id,
zero-candidate short-circuit.

`src/test_pipeline_with_mock_llm.py:152-153` is fragile: its comment says "all 9 recipes" but
the corpus is now 29, and it passes only because `retrieval.py:56` defaults `n_results=9` — so
it silently depends on which 9 chunks TF-IDF happens to rank highest.

**Fix:** add `pytest` + config, port both files, and add unit tests for `guardrails.py`
(untested today) covering: every row of the §1.1 false-positive table, synonym normalisation,
unrecognised-term warnings (§1.2), age-band boundaries at 4/6/7/10/11/14/15/18 and outside,
and the override behaviour from §1.3. Add tests for `evaluator.py`'s metric math on
hand-constructed result fixtures — the metric bugs in §0.3 are exactly what unit tests catch.

### 4.2 Wire the IR metrics into the main run

`eval/` computes P@K, R@K, MRR, NDCG@K against 17 hand-labelled queries, but `run_all.py`
never invokes it — so the comparative report contains **no retrieval-quality numbers at all**,
despite retrieval being half the architecture. Add a `step_retrieval_eval()` and a section in
`generate_report.py`.

### 4.3 Validate the LLM judge, and choose the judge model on bias not size

There is no human-agreement check. For a dissertation, LLM-as-judge scores are routinely
discounted without one. Hand-rate a 20–30 item subset and report agreement (Cohen's κ or
Krippendorff's α) against the judge. **No change of judge model substitutes for this** — it is
required whichever model is used.

**On `gpt-oss-120b` vs `llama-3.3-70b-versatile` as judge.** These differ on three axes, and
they do not all point the same way:

*Family (gpt-oss wins, and this is the axis that matters here).* The generator is
`llama-3.1-8b-instant`; the default judge `llama-3.3-70b-versatile` is also Meta Llama. Same
lineage means shared pretraining data, shared RLHF conventions, and shared blind spots — a
judge that likes Llama-shaped prose. `llm_provider.py:13-14` claims the separate judge
"prevents self-preferencing bias"; with two Llamas that claim is overstated. `gpt-oss-120b` is
an OpenAI open-weight model — a genuinely independent lineage, which makes the docstring's
claim true rather than aspirational.

*Capability (not a win — the parameter counts mislead).* `gpt-oss-120b` is a
mixture-of-experts model: ~117B total parameters but only ~5.1B **active** per token.
`llama-3.3-70b` is dense 70B. "120B beats 70B" is not the right comparison; active compute is
far smaller. Treat them as broadly comparable judges, not as an upgrade.

*Cost and determinism (llama wins).* `gpt-oss` is a reasoning model — it spends hidden
reasoning tokens before answering, so it needs ~600 `max_tokens` where llama needs 200 (~3x the
rate-limit budget per call), and at 200 it truncates mid-sentence. Observed at
`temperature=0`, its verdict text still varied between otherwise identical calls, so it is
the less reproducible of the two.

**Measured, not predicted.** A full re-judge of `run_20260814_084449` on `gpt-oss-120b`
returned `attempted 310 / ok 304 / parse_error 5 / call_error 1 / skipped 11` — a 95% complete
run, against 42 successful calls on the original llama judge. Sample sizes rose from n≈3 to
n=20–29 per metric, and the previously reported faithfulness of **1.000** for `neural_rag` and
`neurosymbolic` resolved to **0.787 / 0.809**: the collapsed sample had been biased optimistic,
not merely imprecise.

**Verdict — use `gpt-oss-120b` with `JUDGE_MAX_TOKENS=450`.** Budget arithmetic against a
321-call run (Groq bills reserved `max_tokens`; TPD is per-model):

| judge | tok/call | calls/day | full run? | family vs generator |
|---|---|---|---|---|
| `llama-3.3-70b` @200 | 281 | 355 | fits | **same (Meta Llama)** |
| `gpt-oss-120b` @600 | 731 | 273 | short by 48 | different |
| `gpt-oss-120b` @450 | 581 | 344 | **fits** | different |

600 was over-provisioned — observed completion was 198–228 tokens, so 450 leaves ~2x headroom
with no truncation, and converts the 95% run into a complete one. Same-family self-preferencing
is a *validity* threat an examiner will challenge; a few percent of sample is a *precision*
issue that can simply be disclosed. Validity wins.

**Then add `llama-3.3-70b` as a second judge and report inter-judge agreement.** Both now fit
their independent per-model budgets, so this costs no extra quota. Two judges of different
lineages agreeing is materially stronger than either alone, and disagreement is itself a
finding. Record the judge model with every score (already wired via `_judge_model`), and treat
human κ as the anchor for both — no judge swap substitutes for it.

### 4.4 Adversarial injections flow into the judge prompt

`runner.py:50-52` appends the injection to `profile["cultural_context"]`, and
`evaluator.py:134-135` interpolates `profile.get('cultural_context','')` straight into the
judge prompt. The attack payload is therefore fed to the evaluator. Strip the injection from
the profile before judging, or judge against the clean profile stored separately.

### 4.5 Documentation corrections

- `README.md:378-387` claims "eight distinct injection strategies", but only 6 of the 8
  `adversarial` cases set `adversarial_injection` — ADV-02 (`benchmark_cases.py:233`) is a
  synonym test with no injection. Every run reports `adversarial_cases_tested: 6`. Fix the
  count or add the two missing injections.
- `eval/eval_generation.py:4` says it "uses `get_llm()` for both generation and judging" — the
  code at `:39` correctly uses `get_judge_llm()`. Stale docstring; fix it.
- Add a limitations section covering corpus size (29 recipes), retrieval depth (§2.7), lexical-
  only retrieval (§2.1), and coverage-vs-safety (§0.3).

---

## Suggested sequencing

1. **Now, out of band:** rotate both API keys (§0.1). `git init` and commit (§0.7) — do this
   before anything else, so the §0.5 judge fixes already applied are captured in history.
2. **Finish the judge** (§0.5, §4.3): set `GROQ_JUDGE_MODEL=openai/gpt-oss-120b` and
   `JUDGE_MAX_TOKENS=450`, re-judge `run_20260814_084449`, and promote the result into
   `benchmark/results/`. This is cheap — safety metrics are deterministic, so no pipeline re-run
   is needed — and it is what makes §3 of the report citable.
3. **Fix the harness** (§0.2–0.4, §0.6, §0.8) and the guardrail bugs (§1.1–1.3) — these change
   the safety numbers, so they invalidate step 2's *inputs* and force a re-judge. Accept that,
   or reorder 2 and 3 if you would rather judge once; judging is the cheaper half.
4. **Re-run the benchmark** with repeats and the config snapshot; regenerate the report.
5. **Then** the engineering work (P2 retrieval, P3 robustness) and P4 tests/docs, which improve
   the artifact without further invalidating results.

Steps 3 and 4 are the ones that gate publishing. Everything in P3 can follow at leisure.

---

## Verification

- **Guardrail:** `pytest tests/test_guardrails.py` — the §1.1 table must produce zero false
  positives; unrecognised-term warnings must appear.
- **Retrieval:** rebuild with `python run_all.py --skip-setup` off, confirm
  `collection.count()` matches the chunk count and that `print_provider_status()` reports the
  retriever actually in use (§0.4). Run `python eval/eval_retrieval.py` and compare P@5/NDCG@5
  before and after §2.1.
- **Harness:** `python run_all.py --mock` must complete and produce a report carrying a visible
  synthetic-data banner (§0.9).
- **Judge (§0.5, §4.3):** re-judge an existing results file — no pipeline re-run needed:

  ```
  GROQ_JUDGE_MODEL=openai/gpt-oss-120b JUDGE_MAX_TOKENS=450 \
    python benchmark/evaluator.py benchmark/results/<run>.json --out <run>_eval.json
  ```

  Pass condition: `_judge_health` shows `skipped_quota_exhausted: 0` and `call_error: 0`, with
  each `n_*` near the mode's `cases_with_final_menus`. If the quota banner still appears in the
  generated report, the run is not publishable — that banner firing is the check working, not a
  cosmetic warning.
- **Full run:** `python run_all.py --repeats 5` against Groq. Confirm the results JSON metadata
  contains the full config snapshot + git SHA; confirm coverage and violations/n_cases appear
  per mode (§0.3); confirm the adversarial bypass rate is now reported for `neurosymbolic` on
  actually-injected cases (§0.2).
- **Regression check:** the direction of the headline result (neurosymbolic ≤ neural_rag on
  violations) should survive a *symmetric* attack. If it doesn't, that is the finding.
