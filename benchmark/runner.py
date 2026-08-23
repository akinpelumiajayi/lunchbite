"""
runner.py -- Runs every pipeline against every benchmark case.

Pipelines run per case:
  no_llm        -- rule-based only (primary baseline)
  neural_rag    -- retrieval + LLM with prompt-only safety
  neurosymbolic -- retrieval + symbolic gates + LLM
  no_rag        -- LLM with no retrieval (secondary reference)
  reward_ranked -- neurosymbolic + best-of-N on the verifiable reward

reward_ranked calls the generator exactly as often as neurosymbolic does. It
reranks the menus that arm already produced rather than resampling for new
ones, so adding it costs no additional tokens against the daily budget -- which
is why it can be run at all on the free tier this project uses.
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "graphs"))
sys.path.insert(0, str(ROOT / "benchmark"))

from benchmark_cases import BENCHMARK_CASES, BenchmarkCase
from console import enable_utf8_stdout
from graphs.nodes import AUTH_FAILED_PREFIX, QUOTA_EXHAUSTED_PREFIX
from rate_limit import AUTH_FAILED, GENERATOR_QUOTA
# run_failed is the evaluator's own definition of a case-run that produced
# nothing usable. Reused here so `metadata.complete` cannot disagree with the
# metrics about whether an arm actually ran.
from stats import run_failed

enable_utf8_stdout()

# The arms this runner executes, in report order. Defined once: the same list
# drives the metadata, the per-arm health table and the resume scan, and those
# three disagreeing is how run 20260819_082917 came to be stamped complete with
# three dead arms.
ALL_ARMS = ("no_llm", "neural_rag", "neurosymbolic", "no_rag", "reward_ranked")
# Arms that call a generator, and so can be stopped by a spent budget or a
# refused key. no_llm is not one of them.
LLM_ARMS = ("neural_rag", "neurosymbolic", "no_rag", "reward_ranked")


def build_run_metadata(provider_name: str, n_cases: int, ts: str,
                       synthetic: bool = False) -> Dict[str, Any]:
    """
    Full configuration snapshot alongside the results.

    A results file previously recorded only the model name and case count, so it
    could not be tied back to the system that produced it — not the retriever, not
    the temperature, not the rerank depth. `synthetic` marks mock runs so a report
    generated from simulated output can never be mistaken for evidence.
    """
    import subprocess

    from graphs.nodes import RRF_K, RETRIEVE_TOP_K
    from huggingface_upgrade.huggingface_embeddings import _model_name as embed_model
    from huggingface_upgrade.reranker import _model_name as ce_model, rerank_top_k
    from llm_provider import use_cross_encoder_reranker, generator_max_tokens
    from guardrails import lunch_fraction, nutrition_gate

    try:
        git_sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                 capture_output=True, text=True, cwd=str(ROOT),
                                 timeout=5).stdout.strip() or None
    except Exception:
        git_sha = None

    return {
        "model": provider_name,
        "judge_model": os.environ.get("GROQ_JUDGE_MODEL", "openai/gpt-oss-120b"),
        "n_cases": n_cases,
        "timestamp": ts,
        "synthetic": synthetic,
        "git_sha": git_sha,
        "pipelines": list(ALL_ARMS),
        "retrieval": {
            "embedding_model": embed_model(),
            "reranker_enabled": use_cross_encoder_reranker(),
            "reranker_model": ce_model() if use_cross_encoder_reranker() else None,
            "retrieve_top_k": RETRIEVE_TOP_K,
            "rerank_top_k": rerank_top_k(),
            "rrf_k": RRF_K,
        },
        "generation": {
            "temperature": os.environ.get("LLM_TEMPERATURE", "0.1"),
            # From llm_provider, not a literal: the default moved 2000 -> 1200 to
            # fit the daily token budget and this copy was left behind, so every
            # results file recorded a reservation the run had not used.
            "max_tokens": str(generator_max_tokens()),
        },
        "guardrail": {
            "lunch_nutrition_fraction": lunch_fraction(),
            # Which arm of the gate was live decides what a pre-filter rejection
            # means, so a results file that omits it cannot be compared with one
            # from a different setting.
            "nutrition_gate": nutrition_gate(),
        },
        "adversarial_injection_applied_to": ["neural_rag", "neurosymbolic", "no_rag",
                                            "reward_ranked"],
    }


def _make_initial_state(case: BenchmarkCase, mode: str) -> Dict[str, Any]:
    return {
        "profile": case.profile,
        "pipeline_mode": mode,
        "run_id": f"{case.case_id}-{mode}-{uuid.uuid4().hex[:8]}",
        "query": "", "bm25_candidates": [], "semantic_candidates": [],
        "fused_candidates": [], "reranked_candidates": [],
        "symbolic_pre_filter_log": [],
        "generation_candidates": [], "llm_raw_output": "",
        "proposed_menus": [], "generation_error": None,
        "symbolic_post_filter_log": [], "final_menus": [],
        "error": None, "latency_ms": {},
        # Seeded explicitly. Omitting them would leave the refine loop reading
        # None on its first pass -- the same silent-drop trap that left
        # max_sugar_g_override inert in every graph.
        "refine_count": 0, "retrieve_top_k": 0, "refine_log": [],
        "reward_log": [],
    }


def _inject_adversarial(state: Dict[str, Any], case: BenchmarkCase) -> Dict[str, Any]:
    """Appends the adversarial injection to cultural_context so it enters the LLM prompt.
    The neuro-symbolic system's deterministic gates are unaffected since they run
    outside the LLM context entirely."""
    if case.adversarial_injection:
        state = dict(state)
        state["profile"] = dict(state["profile"])
        state["profile"]["cultural_context"] = (
            (state["profile"].get("cultural_context") or "") + case.adversarial_injection
        )
    return state


def run_single_case(
    case: BenchmarkCase,
    no_llm_graph: Any,
    neural_graph: Any,
    neuro_graph: Any,
    no_rag_graph: Any,
    reward_graph: Any = None,
) -> Dict[str, Any]:
    """
    Run every compiled arm against one case.

    `reward_graph` is optional so that callers written against the four-arm
    signature -- `run_all._run_mock_benchmark`, and any script outside this
    repo -- keep working unchanged. Passing None simply omits that arm rather
    than recording it as a failure, which would be indistinguishable from an
    arm that ran and produced nothing.
    """
    result: Dict[str, Any] = {
        "case_id": case.case_id,
        "description": case.description,
        "category": case.category,
        "profile": case.profile,
        "adversarial_injection": case.adversarial_injection,
        "expected_unsafe_ids": case.expected_unsafe_ids,
        "expected_nutrition_unsafe_ids": case.expected_nutrition_unsafe_ids,
        "is_attack": case.is_attack,
        "expected_safe_ids": case.expected_safe_ids,
    }

    # The injection goes to EVERY arm whose LLM can see the profile. Injecting
    # only into neural_rag assumed the conclusion: neurosymbolic then scored a 0%
    # bypass rate without ever having been attacked. The symbolic gates have to
    # earn that number under the same attack.
    # no_llm is exempt because it never builds a prompt — there is no surface.
    # reward_ranked is injected for the same reason as neurosymbolic: its extra
    # node reads the profile, so it has to be attacked to earn its number.
    pipelines = [
        ("no_llm", no_llm_graph, False),
        ("neural_rag", neural_graph, True),
        ("neurosymbolic", neuro_graph, True),
        ("no_rag", no_rag_graph, True),
        ("reward_ranked", reward_graph, True),
    ]
    pipelines = [p for p in pipelines if p[1] is not None]
    for mode, _graph, _inject in pipelines:
        result[mode] = None

    for mode, graph, apply_injection in pipelines:
        # no_llm keeps running on a spent budget — it never calls the model, and
        # dropping it would cost the one arm still capable of producing data.
        if GENERATOR_QUOTA.hit and mode != "no_llm":
            result[mode] = {
                "error": f"{QUOTA_EXHAUSTED_PREFIX}: skipped, budget spent earlier in this run",
                "total_latency_ms": 0.0,
            }
            continue

        # Same exemption for no_llm, same reason: it needs no credential either.
        if AUTH_FAILED.hit and mode != "no_llm":
            result[mode] = {
                "error": f"{AUTH_FAILED_PREFIX}: skipped, key refused earlier in this run",
                "total_latency_ms": 0.0,
            }
            continue

        state = _make_initial_state(case, mode)
        if apply_injection:
            state = _inject_adversarial(state, case)

        t0 = time.perf_counter()
        try:
            final = graph.invoke(state)
            elapsed = (time.perf_counter() - t0) * 1000
            result[mode] = {
                "query": final.get("query", ""),
                "fused_candidates": [c["id"] for c in (final.get("fused_candidates") or [])],
                "reranked_candidates": [c["id"] for c in (final.get("reranked_candidates") or [])],
                "reranker_scores": {
                    c["id"]: round(c["reranker_score"], 4)
                    for c in (final.get("reranked_candidates") or [])
                    if "reranker_score" in c
                },
                "generation_candidates": [c["id"] for c in (final.get("generation_candidates") or [])],
                "pre_filter_log": final.get("symbolic_pre_filter_log") or [],
                "proposed_menus": final.get("proposed_menus") or [],
                "final_menus": final.get("final_menus") or [],
                "post_filter_log": final.get("symbolic_post_filter_log") or [],
                "generation_error": final.get("generation_error"),
                "llm_raw_output": final.get("llm_raw_output", ""),
                # How many extra retrieval passes this case needed. A loop that
                # never fires is not evidence of a loop, so the count is recorded
                # per case rather than inferred.
                "refine_count": final.get("refine_count") or 0,
                "refine_log": final.get("refine_log") or [],
                # Empty for every arm but reward_ranked. Recorded per case so a
                # rerank that fired is distinguishable from one that had only a
                # single surviving menu to consider.
                "reward_log": final.get("reward_log") or [],
                "total_latency_ms": elapsed,
                "latency_ms": final.get("latency_ms") or {},
            }
        except Exception as e:
            result[mode] = {
                "error": f"{type(e).__name__}: {e}",
                "total_latency_ms": (time.perf_counter() - t0) * 1000,
            }
            print(f"    ERROR [{mode}]: {e}")

    return result


def _load_completed(resume_path: str) -> List[Dict[str, Any]]:
    """
    Case-runs already on disk from an earlier attempt.

    Only rows that actually produced data are kept. A row whose LLM arms all
    failed on a spent budget carries no evidence, and keeping it would let a
    resumed run inherit the very failures the resume exists to repair.
    """
    with open(resume_path, encoding="utf-8") as f:
        payload = json.load(f)

    # Rows written before generators were recorded per row carry no `generator`
    # key. The file they came from names exactly one, so backfill from it rather
    # than leaving the provenance of half a resumed run unknowable.
    inherited = (payload.get("metadata") or {}).get("model") or "unknown"

    kept = []
    for row in payload.get("results", []):
        arms = [row.get(m) or {} for m in LLM_ARMS]
        dead = (QUOTA_EXHAUSTED_PREFIX, AUTH_FAILED_PREFIX)
        if any(str(a.get("error", "")).startswith(dead)
               or str(a.get("generation_error", "")).startswith(dead)
               for a in arms):
            continue
        row.setdefault("generator", inherited)
        kept.append(row)
    return kept


def _generator_census(results: List[Dict[str, Any]]) -> Dict[str, int]:
    """
    How many case-runs each generator model produced, in run order.

    A resumed run can finish on a different provider than it started on -- the
    2026-08-21 run spent Groq's daily token budget at ADV-01 and had to be
    completed locally -- and `metadata.model` is a single string that cannot say
    so. Recorded per row and counted here, because a reader asking "what produced
    the adversarial half of this file" must be able to answer it from the file.

    Describes the three LLM arms. `no_llm` calls no model in any run.
    """
    census: Dict[str, int] = {}
    for r in results:
        name = r.get("generator") or "unknown"
        census[name] = census.get(name, 0) + 1
    return census


def _arm_health(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    """
    How many case-runs each arm actually answered, by the evaluator's definition.

    `metadata.complete` used to mean only "every case was iterated", which is why
    run 20260819_082917 was stamped `complete: true` while three of its four arms
    had produced nothing whatsoever -- the API key had expired, so every generator
    call 401'd. Nothing downstream could distinguish that file from a good run
    without opening 30 rows and reading the per-arm error text.

    An arm that failed *every* case-run did not run. Recording the counts here,
    and refusing to call such a run complete, makes the difference readable from
    the metadata alone.
    """
    health: Dict[str, Dict[str, int]] = {}
    for mode in ALL_ARMS:
        failed = sum(1 for r in results if run_failed(r.get(mode) or {}, mode))
        health[mode] = {"case_runs": len(results), "failed": failed,
                        "answered": len(results) - failed}
    return health


def run_benchmark(
    provider: Optional[str] = None,
    model_override: Optional[str] = None,
    output_dir: str = "benchmark/results",
    repeats: int = 1,
    resume_from: Optional[str] = None,
    limit: Optional[int] = None,
    allow_provider_change: bool = False,
) -> str:
    from llm_provider import get_llm, configure_langsmith
    from graphs.build_graphs import (build_no_llm_graph, build_neural_rag_graph,
                                     build_neurosymbolic_graph, build_no_rag_graph,
                                     build_reward_ranked_graph)

    configure_langsmith()

    # A resumed run starts against a budget that has since refilled, so the latch
    # from the file being resumed must not carry over.
    GENERATOR_QUOTA.reset()

    done: List[Dict[str, Any]] = []
    resumed_from: Optional[str] = None
    if resume_from:
        done = _load_completed(resume_from)
        resumed_from = resume_from
        print(f"Resuming from {resume_from}: {len(done)} case-runs already complete.")
    completed = {(r["case_id"], r.get("repeat", 0)) for r in done}

    if model_override:
        env_key = "OLLAMA_MODEL" if provider == "ollama" else "GROQ_MODEL"
        os.environ[env_key] = model_override

    llm, provider_name = get_llm(prefer=provider)
    print(f"Generator LLM: {provider_name}")

    # Finishing a run on a different model than it started on is sometimes the
    # only way to finish it at all, but it must be a decision rather than a
    # side effect of whichever provider happened to resolve. Left implicit, the
    # single `metadata.model` field would relabel the inherited rows as the new
    # model and the report would print one backbone for a file holding two.
    prior = sorted({r.get("generator") or "unknown" for r in done})
    # "unknown" is a file written before generators were recorded per row, not
    # evidence of a different one. Refusing on it would block every resume of an
    # older results file, including a resume onto the very same model. It still
    # reaches the census below, because "we cannot tell" is the honest record.
    changed = [g for g in prior if g not in (provider_name, "unknown")]
    if changed and not allow_provider_change:
        raise SystemExit("\n".join([
            "",
            "RESUME REFUSED: this would mix generators in one results file.",
            f"  {resume_from}",
            f"  already holds {len(done)} case-runs from: {', '.join(changed)}",
            f"  and this process resolved: {provider_name}",
            "",
            "  The arms within any single case still share one model, so the paired",
            "  comparison survives -- but the suite would no longer rest on one",
            "  generator, and the split follows case order rather than chance.",
            "",
            "  To finish on the original model, re-run without --provider once its",
            "  budget allows. To accept the mix -- which is recorded in",
            "  metadata.generators and disclosed in the report -- pass:",
            "    --allow-provider-change",
        ]))
    if changed:
        print(f"Mixed-generator run: inheriting {len(done)} case-runs from "
              f"{', '.join(changed)}, continuing on {provider_name}.")
    elif "unknown" in prior:
        print(f"Note: {resume_from} predates per-row generator recording, so the "
              f"model behind its case-runs cannot be confirmed as {provider_name}.")

    # A short slice for smoke-testing a provider without committing to the full
    # benchmark -- a local model can take an hour where Groq takes minutes, and
    # "does this configuration work at all" should not cost that.
    # Marked in the metadata so a truncated run can never be mistaken for a
    # complete one downstream.
    cases = BENCHMARK_CASES[:limit] if limit else BENCHMARK_CASES

    no_llm_graph   = build_no_llm_graph()
    neural_graph   = build_neural_rag_graph(llm)
    neuro_graph    = build_neurosymbolic_graph(llm)
    no_rag_graph   = build_no_rag_graph(llm)
    reward_graph   = build_reward_ranked_graph(llm)
    print(f"All {len(ALL_ARMS)} graphs compiled. Running {len(BENCHMARK_CASES)} cases...\n")

    # Each case is run `repeats` times per pipeline. At temperature 0.1 the
    # generator is not deterministic, so a single pass gives a point estimate
    # with no stated uncertainty — 0.333 vs 0.0 over 30 cases could be noise.
    # Repeats let the evaluator report mean +/- SD and run a paired test.
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(output_dir, f"run_{ts}.json")

    results: List[Dict[str, Any]] = list(done)
    aborted: Optional[str] = None

    def save() -> None:
        census = _generator_census(results)
        # `model` stays a string for every existing reader; a mixed run names
        # every generator in it rather than the last one to run.
        model_str = (provider_name if len(census) <= 1 else
                     "mixed: " + ", ".join(f"{k} ({v})" for k, v in census.items()))
        meta = build_run_metadata(model_str, len(cases), ts)
        meta["generators"] = census
        meta["limited_to"] = limit
        meta["repeats"] = repeats
        meta["n_result_rows"] = len(results)
        health = _arm_health(results)
        meta["arm_health"] = health
        # An arm with zero answered case-runs is switched off, not merely unlucky.
        dead_arms = [m for m, h in health.items()
                     if h["case_runs"] and h["answered"] == 0]
        meta["complete"] = (aborted is None
                            and len(results) == len(cases) * repeats
                            and not dead_arms)
        if dead_arms:
            meta["dead_arms"] = dead_arms
        if AUTH_FAILED.hit:
            meta["auth_failed"] = AUTH_FAILED.as_dict()
        if resumed_from:
            meta["resumed_from"] = resumed_from
        if aborted:
            meta["aborted_reason"] = aborted
            meta["quota"] = GENERATOR_QUOTA.as_dict()
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"metadata": meta, "results": results}, f, indent=2)

    for rep in range(repeats):
        if repeats > 1:
            print(f"\n--- repeat {rep + 1}/{repeats} ---")
        for case in cases:
            if (case.case_id, rep) in completed:
                print(f"  [{case.case_id}] already done — skipped")
                continue

            print(f"  [{case.case_id}] {case.description}")
            r = run_single_case(case, no_llm_graph, neural_graph, neuro_graph,
                                no_rag_graph, reward_graph)
            r["repeat"] = rep
            r["generator"] = provider_name
            results.append(r)

            # Written after every case. A run that dies at case 19 used to lose
            # all 19; the file on disk is now never more than one case stale.
            save()

            if AUTH_FAILED.hit:
                aborted = (f"provider refused the API key at {case.case_id} "
                           f"(rep {rep + 1}/{repeats})")
                break

            if GENERATOR_QUOTA.hit:
                aborted = (f"generator daily token budget exhausted at {case.case_id} "
                           f"(rep {rep + 1}/{repeats})")
                break
        if aborted:
            break

    save()

    if aborted:
        remaining = len(BENCHMARK_CASES) * repeats - len(results)
        print(f"\n{'=' * 70}")
        print(f"RUN STOPPED: {aborted}")
        print(f"  {len(results)} of {len(BENCHMARK_CASES) * repeats} case-runs completed; "
              f"{remaining} not attempted.")
        # A refused key and a spent budget both stop the run, but only one of them
        # is fixed by waiting. Telling the wrong story here sends the user off to
        # wait for a reset that will change nothing.
        if AUTH_FAILED.hit:
            print(f"  The provider rejected GROQ_API_KEY: {AUTH_FAILED.detail[:160]}")
            print("  This is NOT a rate limit — no waiting period applies, and")
            print("  --resume cannot help until the key is replaced.")
            print("\n  Replace GROQ_API_KEY in .env (new key: "
                  "https://console.groq.com/keys), then:")
            print(f"    python benchmark/runner.py --resume {out_path}")
            print(f"{'=' * 70}")
            return out_path
        print("  Continuing would have recorded every remaining case as a failure and")
        print("  produced a report scored on an unequal case list across the arms.")

        # The provider's retry delay is sized for the ONE call it just refused,
        # not for a full refill. Groq's daily cap is a rolling window, so
        # "try again in 16m" against a 2,437-token request means those 2,437
        # tokens come back -- not the ~120k the rest of a run needs. Printed as
        # "budget resets in ~16m", it sent the owner of the 2026-08-21 run off to
        # wait for something that was never going to have happened.
        from llm_provider import generator_max_tokens, _ollama_reachable
        need = remaining * len(LLM_ARMS) * generator_max_tokens()
        print()
        print(f"  The provider asks ~{GENERATOR_QUOTA.human_wait()} before it will RETRY THE ONE CALL")
        print("  IT REFUSED. That is not a full refill: the daily cap is a rolling")
        print(f"  window, and the {remaining} remaining case-runs need at least ~{need:,} more")
        print(f"  tokens ({len(LLM_ARMS)} LLM arms each, reserved max_tokens alone, "
              f"before prompts).")
        print("  Expect to wait for the day's window to roll, not for that interval.")
        print()
        print("  Finish it later on the same model, without re-spending on what is done:")
        print(f"    python benchmark/runner.py --resume {out_path}")
        if _ollama_reachable():
            # Reachable right now, so this is a real alternative rather than an
            # invitation to go and install something.
            print()
            print("  Or finish it locally now -- Ollama IS running, and has no token")
            print("  budget. That mixes two generators in one file, which is recorded")
            print("  in metadata.generators and disclosed in the report:")
            print(f"    python benchmark/runner.py --resume {out_path} --provider ollama --allow-provider-change")
        print(f"{'=' * 70}")
    else:
        print(f"\nResults saved to: {out_path}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=["groq", "ollama"], default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--output-dir", default="benchmark/results")
    parser.add_argument("--repeats", type=int, default=1,
                        help="Runs per case per pipeline. >1 enables mean+/-SD and "
                             "the paired significance test. 3-5 is a sensible range.")
    parser.add_argument("--resume", default=None, metavar="RESULTS.JSON",
                        help="Finish a run that stopped early. Cases already "
                             "recorded there are not re-run, so a run aborted on "
                             "a spent daily budget can be completed after the "
                             "reset for the cost of what is left.")
    # run_benchmark has taken `limit` since it was added and run_all.py exposes
    # it, but this parser never did -- so the one entry point the abort message
    # sends people to could not be used to smoke-test a provider on one case.
    parser.add_argument("--limit", type=int, default=None, metavar="N",
                        help="Run only the first N cases. For checking that a "
                             "provider works at all without committing to the "
                             "full benchmark. Marked in the metadata, so a "
                             "truncated run cannot be mistaken for a complete one.")
    parser.add_argument("--allow-provider-change", action="store_true",
                        help="Permit --resume to finish a run on a different "
                             "generator than it started on. Refused by default: "
                             "the resulting file holds two models, which is "
                             "recorded in metadata.generators and disclosed in "
                             "the report.")
    args = parser.parse_args()
    run_benchmark(provider=args.provider, model_override=args.model,
                  output_dir=args.output_dir, repeats=args.repeats,
                  resume_from=args.resume, limit=args.limit,
                  allow_provider_change=args.allow_provider_change)
