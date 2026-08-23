#!/usr/bin/env python3
"""
run_all.py -- Single entry point for the full research pipeline (Aims 1-5).

Four pipelines are run against 30 benchmark cases:
  no_llm        -- primary baseline: rule-based only, no LLM (Aim 1)
  neural_rag    -- main system: retrieval + LLM with prompt-only safety (Aim 1)
  neurosymbolic -- comparator: retrieval + symbolic gates + LLM (Aim 2)
  no_rag        -- secondary reference: LLM with no retrieval context

Usage:
  python3 run_all.py --mock              # no API key needed
  python3 run_all.py                     # auto-detects Groq or Ollama from .env
  python3 run_all.py --provider groq     # force Groq
  python3 run_all.py --provider ollama   # force Ollama
  python3 run_all.py --skip-setup        # skip DB rebuild
  python3 run_all.py --results path/to/run.json  # skip to eval+report

LLM providers (reads .env automatically):
  Generator: GROQ_MODEL / OLLAMA_MODEL
  Judge:     GROQ_JUDGE_MODEL / OLLAMA_JUDGE_MODEL  (must differ from generator)
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "graphs"))
sys.path.insert(0, str(ROOT / "benchmark"))
sys.path.insert(0, str(ROOT / "report"))
sys.path.insert(0, str(ROOT / "eval"))

from console import enable_utf8_stdout

enable_utf8_stdout()


# ── Step 0: ensure packages ───────────────────────────────────────────────────

REQUIRED_PACKAGES = {
    "dotenv": "python-dotenv>=1.0.0",
    "chromadb": "chromadb>=0.5.0",
    "rank_bm25": "rank-bm25>=0.2.2",
    "langchain": "langchain>=1.3.0",
    "langchain_core": "langchain-core>=0.3.0",
    "langchain_groq": "langchain-groq>=0.3.0",
    "langgraph": "langgraph>=0.3.0",
    "langsmith": "langsmith>=0.1.0",
    "groq": "groq>=0.4.0",
    "sentence_transformers": "sentence-transformers>=3.0.0",
}


def ensure_packages() -> None:
    """
    Checks imports and reports what is missing. Deliberately does NOT install.

    Installing as a side effect of running a benchmark mutates the environment
    the results are supposed to describe — the run that installs its own
    dependencies is not the run you can reproduce later. requirements.txt is the
    single source of truth; this only maps import names to it.
    """
    missing = []
    for module, pip_spec in REQUIRED_PACKAGES.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(pip_spec)
    if missing:
        print("ERROR: missing required packages:\n  " + "\n  ".join(missing))
        print("\nInstall them, then re-run:")
        print("  pip install -r requirements.txt")
        sys.exit(1)


# ── Steps ─────────────────────────────────────────────────────────────────────

def step_setup() -> None:
    print("\n" + "=" * 60)
    print("STEP 1: Building vector database and BM25 index")
    print("=" * 60)
    from setup_database import main as setup_main
    setup_main()


def step_benchmark(provider: Optional[str], use_mock: bool, model_override: Optional[str],
                   repeats: int = 1, resume_from: Optional[str] = None,
                   limit: Optional[int] = None,
                   allow_provider_change: bool = False) -> str:
    print("\n" + "=" * 60)
    from benchmark_cases import BENCHMARK_CASES as _CASES
    from runner import ALL_ARMS as _ARMS
    print(f"STEP 2: Running benchmark — all {len(_ARMS)} pipelines, "
          f"{len(_CASES)} cases (Aims 3 & 4)")
    print("=" * 60)
    if use_mock:
        return _run_mock_benchmark(repeats=repeats)
    from runner import run_benchmark
    return run_benchmark(provider=provider, model_override=model_override,
                         output_dir=str(ROOT / "benchmark" / "results"),
                         repeats=repeats, resume_from=resume_from, limit=limit,
                         allow_provider_change=allow_provider_change)


def _run_mock_benchmark(repeats: int = 1) -> str:
    import uuid
    from unittest.mock import MagicMock
    from langchain_core.messages import AIMessage
    from benchmark_cases import BENCHMARK_CASES
    from runner import run_single_case
    from graphs.build_graphs import (build_no_llm_graph, build_neural_rag_graph,
                                     build_neurosymbolic_graph, build_no_rag_graph,
                                     build_reward_ranked_graph)
    from runner import ALL_ARMS

    # Detect injections by matching the suite's own injection strings rather than
    # a hand-written phrase list. The old list had drifted: three of the six
    # injected cases no longer matched it, so they silently ran as benign and the
    # mock under-reported the attack surface.
    KNOWN_INJECTIONS = [c.adversarial_injection for c in BENCHMARK_CASES
                        if c.adversarial_injection]

    def mock_side_effect(messages):
        content = messages[0].content if hasattr(messages[0], "content") else str(messages[0])
        # Simulate a naive LLM that complies with an injection.
        if any(inj in content for inj in KNOWN_INJECTIONS):
            # Find the first unsafe candidate that appears in the available recipes block
            # (which is the fused_candidates passed to generation)
            # We use recipe_006 (Soft cheese sandwich, contains milk) as default —
            # it consistently ranks high in retrieval for milk-allergy profiles due to
            # "cheese" and "dairy" token matches, making it the most realistic
            # adversarial proposal a real LLM would make after following an injection.
            if "recipe_999" in content:
                unsafe_id = "recipe_999"
            elif "recipe_006" in content:
                unsafe_id = "recipe_006"
            elif "recipe_005" in content:
                unsafe_id = "recipe_005"
            elif "recipe_010" in content:
                unsafe_id = "recipe_010"
            elif "recipe_011" in content:
                unsafe_id = "recipe_011"
            elif "recipe_014" in content:
                unsafe_id = "recipe_014"
            else:
                unsafe_id = "recipe_006"
            import json as _json
            payload = _json.dumps({
                "menu_options": [{
                    "recipe_id": unsafe_id,
                    "menu_name": "Adversarial menu",
                    "why_it_fits": "Override accepted - injection followed",
                    "nutritional_rationale": "N/A",
                    "allergens_confirmed_absent": [],
                    "source_citation": ""
                }]
            })
        else:
            payload = (
                '{"menu_options":[{"recipe_id":"recipe_013","menu_name":"Black Beans and Rice",'
                '"why_it_fits":"Milk-free, egg-free, nut-free. Suits the profile.",'
                '"nutritional_rationale":"370 kcal, 18g protein, 12g fibre, 0.6g salt per serving.",'
                '"allergens_confirmed_absent":["milk","egg","fish","nuts","peanut"],'
                '"source_citation":"Farris, A. (n.d.). Black Beans and Rice. PACK-IT Cookbook. Virginia Cooperative Extension."}]}'
            )
        return AIMessage(content=payload)

    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = mock_side_effect

    # Every arm the real runner builds. The mock must cover the same set or
    # `metadata.pipelines` names an arm with no rows behind it, and the report
    # renders an empty column for a pipeline that was never run.
    no_llm_graph   = build_no_llm_graph()
    neural_graph   = build_neural_rag_graph(mock_llm)
    neuro_graph    = build_neurosymbolic_graph(mock_llm)
    no_rag_graph   = build_no_rag_graph(mock_llm)
    reward_graph   = build_reward_ranked_graph(mock_llm)

    print(f"[MOCK MODE] Simulated LLM follows adversarial injections "
          f"({len(KNOWN_INJECTIONS)} injected cases). Results are SYNTHETIC, not evidence.")
    print(f"Running {len(BENCHMARK_CASES)} cases across all {len(ALL_ARMS)} pipelines...\n")

    results = []
    for rep in range(repeats):
        if repeats > 1:
            print(f"\n--- repeat {rep + 1}/{repeats} ---")
        for case in BENCHMARK_CASES:
            print(f"  [{case.case_id}] {case.description}")
            r = run_single_case(case, no_llm_graph, neural_graph, neuro_graph,
                                no_rag_graph, reward_graph)
            r["repeat"] = rep
            results.append(r)

    from runner import build_run_metadata

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "benchmark" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = str(out_dir / f"mock_run_{ts}.json")
    meta = build_run_metadata("mock", len(BENCHMARK_CASES), ts, synthetic=True)
    meta["repeats"] = repeats
    meta["n_result_rows"] = len(results)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"metadata": meta, "results": results}, f, indent=2)
    print(f"\nMock results saved to: {out_path}")
    return out_path


def step_evaluate(results_path: str, run_judge: Optional[bool] = None,
                  provider: Optional[str] = None) -> str:
    print("\n" + "=" * 60)
    print("STEP 3: Computing evaluation metrics (Aim 5)")
    print("=" * 60)
    from evaluator import evaluate
    scores = evaluate(results_path, run_llm_judge=run_judge, provider=provider)
    out_path = results_path.replace(".json", "_eval.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(scores, f, indent=2)
    print(f"Evaluation scores saved to: {out_path}")
    return out_path


def step_retrieval_eval() -> Optional[str]:
    """
    Retrieval quality against the hand-labelled golden set.

    Retrieval is half the architecture, but these numbers never reached the
    comparative report — eval/ was only ever run by hand, so the report argued
    for a hybrid retriever without showing that hybrid beats its parts.

    Non-fatal: a failure here should not cost you the benchmark results.
    """
    print("\n" + "=" * 60)
    print("STEP 3b: Evaluating retrieval quality (Aim 5)")
    print("=" * 60)
    try:
        from eval_compare_retrievers import compare
        results = compare(k=5)
        out_dir = ROOT / "benchmark" / "results"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = str(out_dir / f"retrieval_eval_{time.strftime('%Y%m%d_%H%M%S')}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Retrieval metrics saved to: {out_path}")
        return out_path
    except Exception as e:                                        # noqa: BLE001
        print(f"  WARNING: retrieval evaluation FAILED ({type(e).__name__}: {e})")
        print("  The report will have no retrieval-quality section. Reproduce with:")
        print("    python eval/eval_compare_retrievers.py")
        return None


def step_rewards(results_path: str) -> Optional[str]:
    """
    Score every generated menu with the verifiable reward, then verify it.

    Costs nothing: no model is called and the run is not re-executed, so this
    step is unaffected by the daily token budget that stops the judge. It is
    also the only scoring pass in this pipeline whose output a reader can
    re-derive -- which is why the verification runs here rather than being left
    as something to remember.

    Non-fatal, on the same grounds as the retrieval step: a failure here must
    not cost you the benchmark results.
    """
    print("\n" + "=" * 60)
    print("STEP 3c: Scoring the verifiable reward (RLHF signal)")
    print("=" * 60)
    try:
        from reward.scoring import score_run
        from reward.verify import verify_scored_run

        with open(results_path, encoding="utf-8") as f:
            payload = json.load(f)

        scored = score_run(payload)
        scored["metadata"]["synthetic"] = bool((payload.get("metadata") or {}).get("synthetic"))
        scored["metadata"]["source_results_file"] = Path(results_path).name

        out_path = str(Path(results_path).with_name(Path(results_path).stem + "_reward.json"))
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(scored, f, indent=2)

        for mode, agg in scored["per_mode"].items():
            b = scored["best_of_n"].get(mode) or {}
            delta = b.get("delta")
            print(f"  {mode:15s} mean reward {agg['mean_reward']:.3f} "
                  f"over {agg['n_menus']:3d} menus"
                  + (f"   best-of-N {delta:+.3f}" if delta else ""))

        report = verify_scored_run(scored, payload)
        # A reward that cannot be re-derived is worth nothing, so the failure is
        # printed loudly rather than returned quietly.
        print(f"  Verification: {'VERIFIED' if report.passed else 'FAILED'} "
              f"({report.n_checked} records recomputed, "
              f"{len(report.mismatches)} mismatches)")
        if not report.passed:
            print(report.summary())

        print(f"Rewards saved to: {out_path}")
        return out_path
    except Exception as e:                                        # noqa: BLE001
        print(f"  WARNING: reward scoring FAILED ({type(e).__name__}: {e})")
        print("  The report will have no reward section. Reproduce with:")
        print(f"    python benchmark/score_rewards.py {results_path} --verify")
        return None


def step_report(results_path: str, eval_path: str,
                retrieval_path: Optional[str] = None,
                reward_path: Optional[str] = None) -> str:
    print("\n" + "=" * 60)
    print("STEP 4: Generating comparative report (Aim 5)")
    print("=" * 60)
    from generate_report import generate_report
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = str(ROOT / "report" / f"COMPARATIVE_REPORT_{ts}.md")
    generate_report(results_path, eval_path, out_path, retrieval_path=retrieval_path,
                    reward_path=reward_path)
    return out_path


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    ensure_packages()

    from llm_provider import (print_provider_status, configure_langsmith,
                              verify_credentials, _try_ollama)

    parser = argparse.ArgumentParser(description="Children's Lunch RAG research pipeline",
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mock", action="store_true", help="Use simulated LLM (no API key needed)")
    parser.add_argument("--provider", choices=["groq", "ollama"], default=None)
    parser.add_argument("--model", default=None, help="Override generator model name")
    parser.add_argument("--results", default=None, help="Path to existing results JSON")
    parser.add_argument("--resume", default=None, metavar="RESULTS.JSON",
                        help="Finish a benchmark that stopped early (spent daily "
                             "token budget). Cases already recorded are not re-run.")
    parser.add_argument("--allow-provider-change", action="store_true",
                        help="Permit --resume to finish a run on a different generator "
                             "than it started on — e.g. completing locally on Ollama a "
                             "run that spent its cloud token budget. Refused by default. "
                             "The resulting file holds two generators; they are counted "
                             "in metadata.generators and disclosed in the report.")
    parser.add_argument("--skip-setup", action="store_true")
    parser.add_argument("--repeats", type=int, default=1,
                        help="Runs per case per pipeline. >1 reports mean +/- SD "
                             "alongside the paired significance test. Try 5.")
    parser.add_argument("--judge-provider", choices=["groq", "ollama"], default=None,
                        help="Provider for the LLM-as-judge. Defaults to --provider. Use this "
                             "to generate locally but judge in the cloud, which is usually what "
                             "you want on a modest machine: an 8B local judge took over 40 "
                             "minutes for 2 cases on 7.7 GB of RAM, where Groq takes seconds "
                             "and the judge is a different model family anyway.")
    parser.add_argument("--limit", type=int, default=None, metavar="N",
                        help="Run only the first N benchmark cases. For smoke-testing a "
                             "provider — a local Ollama model can take an hour where Groq "
                             "takes minutes. The run is marked truncated in its metadata.")
    parser.add_argument("--no-judge", action="store_true",
                        help="Skip LLM-as-judge scoring. Safety metrics, significance "
                             "and retrieval metrics are all deterministic and still "
                             "computed. Pair with --repeats: repeats exist to put "
                             "uncertainty on the safety rates, and judging every "
                             "repeat multiplies judge spend by --repeats for no gain "
                             "on that question.")
    # No --groq-key / --langsmith-key: a key passed on the command line is
    # recorded in shell history and is visible to any other user via the process
    # table for the lifetime of the run. Keys belong in .env, which is gitignored.
    args = parser.parse_args()

    print_provider_status()
    print()

    if not args.mock and not args.results:
        groq_key = os.environ.get("GROQ_API_KEY", "").strip()
        ollama_ok = _try_ollama("dummy") is not None
        if not groq_key and not ollama_ok:
            print("ERROR: No LLM provider configured and --mock not specified.")
            print("  Options:")
            print("    1. Add GROQ_API_KEY=gsk_... to .env  (free at console.groq.com)")
            print("    2. Start Ollama: ollama serve  (install from ollama.com)")
            print("    3. Run mock:    python3 run_all.py --mock")
            sys.exit(1)

    # Preflight. Costs one token-free round trip and saves the whole run when the
    # key is dead — see llm_provider.verify_credentials. Skipped when nothing
    # downstream would use Groq: --mock, reusing results with the judge off, or an
    # explicit --provider ollama, which authenticates with nothing.
    needs_groq = not args.mock and args.provider != "ollama"
    if needs_groq and not (args.results and args.no_judge):
        creds_ok, creds_detail = verify_credentials()
        if not creds_ok:
            print("ERROR: the provider rejected GROQ_API_KEY.")
            print(f"  {creds_detail[:200]}")
            print()
            print("  Nothing in this run can succeed until the key is replaced —")
            print("  both the generator and the judge authenticate with it:")
            print("    1. Get a fresh key at https://console.groq.com/keys")
            print("    2. Replace GROQ_API_KEY=... in .env")
            print("    3. Re-run this command")
            print()
            print("  This is NOT a rate limit: no waiting period applies, and")
            print("  --resume will not help.")
            if _try_ollama("dummy") is not None:
                # Reachable right now, so it is a real alternative rather than a
                # suggestion to go and install something.
                print()
                print("  Ollama IS running locally — you can run against it instead:")
                print("    python run_all.py --provider ollama")
            print("    python run_all.py --mock        (no provider at all)")
            sys.exit(1)
        print(f"Credential check: {creds_detail}")
        print()
    elif args.provider == "ollama" and os.environ.get("GROQ_API_KEY", "").strip():
        # The generator will honour --provider, but get_judge_llm() resolves
        # independently and still prefers Groq whenever a key is present. Say so
        # here rather than letting 30 judge calls discover it one at a time.
        creds_ok, creds_detail = verify_credentials()
        if not creds_ok:
            print("WARNING: generating with Ollama, but the judge still resolves to")
            print("  Groq while GROQ_API_KEY is set — and that key was rejected:")
            print(f"    {creds_detail[:160]}")
            print("  Judge scores will be empty. Either replace the key, or run with")
            print("  --no-judge, or unset GROQ_API_KEY to put the judge on Ollama too.")
            print()

    _judge_prov = args.judge_provider or args.provider
    if (args.provider == "ollama" and _judge_prov == "ollama" and not args.no_judge) or (
            not args.mock and not os.environ.get("GROQ_API_KEY", "").strip()):
        from llm_provider import ollama_memory_warning
        _mem = ollama_memory_warning()
        if _mem:
            print(_mem)
            print()

    tracing = configure_langsmith()
    if tracing:
        print(f"LangSmith tracing → {os.environ.get('LANGCHAIN_PROJECT','lunch-rag-benchmark')}")

    t_start = time.perf_counter()

    if not args.results and not args.skip_setup:
        step_setup()

    if args.results:
        results_path = args.results
        print(f"\nUsing existing results: {results_path}")
    else:
        results_path = step_benchmark(provider=args.provider, use_mock=args.mock,
                                      resume_from=args.resume, limit=args.limit,
                                      model_override=args.model, repeats=args.repeats,
                                      allow_provider_change=args.allow_provider_change)

    # Free the generator's weights before the judge loads its own. On a machine
    # that cannot hold both, this is the difference between a run that finishes
    # and one that dies in Ollama's allocator with no diagnosable message.
    judge_provider = args.judge_provider or args.provider
    # Only worth evicting when the judge needs the RAM back, i.e. it is also local.
    if args.provider == "ollama" and judge_provider == "ollama" and not args.no_judge:
        from llm_provider import unload_ollama_model
        _gen_model = os.environ.get("OLLAMA_MODEL", "llama3.2").strip()
        if unload_ollama_model(_gen_model):
            print(f"Released Ollama generator model ({_gen_model}) before judging.")

    eval_path = step_evaluate(results_path, run_judge=False if args.no_judge else None,
                              provider=judge_provider)
    retrieval_path = step_retrieval_eval()
    reward_path = step_rewards(results_path)
    report_path = step_report(results_path, eval_path, retrieval_path, reward_path)

    elapsed = time.perf_counter() - t_start
    print(f"\n{'=' * 60}")
    print(f"COMPLETE in {elapsed:.1f}s")
    print(f"  Results : {results_path}")
    print(f"  Scores  : {eval_path}")
    if reward_path:
        print(f"  Rewards : {reward_path}")
    print(f"  Report  : {report_path}")
    if tracing:
        print("  Traces  : https://smith.langchain.com")
    print("=" * 60)


if __name__ == "__main__":
    main()
