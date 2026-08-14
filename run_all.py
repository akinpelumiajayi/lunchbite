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
                   repeats: int = 1) -> str:
    print("\n" + "=" * 60)
    print("STEP 2: Running benchmark — all 4 pipelines, 30 cases (Aims 3 & 4)")
    print("=" * 60)
    if use_mock:
        return _run_mock_benchmark(repeats=repeats)
    from runner import run_benchmark
    return run_benchmark(provider=provider, model_override=model_override,
                         output_dir=str(ROOT / "benchmark" / "results"),
                         repeats=repeats)


def _run_mock_benchmark(repeats: int = 1) -> str:
    import uuid
    from unittest.mock import MagicMock
    from langchain_core.messages import AIMessage
    from benchmark_cases import BENCHMARK_CASES
    from runner import run_single_case
    from graphs.build_graphs import (build_no_llm_graph, build_neural_rag_graph,
                                     build_neurosymbolic_graph, build_no_rag_graph)

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

    # Build all 4 graphs
    no_llm_graph   = build_no_llm_graph()
    neural_graph   = build_neural_rag_graph(mock_llm)
    neuro_graph    = build_neurosymbolic_graph(mock_llm)
    no_rag_graph   = build_no_rag_graph(mock_llm)

    print(f"[MOCK MODE] Simulated LLM follows adversarial injections "
          f"({len(KNOWN_INJECTIONS)} injected cases). Results are SYNTHETIC, not evidence.")
    print(f"Running {len(BENCHMARK_CASES)} cases across all 4 pipelines...\n")

    results = []
    for rep in range(repeats):
        if repeats > 1:
            print(f"\n--- repeat {rep + 1}/{repeats} ---")
        for case in BENCHMARK_CASES:
            print(f"  [{case.case_id}] {case.description}")
            r = run_single_case(case, no_llm_graph, neural_graph, neuro_graph, no_rag_graph)
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


def step_evaluate(results_path: str) -> str:
    print("\n" + "=" * 60)
    print("STEP 3: Computing evaluation metrics (Aim 5)")
    print("=" * 60)
    from evaluator import evaluate
    scores = evaluate(results_path)
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
        # Non-fatal so a cross-encoder download failure does not cost you the
        # benchmark — but say loudly what the report will be missing, because a
        # quietly-skipped step is how retrieval went unmeasured in the first place.
        print(f"  WARNING: retrieval evaluation FAILED ({type(e).__name__}: {e})")
        print("  The report will have no retrieval-quality section. Reproduce with:")
        print("    python eval/eval_compare_retrievers.py")
        return None


def step_report(results_path: str, eval_path: str,
                retrieval_path: Optional[str] = None) -> str:
    print("\n" + "=" * 60)
    print("STEP 4: Generating comparative report (Aim 5)")
    print("=" * 60)
    from generate_report import generate_report
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = str(ROOT / "report" / f"COMPARATIVE_REPORT_{ts}.md")
    generate_report(results_path, eval_path, out_path, retrieval_path=retrieval_path)
    return out_path


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    ensure_packages()

    from llm_provider import print_provider_status, configure_langsmith, _try_ollama

    parser = argparse.ArgumentParser(description="Children's Lunch RAG research pipeline",
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mock", action="store_true", help="Use simulated LLM (no API key needed)")
    parser.add_argument("--provider", choices=["groq", "ollama"], default=None)
    parser.add_argument("--model", default=None, help="Override generator model name")
    parser.add_argument("--results", default=None, help="Path to existing results JSON")
    parser.add_argument("--skip-setup", action="store_true")
    parser.add_argument("--repeats", type=int, default=1,
                        help="Runs per case per pipeline. >1 reports mean +/- SD "
                             "alongside the paired significance test. Try 5.")
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
                                      model_override=args.model, repeats=args.repeats)

    eval_path = step_evaluate(results_path)
    retrieval_path = step_retrieval_eval()
    report_path = step_report(results_path, eval_path, retrieval_path)

    elapsed = time.perf_counter() - t_start
    print(f"\n{'=' * 60}")
    print(f"COMPLETE in {elapsed:.1f}s")
    print(f"  Results : {results_path}")
    print(f"  Scores  : {eval_path}")
    print(f"  Report  : {report_path}")
    if tracing:
        print("  Traces  : https://smith.langchain.com")
    print("=" * 60)


if __name__ == "__main__":
    main()
