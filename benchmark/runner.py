"""
runner.py -- Runs all four pipelines against every benchmark case.

Pipelines run per case:
  no_llm        -- rule-based only (primary baseline)
  neural_rag    -- retrieval + LLM with prompt-only safety
  neurosymbolic -- retrieval + symbolic gates + LLM
  no_rag        -- LLM with no retrieval (secondary reference)
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

enable_utf8_stdout()


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
    from llm_provider import use_cross_encoder_reranker
    from guardrails import lunch_fraction

    try:
        git_sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                 capture_output=True, text=True, cwd=str(ROOT),
                                 timeout=5).stdout.strip() or None
    except Exception:
        git_sha = None

    return {
        "model": provider_name,
        "judge_model": os.environ.get("GROQ_JUDGE_MODEL", "llama-3.3-70b-versatile"),
        "n_cases": n_cases,
        "timestamp": ts,
        "synthetic": synthetic,
        "git_sha": git_sha,
        "pipelines": ["no_llm", "neural_rag", "neurosymbolic", "no_rag"],
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
            "max_tokens": os.environ.get("LLM_MAX_TOKENS", "2000"),
        },
        "guardrail": {"lunch_nutrition_fraction": lunch_fraction()},
        "adversarial_injection_applied_to": ["neural_rag", "neurosymbolic", "no_rag"],
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
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "case_id": case.case_id,
        "description": case.description,
        "category": case.category,
        "profile": case.profile,
        "adversarial_injection": case.adversarial_injection,
        "expected_unsafe_ids": case.expected_unsafe_ids,
        "expected_safe_ids": case.expected_safe_ids,
        "no_llm": None,
        "neural_rag": None,
        "neurosymbolic": None,
        "no_rag": None,
    }

    # The injection goes to EVERY arm whose LLM can see the profile. Injecting
    # only into neural_rag assumed the conclusion: neurosymbolic then scored a 0%
    # bypass rate without ever having been attacked. The symbolic gates have to
    # earn that number under the same attack.
    # no_llm is exempt because it never builds a prompt — there is no surface.
    pipelines = [
        ("no_llm", no_llm_graph, False),
        ("neural_rag", neural_graph, True),
        ("neurosymbolic", neuro_graph, True),
        ("no_rag", no_rag_graph, True),
    ]

    for mode, graph, apply_injection in pipelines:
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


def run_benchmark(
    provider: Optional[str] = None,
    model_override: Optional[str] = None,
    output_dir: str = "benchmark/results",
) -> str:
    from llm_provider import get_llm, configure_langsmith
    from graphs.build_graphs import (build_no_llm_graph, build_neural_rag_graph,
                                     build_neurosymbolic_graph, build_no_rag_graph)

    configure_langsmith()

    if model_override:
        env_key = "OLLAMA_MODEL" if provider == "ollama" else "GROQ_MODEL"
        os.environ[env_key] = model_override

    llm, provider_name = get_llm(prefer=provider)
    print(f"Generator LLM: {provider_name}")

    no_llm_graph   = build_no_llm_graph()
    neural_graph   = build_neural_rag_graph(llm)
    neuro_graph    = build_neurosymbolic_graph(llm)
    no_rag_graph   = build_no_rag_graph(llm)
    print(f"All 4 graphs compiled. Running {len(BENCHMARK_CASES)} cases...\n")

    results: List[Dict[str, Any]] = []
    for case in BENCHMARK_CASES:
        print(f"  [{case.case_id}] {case.description}")
        results.append(run_single_case(case, no_llm_graph, neural_graph, neuro_graph, no_rag_graph))

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(output_dir, f"run_{ts}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"metadata": build_run_metadata(provider_name, len(results), ts),
                   "results": results}, f, indent=2)
    print(f"\nResults saved to: {out_path}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=["groq", "ollama"], default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--output-dir", default="benchmark/results")
    parser.add_argument("--groq-key", default=None)
    args = parser.parse_args()
    if args.groq_key:
        os.environ["GROQ_API_KEY"] = args.groq_key
    run_benchmark(provider=args.provider, model_override=args.model, output_dir=args.output_dir)
