"""
main.py -- Interactive pipeline orchestration, driving the neurosymbolic LangGraph.

This module used to call a second, hand-rolled implementation of the pipeline
(`retrieval.py` + `generation.py` + `post_filter.py`) while the benchmark ran the
LangGraph in `graphs/`. The two had drifted apart: different prompts, different
JSON schemas (only the graph asked for `source_citation`), a different generation
pool, and no citation verification on the legacy side. **The CLI and the
benchmark therefore measured different systems**, which meant no result reported
in the dissertation described what a user of `cli.py` actually got.

There is now one implementation. `recommend_lunches` invokes the same compiled
`build_neurosymbolic_graph` the benchmark's `neurosymbolic` arm invokes, and maps
its terminal state onto the display shape `print_final_recommendations` expects.

LLM provider is resolved from .env automatically via llm_provider.
"""

from __future__ import annotations
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "graphs"))

from guardrails import ChildProfile


@lru_cache(maxsize=1)
def _graph():
    """Compiled once per process — building it pulls in the embedding model."""
    from graphs.build_graphs import build_neurosymbolic_graph
    from llm_provider import get_llm
    llm, _ = get_llm()
    return build_neurosymbolic_graph(llm)


def _initial_state(profile: ChildProfile) -> Dict[str, Any]:
    """The graph's state is a plain dict keyed exactly as PipelineState declares."""
    return {
        "profile": profile.__dict__ if hasattr(profile, "__dict__") else dict(profile),
        "pipeline_mode": "neurosymbolic",
        "run_id": f"cli-{uuid4().hex[:8]}",
        "query": "", "bm25_candidates": [], "semantic_candidates": [],
        "fused_candidates": [], "reranked_candidates": [],
        "symbolic_pre_filter_log": [],
        "generation_candidates": [], "llm_raw_output": "",
        "proposed_menus": [], "generation_error": None,
        "symbolic_post_filter_log": [], "final_menus": [],
        "error": None, "latency_ms": {},
    }


def run_pipeline(profile: ChildProfile) -> Dict[str, Any]:
    """
    Runs the neurosymbolic graph and returns its raw terminal state.

    The single entry point for anything needing the pipeline's internals — the
    retrieval candidates, the two filter logs, the un-filtered proposals. Callers
    that only want something printable should use `recommend_lunches`.
    """
    return _graph().invoke(_initial_state(profile))


def supporting_context(query: str, n_results: int = 4) -> List[Dict[str, Any]]:
    """
    Nutrition-guideline and allergen-rule chunks for the query.

    Not part of the recommendation path — the graph grounds generation in the
    recipe records themselves. This exists for the evaluation harness, which
    checks generated claims against the source documents.
    """
    from graphs.nodes import _vector_semantic_search
    chunks = _vector_semantic_search(query, n_results=n_results,
                                     source_types=["nutrition_guideline"])
    chunks += _vector_semantic_search(query, n_results=2,
                                      source_types=["allergen_rule"])
    return chunks


def shape_result(final: Dict[str, Any], profile: ChildProfile) -> Dict[str, Any]:
    """
    Maps a terminal PipelineState onto the display contract, without printing.

    Pure, and the single definition of that contract. `recommend_lunches` and the
    Streamlit app both end here: the one thing this module's history argues
    against is a second implementation of the same mapping drifting away from
    the first, and a dashboard that reshaped the state itself would be exactly
    that.

    Works for every arm, not just neurosymbolic -- the arms without symbolic
    gates simply leave both filter logs empty, so their rejection lists come
    back empty rather than absent.
    """
    pre_log: List[Dict[str, Any]] = final.get("symbolic_pre_filter_log") or []
    post_log: List[Dict[str, Any]] = final.get("symbolic_post_filter_log") or []
    approved = final.get("generation_candidates") or []

    explanation: Optional[str] = None
    if not approved:
        explanation = (
            "No recipes in the knowledge base satisfy this child's combined constraints. "
            "Consider expanding the recipe knowledge base for this combination of restrictions."
        )

    return {
        "profile": profile,
        "query": final.get("query", ""),
        "final_recommendations": final.get("final_menus") or [],
        "rejected_at_retrieval": [e for e in pre_log if not e.get("passed")],
        "rejected_at_post_filter": [e for e in post_log if not e.get("survived")],
        "generation_error": final.get("generation_error"),
        "explanation": explanation,
        # Kept for the display contract; the graph rejects rather than defers, so
        # nothing lands here. Left explicit rather than removed so the CLI's
        # output shape stays stable.
        "unverifiable": [],
    }


def recommend_lunches(profile: ChildProfile, verbose: bool = True) -> Dict[str, Any]:
    """Retrieval -> symbolic pre-filter -> LLM -> symbolic post-filter."""
    final = run_pipeline(profile)

    pre_log: List[Dict[str, Any]] = final.get("symbolic_pre_filter_log") or []
    post_log: List[Dict[str, Any]] = final.get("symbolic_post_filter_log") or []
    rejected_pre = [e for e in pre_log if not e.get("passed")]
    rejected_post = [e for e in post_log if not e.get("survived")]
    approved = final.get("generation_candidates") or []

    if verbose:
        print(f"\n[Retrieval] Query: \"{final.get('query', '')}\"")
        print(f"[Retrieval] {len(pre_log)} candidates considered.")
        print(f"[Retrieval] {len(approved)} passed the safety/nutrition guardrail.")
        for e in rejected_pre:
            print(f"    - {e.get('recipe_name')}: {e.get('reasons')}")
        # Guardrail warnings name a restriction the system could not interpret
        # (an unrecognised allergy term). Silence here would mean a restriction
        # the user typed was never enforced and they were never told.
        for e in pre_log:
            for w in (e.get("warnings") or []):
                print(f"    ! {w}")

    if final.get("generation_error") and verbose:
        print(f"\n[Generation] {final['generation_error']}")
    elif verbose:
        print(f"\n[Generation] {len(final.get('proposed_menus') or [])} menu(s) proposed.")

    if verbose:
        print(f"[Post-filter] {len(final.get('final_menus') or [])} menu(s) passed final verification.")
        if rejected_post:
            print(f"[Post-filter] WARNING: {len(rejected_post)} menu(s) rejected at final gate:")
            for e in rejected_post:
                print(f"    - {e.get('menu_name') or e.get('recipe_id')}: "
                      f"{e.get('rejection_reason')}")
        corrected = [e for e in post_log if e.get("citation_corrected")]
        if corrected:
            print(f"[Post-filter] {len(corrected)} fabricated citation(s) corrected "
                  f"against the recipe record.")

    return shape_result(final, profile)


def print_final_recommendations(result: Dict[str, Any]) -> None:
    print("\n" + "=" * 70)
    print("FINAL LUNCH RECOMMENDATIONS")
    print("=" * 70)

    if not result["final_recommendations"]:
        print("\nNo recommendations could be produced.")
        if result.get("explanation"):
            print(f"Reason: {result['explanation']}")
        if result.get("generation_error"):
            print(f"Generation error: {result['generation_error']}")
        return

    for i, rec in enumerate(result["final_recommendations"], 1):
        print(f"\n--- Option {i}: {rec.get('menu_name', '?')} ---")
        print(f"Why it fits: {rec.get('why_it_fits', '')}")
        print(f"Nutritional rationale: {rec.get('nutritional_rationale', '')}")
        confirmed = ', '.join(rec.get('allergens_confirmed_absent') or []) or 'none specified'
        print(f"Allergens confirmed absent: {confirmed}")
        # Verified by the post-filter against the recipe record, so it is safe to
        # display as provenance rather than as the model's unchecked assertion.
        if rec.get("source_citation"):
            print(f"Source: {rec['source_citation']}")


if __name__ == "__main__":
    from llm_provider import print_provider_status
    print_provider_status()
    print()

    example_profile = ChildProfile(
        age_years=7,
        intolerances=["lactose-intolerant"],
        likes=["rice", "fish"],
        school_nut_free=True,
        cultural_context="British primary school",
    )
    result = recommend_lunches(example_profile)
    print_final_recommendations(result)
