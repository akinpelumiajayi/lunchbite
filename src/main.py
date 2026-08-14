"""
main.py -- Original (non-LangGraph) pipeline orchestration.

Used by cli.py for interactive use. The LangGraph pipeline (run_all.py)
is separate and recommended for research/benchmark runs.

LLM provider is resolved from .env automatically via llm_provider.
"""

from __future__ import annotations
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from guardrails import ChildProfile
from retrieval import run_retrieval
from generation import generate_menus
from post_filter import post_filter_menus


def recommend_lunches(profile: ChildProfile, verbose: bool = True) -> Dict[str, Any]:
    """Full pipeline: retrieval -> guardrail filter -> generation -> post-filter."""

    # Retrieval + symbolic pre-filter
    retrieval_result = run_retrieval(profile)

    if verbose:
        print(f"\n[Retrieval] Query: \"{retrieval_result['query']}\"")
        print(f"[Retrieval] {retrieval_result['candidates_considered']} candidates considered.")
        print(f"[Retrieval] {len(retrieval_result['safe_candidates'])} passed the safety/nutrition guardrail.")
        if retrieval_result["rejected_candidates"]:
            print(f"[Retrieval] {len(retrieval_result['rejected_candidates'])} rejected before LLM:")
            for r in retrieval_result["rejected_candidates"]:
                print(f"    - {r['recipe_name']}: {r['reasons']}")

    if not retrieval_result["safe_candidates"]:
        if verbose:
            print("\n[Pipeline] No safe candidates — skipping generation.")
        return {
            "profile": profile,
            "query": retrieval_result["query"],
            "final_recommendations": [],
            "explanation": (
                "No recipes in the knowledge base satisfy this child's combined constraints. "
                "Consider expanding the recipe knowledge base for this combination of restrictions."
            ),
            "rejected_at_retrieval": retrieval_result["rejected_candidates"],
        }

    # Generation
    generation_output = generate_menus(
        profile,
        retrieval_result["safe_candidates"],
        retrieval_result["supporting_context"],
    )

    if verbose:
        if generation_output.get("generation_error"):
            print(f"\n[Generation] {generation_output['generation_error']}")
        else:
            print(f"\n[Generation] {len(generation_output.get('menu_options', []))} menu(s) proposed.")

    # Post-filter re-verification
    post_filter_result = post_filter_menus(profile, generation_output)

    if verbose:
        n_final = len(post_filter_result["final_recommendations"])
        print(f"[Post-filter] {n_final} menu(s) passed final verification.")
        if post_filter_result["rejected_at_post_filter"]:
            print(f"[Post-filter] WARNING: "
                  f"{len(post_filter_result['rejected_at_post_filter'])} "
                  f"menu(s) rejected at final gate:")
            for r in post_filter_result["rejected_at_post_filter"]:
                print(f"    - {r['recipe_name']}: {r['reasons']}")

    return {
        "profile": profile,
        "query": retrieval_result["query"],
        "final_recommendations": post_filter_result["final_recommendations"],
        "rejected_at_retrieval": retrieval_result["rejected_candidates"],
        "rejected_at_post_filter": post_filter_result["rejected_at_post_filter"],
        "unverifiable": post_filter_result["unverifiable"],
        "generation_error": generation_output.get("generation_error"),
    }


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
        print(f"\n--- Option {i}: {rec['menu_name']} ---")
        print(f"Why it fits: {rec['why_it_fits']}")
        print(f"Nutritional rationale: {rec['nutritional_rationale']}")
        confirmed = ', '.join(rec['allergens_confirmed_absent']) or 'none specified'
        print(f"Allergens confirmed absent: {confirmed}")
        if rec.get("guardrail_warnings"):
            print(f"Warnings: {rec['guardrail_warnings']}")


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
