"""
retrieval.py

Implements the "Retrieval layer" from the project's data flow:
  2. System builds a query from the child profile.
  3. Retrieval: semantic search over recipes and guidelines, filtered by
     allergy and school-safe rules.

This module sits between the raw vector_store (generic semantic search) and
the generation layer (LLM). It is responsible for:
  - turning a ChildProfile into a natural-language query string
  - running hybrid search (semantic ranking + exact metadata filtering)
  - pulling supporting guideline/allergen-rule chunks for the LLM's
    nutritional rationale
  - applying the hard guardrail filter to recipe candidates BEFORE they ever
    reach the LLM, so the LLM is never even shown an unsafe recipe to choose
    from in the first place (defense in depth: even if the LLM ignored
    instructions, it physically cannot suggest a recipe it was never given)
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from vector_store import semantic_search
from guardrails import ChildProfile, filter_recipes
from document_loader import _load_json


def build_query_string(profile: ChildProfile) -> str:
    """
    Turns a structured ChildProfile into the kind of natural-language query
    the spec describes, e.g.:
    "lactose-intolerant, age 7, likes rice, nut-free school"
    """
    parts = [f"age {profile.age_years}"]

    if profile.allergies:
        parts.append(f"allergic to {', '.join(profile.allergies)}")
    if profile.intolerances:
        parts.append(f"{', '.join(profile.intolerances)} intolerant")
    if profile.school_nut_free:
        parts.append("nut-free school")
    if profile.likes:
        parts.append(f"likes {', '.join(profile.likes)}")
    if profile.dislikes:
        parts.append(f"dislikes {', '.join(profile.dislikes)}")
    if profile.cultural_context:
        parts.append(profile.cultural_context)

    return ", ".join(parts)


def retrieve_candidate_recipes(profile: ChildProfile, query: str, n_results: int = 9) -> dict:
    """
    Step 1 of retrieval: semantic search over ALL recipes (no metadata
    pre-filtering on allergens here, because we want the semantic ranking to
    consider the full recipe set on relevance grounds first -- the actual
    safety filtering is then applied deterministically in guardrails.py,
    which is the authoritative gate, not a vector similarity score).
    """
    raw_recipes = {r["id"]: r for r in _load_json("recipes.json")}

    hits = semantic_search(query, n_results=n_results, source_types=["recipe"])

    ranked_recipes = []
    for h in hits:
        recipe_id = h["id"]
        if recipe_id in raw_recipes:
            ranked_recipes.append(raw_recipes[recipe_id])

    # Apply the deterministic, non-negotiable safety + nutrition gate.
    safe_recipes, rejected_recipes = filter_recipes(ranked_recipes, profile)

    return {
        "query_used": query,
        "candidates_considered": len(ranked_recipes),
        "safe_candidates": safe_recipes,
        "rejected_candidates": rejected_recipes,
    }


def retrieve_supporting_context(profile: ChildProfile, query: str, n_results: int = 4) -> list[dict]:
    """
    Step 2 of retrieval: pull relevant nutrition-guideline and allergen-rule
    chunks to ground the LLM's nutritional rationale and allergen
    explanations in the actual source documents, rather than letting it
    invent generic-sounding advice.
    """
    guideline_hits = semantic_search(query, n_results=n_results, source_types=["nutrition_guideline"])
    allergen_hits = []
    if profile.all_restricted_allergens():
        # If the child has restrictions, also pull the specific allergen-rule
        # chunks for those allergens so the LLM can cite WHY something is
        # excluded, grounded in the actual EU FIC text.
        for allergen in profile.all_restricted_allergens():
            allergen_hits.extend(semantic_search(allergen, n_results=2, source_types=["allergen_rule"]))

    # Dedup by id while preserving order
    seen = set()
    combined = []
    for h in guideline_hits + allergen_hits:
        if h["id"] not in seen:
            seen.add(h["id"])
            combined.append(h)
    return combined


def run_retrieval(profile: ChildProfile) -> dict:
    """Full retrieval pipeline entry point: profile -> query -> safe candidates + context."""
    query = build_query_string(profile)
    recipe_results = retrieve_candidate_recipes(profile, query)
    context_chunks = retrieve_supporting_context(profile, query)

    return {
        "query": query,
        "safe_candidates": recipe_results["safe_candidates"],
        "rejected_candidates": recipe_results["rejected_candidates"],
        "candidates_considered": recipe_results["candidates_considered"],
        "supporting_context": context_chunks,
    }


if __name__ == "__main__":
    profile = ChildProfile(
        age_years=7,
        intolerances=["lactose-intolerant"],
        likes=["rice", "chicken"],
        school_nut_free=True,
    )
    result = run_retrieval(profile)
    print(f"Query: {result['query']}")
    print(f"Candidates considered: {result['candidates_considered']}")
    print(f"Safe candidates: {[r['name'] for r in result['safe_candidates']]}")
    print(f"Rejected: {[(r['recipe_name'], r['reasons']) for r in result['rejected_candidates']]}")
    print(f"Supporting context chunks: {[c['id'] for c in result['supporting_context']]}")
