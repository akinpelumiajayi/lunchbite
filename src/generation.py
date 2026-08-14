"""
generation.py -- Original (non-LangGraph) generation backend.

Used by src/main.py, src/cli.py, and eval/eval_generation.py.
Now resolves LLM from llm_provider.py (reads .env automatically),
supporting both Groq and Ollama.

The LangGraph pipeline (src/graphs/) has its own generation node in
nodes.py and does not use this file.
"""

from __future__ import annotations
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from guardrails import ChildProfile

MODEL_FALLBACK = "llama-3.3-70b-versatile"


def _build_prompt(
    profile: ChildProfile,
    safe_candidates: List[Dict[str, Any]],
    supporting_context: List[Dict[str, Any]],
) -> str:
    context_text = "\n\n".join(
        f"[{c['metadata']['source_type']}] {c['text']}" for c in supporting_context
    )
    recipe_lines = []
    for r in safe_candidates:
        n = r.get("nutrition_per_serving", {})
        recipe_lines.append(
            f"- {r['name']} (id: {r['id']})\n"
            f"  Category: {r['meal_category']}\n"
            f"  Ingredients: {', '.join(r.get('ingredients', []))}\n"
            f"  Nutrition: {n.get('energy_kcal', '?')} kcal, "
            f"{n.get('sugars_g', '?')}g sugars, {n.get('salt_g', '?')}g salt, "
            f"{n.get('protein_g', '?')}g protein\n"
            f"  Allergens present: {', '.join(r.get('allergens_present', [])) or 'none declared'}\n"
            f"  Diet tags: {', '.join(r.get('diet_tags', []))}"
        )
    recipes_block = "\n\n".join(recipe_lines)
    restricted = sorted(profile.all_restricted_allergens())

    return (
        "You are a child nutrition assistant helping plan school lunches. "
        "Work ONLY from the recipes provided — do not invent new dishes.\n\n"
        f"CHILD PROFILE:\n"
        f"- Age: {profile.age_years} years\n"
        f"- Allergies/intolerances to avoid: {', '.join(restricted) if restricted else 'none'}\n"
        f"- School nut-free: {'yes' if profile.school_nut_free else 'no'}\n"
        f"- Likes: {', '.join(profile.likes) if profile.likes else 'not specified'}\n"
        f"- Dislikes: {', '.join(profile.dislikes) if profile.dislikes else 'not specified'}\n"
        f"- Cultural context: {profile.cultural_context or 'not specified'}\n\n"
        "These recipes have been verified safe by a deterministic rule-based system:\n\n"
        f"{recipes_block}\n\n"
        "SUPPORTING REFERENCE MATERIAL:\n"
        f"{context_text}\n\n"
        "Propose 1 to 3 lunch menu options from the safe recipes above. "
        "Explain why each fits this child and give a brief nutritional rationale.\n\n"
        "Respond ONLY with valid JSON (no markdown fences):\n"
        '{"menu_options": [{"recipe_id": "recipe_001", "menu_name": "...", '
        '"why_it_fits": "...", "nutritional_rationale": "...", '
        '"allergens_confirmed_absent": ["..."]}]}\n\n'
        'If no safe recipe is suitable, return {"menu_options": []}.'
    )


def _parse_response(raw: str) -> Dict[str, Any]:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        return {
            "menu_options": [],
            "generation_error": f"Could not parse model output as JSON: {e}",
            "raw_output": raw,
        }


def generate_menus(
    profile: ChildProfile,
    safe_candidates: List[Dict[str, Any]],
    supporting_context: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Generates lunch menu proposals using the configured LLM provider.
    Reads .env automatically via llm_provider.
    Short-circuits (returns empty) if no candidates are provided.
    """
    if not safe_candidates:
        return {"menu_options": []}

    from llm_provider import get_llm
    from langchain_core.messages import HumanMessage

    try:
        llm, provider_name = get_llm()
    except RuntimeError as e:
        return {
            "menu_options": [],
            "generation_error": str(e),
        }

    prompt = _build_prompt(profile, safe_candidates, supporting_context)

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        raw = response.content
    except Exception as e:
        return {
            "menu_options": [],
            "generation_error": f"LLM call failed ({provider_name}): {type(e).__name__}: {e}",
        }

    return _parse_response(raw)


if __name__ == "__main__":
    from retrieval import run_retrieval
    from llm_provider import print_provider_status

    print_provider_status()
    print()

    profile = ChildProfile(
        age_years=8,
        allergies=["peanut"],
        intolerances=["lactose"],
        likes=["fish", "wraps"],
        school_nut_free=True,
        cultural_context="British school lunch",
    )
    retrieval_result = run_retrieval(profile)
    print(f"Safe candidates: {[r['name'] for r in retrieval_result['safe_candidates']]}")

    menus = generate_menus(
        profile,
        retrieval_result["safe_candidates"],
        retrieval_result["supporting_context"],
    )
    print(json.dumps(menus, indent=2))
