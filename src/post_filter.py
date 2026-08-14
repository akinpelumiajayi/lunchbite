"""
post_filter.py

Step 5 of the project's data flow: "Post-filter: Reject menus violating
constraints. Return final recommendations."

This module is the LAST line of defense before anything reaches the end
user. Even though:
  1. retrieval.py already filtered candidates to only safe recipes
     (guardrails.py), and
  2. generation.py instructed the LLM to choose only from that safe list,

we do NOT trust that the LLM's `recipe_id` references are necessarily
correct, that it didn't subtly alter ingredients in its prose, or that
nothing was lost in the prompt/response round-trip. This module re-runs the
exact same deterministic guardrail check on whatever recipe_id the LLM
claims to have used, using the ORIGINAL recipe data (not the LLM's
restated version of it), and rejects anything that does not check out.

This means: even if every other layer in this system had a bug or was
compromised, a child with a peanut allergy still cannot be served a peanut
recipe by this system, because the final gate re-derives the answer from
the source-of-truth recipe database every time.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from guardrails import ChildProfile, check_recipe_against_profile
from document_loader import _load_json


def post_filter_menus(profile: ChildProfile, generation_output: dict) -> dict:
    """
    Re-validates each proposed menu option against the ORIGINAL recipe
    record (looked up fresh by recipe_id, not trusting the LLM's restated
    ingredients/nutrition numbers) and the same deterministic guardrail
    logic used in retrieval.

    Returns a dict with:
      - final_recommendations: menus that passed re-verification, enriched
        with the original structured recipe data
      - rejected_at_post_filter: anything the LLM proposed that did NOT
        survive re-verification, with reasons (this should be rare/never if
        upstream layers are working, but the system must not silently drop
        this information -- it's valuable for auditing/debugging the
        pipeline)
      - unverifiable: menu options whose recipe_id didn't match any known
        recipe at all (e.g. the LLM hallucinated an id)
    """
    raw_recipes = {r["id"]: r for r in _load_json("recipes.json")}

    final_recommendations = []
    rejected_at_post_filter = []
    unverifiable = []

    for option in generation_output.get("menu_options", []):
        recipe_id = option.get("recipe_id")
        original_recipe = raw_recipes.get(recipe_id)

        if original_recipe is None:
            unverifiable.append({
                "llm_proposed": option,
                "reason": f"recipe_id '{recipe_id}' does not match any known recipe in the database.",
            })
            continue

        # Re-run the SAME deterministic check used during retrieval, against
        # the ORIGINAL recipe record, not the LLM's restated version of it.
        result = check_recipe_against_profile(original_recipe, profile)

        if not result.passed:
            rejected_at_post_filter.append({
                "recipe_id": recipe_id,
                "recipe_name": original_recipe["name"],
                "llm_proposed": option,
                "reasons": result.reasons_for_rejection,
            })
            continue

        # Cross-check the LLM's self-reported "allergens_confirmed_absent"
        # against the real data, and flag (don't just trust) any mismatch.
        llm_claimed_absent = set(a.lower() for a in option.get("allergens_confirmed_absent", []))
        actual_present = set(a.lower() for a in original_recipe.get("allergens_present", []))
        contradicted_claims = llm_claimed_absent & actual_present
        verification_notes = []
        if contradicted_claims:
            verification_notes.append(
                f"LLM claimed these allergens were absent, but they ARE present per the source recipe data: "
                f"{', '.join(sorted(contradicted_claims))}. Flagging for review even though the recipe itself "
                f"does not violate this child's specific restrictions."
            )

        final_recommendations.append({
            "recipe_id": recipe_id,
            "menu_name": option.get("menu_name", original_recipe["name"]),
            "why_it_fits": option.get("why_it_fits", ""),
            "nutritional_rationale": option.get("nutritional_rationale", ""),
            "allergens_confirmed_absent": option.get("allergens_confirmed_absent", []),
            "verification_notes": verification_notes,
            "guardrail_warnings": result.warnings,
            "full_recipe": original_recipe,
        })

    return {
        "final_recommendations": final_recommendations,
        "rejected_at_post_filter": rejected_at_post_filter,
        "unverifiable": unverifiable,
    }


if __name__ == "__main__":
    # Self-test using a fabricated generation_output to prove the post-filter
    # catches a bad recipe_id even if "generation" claimed it was safe --
    # simulating what would happen if the LLM made a mistake.
    profile = ChildProfile(age_years=8, allergies=["milk"])

    fake_llm_output = {
        "menu_options": [
            {
                "recipe_id": "recipe_006",  # Soft cheese sandwich -- ACTUALLY contains milk!
                "menu_name": "Soft cheese and salad sandwich",
                "why_it_fits": "A nice dairy-based lunch",
                "nutritional_rationale": "Good source of calcium",
                "allergens_confirmed_absent": ["milk"],  # LLM incorrectly claims this is milk-free
            },
            {
                "recipe_id": "recipe_009",  # Tuna sandwich -- genuinely milk-free
                "menu_name": "Tuna mayonnaise and sweetcorn sandwich",
                "why_it_fits": "Milk-free, good protein",
                "nutritional_rationale": "Lower in saturated fat",
                "allergens_confirmed_absent": ["milk"],
            },
            {
                "recipe_id": "recipe_999",  # Doesn't exist
                "menu_name": "Hallucinated recipe",
                "why_it_fits": "n/a",
                "nutritional_rationale": "n/a",
                "allergens_confirmed_absent": [],
            },
        ]
    }

    result = post_filter_menus(profile, fake_llm_output)
    print("FINAL RECOMMENDATIONS:")
    for r in result["final_recommendations"]:
        print(f"  - {r['menu_name']} (notes: {r['verification_notes']})")

    print("\nREJECTED AT POST-FILTER (caught despite LLM/generation saying it was fine):")
    for r in result["rejected_at_post_filter"]:
        print(f"  - {r['recipe_name']}: {r['reasons']}")

    print("\nUNVERIFIABLE (hallucinated recipe_id):")
    for r in result["unverifiable"]:
        print(f"  - {r['reason']}")
