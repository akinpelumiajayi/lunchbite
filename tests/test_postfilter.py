"""
Unit tests for the symbolic post-filter (src/graphs/nodes.py::symbolic_postfilter).

This is the gate the neurosymbolic arm relies on, and it is the only place the
LLM's *claims* about its own output are checked against the recipe record. Three
distinct things can go wrong, and they need separating:

  hallucinated id     — the recipe does not exist. Nothing downstream is meaningful.
  false allergen claim— the model asserts an allergen is absent when the record
                        says it is present. This is what a reader would rely on,
                        so it is a rejection, not a warning.
  citation mismatch   — the model returns a source that is not the one attached to
                        the recipe. The recipe is still safe, so this is repaired
                        rather than rejected, but it is counted.

Run:  pytest tests/ -v
"""

from __future__ import annotations

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Both are needed: nodes.py imports `state` as a top-level module, which only
# resolves with src/graphs on the path (see §3.5 — the package layout is the
# real fix; this mirrors what the runtime already does).
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "src", "graphs"))

from graphs.nodes import symbolic_postfilter  # noqa: E402


CITATION = "NHS Eatwell Guide (2024)"


@pytest.fixture
def patched_recipes(monkeypatch):
    """Pin the recipe lookup so tests do not depend on data/recipes.json."""
    recipes = {
        "recipe_001": {
            "id": "recipe_001",
            "name": "Tomato pasta",
            "ingredients": ["pasta", "tomato", "olive oil"],
            "allergens_present": ["cereals containing gluten"],
            "nutrition_per_serving": {"energy_kcal": 400, "sugars_g": 4.0, "salt_g": 0.5},
            "citation": CITATION,
        },
    }
    monkeypatch.setattr("graphs.nodes._recipes_by_id", lambda: recipes)
    return recipes


def make_state(menus, allergies=None):
    return {
        "profile": {
            "age_years": 7,
            "allergies": allergies or [],
            "intolerances": [],
            "dislikes": [],
            "likes": [],
            "cultural_context": "",
        },
        "proposed_menus": menus,
        "latency_ms": {},
    }


def menu(**overrides):
    base = {
        "recipe_id": "recipe_001",
        "menu_name": "Tomato pasta",
        "why_it_fits": "simple and safe",
        "nutritional_rationale": "balanced",
        "allergens_confirmed_absent": ["milk", "egg"],
        "source_citation": CITATION,
    }
    base.update(overrides)
    return base


def test_clean_menu_survives(patched_recipes):
    out = symbolic_postfilter(make_state([menu()]))
    assert len(out["final_menus"]) == 1
    entry = out["symbolic_post_filter_log"][0]
    assert entry["survived"] is True
    assert entry.get("citation_corrected") is None


def test_hallucinated_id_rejected(patched_recipes):
    out = symbolic_postfilter(make_state([menu(recipe_id="recipe_999")]))
    assert out["final_menus"] == []
    assert "hallucinated" in out["symbolic_post_filter_log"][0]["rejection_reason"]


def test_false_allergen_claim_rejected(patched_recipes):
    """
    The recipe genuinely contains gluten. A child with no gluten allergy means the
    guardrail passes — so this rejection can only come from the claim check, which
    is exactly the gap being covered.
    """
    out = symbolic_postfilter(
        make_state([menu(allergens_confirmed_absent=["cereals containing gluten"])])
    )
    assert out["final_menus"] == []
    entry = out["symbolic_post_filter_log"][0]
    assert entry["false_allergen_claim"] == ["cereals containing gluten"]
    assert "claimed absent" in entry["rejection_reason"]


def test_false_allergen_claim_is_case_insensitive(patched_recipes):
    out = symbolic_postfilter(
        make_state([menu(allergens_confirmed_absent=["Cereals Containing Gluten"])])
    )
    assert out["final_menus"] == []


def test_fabricated_citation_is_repaired_not_rejected(patched_recipes):
    out = symbolic_postfilter(
        make_state([menu(source_citation="Journal of Invented Sources (2019)")])
    )
    # Survives — the recipe is safe; shipping the false source is the harm.
    assert len(out["final_menus"]) == 1
    assert out["final_menus"][0]["source_citation"] == CITATION
    entry = out["symbolic_post_filter_log"][0]
    assert entry["citation_corrected"] is True
    assert entry["citation_returned"] == "Journal of Invented Sources (2019)"


def test_missing_citation_is_filled_in(patched_recipes):
    out = symbolic_postfilter(make_state([menu(source_citation="")]))
    assert out["final_menus"][0]["source_citation"] == CITATION
    assert out["symbolic_post_filter_log"][0]["citation_corrected"] is True


def test_guardrail_rejection_still_takes_precedence(patched_recipes):
    """An unsafe-for-this-child recipe is rejected on safety, not on the claim."""
    out = symbolic_postfilter(
        make_state([menu()], allergies=["gluten"])
    )
    assert out["final_menus"] == []
    entry = out["symbolic_post_filter_log"][0]
    assert entry.get("false_allergen_claim") is None
    assert "claimed absent" not in (entry["rejection_reason"] or "")


def test_original_menu_dict_is_not_mutated(patched_recipes):
    """Citation repair must not write back into the caller's proposed_menus."""
    m = menu(source_citation="wrong")
    symbolic_postfilter(make_state([m]))
    assert m["source_citation"] == "wrong"
