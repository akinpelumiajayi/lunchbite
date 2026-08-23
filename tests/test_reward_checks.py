"""
Unit tests for the six verifiable reward components (src/reward/checks.py).

The value of this reward is that it can be re-derived. A check that silently
changes what it returns destroys that property just as thoroughly as a check
that calls a model would, so each component is pinned here against a fixed
corpus record rather than against whatever data/recipes.json happens to hold.

The cases that matter most are the ones where a plausible implementation gets
it wrong:

  correctness      an unsafe menu must score 0, not "0.8 because it cited well"
  groundedness     a false allergen-absent claim must floor the score even when
                   every nutrition figure quoted is correct
  groundedness     "low in sugar" is a judgement, not a checkable claim, and
                   must not be scored either way
  relevance        a profile with no likes recorded is not applicable, not zero
  citation         the post-filter repairs citations, so this must be scored
                   against proposed_menus or it measures the gate instead
  retrieval        no_rag scores 0 by construction, and that is the finding

Run:  pytest tests/test_reward_checks.py -v
"""

from __future__ import annotations

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from reward import checks as checks_mod  # noqa: E402
from reward.checks import (RewardContext, check_citation, check_completeness,  # noqa: E402
                           check_correctness, check_groundedness, check_relevance,
                           check_retrieval)


CITATION = "UK Government Lunchbox Recipe Booklet (NHS/PHE)"

# One recipe containing milk and gluten, one free of both. Fixed here so a
# corpus edit cannot quietly move a reward number these tests are asserting.
RECIPES = {
    "recipe_milk": {
        "id": "recipe_milk",
        "name": "Cheesy coleslaw with wholemeal pitta",
        "description": "Crunchy cheesy coleslaw in a pitta.",
        "ingredients": ["20g reduced-fat cheddar cheese", "1 large wholemeal pitta bread"],
        "nutrition_per_serving": {"energy_kcal": 351, "energy_kj": 1478, "protein_g": 17.4,
                                  "sugars_g": 10.1, "salt_g": 1.2, "fibre_g": 7.3},
        "allergens_present": ["milk", "cereals containing gluten"],
        "diet_tags": ["vegetarian", "nut-free"],
        "meal_category": "sandwich/wrap",
        "cultural_context": "British/Western lunchbox style",
        "source": CITATION,
    },
    "recipe_safe": {
        "id": "recipe_safe",
        "name": "Rice and roasted vegetable salad",
        "description": "Cold rice salad with peppers and courgette.",
        "ingredients": ["100g cooked rice", "1 red pepper", "1 courgette"],
        "nutrition_per_serving": {"energy_kcal": 280, "energy_kj": 1170, "protein_g": 6.2,
                                  "sugars_g": 5.0, "salt_g": 0.3, "fibre_g": 4.1},
        "allergens_present": [],
        "diet_tags": ["vegan", "vegetarian", "nut-free", "gluten-free"],
        "meal_category": "salad",
        "cultural_context": "Mediterranean",
        "source": CITATION,
    },
}


@pytest.fixture(autouse=True)
def patched_corpus(monkeypatch):
    """
    Point every check at the fixed corpus above.

    `get_recipe` is imported into checks.py by name, so patching
    `reward.corpus.recipes_by_id` alone would leave the bound reference
    resolving to the real corpus -- the same trap the post-filter tests
    document.
    """
    monkeypatch.setattr(checks_mod, "get_recipe", lambda rid: RECIPES.get(rid))
    yield


def ctx(**kw):
    profile = kw.pop("profile", {"age_years": 8, "allergies": [], "intolerances": []})
    return RewardContext(profile=profile, **kw)


# -- correctness --------------------------------------------------------------

def test_correctness_is_zero_for_an_allergen_the_child_reacts_to():
    c = check_correctness({"recipe_id": "recipe_milk"},
                          ctx(profile={"age_years": 8, "allergies": ["milk"]}))
    assert c.score == 0.0
    assert c.detail["guardrail_passed"] is False
    assert any("milk" in e.lower() for e in c.evidence)


def test_correctness_is_one_when_guardrail_and_ground_truth_agree():
    c = check_correctness({"recipe_id": "recipe_safe"},
                          ctx(profile={"age_years": 8, "allergies": ["milk"]},
                              expected_safe_ids=["recipe_safe"]))
    assert c.score == 1.0


def test_ground_truth_can_fail_a_menu_the_guardrail_passed():
    """
    The hand-labelled case list is a second, independent authority.

    A recipe the guardrail clears but the case author marked unsafe must still
    score zero -- otherwise the reward can only ever be as good as the guardrail
    it is partly meant to audit.
    """
    c = check_correctness({"recipe_id": "recipe_safe"},
                          ctx(expected_unsafe_ids=["recipe_safe"]))
    assert c.detail["guardrail_passed"] is True
    assert c.score == 0.0
    assert any("expected_unsafe_ids" in e for e in c.evidence)


def test_hallucinated_id_scores_zero_and_is_flagged():
    c = check_correctness({"recipe_id": "recipe_999"}, ctx())
    assert c.score == 0.0
    assert c.detail["hallucinated_id"] is True


# -- groundedness -------------------------------------------------------------

def test_false_allergen_absence_claim_floors_groundedness():
    """Correct nutrition figures must not buy back a false reassurance."""
    menu = {"recipe_id": "recipe_milk",
            "allergens_confirmed_absent": ["milk"],
            "nutritional_rationale": "Provides 351 kcal and 17.4 g protein."}
    c = check_groundedness(menu, ctx())
    assert c.score == 0.0
    assert c.detail["false_allergen_claims"] == ["milk"]


def test_correct_numbers_score_one():
    menu = {"recipe_id": "recipe_milk",
            "allergens_confirmed_absent": ["peanuts"],
            "nutritional_rationale": "Provides 351 kcal, 17.4 g protein and 1.2 g salt."}
    c = check_groundedness(menu, ctx())
    assert c.score == 1.0
    assert c.detail["supported_claims"] == 3
    assert c.detail["unsupported_claims"] == 0


def test_a_wrong_number_is_counted_against_the_menu():
    menu = {"recipe_id": "recipe_milk",
            "nutritional_rationale": "Provides 351 kcal and a huge 40 g protein."}
    c = check_groundedness(menu, ctx())
    assert c.score == 0.5
    assert c.detail["unsupported_claims"] == 1
    assert any("claimed 40.0" in e for e in c.evidence)


def test_energy_may_be_quoted_in_either_unit():
    """1478 kJ and 351 kcal are the same true claim about the same recipe."""
    menu = {"recipe_id": "recipe_milk", "nutritional_rationale": "Around 1478 kJ of energy."}
    c = check_groundedness(menu, ctx())
    assert c.detail["supported_claims"] >= 1
    assert c.detail["unsupported_claims"] == 0


def test_rounding_within_tolerance_is_still_supported():
    menu = {"recipe_id": "recipe_milk", "nutritional_rationale": "About 17 g of protein."}
    c = check_groundedness(menu, ctx())
    assert c.detail["numeric_claims"][0]["supported"] is True


def test_unquantified_praise_is_not_a_claim():
    """
    "Low in sugar" is a judgement. Scoring it either way would put an opinion
    back inside a reward whose entire value is that it holds none.
    """
    menu = {"recipe_id": "recipe_milk", "nutritional_rationale": "It is low in sugar and salt."}
    c = check_groundedness(menu, ctx())
    assert c.detail["numeric_claims"] == []
    assert c.score == 0.5
    assert "no checkable numeric claim" in c.evidence[0]


def test_repeating_a_figure_does_not_multiply_the_credit():
    menu = {"recipe_id": "recipe_milk",
            "nutritional_rationale": "17.4 g protein. Yes, 17.4 g protein, a good 17.4 g protein."}
    c = check_groundedness(menu, ctx())
    assert len(c.detail["numeric_claims"]) == 1


# -- completeness -------------------------------------------------------------

def test_completeness_counts_the_fields_the_prompt_asked_for():
    menu = {"recipe_id": "recipe_safe", "menu_name": "Rice salad",
            "why_it_fits": "A cold rice salad that suits a child who dislikes bread.",
            "nutritional_rationale": "Provides 280 kcal and 6.2 g of protein per serving.",
            "allergens_confirmed_absent": ["milk"], "source_citation": CITATION}
    c = check_completeness(menu, ctx(profile={"age_years": 8, "allergies": ["milk"]}))
    assert c.score == 1.0
    assert c.detail["missing"] == []


def test_a_rationale_without_numbers_is_incomplete():
    menu = {"recipe_id": "recipe_safe", "menu_name": "Rice salad",
            "why_it_fits": "A cold rice salad that suits this child well.",
            "nutritional_rationale": "It is nutritionally balanced and wholesome.",
            "allergens_confirmed_absent": [], "source_citation": CITATION}
    c = check_completeness(menu, ctx())
    assert "rationale_cites_numbers" in c.detail["missing"]
    assert c.score < 1.0


def test_allergen_coverage_is_skipped_when_the_child_has_no_restrictions():
    """A child with no allergies cannot fail to address them."""
    menu = {"recipe_id": "recipe_safe", "menu_name": "Rice salad",
            "why_it_fits": "A cold rice salad that suits this child well.",
            "nutritional_rationale": "Provides 280 kcal per serving of energy.",
            "allergens_confirmed_absent": [], "source_citation": CITATION}
    c = check_completeness(menu, ctx())
    assert c.detail["signals"]["allergens_addressed"] is True


# -- relevance ----------------------------------------------------------------

def test_relevance_is_not_applicable_without_preferences():
    """None, not zero. A sparse profile must not drag the mean down."""
    c = check_relevance({"recipe_id": "recipe_safe"},
                        ctx(profile={"age_years": 8, "allergies": []}))
    assert c.score is None


def test_a_disliked_ingredient_costs_relevance():
    c = check_relevance({"recipe_id": "recipe_safe"},
                        ctx(profile={"age_years": 8, "dislikes": ["courgette"]}))
    assert c.detail["sub_signals"]["dislikes_avoided"] == 0.0


def test_a_matched_like_earns_relevance():
    c = check_relevance({"recipe_id": "recipe_safe"},
                        ctx(profile={"age_years": 8, "likes": ["rice"]}))
    assert c.detail["sub_signals"]["likes_matched"] == 1.0


# -- citation -----------------------------------------------------------------

def test_exact_citation_scores_one():
    c = check_citation({"recipe_id": "recipe_safe", "source_citation": CITATION}, ctx())
    assert c.score == 1.0


def test_fabricated_citation_scores_zero():
    c = check_citation({"recipe_id": "recipe_safe",
                        "source_citation": "Journal of Imaginary Nutrition (2019)"}, ctx())
    assert c.score == 0.0
    assert c.detail["expected"] == CITATION


def test_case_and_whitespace_differences_are_a_rendering_fault_not_a_fabrication():
    c = check_citation({"recipe_id": "recipe_safe",
                        "source_citation": "  uk government   lunchbox recipe booklet (nhs/phe)  "},
                       ctx())
    assert c.score == 0.5


def test_missing_citation_scores_zero():
    c = check_citation({"recipe_id": "recipe_safe", "source_citation": ""}, ctx())
    assert c.score == 0.0


# -- retrieval ----------------------------------------------------------------

def test_a_recipe_offered_to_the_generator_scores_one():
    c = check_retrieval({"recipe_id": "recipe_safe"},
                        ctx(generation_candidates=["recipe_milk", "recipe_safe"]))
    assert c.score == 1.0
    assert c.detail["rank"] == 2


def test_retrieved_but_filtered_out_scores_half():
    c = check_retrieval({"recipe_id": "recipe_safe"},
                        ctx(generation_candidates=["recipe_milk"],
                            reranked_candidates=["recipe_milk", "recipe_safe"]))
    assert c.score == 0.5


def test_a_recipe_never_retrieved_scores_zero():
    c = check_retrieval({"recipe_id": "recipe_safe"},
                        ctx(generation_candidates=["recipe_milk"],
                            reranked_candidates=["recipe_milk"]))
    assert c.score == 0.0


def test_no_rag_scores_zero_and_says_why():
    """
    The no-RAG arm has no retrieval stage at all. Scoring it 0.0 is the finding,
    not a gap in measurement, so the evidence has to state it in words.
    """
    c = check_retrieval({"recipe_id": "recipe_safe"}, ctx())
    assert c.score == 0.0
    assert c.detail["no_retrieval"] is True
    assert "no retrieval stage" in c.evidence[0]
