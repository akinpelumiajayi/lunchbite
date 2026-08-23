"""
Aggregation, gating and weighting of the verifiable reward (src/reward/model.py).

Two properties are load-bearing here and each has its own failure mode.

The correctness gate. Without it a reward-maximising policy can buy a safety
violation with good prose -- five well-cited paragraphs about a recipe the child
is allergic to would outscore a plain, safe one. That is the exact failure this
project exists to argue against, reproduced inside the reward function meant to
detect it, so the gate is asserted rather than trusted.

Renormalisation over applicable checks. A profile recording no likes cannot be
scored on relevance. Treating that as zero would punish sparse profiles for
information the annotator never had, and would make rewards from rich and thin
profiles incomparable.

Run:  pytest tests/test_reward_model.py -v
"""

from __future__ import annotations

import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from reward import checks as checks_mod  # noqa: E402
from reward.checks import RewardContext  # noqa: E402
from reward.model import (DEFAULT_WEIGHTS, load_weights, rank_menus,  # noqa: E402
                          score_menu)

CITATION = "UK Government Lunchbox Recipe Booklet (NHS/PHE)"

RECIPES = {
    "recipe_milk": {
        "id": "recipe_milk", "name": "Cheese pitta",
        "description": "Cheesy coleslaw in a pitta.", "ingredients": ["cheddar cheese"],
        "nutrition_per_serving": {"energy_kcal": 351, "protein_g": 17.4},
        "allergens_present": ["milk"], "diet_tags": ["vegetarian"],
        "meal_category": "sandwich/wrap", "source": CITATION,
    },
    "recipe_safe": {
        "id": "recipe_safe", "name": "Rice salad",
        "description": "Cold rice salad.", "ingredients": ["cooked rice"],
        "nutrition_per_serving": {"energy_kcal": 280, "protein_g": 6.2},
        "allergens_present": [], "diet_tags": ["vegan"],
        "meal_category": "salad", "source": CITATION,
    },
}


@pytest.fixture(autouse=True)
def patched_corpus(monkeypatch):
    monkeypatch.setattr(checks_mod, "get_recipe", lambda rid: RECIPES.get(rid))
    monkeypatch.delenv("REWARD_WEIGHTS", raising=False)
    yield


def good_menu(recipe_id="recipe_safe", **over):
    menu = {"recipe_id": recipe_id, "menu_name": "Lunch",
            "why_it_fits": "A cold rice salad that suits this child nicely.",
            "nutritional_rationale": "Provides 280 kcal and 6.2 g protein per serving.",
            "allergens_confirmed_absent": ["milk"], "source_citation": CITATION}
    menu.update(over)
    return menu


def ctx(**kw):
    profile = kw.pop("profile", {"age_years": 8, "allergies": ["milk"], "likes": ["rice"]})
    return RewardContext(profile=profile, **kw)


# -- the correctness gate -----------------------------------------------------

def test_an_unsafe_menu_scores_zero_however_good_the_rest_is():
    """Perfect citation, correct figures, matched preference -- still zero."""
    menu = good_menu("recipe_milk",
                     why_it_fits="A cheesy pitta this child would really enjoy.",
                     nutritional_rationale="Provides 351 kcal and 17.4 g protein.",
                     allergens_confirmed_absent=["peanuts"])
    result = score_menu(menu, ctx(generation_candidates=["recipe_milk"]))
    assert result.reward == 0.0
    assert result.gated is True
    # The pre-gate figure is kept so nothing about the decision is hidden.
    assert result.weighted_score > 0.5


def test_the_gate_can_be_lifted_to_show_what_it_suppressed():
    menu = good_menu("recipe_milk", allergens_confirmed_absent=["peanuts"])
    gated = score_menu(menu, ctx())
    ungated = score_menu(menu, ctx(), gate_on_correctness=False)
    assert gated.reward == 0.0
    assert ungated.reward == pytest.approx(ungated.weighted_score)
    assert ungated.reward > 0.0


def test_a_safe_menu_is_not_gated():
    result = score_menu(good_menu(), ctx(generation_candidates=["recipe_safe"]))
    assert result.gated is False
    assert result.reward == pytest.approx(result.weighted_score)
    assert result.reward > 0.8


# -- renormalisation ----------------------------------------------------------

def test_inapplicable_checks_are_excluded_from_the_weight_mass():
    """
    A profile with nothing to match on skips relevance entirely, so the mass
    drops by exactly that weight rather than the score dropping toward zero.
    """
    bare = {"age_years": 8, "allergies": ["milk"]}
    result = score_menu(good_menu(), ctx(profile=bare, generation_candidates=["recipe_safe"]))
    assert result.component_scores()["relevance"] is None
    assert result.applicable_weight == pytest.approx(
        sum(DEFAULT_WEIGHTS.values()) - DEFAULT_WEIGHTS["relevance"])


def test_a_sparse_profile_is_not_punished_for_what_was_never_recorded():
    rich = score_menu(good_menu(), ctx(generation_candidates=["recipe_safe"]))
    bare = score_menu(good_menu(), ctx(profile={"age_years": 8, "allergies": ["milk"]},
                                       generation_candidates=["recipe_safe"]))
    # The rich profile matched its like, so both should sit at the top of the
    # range; the sparse one must not be dragged down by the skipped check.
    assert bare.reward == pytest.approx(rich.reward, abs=0.05)


# -- weights ------------------------------------------------------------------

def test_weights_come_from_the_environment_when_set(monkeypatch):
    monkeypatch.setenv("REWARD_WEIGHTS", json.dumps({"correctness": 0.9}))
    assert load_weights()["correctness"] == 0.9
    # Unnamed checks keep their defaults rather than being dropped.
    assert load_weights()["groundedness"] == DEFAULT_WEIGHTS["groundedness"]


def test_a_typo_in_the_weights_is_refused_rather_than_ignored(monkeypatch):
    """
    Silently dropping an unknown key would leave the intended reweighting inert
    while the run looked like it had been applied.
    """
    monkeypatch.setenv("REWARD_WEIGHTS", json.dumps({"correctnes": 0.9}))
    with pytest.raises(ValueError, match="unknown checks"):
        load_weights()


def test_malformed_weight_json_is_refused(monkeypatch):
    monkeypatch.setenv("REWARD_WEIGHTS", "{not json")
    with pytest.raises(ValueError, match="not valid JSON"):
        load_weights()


def test_negative_and_empty_weights_are_refused():
    with pytest.raises(ValueError, match="non-negative"):
        load_weights({"correctness": -1.0})
    with pytest.raises(ValueError, match="sum to zero"):
        load_weights({k: 0.0 for k in DEFAULT_WEIGHTS})


def test_reweighting_moves_the_score_in_the_stated_direction():
    """Leaning the weights onto a component the menu fails must lower the reward."""
    menu = good_menu(source_citation="Invented Journal (2019)")
    base = score_menu(menu, ctx(generation_candidates=["recipe_safe"]))
    heavy = score_menu(menu, ctx(generation_candidates=["recipe_safe"]),
                       weights={"citation_accuracy": 0.60})
    assert heavy.reward < base.reward


# -- reproducibility ----------------------------------------------------------

def test_the_same_input_gives_the_same_reward_and_digest():
    a = score_menu(good_menu(), ctx(generation_candidates=["recipe_safe"]))
    b = score_menu(good_menu(), ctx(generation_candidates=["recipe_safe"]))
    assert a.reward == b.reward
    assert a.input_digest == b.input_digest


def test_changing_the_input_changes_the_digest():
    a = score_menu(good_menu(), ctx(generation_candidates=["recipe_safe"]))
    b = score_menu(good_menu(menu_name="Something else"),
                   ctx(generation_candidates=["recipe_safe"]))
    assert a.input_digest != b.input_digest


def test_every_component_is_model_free():
    """
    The one property that separates this from the LLM judge. If a component
    ever grows a model call, the fraction drops below 1.0 and this fails.
    """
    result = score_menu(good_menu(), ctx(generation_candidates=["recipe_safe"]))
    assert all(c.verifiable for c in result.checks)
    assert result.verifiable_fraction == 1.0


def test_the_result_serialises_with_its_evidence():
    d = score_menu(good_menu(), ctx(generation_candidates=["recipe_safe"])).as_dict()
    assert set(d["components"]) == set(DEFAULT_WEIGHTS)
    assert d["components"]["citation_accuracy"]["evidence"] == ["exact match"]
    assert d["reward_version"] >= 1


# -- best-of-N ----------------------------------------------------------------

def test_rank_menus_promotes_the_safe_menu_over_the_unsafe_one():
    menus = [good_menu("recipe_milk", allergens_confirmed_absent=["peanuts"]), good_menu()]
    ranked = rank_menus(menus, ctx(generation_candidates=["recipe_milk", "recipe_safe"]))
    assert ranked[0]["menu"]["recipe_id"] == "recipe_safe"
    assert ranked[0]["original_rank"] == 2
    assert ranked[1]["reward"].reward == 0.0


def test_ties_keep_the_generator_order():
    """
    A reward that cannot separate two menus must leave the pipeline behaving
    exactly as it did before, or reranking becomes a source of churn.
    """
    menus = [good_menu(menu_name="A"), good_menu(menu_name="B")]
    ranked = rank_menus(menus, ctx(generation_candidates=["recipe_safe"]))
    assert [r["menu"]["menu_name"] for r in ranked] == ["A", "B"]
    assert [r["original_rank"] for r in ranked] == [1, 2]
