"""
Weight sensitivity of the verifiable reward (src/reward/sensitivity.py).

The six weights in `model.DEFAULT_WEIGHTS` were chosen by argument, not fitted.
This analysis is the defence: rather than justify 0.35, show the ordering of the
arms survives changing it. That defence is only worth anything if the analysis
is capable of reporting instability when instability exists, so the case that
matters most here is `test_an_ordering_that_flips_is_reported_as_unstable` --
an analysis that always prints "stable" proves nothing.

Two other properties are load-bearing:

  the gate is a veto, not a term    `hostile` weights correctness at 0.02. If
                                    that let an unsafe menu score above zero,
                                    the weighting could buy a safety violation
                                    and the whole gate would be decorative.
  two implementations must agree    `reaggregate` recomputes what
                                    `model.score_menu` already did. If they
                                    disagree, every sensitivity number is noise.

Run:  pytest tests/test_sensitivity.py -v
"""

from __future__ import annotations

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from reward import checks as checks_mod  # noqa: E402
from reward.checks import RewardContext  # noqa: E402
from reward.model import DEFAULT_WEIGHTS, score_menu  # noqa: E402
from reward.scoring import score_run  # noqa: E402
from reward.sensitivity import (WEIGHTINGS, _ranking, reaggregate,  # noqa: E402
                                sensitivity, verify_reaggregation)

CITATION = "UK Government Lunchbox Recipe Booklet (NHS/PHE)"

RECIPES = {
    "recipe_safe": {
        "id": "recipe_safe", "name": "Rice salad", "description": "Cold rice salad.",
        "ingredients": ["cooked rice"],
        "nutrition_per_serving": {"energy_kcal": 280, "protein_g": 6.2},
        "allergens_present": [], "diet_tags": ["vegan"],
        "meal_category": "salad", "source": CITATION,
    },
    "recipe_milk": {
        "id": "recipe_milk", "name": "Cheese pitta", "description": "Cheesy pitta.",
        "ingredients": ["cheddar cheese"],
        "nutrition_per_serving": {"energy_kcal": 351, "protein_g": 17.4},
        "allergens_present": ["milk"], "diet_tags": ["vegetarian"],
        "meal_category": "sandwich/wrap", "source": CITATION,
    },
}


@pytest.fixture(autouse=True)
def patched_corpus(monkeypatch):
    monkeypatch.setattr(checks_mod, "get_recipe", lambda rid: RECIPES.get(rid))
    monkeypatch.delenv("REWARD_WEIGHTS", raising=False)
    yield


def menu(recipe_id, **over):
    m = {"recipe_id": recipe_id, "menu_name": "Lunch",
         "why_it_fits": "A cold rice salad that suits this child nicely.",
         "nutritional_rationale": "Provides 280 kcal and 6.2 g protein per serving.",
         "allergens_confirmed_absent": ["milk"], "source_citation": CITATION}
    m.update(over)
    return m


def record(mode, **components):
    """A minimal scored record: only component scores matter to re-aggregation."""
    return {"case_id": "STD-01", "repeat": 0, "mode": mode, "menu_index": 0,
            "reward": 0.0,
            "components": {k: {"score": v, "verifiable": True}
                           for k, v in components.items()}}


RUN = {
    "metadata": {"timestamp": "20260821_114840"},
    "results": [{
        "case_id": "STD-01", "repeat": 0,
        "profile": {"age_years": 8, "allergies": ["milk"], "likes": ["rice"]},
        "expected_unsafe_ids": ["recipe_milk"],
        "neurosymbolic": {"generation_candidates": ["recipe_safe"],
                          "proposed_menus": [menu("recipe_safe")]},
        "no_rag": {"generation_candidates": [],
                   "proposed_menus": [menu("recipe_milk",
                                           allergens_confirmed_absent=["peanuts"])]},
    }],
}


# -- the two implementations must agree ---------------------------------------

def test_reaggregation_reproduces_the_recorded_reward():
    scored = score_run(RUN)
    agrees, worst = verify_reaggregation(scored)
    assert agrees, "re-aggregation drifted by %.2e" % worst
    assert worst <= 1e-6


def test_reaggregation_matches_score_menu_directly():
    ctx = RewardContext(profile={"age_years": 8, "allergies": ["milk"], "likes": ["rice"]},
                        generation_candidates=["recipe_safe"])
    result = score_menu(menu("recipe_safe"), ctx)
    rec = {"components": {c.name: {"score": c.score, "verifiable": c.verifiable}
                          for c in result.checks}}
    assert reaggregate(rec, DEFAULT_WEIGHTS) == pytest.approx(result.reward, abs=1e-9)


# -- the gate is a veto, not a term -------------------------------------------

def test_the_hostile_weighting_cannot_rescue_an_unsafe_menu():
    """
    `hostile` weights correctness at 0.02. If lowering that weight let an unsafe
    menu score above zero, the gate would be a term in a sum rather than a veto
    -- and a policy could buy a safety violation by paying for it elsewhere.
    """
    unsafe = record("neural_rag", correctness=0.0, groundedness=1.0,
                    completeness=1.0, relevance=1.0,
                    citation_accuracy=1.0, retrieval_accuracy=1.0)
    assert reaggregate(unsafe, WEIGHTINGS["hostile"]) == 0.0
    assert reaggregate(unsafe, WEIGHTINGS["default"]) == 0.0


def test_the_gate_can_be_lifted_deliberately():
    unsafe = record("neural_rag", correctness=0.0, groundedness=1.0,
                    completeness=1.0, relevance=1.0,
                    citation_accuracy=1.0, retrieval_accuracy=1.0)
    ungated = reaggregate(unsafe, WEIGHTINGS["hostile"], gate_on_correctness=False)
    assert ungated is not None and ungated > 0.9


# -- stability reporting ------------------------------------------------------

def test_a_stable_ordering_is_reported_as_stable():
    scored = {"metadata": {"weights": dict(DEFAULT_WEIGHTS)}, "records": [
        record("good", correctness=1.0, groundedness=1.0, completeness=1.0,
               relevance=1.0, citation_accuracy=1.0, retrieval_accuracy=1.0),
        record("bad", correctness=1.0, groundedness=0.1, completeness=0.1,
               relevance=0.1, citation_accuracy=0.1, retrieval_accuracy=0.1),
    ]}
    out = sensitivity(scored)
    assert out["ordering_stable"] is True
    assert out["distinct_orderings"] == [["good", "bad"]]
    assert out["rank_range"]["good"] == {"best": 1, "worst": 1}


def test_an_ordering_that_flips_is_reported_as_unstable():
    """
    The test the whole analysis rests on. `a` wins on citation, `b` wins on
    groundedness, so weighting one over the other swaps them. An analysis that
    could not detect this would print "stable" unconditionally and be worthless
    as a defence of the weights.
    """
    scored = {"metadata": {"weights": dict(DEFAULT_WEIGHTS)}, "records": [
        record("a", correctness=1.0, groundedness=0.0, completeness=0.5,
               relevance=0.5, citation_accuracy=1.0, retrieval_accuracy=0.5),
        record("b", correctness=1.0, groundedness=1.0, completeness=0.5,
               relevance=0.5, citation_accuracy=0.0, retrieval_accuracy=0.5),
    ]}
    out = sensitivity(scored)
    assert out["ordering_stable"] is False
    assert len(out["distinct_orderings"]) > 1
    # Both arms move, and the report names them.
    moved = [m for m, r in out["rank_range"].items() if r["best"] != r["worst"]]
    assert set(moved) == {"a", "b"}


def test_every_named_weighting_is_reported():
    scored = {"metadata": {"weights": dict(DEFAULT_WEIGHTS)},
              "records": [record("only", correctness=1.0, groundedness=1.0,
                                 completeness=1.0, relevance=1.0,
                                 citation_accuracy=1.0, retrieval_accuracy=1.0)]}
    out = sensitivity(scored)
    assert set(out["per_weighting"]) == set(WEIGHTINGS)
    assert out["weightings_tested"] == list(WEIGHTINGS)


# -- degenerate inputs --------------------------------------------------------

def test_weights_are_validated_the_same_way_a_real_scoring_run_validates_them():
    scored = {"metadata": {}, "records": [record("m", correctness=1.0)]}
    with pytest.raises(ValueError, match="unknown checks"):
        sensitivity(scored, {"typo": {"correctnes": 1.0}})
    with pytest.raises(ValueError, match="non-negative"):
        sensitivity(scored, {"negative": {"correctness": -1.0}})


def test_an_arm_with_no_applicable_component_is_omitted_not_ranked_last():
    """
    None means "not measurable here", which is not the same as "worst". Ranking
    it last would invent a comparison the data does not support.
    """
    assert _ranking({"a": 0.5, "b": None, "c": 0.9}) == ["c", "a"]


def test_a_record_with_no_applicable_component_reaggregates_to_none():
    empty = {"components": {"relevance": {"score": None, "verifiable": True}}}
    assert reaggregate(empty, DEFAULT_WEIGHTS) is None
