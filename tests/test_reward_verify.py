"""
The verification harness for the reward (src/reward/verify.py).

A reward is only verifiable if someone other than its author can re-derive it,
so the harness that claims to do that has to be shown actually failing when the
numbers are wrong. A verifier that passes everything is worse than no verifier:
it converts an unchecked number into one carrying a false guarantee.

Each test below tampers with exactly one thing in a scored file and asserts the
verdict flips. Between them they cover the three properties the report leans on:
reproducible, model-free, and scored under a comparable reward_version.

Run:  pytest tests/test_reward_verify.py -v
"""

from __future__ import annotations

import copy
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from reward import checks as checks_mod  # noqa: E402
from reward.scoring import score_run  # noqa: E402
from reward.verify import (load_and_verify, verify_determinism,  # noqa: E402
                           verify_scored_run)
from reward.checks import RewardContext  # noqa: E402

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


def menu(recipe_id, **over):
    m = {"recipe_id": recipe_id, "menu_name": "Lunch",
         "why_it_fits": "A cold rice salad that suits this child nicely.",
         "nutritional_rationale": "Provides 280 kcal and 6.2 g protein per serving.",
         "allergens_confirmed_absent": ["milk"], "source_citation": CITATION}
    m.update(over)
    return m


RUN = {
    "metadata": {"timestamp": "20260821_114840", "git_sha": "abc1234",
                 "model": "groq/qwen/qwen3.6-27b", "synthetic": False},
    "results": [
        {
            "case_id": "STD-01", "repeat": 0,
            "profile": {"age_years": 8, "allergies": ["milk"], "likes": ["rice"]},
            "expected_unsafe_ids": ["recipe_milk"], "expected_safe_ids": ["recipe_safe"],
            "neurosymbolic": {
                "generation_candidates": ["recipe_safe"],
                "reranked_candidates": ["recipe_safe", "recipe_milk"],
                "proposed_menus": [menu("recipe_safe")],
                "final_menus": [menu("recipe_safe")],
            },
            "no_rag": {
                "generation_candidates": [], "reranked_candidates": [],
                "proposed_menus": [menu("recipe_milk", allergens_confirmed_absent=["peanuts"])],
                "final_menus": [menu("recipe_milk", allergens_confirmed_absent=["peanuts"])],
            },
        },
    ],
}


@pytest.fixture(autouse=True)
def patched_corpus(monkeypatch):
    monkeypatch.setattr(checks_mod, "get_recipe", lambda rid: RECIPES.get(rid))
    monkeypatch.delenv("REWARD_WEIGHTS", raising=False)
    yield


@pytest.fixture
def scored():
    return score_run(copy.deepcopy(RUN))


# -- the happy path -----------------------------------------------------------

def test_an_untouched_scored_file_verifies(scored):
    report = verify_scored_run(scored, RUN)
    assert report.passed, report.summary()
    assert report.reproducible and report.model_free and report.version_matches
    assert report.n_checked == report.n_records == len(scored["records"])


def test_the_summary_states_the_verdict(scored):
    text = verify_scored_run(scored, RUN).summary()
    assert "VERDICT: VERIFIED" in text
    assert "model-free           : yes" in text


def test_scoring_is_deterministic():
    ctx = RewardContext(profile={"age_years": 8, "allergies": ["milk"]},
                        generation_candidates=["recipe_safe"])
    assert verify_determinism(menu("recipe_safe"), ctx) is True


# -- tampering ----------------------------------------------------------------

def test_an_edited_reward_is_caught(scored):
    scored["records"][0]["reward"] = 0.999
    report = verify_scored_run(scored, RUN)
    assert not report.passed
    assert any(m.field == "reward" for m in report.mismatches)
    assert "VERDICT: FAILED" in report.summary()


def test_an_edited_component_score_is_caught(scored):
    scored["records"][0]["components"]["citation_accuracy"]["score"] = 1.0
    scored["records"][0]["components"]["citation_accuracy"]["score"] = 0.0
    report = verify_scored_run(scored, RUN)
    assert any(m.field == "citation_accuracy" for m in report.mismatches)


def test_an_edited_input_digest_is_caught(scored):
    scored["records"][0]["input_digest"] = "0" * 16
    report = verify_scored_run(scored, RUN)
    assert any(m.field == "input_digest" for m in report.mismatches)


def test_a_record_with_no_matching_run_row_is_caught(scored):
    scored["records"][0]["case_id"] = "STD-99"
    report = verify_scored_run(scored, RUN)
    assert any(m.field == "source_row" for m in report.mismatches)


def test_a_menu_index_past_the_end_of_the_run_is_caught(scored):
    scored["records"][0]["menu_index"] = 7
    report = verify_scored_run(scored, RUN)
    assert any(m.field == "menu_index" for m in report.mismatches)


def test_a_component_claiming_to_be_model_backed_fails_the_model_free_check(scored):
    """
    The reward is worth having only while nothing in it needs a model. A
    component that quietly grew one must fail verification rather than be
    averaged in beside the deterministic ones.
    """
    scored["records"][0]["components"]["relevance"]["verifiable"] = False
    report = verify_scored_run(scored, RUN)
    assert not report.model_free
    assert "relevance" in report.non_verifiable_components
    assert not report.passed


def test_a_reward_from_a_different_version_is_flagged_not_silently_compared(scored):
    scored["metadata"]["reward_version"] = 99
    report = verify_scored_run(scored, RUN)
    assert not report.version_matches
    assert not report.passed
    assert any("not comparable" in n for n in report.notes)


def test_reweighting_after_the_fact_is_caught(scored):
    """
    Comparing rewards alone would not catch this. Both records in this fixture
    sit at an extreme -- one scores 1.0 on every applicable check, the other is
    gated to 0.0 -- and neither moves under any weighting, which is why each
    record commits to a digest of the weights it was scored under.
    """
    scored["metadata"]["weights"]["correctness"] = 0.99
    report = verify_scored_run(scored, RUN)
    assert not report.reproducible
    assert any(m.field == "weights_digest" for m in report.mismatches)


# -- the file-level entry point -----------------------------------------------

def test_load_and_verify_finds_the_run_file_by_convention(tmp_path, scored):
    run_path = tmp_path / "run_20260821_114840.json"
    reward_path = tmp_path / "run_20260821_114840_reward.json"
    run_path.write_text(json.dumps(RUN), encoding="utf-8")
    reward_path.write_text(json.dumps(scored), encoding="utf-8")

    report = load_and_verify(str(reward_path))
    assert report.passed, report.summary()


def test_an_unconventional_filename_asks_for_the_run_file_explicitly(tmp_path, scored):
    odd = tmp_path / "scores.json"
    odd.write_text(json.dumps(scored), encoding="utf-8")
    with pytest.raises(ValueError, match="cannot infer"):
        load_and_verify(str(odd))
