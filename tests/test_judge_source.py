"""
Unit tests for the SOURCE text handed to the LLM judge (evaluator.source_text).

These guard the fix for the 0.000 faithfulness floor. In run 20260816_160852
every no_llm menu (n=22) and every neurosymbolic menu (n=25) scored exactly
0.000 with a [0.000, 0.000] interval — including menus whose every nutrition
figure matched the corpus to the decimal. The cause was not the models: SOURCE
omitted the allergen fields, so "free from milk" had nothing to be checked
against, and the judge said so in its own reasoning ("the source does not
mention the absence of various allergens").

What must hold is that every claim type the menus actually make is *decidable*
from SOURCE. If a field disappears from here again, faithfulness silently
becomes unscoreable rather than failing loudly.

Run:  pytest tests/test_judge_source.py -v
"""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "benchmark"))

import evaluator  # noqa: E402
from document_loader import ALL_14_ALLERGENS, _load_json  # noqa: E402

RECIPE = {
    "id": "recipe_005",
    "name": "Salmon and salad bagel",
    "ingredients": ["1 wholemeal bagel", "small can of pink salmon",
                    "1 tablespoon reduced-calorie mayonnaise"],
    "nutrition_per_serving": {"energy_kcal": 376, "fat_g": 11.0, "saturates_g": 1.9,
                              "carbohydrate_g": 40.2, "sugars_g": 4.4, "fibre_g": 5.6,
                              "protein_g": 26.2, "salt_g": 1.5},
    "allergens_present": ["fish", "cereals containing gluten", "egg"],
}


def test_declared_present_allergens_appear():
    src = evaluator.source_text(RECIPE)
    assert "Allergens declared present" in src
    for allergen in RECIPE["allergens_present"]:
        assert allergen in src


def test_absent_allergens_are_stated_not_left_to_inference():
    """
    The claim under test is the commonest one in this domain: "free from milk".
    The corpus only records allergens *present*, so unless the absent list is
    written out the judge is asked to verify a negative against silence.
    """
    src = evaluator.source_text(RECIPE)
    absent_line = [ln for ln in src.splitlines()
                   if ln.startswith("Allergens declared absent")][0]
    assert "milk" in absent_line
    assert "peanut" in absent_line
    # and the present ones are not also listed as absent
    for allergen in RECIPE["allergens_present"]:
        assert allergen not in absent_line


def test_present_and_absent_partition_the_fourteen():
    src = evaluator.source_text(RECIPE)
    present = set(RECIPE["allergens_present"])
    absent = {a.strip() for a in
              src.split("Allergens declared absent: ")[1].split(", ")}
    assert present | absent == set(ALL_14_ALLERGENS)
    assert not (present & absent)


def test_nutrition_figures_are_labelled_and_verbatim():
    """Menus quote these numbers exactly; the judge must be able to match them."""
    src = evaluator.source_text(RECIPE)
    assert "376 kcal" in src
    assert "4.4 g sugars" in src
    assert "1.5 g salt" in src
    assert "26.2 g protein" in src
    # Not a Python dict repr — those read as one opaque blob to the judge.
    assert "{" not in src and "'" not in src


def test_missing_nutrition_is_marked_not_dropped_silently():
    src = evaluator.source_text({"name": "x", "ingredients": [], "allergens_present": []})
    assert "(not stated)" in src


def test_recipe_with_no_declared_allergens_says_none():
    src = evaluator.source_text({**RECIPE, "allergens_present": []})
    assert "Allergens declared present (EU FIC 14): none" in src


def test_every_corpus_recipe_renders():
    """The real corpus, not just the fixture — a missing field must not raise."""
    for recipe in _load_json("recipes.json"):
        src = evaluator.source_text(recipe)
        assert src.startswith("Name: ")
        assert "Allergens declared absent:" in src


def test_rubric_tells_the_judge_how_to_score_an_absence_claim():
    """
    Rendering the absent list is only half the fix; the rubric has to say that a
    negative claim is scoreable. Without this the judge defaulted to treating
    "asserts an absence" as "unsupported".
    """
    assert "absence claim is always decidable" in evaluator._RUBRIC
    assert "Allergens declared absent" in evaluator._RUBRIC
    # Process descriptions ("selected by rule-based scoring") are what dragged
    # the no_llm arm down; they must be explicitly excluded from the claim count.
    assert "neither supported nor unsupported" in evaluator._RUBRIC
