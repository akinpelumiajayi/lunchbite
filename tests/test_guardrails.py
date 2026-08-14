"""
Unit tests for src/guardrails.py — the only module that makes safety decisions.

Two failure directions matter and they are not symmetric:

  false positive — a safe recipe is rejected. Costs coverage. On this corpus that
                   is the dominant failure, and it silently starves the pipelines
                   that use the gate.
  false negative — an unsafe recipe passes. Costs safety. Unacceptable.

Run:  pytest tests/ -v
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from guardrails import (  # noqa: E402
    ALLERGEN_KEYWORDS,
    ChildProfile,
    DEFAULT_LUNCH_FRACTION,
    _daily_to_lunch_fraction,
    _load_age_band_limits,
    check_recipe_against_profile,
    keyword_hit,
    lunch_fraction,
    normalize_allergy_terms,
    normalize_allergy_terms_with_unknowns,
)


def make_recipe(ingredients, allergens=None, sugars=1.0, salt=0.1, extras=None):
    return {
        "id": "recipe_test",
        "name": "Test recipe",
        "ingredients": list(ingredients),
        "allergens_present": list(allergens or []),
        "extras_suggested": list(extras or []),
        "nutrition_per_serving": {"sugars_g": sugars, "salt_g": salt},
    }


def passes(ingredients, allergy, **kw):
    recipe = make_recipe(ingredients, **kw)
    return check_recipe_against_profile(recipe, ChildProfile(age_years=8, allergies=[allergy]))


# ── False positives: safe food must not be rejected ───────────────────────────

@pytest.mark.parametrize("ingredients,allergy,why", [
    (["butternut squash", "rice"],        "milk",   "'butter' inside 'butternut'"),
    (["coconut milk", "rice"],            "milk",   "coconut milk is not dairy"),
    (["oat milk", "banana"],              "milk",   "oat milk is not dairy"),
    (["soy milk", "cereal"],              "milk",   "soy milk is not dairy"),
    (["almond butter", "bread"],          "milk",   "nut butter is not dairy"),
    (["peanut butter", "bread"],          "milk",   "'butter' in peanut butter"),
    (["eggplant", "tomato"],              "egg",    "'egg' inside 'eggplant'"),
    (["goat stew", "potato"],             "gluten", "'oat' inside 'goat'"),
    (["nutmeg", "apple"],                 "nuts",   "nutmeg is not a tree nut"),
    (["coconut flakes", "rice"],          "nuts",   "'nut' inside 'coconut'"),
])
def test_safe_ingredients_are_not_rejected(ingredients, allergy, why):
    result = passes(ingredients, allergy)
    assert result.passed, f"false positive ({why}): {result.reasons_for_rejection}"


# ── False negatives: genuinely unsafe food must be rejected ───────────────────

@pytest.mark.parametrize("ingredients,allergy", [
    (["whole milk", "cereal"],            "milk"),
    (["cheddar cheese", "bread"],         "milk"),
    (["butter", "toast"],                 "milk"),
    (["greek yoghurt", "berries"],        "milk"),
    (["egg", "toast"],                    "egg"),
    (["mayonnaise", "tuna"],              "egg"),
    (["rolled oats", "raisins"],          "gluten"),
    (["wheat flour", "water"],            "gluten"),
    (["peanut butter", "jam"],            "peanut"),
    (["walnut pieces", "salad"],          "nuts"),
    (["tahini", "chickpeas"],             "sesame"),
    (["salmon fillet", "rice"],           "fish"),
])
def test_unsafe_ingredients_are_rejected(ingredients, allergy):
    result = passes(ingredients, allergy)
    assert not result.passed, f"false negative: {ingredients} should violate '{allergy}'"


def test_tagged_allergen_is_rejected_even_without_keyword():
    """The tagged list is authoritative; it must not depend on ingredient wording."""
    recipe = make_recipe(["mystery filling"], allergens=["milk"])
    result = check_recipe_against_profile(recipe, ChildProfile(age_years=8, allergies=["milk"]))
    assert not result.passed


def test_oat_milk_still_flags_for_gluten():
    """False-friend masking is per allergen: oats in oat milk still matter for gluten."""
    result = passes(["oat milk", "banana"], "gluten")
    assert not result.passed


# ── Synonym normalisation ─────────────────────────────────────────────────────

@pytest.mark.parametrize("term,expected", [
    ("dairy", "milk"), ("lactose", "milk"), ("groundnut", "peanut"),
    ("coeliac", "cereals containing gluten"), ("shellfish", "crustaceans"),
    ("eggs", "egg"), ("soy", "soybeans"), ("tree nuts", "nuts"),
])
def test_synonyms_map_to_canonical_allergens(term, expected):
    assert expected in normalize_allergy_terms([term])


def test_unrecognised_term_is_reported():
    _, unknown = normalize_allergy_terms_with_unknowns(["kiwi"])
    assert "kiwi" in unknown


def test_unrecognised_term_surfaces_as_a_warning():
    """A weaker check must be visible, not silent."""
    recipe = make_recipe(["chicken", "rice"])
    result = check_recipe_against_profile(
        recipe, ChildProfile(age_years=8, allergies=["kiwi"]))
    assert any("Unrecognised allergy term" in w for w in result.warnings)


def test_known_term_produces_no_unknown_warning():
    recipe = make_recipe(["chicken", "rice"])
    result = check_recipe_against_profile(
        recipe, ChildProfile(age_years=8, allergies=["milk"]))
    assert not any("Unrecognised allergy term" in w for w in result.warnings)


# ── School nut-free policy ────────────────────────────────────────────────────

def test_school_nut_free_blocks_nuts_without_a_declared_allergy():
    recipe = make_recipe(["walnut pieces"], allergens=["nuts"])
    result = check_recipe_against_profile(
        recipe, ChildProfile(age_years=8, school_nut_free=True))
    assert not result.passed


# ── Nutrition ceilings ────────────────────────────────────────────────────────

@pytest.mark.parametrize("age", [4, 6, 7, 10, 11, 14, 15, 18])
def test_supported_age_bands_resolve(age):
    assert _load_age_band_limits(age) is not None


@pytest.mark.parametrize("age", [3, 19, 0, 99])
def test_ages_outside_the_supported_range_warn_rather_than_crash(age):
    recipe = make_recipe(["rice"])
    result = check_recipe_against_profile(recipe, ChildProfile(age_years=age))
    assert any("outside supported" in w for w in result.warnings)


def test_zero_override_is_honoured_not_treated_as_unset():
    """A 0.0 ceiling is falsy; `or` silently discarded it and used the default."""
    recipe = make_recipe(["rice"], sugars=1.0)
    result = check_recipe_against_profile(
        recipe, ChildProfile(age_years=8, max_sugar_g_override=0.0))
    assert not result.passed, "a 0 g sugar ceiling must reject a 1 g recipe"


def test_generous_override_admits_a_sugary_recipe():
    recipe = make_recipe(["rice"], sugars=40.0)
    result = check_recipe_against_profile(
        recipe, ChildProfile(age_years=8, max_sugar_g_override=100.0))
    assert result.passed


def test_daily_to_lunch_fraction_uses_the_conservative_value():
    assert _daily_to_lunch_fraction({"male": 24, "female": 23}, 0.5) == 11.5


def test_lunch_fraction_default_and_override(monkeypatch):
    monkeypatch.delenv("LUNCH_NUTRITION_FRACTION", raising=False)
    assert lunch_fraction() == DEFAULT_LUNCH_FRACTION
    monkeypatch.setenv("LUNCH_NUTRITION_FRACTION", "0.6")
    assert lunch_fraction() == pytest.approx(0.6)
    for bad in ("abc", "0", "1.5", "-0.2"):
        monkeypatch.setenv("LUNCH_NUTRITION_FRACTION", bad)
        assert lunch_fraction() == DEFAULT_LUNCH_FRACTION, f"{bad!r} should fall back"


# ── Determinism ───────────────────────────────────────────────────────────────

def test_check_is_deterministic():
    recipe = make_recipe(["cheddar cheese"], allergens=["milk"])
    profile = ChildProfile(age_years=8, allergies=["milk"])
    first = check_recipe_against_profile(recipe, profile)
    for _ in range(5):
        again = check_recipe_against_profile(recipe, profile)
        assert again.passed == first.passed
        assert again.reasons_for_rejection == first.reasons_for_rejection


def test_keyword_hit_respects_word_boundaries():
    assert keyword_hit("goat cheese", "goat", "milk") is True
    assert keyword_hit("goat cheese", "oat", "cereals containing gluten") is False
    assert keyword_hit("butternut squash", "butter", "milk") is False
    assert keyword_hit("melted butter", "butter", "milk") is True


def test_every_canonical_allergen_has_keywords():
    for allergen, keywords in ALLERGEN_KEYWORDS.items():
        assert keywords, f"{allergen} has no keywords"
