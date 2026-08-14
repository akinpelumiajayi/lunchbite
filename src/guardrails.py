"""
guardrails.py

SAFETY-CRITICAL MODULE. Deterministic allergen and nutrition checking.
No LLM calls. Same input always produces same output.
Used by symbolic_prefilter and symbolic_postfilter nodes in both pipelines.
"""

from __future__ import annotations
import json
import os
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Dict, List, Optional, Set, Tuple

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")

DEFAULT_LUNCH_FRACTION = 0.40


def lunch_fraction() -> float:
    """
    Share of the daily nutrient maximum allotted to one lunch.

    Override with LUNCH_NUTRITION_FRACTION in .env. The default is unchanged at
    40%; see eval/check_data_quality.py before raising it, because on the current
    corpus the binding constraint is unreliable sugar data rather than the
    fraction itself.
    """
    try:
        value = float(os.environ.get("LUNCH_NUTRITION_FRACTION", DEFAULT_LUNCH_FRACTION))
    except (TypeError, ValueError):
        return DEFAULT_LUNCH_FRACTION
    return value if 0.0 < value <= 1.0 else DEFAULT_LUNCH_FRACTION

ALLERGEN_KEYWORDS: Dict[str, List[str]] = {
    "cereals containing gluten": ["wheat", "barley", "rye", "oat", "spelt", "khorasan",
                                   "gluten", "bread", "pasta", "pitta", "bagel", "bap", "wrap", "flour"],
    "crustaceans": ["prawn", "shrimp", "lobster", "crab", "crayfish"],
    "egg": ["egg", "mayonnaise", "mayo"],
    "fish": ["fish", "salmon", "tuna", "cod", "mackerel", "sardine", "kipper"],
    "peanut": ["peanut", "groundnut"],
    "soybeans": ["soya", "soy", "soybean", "tofu", "edamame"],
    "milk": ["milk", "cheese", "yoghurt", "yogurt", "butter", "cream", "lactose", "whey", "casein", "dairy"],
    "nuts": ["almond", "hazelnut", "walnut", "cashew", "pecan", "brazil nut", "pistachio", "macadamia"],
    "celery": ["celery", "celeriac"],
    "mustard": ["mustard"],
    "sesame": ["sesame", "tahini"],
    "sulphites": ["sulphite", "sulfite", "sulphur dioxide"],
    "lupin": ["lupin"],
    "molluscs": ["clam", "oyster", "scallop", "snail", "squid", "mussel"],
}

# Phrases that contain an allergen keyword as a whole word but do NOT indicate
# that allergen. Masked out of the text before scanning, per allergen — "oat milk"
# is masked for milk but deliberately NOT for gluten, where the oats do matter.
ALLERGEN_FALSE_FRIENDS: Dict[str, List[str]] = {
    "milk": [
        "coconut milk", "almond milk", "oat milk", "soy milk", "soya milk",
        "rice milk", "cashew milk", "hazelnut milk", "hemp milk",
        "peanut butter", "nut butter", "almond butter", "cashew butter",
        "cocoa butter", "shea butter", "apple butter", "milk thistle",
        "milk-free", "dairy-free", "non-dairy", "milk free", "dairy free",
    ],
    "egg": ["eggplant", "egg-free", "egg free"],
    "fish": ["fish-free", "fish sauce substitute"],
    "cereals containing gluten": ["gluten-free", "gluten free", "buckwheat"],
    "peanut": ["peanut-free", "peanut free"],
    "nuts": ["nut-free", "nut free", "nutmeg", "water chestnut"],
    "soybeans": ["soy-free", "soya-free", "soy free"],
    "celery": ["celery-free"],
    "sesame": ["sesame-free"],
}

_WORD_RE_CACHE: Dict[str, Any] = {}


def _word_regex(keyword: str):
    """
    Whole-word matcher, tolerant of a regular plural.

    Word boundaries stop 'oat' matching 'goat', 'egg' matching 'eggplant', and
    'butter' matching 'butternut'. The optional plural suffix is required so that
    tightening those boundaries does not lose real hits: 'oat' must still match
    'rolled oats', 'sulphite' must still match 'sulphites'.
    """
    rx = _WORD_RE_CACHE.get(keyword)
    if rx is None:
        rx = re.compile(rf"(?<![a-z]){re.escape(keyword)}(?:es|s)?(?![a-z])")
        _WORD_RE_CACHE[keyword] = rx
    return rx


def _mask_false_friends(text: str, allergen: str) -> str:
    """Blank out phrases that look like this allergen but are not."""
    for phrase in ALLERGEN_FALSE_FRIENDS.get(allergen, []):
        text = text.replace(phrase, " " * len(phrase))
    return text


def keyword_hit(text: str, keyword: str, allergen: str) -> bool:
    """True when `keyword` appears as a whole word, ignoring known false friends."""
    return _word_regex(keyword).search(_mask_false_friends(text, allergen)) is not None


ALLERGY_SYNONYMS: Dict[str, str] = {
    "dairy": "milk",
    "lactose": "milk",
    "lactose intolerant": "milk",
    "lactose-intolerant": "milk",
    "nut": "nuts",
    "tree nut": "nuts",
    "tree nuts": "nuts",
    "peanuts": "peanut",
    "groundnut": "peanut",
    "groundnuts": "peanut",
    "wheat": "cereals containing gluten",
    "gluten": "cereals containing gluten",
    "coeliac": "cereals containing gluten",
    "celiac": "cereals containing gluten",
    "eggs": "egg",
    "soy": "soybeans",
    "soya": "soybeans",
    "shellfish": "crustaceans",
    "fish allergy": "fish",
}


def normalize_allergy_terms(raw_terms: List[str]) -> Set[str]:
    """Map free-text allergy terms onto the canonical 14-allergen vocabulary."""
    normalized, _ = normalize_allergy_terms_with_unknowns(raw_terms)
    return normalized


def normalize_allergy_terms_with_unknowns(
    raw_terms: List[str],
) -> Tuple[Set[str], List[str]]:
    """
    Same as normalize_allergy_terms, but also reports terms that could not be
    mapped onto a known allergen.

    Unmapped terms are still enforced — they are matched literally against the
    ingredient text — but that only works when the recipe happens to use the same
    wording. Callers surface the unknowns as a warning so a silently weaker check
    is visible rather than invisible.
    """
    normalized: Set[str] = set()
    unknown: List[str] = []
    for term in raw_terms:
        t = term.strip().lower()
        if not t:
            continue
        if t in ALLERGY_SYNONYMS:
            normalized.add(ALLERGY_SYNONYMS[t])
        elif t in ALLERGEN_KEYWORDS:
            normalized.add(t)
        else:
            matched = False
            for canonical in ALLERGEN_KEYWORDS:
                if t in canonical or canonical in t:
                    normalized.add(canonical)
                    matched = True
            if not matched:
                normalized.add(t)
                unknown.append(t)
    return normalized, unknown


@dataclass
class ChildProfile:
    age_years: int
    allergies: List[str] = field(default_factory=list)
    intolerances: List[str] = field(default_factory=list)
    dislikes: List[str] = field(default_factory=list)
    likes: List[str] = field(default_factory=list)
    school_nut_free: bool = False
    cultural_context: str = ""
    max_sugar_g_override: Optional[float] = None
    max_salt_g_override: Optional[float] = None

    def all_restricted_allergens(self) -> Set[str]:
        restricted = normalize_allergy_terms(self.allergies + self.intolerances)
        if self.school_nut_free:
            restricted.add("nuts")
            restricted.add("peanut")
        return restricted


@dataclass
class GuardrailResult:
    passed: bool
    reasons_for_rejection: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@lru_cache(maxsize=1)
def _load_guidelines() -> Dict[str, Any]:
    """Cached — this was re-read from disk on every recipe check."""
    path = os.path.join(DATA_DIR, "nutrition_guidelines.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=32)
def _load_age_band_limits(age_years: int) -> Optional[Dict[str, Any]]:
    bands = _load_guidelines()["child_age_band_recommendations"]
    if 4 <= age_years <= 6:
        return bands["age_4_6"]
    elif 7 <= age_years <= 10:
        return bands["age_7_10"]
    elif 11 <= age_years <= 14:
        return bands["age_11_14"]
    elif 15 <= age_years <= 18:
        return bands["age_15_18"]
    return None


def _daily_to_lunch_fraction(daily_value: Any, fraction: Optional[float] = None) -> float:
    """Per-lunch ceiling from a daily maximum. `fraction` defaults to lunch_fraction()."""
    if fraction is None:
        fraction = lunch_fraction()
    if isinstance(daily_value, dict):
        daily_value = min(daily_value.values())   # most conservative of the sexes
    return round(float(daily_value) * fraction, 1)


def check_recipe_against_profile(recipe: Dict[str, Any], profile: ChildProfile) -> GuardrailResult:
    """Core deterministic gate. Pure function — no LLM, no side effects."""
    reasons: List[str] = []
    warnings: List[str] = []

    restricted = profile.all_restricted_allergens()
    _, unknown_terms = normalize_allergy_terms_with_unknowns(
        profile.allergies + profile.intolerances
    )
    if unknown_terms:
        warnings.append(
            f"Unrecognised allergy term(s): {', '.join(sorted(unknown_terms))}. "
            f"Not in the 14-allergen vocabulary, so these are only matched literally "
            f"against ingredient text — check manually."
        )

    recipe_allergens = set(a.lower() for a in recipe.get("allergens_present", []))

    direct_hits = restricted & recipe_allergens
    if direct_hits:
        reasons.append(
            f"Contains restricted allergen(s): {', '.join(sorted(direct_hits))}."
        )

    ingredient_text = " ".join(recipe.get("ingredients", [])).lower()
    for allergen in restricted:
        if allergen in direct_hits:
            continue
        for kw in ALLERGEN_KEYWORDS.get(allergen, [allergen]):
            if keyword_hit(ingredient_text, kw, allergen):
                reasons.append(
                    f"Ingredient text contains '{kw}' (restricted allergen: '{allergen}'). "
                    f"Not in tagged allergen list — verify manually."
                )
                break

    if profile.school_nut_free:
        nut_flagged = {"nuts", "peanut"} & direct_hits
        if ("nuts" in recipe_allergens or "peanut" in recipe_allergens) and not nut_flagged:
            reasons.append("Violates school nut-free policy.")

    extras_text = " ".join(recipe.get("extras_suggested", [])).lower()
    for allergen in restricted:
        for kw in ALLERGEN_KEYWORDS.get(allergen, [allergen]):
            if keyword_hit(extras_text, kw, allergen):
                warnings.append(
                    f"Suggested side item contains '{kw}' (restricted: {allergen}). "
                    f"Swap the side item for a safe alternative."
                )
                break

    band = _load_age_band_limits(profile.age_years)
    nutrition = recipe.get("nutrition_per_serving", {})

    if band is None:
        warnings.append(f"Age {profile.age_years} outside supported 4-18 range; nutrition limits not checked.")
    else:
        daily_sugar = band["free_sugars_g_day_max"]
        daily_salt = band["salt_g_day_max"]
        display_sugar = min(daily_sugar.values()) if isinstance(daily_sugar, dict) else daily_sugar
        display_salt = min(daily_salt.values()) if isinstance(daily_salt, dict) else daily_salt

        # `is not None`, not `or`: an override of 0.0 is falsy and was silently
        # discarded, falling back to the default ceiling — the opposite of intent.
        sugar_limit = (profile.max_sugar_g_override if profile.max_sugar_g_override is not None
                       else _daily_to_lunch_fraction(daily_sugar))
        salt_limit = (profile.max_salt_g_override if profile.max_salt_g_override is not None
                      else _daily_to_lunch_fraction(daily_salt))

        recipe_sugar = nutrition.get("sugars_g")
        recipe_salt = nutrition.get("salt_g")

        if recipe_sugar is not None and recipe_sugar > sugar_limit:
            reasons.append(
                f"Sugar content ({recipe_sugar}g) exceeds per-lunch ceiling "
                f"({sugar_limit}g, ~40% of {display_sugar}g daily max for age {profile.age_years})."
            )
        elif recipe_sugar is not None and recipe_sugar > sugar_limit * 0.85:
            warnings.append(f"Sugar content ({recipe_sugar}g) is near the per-lunch ceiling ({sugar_limit}g).")

        if recipe_salt is not None and recipe_salt > salt_limit:
            reasons.append(
                f"Salt content ({recipe_salt}g) exceeds per-lunch ceiling "
                f"({salt_limit}g, ~40% of {display_salt}g daily max for age {profile.age_years})."
            )
        elif recipe_salt is not None and recipe_salt > salt_limit * 0.85:
            warnings.append(f"Salt content ({recipe_salt}g) is near the per-lunch ceiling ({salt_limit}g).")

    return GuardrailResult(passed=len(reasons) == 0, reasons_for_rejection=reasons, warnings=warnings)


def filter_recipes(
    recipes: List[Dict[str, Any]], profile: ChildProfile
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    accepted, rejected = [], []
    for r in recipes:
        result = check_recipe_against_profile(r, profile)
        if result.passed:
            r_copy = dict(r)
            r_copy["_guardrail_warnings"] = result.warnings
            accepted.append(r_copy)
        else:
            rejected.append({"recipe_name": r.get("name", "unknown"), "reasons": result.reasons_for_rejection})
    return accepted, rejected
