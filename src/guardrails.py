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


NUTRITION_GATE_MODES = ("advisory", "hard", "off")
DEFAULT_NUTRITION_GATE = "advisory"


def nutrition_gate() -> str:
    """
    What a band-derived sugar/salt ceiling does when a recipe exceeds it:

      advisory (default) -- record it as a warning; the recipe still reaches the
                            LLM, and the warning travels with it
      hard               -- reject the recipe, as an allergen hit does
      off                -- do not check band ceilings at all

    Why advisory is the default
    ---------------------------
    Two independent defects make a *hard* band ceiling reject safe food:

    1. Unit mismatch. The recipe field is `sugars_g` -- TOTAL sugars -- and the
       guideline field is `free_sugars_g_day_max`. Lactose in yoghurt and
       fructose in fruit count toward the first and explicitly not the second,
       so every fruit- or dairy-containing lunch is charged sugar it does not
       owe. NHS recipe_001, a cheesy coleslaw pitta from the UK government
       lunchbox booklet, is rejected at age 7 on 10.1 g of mostly-vegetable
       sugar.

    2. Unreliable source data. eval/check_data_quality.py finds savoury dishes
       carrying 44 g of sugar and round placeholder values repeated across
       unrelated recipes. Nine UK Gov recipes median 4.4 g; twenty PACK-IT
       recipes median 28.5 g, at comparable energy.

    Together these rejected 21 of 29 recipes at age 7-10 on sugar alone --
    before a single allergen was considered. Measured over the 2026-08-18 run:
    161 of 199 pre-filter rejections were nutrition, not allergens; they dragged
    pre-filter precision to 0.477 and left 5 cases with zero safe candidates.
    Gating on allergens alone takes precision to 0.939 and zero-candidate cases
    to 0.

    The distinction being drawn is between *unsafe* and *less ideal*. An
    allergen in a recipe can hospitalise the child it is served to, and no
    prompt may bypass that gate. A lunch above a per-meal sugar guideline is a
    nutrition-quality judgement, and suppressing it entirely is what starves the
    generator of options. So it is surfaced, not enforced.

    Set NUTRITION_GATE=hard once the corpus carries free-sugar figures that
    eval/check_data_quality.py passes clean. A ceiling set explicitly by the
    caller (`max_sugar_g_override` / `max_salt_g_override` on ChildProfile) is
    always enforced regardless of this setting -- it is a deliberate instruction
    about one child, not an inference from suspect corpus data.
    """
    value = os.environ.get("NUTRITION_GATE", DEFAULT_NUTRITION_GATE).strip().lower()
    return value if value in NUTRITION_GATE_MODES else DEFAULT_NUTRITION_GATE


DIET_GATE_MODES = ("hard", "advisory", "off")
DEFAULT_DIET_GATE = "hard"


def diet_gate() -> str:
    """
    What a diet-requirement miss does: reject (`hard`, the default), warn
    (`advisory`), or nothing (`off`). Override with DIET_GATE in .env.

    Why this defaults to `hard` where NUTRITION_GATE defaults to `advisory`
    ----------------------------------------------------------------------
    The nutrition gate is advisory because its *data* is unreliable -- total
    sugars compared against a free-sugars guideline, on corpus figures that
    check_data_quality.py already flags. Nothing is wrong with the diet data:
    every one of the 29 recipes carries `diet_tags`, and the exclusions below
    are read from the ingredient list, the same field the allergen keyword scan
    trusts. A vegetarian profile served chicken is a plain failure to honour a
    stated requirement, not a judgement call about food quality.

    What this gate does NOT claim
    -----------------------------
    For `halal` and `kosher` it enforces *ingredient exclusions only* -- pork
    and its derivatives, plus shellfish for kosher. It cannot speak to slaughter
    method, certification, utensil separation, or the meat-and-dairy rule, none
    of which are represented anywhere in the corpus. Those requirements are
    marked non-certifiable below and always attach a warning saying so, because
    a system that answered "kosher: passed" off an ingredient scan would be
    making a claim it cannot support.
    """
    value = os.environ.get("DIET_GATE", DEFAULT_DIET_GATE).strip().lower()
    return value if value in DIET_GATE_MODES else DEFAULT_DIET_GATE

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


# ── Diet requirements ─────────────────────────────────────────────────────────
#
# Diet compliance used to be carried only in `cultural_context` -- free text that
# reached the generator's prompt and nothing else. That put halal, kosher and
# vegetarian on exactly the footing this project exists to argue against: the
# model was asked to honour them, and nothing checked that it had.

DIET_SYNONYMS: Dict[str, str] = {
    "veggie": "vegetarian",
    "meat-free": "vegetarian",
    "meat free": "vegetarian",
    "no meat": "vegetarian",
    "vegetarian household": "vegetarian",
    "plant-based": "vegan",
    "plant based": "vegan",
    "vegan-friendly": "vegan",
    "vegan household": "vegan",
    "pescetarian": "pescatarian",
    "halal diet": "halal",
    "kosher diet": "kosher",
}

_MEAT_TERMS = [
    "pork", "bacon", "ham", "gammon", "lard", "gelatin", "gelatine", "chicken",
    "beef", "turkey", "lamb", "mutton", "veal", "venison", "sausage",
    "pepperoni", "salami", "prosciutto", "chorizo", "meat",
]
_FISH_TERMS = [
    "fish", "tuna", "salmon", "cod", "haddock", "mackerel", "sardine",
    "anchovy", "prawn", "shrimp", "crab", "lobster", "shellfish",
]
_PORK_TERMS = [
    "pork", "bacon", "ham", "gammon", "lard", "gelatin", "gelatine",
    "pepperoni", "salami", "prosciutto", "chorizo",
]
_SHELLFISH_TERMS = [
    "prawn", "shrimp", "crab", "lobster", "shellfish", "clam", "oyster",
    "mussel", "scallop",
]

# `tags`       -- corpus diet_tags that satisfy the requirement; None means the
#                 corpus does not label this diet, so only exclusions apply.
# `excludes`   -- ingredient terms that violate it.
# `allergens`  -- canonical allergens whose presence violates it, read from the
#                 recipe's tagged allergen list rather than its prose.
# `certifiable`-- False when the check is necessarily partial; a warning saying
#                 so is attached on every pass, not just on failures.
DIET_SPECS: Dict[str, Dict[str, Any]] = {
    "vegetarian": {
        "tags": {"vegetarian", "vegan-friendly"},
        "excludes": _MEAT_TERMS + _FISH_TERMS,
        "allergens": set(),
        "certifiable": True,
    },
    "vegan": {
        "tags": {"vegan-friendly"},
        "excludes": _MEAT_TERMS + _FISH_TERMS,
        "allergens": {"milk", "egg"},
        "certifiable": True,
    },
    "pescatarian": {
        "tags": {"pescatarian", "vegetarian", "vegan-friendly"},
        "excludes": _MEAT_TERMS,
        "allergens": set(),
        "certifiable": True,
    },
    "halal": {
        "tags": None,
        "excludes": _PORK_TERMS,
        "allergens": set(),
        "certifiable": False,
    },
    "kosher": {
        "tags": None,
        "excludes": _PORK_TERMS + _SHELLFISH_TERMS,
        "allergens": set(),
        "certifiable": False,
    },
}


def normalize_diet_terms(raw_terms: List[str]) -> Tuple[Set[str], List[str]]:
    """
    Map free-text diet terms onto the vocabulary above.

    Returns (recognised, unknown). An unrecognised term is NOT silently enforced
    the way an unknown allergy term is -- there is no ingredient list to match it
    against -- so it is returned for the caller to surface as a warning. Claiming
    to honour a diet nobody implemented is the failure mode being avoided here.
    """
    recognised: Set[str] = set()
    unknown: List[str] = []
    for term in raw_terms:
        t = term.strip().lower()
        if not t:
            continue
        if t in DIET_SYNONYMS:
            recognised.add(DIET_SYNONYMS[t])
        elif t in DIET_SPECS:
            recognised.add(t)
        else:
            hit = next((d for d in DIET_SPECS if d in t), None)
            syn = next((v for k, v in DIET_SYNONYMS.items() if k in t), None)
            if hit:
                recognised.add(hit)
            elif syn:
                recognised.add(syn)
            else:
                unknown.append(t)
    # vegan is stricter than vegetarian; enforcing both would double-report the
    # same miss, so the stricter one stands alone.
    if "vegan" in recognised:
        recognised.discard("vegetarian")
    return recognised, unknown


@dataclass
class ChildProfile:
    age_years: int
    allergies: List[str] = field(default_factory=list)
    intolerances: List[str] = field(default_factory=list)
    dislikes: List[str] = field(default_factory=list)
    likes: List[str] = field(default_factory=list)
    school_nut_free: bool = False
    cultural_context: str = ""
    diet_requirements: List[str] = field(default_factory=list)
    max_sugar_g_override: Optional[float] = None
    max_salt_g_override: Optional[float] = None

    def all_restricted_allergens(self) -> Set[str]:
        restricted = normalize_allergy_terms(self.allergies + self.intolerances)
        if self.school_nut_free:
            restricted.add("nuts")
            restricted.add("peanut")
        return restricted

    def required_diets(self) -> Tuple[Set[str], List[str]]:
        """
        Recognised diet requirements and any terms that could not be mapped.

        `cultural_context` is read as well as the explicit list: existing
        benchmark profiles carry "vegetarian household" and "halal diet required
        - no pork products" there, and a requirement that was already being
        stated should not need restating to be enforced.
        """
        return normalize_diet_terms(self.diet_requirements + [self.cultural_context])


@dataclass
class GuardrailResult:
    passed: bool
    reasons_for_rejection: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    # Band-ceiling breaches on their own, so a caller can tell a
    # nutrition-quality note from an allergen near-miss without parsing prose.
    # Populated in every gate mode except "off". The same strings also appear in
    # `warnings` under an advisory gate and in `reasons_for_rejection` under a
    # hard one — never in both, since a rejection is not a warning.
    nutrition_flags: List[str] = field(default_factory=list)


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


def _band_label(age_years: int) -> str:
    """The nutrition_guidelines.json band key covering this age, for attribution."""
    for lo, hi in ((4, 6), (7, 10), (11, 14), (15, 18)):
        if lo <= age_years <= hi:
            return f"age_{lo}_{hi}"
    return "unsupported"


def lunch_limits_for_age(age_years: int) -> Optional[Dict[str, Any]]:
    """
    The per-lunch nutrient ceilings check_recipe_against_profile applies, plus
    the guideline kcal target, exposed for display.

    Returns None outside the 4-18 range the corpus covers -- the same condition
    under which the gate stops checking bands and warns instead. Callers must
    read None as "not checked", never as "a ceiling of zero".

    The sugar figure is a *free sugars* ceiling being compared against the
    corpus's *total* sugars, which is why breaching it is advisory by default
    (see nutrition_gate). `band_label` names the band the numbers came from so a
    caller can attribute them rather than present them as universal.
    """
    band = _load_age_band_limits(age_years)
    if band is None:
        return None

    fraction = lunch_fraction()
    daily_sugar = band["free_sugars_g_day_max"]
    daily_salt = band["salt_g_day_max"]

    return {
        "sugars_g": _daily_to_lunch_fraction(daily_sugar, fraction),
        "salt_g": _daily_to_lunch_fraction(daily_salt, fraction),
        # Most conservative of the sexes, matching the gate's own convention.
        "daily_sugars_g": (min(daily_sugar.values())
                           if isinstance(daily_sugar, dict) else daily_sugar),
        "daily_salt_g": (min(daily_salt.values())
                         if isinstance(daily_salt, dict) else daily_salt),
        "kcal_target": band.get("approx_lunch_target_kcal"),
        "fraction": fraction,
        "band_label": _band_label(age_years),
    }


def check_recipe_against_profile(recipe: Dict[str, Any], profile: ChildProfile) -> GuardrailResult:
    """Core deterministic gate. Pure function — no LLM, no side effects."""
    reasons: List[str] = []
    warnings: List[str] = []
    nutrition_flags: List[str] = []
    gate = nutrition_gate()

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

    # ── Diet requirements ────────────────────────────────────────────────────
    d_gate = diet_gate()
    if d_gate != "off":
        diets, unknown_diets = profile.required_diets()
        if unknown_diets:
            warnings.append(
                f"Unrecognised diet requirement(s): {', '.join(sorted(unknown_diets))}. "
                f"Not in the diet vocabulary, so these are NOT enforced — check manually."
            )
        recipe_tags = {t.lower() for t in recipe.get("diet_tags", [])}
        diet_text = " ".join(recipe.get("ingredients", [])).lower()

        for diet in sorted(diets):
            spec = DIET_SPECS[diet]
            misses: List[str] = []

            # The corpus tag is authoritative where it exists: it is curated per
            # recipe, where the scan below only sees whatever the ingredient
            # prose happens to name.
            if spec["tags"] is not None and not (recipe_tags & spec["tags"]):
                misses.append(f"not tagged {' or '.join(sorted(spec['tags']))}")

            bad_allergens = sorted(spec["allergens"] & recipe_allergens)
            if bad_allergens:
                misses.append(f"contains {', '.join(bad_allergens)}")

            for kw in spec["excludes"]:
                if keyword_hit(diet_text, kw, diet):
                    misses.append(f"ingredients contain '{kw}'")
                    break

            if misses:
                msg = f"Does not meet '{diet}' requirement: {'; '.join(misses)}."
                (reasons if d_gate == "hard" else warnings).append(msg)

            if not spec["certifiable"]:
                warnings.append(
                    f"'{diet}' is checked as an ingredient exclusion only. Certification, "
                    f"slaughter method and preparation separation are not represented in "
                    f"the corpus and are NOT verified."
                )

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

    # `is not None`, not `or`: an override of 0.0 is falsy and was silently
    # discarded, falling back to the default ceiling — the opposite of intent.
    sugar_is_explicit = profile.max_sugar_g_override is not None
    salt_is_explicit = profile.max_salt_g_override is not None

    # "off" suppresses the band-derived ceilings only. A ceiling the caller set
    # for this child is an instruction about one child rather than an inference
    # from the corpus, so it survives every gate mode.
    check_band = gate != "off" and band is not None
    check_explicit = sugar_is_explicit or salt_is_explicit

    if gate != "off" and band is None:
        warnings.append(f"Age {profile.age_years} outside supported 4-18 range; nutrition limits not checked.")

    if check_band or check_explicit:
        daily_sugar = band["free_sugars_g_day_max"] if band else None
        daily_salt = band["salt_g_day_max"] if band else None
        display_sugar = (min(daily_sugar.values()) if isinstance(daily_sugar, dict)
                         else daily_sugar)
        display_salt = (min(daily_salt.values()) if isinstance(daily_salt, dict)
                        else daily_salt)

        # None means "no ceiling applies" — an unchecked nutrient, not a zero one.
        sugar_limit = (profile.max_sugar_g_override if sugar_is_explicit
                       else _daily_to_lunch_fraction(daily_sugar) if check_band else None)
        salt_limit = (profile.max_salt_g_override if salt_is_explicit
                      else _daily_to_lunch_fraction(daily_salt) if check_band else None)

        recipe_sugar = nutrition.get("sugars_g")
        recipe_salt = nutrition.get("salt_g")

        pct = int(round(lunch_fraction() * 100))

        if sugar_limit is not None and recipe_sugar is not None and recipe_sugar > sugar_limit:
            note = (
                f"Sugar content ({recipe_sugar}g) exceeds per-lunch ceiling "
                f"({sugar_limit}g, ~{pct}% of {display_sugar}g daily max for age "
                f"{profile.age_years})."
            )
            if sugar_is_explicit:
                reasons.append(note)
            else:
                nutrition_flags.append(note)
        elif sugar_limit is not None and recipe_sugar is not None and recipe_sugar > sugar_limit * 0.85:
            nutrition_flags.append(
                f"Sugar content ({recipe_sugar}g) is near the per-lunch ceiling ({sugar_limit}g).")

        if salt_limit is not None and recipe_salt is not None and recipe_salt > salt_limit:
            note = (
                f"Salt content ({recipe_salt}g) exceeds per-lunch ceiling "
                f"({salt_limit}g, ~{pct}% of {display_salt}g daily max for age "
                f"{profile.age_years})."
            )
            if salt_is_explicit:
                reasons.append(note)
            else:
                nutrition_flags.append(note)
        elif salt_limit is not None and recipe_salt is not None and recipe_salt > salt_limit * 0.85:
            nutrition_flags.append(
                f"Salt content ({recipe_salt}g) is near the per-lunch ceiling ({salt_limit}g).")

    if gate == "hard":
        # Only the band-derived flags land here; a breach of an explicit override
        # was appended to `reasons` above, so this cannot duplicate one. They are
        # not also copied into `warnings` — a rejection is not a warning, and a
        # caller listing both would print each breach twice.
        reasons.extend(nutrition_flags)
    else:
        warnings.extend(nutrition_flags)

    return GuardrailResult(
        passed=len(reasons) == 0,
        reasons_for_rejection=reasons,
        warnings=warnings,
        nutrition_flags=nutrition_flags,
    )


def filter_recipes(
    recipes: List[Dict[str, Any]], profile: ChildProfile
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    accepted, rejected = [], []
    for r in recipes:
        result = check_recipe_against_profile(r, profile)
        if result.passed:
            r_copy = dict(r)
            r_copy["_guardrail_warnings"] = result.warnings
            r_copy["_nutrition_flags"] = result.nutrition_flags
            accepted.append(r_copy)
        else:
            rejected.append({"recipe_name": r.get("name", "unknown"), "reasons": result.reasons_for_rejection})
    return accepted, rejected
