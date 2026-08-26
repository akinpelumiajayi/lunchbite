"""
vocab.py -- Form options derived from the guardrail vocabularies.

The point is that the form cannot offer a term the gate cannot enforce. Every
option here comes out of `guardrails` or `document_loader` at import time, so
adding an allergen synonym or a diet spec to the safety module changes the form
without anyone remembering to update it.

Free text is still allowed everywhere the profile allows it -- the guardrail
already handles unrecognised terms by matching them literally and attaching a
warning, and suppressing that path in the UI would hide a real weakening of the
check rather than prevent it.
"""

from __future__ import annotations

from typing import Dict, List, Set

from lunchbite import bootstrap  # noqa: F401  (path setup must precede src imports)

from document_loader import ALL_14_ALLERGENS
from guardrails import (ALLERGY_SYNONYMS, DIET_SPECS, DIET_SYNONYMS,
                        normalize_allergy_terms_with_unknowns,
                        normalize_diet_terms)

# The canonical EU-14 vocabulary, in the corpus's own order.
ALLERGEN_OPTIONS: List[str] = list(ALL_14_ALLERGENS)

# Everyday words the gate maps onto a canonical allergen. Offered alongside the
# formal names because "dairy" and "gluten" are what a parent actually types.
ALLERGY_ALIAS_OPTIONS: List[str] = sorted(ALLERGY_SYNONYMS)

DIET_OPTIONS: List[str] = sorted(DIET_SPECS)
DIET_ALIAS_OPTIONS: List[str] = sorted(DIET_SYNONYMS)

# Diets whose check is necessarily partial. `check_recipe_against_profile`
# attaches a warning on every pass for these, not only on failures; the form
# says so up front rather than letting the warning be the first the user hears
# of it.
UNCERTIFIABLE_DIETS: List[str] = sorted(
    name for name, spec in DIET_SPECS.items() if not spec["certifiable"]
)

AGE_MIN, AGE_MAX = 4, 18


def allergen_label(term: str) -> str:
    """`dairy -> dairy (milk)` so an alias shows what it will be enforced as."""
    canonical = ALLERGY_SYNONYMS.get(term)
    return f"{term} ({canonical})" if canonical and canonical != term else term


def diet_label(term: str) -> str:
    canonical = DIET_SYNONYMS.get(term)
    if canonical and canonical != term:
        return f"{term} ({canonical})"
    if term in UNCERTIFIABLE_DIETS:
        return f"{term} - ingredient exclusion only"
    return term


def split_free_text(raw: str) -> List[str]:
    """Comma-separated free text -> a clean list, matching cli.py's prompt_list."""
    return [part.strip() for part in (raw or "").split(",") if part.strip()]


def preview_enforcement(allergy_terms: List[str], diet_terms: List[str],
                        cultural_context: str = "") -> Dict[str, object]:
    """
    What the gate will actually enforce, computed before the run.

    Runs the same normalisers `ChildProfile` uses, so the form can warn about an
    uninterpretable restriction while the user is still typing it rather than
    after an LLM call has been spent.

    `cultural_context` is kept separate from `diet_terms` on purpose.
    `ChildProfile.required_diets` scans both, so a term it cannot map in either
    one comes back in the same `unknown` list -- and the guardrail then warns
    that it is "NOT enforced". For a declared diet requirement that warning is
    exactly right. For cultural context it is a false alarm: the field is prose
    describing the setting, scanned opportunistically in case it happens to name
    a diet, and "British primary school" naming none is the ordinary case rather
    than a restriction going unchecked. Splitting them lets the form say the
    accurate thing about each.
    """
    allergens, unknown_allergens = normalize_allergy_terms_with_unknowns(allergy_terms)
    diets, unknown_diets = normalize_diet_terms(diet_terms)

    context_diets: Set[str] = set()
    context_unmatched = ""
    if cultural_context.strip():
        context_diets, context_unknown = normalize_diet_terms([cultural_context])
        if context_unknown and not context_diets:
            context_unmatched = cultural_context.strip()

    all_diets = diets | context_diets
    return {
        "allergens": sorted(allergens - set(unknown_allergens)),
        "unknown_allergens": sorted(unknown_allergens),
        "diets": sorted(all_diets),
        "diets_from_context": sorted(context_diets - diets),
        "unknown_diets": sorted(unknown_diets),
        "context_unmatched": context_unmatched,
        "uncertifiable": sorted(all_diets & set(UNCERTIFIABLE_DIETS)),
    }


# A few profiles worth one click, drawn from constraint shapes the corpus
# actually exercises.
#
# The last two are deliberate: allergen restrictions alone cannot empty this
# corpus, because one recipe (Black Beans and Rice) carries no tagged allergen
# and is vegan- and gluten-free, so it survives even a profile restricting all
# 14. The empty state is reachable only through an explicit per-lunch ceiling,
# which is a hard rejection where a band breach is merely advisory. Labelling a
# preset "expect no result" when it in fact returns one would be exactly the
# kind of claim this project checks everywhere else.
PRESETS: Dict[str, Dict[str, object]] = {
    "Milk allergy, nut-free school": {
        "age_years": 7,
        "allergies": ["milk"],
        "school_nut_free": True,
        "likes": ["sandwiches", "pasta"],
        "cultural_context": "British primary school",
    },
    "Coeliac, age 9": {
        "age_years": 9,
        "allergies": ["gluten"],
        "likes": ["rice", "chicken"],
    },
    "Vegetarian, egg allergy": {
        "age_years": 8,
        "allergies": ["egg"],
        "diet_requirements": ["vegetarian"],
        "likes": ["cheese", "beans"],
    },
    "Vegan, age 11": {
        "age_years": 11,
        "diet_requirements": ["vegan"],
        "dislikes": ["mushrooms"],
    },
    "Halal, fish allergy": {
        "age_years": 10,
        "allergies": ["fish"],
        "diet_requirements": ["halal"],
    },
    "Tightly restricted (narrows to one recipe)": {
        "age_years": 7,
        "allergies": ["milk", "egg", "gluten", "fish", "soya", "sesame"],
        "diet_requirements": ["vegan"],
        "school_nut_free": True,
    },
    "Impossible ceiling (shows the empty state)": {
        "age_years": 7,
        "diet_requirements": ["vegan"],
        "max_sugar_g_override": 1.0,
    },
}
