"""
corpus.py -- Ground-truth lookups for the verifiable reward.

Deliberately dependency-light. The reward scorer runs offline over results
files already on disk, so importing `graphs.nodes` for `_recipes_by_id` would
drag in rank_bm25, chromadb and torch to read one JSON file — and would make
the reward depend on the retrieval stack it is supposed to be scoring.

Everything here is a pure read of `data/*.json`. Nothing calls a model.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"


def _read(name: str) -> Any:
    with open(_DATA_DIR / name, encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def recipes_by_id() -> Dict[str, Dict[str, Any]]:
    """recipe_id -> full recipe record, from data/recipes.json."""
    payload = _read("recipes.json")
    rows: List[Dict[str, Any]] = payload if isinstance(payload, list) else payload.get("recipes", [])
    return {r["id"]: r for r in rows if r.get("id")}


def get_recipe(recipe_id: Optional[str]) -> Optional[Dict[str, Any]]:
    if not recipe_id:
        return None
    return recipes_by_id().get(recipe_id)


def expected_citation(recipe: Dict[str, Any]) -> str:
    """
    The citation a menu for this recipe must carry.

    Mirrors `graphs.nodes.symbolic_postfilter` exactly -- `citation` if the
    corpus carries one, else `source`. If these two ever disagree the
    post-filter would repair a citation the reward then scored as wrong, so the
    fallback order is duplicated deliberately rather than approximated.
    """
    return (recipe.get("citation") or recipe.get("source") or "").strip()


# Nutrition keys the corpus records, mapped to the words a generated rationale
# uses for them. A numeric claim is only checkable when it names one of these.
NUTRIENT_ALIASES: Dict[str, tuple] = {
    "energy_kcal":     ("kcal", "calorie", "calories", "energy"),
    "energy_kj":       ("kj",),
    "protein_g":       ("protein",),
    "fibre_g":         ("fibre", "fiber"),
    "sugars_g":        ("sugar", "sugars"),
    "salt_g":          ("salt", "sodium"),
    "fat_g":           ("fat",),
    "saturates_g":     ("saturate", "saturates", "saturated"),
    "carbohydrate_g":  ("carbohydrate", "carbohydrates", "carbs", "carb"),
}
