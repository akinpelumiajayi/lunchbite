"""
document_loader.py

Loads data/*.json knowledge base files and converts each record into
retrievable text Chunk objects with attached metadata.

Key design: each recipe chunk includes an explicit "free from X, Y, Z"
sentence listing all 14 allergens NOT present in that recipe. This gives
both BM25 and TF-IDF embedding a real lexical signal for allergen absence
(not just presence), partially mitigating the negation blind spot that
pure bag-of-words methods have.
"""

from __future__ import annotations
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")

# Canonical EU FIC / UK 14 declarable allergens
ALL_14_ALLERGENS: List[str] = [
    "cereals containing gluten", "crustaceans", "egg", "fish", "peanut",
    "soybeans", "milk", "nuts", "celery", "mustard", "sesame",
    "sulphites", "lupin", "molluscs",
]


@dataclass
class Chunk:
    """A single retrievable unit of text plus structured metadata."""
    id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


def _load_json(filename: str) -> Any:
    path = os.path.join(DATA_DIR, filename)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_recipe_chunks() -> List[Chunk]:
    recipes = _load_json("recipes.json")
    chunks = []
    for r in recipes:
        nutrition = r["nutrition_per_serving"]
        present = set(a.lower() for a in r["allergens_present"])
        absent = sorted(a for a in ALL_14_ALLERGENS if a not in present)
        free_from = (
            f"This recipe is free from, does not contain, and excludes: {', '.join(absent)}."
            if absent else "This recipe may contain all 14 major allergens."
        )

        text = (
            f"Recipe: {r['name']}\n"
            f"Category: {r['meal_category']}\n"
            f"Description: {r['description']}\n"
            f"Ingredients: {', '.join(r['ingredients'])}\n"
            f"Extras suggested: {', '.join(r.get('extras_suggested', []))}\n"
            f"Nutrition per serving: {nutrition['energy_kcal']} kcal, "
            f"{nutrition['fat_g']}g fat ({nutrition['saturates_g']}g saturates), "
            f"{nutrition['carbohydrate_g']}g carbohydrate ({nutrition['sugars_g']}g sugars), "
            f"{nutrition['fibre_g']}g fibre, {nutrition['protein_g']}g protein, "
            f"{nutrition['salt_g']}g salt.\n"
            f"Allergens present: {', '.join(r['allergens_present']) if r['allergens_present'] else 'none of the 14 major allergens'}.\n"
            f"{free_from}\n"
            f"Allergen notes: {r['allergen_notes']}\n"
            f"Diet tags: {', '.join(r['diet_tags'])}\n"
            f"Cultural context: {r.get('cultural_context', '')}"
        )
        chunks.append(Chunk(
            id=r["id"],
            text=text,
            metadata={
                "source_type": "recipe",
                "source_doc": r["source"],
                "name": r["name"],
                "allergens_present": r["allergens_present"],
                "diet_tags": r["diet_tags"],
                "meal_category": r["meal_category"],
                "energy_kcal": nutrition["energy_kcal"],
                "sugars_g": nutrition["sugars_g"],
                "salt_g": nutrition["salt_g"],
                "saturates_g": nutrition["saturates_g"],
                "protein_g": nutrition["protein_g"],
                "raw_recipe": json.dumps(r),
            },
        ))
    return chunks


def load_allergen_chunks() -> List[Chunk]:
    data = _load_json("allergen_rules.json")
    chunks = []
    for a in data["the_14_allergens"]:
        text = (
            f"Allergen: {a['name']} (EU FIC Annex II)\n"
            f"Includes: {', '.join(a['includes'])}\n"
            f"Declaration note: {a['declaration_note']}"
        )
        chunks.append(Chunk(
            id=f"allergen_{a['id']}",
            text=text,
            metadata={"source_type": "allergen_rule", "allergen_id": a["id"], "allergen_name": a["name"]},
        ))

    facts_text = "Key facts for schools and caterers on allergen management:\n" + "\n".join(
        f"- {fact}" for fact in data["key_facts_for_schools_and_caterers"]
    )
    chunks.append(Chunk(
        id="allergen_school_facts",
        text=facts_text,
        metadata={"source_type": "allergen_rule", "allergen_id": "school_caterer_facts"},
    ))

    for intol in data["common_intolerances_not_legally_classed_as_allergens"]:
        safe_id = intol["name"].lower().replace(" ", "_").replace("/", "_").replace("__", "_")
        chunks.append(Chunk(
            id=f"intolerance_{safe_id}",
            text=f"Intolerance (not Annex II): {intol['name']}\n{intol['note']}",
            metadata={"source_type": "allergen_rule", "allergen_id": intol["name"].lower()},
        ))
    return chunks


def load_nutrition_guideline_chunks() -> List[Chunk]:
    data = _load_json("nutrition_guidelines.json")
    chunks = []

    for principle in data["eatwell_guide_principles"]:
        safe_id = (principle["food_group"].lower()
                   .replace(" ", "_").replace(",", "").replace("/", "_"))
        text = (
            f"Eatwell Guide food group: {principle['food_group']}\n"
            f"Guidance: {principle['guidance']}\n"
            f"Lunchbox application: {principle['lunchbox_application']}\n"
            f"Caveats: {principle.get('caveats', 'none')}"
        )
        chunks.append(Chunk(
            id=f"eatwell_{safe_id}",
            text=text,
            metadata={"source_type": "nutrition_guideline", "food_group": principle["food_group"]},
        ))

    for age_band, vals in data["child_age_band_recommendations"].items():
        if age_band == "note":
            continue
        text = (
            f"Government dietary recommendation for age band "
            f"{age_band.replace('age_', '').replace('_', '-')} years:\n"
            + json.dumps(vals, indent=2)
        )
        chunks.append(Chunk(
            id=f"dietrec_{age_band}",
            text=text,
            metadata={"source_type": "nutrition_guideline", "age_band": age_band},
        ))

    chunks.append(Chunk(
        id="school_safe_constraints",
        text="School-safe lunch constraints:\n" + "\n".join(
            f"- {c}" for c in data["school_safe_constraints"]
        ),
        metadata={"source_type": "nutrition_guideline", "topic": "school_safe_constraints"},
    ))
    return chunks


def load_all_chunks() -> List[Chunk]:
    return load_recipe_chunks() + load_allergen_chunks() + load_nutrition_guideline_chunks()
