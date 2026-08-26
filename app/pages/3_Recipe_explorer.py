"""
Recipe explorer -- the whole corpus, filterable, with no pipeline run needed.

Everything the recommender can possibly return is here: 29 recipes and nothing
else. That bound is worth being able to see. When a tight profile comes back
empty, this page is how you check whether the constraint was unsatisfiable or
the retrieval simply missed -- and it answers that without a key, a network
call, or an LLM.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lunchbite import bootstrap  # noqa: F401
from lunchbite import components, service, theme

from document_loader import ALL_14_ALLERGENS

st.set_page_config(page_title="Recipe explorer - LunchBite",
                   page_icon="\N{OPEN BOOK}", layout="wide")
components.inject_css()

st.title("Recipe explorer")

with st.spinner("Loading the recipe corpus."):
    recipes: List[Dict[str, Any]] = list(service.recipe_index().values())
st.caption(
    f"All {len(recipes)} recipes in the knowledge base. This is the complete "
    f"set the recommender draws from -- it never invents one."
)


@st.cache_data(show_spinner=False)
def corpus_frame() -> pd.DataFrame:
    """One flat row per recipe, for filtering and for the charts."""
    rows = []
    for recipe in service.recipe_index().values():
        nutrition = recipe.get("nutrition_per_serving") or {}
        rows.append({
            "id": recipe.get("id"),
            "Recipe": recipe.get("name"),
            "Category": recipe.get("meal_category"),
            "kcal": nutrition.get("energy_kcal"),
            "Sugars (g)": nutrition.get("sugars_g"),
            "Salt (g)": nutrition.get("salt_g"),
            "Protein (g)": nutrition.get("protein_g"),
            "Fibre (g)": nutrition.get("fibre_g"),
            "Prep (min)": recipe.get("prep_time_mins"),
            # Present on 20 of 29 recipes; None is "not recorded", not free.
            "Cost (USD)": recipe.get("lunch_cost_usd"),
            "Allergens": ", ".join(sorted(recipe.get("allergens_present") or [])),
            "Diet tags": ", ".join(sorted(recipe.get("diet_tags") or [])),
            "Source": recipe.get("source"),
        })
    return pd.DataFrame(rows)


frame = corpus_frame()

# -- Filters ------------------------------------------------------------------

st.markdown("### Filter")
row1 = st.columns(3)
with row1[0]:
    categories = st.multiselect("Category", sorted({r.get("meal_category") or ""
                                                    for r in recipes} - {""}))
with row1[1]:
    exclude = st.multiselect("Must not contain", ALL_14_ALLERGENS,
                             help="Uses the recipe's tagged allergen list, the "
                                  "same field the guardrail reads.")
with row1[2]:
    diet_tags = sorted({tag for r in recipes for tag in (r.get("diet_tags") or [])})
    require_tags = st.multiselect("Must be tagged", diet_tags)

row2 = st.columns(3)
kcal_values = [r["kcal"] for _, r in frame.iterrows() if pd.notna(r["kcal"])]
with row2[0]:
    kcal_range = st.slider("Energy (kcal)", int(min(kcal_values)),
                           int(max(kcal_values)),
                           (int(min(kcal_values)), int(max(kcal_values))))
with row2[1]:
    max_sugar = st.slider("Sugars at most (g)", 0.0,
                          float(frame["Sugars (g)"].max()),
                          float(frame["Sugars (g)"].max()), 0.5)
with row2[2]:
    max_salt = st.slider("Salt at most (g)", 0.0, float(frame["Salt (g)"].max()),
                         float(frame["Salt (g)"].max()), 0.1)

filtered = frame.copy()
if categories:
    filtered = filtered[filtered["Category"].isin(categories)]
if exclude:
    keep = []
    for _, row in filtered.iterrows():
        present = {a.strip() for a in str(row["Allergens"]).split(",") if a.strip()}
        keep.append(not (present & set(exclude)))
    filtered = filtered[keep]
if require_tags:
    keep = []
    for _, row in filtered.iterrows():
        tags = {t.strip() for t in str(row["Diet tags"]).split(",") if t.strip()}
        keep.append(set(require_tags).issubset(tags))
    filtered = filtered[keep]
filtered = filtered[
    filtered["kcal"].between(kcal_range[0], kcal_range[1])
    & (filtered["Sugars (g)"] <= max_sugar)
    & (filtered["Salt (g)"] <= max_salt)
]

st.markdown(f"### {len(filtered)} of {len(frame)} recipes match")
if filtered.empty:
    st.info(
        "Nothing matches. On a corpus of 29 that is easy to reach, and it is "
        "the same wall a tightly constrained child profile runs into."
    )
else:
    st.dataframe(filtered.drop(columns=["id"]), width="stretch", hide_index=True)

# -- Detail -------------------------------------------------------------------

if not filtered.empty:
    st.markdown("### Look at one")
    options = list(filtered["id"])
    chosen = st.selectbox(
        "Recipe", options,
        format_func=lambda rid: service.recipe_index()[rid].get("name", rid),
    )
    recipe = service.recipe_index()[chosen]
    # Rendered through the same card the recommender uses, with the recipe's own
    # fields standing in for a MenuOption -- so a recipe looks the same here as
    # it does when recommended.
    components.menu_card(
        {
            "recipe_id": recipe["id"],
            "menu_name": recipe.get("name"),
            "why_it_fits": recipe.get("description", ""),
            "nutritional_rationale": "",
            "allergens_confirmed_absent": sorted(
                set(ALL_14_ALLERGENS) - {a.lower() for a in recipe.get("allergens_present") or []}
            ),
            "source_citation": recipe.get("citation") or recipe.get("source", ""),
        },
        recipe,
        None,  # no age selected here, so no per-lunch ceilings to draw against
        1,
    )

# -- Corpus shape -------------------------------------------------------------

st.markdown("### The shape of the corpus")
st.caption(
    "Worth knowing before reading any recommendation: what the corpus is heavy "
    "in, and which restrictions it can barely serve."
)

left, right = st.columns(2)

with left:
    counts = (frame["Category"].value_counts().rename_axis("Category")
              .reset_index(name="Recipes"))
    components.bar_chart(counts, "Category", "Recipes", "Recipes per category")

with right:
    allergen_rows = []
    for allergen in ALL_14_ALLERGENS:
        n = sum(1 for r in recipes if allergen in (r.get("allergens_present") or []))
        allergen_rows.append({"Allergen": allergen, "Recipes containing": n})
    allergen_frame = pd.DataFrame(allergen_rows).sort_values("Recipes containing",
                                                             ascending=False)
    components.bar_chart(allergen_frame, "Allergen", "Recipes containing",
                         "Recipes containing each allergen")
    st.caption(
        "The tall bars are the restrictions that cost the most coverage: an "
        "allergy to one of them removes that many recipes from 29 before "
        "anything else is considered."
    )

st.markdown("#### Where each recipe sits on sugar and salt")
st.caption(
    "Both axes are per serving. The guideline ceilings depend on the child's "
    "age, so no threshold is drawn here -- pick an age on the main page and the "
    "cards show each lunch against that band."
)
components.scatter(
    frame[["Recipe", "Sugars (g)", "Salt (g)", "kcal"]],
    x="Sugars (g)", y="Salt (g)", label="Recipe",
    title="Sugar against salt, per serving",
)

missing_cost = int(frame["Cost (USD)"].isna().sum())
missing_steps = sum(1 for r in recipes if not r.get("method_steps"))
st.caption(
    f"Coverage gaps worth knowing: {missing_cost} of {len(recipes)} recipes "
    f"record no cost, and {missing_steps} carry no method steps. Blank means "
    f"not recorded in the source, never zero."
)
