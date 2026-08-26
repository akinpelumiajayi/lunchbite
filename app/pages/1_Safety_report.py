"""
Safety report -- what the guardrail rejected, and on what grounds.

This is the page the project exists to make visible. The claim under test is
that a deterministic constraint layer catches things a prompt-instructed model
does not, and that claim is only inspectable if every rejection states its own
reason next to the recipe it removed.

Two distinctions are load-bearing here and the page keeps them apart:

* An **allergen** rejection is a hard safety gate. A **nutrition** flag is
  advisory by default -- the recipe still reached the model. Presenting them in
  one undifferentiated list would let a reader conclude that every rejected
  recipe was dangerous.
* Rejections **before** generation are the model never seeing the recipe.
  Rejections **after** are the model having proposed something its own claims
  did not survive. The second is the more interesting result.
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

st.set_page_config(page_title="Safety report - LunchBite",
                   page_icon="\N{SHIELD}", layout="wide")
components.inject_css()

st.title("Safety report")

payload = service.require_last_run()
if payload is None:
    st.stop()

result: Dict[str, Any] = payload["result"]
state: Dict[str, Any] = payload["state"]
arm: str = payload["arm"]

st.caption(
    f"From the **{theme.ARMS[arm]['label']}** run for a "
    f"{result['profile'].age_years}-year-old."
)

# The arms without symbolic gates leave both logs empty. That is the comparison,
# not a missing page, so it is said plainly rather than shown as no data.
if arm in ("neural_rag", "no_rag"):
    st.warning(
        f"**{theme.ARMS[arm]['label']} has no symbolic gate.** Nothing was "
        f"filtered before the model was asked and nothing was verified after, "
        f"so both tables below are empty by construction. Whatever that run "
        f"recommended was accepted on the model's word alone. Re-run under "
        f"**Neurosymbolic** to see the same profile with the gates in place."
    )

# -- Funnel -------------------------------------------------------------------

st.markdown("### Where the candidates went")
stages = service.funnel_counts(state, result)
components.funnel(stages, "Recipes surviving each stage")

# The first drop is usually not a rejection and should not be read as one.
retrieved, checked = stages[0][1], stages[1][1]
if retrieved > checked:
    st.caption(
        f"The drop from {retrieved} retrieved to {checked} checked is the "
        f"cross-encoder keeping only its top {checked}, not the guardrail "
        f"rejecting anything. Rejections start at the next bar."
    )
if state.get("refine_count"):
    st.caption(
        f"Retrieval ran {state['refine_count']} extra time(s) because the first "
        f"pass came up short, so more recipes were retrieved than a single pass "
        f"would have produced."
    )

# -- Rejected before generation ----------------------------------------------

rejected_pre: List[Dict[str, Any]] = result.get("rejected_at_retrieval") or []
approved = len(state.get("generation_candidates") or [])

st.markdown("### Rejected before the model was asked")

if not rejected_pre:
    if arm in ("neural_rag", "no_rag"):
        st.caption("No pre-filter runs in this arm.")
    else:
        st.success(
            f"Nothing was rejected. All {approved} retrieved recipe(s) met "
            f"every allergen and diet constraint for this profile."
        )
else:
    rows = []
    for entry in rejected_pre:
        reasons = entry.get("reasons") or []
        rows.append({
            "Recipe": entry.get("recipe_name") or entry.get("recipe_id"),
            "Rejected because": " ".join(str(r) for r in reasons),
            "Also flagged": " ".join(str(f) for f in (entry.get("nutrition_flags") or [])),
        })
    frame = pd.DataFrame(rows)
    st.dataframe(frame, width="stretch", hide_index=True)

    st.caption(
        f"{len(rejected_pre)} of {len(state.get('symbolic_pre_filter_log') or [])} "
        f"checked recipes were removed before generation. Every reason above is "
        f"a deterministic check against the recipe record -- no model was "
        f"involved in any of these decisions."
    )
    st.info(
        "**A rejection here is not proof a recipe was unsafe.** The pre-filter "
        "over-blocks: measured precision is 0.477, so roughly half of the "
        "recipes it removes would in fact have been fine. It is deliberately "
        "tuned that way -- missing a real allergen costs more than dropping an "
        "acceptable lunch -- and it is why this arm returns fewer options than "
        "the ungated one. That trade is the finding, not a defect."
    )

# -- Advisory flags on recipes that passed ------------------------------------

flagged_passing = [
    e for e in (state.get("symbolic_pre_filter_log") or [])
    if e.get("passed") and e.get("nutrition_flags")
]
if flagged_passing:
    st.markdown("### Passed, but flagged on nutrition")
    st.caption(
        "The nutrition gate is advisory by default, so these recipes still "
        "reached the model. The corpus records total sugars while the guideline "
        "specifies free sugars, so a breach here is a quality note rather than "
        "a safety finding -- hard-gating on it rejected 21 of 29 recipes at "
        "ages 7-10 on sugar alone."
    )
    st.dataframe(
        pd.DataFrame([{
            "Recipe": e.get("recipe_name") or e.get("recipe_id"),
            "Flag": " ".join(str(f) for f in (e.get("nutrition_flags") or [])),
        } for e in flagged_passing]),
        width="stretch", hide_index=True,
    )

# -- Rejected after generation ------------------------------------------------

post_log: List[Dict[str, Any]] = state.get("symbolic_post_filter_log") or []
rejected_post: List[Dict[str, Any]] = result.get("rejected_at_post_filter") or []

st.markdown("### The model's own claims, checked")

if not post_log:
    if arm in ("neural_rag", "no_rag"):
        st.caption("No post-filter runs in this arm -- nothing was checked.")
    else:
        st.caption("The model proposed nothing, so there was nothing to verify.")
else:
    survived = [e for e in post_log if e.get("survived")]
    corrected = [e for e in post_log if e.get("citation_corrected")]
    false_claims = [e for e in post_log if e.get("false_allergen_claim")]

    cols = st.columns(4)
    cols[0].metric("Proposed", len(post_log))
    cols[1].metric("Survived", len(survived))
    cols[2].metric("Rejected", len(rejected_post))
    cols[3].metric("Citations repaired", len(corrected))

    if rejected_post:
        st.markdown("**Rejected at the final gate**")
        st.dataframe(
            pd.DataFrame([{
                "Menu": e.get("menu_name") or e.get("recipe_id"),
                "Why": e.get("rejection_reason") or "",
                "False allergen claim": ", ".join(e.get("false_allergen_claim") or []),
            } for e in rejected_post]),
            width="stretch", hide_index=True,
        )
        st.caption(
            "These are menus the model proposed and the symbolic layer refused. "
            "In an arm without that layer they would have been shown to the user."
        )

    if false_claims:
        st.error(
            f"**{len(false_claims)} menu(s) claimed an allergen was absent that "
            f"the recipe record says is present.** This is the failure mode the "
            f"post-filter exists for: the model asserted a safety property it "
            f"had no basis for, and the check against the record caught it."
        )

    if corrected:
        st.markdown("**Citations repaired**")
        st.dataframe(
            pd.DataFrame([{
                "Menu": e.get("menu_name") or e.get("recipe_id"),
                "Corrected to": e.get("citation_returned") or "",
            } for e in corrected]),
            width="stretch", hide_index=True,
        )
        st.caption(
            "The model returned a source that did not match the recipe record. "
            "The post-filter replaced it with the real citation rather than "
            "letting fabricated provenance reach the page."
        )

    if not rejected_post and not corrected and not false_claims:
        st.success(
            "Every proposed menu survived verification, with accurate allergen "
            "claims and correct provenance. On this profile the model and the "
            "symbolic layer agreed."
        )

# -- Interpretation warnings --------------------------------------------------

profile_level = service.profile_warnings(state)
if profile_level:
    st.markdown("### Restrictions the system could not fully interpret")
    for warning in profile_level:
        st.warning(warning)
    st.caption(
        "These are the most important lines on the page. A restriction that "
        "falls outside the vocabulary is matched literally against ingredient "
        "wording, so it is only caught when the recipe happens to use the same "
        "word. Silence here would mean a weaker check with no sign of it."
    )

other = service.warnings_by_recipe(state)
if other:
    with st.expander(f"Per-recipe guardrail notes ({len(other)} recipes)"):
        index = service.recipe_index()
        for recipe_id, warnings in other.items():
            name = index.get(recipe_id, {}).get("name", recipe_id)
            st.markdown(f"**{name}**")
            for warning in warnings:
                components.status_line("warning", warning)
