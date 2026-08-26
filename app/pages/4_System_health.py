"""
System health -- is this thing configured, and what is it configured as?

Answers the two questions that come up when a run fails: is a model reachable,
and is the index built. Then it lists the gate settings, because the same
profile produces different results under different gate modes and a reader
comparing two runs needs to know which was in force.

No key value is ever printed here. Presence and provider name only.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lunchbite import bootstrap  # noqa: F401
from lunchbite import components, service

from guardrails import DIET_GATE_MODES, diet_gate, lunch_fraction, nutrition_gate

st.set_page_config(page_title="System health - LunchBite",
                   page_icon="\N{HIGH VOLTAGE SIGN}", layout="wide")
components.inject_css()

st.title("System health")

if st.button("Re-check"):
    service.provider_status.clear()
    service.index_status.clear()
    st.rerun()

for message in service.shadow_warnings():
    st.warning(message)

# -- Retrieval ----------------------------------------------------------------

st.markdown("### Recipe index")
with st.spinner("Checking the vector store."):
    index = service.index_status()
if index["ok"]:
    detail = index["detail"]
    components.status_line("good", "The vector store is built and reachable.")
    st.dataframe(
        pd.DataFrame([{
            "Collection": detail.get("collection"),
            "Chunks": detail.get("chunks"),
            "Embedding model": detail.get("embedding_model"),
            "Dimensions": detail.get("dimension"),
            "Distance": detail.get("space"),
        }]),
        width="stretch", hide_index=True,
    )
    st.caption(
        "Chunks cover recipes, nutrition guidelines and allergen rules. Each "
        "recipe chunk carries an explicit \"free from …\" sentence so absence "
        "has some lexical signal, which bag-of-words retrieval otherwise has no "
        "way to represent."
    )
else:
    components.status_line("critical", "The vector store is not usable.")
    st.code("python src/setup_database.py", language="bash")
    st.caption(index["error"])

# -- Model --------------------------------------------------------------------

st.markdown("### Language model")
provider = service.provider_status()
if provider["available"]:
    components.status_line("good", f"Reachable via {provider['provider']}.")
    st.caption(
        f"Generator model `{provider['model']}`. The judge used by the benchmark "
        f"is deliberately a different model family, so it is not grading its own "
        f"output."
    )
else:
    components.status_line("warning", "No language model is configured.")
    if provider["groq_key_present"]:
        st.caption(
            "GROQ_API_KEY is set but the provider did not accept it. An expired "
            "or revoked key looks exactly like this; waiting will not fix it."
        )
    else:
        st.caption(
            "Neither GROQ_API_KEY nor a reachable Ollama server was found. Set "
            "one in .env, or use the rule-based arm, which needs neither."
        )
    if provider["error"]:
        with st.expander("Provider error"):
            st.code(provider["error"])
    st.info(
        "**The rule-based arm still works.** It runs retrieval and the "
        "deterministic guardrail with no model at all -- select **No LLM "
        "(rule-based)** on the main page."
    )

# -- Gates --------------------------------------------------------------------

st.markdown("### Gate configuration")
st.caption(
    "These change what a run does, so two results are only comparable when "
    "these matched. They are read from .env at import time."
)

gate = nutrition_gate()
d_gate = diet_gate()

st.dataframe(
    pd.DataFrame([
        {"Setting": "NUTRITION_GATE", "Value": gate,
         "Effect": {
             "advisory": "Sugar/salt breaches warn but do not reject.",
             "hard": "A breach rejects the recipe outright.",
             "off": "Band ceilings are not checked at all.",
         }.get(gate, "unrecognised value")},
        {"Setting": "DIET_GATE", "Value": d_gate,
         "Effect": {
             "hard": "A diet miss rejects the recipe.",
             "advisory": "A diet miss warns but does not reject.",
             "off": "Diet requirements are not enforced.",
         }.get(d_gate, "unrecognised value")},
        {"Setting": "LUNCH_NUTRITION_FRACTION", "Value": f"{lunch_fraction():.2f}",
         "Effect": "Share of the daily maximum allotted to one lunch."},
        {"Setting": "USE_CROSS_ENCODER_RERANKER",
         "Value": os.environ.get("USE_CROSS_ENCODER_RERANKER", "true"),
         "Effect": "Reranking lifts NDCG@5 from 0.622 to 0.727 on the eval set."},
        {"Setting": "CROSS_ENCODER_MODEL",
         "Value": os.environ.get("CROSS_ENCODER_MODEL",
                                 "cross-encoder/ms-marco-MiniLM-L-6-v2"),
         "Effect": "Reorders the fused candidates before the guardrail."},
    ]),
    width="stretch", hide_index=True,
)

if gate == "advisory":
    st.caption(
        "The nutrition gate is advisory by design. The corpus records total "
        "sugars while the guideline specifies free sugars, and hard-gating on "
        "that mismatch rejected 21 of 29 recipes at ages 7-10 on sugar alone. "
        "Allergens are unaffected -- those are always a hard rejection."
    )
elif gate == "hard":
    st.warning(
        "The nutrition gate is **hard**: a sugar or salt breach now rejects the "
        "recipe. Expect far fewer recommendations, and note that the comparison "
        "is against total rather than free sugars."
    )
if d_gate not in DIET_GATE_MODES:
    st.error(f"DIET_GATE is set to an unrecognised value: {d_gate!r}.")

# -- Where things are ---------------------------------------------------------

st.markdown("### Paths")
st.dataframe(
    pd.DataFrame([
        {"What": "Repository", "Where": str(bootstrap.ROOT)},
        {"What": "Recipe corpus", "Where": str(bootstrap.DATA / "recipes.json")},
        {"What": "Vector store", "Where": str(bootstrap.ROOT / "vectordb")},
        {"What": ".env",
         "Where": str(bootstrap.ROOT / ".env")
                  + ("" if (bootstrap.ROOT / ".env").exists() else "  (missing)")},
    ]),
    width="stretch", hide_index=True,
)

st.caption(
    "LunchBite shows no photographs. data/recipes.json carries no image field "
    "for any recipe, and a stock photo of a similar dish would assert something "
    "about the recommendation that is not true."
)
