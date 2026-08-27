"""
LunchBite -- children's lunch recommendations, and the reasoning behind them.

Run with:  streamlit run app/LunchBite.py

This page owns the profile form and the recommendations. The pages in the
sidebar read the run this page stored; none of them re-invoke the pipeline, so
moving between them never spends another LLM call.

There are no photographs anywhere in LunchBite. data/recipes.json has no image
field for any of the 29 recipes, and a stock photo of a similar dish would
assert something about the recommendation that is not true. Each card carries a
category mark and the real figures instead.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lunchbite import bootstrap  # noqa: F401  (path setup must precede src imports)
from lunchbite import components, service, theme, vocab

from guardrails import ChildProfile, lunch_limits_for_age

st.set_page_config(page_title="LunchBite", page_icon="\N{BENTO BOX}",
                   layout="wide", initial_sidebar_state="expanded")
components.inject_css()


# -- Sidebar: the profile -----------------------------------------------------

# Widget defaults live in session_state, never in a `value=`/`default=`
# argument. Passing both makes Streamlit warn on every rerun, and the presets
# below work by writing these keys -- which only takes effect if the widget is
# reading from session_state in the first place.
_FORM_DEFAULTS: Dict[str, Any] = {
    "lb_age": 7,
    "lb_allergies": [],
    "lb_intolerances": [],
    "lb_other": "",
    "lb_diets": [],
    "lb_nut_free": False,
    "lb_likes": "",
    "lb_dislikes": "",
    "lb_culture": "",
    "lb_sugar_on": False,
    "lb_sugar": 15.0,
    "lb_salt_on": False,
    "lb_salt": 2.0,
}


def _seed_defaults() -> None:
    for key, value in _FORM_DEFAULTS.items():
        st.session_state.setdefault(key, value)


def _apply_preset() -> None:
    """Writes a preset into the widget keys; the rerun redraws the form."""
    name = st.session_state.get("lb_preset", "-")
    if name == "-":
        return
    preset = vocab.PRESETS[name]
    st.session_state["lb_age"] = preset.get("age_years", 7)
    st.session_state["lb_allergies"] = list(preset.get("allergies", []))
    st.session_state["lb_intolerances"] = list(preset.get("intolerances", []))
    st.session_state["lb_diets"] = list(preset.get("diet_requirements", []))
    st.session_state["lb_likes"] = ", ".join(preset.get("likes", []))
    st.session_state["lb_dislikes"] = ", ".join(preset.get("dislikes", []))
    st.session_state["lb_nut_free"] = bool(preset.get("school_nut_free", False))
    st.session_state["lb_culture"] = preset.get("cultural_context", "")
    st.session_state["lb_other"] = ""
    # An override is a ceiling set for this child, so it is a hard rejection
    # rather than the advisory band breach. A preset carrying one has to switch
    # the checkbox on too, or the number would sit there doing nothing.
    sugar = preset.get("max_sugar_g_override")
    st.session_state["lb_sugar_on"] = sugar is not None
    st.session_state["lb_sugar"] = float(sugar) if sugar is not None else 15.0
    salt = preset.get("max_salt_g_override")
    st.session_state["lb_salt_on"] = salt is not None
    st.session_state["lb_salt"] = float(salt) if salt is not None else 2.0


def build_profile() -> ChildProfile:
    """The sidebar form. Every option comes from the guardrail vocabularies."""
    _seed_defaults()
    st.sidebar.markdown("### The child")

    st.sidebar.selectbox(
        "Start from an example", ["-"] + list(vocab.PRESETS), key="lb_preset",
        on_change=_apply_preset,
        help="Fills the form below. Everything stays editable.",
    )

    age = st.sidebar.slider("Age", vocab.AGE_MIN, vocab.AGE_MAX, key="lb_age")

    allergy_options = vocab.ALLERGEN_OPTIONS + vocab.ALLERGY_ALIAS_OPTIONS
    allergies = st.sidebar.multiselect(
        "Allergies", allergy_options, format_func=vocab.allergen_label,
        key="lb_allergies",
        help="The 14 declarable allergens, plus the everyday words the "
             "guardrail maps onto them.",
    )
    intolerances = st.sidebar.multiselect(
        "Intolerances", allergy_options, format_func=vocab.allergen_label,
        key="lb_intolerances",
    )
    other_allergies = st.sidebar.text_input(
        "Anything else to avoid", key="lb_other",
        placeholder="comma-separated",
        help="Free text is allowed, but a term outside the 14-allergen "
             "vocabulary is only matched literally against ingredient wording. "
             "LunchBite says so before you run.",
    )

    diets = st.sidebar.multiselect(
        "Diet", vocab.DIET_OPTIONS, format_func=vocab.diet_label, key="lb_diets",
    )
    nut_free = st.sidebar.toggle("School is nut-free", key="lb_nut_free")

    st.sidebar.markdown("### Preferences")
    likes = st.sidebar.text_input("Likes", key="lb_likes", placeholder="pasta, cheese")
    dislikes = st.sidebar.text_input("Dislikes", key="lb_dislikes",
                                     placeholder="mushrooms")
    culture = st.sidebar.text_input(
        "Cultural context", key="lb_culture",
        placeholder="e.g. British primary school",
        help="Read by the guardrail as well as the prompt -- a diet named here "
             "is enforced even if you did not also pick it above.",
    )

    with st.sidebar.expander("Override the nutrition ceilings"):
        st.caption(
            "By default the per-lunch sugar and salt ceilings come from the "
            "age band and a breach is advisory. A ceiling you set here is an "
            "instruction about this child, so it becomes a hard rejection."
        )
        sugar_on = st.checkbox("Set a sugar ceiling", key="lb_sugar_on")
        sugar = st.number_input("Max sugars (g per lunch)", 0.0, 100.0, step=0.5,
                                key="lb_sugar", disabled=not sugar_on)
        salt_on = st.checkbox("Set a salt ceiling", key="lb_salt_on")
        salt = st.number_input("Max salt (g per lunch)", 0.0, 10.0, step=0.1,
                               key="lb_salt", disabled=not salt_on)

    return ChildProfile(
        age_years=age,
        allergies=allergies + vocab.split_free_text(other_allergies),
        intolerances=intolerances,
        likes=vocab.split_free_text(likes),
        dislikes=vocab.split_free_text(dislikes),
        school_nut_free=nut_free,
        cultural_context=culture.strip(),
        diet_requirements=diets,
        max_sugar_g_override=float(sugar) if sugar_on else None,
        max_salt_g_override=float(salt) if salt_on else None,
    )


def choose_arm(llm_ready: bool) -> str:
    """Arm selector. Arms needing an LLM are disabled when none is configured."""
    st.sidebar.markdown("### Pipeline")
    options = [a for a in theme.ARMS if llm_ready or not theme.ARMS[a]["needs_llm"]]
    # Seeded rather than passed as `index=`, for the same reason as the form
    # fields. The stored choice is also re-validated: a key that disappears
    # between reruns would otherwise leave an unavailable arm selected.
    if st.session_state.get("lb_arm") not in options:
        st.session_state["lb_arm"] = "neurosymbolic" if llm_ready else "no_llm"

    arm = st.sidebar.radio(
        "Which system should answer", options,
        format_func=lambda a: str(theme.ARMS[a]["label"]), key="lb_arm",
    )
    st.sidebar.caption(str(theme.ARMS[arm]["blurb"]))

    if not llm_ready:
        st.sidebar.caption(
            "The other four arms need a language model and are hidden until "
            "one is configured. See **System health**."
        )
    return arm


# -- Enforcement preview ------------------------------------------------------

def enforcement_preview(profile: ChildProfile) -> None:
    """
    What the guardrail will enforce, shown before the run rather than after.

    An unrecognised term still gets enforced -- literally, against ingredient
    wording -- but only when the recipe happens to use the same word. That is a
    materially weaker check, and finding out afterwards costs an LLM call.
    """
    preview = vocab.preview_enforcement(
        profile.allergies + profile.intolerances,
        profile.diet_requirements,
        profile.cultural_context,
    )
    restricted = sorted(profile.all_restricted_allergens() - set(preview["unknown_allergens"]))

    if not any((restricted, preview["diets"], preview["unknown_allergens"],
                preview["unknown_diets"])):
        st.caption(
            "No restrictions set, so all 29 recipes are eligible and the "
            "guardrail will reject nothing. Add an allergy or a diet in the "
            "sidebar to see it work."
        )
        return

    if restricted:
        st.markdown(components.label("Will be excluded"), unsafe_allow_html=True)
        st.markdown(components.chips(restricted, "danger"), unsafe_allow_html=True)
        if profile.school_nut_free:
            st.caption("Nuts and peanuts are included because the school is nut-free.")

    if preview["diets"]:
        st.markdown(components.label("Diet enforced"), unsafe_allow_html=True)
        st.markdown(components.chips(preview["diets"], "muted"), unsafe_allow_html=True)
        if preview["diets_from_context"]:
            st.caption(
                f"**{', '.join(preview['diets_from_context'])}** was picked up "
                f"from the cultural context, not the diet field. A requirement "
                f"already stated does not need restating to be enforced."
            )

    if preview["uncertifiable"]:
        st.warning(
            f"**{', '.join(preview['uncertifiable'])}** is checked as an ingredient "
            f"exclusion only. Certification, slaughter method and preparation "
            f"separation are not in the corpus and are not verified."
        )

    if preview["unknown_allergens"]:
        st.warning(
            f"**{', '.join(preview['unknown_allergens'])}** is not in the "
            f"14-allergen vocabulary. It will only be matched literally against "
            f"ingredient wording, so a recipe that spells it differently will "
            f"not be caught. Check those results by hand."
        )
    if preview["unknown_diets"]:
        st.warning(
            f"**{', '.join(preview['unknown_diets'])}** is not in the diet "
            f"vocabulary and will not be enforced at all."
        )

    # Not a warning. The cultural context field is prose about the setting; it
    # is scanned in case it names a diet, and usually it does not. The guardrail
    # reports that as an unrecognised diet requirement, which reads as a
    # restriction going unchecked when nothing was being restricted.
    if preview["context_unmatched"]:
        st.caption(
            f"No diet requirement was recognised in \"{preview['context_unmatched']}\", "
            f"so nothing is enforced from it. It still reaches the model as "
            f"context. Use the **Diet** field for a requirement that must be "
            f"enforced."
        )


# -- Running ------------------------------------------------------------------

def execute(profile: ChildProfile, arm: str) -> None:
    """Runs the arm with live per-node progress, and stores the run."""
    progress = st.progress(0.0)
    status = st.empty()
    # Nodes actually visited, not the full sequence: the arms differ in length
    # and the refine loop can revisit retrieval, so this is an estimate that
    # never claims more precision than it has.
    expected = max(4, len([n for n in service.NODE_SEQUENCE]) // 2)

    def on_node(node: str, seen: int) -> None:
        label = service.NODE_LABELS.get(node, node.replace("_", " "))
        status.caption(f"{label}…")
        progress.progress(min(0.95, seen / expected))

    try:
        payload = service.run(profile, arm, on_node=on_node)
    except service.PipelineUnavailable as exc:
        progress.empty()
        status.empty()
        st.error(f"**The pipeline could not start.** {exc}")
        if exc.remedy:
            st.info(exc.remedy)
        return
    except Exception as exc:  # noqa: BLE001 - the UI must survive any node failure
        progress.empty()
        status.empty()
        st.error(f"**The run failed.** {type(exc).__name__}: {exc}")
        st.caption("System health in the sidebar checks the index and the provider.")
        return

    progress.progress(1.0)
    progress.empty()
    status.empty()
    service.store_run(payload)


# -- Rendering ----------------------------------------------------------------

def render_error(result: Dict[str, Any]) -> None:
    """Generation failures, told apart so the response to each is different."""
    message = result.get("generation_error")
    if not message:
        return
    kind = service.fatal_error_kind(message)
    if kind == "quota":
        st.error(f"**Daily quota exhausted.** {message}")
        st.info("The key is valid; the allowance is spent. Wait for the reset, "
                "or use the **No LLM (rule-based)** arm, which calls nothing.")
    elif kind == "auth":
        st.error(f"**The provider rejected the credentials.** {message}")
        st.info("The key is missing, expired or revoked. Replace GROQ_API_KEY "
                "in .env and restart the app -- waiting will not fix this one.")
    else:
        st.error(f"**Generation failed.** {message}")


def render_recommendations(payload: Dict[str, Any]) -> None:
    result: Dict[str, Any] = payload["result"]
    state: Dict[str, Any] = payload["state"]
    profile: ChildProfile = result["profile"]
    menus: List[Dict[str, Any]] = result.get("final_recommendations") or []

    limits = lunch_limits_for_age(profile.age_years)

    top = st.columns(4)
    top[0].metric("Lunches found", len(menus))
    top[1].metric("Recipes checked", len(state.get("symbolic_pre_filter_log") or []))
    top[2].metric("Passed the guardrail", len(state.get("generation_candidates") or []))
    top[3].metric("Time", f"{payload['wall_ms'] / 1000:.1f}s")

    if state.get("refine_count"):
        st.caption(
            f"The first search came up short, so LunchBite widened it "
            f"{state['refine_count']} more time(s)."
        )

    render_error(result)

    # No page-level warning loop here on purpose. The guardrail raises its
    # profile-level warnings once per candidate, and the "What will be enforced"
    # panel already states the same facts from the profile itself -- and states
    # them more accurately, since it distinguishes an unrecognised *diet
    # requirement* from cultural-context prose that simply named no diet. The
    # unfiltered log is on the safety report, where the raw wording belongs.

    if not menus:
        st.markdown("### No lunch met every constraint")
        if result.get("explanation"):
            st.info(result["explanation"])
        rejected = result.get("rejected_at_retrieval") or []
        if rejected:
            st.caption(
                f"{len(rejected)} recipe(s) were rejected before the model was "
                f"asked. **Safety report** in the sidebar names each one and why."
            )
        st.caption(
            "The corpus holds 29 recipes. A tight combination of restrictions "
            "can genuinely exhaust it -- that is a real answer, not a failure, "
            "and it is why the system abstains rather than improvising."
        )
        return

    st.markdown("### Recommended lunches")
    if limits is None:
        st.caption(
            f"Age {profile.age_years} is outside the 4-18 range the nutrition "
            f"guidelines cover, so sugar and salt were not checked against a band."
        )

    per_recipe = service.warnings_by_recipe(state)
    for i, menu in enumerate(menus, 1):
        components.menu_card(menu, service.recipe_for(menu), limits, i)
        # Warnings the guardrail raised against this recipe specifically. A
        # recommended lunch whose suggested side item carries the allergen is
        # still actionable advice, and it appears nowhere else on this page.
        for warning in per_recipe.get(menu.get("recipe_id", ""), []):
            components.status_line("warning", warning)

    if payload["arm"] not in ("neurosymbolic", "reward_ranked", "no_llm"):
        st.warning(
            f"**{theme.ARMS[payload['arm']]['label']}** has no symbolic "
            f"verification. Nothing above was checked against the recipe "
            f"records after the model wrote it -- including the allergen claims. "
            f"It is here for comparison, not for feeding a child."
        )


# -- Page ---------------------------------------------------------------------

st.title("LunchBite")
st.caption(
    "Lunch recommendations for children, filtered by a deterministic allergen "
    "and nutrition guardrail before and after the model is asked."
)

for message in service.shadow_warnings():
    st.warning(message)

# First call opens ChromaDB and loads the embedding model, which takes a good
# fifteen seconds cold and renders nothing while it happens -- and on a fresh
# deployment it also builds the index and downloads the model, which is a
# minute or two. The spinner is the difference between "starting up" and
# "broken"; the result is cached, so every later rerun passes straight through.
with st.spinner("Opening the recipe index and loading the embedding model. "
                "On a fresh deployment this also builds the index and "
                "downloads the models -- a minute or two, once per server."):
    index = service.index_status()

if not index["ok"]:
    # `service.ensure_index` already tried to build it, so reaching here means
    # the build itself failed -- most often no network for the model download.
    # The local remedy is still worth printing; deployed, there is no shell to
    # print it to, so the error text is what has to carry the diagnosis.
    st.error("**The recipe index is unavailable**, so retrieval cannot run.")
    st.caption("Building it needs network access to download the embedding "
               "model (~90 MB, cached afterwards). Locally you can build it "
               "ahead of time:")
    st.code("python src/setup_database.py", language="bash")
    st.caption(index["error"])
    st.stop()

provider = service.provider_status()
if not provider["available"]:
    st.info(
        "**No language model is configured.** The rule-based arm still works "
        "and needs neither a key nor a network -- it is selected in the sidebar. "
        "**System health** explains what is missing."
    )

profile = build_profile()
arm = choose_arm(bool(provider["available"]))

left, right = st.columns([2, 1])
with right:
    st.markdown("#### What will be enforced")
    enforcement_preview(profile)

with left:
    if st.button("Find lunches", type="primary", width="stretch"):
        execute(profile, arm)

payload = service.last_run()
if payload is None:
    with left:
        st.markdown(
            "Build a profile in the sidebar, or pick one of the examples, then "
            "press **Find lunches**."
        )
        st.caption(
            "Retrieval is hybrid -- BM25 and embeddings fused, then reranked by "
            "a cross-encoder. The guardrail runs twice: once to decide what the "
            "model may see, and once to check what it claimed."
        )
else:
    render_recommendations(payload)
    st.divider()
    st.caption(
        f"Answered by **{theme.ARMS[payload['arm']]['label']}**. "
        f"**Safety report** shows what was rejected and why; **pipeline trace** "
        f"shows the retrieval stages and timings."
    )
