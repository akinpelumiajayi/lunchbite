"""
components.py -- Shared rendering. No module here calls the pipeline.

Chart conventions, applied everywhere in the app:

* One measure across named categories is a horizontal bar chart in a single
  hue, with a direct value label on every bar. Three hues in the palette fall
  below 3:1 contrast on this surface, so labels are relief, not decoration --
  magnitude is never left to colour alone.
* Every chart ships a table view beside it.
* Status colours (good / warning / critical) are reserved and always carry a
  word, never a colour on its own.
* Pipeline stages sort by pipeline position, not by magnitude. A funnel sorted
  largest-first would still look like a funnel while saying nothing true.
"""

from __future__ import annotations

import html
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import altair as alt
import pandas as pd
import streamlit as st

from lunchbite import theme

# Guardrail's own "near the ceiling" threshold, so the meter and the flag it is
# drawn beside can never disagree about what counts as close.
NEAR_CEILING = 0.85


def inject_css() -> None:
    st.markdown(theme.CSS, unsafe_allow_html=True)


# -- Small pieces -------------------------------------------------------------

def chips(items: Iterable[str], kind: str = "muted", limit: Optional[int] = None) -> str:
    """Pill list as an HTML fragment. `kind` in safe / danger / warn / muted."""
    values = [str(i) for i in items if str(i).strip()]
    if not values:
        return ""
    shown, rest = (values[:limit], values[limit:]) if limit else (values, [])
    out = "".join(
        f'<span class="lb-chip lb-chip-{kind}">{html.escape(v)}</span>' for v in shown
    )
    if rest:
        out += f'<span class="lb-chip lb-chip-muted">+{len(rest)} more</span>'
    return out


def label(text: str) -> str:
    return f'<div class="lb-label">{html.escape(text)}</div>'


def _fmt(value: Any, suffix: str = "") -> str:
    if value is None or value == "":
        return "-"
    if isinstance(value, float):
        return f"{value:g}{suffix}"
    return f"{value}{suffix}"


# -- Nutrient meters ----------------------------------------------------------

def _meter_status(value: float, ceiling: float) -> Tuple[str, str, str]:
    """(status key, colour, word) for a value against a per-lunch ceiling."""
    if value > ceiling:
        return "critical", theme.STATUS["critical"], "over the guideline"
    if value > ceiling * NEAR_CEILING:
        return "warning", theme.STATUS["warning"], "near the guideline"
    return "good", theme.STATUS["good"], "within the guideline"


def nutrient_meter(name: str, value: Optional[float], ceiling: Optional[float],
                   unit: str = "g") -> str:
    """
    One nutrient against its per-lunch ceiling, with the ceiling drawn as a tick.

    The ceiling is marked rather than implied by the bar filling the track, so a
    value over it can overflow visibly instead of silently pinning at 100%.
    """
    if value is None:
        return (f'<div class="lb-meter-row"><div class="lb-meter-head">'
                f'<span>{html.escape(name)}</span>'
                f'<span class="lb-note">not recorded</span></div></div>')

    if ceiling is None:
        # Outside the 4-18 band the corpus covers, the gate stops checking and
        # warns. "Not checked" is the honest label; a bar with no ceiling would
        # invite the reader to supply one.
        return (f'<div class="lb-meter-row"><div class="lb-meter-head">'
                f'<span>{html.escape(name)}</span>'
                f'<span><span class="lb-meter-val">{_fmt(value, unit)}</span>'
                f' &middot; <span class="lb-note">no guideline for this age</span>'
                f'</span></div></div>')

    _key, colour, word = _meter_status(value, ceiling)
    # The track runs to the ceiling or the value, whichever is larger, so a
    # breach is visible as overflow past the tick rather than a full bar.
    span = max(value, ceiling) * 1.15
    fill_pct = min(100.0, value / span * 100.0)
    tick_pct = min(100.0, ceiling / span * 100.0)

    return (
        f'<div class="lb-meter-row">'
        f'<div class="lb-meter-head">'
        f'<span>{html.escape(name)}</span>'
        f'<span><span class="lb-meter-val">{_fmt(value, unit)}</span>'
        f' <span class="lb-note">of {_fmt(ceiling, unit)} &middot; {word}</span></span>'
        f'</div>'
        f'<div class="lb-meter-track">'
        f'<div class="lb-meter-fill" style="width:{fill_pct:.1f}%;background:{colour};"></div>'
        f'<div class="lb-meter-tick" style="left:{tick_pct:.1f}%;"></div>'
        f'</div></div>'
    )


_KCAL_RANGE = re.compile(r"(\d+)\s*-\s*(\d+)")


def kcal_note(kcal: Optional[float], target: Optional[str]) -> str:
    """
    Energy against the guideline lunch band.

    Deliberately not a status meter. Sugar and salt have ceilings the guardrail
    enforces; energy has a target band, and a lunch under it is not a failure
    the way an allergen is. So this states both numbers and stops.
    """
    if kcal is None:
        return ""
    if not target:
        return f'<div class="lb-note">{_fmt(kcal)} kcal</div>'

    match = _KCAL_RANGE.search(str(target))
    if not match:
        return (f'<div class="lb-note">{_fmt(kcal)} kcal &middot; '
                f'guideline lunch {html.escape(str(target))}</div>')

    low, high = int(match.group(1)), int(match.group(2))
    if kcal < low:
        where = "below"
    elif kcal > high:
        where = "above"
    else:
        where = "inside"
    return (f'<div class="lb-note">{_fmt(kcal)} kcal &middot; '
            f'{where} the guideline lunch band of {low}-{high} kcal</div>')


# -- Menu card ----------------------------------------------------------------

def menu_card(menu: Dict[str, Any], recipe: Dict[str, Any],
              limits: Optional[Dict[str, Any]], index: int) -> None:
    """
    One recommendation.

    `menu` is a MenuOption -- six keys, none of them ingredients or nutrition.
    Everything concrete comes from `recipe`, the corpus record joined on
    recipe_id. `recipe` is {} when the id is not in the corpus, which the
    no_rag arm can produce and which the card says outright.
    """
    meta = theme.category_meta(recipe.get("meal_category"))
    name = menu.get("menu_name") or recipe.get("name") or menu.get("recipe_id", "?")
    category = recipe.get("meal_category") or "not in the recipe corpus"

    head = (
        f'<div class="lb-card"><div class="lb-card-head" '
        f'style="background:{meta["color"]};">'
        f'<div class="lb-card-glyph">{meta["glyph"]}</div>'
        f'<div><div class="lb-card-title">{index}. {html.escape(str(name))}</div>'
        f'<div class="lb-card-cat">{html.escape(str(category))}</div></div>'
        f'</div><div class="lb-card-body">'
    )

    body: List[str] = []

    if not recipe:
        body.append(
            '<div class="lb-chip lb-chip-danger">recipe id not found in the corpus</div>'
            '<div class="lb-note">This arm returned an id that does not exist in '
            'data/recipes.json, so none of the ingredients, nutrition or '
            'provenance below could be verified against a real record.</div>'
        )

    if menu.get("why_it_fits"):
        body.append(label("Why it fits"))
        body.append(f'<div>{html.escape(str(menu["why_it_fits"]))}</div>')

    if menu.get("nutritional_rationale"):
        body.append(label("Nutrition"))
        body.append(f'<div>{html.escape(str(menu["nutritional_rationale"]))}</div>')

    nutrition = recipe.get("nutrition_per_serving") or {}
    if nutrition:
        body.append(label("Against the guideline for this age"))
        body.append(kcal_note(nutrition.get("energy_kcal"),
                              (limits or {}).get("kcal_target")))
        body.append('<div style="margin-top:0.5rem;">')
        body.append(nutrient_meter("Sugars", nutrition.get("sugars_g"),
                                   (limits or {}).get("sugars_g")))
        body.append(nutrient_meter("Salt", nutrition.get("salt_g"),
                                   (limits or {}).get("salt_g")))
        body.append("</div>")

    confirmed = menu.get("allergens_confirmed_absent") or []
    if confirmed:
        body.append(label("Confirmed absent"))
        body.append(chips(sorted(confirmed), "safe", limit=14))

    present = recipe.get("allergens_present") or []
    if present:
        body.append(label("Contains"))
        body.append(chips(sorted(present), "danger"))
    elif recipe:
        body.append(label("Contains"))
        body.append('<span class="lb-chip lb-chip-safe">none of the 14 declarable allergens</span>')

    if recipe.get("allergen_notes"):
        body.append(f'<div class="lb-note" style="margin-top:0.4rem;">'
                    f'{html.escape(str(recipe["allergen_notes"]))}</div>')

    st.markdown(head + "".join(body) + "</div></div>", unsafe_allow_html=True)

    if recipe:
        _card_details(menu, recipe)


def _card_details(menu: Dict[str, Any], recipe: Dict[str, Any]) -> None:
    """The long tail of a recipe, folded away so the card stays scannable."""
    name = recipe.get("name") or menu.get("recipe_id", "recipe")
    with st.expander(f"Ingredients, method and source for {name}"):
        meta_bits = []
        if recipe.get("prep_time_mins") is not None:
            meta_bits.append(f"{recipe['prep_time_mins']} min prep")
        if recipe.get("cook_time_mins") is not None:
            meta_bits.append(f"{recipe['cook_time_mins']} min cook")
        if recipe.get("serves") is not None:
            meta_bits.append(f"serves {recipe['serves']}")
        # Present on 20 of 29 recipes.
        if recipe.get("lunch_cost_usd") is not None:
            meta_bits.append(f"about ${recipe['lunch_cost_usd']:.2f} per lunch")
        if meta_bits:
            st.caption(" &middot; ".join(meta_bits))

        if recipe.get("description"):
            st.write(recipe["description"])

        left, right = st.columns(2)
        with left:
            st.markdown("**Ingredients**")
            for item in recipe.get("ingredients") or []:
                st.markdown(f"- {item}")
        with right:
            if recipe.get("extras_suggested"):
                st.markdown("**Suggested extras**")
                for item in recipe["extras_suggested"]:
                    st.markdown(f"- {item}")
                st.caption(
                    "Extras are suggestions attached to the recipe, not part of "
                    "the nutrition figures above. The guardrail scans them "
                    "separately and warns if one carries a restricted allergen."
                )

        # method_steps is present on only 8 of the 29 recipes.
        steps = recipe.get("method_steps") or []
        if steps:
            st.markdown("**Method**")
            for i, step in enumerate(steps, 1):
                st.markdown(f"{i}. {step}")

        if recipe.get("diet_tags"):
            st.markdown("**Diet tags**")
            st.markdown(chips(sorted(recipe["diet_tags"]), "muted"),
                        unsafe_allow_html=True)

        citation = menu.get("source_citation") or recipe.get("citation") or recipe.get("source")
        if citation:
            st.markdown("**Source**")
            st.caption(str(citation))
            st.caption(
                "Verified against the recipe record by the post-filter, which "
                "replaces a citation the model invented with the real one."
            )
        if recipe.get("source_url"):
            st.markdown(f"[Open the original source]({recipe['source_url']})")


# -- Charts -------------------------------------------------------------------

def _table_view(frame: pd.DataFrame, caption: str = "Table view") -> None:
    with st.expander(caption):
        st.dataframe(frame, width="stretch", hide_index=True)


def bar_chart(frame: pd.DataFrame, category: str, value: str, title: str,
              colour: Optional[str] = None, height: Optional[int] = None,
              sort_by_value: bool = True, value_format: str = "d") -> None:
    """
    Horizontal bars, one hue, a direct label on every bar, plus a table view.

    `sort_by_value=False` keeps the frame's own order, which is what pipeline
    stages and node sequences need -- reordering those by magnitude would
    destroy the only thing they mean.
    """
    if frame.empty:
        st.caption(f"{title}: nothing to show.")
        return

    colour = colour or theme.SERIES[0]
    order = "-x" if sort_by_value else None
    rows = len(frame)
    height = height or max(120, 26 * rows + 30)

    base = alt.Chart(frame).encode(
        y=alt.Y(f"{category}:N", sort=order, title=None,
                axis=alt.Axis(labelColor=theme.INK_2, labelLimit=260,
                              domainColor=theme.BASELINE, tickSize=0)),
        x=alt.X(f"{value}:Q", title=None,
                axis=alt.Axis(grid=True, gridColor=theme.GRID, labelColor=theme.MUTED,
                              domain=False, tickSize=0),
                scale=alt.Scale(nice=True)),
        tooltip=list(frame.columns),
    )
    bars = base.mark_bar(color=colour, height=13, cornerRadiusEnd=4)
    labels = base.mark_text(align="left", dx=5, color=theme.INK_2, fontSize=11).encode(
        text=alt.Text(f"{value}:Q", format=value_format)
    )

    st.markdown(f"**{title}**")
    st.altair_chart(
        (bars + labels).properties(height=height, background=theme.SURFACE),
        width="stretch",
    )
    _table_view(frame)


def scatter(frame: pd.DataFrame, x: str, y: str, label: str, title: str,
            colour: Optional[str] = None, height: int = 340) -> None:
    """
    Two measures across many items, one hue, hover to identify.

    No size encoding. A third variable mapped to area invites the reader to
    compare areas, which people do badly, and Streamlit's built-in size legend
    starts its ramp at zero -- so a 190 kcal lunch and a 610 kcal one look
    nearly identical. Identity goes on the tooltip, where it is exact.
    """
    if frame.empty:
        st.caption(f"{title}: nothing to show.")
        return

    colour = colour or theme.SERIES[0]
    chart = (
        alt.Chart(frame)
        .mark_circle(size=110, color=colour, opacity=0.75,
                     stroke=theme.SURFACE, strokeWidth=2)
        .encode(
            x=alt.X(f"{x}:Q", title=x,
                    axis=alt.Axis(grid=True, gridColor=theme.GRID,
                                  labelColor=theme.MUTED, titleColor=theme.INK_2,
                                  domainColor=theme.BASELINE, tickSize=0),
                    scale=alt.Scale(nice=True, zero=False)),
            y=alt.Y(f"{y}:Q", title=y,
                    axis=alt.Axis(grid=True, gridColor=theme.GRID,
                                  labelColor=theme.MUTED, titleColor=theme.INK_2,
                                  domain=False, tickSize=0),
                    scale=alt.Scale(nice=True, zero=False)),
            tooltip=[alt.Tooltip(f"{label}:N", title="Recipe"),
                     alt.Tooltip(f"{x}:Q"), alt.Tooltip(f"{y}:Q")],
        )
        .properties(height=height, background=theme.SURFACE)
    )

    st.markdown(f"**{title}**")
    st.altair_chart(chart, width="stretch")
    _table_view(frame)


def funnel(stages: Sequence[Tuple[str, int]], title: str = "Where candidates went") -> None:
    """
    Stage-by-stage survivor counts, in pipeline order.

    Not sorted by magnitude: the stages are a sequence, and a funnel is only
    honest if the bars run in the order the pipeline actually ran them.
    """
    frame = pd.DataFrame(stages, columns=["Stage", "Recipes"])
    bar_chart(frame, "Stage", "Recipes", title, sort_by_value=False)


def status_line(kind: str, text: str) -> None:
    """A reserved status colour, always beside a word. Never colour alone."""
    colour = theme.STATUS.get(kind, theme.INK_2)
    st.markdown(
        f'<div style="display:flex;gap:0.5rem;align-items:baseline;margin:0.2rem 0;">'
        f'<span style="color:{colour};font-weight:700;">&#9679;</span>'
        f'<span>{html.escape(text)}</span></div>',
        unsafe_allow_html=True,
    )
