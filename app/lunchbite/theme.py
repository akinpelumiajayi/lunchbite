"""
theme.py -- One definition of every colour and mark style the dashboard uses.

The palette is lifted unchanged from the "Chart Theme" cell of
`notebooks/lunch_rag_pipeline.ipynb` so the dashboard and the dissertation
figures read as one system. It validates clean for colour-vision deficiency
(worst adjacent pair dE 9.1 protan) against this surface, with one caveat the
charts here honour: three hues fall below 3:1 contrast on the surface, so every
bar carries a direct value label and every chart offers a table view. Identity
and magnitude are never left to colour alone.

Categorical hues are applied in fixed slot order, never cycled.
"""

from __future__ import annotations

from typing import Dict

# Categorical hues, fixed order.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
          "#e87ba4", "#008300", "#4a3aa7", "#e34948"]

# Reserved status colours -- never used as a series colour.
STATUS: Dict[str, str] = {
    "good": "#0ca30c",
    "warning": "#fab219",
    "critical": "#d03b3b",
}

# Single-hue sequential ramp, light -> dark, for magnitude.
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
            "#256abf", "#184f95", "#0d366b"]

INK = "#0b0b0b"        # primary text
INK_2 = "#52514e"      # secondary text
MUTED = "#898781"      # axis labels
GRID = "#e1e0d9"       # hairline gridlines
BASELINE = "#c3c2b7"   # axis line
SURFACE = "#fcfcfb"    # chart surface
CARD = "#ffffff"
BORDER = "#e6e5de"

# ── Meal categories ──────────────────────────────────────────────────────────
#
# The nine `meal_category` values in data/recipes.json. The colour is a card
# accent, not a data encoding: the category name is always written out in text
# beside it, so nothing is carried by hue alone and the ninth value can take a
# neutral rather than forcing a generated ninth hue.
#
# The glyph is decoration standing in for a photograph. data/recipes.json has no
# image field for any of the 29 recipes, and showing a stock photo of a
# different dish would assert something about the recommendation that is not
# true -- so LunchBite shows a category mark and the real numbers instead.

CATEGORY_META: Dict[str, Dict[str, str]] = {
    "sandwich/wrap":         {"color": SERIES[0], "glyph": "\N{SANDWICH}"},
    "salad":                 {"color": SERIES[2], "glyph": "\N{GREEN SALAD}"},
    "salad/grain bowl":      {"color": SERIES[5], "glyph": "\N{SHALLOW PAN OF FOOD}"},
    "soup/hot":              {"color": SERIES[1], "glyph": "\N{POT OF FOOD}"},
    "snack-box/bento":       {"color": SERIES[6], "glyph": "\N{BENTO BOX}"},
    "dip/snack-style lunch": {"color": SERIES[4], "glyph": "\N{AMPHORA}"},
    "hot main / leftover":   {"color": SERIES[7], "glyph": "\N{CURRY AND RICE}"},
    "pizza/hot or cold":     {"color": SERIES[3], "glyph": "\N{SLICE OF PIZZA}"},
    "breakfast-for-lunch":   {"color": MUTED,     "glyph": "\N{COOKING}"},
}

_FALLBACK = {"color": MUTED, "glyph": "\N{FORK AND KNIFE WITH PLATE}"}


def category_meta(category: str | None) -> Dict[str, str]:
    """Accent colour and glyph for a meal_category, with a neutral fallback."""
    return CATEGORY_META.get((category or "").strip().lower(), _FALLBACK)


# ── Pipeline arms ────────────────────────────────────────────────────────────
#
# Mirrors the five modes documented in src/graphs/state.py. `needs_llm` decides
# whether the app may offer the arm with no API key configured.

ARMS: Dict[str, Dict[str, object]] = {
    "neurosymbolic": {
        "label": "Neurosymbolic",
        "needs_llm": True,
        "blurb": "Retrieval + symbolic pre-filter + LLM + symbolic post-filter. "
                 "The system this project argues for.",
    },
    "no_llm": {
        "label": "No LLM (rule-based)",
        "needs_llm": False,
        "blurb": "Retrieval + deterministic guardrail only. No API key needed, "
                 "no network call, instant.",
    },
    "neural_rag": {
        "label": "Neural RAG",
        "needs_llm": True,
        "blurb": "Retrieval + LLM with constraints stated in the prompt only. "
                 "No symbolic gate -- the comparison baseline.",
    },
    "reward_ranked": {
        "label": "Reward-ranked",
        "needs_llm": True,
        "blurb": "Neurosymbolic plus best-of-N reranking on the verifiable reward.",
    },
    "no_rag": {
        "label": "No RAG",
        "needs_llm": True,
        "blurb": "LLM with the profile alone, no retrieved recipes. "
                 "Reference control -- it can invent recipes that do not exist.",
    },
}

CSS = f"""
<style>
  .lb-card {{
    border: 1px solid {BORDER};
    border-radius: 10px;
    background: {CARD};
    overflow: hidden;
    margin-bottom: 1.1rem;
  }}
  .lb-card-head {{
    padding: 0.85rem 1.1rem;
    color: #fff;
    display: flex;
    align-items: center;
    gap: 0.7rem;
  }}
  .lb-card-glyph {{ font-size: 1.7rem; line-height: 1; }}
  .lb-card-title {{ font-size: 1.12rem; font-weight: 650; line-height: 1.25; }}
  .lb-card-cat {{ font-size: 0.78rem; opacity: 0.92; letter-spacing: 0.02em; }}
  .lb-card-body {{ padding: 0.9rem 1.1rem 1.1rem; }}

  .lb-chip {{
    display: inline-block;
    padding: 0.12rem 0.55rem;
    margin: 0.12rem 0.25rem 0.12rem 0;
    border-radius: 999px;
    font-size: 0.76rem;
    border: 1px solid transparent;
    white-space: nowrap;
  }}
  .lb-chip-safe {{ background: #e7f6e7; color: #10620f; border-color: #bfe4bf; }}
  .lb-chip-danger {{ background: #fdecec; color: #8c1f1f; border-color: #f3c4c4; }}
  .lb-chip-warn {{ background: #fdf3dc; color: #7a5300; border-color: #f2ddac; }}
  .lb-chip-muted {{ background: #f2f1ec; color: {INK_2}; border-color: {BORDER}; }}

  .lb-label {{
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: {MUTED};
    margin: 0.75rem 0 0.28rem;
    font-weight: 600;
  }}
  .lb-note {{ font-size: 0.82rem; color: {INK_2}; }}
  .lb-cite {{
    font-size: 0.76rem;
    color: {INK_2};
    border-top: 1px solid {BORDER};
    padding-top: 0.55rem;
    margin-top: 0.85rem;
  }}

  /* Nutrient meter: a bar against a per-lunch ceiling, with the ceiling drawn
     as a marker rather than implied by the bar running out of room. */
  .lb-meter-row {{ margin-bottom: 0.55rem; }}
  .lb-meter-head {{
    display: flex; justify-content: space-between;
    font-size: 0.78rem; color: {INK_2}; margin-bottom: 0.18rem;
  }}
  .lb-meter-val {{ font-variant-numeric: tabular-nums; color: {INK}; font-weight: 600; }}
  .lb-meter-track {{
    position: relative; height: 9px; border-radius: 4px;
    background: #efeee8; overflow: visible;
  }}
  .lb-meter-fill {{ height: 9px; border-radius: 4px; }}
  .lb-meter-tick {{
    position: absolute; top: -3px; width: 2px; height: 15px;
    background: {INK_2}; border-radius: 1px;
  }}
</style>
"""
