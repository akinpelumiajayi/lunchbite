"""
Iterative retrieve-and-refine (objective 1(ii)).

Objective 1(ii) specifies a Pic2Plate-style iterative retrieval-and-refine loop.
The pipelines were strictly single-pass: `build_graphs.py` used `add_edge`
exclusively, there was not one `add_conditional_edges` in the file, and
`symbolic_postfilter` dropped unsafe menus without ever asking for more. A pass
that came up short simply returned a short answer.

The properties that matter here are not "a loop exists" but what it is allowed to
relax on the way round. Widening the search must never widen what counts as safe.

Run:  pytest tests/test_refine_loop.py -v
"""

from __future__ import annotations

import json
import os
import sys
from unittest.mock import MagicMock

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "src", "graphs"))

from nodes import (  # noqa: E402
    should_refine, widen_query, _refine_query,
    REFINE_MAX_PASSES, REFINE_TARGET_MENUS, RETRIEVE_TOP_K,
)

PROFILE = {
    "age_years": 8, "allergies": ["milk"], "intolerances": ["egg"],
    "likes": ["pasta"], "dislikes": ["hummus"], "school_nut_free": True,
    "cultural_context": "", "diet_requirements": ["vegetarian"],
}


def _state(**over):
    base = dict(
        profile=PROFILE, pipeline_mode="neural_rag", run_id="t", query="",
        bm25_candidates=[], semantic_candidates=[], fused_candidates=[],
        reranked_candidates=[], symbolic_pre_filter_log=[], generation_candidates=[],
        llm_raw_output="", proposed_menus=[], generation_error=None,
        symbolic_post_filter_log=[], final_menus=[], error=None, latency_ms={},
        refine_count=0, retrieve_top_k=0, refine_log=[],
    )
    base.update(over)
    return base


# ── When the loop runs ────────────────────────────────────────────────────────

def test_a_short_pass_with_a_thin_pool_is_retried():
    """Retrieval was the bottleneck: fewer candidates than the target."""
    assert should_refine(_state(
        final_menus=[{"recipe_id": "recipe_013"}],
        generation_candidates=[{"id": "recipe_013"}],
    )) == "widen_query"


def test_a_terse_model_with_an_ample_pool_is_not_retried():
    """
    The model was handed plenty and returned one anyway. Re-retrieving re-runs
    BM25, dense search, fusion and the cross-encoder to offer options it already
    declined — pure cost. Gating on menu count alone sent every mock case to the
    pass cap and roughly tripled the run.
    """
    ample = [{"id": f"recipe_{i:03d}"} for i in range(1, 10)]
    assert should_refine(_state(
        final_menus=[{"recipe_id": "recipe_013"}],
        generation_candidates=ample,
    )) == "done"


def test_nothing_usable_is_always_retried():
    """Even with an ample pool: the gates rejected all of it, so widen."""
    ample = [{"id": f"recipe_{i:03d}"} for i in range(1, 10)]
    assert should_refine(_state(final_menus=[], generation_candidates=ample)) == "widen_query"


def test_a_full_pass_is_not_retried():
    full = [{"recipe_id": f"recipe_{i:03d}"} for i in range(1, REFINE_TARGET_MENUS + 1)]
    assert should_refine(_state(final_menus=full)) == "done"


def test_the_pass_cap_holds():
    assert should_refine(_state(refine_count=REFINE_MAX_PASSES)) == "done"


def test_a_generation_error_is_not_retried():
    """
    Re-retrieving cannot fix a dead key, an exhausted quota or an unparseable
    response, and retrying multiplies the spend on a call that will fail again.
    """
    assert should_refine(_state(generation_error="401 invalid api key")) == "done"
    assert should_refine(_state(error="quota exhausted")) == "done"


def test_exhausted_corpus_stops_the_loop():
    """Widening past the corpus size cannot surface anything new."""
    assert should_refine(_state(retrieve_top_k=10_000)) == "done"


# ── What the loop may and may not relax ───────────────────────────────────────

@pytest.mark.parametrize("n", [0, 1, 2, 3])
def test_hard_constraints_survive_every_pass(n):
    """
    The whole risk of a widening loop: relaxing the query until something is
    found, including the allergy that made the search hard.
    """
    q = _refine_query(PROFILE, n).lower()
    assert "milk" in q, "allergy dropped from the query"
    assert "egg" in q, "intolerance dropped from the query"
    assert "nut-free school" in q, "school policy dropped from the query"
    assert "vegetarian" in q, "diet requirement dropped from the query"


def test_preferences_are_shed_as_passes_rise():
    """
    Preferences are soft: an unloved safe lunch beats no lunch.

    Pass 0 is built by `build_query`, not by `_refine_query`, so the shedding
    starts at pass 1 — dislikes go first, likes survive one more pass.
    """
    from nodes import build_query

    pass0 = build_query({"profile": PROFILE, "latency_ms": {}})["query"]
    assert "hummus" in pass0 and "pasta" in pass0

    assert "hummus" not in _refine_query(PROFILE, 1)   # dislikes shed first
    assert "pasta" in _refine_query(PROFILE, 1)        # likes survive one pass
    assert "pasta" not in _refine_query(PROFILE, 2)    # then shed too


def test_an_empty_candidate_set_is_refined_not_abandoned():
    """
    The pre-filter rejecting everything is a retrieval shortfall, not a
    generation failure — and it is the case the loop most needs to handle, since
    it is how the neuro-symbolic arm runs out of options on a small corpus.
    """
    from nodes import NO_CANDIDATES_ERROR

    assert should_refine(_state(generation_error=NO_CANDIDATES_ERROR)) == "widen_query"


def test_each_pass_widens_the_window():
    s = _state(final_menus=[])
    first = widen_query(s)
    assert first["retrieve_top_k"] > RETRIEVE_TOP_K
    second = widen_query(_state(**{**s, **first}))
    assert second["retrieve_top_k"] > first["retrieve_top_k"]
    assert second["refine_count"] == 2


def test_refine_log_records_why():
    out = widen_query(_state(final_menus=[]))
    assert out["refine_log"], "a loop that fires without saying so is unauditable"
    entry = out["refine_log"][0]
    assert entry["pass"] == 1
    assert str(REFINE_TARGET_MENUS) in entry["reason"]


# ── End to end, through the compiled graphs ───────────────────────────────────

def _llm_returning(n_menus):
    """n_menus=0 leaves nothing usable, which is what now drives the loop."""
    payload = {"menu_options": [
        {"recipe_id": "recipe_013", "menu_name": f"Option {i}", "why_it_fits": "y",
         "nutritional_rationale": "z", "allergens_confirmed_absent": [],
         "source_citation": "s"}
        for i in range(n_menus)
    ]}
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content=json.dumps(payload))
    return llm


@pytest.mark.parametrize("builder_name", ["build_neural_rag_graph", "build_neurosymbolic_graph"])
def test_both_llm_arms_carry_the_loop(builder_name):
    """
    If only one arm looped, neural_rag and neurosymbolic would differ by the
    gates AND by the loop, and the objective-4 comparison would stop isolating
    the symbolic layer.
    """
    import build_graphs
    graph = getattr(build_graphs, builder_name)(_llm_returning(0))
    mode = "neural_rag" if "neural" in builder_name else "neurosymbolic"

    out = graph.invoke(_state(pipeline_mode=mode))
    assert out["refine_count"] == REFINE_MAX_PASSES, "loop did not run to its cap"
    assert len(out["refine_log"]) == REFINE_MAX_PASSES


def test_a_satisfied_first_pass_does_not_loop():
    import build_graphs
    graph = build_graphs.build_neural_rag_graph(_llm_returning(REFINE_TARGET_MENUS))
    out = graph.invoke(_state(pipeline_mode="neural_rag"))
    assert out["refine_count"] == 0, "looped despite the first pass being sufficient"
