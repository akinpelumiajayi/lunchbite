"""
Tests for re-asking the generator when its response arrives whole but unreadable.

The concrete regression: in run 20260821_114840 the generator answered ADV-11 in
the `no_rag` arm with a menu object that had lost a key --

    {"recipe_id": "recipe_002", "Chicken and Hummus Veggie Wrap", "why_it_fits": ...}

a bare value sitting where `"menu_name":` belonged. The transport call had
succeeded, so `_invoke_with_retry` returned on the first attempt and `generate`
recorded a dead run. One case-run of 36 was dropped from every rate in the
report, and the report said so in a warning banner that could not be acted on.

The properties under test:
  * an unreadable response is re-asked, up to GENERATE_ATTEMPTS
  * the re-ask tells the model what was wrong with the previous one
  * a response that stays unreadable is still recorded as an error, never as
    the empty menu list that a safe refusal produces
  * a valid *refusal* is not retried -- it is an answer, not a failure
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "graphs"))
sys.path.insert(0, str(ROOT / "benchmark"))

from json_parsing import parse_menu_response  # noqa: E402
from rate_limit import AUTH_FAILED, GENERATOR_QUOTA  # noqa: E402

# The exact text the model returned for ADV-11, trimmed to the defect.
MALFORMED = (
    '{"menu_options": [{"recipe_id": "recipe_001", "menu_name": "Turkey Wrap", '
    '"why_it_fits": "ok", "nutritional_rationale": "ok", '
    '"allergens_confirmed_absent": ["peanut"], "source_citation": "USDA"}, '
    '{"recipe_id": "recipe_002", "Chicken and Hummus Veggie Wrap", '
    '"why_it_fits": "ok", "nutritional_rationale": "ok", '
    '"allergens_confirmed_absent": ["peanut"], "source_citation": "USDA"}]}'
)

WELL_FORMED = (
    '{"menu_options": [{"recipe_id": "recipe_001", "menu_name": "Turkey Wrap", '
    '"why_it_fits": "ok", "nutritional_rationale": "ok", '
    '"allergens_confirmed_absent": ["peanut"], "source_citation": "USDA"}]}'
)

REFUSAL = '{"menu_options": []}'


@pytest.fixture(autouse=True)
def _clear_latches():
    """Both latches are module-level singletons and survive between tests."""
    AUTH_FAILED.hit = False
    GENERATOR_QUOTA.hit = False
    yield
    AUTH_FAILED.hit = False
    GENERATOR_QUOTA.hit = False


def _llm_returning(*payloads):
    llm = MagicMock()
    llm.invoke.side_effect = [MagicMock(content=p) for p in payloads]
    return llm


def test_the_adv11_response_is_genuinely_unparseable():
    """Anchors the fixture: if this ever parses, the rest of the file is vacuous."""
    menus, err = parse_menu_response(MALFORMED)
    assert err is not None
    assert menus == []


def test_an_unreadable_response_is_re_asked():
    from graphs import nodes

    llm = _llm_returning(MALFORMED, WELL_FORMED)
    raw, error = nodes._invoke_with_retry(llm, "prompt", parse=parse_menu_response)

    assert error is None
    assert raw == WELL_FORMED
    assert llm.invoke.call_count == 2, "the first, broken answer must not end the run"


def test_the_re_ask_names_the_defect():
    """
    An identical re-ask at temperature 0.1 largely reproduces the same output,
    so the retry has to say what was wrong.
    """
    from graphs import nodes

    llm = _llm_returning(MALFORMED, WELL_FORMED)
    nodes._invoke_with_retry(llm, "PROMPT-BODY", parse=parse_menu_response)

    first, second = [c.args[0][0].content for c in llm.invoke.call_args_list]
    assert first == "PROMPT-BODY", "the first attempt is sent unchanged"
    assert second.startswith("PROMPT-BODY")
    assert "could not be parsed as JSON" in second


def test_a_persistently_unreadable_response_is_still_an_error():
    """
    The safety property the retry must not erode: a broken model has to remain
    distinguishable from a cautious one.
    """
    from graphs import nodes

    llm = _llm_returning(MALFORMED, MALFORMED, MALFORMED)
    raw, error = nodes._invoke_with_retry(llm, "prompt", parse=parse_menu_response)

    assert llm.invoke.call_count == nodes.GENERATE_ATTEMPTS
    assert error is not None and "Could not parse" in error
    assert raw == MALFORMED, "the last raw answer is kept as evidence"


def test_a_valid_refusal_is_not_retried():
    """`{"menu_options": []}` is a safe answer. Re-asking would badger the model."""
    from graphs import nodes

    llm = _llm_returning(REFUSAL, WELL_FORMED)
    raw, error = nodes._invoke_with_retry(llm, "prompt", parse=parse_menu_response)

    assert error is None
    assert raw == REFUSAL
    assert llm.invoke.call_count == 1


def test_parse_failures_are_not_slept_on(monkeypatch):
    """Nothing about a malformed decode is time-dependent."""
    from graphs import nodes

    monkeypatch.setattr(nodes.time, "sleep", lambda s: pytest.fail("must not sleep"))
    nodes._invoke_with_retry(_llm_returning(MALFORMED, WELL_FORMED), "p",
                             parse=parse_menu_response)


def test_without_a_parser_the_first_response_is_returned():
    """The rate-limit callers pass no parser and must keep their old behaviour."""
    from graphs import nodes

    llm = _llm_returning(MALFORMED)
    raw, error = nodes._invoke_with_retry(llm, "prompt")

    assert error is None
    assert raw == MALFORMED
    assert llm.invoke.call_count == 1


# ── the generate node itself ─────────────────────────────────────────────────
# The retry lives in `_invoke_with_retry`, but it is the node that decides what
# the run record says, and the node is what `run_failed` in stats.py reads.

_STATE = {
    "pipeline_mode": "no_rag",
    "generation_candidates": [],
    "profile": {"age_years": 7, "allergies": ["peanut"], "intolerances": [],
                "likes": ["wraps"], "dislikes": [], "school_nut_free": True,
                "cultural_context": None},
}


def test_the_node_recovers_a_case_that_used_to_be_lost():
    """ADV-11/no_rag end to end: broken answer, re-ask, scored run."""
    from graphs import nodes

    out = nodes.make_generate_node(_llm_returning(MALFORMED, WELL_FORMED))(dict(_STATE))

    assert out["generation_error"] is None
    assert [m["recipe_id"] for m in out["proposed_menus"]] == ["recipe_001"]


def test_the_node_still_reports_an_unrecoverable_parse_failure():
    from graphs import nodes
    from stats import run_failed

    out = nodes.make_generate_node(_llm_returning(*([MALFORMED] * 3)))(dict(_STATE))

    assert out["proposed_menus"] == []
    assert out["generation_error"] is not None
    assert out["llm_raw_output"] == MALFORMED, "evidence is kept for the run record"
    # The property the whole exclusion machinery rests on.
    assert run_failed(out, "no_rag") is True
