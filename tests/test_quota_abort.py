"""
Tests for the daily-budget latch — src/rate_limit.py and its two callers.

What this is guarding against, concretely: on 2026-08-18 the generator spent its
200,000 tokens/day at case ADV-04 of 30. The runner carried on, issuing 39 more
calls that could not succeed, and the resulting report was computed from 17 of 30
cases for `neural_rag`, 18 for `neurosymbolic` and 17 for `no_rag` — three
different case lists, presented as a paired comparison.

The property under test is that a spent daily budget stops the run, and a
momentary per-minute limit does not.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "graphs"))
sys.path.insert(0, str(ROOT / "benchmark"))

from rate_limit import (  # noqa: E402
    QuotaState,
    is_daily_quota,
    retry_after_secs,
)

# Groq's own wording, both windows.
PER_DAY = (
    "RateLimitError: Error code: 429 - {'error': {'message': 'Rate limit reached "
    "for model `qwen/qwen3.6-27b` in organization `org_01ks` service tier "
    "`on_demand` on tokens per day (TPD): Limit 200000, Used 199351, Requested "
    "2453. Please try again in 12m59.328s."
)
PER_MINUTE = (
    "RateLimitError: Error code: 429 - {'error': {'message': 'Rate limit reached "
    "for model `qwen/qwen3.6-27b` on tokens per minute (TPM): Limit 6000. "
    "Please try again in 7.5s."
)


# ── parsing ──────────────────────────────────────────────────────────────────

def test_retry_after_reads_minutes_and_seconds():
    assert retry_after_secs(PER_DAY) == pytest.approx(12 * 60 + 59.328)
    assert retry_after_secs(PER_MINUTE) == pytest.approx(7.5)


def test_retry_after_is_none_for_a_non_rate_limit_error():
    """Callers use this as the rate-limit test, so a timeout must not look like one."""
    assert retry_after_secs("APIConnectionError: connection reset") is None
    assert retry_after_secs("") is None


def test_daily_and_per_minute_are_told_apart():
    assert is_daily_quota(PER_DAY) is True
    assert is_daily_quota(PER_MINUTE) is False


def test_a_long_delay_counts_as_daily_even_without_the_window_named():
    """The named window is the primary signal; an unsittable delay is the backstop."""
    assert is_daily_quota("Please try again in 45m0s") is True


def test_wait_threshold_is_configurable(monkeypatch):
    monkeypatch.setenv("LLM_MAX_RATE_LIMIT_WAIT", "5")
    assert is_daily_quota(PER_MINUTE) is True


# ── the latch ────────────────────────────────────────────────────────────────

def test_latch_keeps_the_first_message_and_survives_later_calls():
    q = QuotaState("generator")
    assert not q.hit

    q.record(PER_DAY)
    assert q.hit and "tokens per day" in q.detail
    first = q.detail

    q.record("something else entirely")
    assert q.detail == first, "the latch records the cause, not the last symptom"

    q.reset()
    assert not q.hit and q.detail == ""


def test_human_wait_is_readable():
    q = QuotaState("generator")
    q.record(PER_DAY)
    assert q.human_wait() == "13m"


# ── the generation node ──────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _clean_latch():
    from rate_limit import GENERATOR_QUOTA
    GENERATOR_QUOTA.reset()
    yield
    GENERATOR_QUOTA.reset()


def _llm_raising(*errors):
    """An llm whose invoke raises the given errors in order, then succeeds."""
    from langchain_core.messages import AIMessage

    llm = MagicMock()
    replies = list(errors)

    def invoke(_messages):
        if replies:
            raise RuntimeError(replies.pop(0))
        return AIMessage(content=json.dumps({"menu_options": []}))

    llm.invoke.side_effect = invoke
    return llm


def test_a_per_minute_limit_is_slept_off_and_retried(monkeypatch):
    from graphs import nodes

    slept = []
    monkeypatch.setattr(nodes.time, "sleep", lambda s: slept.append(s))

    raw, error = nodes._invoke_with_retry(_llm_raising(PER_MINUTE), "prompt")

    assert error is None, "a limit that clears in 7.5s must not fail the case"
    assert slept and slept[0] == pytest.approx(8.0), "waits the interval Groq named"


def test_a_daily_limit_latches_and_is_not_retried(monkeypatch):
    from graphs import nodes
    from rate_limit import GENERATOR_QUOTA

    monkeypatch.setattr(nodes.time, "sleep", lambda s: pytest.fail("must not sleep"))
    llm = _llm_raising(PER_DAY)

    raw, error = nodes._invoke_with_retry(llm, "prompt")

    assert error.startswith(nodes.QUOTA_EXHAUSTED_PREFIX)
    assert GENERATOR_QUOTA.hit
    assert llm.invoke.call_count == 1, "a spent daily budget is not worth a second call"


def test_once_latched_no_further_call_is_made():
    from graphs import nodes
    from rate_limit import GENERATOR_QUOTA

    GENERATOR_QUOTA.record(PER_DAY)
    llm = MagicMock()

    raw, error = nodes._invoke_with_retry(llm, "prompt")

    assert error.startswith(nodes.QUOTA_EXHAUSTED_PREFIX)
    llm.invoke.assert_not_called()


def test_a_non_rate_limit_error_is_not_mistaken_for_a_quota():
    from graphs import nodes
    from rate_limit import GENERATOR_QUOTA

    raw, error = nodes._invoke_with_retry(_llm_raising("APIConnectionError: reset"), "p")

    assert error.startswith("LLM call failed")
    assert not GENERATOR_QUOTA.hit, "a dropped connection is not a spent budget"


# ── the runner ───────────────────────────────────────────────────────────────

def test_run_single_case_skips_llm_arms_once_latched_but_keeps_no_llm():
    """
    no_llm never calls the model, so it is the one arm that still yields data on
    a spent budget — and it is the deterministic safety floor the report leans on.
    """
    import runner
    from benchmark_cases import BENCHMARK_CASES
    from rate_limit import GENERATOR_QUOTA

    GENERATOR_QUOTA.record(PER_DAY)

    no_llm_graph = MagicMock()
    no_llm_graph.invoke.return_value = {"final_menus": [], "latency_ms": {}}
    llm_graph = MagicMock()

    result = runner.run_single_case(BENCHMARK_CASES[0], no_llm_graph,
                                    llm_graph, llm_graph, llm_graph)

    no_llm_graph.invoke.assert_called_once()
    llm_graph.invoke.assert_not_called()
    for mode in ("neural_rag", "neurosymbolic", "no_rag"):
        assert result[mode]["error"].startswith(runner.QUOTA_EXHAUSTED_PREFIX)


def test_resume_drops_the_rows_that_failed_on_a_spent_budget(tmp_path):
    """
    A resumed run must re-run the cases the budget killed. Keeping those rows
    would carry the outage into the new results file — the resume would 'finish'
    the run while leaving the same 13 holes in it.
    """
    import runner

    path = tmp_path / "run_partial.json"
    path.write_text(json.dumps({
        "metadata": {"aborted_reason": "budget"},
        "results": [
            {"case_id": "STD-01", "repeat": 0,
             "neural_rag": {"final_menus": [{"recipe_id": "recipe_001"}]},
             "neurosymbolic": {"final_menus": []}, "no_rag": {"final_menus": []}},
            {"case_id": "ADV-04", "repeat": 0,
             "neural_rag": {"error": f"{runner.QUOTA_EXHAUSTED_PREFIX}: TPD"},
             "neurosymbolic": {"final_menus": []}, "no_rag": {"final_menus": []}},
        ],
    }), encoding="utf-8")

    kept = runner._load_completed(str(path))

    assert [r["case_id"] for r in kept] == ["STD-01"]


def test_resume_also_drops_a_row_whose_failure_was_recorded_as_generation_error(tmp_path):
    """The graph records the outage on `generation_error`; only an exception
    escaping the graph lands on `error`. Both have to be recognised."""
    import runner

    path = tmp_path / "run_partial.json"
    path.write_text(json.dumps({
        "metadata": {},
        "results": [
            {"case_id": "ADV-05", "repeat": 0,
             "neural_rag": {"final_menus": [],
                            "generation_error": f"{runner.QUOTA_EXHAUSTED_PREFIX}: TPD"},
             "neurosymbolic": {"final_menus": []}, "no_rag": {"final_menus": []}},
        ],
    }), encoding="utf-8")

    assert runner._load_completed(str(path)) == []


# ── the resume loop ──────────────────────────────────────────────────────────

def _stub_graphs(monkeypatch, calls):
    """Swap the real graph builders for stubs that just record which case ran."""
    import llm_provider
    from graphs import build_graphs

    monkeypatch.setattr(llm_provider, "get_llm", lambda prefer=None: (MagicMock(), "stub"))
    monkeypatch.setattr(llm_provider, "configure_langsmith", lambda *a, **k: False)

    def make(_llm=None):
        g = MagicMock()
        g.invoke.side_effect = lambda state: (
            calls.append((state["profile"]["age_years"], state["pipeline_mode"]))
            or {"final_menus": [], "latency_ms": {}}
        )
        return g

    # Driven off build_graphs.BUILDER_NAMES rather than a literal list, so a
    # new arm is stubbed here automatically instead of reaching a real model.
    for name in build_graphs.BUILDER_NAMES.values():
        monkeypatch.setattr(build_graphs, name, make)


def test_resume_runs_only_the_cases_that_are_missing(monkeypatch, tmp_path):
    """
    The point of --resume: after the daily budget refills, pay only for what is
    left. Re-running the finished cases would spend the same budget again and
    could not finish either.
    """
    import runner
    from benchmark_cases import BENCHMARK_CASES

    already = [
        {"case_id": c.case_id, "repeat": 0,
         "neural_rag": {"final_menus": []}, "neurosymbolic": {"final_menus": []},
         "no_rag": {"final_menus": []}}
        for c in BENCHMARK_CASES[:5]
    ]
    partial = tmp_path / "run_partial.json"
    partial.write_text(json.dumps({"metadata": {}, "results": already}), encoding="utf-8")

    calls: list = []
    _stub_graphs(monkeypatch, calls)

    out = runner.run_benchmark(output_dir=str(tmp_path),
                               resume_from=str(partial))

    payload = json.loads(Path(out).read_text(encoding="utf-8"))
    ran = {r["case_id"] for r in payload["results"]}

    assert ran == {c.case_id for c in BENCHMARK_CASES}, "every case is present"
    assert len(payload["results"]) == len(BENCHMARK_CASES), "and none is duplicated"
    # 4 pipelines x the 25 cases that were missing
    # One graph invocation per arm per case. Counted off runner.ALL_ARMS so
    # the assertion tracks the arm list instead of a literal that goes stale
    # the moment a pipeline is added.
    assert len(calls) == len(runner.ALL_ARMS) * (len(BENCHMARK_CASES) - 5)
    assert payload["metadata"]["complete"] is True
    assert payload["metadata"]["resumed_from"] == str(partial)


def test_a_completed_run_is_marked_complete(monkeypatch, tmp_path):
    import runner
    from benchmark_cases import BENCHMARK_CASES

    calls: list = []
    _stub_graphs(monkeypatch, calls)

    out = runner.run_benchmark(output_dir=str(tmp_path))
    payload = json.loads(Path(out).read_text(encoding="utf-8"))

    assert payload["metadata"]["complete"] is True
    assert "aborted_reason" not in payload["metadata"]
    assert len(payload["results"]) == len(BENCHMARK_CASES)


def test_the_run_stops_and_is_marked_incomplete_when_the_budget_goes(monkeypatch, tmp_path):
    """
    The behaviour the 2026-08-18 report needed and did not have: stop, say so in
    the file, and do not record 39 more calls that could never succeed.
    """
    import runner
    from benchmark_cases import BENCHMARK_CASES
    from rate_limit import GENERATOR_QUOTA

    calls: list = []
    _stub_graphs(monkeypatch, calls)

    real_run_single = runner.run_single_case

    def run_then_die(case, *graphs):
        result = real_run_single(case, *graphs)
        if case.case_id == BENCHMARK_CASES[2].case_id:
            GENERATOR_QUOTA.record("on tokens per day (TPD). try again in 12m0s")
        return result

    monkeypatch.setattr(runner, "run_single_case", run_then_die)

    out = runner.run_benchmark(output_dir=str(tmp_path))
    payload = json.loads(Path(out).read_text(encoding="utf-8"))

    assert len(payload["results"]) == 3, "stops at the case that spent the budget"
    assert payload["metadata"]["complete"] is False
    assert "budget exhausted" in payload["metadata"]["aborted_reason"]
    assert payload["metadata"]["quota"]["hit"] is True
