"""
Tests for the rejected-credential latch — src/rate_limit.py and its callers.

What this guards against, concretely: on 2026-08-19 the GROQ_API_KEY in .env had
expired. Every one of the 90 generator calls returned 401 `expired_api_key`, and
the three LLM arms produced nothing at all. The judge then made 30 more calls to
the same endpoint with the same key, retrying each one three times — about 90
seconds of backoff spent re-confirming a fact established by the first call. The
run was nonetheless written out as `complete: true`, and a report was generated
from the single arm that never calls a model.

The property under test is that a refused credential stops the run immediately,
is never retried, and is not confused with a rate limit — a quota refills
overnight, a dead key does not.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "graphs"))
sys.path.insert(0, str(ROOT / "benchmark"))

from rate_limit import (  # noqa: E402
    AUTH_FAILED,
    CredentialState,
    is_auth_failure,
    is_daily_quota,
    retry_after_secs,
)

# Groq's verbatim reply to the expired key, as langchain_groq surfaced it.
EXPIRED = (
    "AuthenticationError: Error code: 401 - {'error': {'message': 'Invalid API Key', "
    "'type': 'invalid_request_error', 'code': 'expired_api_key'}}"
)
PER_DAY = (
    "RateLimitError: Error code: 429 - Rate limit reached on tokens per day (TPD): "
    "Limit 200000. Please try again in 12m59.328s."
)
PER_MINUTE = (
    "RateLimitError: Error code: 429 - Rate limit reached on tokens per minute "
    "(TPM). Please try again in 7.5s."
)


@pytest.fixture(autouse=True)
def _clean_latches():
    from rate_limit import GENERATOR_QUOTA, JUDGE_QUOTA

    for latch in (AUTH_FAILED, GENERATOR_QUOTA, JUDGE_QUOTA):
        latch.reset()
    yield
    for latch in (AUTH_FAILED, GENERATOR_QUOTA, JUDGE_QUOTA):
        latch.reset()


# ── classification ───────────────────────────────────────────────────────────

def test_the_real_401_is_recognised():
    assert is_auth_failure(EXPIRED)


def test_a_rate_limit_is_not_an_auth_failure():
    """The two both mean 'stop calling', but only one of them clears on its own."""
    assert not is_auth_failure(PER_DAY)
    assert not is_auth_failure(PER_MINUTE)
    assert not is_auth_failure("APIConnectionError: connection reset")
    assert not is_auth_failure("")


def test_an_auth_failure_is_not_mistaken_for_a_daily_quota():
    """
    The inverse direction matters just as much: a 401 carries no `try again in`
    hint, so it must not fall into the quota branch and tell the user to wait for
    a reset that will not fix anything.
    """
    assert retry_after_secs(EXPIRED) is None
    assert not is_daily_quota(EXPIRED)


def test_a_disabled_key_403_is_recognised():
    assert is_auth_failure("PermissionDeniedError: Error code: 403 - key disabled")


# ── the latch ────────────────────────────────────────────────────────────────

def test_the_latch_keeps_the_first_message():
    state = CredentialState()
    assert not state.hit

    state.record(EXPIRED)
    state.record("something else entirely")

    assert state.hit
    assert "expired_api_key" in state.detail, "the first message is the diagnosis"


def test_the_latch_is_shared_between_roles_not_split_by_role():
    """
    Unlike the daily budgets, both models authenticate with the same key, so
    there is deliberately one latch and it carries no `role`.
    """
    assert not hasattr(AUTH_FAILED, "role")


# ── the generation node ──────────────────────────────────────────────────────

def _llm_raising(message):
    llm = MagicMock()
    llm.invoke.side_effect = RuntimeError(message)
    return llm


def test_a_401_is_latched_and_never_retried(monkeypatch):
    from graphs import nodes

    monkeypatch.setattr(nodes.time, "sleep", lambda s: pytest.fail("must not sleep"))
    llm = _llm_raising(EXPIRED)

    raw, error = nodes._invoke_with_retry(llm, "prompt")

    assert error.startswith(nodes.AUTH_FAILED_PREFIX)
    assert AUTH_FAILED.hit
    assert llm.invoke.call_count == 1, "a refused key is not worth a second call"


def test_a_401_does_not_latch_the_quota():
    """Latching the quota instead would route the user to --resume after a wait."""
    from graphs import nodes
    from rate_limit import GENERATOR_QUOTA

    nodes._invoke_with_retry(_llm_raising(EXPIRED), "prompt")

    assert not GENERATOR_QUOTA.hit


def test_once_the_key_is_refused_no_further_call_is_made():
    from graphs import nodes

    AUTH_FAILED.record(EXPIRED)
    llm = MagicMock()

    raw, error = nodes._invoke_with_retry(llm, "prompt")

    assert error.startswith(nodes.AUTH_FAILED_PREFIX)
    llm.invoke.assert_not_called()


# ── the judge ────────────────────────────────────────────────────────────────

_FRESH_STATS = {"attempted": 0, "ok": 0, "parse_error": 0, "call_error": 0,
                "skipped_quota_exhausted": 0, "skipped_auth_failed": 0,
                "last_error": ""}


def test_the_judge_does_not_retry_a_refused_key(monkeypatch):
    """
    The regression this exists for: the judge's own loop slept 1s then 2s and
    called three times per menu, ~90s across the 30 menus of run 20260819_082917.
    """
    import evaluator

    evaluator.JUDGE_STATS.update(_FRESH_STATS)
    monkeypatch.setattr(evaluator.time, "sleep",
                        lambda s: pytest.fail("must not sleep"))
    llm = _llm_raising(EXPIRED)
    monkeypatch.setattr(evaluator, "_get_judge", lambda: llm)

    verdict = evaluator._judge_call("score this")

    assert "credentials rejected" in verdict["error"]
    assert llm.invoke.call_count == 1
    assert AUTH_FAILED.hit
    assert evaluator.JUDGE_STATS["call_error"] == 1
    assert "expired_api_key" in evaluator.JUDGE_STATS["last_error"], \
        "the console warning has to be able to name the cause"


def test_the_judge_skips_every_later_menu_without_calling(monkeypatch):
    import evaluator

    evaluator.JUDGE_STATS.update(_FRESH_STATS)
    AUTH_FAILED.record(EXPIRED)
    llm = MagicMock()
    monkeypatch.setattr(evaluator, "_get_judge", lambda: llm)

    verdict = evaluator._judge_call("score this")

    llm.invoke.assert_not_called()
    assert evaluator.JUDGE_STATS["skipped_auth_failed"] == 1
    assert evaluator.JUDGE_STATS["attempted"] == 0, \
        "a call never made is not an attempt — it must not shrink the denominator"
    assert "credentials rejected" in verdict["error"]


# ── the runner ───────────────────────────────────────────────────────────────

def test_run_single_case_skips_llm_arms_once_refused_but_keeps_no_llm():
    """no_llm needs no credential, and it is the deterministic safety floor."""
    import runner
    from benchmark_cases import BENCHMARK_CASES

    AUTH_FAILED.record(EXPIRED)
    no_llm_graph = MagicMock()
    no_llm_graph.invoke.return_value = {"final_menus": [], "latency_ms": {}}
    llm_graph = MagicMock()

    result = runner.run_single_case(BENCHMARK_CASES[0], no_llm_graph,
                                    llm_graph, llm_graph, llm_graph)

    no_llm_graph.invoke.assert_called_once()
    llm_graph.invoke.assert_not_called()
    for mode in ("neural_rag", "neurosymbolic", "no_rag"):
        assert result[mode]["error"].startswith(runner.AUTH_FAILED_PREFIX)


def test_resume_drops_rows_killed_by_a_refused_key(tmp_path):
    """
    Those case-runs carry no evidence. Keeping them would let a run resumed after
    the key was replaced 'finish' with the same holes the outage left in it.
    """
    import runner

    path = tmp_path / "run_partial.json"
    path.write_text(json.dumps({
        "metadata": {},
        "results": [
            {"case_id": "STD-01", "repeat": 0,
             "neural_rag": {"final_menus": [{"recipe_id": "recipe_001"}]},
             "neurosymbolic": {"final_menus": []}, "no_rag": {"final_menus": []}},
            {"case_id": "STD-02", "repeat": 0,
             "neural_rag": {"final_menus": [],
                            "generation_error": f"{runner.AUTH_FAILED_PREFIX}: 401"},
             "neurosymbolic": {"final_menus": []}, "no_rag": {"final_menus": []}},
        ],
    }), encoding="utf-8")

    assert [r["case_id"] for r in runner._load_completed(str(path))] == ["STD-01"]


# ── the completeness flag ────────────────────────────────────────────────────

def test_arm_health_counts_what_each_arm_actually_answered():
    import runner

    results = [
        {"no_llm": {"final_menus": [{"recipe_id": "recipe_001"}]},
         "neural_rag": {"final_menus": [], "generation_error": "LLM call failed: 401",
                        "generation_candidates": ["recipe_001"]},
         "neurosymbolic": {"final_menus": []},
         "no_rag": {"final_menus": []}},
    ]
    health = runner._arm_health(results)

    assert health["no_llm"]["answered"] == 1
    assert health["neural_rag"]["answered"] == 0
    assert health["neural_rag"]["failed"] == 1


def test_a_run_whose_llm_arms_all_died_is_not_marked_complete(monkeypatch, tmp_path):
    """
    Run 20260819_082917 was stamped `complete: true` with three of four arms
    empty. Iterating every case is not the same as producing data for it.
    """
    import llm_provider
    import runner
    from benchmark_cases import BENCHMARK_CASES
    from graphs import build_graphs

    monkeypatch.setattr(llm_provider, "get_llm",
                        lambda prefer=None: (MagicMock(), "stub"))
    monkeypatch.setattr(llm_provider, "configure_langsmith", lambda *a, **k: False)

    def make_ok(_llm=None):
        g = MagicMock()
        g.invoke.return_value = {"final_menus": [{"recipe_id": "recipe_001"}],
                                 "latency_ms": {}}
        return g

    def make_dead(_llm=None):
        # A non-auth failure, so the latch does not abort the run: this test is
        # about the completeness flag, not about the abort.
        g = MagicMock()
        g.invoke.return_value = {
            "final_menus": [],
            "generation_candidates": [{"id": "recipe_001"}],
            "generation_error": "LLM call failed: APIConnectionError: reset",
            "latency_ms": {},
        }
        return g

    monkeypatch.setattr(build_graphs, "build_no_llm_graph", make_ok)
    # Every arm that calls a generator dies; no_llm is the only one that can
    # still answer. Driven off runner.LLM_ARMS so a new generator arm joins the
    # dead set automatically rather than quietly reaching a real model.
    for mode in runner.LLM_ARMS:
        monkeypatch.setattr(build_graphs, build_graphs.BUILDER_NAMES[mode], make_dead)

    out = runner.run_benchmark(output_dir=str(tmp_path))
    payload = json.loads(Path(out).read_text(encoding="utf-8"))
    meta = payload["metadata"]

    assert len(payload["results"]) == len(BENCHMARK_CASES), \
        "every case was still iterated"
    assert meta["complete"] is False, "but three arms produced nothing"
    assert set(meta["dead_arms"]) == set(runner.LLM_ARMS)
    assert meta["arm_health"]["no_llm"]["answered"] == len(BENCHMARK_CASES)


def test_a_refused_key_stops_the_run_at_the_first_case(monkeypatch, tmp_path):
    import llm_provider
    import runner
    from graphs import build_graphs

    monkeypatch.setattr(llm_provider, "get_llm",
                        lambda prefer=None: (MagicMock(), "stub"))
    monkeypatch.setattr(llm_provider, "configure_langsmith", lambda *a, **k: False)

    def make(_llm=None):
        g = MagicMock()
        g.invoke.return_value = {"final_menus": [], "latency_ms": {}}
        return g

    # Driven off build_graphs.BUILDER_NAMES rather than a literal list, so a
    # new arm is stubbed here automatically instead of reaching a real model.
    for name in build_graphs.BUILDER_NAMES.values():
        monkeypatch.setattr(build_graphs, name, make)

    real_run_single = runner.run_single_case

    def run_then_refuse(case, *graphs):
        result = real_run_single(case, *graphs)
        AUTH_FAILED.record(EXPIRED)
        return result

    monkeypatch.setattr(runner, "run_single_case", run_then_refuse)

    out = runner.run_benchmark(output_dir=str(tmp_path))
    payload = json.loads(Path(out).read_text(encoding="utf-8"))

    assert len(payload["results"]) == 1, "no point running 29 more cases"
    assert payload["metadata"]["complete"] is False
    assert "refused the API key" in payload["metadata"]["aborted_reason"]
    assert payload["metadata"]["auth_failed"]["hit"] is True


# ── the preflight ────────────────────────────────────────────────────────────

def test_verify_credentials_fails_closed_on_a_401(monkeypatch):
    import groq
    import llm_provider

    monkeypatch.setenv("GROQ_API_KEY", "gsk_dead")

    def boom(*a, **k):
        raise RuntimeError(EXPIRED)

    monkeypatch.setattr(groq, "Groq", boom)

    ok, detail = llm_provider.verify_credentials()

    assert ok is False
    assert "expired_api_key" in detail


def test_verify_credentials_does_not_block_a_run_on_a_flaky_probe(monkeypatch):
    """
    A timeout says nothing about the key. Refusing to start on one would be a
    worse failure than the one the preflight prevents.
    """
    import groq
    import llm_provider

    monkeypatch.setenv("GROQ_API_KEY", "gsk_probably_fine")

    def boom(*a, **k):
        raise RuntimeError("APITimeoutError: request timed out")

    monkeypatch.setattr(groq, "Groq", boom)

    ok, detail = llm_provider.verify_credentials()

    assert ok is True
    assert "inconclusive" in detail


def test_verify_credentials_has_nothing_to_check_without_a_groq_key(monkeypatch):
    """An Ollama-only setup authenticates with nothing."""
    import llm_provider

    monkeypatch.setenv("GROQ_API_KEY", "")

    ok, detail = llm_provider.verify_credentials()

    assert ok is True
    assert "nothing to verify" in detail


def test_verify_credentials_accepts_a_working_key(monkeypatch):
    import groq
    import llm_provider

    monkeypatch.setenv("GROQ_API_KEY", "gsk_good")
    client = MagicMock()
    monkeypatch.setattr(groq, "Groq", lambda *a, **k: client)

    ok, detail = llm_provider.verify_credentials()

    assert ok is True
    assert detail == "Groq key accepted"
    client.models.list.assert_called_once()
