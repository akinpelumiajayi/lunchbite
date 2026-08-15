"""
Plumbing tests for the generation eval pipeline (eval/eval_generation.py).

Ported from eval/test_eval_generation_with_mock.py (bare asserts, run as a
script) so it runs under the same pytest suite as everything else.

The judge is mocked, so these confirm the wiring — claim extraction, claim
verification, score arithmetic — before any real API tokens are spent on it.

Run:  pytest tests/test_eval_generation.py -v
"""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "src", "graphs"))
sys.path.insert(0, os.path.join(ROOT, "eval"))

import llm_provider  # noqa: E402
import eval_generation as eg  # noqa: E402


def mock_judge(json_response: str):
    """A judge LLM returning a fixed JSON string.

    `bind` returns the same mock: _judge_call binds a per-call max_tokens, and a
    bare MagicMock would otherwise hand back a fresh child mock whose invoke()
    returns a MagicMock rather than a message.
    """
    from langchain_core.messages import AIMessage
    m = MagicMock()
    m.invoke.return_value = AIMessage(content=json_response)
    m.bind.return_value = m
    return m


@pytest.fixture(autouse=True)
def clear_judge_cache():
    """The judge client is cached module-level; a mock leaking from one test
    into the next would make results order-dependent."""
    eg._judge_llm_cache.clear()
    yield
    eg._judge_llm_cache.clear()


def with_judge(m):
    return patch.object(llm_provider, "get_judge_llm", return_value=(m, "mock/judge"))


# ── claim extraction ─────────────────────────────────────────────────────────

def test_claim_extraction_returns_a_list():
    m = mock_judge('{"claims": ["The recipe contains 300 kcal.", '
                   '"It is free from nuts.", "The protein content is 15g."]}')
    with with_judge(m):
        claims = eg.extract_claims("300 kcal, 15g protein, nut-free.")
    assert isinstance(claims, list)
    assert len(claims) == 3


def test_unparseable_judge_response_yields_no_claims():
    """A malformed response must not become a phantom claim that then scores."""
    with with_judge(mock_judge("I'm afraid I can't do that.")):
        assert eg.extract_claims("anything") == []


# ── claim verification ───────────────────────────────────────────────────────

def test_contradiction_is_detected():
    m = mock_judge('{"verifications": [{"claim": "The salmon bagel is fish-free.", '
                   '"verdict": "CONTRADICTED", "evidence": "Salmon is a fish."}]}')
    with with_judge(m):
        results = eg.verify_claims(
            ["The salmon bagel recipe is completely fish-free."],
            "Recipe: Salmon and salad bagel. Contains: salmon, cream cheese.")
    assert results[0]["verdict"] == "CONTRADICTED"


# ── faithfulness arithmetic ──────────────────────────────────────────────────

def scripted_judge(*json_responses: str):
    """One judge client returning each payload in turn.

    compute_faithfulness makes two judge calls -- extract, then verify -- but
    only one client: _judge_call caches it. Scripting invoke (rather than
    get_judge_llm) is what matches the real call pattern.
    """
    from langchain_core.messages import AIMessage
    m = MagicMock()
    m.invoke.side_effect = [AIMessage(content=r) for r in json_responses]
    m.bind.return_value = m
    return m


def test_all_supported_claims_score_one():
    m = scripted_judge(
        '{"claims": ["Claim A.", "Claim B."]}',
        '{"verifications": [{"claim": "Claim A.", "verdict": "SUPPORTED", "evidence": "."},'
        '{"claim": "Claim B.", "verdict": "SUPPORTED", "evidence": "."}]}')
    with with_judge(m):
        result = eg.compute_faithfulness("Good text.", [], [])
    assert result["faithfulness_score"] == 1.0
    assert result["n_claims"] == 2


def test_unsupported_claims_lower_the_score():
    """The metric has to be sensitive to fabrication, or it measures nothing."""
    m = scripted_judge(
        '{"claims": ["Claim A.", "Claim B."]}',
        '{"verifications": [{"claim": "Claim A.", "verdict": "SUPPORTED", "evidence": "."},'
        '{"claim": "Claim B.", "verdict": "CONTRADICTED", "evidence": "."}]}')
    with with_judge(m):
        result = eg.compute_faithfulness("Half-invented text.", [], [])
    assert result["faithfulness_score"] < 1.0


def test_no_claims_reports_none_not_a_perfect_score():
    """Zero extracted claims means nothing was checked. Scoring that 1.0 would
    reward output so vague it makes no checkable assertion at all."""
    m = scripted_judge('{"claims": []}', '{"verifications": []}')
    with with_judge(m):
        result = eg.compute_faithfulness("Lunch is nice.", [], [])
    assert result["faithfulness_score"] is None
    assert result["n_claims"] == 0


def test_the_judge_client_is_built_once_not_per_call():
    """Four calls per profile used to construct four HTTP clients."""
    m = scripted_judge('{"a": 1}', '{"b": 2}', '{"c": 3}')
    with with_judge(m) as patched:
        eg._judge_call("one")
        eg._judge_call("two")
        eg._judge_call("three")
    assert patched.call_count == 1


# ── relevancy ────────────────────────────────────────────────────────────────

def test_answer_relevancy_returns_a_score_in_range():
    from guardrails import ChildProfile
    profile = ChildProfile(age_years=8, allergies=["milk"], likes=["fish"])
    menu = {"recipe_id": "recipe_008", "menu_name": "Tuna and bean salad",
            "why_it_fits": "Milk-free and high protein for age 8.",
            "nutritional_rationale": "304 kcal, 25g protein.",
            "allergens_confirmed_absent": ["milk"],
            "source_citation": "UK Gov Lunchbox Recipes"}
    m = mock_judge('{"relevancy_score": 4, "reasoning": "Addresses the milk allergy."}')
    with with_judge(m):
        result = eg.compute_answer_relevancy(profile, menu)
    assert 1 <= result["relevancy_score"] <= 5


# ── per-call token budget ────────────────────────────────────────────────────

def test_max_tokens_is_actually_applied():
    """It was accepted and discarded for four call sites passing four different
    values; a regression here is silent and only shows up as truncation."""
    m = mock_judge('{"ok": true}')
    with with_judge(m):
        eg._judge_call("prompt", max_tokens=777)
    m.bind.assert_called_once_with(max_tokens=777)
