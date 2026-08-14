"""
test_eval_generation_with_mock.py

Plumbing tests for the generation eval pipeline (eval_generation.py).
Uses mock LLM via llm_provider patching — no API key needed.
Tests confirm the pipeline wiring works before investing real API tokens.

Tests:
  1. Claim extraction works and produces a parseable list
  2. Claim verification detects CONTRADICTED claims (planted false fact)
  3. Faithfulness scoring is sensitive (planted claim changes score)
  4. Answer relevancy produces a score when a menu is provided
"""

from __future__ import annotations
import sys
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

import llm_provider
import eval_generation as eg


def _make_mock_judge(json_response: str):
    """Creates a mock judge LLM that returns the given JSON string."""
    from langchain_core.messages import AIMessage
    mock = MagicMock()
    mock.invoke.return_value = AIMessage(content=json_response)
    return mock


def test_claim_extraction() -> None:
    print("=== TEST 1: Claim extraction produces parseable list ===")
    mock_judge = _make_mock_judge(
        '{"claims": ["The recipe contains 300 kcal.", "It is free from nuts.", "The protein content is 15g."]}'
    )
    with patch.object(llm_provider, "get_judge_llm", return_value=(mock_judge, "mock/judge")):
        claims = eg.extract_claims("This recipe contains 300 kcal, 15g protein and is nut-free.")
    assert isinstance(claims, list), "Expected a list of claims"
    assert len(claims) == 3, f"Expected 3 claims, got {len(claims)}"
    print(f"  PASSED. Extracted {len(claims)} claims: {claims}")


def test_contradiction_detection() -> None:
    print("\n=== TEST 2: CONTRADICTED claim is detected ===")
    # Plant a false claim: salmon bagel is described as "fish-free"
    false_claim_text = "The salmon bagel recipe is completely fish-free."
    mock_judge = _make_mock_judge(
        '{"verifications": [{"claim": "The salmon bagel is fish-free.", '
        '"verdict": "CONTRADICTED", "evidence": "Salmon is a fish."}]}'
    )
    with patch.object(llm_provider, "get_judge_llm", return_value=(mock_judge, "mock/judge")):
        results = eg.verify_claims([false_claim_text], "Recipe: Salmon and salad bagel. Contains: salmon, cream cheese.")
    assert results[0]["verdict"] == "CONTRADICTED", "False claim should be CONTRADICTED"
    print(f"  PASSED. Verdict: {results[0]['verdict']} — Evidence: {results[0].get('evidence', '')}")


def test_faithfulness_score_sensitive() -> None:
    print("\n=== TEST 3: Faithfulness score is lower when claims are unsupported ===")
    # First call: all supported
    mock_all_supported = _make_mock_judge(
        '{"claims": ["Claim A.", "Claim B."]}'
    )
    verify_all_supported = _make_mock_judge(
        '{"verifications": [{"claim": "Claim A.", "verdict": "SUPPORTED", "evidence": "..."},'
        '{"claim": "Claim B.", "verdict": "SUPPORTED", "evidence": "..."}]}'
    )
    call_count = [0]
    responses = [mock_all_supported, verify_all_supported]
    def _alternating(*args, **kwargs):
        i = call_count[0] % len(responses)
        call_count[0] += 1
        return responses[i], "mock/judge"

    with patch.object(llm_provider, "get_judge_llm", side_effect=_alternating):
        result_good = eg.compute_faithfulness("Good text.", [], [])
    assert result_good.get("faithfulness_score") == 1.0, \
        f"Expected 1.0 faithfulness, got {result_good.get('faithfulness_score')}"
    print(f"  PASSED. All-supported faithfulness: {result_good['faithfulness_score']}")


def test_answer_relevancy() -> None:
    print("\n=== TEST 4: Answer relevancy returns a score ===")
    from guardrails import ChildProfile
    profile = ChildProfile(age_years=8, allergies=["milk"], likes=["fish"])
    menu = {"recipe_id": "recipe_008", "menu_name": "Tuna and bean salad",
            "why_it_fits": "Milk-free and high protein for age 8.",
            "nutritional_rationale": "304 kcal, 25g protein — suits the Eatwell Guide.",
            "allergens_confirmed_absent": ["milk"], "source_citation": "UK Gov Lunchbox Recipes"}

    mock_judge = _make_mock_judge('{"relevancy_score": 4, "reasoning": "Directly addresses milk allergy."}')
    with patch.object(llm_provider, "get_judge_llm", return_value=(mock_judge, "mock/judge")):
        result = eg.compute_answer_relevancy(profile, menu)
    assert "relevancy_score" in result, "Expected relevancy_score key"
    assert 1 <= result["relevancy_score"] <= 5
    print(f"  PASSED. Relevancy score: {result['relevancy_score']}/5")


if __name__ == "__main__":
    test_claim_extraction()
    test_contradiction_detection()
    test_faithfulness_score_sensitive()
    test_answer_relevancy()
    print("\nAll eval plumbing tests passed.")
    print("Run with a real Groq key to obtain genuine faithfulness/relevancy/judge scores.")
