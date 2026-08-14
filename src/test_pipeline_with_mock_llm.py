"""
test_pipeline_with_mock_llm.py

Integration tests proving the guardrail safety net works correctly
using a mocked LLM, without needing a live API key or network access.

Now patches llm_provider.get_llm (the single resolution point used by
generation.py, evaluator.py, and eval_generation.py) instead of the
old provider-specific anthropic/groq patches.

Tests:
  1. Happy path: well-behaved LLM passes through correctly
  2. LLM proposes an unsafe recipe -> caught by post-filter
  3. LLM hallucinates a recipe_id -> caught as unverifiable
  4. Zero safe candidates -> LLM is never called
"""

from __future__ import annotations
import json
import os
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from guardrails import ChildProfile
from retrieval import run_retrieval
from post_filter import post_filter_menus
import generation
import llm_provider


def _make_mock_llm(json_payload: dict) -> Any:
    """Builds a LangChain-compatible mock LLM returning the given JSON payload."""
    from langchain_core.messages import AIMessage

    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(content=json.dumps(json_payload))
    return mock_llm


def _patch_llm(mock_llm: Any):
    """Context manager that patches llm_provider.get_llm to return mock_llm."""
    return patch.object(llm_provider, "get_llm", return_value=(mock_llm, "mock"))


def test_happy_path() -> None:
    print("=== TEST: happy path (well-behaved LLM) ===")
    profile = ChildProfile(age_years=9, allergies=["fish"], school_nut_free=True)
    retrieval_result = run_retrieval(profile)

    safe_ids = [r["id"] for r in retrieval_result["safe_candidates"]]
    assert safe_ids, "Expected at least one safe candidate"
    assert "recipe_005" not in safe_ids, "Salmon bagel must not be safe for fish allergy"
    assert "recipe_008" not in safe_ids, "Tuna salad must not be safe for fish allergy"

    payload = {
        "menu_options": [{
            "recipe_id": safe_ids[0],
            "menu_name": "Test menu",
            "why_it_fits": "Fish-free, suits the profile",
            "nutritional_rationale": "Within age-appropriate limits",
            "allergens_confirmed_absent": ["fish"],
        }]
    }

    mock_llm = _make_mock_llm(payload)
    with _patch_llm(mock_llm):
        gen_output = generation.generate_menus(
            profile,
            retrieval_result["safe_candidates"],
            retrieval_result["supporting_context"],
        )

    assert gen_output["menu_options"], "Expected parsed menu options"
    post_result = post_filter_menus(profile, gen_output)
    assert len(post_result["final_recommendations"]) == 1, "Well-behaved menu should pass"
    assert not post_result["rejected_at_post_filter"]
    print(f"  PASSED. Final: {post_result['final_recommendations'][0]['menu_name']}")


def test_llm_proposes_unsafe_recipe() -> None:
    print("\n=== TEST: LLM ignores instructions, proposes allergen-containing recipe ===")
    profile = ChildProfile(age_years=9, allergies=["fish"], school_nut_free=True)
    retrieval_result = run_retrieval(profile)

    # Salmon bagel contains fish — should be caught by post-filter
    payload = {
        "menu_options": [{
            "recipe_id": "recipe_005",
            "menu_name": "Salmon and salad bagel",
            "why_it_fits": "Great omega-3 source",
            "nutritional_rationale": "High protein",
            "allergens_confirmed_absent": ["fish"],  # LLM claims fish-free — wrong
        }]
    }

    mock_llm = _make_mock_llm(payload)
    with _patch_llm(mock_llm):
        gen_output = generation.generate_menus(
            profile,
            retrieval_result["safe_candidates"],
            retrieval_result["supporting_context"],
        )

    post_result = post_filter_menus(profile, gen_output)
    assert len(post_result["final_recommendations"]) == 0, \
        "CRITICAL: unsafe recipe reached final output!"
    assert len(post_result["rejected_at_post_filter"]) == 1
    rejected = post_result["rejected_at_post_filter"][0]
    assert "fish" in rejected["reasons"][0].lower()
    print(f"  PASSED. Post-filter correctly rejected: {rejected['recipe_name']}")
    print(f"  Reason: {rejected['reasons']}")


def test_llm_hallucinates_recipe_id() -> None:
    print("\n=== TEST: LLM invents a recipe_id that does not exist ===")
    profile = ChildProfile(age_years=10)
    retrieval_result = run_retrieval(profile)

    payload = {
        "menu_options": [{
            "recipe_id": "recipe_does_not_exist_42",
            "menu_name": "Imaginary lunch",
            "why_it_fits": "n/a",
            "nutritional_rationale": "n/a",
            "allergens_confirmed_absent": [],
        }]
    }

    mock_llm = _make_mock_llm(payload)
    with _patch_llm(mock_llm):
        gen_output = generation.generate_menus(
            profile,
            retrieval_result["safe_candidates"],
            retrieval_result["supporting_context"],
        )

    post_result = post_filter_menus(profile, gen_output)
    assert len(post_result["final_recommendations"]) == 0
    assert len(post_result["unverifiable"]) == 1
    print(f"  PASSED. Correctly flagged: {post_result['unverifiable'][0]['reason']}")


def test_zero_candidates_skips_llm() -> None:
    print("\n=== TEST: zero safe candidates — LLM must never be called ===")
    profile = ChildProfile(age_years=7, allergies=["fish"], intolerances=["gluten"])
    retrieval_result = run_retrieval(profile)
    assert not retrieval_result["safe_candidates"], \
        "This test requires zero safe candidates (fish + gluten excludes all 9 recipes)"

    mock_llm = MagicMock()
    with _patch_llm(mock_llm):
        gen_output = generation.generate_menus(
            profile,
            retrieval_result["safe_candidates"],
            retrieval_result["supporting_context"],
        )
    mock_llm.invoke.assert_not_called()
    assert gen_output == {"menu_options": []}
    print("  PASSED. generate_menus() short-circuited; LLM was never called.")


if __name__ == "__main__":
    test_happy_path()
    test_llm_proposes_unsafe_recipe()
    test_llm_hallucinates_recipe_id()
    test_zero_candidates_skips_llm()
    print("\nAll integration tests passed.")
