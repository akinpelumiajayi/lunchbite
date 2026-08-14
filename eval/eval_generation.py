"""
eval_generation.py -- Evaluates generation quality using LLM-as-judge.

Uses llm_provider.get_llm() for both generation and judging, which
reads from .env automatically and supports Groq or Ollama.

Metrics:
  Faithfulness  -- are factual claims grounded in the retrieved context?
  Answer Relevancy -- does the menu suit this specific child's profile?
  LLM-as-Judge holistic rubric -- correctness, clarity, nutritional
                                   soundness, tone appropriateness
"""

from __future__ import annotations
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from guardrails import ChildProfile
from retrieval import run_retrieval
from generation import generate_menus


# ── Judge call helper ─────────────────────────────────────────────────────────

def _judge_call(prompt: str, max_tokens: int = 1500) -> Dict[str, Any]:
    """
    Calls the configured LLM provider as a judge.
    Reads provider config from .env via llm_provider.
    """
    from llm_provider import get_judge_llm
    from langchain_core.messages import HumanMessage

    llm, _ = get_judge_llm()  # Judge uses a different model than generator
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        raw = response.content.strip()
        if raw.startswith("```"):
            raw = raw.strip("`")
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"error": raw}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


# ── Faithfulness ──────────────────────────────────────────────────────────────

def extract_claims(generated_text: str) -> List[str]:
    prompt = (
        "Break the following text into short, atomic factual claims "
        "(one discrete fact per item). Only extract checkable factual claims "
        "(nutrition numbers, allergen presence/absence, ingredient facts, "
        "guideline references). Skip purely subjective phrases.\n\n"
        f"TEXT:\n{generated_text}\n\n"
        'Respond ONLY with valid JSON: {"claims": ["claim 1", "claim 2", ...]}'
    )
    result = _judge_call(prompt, max_tokens=1000)
    return result.get("claims", [])


def verify_claims(claims: List[str], context_text: str) -> List[Dict[str, Any]]:
    if not claims:
        return []
    prompt = (
        "For each claim below, decide if it is SUPPORTED, CONTRADICTED, or "
        "NOT_MENTIONED by the source context. Use ONLY the source context.\n\n"
        f"SOURCE CONTEXT:\n{context_text}\n\n"
        f"CLAIMS:\n{json.dumps(claims, indent=2)}\n\n"
        'Respond ONLY with valid JSON: {"verifications": [{'
        '"claim": "...", "verdict": "SUPPORTED|CONTRADICTED|NOT_MENTIONED", '
        '"evidence": "..."}]}'
    )
    result = _judge_call(prompt, max_tokens=2000)
    return result.get("verifications", [])


def compute_faithfulness(
    generated_text: str,
    context_chunks: List[Dict[str, Any]],
    recipe_data: List[Dict[str, Any]],
) -> Dict[str, Any]:
    context_text = "\n\n".join(c["text"] for c in context_chunks)
    context_text += "\n\n" + "\n\n".join(json.dumps(r) for r in recipe_data)

    claims = extract_claims(generated_text)
    verifications = verify_claims(claims, context_text)

    n = len(verifications)
    if n == 0:
        return {"faithfulness_score": None, "n_claims": 0, "verifications": [],
                "note": "No checkable claims extracted."}

    supported = sum(1 for v in verifications if v.get("verdict") == "SUPPORTED")
    contradicted = sum(1 for v in verifications if v.get("verdict") == "CONTRADICTED")
    not_mentioned = sum(1 for v in verifications if v.get("verdict") == "NOT_MENTIONED")

    return {
        "faithfulness_score": supported / n,
        "n_claims": n,
        "n_supported": supported,
        "n_contradicted": contradicted,
        "n_not_mentioned": not_mentioned,
        "verifications": verifications,
    }


# ── Answer relevancy ──────────────────────────────────────────────────────────

def compute_answer_relevancy(
    profile: ChildProfile,
    generated_menu: Dict[str, Any],
) -> Dict[str, Any]:
    prompt = (
        "Evaluate whether this lunch recommendation is RELEVANT to the specific "
        "child profile. Relevancy means: does it engage with the child's stated "
        "likes/age/context, not just generic advice?\n\n"
        f"CHILD PROFILE:\n"
        f"- Age: {profile.age_years}\n"
        f"- Allergies/intolerances: {', '.join(profile.allergies + profile.intolerances) or 'none'}\n"
        f"- Likes: {', '.join(profile.likes) or 'not specified'}\n"
        f"- Dislikes: {', '.join(profile.dislikes) or 'not specified'}\n"
        f"- Cultural context: {profile.cultural_context or 'not specified'}\n\n"
        f"GENERATED MENU:\n{json.dumps(generated_menu, indent=2)}\n\n"
        "Rate 1-5 (5=directly engages with this child's profile, "
        "3=generic, 1=irrelevant).\n"
        'Respond ONLY with valid JSON: '
        '{"relevancy_score": <int 1-5>, "reasoning": "..."}'
    )
    return _judge_call(prompt, max_tokens=500)


# ── LLM-as-judge holistic ─────────────────────────────────────────────────────

def llm_judge_holistic(
    profile: ChildProfile,
    generated_menu: Dict[str, Any],
    safe_recipe_names: List[str],
) -> Dict[str, Any]:
    prompt = (
        "You are an expert pediatric nutritionist reviewing an AI-generated "
        "lunch recommendation. Score 1-5 on each dimension, give reasoning "
        "BEFORE the score.\n\n"
        f"CHILD: age {profile.age_years}, "
        f"restrictions: {', '.join(profile.allergies + profile.intolerances) or 'none'}, "
        f"likes: {', '.join(profile.likes) or 'n/a'}\n\n"
        f"ALLOWED RECIPES: {', '.join(safe_recipe_names)}\n\n"
        f"GENERATED OUTPUT:\n{json.dumps(generated_menu, indent=2)}\n\n"
        "Score these dimensions:\n"
        "1. correctness: Is the recipe in the allowed list with accurate facts?\n"
        "2. clarity: Is the explanation clear for a parent/caterer?\n"
        "3. nutritional_soundness: Is the rationale grounded in real numbers?\n"
        "4. tone_appropriateness: Warm but not condescending, no overclaiming?\n\n"
        'Respond ONLY with valid JSON: {'
        '"correctness": {"score": <1-5>, "reasoning": "..."}, '
        '"clarity": {"score": <1-5>, "reasoning": "..."}, '
        '"nutritional_soundness": {"score": <1-5>, "reasoning": "..."}, '
        '"tone_appropriateness": {"score": <1-5>, "reasoning": "..."}, '
        '"overall_score": <1-5>}'
    )
    return _judge_call(prompt, max_tokens=1200)


# ── Orchestration ─────────────────────────────────────────────────────────────

def evaluate_generation_for_profile(profile: ChildProfile) -> Dict[str, Any]:
    retrieval_result = run_retrieval(profile)
    if not retrieval_result["safe_candidates"]:
        return {
            "error": "No safe candidates for this profile.",
            "profile_age": profile.age_years,
        }

    gen_output = generate_menus(
        profile,
        retrieval_result["safe_candidates"],
        retrieval_result["supporting_context"],
    )
    if gen_output.get("generation_error"):
        return {"error": gen_output["generation_error"], "profile_age": profile.age_years}
    if not gen_output.get("menu_options"):
        return {"error": "Generation produced zero menu options.", "profile_age": profile.age_years}

    menu = gen_output["menu_options"][0]
    generated_text = (
        f"{menu.get('why_it_fits', '')} "
        f"{menu.get('nutritional_rationale', '')} "
        f"Allergens confirmed absent: {', '.join(menu.get('allergens_confirmed_absent', []))}."
    )

    faithfulness = compute_faithfulness(
        generated_text,
        retrieval_result["supporting_context"],
        retrieval_result["safe_candidates"],
    )
    relevancy = compute_answer_relevancy(profile, menu)
    holistic = llm_judge_holistic(
        profile,
        menu,
        [r["name"] for r in retrieval_result["safe_candidates"]],
    )

    return {
        "profile_age": profile.age_years,
        "query": retrieval_result["query"],
        "generated_menu": menu,
        "faithfulness": faithfulness,
        "answer_relevancy": relevancy,
        "llm_judge_holistic": holistic,
    }


if __name__ == "__main__":
    from llm_provider import print_provider_status, get_llm

    print_provider_status()
    print()

    try:
        get_llm()
    except RuntimeError as e:
        print(f"\nERROR: {e}")
        sys.exit(1)

    test_profile = ChildProfile(
        age_years=8,
        allergies=["peanut"],
        intolerances=["lactose"],
        likes=["fish", "wraps"],
        school_nut_free=True,
        cultural_context="British school lunch",
    )

    result = evaluate_generation_for_profile(test_profile)
    print(json.dumps(result, indent=2, default=str))
