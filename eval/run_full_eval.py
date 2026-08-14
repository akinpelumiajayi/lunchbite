"""
run_full_eval.py -- Batch generation eval across representative profiles.

Uses llm_provider.get_llm() which reads .env automatically.
Works with Groq or Ollama.

Usage:
  python3 eval/run_full_eval.py
  python3 eval/run_full_eval.py --provider ollama
"""

from __future__ import annotations
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

from guardrails import ChildProfile
from eval_generation import evaluate_generation_for_profile

EVAL_PROFILES = [
    ChildProfile(
        age_years=7,
        school_nut_free=True,
        likes=["rice", "chicken"],
        cultural_context="British primary school",
    ),
    ChildProfile(
        age_years=9,
        allergies=["fish"],
        school_nut_free=True,
    ),
    ChildProfile(
        age_years=8,
        allergies=["peanut"],
        intolerances=["lactose"],
        likes=["fish", "wraps"],
        cultural_context="British school lunch",
    ),
    ChildProfile(
        age_years=12,
        intolerances=["gluten"],
        likes=["sandwiches"],
    ),
    ChildProfile(
        age_years=7,
        allergies=["fish"],
        intolerances=["gluten"],
    ),
]


def run_all(provider: Optional[str] = None) -> None:
    from llm_provider import get_llm, print_provider_status

    print_provider_status()
    print()

    # Apply provider preference if given
    if provider:
        os.environ["_LLM_PROVIDER_PREFER"] = provider

    try:
        llm, provider_name = get_llm(prefer=provider)
        print(f"Running evaluation with: {provider_name}\n")
    except RuntimeError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    results = []
    for i, profile in enumerate(EVAL_PROFILES, 1):
        print(f"--- Profile {i}/{len(EVAL_PROFILES)}: "
              f"age={profile.age_years}, "
              f"allergies={profile.allergies}, "
              f"intolerances={profile.intolerances} ---")
        result = evaluate_generation_for_profile(profile)
        results.append(result)
        if result.get("error"):
            print(f"  SKIPPED: {result['error']}")
        else:
            f = result.get("faithfulness", {})
            r = result.get("answer_relevancy", {})
            j = result.get("llm_judge_holistic", {})
            print(f"  Faithfulness : {f.get('faithfulness_score')} "
                  f"({f.get('n_supported')}/{f.get('n_claims')} claims supported)")
            print(f"  Relevancy    : {r.get('relevancy_score')}/5")
            print(f"  Judge overall: {j.get('overall_score')}/5")

    # Aggregate
    scored = [r for r in results if not r.get("error")]
    skipped = [r for r in results if r.get("error")]

    print(f"\n{'=' * 60}")
    print(f"AGGREGATE  ({len(scored)} scored, {len(skipped)} skipped)")
    print(f"{'=' * 60}")

    if scored:
        def avg(key_path: str) -> Optional[float]:
            vals = []
            for r in scored:
                parts = key_path.split(".")
                v = r
                for p in parts:
                    v = (v or {}).get(p)
                if v is not None:
                    try:
                        vals.append(float(v))
                    except (TypeError, ValueError):
                        pass
            return round(sum(vals) / len(vals), 3) if vals else None

        print(f"  Mean faithfulness       : {avg('faithfulness.faithfulness_score')}")
        print(f"  Mean relevancy (1-5)    : {avg('answer_relevancy.relevancy_score')}")
        print(f"  Mean judge overall (1-5): {avg('llm_judge_holistic.overall_score')}")
        contradicted = sum(
            r.get("faithfulness", {}).get("n_contradicted", 0) for r in scored
        )
        print(f"  Total CONTRADICTED claims: {contradicted} (should be 0)")

    if skipped:
        print(f"\n  Skipped profiles:")
        for r in skipped:
            print(f"    age={r.get('profile_age')}: {r['error']}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=["groq", "ollama"], default=None,
                        help="Force a specific LLM provider (default: auto-detect from .env)")
    # No --groq-key: a key on the command line lands in shell history and is
    # visible in the process table. Use .env (gitignored).
    parser.add_argument("--anthropic-key", default=None,
                        help="(legacy) this eval now uses Groq/Ollama, not Anthropic")
    args = parser.parse_args()

    if args.anthropic_key:
        print("Note: --anthropic-key is no longer used. "
              "This eval now uses Groq or Ollama (set in .env).")

    run_all(provider=args.provider)
