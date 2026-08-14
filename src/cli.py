"""
cli.py -- Interactive command-line interface.

Prompts for a child profile, then runs the original (non-LangGraph)
pipeline to produce lunch recommendations.
LLM provider is resolved from .env automatically.

Usage:
  python3 src/cli.py
"""

from __future__ import annotations
import os
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent))

from guardrails import ChildProfile
from main import recommend_lunches, print_final_recommendations


def prompt_list(label: str) -> List[str]:
    raw = input(f"{label} (comma-separated, leave blank if none): ").strip()
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def prompt_bool(label: str) -> bool:
    raw = input(f"{label} (y/n): ").strip().lower()
    return raw in ("y", "yes")


def main() -> None:
    from llm_provider import print_provider_status, get_llm

    print("=" * 70)
    print("Children's Lunch Recommendation System")
    print("=" * 70)
    print()
    print_provider_status()
    print()

    try:
        _, provider_name = get_llm()
        print(f"Using: {provider_name}\n")
    except RuntimeError as e:
        print(f"\nERROR: {e}")
        sys.exit(1)

    print("Enter the child's profile:\n")

    while True:
        try:
            age = int(input("Child's age (years): ").strip())
            break
        except ValueError:
            print("Please enter a whole number.")

    allergies = prompt_list("Allergies")
    intolerances = prompt_list("Intolerances")
    likes = prompt_list("Foods the child likes")
    dislikes = prompt_list("Foods the child dislikes")
    school_nut_free = prompt_bool("Is the school nut-free")
    cultural_context = input(
        "Cultural context (optional, e.g. 'halal', 'vegetarian household'): "
    ).strip()

    profile = ChildProfile(
        age_years=age,
        allergies=allergies,
        intolerances=intolerances,
        likes=likes,
        dislikes=dislikes,
        school_nut_free=school_nut_free,
        cultural_context=cultural_context,
    )

    result = recommend_lunches(profile)
    print_final_recommendations(result)


if __name__ == "__main__":
    main()
