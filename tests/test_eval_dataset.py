"""
Guards the IR golden set in eval/eval_dataset.py against corpus drift.

This is the bug these tests exist to prevent recurring: the golden set was
hand-labelled when the corpus held 9 recipes and was never updated when it grew
to 29. Recipes 010-029 that genuinely answered a query were counted as false
positives, so every precision/recall figure in the retrieval evaluation
described a corpus that no longer existed — and nothing failed, because a stale
label set is still a valid Python literal.

The rule for each labelled query is re-derived here from data/recipes.json. If
someone adds a recipe, these fail loudly instead of quietly corrupting the
metrics.

Run:  pytest tests/ -v
"""

from __future__ import annotations

import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "eval"))

from eval_dataset import ALL_QUERIES, RECIPE_QUERIES  # noqa: E402


@pytest.fixture(scope="module")
def recipes():
    with open(os.path.join(ROOT, "data", "recipes.json"), encoding="utf-8") as f:
        return json.load(f)


def allergens(r):
    return [a.lower() for a in r.get("allergens_present", [])]


def haystack(r):
    return " ".join(str(x).lower() for x in r.get("ingredients", [])) + " " + r["name"].lower()


def tags(r):
    return r.get("diet_tags") or []


def by_query(q_text):
    for q in RECIPE_QUERIES:
        if q.query == q_text:
            return q
    raise AssertionError(f"query not found: {q_text}")


# ── The corpus itself ────────────────────────────────────────────────────────

def test_corpus_size_is_what_the_labels_assume(recipes):
    """
    The labels below are derived for this corpus size. If it changes, the labels
    must be re-derived — that is the whole point of this file.
    """
    assert len(recipes) == 29, (
        f"Corpus is now {len(recipes)} recipes, not 29. Re-derive relevant_ids in "
        "eval/eval_dataset.py and update this test."
    )


def test_every_labelled_id_exists(recipes):
    """A label pointing at a deleted recipe silently depresses recall."""
    known = {r["id"] for r in recipes}
    for q in RECIPE_QUERIES:
        unknown = q.relevant_ids - known
        assert not unknown, f"query {q.query!r} references non-existent ids: {sorted(unknown)}"


# ── Per-query rules, re-derived from the data ────────────────────────────────

def test_fish_query(recipes):
    expected = {r["id"] for r in recipes if "fish" in allergens(r)}
    assert by_query("fish lunch for a child").relevant_ids == expected


def test_milk_free_query(recipes):
    expected = {r["id"] for r in recipes if "milk" not in allergens(r)}
    assert by_query("milk-free dairy-free lunch").relevant_ids == expected


def test_egg_free_query(recipes):
    expected = {r["id"] for r in recipes if "egg" not in allergens(r)}
    assert by_query("egg-free lunch option").relevant_ids == expected


def test_gluten_free_query(recipes):
    """
    Was the empty set on the 9-recipe corpus and is not any more. A retriever
    returning nothing used to be correct here and is now wrong.
    """
    expected = {r["id"] for r in recipes
                if "cereals containing gluten" not in allergens(r)}
    q = by_query("gluten-free wheat-free lunch")
    assert q.relevant_ids == expected
    assert q.relevant_ids, "gluten-free is no longer a zero-answer query"


def test_chicken_query(recipes):
    expected = {r["id"] for r in recipes
                if "chicken" in haystack(r) and "high-protein" in tags(r)}
    assert by_query("high protein chicken lunch").relevant_ids == expected


def test_vegetarian_cheese_query(recipes):
    expected = {r["id"] for r in recipes
                if "cheese" in haystack(r) and "vegetarian" in tags(r)}
    assert by_query("vegetarian sandwich with cheese").relevant_ids == expected


def test_hummus_query(recipes):
    expected = {r["id"] for r in recipes
                if "hummus" in haystack(r) or "chickpea" in haystack(r)}
    assert by_query("hummus chickpea recipe").relevant_ids == expected


# ── Honesty about what the set can measure ───────────────────────────────────

def test_low_discrimination_queries_are_flagged(recipes):
    """
    A query whose relevant set is most of the corpus cannot distinguish a good
    retriever from a bad one. Those queries are allowed, but they must say so in
    their notes, so the number is not quoted as if it were discriminative.
    """
    n = len(recipes)
    for q in RECIPE_QUERIES:
        if q.relevant_ids and len(q.relevant_ids) / n > 0.4:
            assert "LOW DISCRIMINATION" in q.notes, (
                f"query {q.query!r} covers {len(q.relevant_ids)}/{n} of the corpus "
                "but is not flagged as low-discrimination in its notes"
            )


def test_queries_have_auditable_notes():
    """Every label must carry the reasoning that justifies it."""
    for q in ALL_QUERIES:
        assert q.notes.strip(), f"query {q.query!r} has no notes explaining its labels"
