"""
Diet-requirement enforcement in the symbolic layer.

Objective 1(iii) names three things the constraint layer must check: allergen /
exclusion rules, nutrition thresholds, and *diet-tag compliance*. The first two
were implemented; the third was not. Diet tags were read into the retrieval chunk
text (`src/document_loader.py:70`), the chunk metadata (`:81`) and the generator
prompt (`src/graphs/nodes.py:490`) — that is, handled neurally — while
`check_recipe_against_profile` never looked at them. Halal, kosher and vegetarian
compliance rested entirely on the model choosing to honour them, which is the
arrangement this project exists to argue against.

The benchmark recorded the same gap: CUL-02 and CUL-03 carried
`expected_unsafe_ids=[]` with the comment "guardrails don't enforce vegetarian
(not an allergen)". An empty unsafe list cannot fail, so every arm scored those
cases safe regardless of what it recommended.

Run:  pytest tests/test_diet_gate.py -v
"""

from __future__ import annotations

import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "benchmark"))

from guardrails import (  # noqa: E402
    ChildProfile, check_recipe_against_profile, normalize_diet_terms,
    DIET_SPECS, diet_gate, DEFAULT_DIET_GATE,
)

RECIPES = json.load(open(os.path.join(ROOT, "data", "recipes.json"), encoding="utf-8"))
BY_ID = {r["id"]: r for r in RECIPES}


@pytest.fixture(autouse=True)
def _clean_gate(monkeypatch):
    monkeypatch.delenv("DIET_GATE", raising=False)


def _check(rid, **profile_kwargs):
    return check_recipe_against_profile(BY_ID[rid], ChildProfile(age_years=10, **profile_kwargs))


# ── The vocabulary ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("raw,expected", [
    ("vegetarian", "vegetarian"),
    ("veggie", "vegetarian"),
    ("meat-free", "vegetarian"),
    ("vegetarian household", "vegetarian"),
    ("plant-based", "vegan"),
    ("vegan household", "vegan"),
    ("halal diet required — no pork products", "halal"),
    ("kosher diet required", "kosher"),
    ("pescetarian", "pescatarian"),
])
def test_diet_synonyms_normalise(raw, expected):
    recognised, unknown = normalize_diet_terms([raw])
    assert expected in recognised, f"{raw!r} did not map to {expected!r}"
    assert not unknown


def test_vegan_supersedes_vegetarian():
    """Both would fire on the same recipe; reporting the miss twice helps nobody."""
    recognised, _ = normalize_diet_terms(["vegan", "vegetarian"])
    assert recognised == {"vegan"}


def test_unknown_diet_is_reported_not_silently_enforced():
    """
    An unmapped allergy term is still matched literally against ingredient text.
    A diet has no such fallback, so claiming to honour it would be a lie —
    it must surface as a warning instead.
    """
    recognised, unknown = normalize_diet_terms(["jain"])
    assert recognised == set()
    assert unknown == ["jain"]

    result = _check("recipe_017", diet_requirements=["jain"])
    assert any("Unrecognised diet requirement" in w for w in result.warnings)


# ── Enforcement ───────────────────────────────────────────────────────────────

def test_vegetarian_rejects_meat():
    result = _check("recipe_007", diet_requirements=["vegetarian"])   # chicken
    assert not result.passed
    assert any("vegetarian" in r for r in result.reasons_for_rejection)


def test_vegetarian_rejects_fish():
    """Fish is not vegetarian. The benchmark's own comment listed only meat."""
    result = _check("recipe_005", diet_requirements=["vegetarian"])   # salmon
    assert not result.passed


def test_vegetarian_accepts_a_vegetarian_recipe():
    assert _check("recipe_002", diet_requirements=["vegetarian"]).passed


def test_vegan_rejects_dairy_via_the_tagged_allergen_list():
    """Not via an ingredient scan: the tagged allergen field is authoritative."""
    result = _check("recipe_001", diet_requirements=["vegan"])
    assert not result.passed
    assert any("milk" in r for r in result.reasons_for_rejection)


def test_halal_rejects_pork_and_ham():
    for rid in ("recipe_017", "recipe_020", "recipe_022"):
        assert not _check(rid, diet_requirements=["halal"]).passed, rid


def test_halal_accepts_a_non_pork_recipe():
    assert _check("recipe_005", diet_requirements=["halal"]).passed


def test_requirement_is_read_from_cultural_context_too():
    """Existing profiles state the requirement there; it should not need restating."""
    result = _check("recipe_017", cultural_context="halal diet required — no pork products")
    assert not result.passed


# ── The gate modes ────────────────────────────────────────────────────────────

def test_gate_defaults_to_hard():
    assert DEFAULT_DIET_GATE == "hard"
    assert diet_gate() == "hard"


def test_advisory_gate_warns_instead_of_rejecting(monkeypatch):
    monkeypatch.setenv("DIET_GATE", "advisory")
    result = _check("recipe_007", diet_requirements=["vegetarian"])
    assert result.passed
    assert any("vegetarian" in w for w in result.warnings)


def test_off_gate_skips_the_check_entirely(monkeypatch):
    monkeypatch.setenv("DIET_GATE", "off")
    result = _check("recipe_007", diet_requirements=["vegetarian"])
    assert result.passed
    assert not any("vegetarian" in w for w in result.warnings)


def test_allergen_gating_is_untouched_by_the_diet_gate(monkeypatch):
    """
    The nutrition gate is tunable because its data is unreliable. Neither gate may
    ever relax an allergen, in any mode.
    """
    for mode in ("hard", "advisory", "off"):
        monkeypatch.setenv("DIET_GATE", mode)
        result = _check("recipe_001", allergies=["milk"])
        assert not result.passed, f"milk allergy leaked under DIET_GATE={mode}"


# ── Claims the check must not make ────────────────────────────────────────────

@pytest.mark.parametrize("diet", ["halal", "kosher"])
def test_non_certifiable_diets_always_disclose_their_limits(diet):
    """
    Passing an ingredient scan is not certification. A system reporting
    "kosher: passed" off the ingredient list would be asserting something the
    corpus cannot support, so the caveat rides on passes as well as failures.
    """
    assert DIET_SPECS[diet]["certifiable"] is False
    result = _check("recipe_002", diet_requirements=[diet])   # a passing recipe
    assert result.passed
    assert any("ingredient exclusion only" in w for w in result.warnings)


@pytest.mark.parametrize("diet", ["vegetarian", "vegan", "pescatarian"])
def test_corpus_backed_diets_make_no_such_disclaimer(diet):
    result = _check("recipe_013", diet_requirements=[diet])
    assert not any("ingredient exclusion only" in w for w in result.warnings)


# ── The benchmark's declared sets must match the corpus ───────────────────────

def test_declared_unsafe_sets_match_the_corpus():
    """
    benchmark_cases.py hardcodes the diet unsafe-sets so the expectation is not
    simply a transcript of whatever the guardrail currently does. That only holds
    if something checks the two against each other.
    """
    import benchmark_cases as bc

    for name, declared, diet in [
        ("VEGETARIAN_UNSAFE", bc.VEGETARIAN_UNSAFE, "vegetarian"),
        ("HALAL_UNSAFE", bc.HALAL_UNSAFE, "halal"),
        ("KOSHER_UNSAFE", bc.KOSHER_UNSAFE, "kosher"),
        ("VEGAN_UNSAFE", bc.VEGAN_UNSAFE, "vegan"),
    ]:
        actual = sorted(
            r["id"] for r in RECIPES
            if not check_recipe_against_profile(
                r, ChildProfile(age_years=10, diet_requirements=[diet])).passed
        )
        assert sorted(declared) == actual, (
            f"{name} disagrees with the corpus: "
            f"declared-only={sorted(set(declared) - set(actual))}, "
            f"corpus-only={sorted(set(actual) - set(declared))}"
        )


def test_cultural_cases_are_no_longer_unfailable():
    """Every cultural case must be able to fail, or it tests nothing."""
    import benchmark_cases as bc

    cultural = [c for c in bc.BENCHMARK_CASES if c.category == "cultural"]
    assert cultural
    for case in cultural:
        assert case.expected_unsafe_ids, (
            f"{case.case_id} has an empty expected_unsafe_ids, so no output can fail it"
        )
        assert case.profile.get("diet_requirements"), (
            f"{case.case_id} states its requirement only as prose"
        )


# ── Every declared-unsafe recipe must actually be unsafe ─────────────────────

def test_no_case_declares_a_recipe_unsafe_that_the_guardrail_passes():
    """
    The dangerous direction of a benchmark/corpus disagreement.

    If a case lists a recipe under expected_unsafe_ids that the guardrail
    considers safe for that profile, then a system doing the right thing is
    scored as committing a violation. That inflates every arm's violation rate
    and penalises correct behaviour — the opposite of what the metric is for.

    Found by ADV-14: SESAME_UNSAFE listed recipe_002, the NHS yoghurt-based
    hummus, which contains no tahini and is not tagged sesame. `no_llm` — an arm
    with no LLM and therefore no way to misbehave — was reported as violating.

    The reverse direction is deliberately NOT asserted: the guardrail rejecting
    more than the case declares is the documented ingredient-keyword
    over-blocking, measured as pre-filter precision, and is a known cost rather
    than a defect.
    """
    import benchmark_cases as bc

    problems = []
    for case in bc.BENCHMARK_CASES:
        profile = ChildProfile(
            age_years=case.profile["age_years"],
            allergies=case.profile.get("allergies", []),
            intolerances=case.profile.get("intolerances", []),
            school_nut_free=case.profile.get("school_nut_free", False),
            cultural_context=case.profile.get("cultural_context", ""),
            diet_requirements=case.profile.get("diet_requirements", []),
        )
        for rid in case.expected_unsafe_ids:
            if check_recipe_against_profile(BY_ID[rid], profile).passed:
                problems.append(f"{case.case_id}: {rid} declared unsafe but the guardrail passes it")

    assert not problems, "declared-unsafe recipes the guardrail considers safe:\n  " + "\n  ".join(problems)
