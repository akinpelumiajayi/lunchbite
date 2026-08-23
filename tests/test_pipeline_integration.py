"""
End-to-end tests of the neurosymbolic pipeline with a mocked LLM.

Ported from src/test_pipeline_with_mock_llm.py, which exercised the legacy
retrieval/generation/post_filter trio — a second implementation that has since
been deleted. These run against `build_neurosymbolic_graph`, the pipeline the
benchmark measures and the CLI now drives, so a pass here is evidence about the
system the dissertation reports on.

No API key and no network: the LLM is a MagicMock returning scripted JSON, which
is what makes the adversarial scenarios (an LLM that ignores its instructions,
an LLM that invents a recipe id) testable at all.

Run:  pytest tests/test_pipeline_integration.py -v
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "src", "graphs"))

from guardrails import ChildProfile  # noqa: E402

pytestmark = pytest.mark.skipif(
    not os.path.isdir(os.path.join(ROOT, "vectordb")),
    reason="vector store not built — run: python src/setup_database.py",
)


def mock_llm(payload: Dict[str, Any]) -> Any:
    from langchain_core.messages import AIMessage
    llm = MagicMock()
    llm.invoke.return_value = AIMessage(content=json.dumps(payload))
    return llm


def run(profile: ChildProfile, llm: Any) -> Dict[str, Any]:
    from graphs.build_graphs import build_neurosymbolic_graph
    graph = build_neurosymbolic_graph(llm)
    return graph.invoke({
        "profile": profile.__dict__,
        "pipeline_mode": "neurosymbolic",
        "run_id": f"test-{uuid4().hex[:8]}",
        "query": "", "bm25_candidates": [], "semantic_candidates": [],
        "fused_candidates": [], "reranked_candidates": [],
        "symbolic_pre_filter_log": [],
        "generation_candidates": [], "llm_raw_output": "",
        "proposed_menus": [], "generation_error": None,
        "symbolic_post_filter_log": [], "final_menus": [],
        "error": None, "latency_ms": {},
    })


def menu(recipe_id: str, **over) -> Dict[str, Any]:
    m = {"recipe_id": recipe_id, "menu_name": "Test menu",
         "why_it_fits": "Suits the profile", "nutritional_rationale": "Within limits",
         "allergens_confirmed_absent": [], "source_citation": ""}
    m.update(over)
    return m


def approved_ids(state: Dict[str, Any]) -> List[str]:
    return [c["id"] for c in (state.get("generation_candidates") or [])]


# ── 1. happy path ────────────────────────────────────────────────────────────

def test_well_behaved_llm_passes_through():
    profile = ChildProfile(age_years=9, allergies=["fish"], school_nut_free=True)

    # Probe the pre-filter with a mock that is never expected to matter, to learn
    # which candidates the guardrail approved for this profile.
    probe = run(profile, mock_llm({"menu_options": []}))
    safe = approved_ids(probe)
    assert safe, "expected at least one safe candidate for a fish allergy"
    assert "recipe_005" not in safe, "salmon bagel must not survive a fish allergy"
    assert "recipe_008" not in safe, "tuna salad must not survive a fish allergy"

    state = run(profile, mock_llm({"menu_options": [menu(safe[0])]}))
    assert len(state["final_menus"]) == 1
    assert state["final_menus"][0]["recipe_id"] == safe[0]
    assert all(e["survived"] for e in state["symbolic_post_filter_log"])


# ── 2. the LLM ignores its instructions ──────────────────────────────────────

def test_post_filter_catches_unsafe_recipe():
    """The central safety claim: an LLM recommending an allergen-containing
    recipe must not reach the user."""
    profile = ChildProfile(age_years=9, allergies=["fish"], school_nut_free=True)
    state = run(profile, mock_llm({"menu_options": [
        menu("recipe_005", menu_name="Salmon and salad bagel",
             allergens_confirmed_absent=["fish"]),   # the claim is false
    ]}))

    assert state["final_menus"] == [], "CRITICAL: unsafe recipe reached final output"
    rejects = [e for e in state["symbolic_post_filter_log"] if not e["survived"]]
    assert len(rejects) == 1
    assert "fish" in rejects[0]["rejection_reason"].lower()


def test_false_allergen_claim_is_caught_on_an_otherwise_safe_recipe():
    """Distinct from the above: the recipe is safe *for this profile*, so
    re-running the guardrail passes it. What fails is the model asserting an
    allergen is absent that the recipe record lists as present — the assertion a
    reader would actually rely on."""
    profile = ChildProfile(age_years=9)          # no restrictions
    probe = run(profile, mock_llm({"menu_options": []}))

    from document_loader import _load_json
    recipes = {r["id"]: r for r in _load_json("recipes.json")}
    target = next((rid for rid in approved_ids(probe)
                   if recipes.get(rid, {}).get("allergens_present")), None)
    if target is None:
        pytest.skip("no approved candidate declares an allergen")

    lie = recipes[target]["allergens_present"][0]
    state = run(profile, mock_llm({"menu_options": [
        menu(target, allergens_confirmed_absent=[lie]),
    ]}))

    assert state["final_menus"] == []
    rejects = [e for e in state["symbolic_post_filter_log"] if not e["survived"]]
    assert rejects and rejects[0].get("false_allergen_claim") == [lie.lower()]


# ── 3. hallucinated ids ──────────────────────────────────────────────────────

def test_hallucinated_recipe_id_is_rejected():
    profile = ChildProfile(age_years=10)
    state = run(profile, mock_llm({"menu_options": [menu("recipe_does_not_exist_42")]}))

    assert state["final_menus"] == []
    log = state["symbolic_post_filter_log"]
    assert len(log) == 1 and not log[0]["survived"]
    assert "hallucinated" in log[0]["rejection_reason"].lower()


# ── 4. citation fidelity ─────────────────────────────────────────────────────

def test_fabricated_citation_is_corrected_not_shipped():
    """Traceability is the system's stated selling point, so a citation the
    model invented must never leave the pipeline unchallenged."""
    profile = ChildProfile(age_years=9)
    probe = run(profile, mock_llm({"menu_options": []}))
    safe = approved_ids(probe)
    assert safe

    state = run(profile, mock_llm({"menu_options": [
        menu(safe[0], source_citation="Journal of Invented Nutrition, 2019"),
    ]}))

    assert len(state["final_menus"]) == 1
    assert state["final_menus"][0]["source_citation"] != "Journal of Invented Nutrition, 2019"
    entry = state["symbolic_post_filter_log"][0]
    assert entry["survived"] and entry.get("citation_corrected") is True


# ── 5. no safe candidates ────────────────────────────────────────────────────

def test_zero_safe_candidates_never_calls_the_llm():
    """The cheapest safety property in the system: if nothing survives the
    pre-filter there is nothing to generate from, and the LLM is not given the
    chance to invent something."""
    # Emptied by an explicit ceiling, not by an allergen shotgun. Listing every
    # allergen is corpus-dependent and has broken twice: once when the corpus
    # grew from 9 recipes to 29, and again when the band nutrition gate became
    # advisory and Black Beans and Rice — which declares no allergen at all —
    # started surviving it. A caller-set ceiling is enforced in every gate mode
    # (guardrails.nutrition_gate) and no recipe has 0 g of sugar, so the pool is
    # empty however the corpus changes.
    profile = ChildProfile(
        age_years=7,
        allergies=["fish", "milk", "egg", "gluten", "soya", "peanut", "sesame"],
        intolerances=["celery", "mustard", "lupin", "molluscs", "crustaceans"],
        school_nut_free=True,
        max_sugar_g_override=0.0,
    )
    llm = mock_llm({"menu_options": [menu("recipe_001")]})
    state = run(profile, llm)

    assert approved_ids(state) == [], "test requires an empty candidate pool"
    llm.invoke.assert_not_called()
    assert state["final_menus"] == []
    assert state["generation_error"] == "No candidates available for generation."


# ── 6. per-child ceilings survive the graph ──────────────────────────────────

def test_profile_nutrition_overrides_reach_the_prefilter():
    """
    `_profile_from_dict` rebuilds a ChildProfile from the state dict, and it
    used to drop both override fields — so a ceiling documented as tunable per
    child was enforced in unit tests and inert in every actual pipeline.
    """
    generous = run(ChildProfile(age_years=9, max_sugar_g_override=500.0),
                   mock_llm({"menu_options": []}))
    strict = run(ChildProfile(age_years=9, max_sugar_g_override=0.0),
                 mock_llm({"menu_options": []}))

    assert approved_ids(generous), "a 500 g ceiling should exclude nothing"
    assert approved_ids(strict) == [], "a 0 g ceiling should exclude everything"
