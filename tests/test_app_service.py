"""
Tests for the LunchBite dashboard layer.

The load-bearing one is `test_shape_result_matches_recommend_lunches`. The
dashboard obtains a terminal state by streaming rather than invoking, so it
cannot reuse `recommend_lunches` directly; what it must never do is grow its
own copy of the mapping from state to display. `src/main.py` documents at
length what that cost last time. This asserts the two agree.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "graphs"))
sys.path.insert(0, str(ROOT / "app"))

from document_loader import ALL_14_ALLERGENS, recipes_by_id  # noqa: E402
from guardrails import (ChildProfile, DIET_SPECS, _daily_to_lunch_fraction,  # noqa: E402
                        _load_age_band_limits, check_recipe_against_profile,
                        lunch_fraction, lunch_limits_for_age,
                        normalize_allergy_terms_with_unknowns, normalize_diet_terms)
from main import recommend_lunches, shape_result  # noqa: E402

streamlit = pytest.importorskip("streamlit", reason="the dashboard needs streamlit")


# -- document_loader.recipes_by_id --------------------------------------------

def test_recipes_by_id_covers_the_corpus():
    index = recipes_by_id()
    assert len(index) == 29
    assert all(key == recipe["id"] for key, recipe in index.items())


def test_recipes_by_id_is_cached():
    assert recipes_by_id() is recipes_by_id()


def test_graph_and_loader_share_one_index():
    """nodes._recipes_by_id delegates, so the app and the graph cannot diverge."""
    from graphs.nodes import _recipes_by_id

    assert _recipes_by_id() is recipes_by_id()


# -- guardrails.lunch_limits_for_age ------------------------------------------

@pytest.mark.parametrize("age", [5, 8, 12, 16])
def test_lunch_limits_match_the_ceilings_the_gate_applies(age):
    """
    The displayed ceilings must be the ones the gate enforces.

    Recomputed here from the private helpers rather than hard-coded, so the
    test tracks a change to the guidelines file instead of failing on one.
    """
    limits = lunch_limits_for_age(age)
    band = _load_age_band_limits(age)
    assert limits is not None and band is not None

    assert limits["sugars_g"] == _daily_to_lunch_fraction(band["free_sugars_g_day_max"])
    assert limits["salt_g"] == _daily_to_lunch_fraction(band["salt_g_day_max"])
    assert limits["fraction"] == lunch_fraction()
    assert limits["kcal_target"] == band.get("approx_lunch_target_kcal")


@pytest.mark.parametrize("age", [3, 19, 20, 65])
def test_lunch_limits_are_none_outside_the_supported_band(age):
    """None means 'not checked'. A caller reading it as zero would invert it."""
    assert lunch_limits_for_age(age) is None


def test_displayed_sugar_ceiling_is_the_one_that_triggers_the_flag():
    """A recipe just over the displayed ceiling must actually be flagged."""
    limits = lunch_limits_for_age(8)
    profile = ChildProfile(age_years=8)
    recipe = {
        "id": "test", "name": "test", "ingredients": [], "allergens_present": [],
        "diet_tags": [], "extras_suggested": [],
        "nutrition_per_serving": {"sugars_g": limits["sugars_g"] + 1.0, "salt_g": 0.1},
    }
    result = check_recipe_against_profile(recipe, profile)
    assert any("Sugar" in flag for flag in result.nutrition_flags)


def test_band_label_names_the_band_used():
    assert lunch_limits_for_age(5)["band_label"] == "age_4_6"
    assert lunch_limits_for_age(8)["band_label"] == "age_7_10"
    assert lunch_limits_for_age(12)["band_label"] == "age_11_14"


# -- main.shape_result --------------------------------------------------------

DISPLAY_KEYS = {
    "profile", "query", "final_recommendations", "rejected_at_retrieval",
    "rejected_at_post_filter", "generation_error", "explanation", "unverifiable",
}


def _canned_state() -> Dict[str, Any]:
    return {
        "query": "age 7, allergic to milk",
        "symbolic_pre_filter_log": [
            {"recipe_id": "recipe_001", "recipe_name": "A", "passed": False,
             "reasons": ["Contains restricted allergen(s): milk."],
             "warnings": [], "nutrition_flags": []},
            {"recipe_id": "recipe_005", "recipe_name": "B", "passed": True,
             "reasons": [], "warnings": [], "nutrition_flags": []},
        ],
        "generation_candidates": [{"id": "recipe_005", "name": "B"}],
        "proposed_menus": [{"recipe_id": "recipe_005", "menu_name": "B"}],
        "symbolic_post_filter_log": [
            {"recipe_id": "recipe_005", "menu_name": "B", "survived": True,
             "rejection_reason": None, "symbolic_verified": True},
        ],
        "final_menus": [{"recipe_id": "recipe_005", "menu_name": "B"}],
        "generation_error": None,
    }


def test_shape_result_returns_the_display_contract():
    result = shape_result(_canned_state(), ChildProfile(age_years=7))
    assert set(result) == DISPLAY_KEYS
    assert result["query"] == "age 7, allergic to milk"
    assert [m["recipe_id"] for m in result["final_recommendations"]] == ["recipe_005"]
    assert [e["recipe_id"] for e in result["rejected_at_retrieval"]] == ["recipe_001"]
    assert result["rejected_at_post_filter"] == []
    assert result["explanation"] is None


def test_shape_result_explains_an_empty_candidate_pool():
    state = _canned_state()
    state["generation_candidates"] = []
    result = shape_result(state, ChildProfile(age_years=7))
    assert result["explanation"]
    assert "knowledge base" in result["explanation"]


def test_shape_result_matches_recommend_lunches(monkeypatch, capsys):
    """
    The two paths into the display contract must produce the same thing.

    `recommend_lunches` runs the graph and prints; the dashboard streams the
    same graph and does not. Only the shaping is compared, with the pipeline
    itself stubbed out so the test needs no model and no network.
    """
    import main as main_module

    state = _canned_state()
    profile = ChildProfile(age_years=7, allergies=["milk"])
    monkeypatch.setattr(main_module, "run_pipeline", lambda _profile: state)

    from_cli = recommend_lunches(profile, verbose=True)
    capsys.readouterr()  # discard the CLI's printing
    from_app = shape_result(state, profile)

    assert from_cli == from_app


def test_shape_result_is_pure():
    """It must not mutate the state the caller is still holding."""
    state = _canned_state()
    before = repr(state)
    shape_result(state, ChildProfile(age_years=7))
    assert repr(state) == before


def test_profile_survives_as_a_dataclass():
    """Callers serialising the result have to know this is not a dict."""
    profile = ChildProfile(age_years=7)
    result = shape_result(_canned_state(), profile)
    assert result["profile"] is profile
    assert not isinstance(result["profile"], dict)


# -- vocab --------------------------------------------------------------------

def test_every_allergy_option_is_understood_by_the_gate():
    """The form must not offer a term the guardrail cannot interpret."""
    from lunchbite import vocab

    options = vocab.ALLERGEN_OPTIONS + vocab.ALLERGY_ALIAS_OPTIONS
    _, unknown = normalize_allergy_terms_with_unknowns(options)
    assert unknown == []


def test_every_diet_option_is_understood_by_the_gate():
    from lunchbite import vocab

    _, unknown = normalize_diet_terms(vocab.DIET_OPTIONS + vocab.DIET_ALIAS_OPTIONS)
    assert unknown == []


def test_allergen_options_are_the_canonical_fourteen():
    from lunchbite import vocab

    assert vocab.ALLERGEN_OPTIONS == list(ALL_14_ALLERGENS)


def test_uncertifiable_diets_match_the_specs():
    from lunchbite import vocab

    expected = sorted(n for n, s in DIET_SPECS.items() if not s["certifiable"])
    assert vocab.UNCERTIFIABLE_DIETS == expected


def test_preview_enforcement_reports_unknown_terms():
    from lunchbite import vocab

    preview = vocab.preview_enforcement(["dairy", "fairy dust"], ["vegan"])
    assert "milk" in preview["allergens"]
    assert preview["unknown_allergens"] == ["fairy dust"]
    assert "vegan" in preview["diets"]


def test_cultural_context_is_not_reported_as_an_unenforced_diet():
    """
    Prose in the cultural-context field is not a diet requirement going unmet.

    `ChildProfile.required_diets` scans cultural_context as well as the diet
    list, so the guardrail reports "British primary school" as an unrecognised
    diet requirement that is "NOT enforced". Nothing was being restricted, so
    presenting that as a warning would be a false alarm on the most common
    profile there is. It is surfaced as a note instead.
    """
    from lunchbite import vocab

    preview = vocab.preview_enforcement(["milk"], [], "British primary school")
    assert preview["unknown_diets"] == []
    assert preview["context_unmatched"] == "British primary school"
    assert preview["diets"] == []


def test_a_diet_named_only_in_cultural_context_is_still_enforced():
    """The guardrail reads it, so the form must show it as enforced."""
    from lunchbite import vocab

    preview = vocab.preview_enforcement([], [], "vegetarian household")
    assert preview["diets"] == ["vegetarian"]
    assert preview["diets_from_context"] == ["vegetarian"]
    assert preview["context_unmatched"] == ""


def test_an_unrecognised_declared_diet_is_still_a_warning():
    """The distinction only applies to context prose, never to the diet field."""
    from lunchbite import vocab

    preview = vocab.preview_enforcement([], ["fairy diet"], "")
    assert preview["unknown_diets"] == ["fairy diet"]


def test_preview_diets_match_what_the_profile_will_enforce():
    """The preview and ChildProfile.required_diets must not disagree."""
    from lunchbite import vocab

    for diets, context in (([], "vegetarian household"), (["vegan"], ""),
                           (["halal"], "British primary school"),
                           ([], "halal diet required - no pork products")):
        profile = ChildProfile(age_years=7, diet_requirements=list(diets),
                               cultural_context=context)
        enforced, _ = profile.required_diets()
        preview = vocab.preview_enforcement([], diets, context)
        assert set(preview["diets"]) == enforced, (diets, context)


def test_split_free_text_matches_the_cli():
    from lunchbite import vocab

    assert vocab.split_free_text(" a, b ,, c ") == ["a", "b", "c"]
    assert vocab.split_free_text("") == []


@pytest.mark.parametrize("name", [
    "Milk allergy, nut-free school", "Coeliac, age 9", "Vegetarian, egg allergy",
    "Vegan, age 11", "Halal, fish allergy",
    "Tightly restricted (narrows to one recipe)",
    "Impossible ceiling (shows the empty state)",
])
def test_presets_describe_what_they_actually_do(name):
    """
    A preset's label is a claim about the corpus, so it is checked against it.

    Run against the guardrail over all 29 recipes -- no retrieval, no model, so
    this measures what is satisfiable rather than what happened to be retrieved.
    """
    from lunchbite import vocab

    preset = vocab.PRESETS[name]
    profile = ChildProfile(
        age_years=preset.get("age_years", 7),
        allergies=list(preset.get("allergies", [])),
        intolerances=list(preset.get("intolerances", [])),
        likes=list(preset.get("likes", [])),
        dislikes=list(preset.get("dislikes", [])),
        school_nut_free=bool(preset.get("school_nut_free", False)),
        cultural_context=preset.get("cultural_context", ""),
        diet_requirements=list(preset.get("diet_requirements", [])),
        max_sugar_g_override=preset.get("max_sugar_g_override"),
        max_salt_g_override=preset.get("max_salt_g_override"),
    )
    survivors = [r for r in recipes_by_id().values()
                 if check_recipe_against_profile(r, profile).passed]

    if "empty state" in name:
        assert survivors == [], f"{name!r} claims no result but {len(survivors)} survive"
    elif "narrows to one" in name:
        assert len(survivors) == 1, f"{name!r} claims one but {len(survivors)} survive"
    else:
        assert survivors, f"{name!r} leaves nothing for the recommender to choose"


# -- service ------------------------------------------------------------------

def test_recipe_join_resolves_every_id_the_pipeline_can_emit():
    from lunchbite import service

    for recipe_id in recipes_by_id():
        assert service.recipe_for({"recipe_id": recipe_id})["id"] == recipe_id


def test_recipe_join_returns_empty_for_an_invented_id():
    """The no_rag arm can name a recipe that does not exist; the card says so."""
    from lunchbite import service

    assert service.recipe_for({"recipe_id": "recipe_999"}) == {}
    assert service.recipe_for({}) == {}


def test_initial_state_covers_every_pipeline_state_key():
    """A key the graph reads but the app never seeds would fail at runtime."""
    from lunchbite import service
    from state import PipelineState

    seeded = set(service._initial_state(ChildProfile(age_years=7), "neurosymbolic", "id"))
    declared = set(PipelineState.__annotations__)
    assert declared - seeded == set()


def test_profile_warnings_separate_from_per_recipe_warnings():
    """
    Told apart by which recipes carry them, never by matching their wording.

    plan.md §1.4 records reason-string matching breaking when a message was
    reworded; this keeps that class of bug out of the display layer.
    """
    from lunchbite import service

    state = {"symbolic_pre_filter_log": [
        {"recipe_id": "a", "warnings": ["unrecognised term", "side item has milk"]},
        {"recipe_id": "b", "warnings": ["unrecognised term"]},
        {"recipe_id": "c", "warnings": ["unrecognised term"]},
    ]}
    assert service.profile_warnings(state) == ["unrecognised term"]
    by_recipe = service.warnings_by_recipe(state)
    assert "side item has milk" in by_recipe["a"]


def test_warnings_by_recipe_drops_profile_level_by_default():
    """
    A profile-level warning must not repeat on every card.

    The guardrail attaches it to every candidate it checks, so an unfiltered
    per-recipe view shows the same sentence once per lunch -- which is how the
    cultural-context note ended up on all three recommendations.
    """
    from lunchbite import service

    state = {"symbolic_pre_filter_log": [
        {"recipe_id": "a", "warnings": ["about the profile", "side item has milk"]},
        {"recipe_id": "b", "warnings": ["about the profile"]},
        {"recipe_id": "c", "warnings": ["about the profile"]},
    ]}
    filtered = service.warnings_by_recipe(state)
    assert filtered == {"a": ["side item has milk"]}

    raw = service.warnings_by_recipe(state, include_profile_level=True)
    assert raw["b"] == ["about the profile"]
    assert len(raw) == 3


def test_profile_warnings_abstain_below_two_candidates():
    """With one candidate there is no evidence a warning is profile-level."""
    from lunchbite import service

    state = {"symbolic_pre_filter_log": [{"recipe_id": "a", "warnings": ["x"]}]}
    assert service.profile_warnings(state) == []


def test_fatal_error_kinds():
    from graphs.nodes import AUTH_FAILED_PREFIX, QUOTA_EXHAUSTED_PREFIX
    from lunchbite import service

    assert service.fatal_error_kind(QUOTA_EXHAUSTED_PREFIX + ": spent") == "quota"
    assert service.fatal_error_kind(AUTH_FAILED_PREFIX + ": bad key") == "auth"
    assert service.fatal_error_kind("could not parse JSON") is None
    assert service.fatal_error_kind(None) is None


def test_funnel_counts_are_in_pipeline_order():
    from lunchbite import service

    state = _canned_state()
    state["fused_candidates"] = [{"id": "recipe_001"}, {"id": "recipe_005"}]
    stages = service.funnel_counts(state, shape_result(state, ChildProfile(age_years=7)))
    assert [name for name, _ in stages] == [
        "Retrieved", "Checked by the guardrail", "Passed the guardrail",
        "Proposed by the model", "Survived verification",
    ]
    assert [n for _, n in stages] == [2, 2, 1, 1, 1]


# -- components ---------------------------------------------------------------

def test_meter_status_matches_the_guardrails_own_threshold():
    """The meter and the flag beside it must agree on what counts as near."""
    from lunchbite import components

    assert components._meter_status(5.0, 10.0)[0] == "good"
    assert components._meter_status(9.0, 10.0)[0] == "warning"
    assert components._meter_status(11.0, 10.0)[0] == "critical"
    assert components.NEAR_CEILING == 0.85


def test_meter_handles_a_missing_value_and_a_missing_ceiling():
    from lunchbite import components

    assert "not recorded" in components.nutrient_meter("Salt", None, 2.0)
    assert "no guideline" in components.nutrient_meter("Salt", 1.2, None)


def test_meter_escapes_its_label():
    from lunchbite import components

    assert "<script>" not in components.nutrient_meter("<script>", 1.0, 2.0)


def test_chips_escape_their_content():
    from lunchbite import components

    assert "<b>" not in components.chips(["<b>milk</b>"])


def test_every_meal_category_has_an_accent():
    """A category with no entry would fall back and lose its identity."""
    from lunchbite import theme

    for recipe in recipes_by_id().values():
        category = recipe["meal_category"]
        assert category in theme.CATEGORY_META, f"no accent for {category!r}"


def test_arms_cover_every_declared_pipeline_mode():
    from lunchbite import theme

    assert set(theme.ARMS) == {"no_llm", "neural_rag", "neurosymbolic",
                               "no_rag", "reward_ranked"}
    assert theme.ARMS["no_llm"]["needs_llm"] is False
    assert all(theme.ARMS[a]["needs_llm"] for a in theme.ARMS if a != "no_llm")


# -- End to end ---------------------------------------------------------------

@pytest.mark.skipif(not (ROOT / "vectordb").exists(),
                    reason="needs a built index (python src/setup_database.py)")
def test_app_produces_a_safe_recommendation_without_an_llm(monkeypatch):
    """
    A real run of the whole page in the arm that calls no model.

    No API key, no network. Asserts the page renders and that what it
    recommended genuinely excludes the restricted allergens -- checked against
    the recipe records, not against the model's claim about them.
    """
    from streamlit.testing.v1 import AppTest

    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    app = AppTest.from_file(str(ROOT / "app" / "LunchBite.py"), default_timeout=900)
    app.run()
    assert not app.exception

    app.session_state["lb_arm"] = "no_llm"
    app.session_state["lb_allergies"] = ["milk"]
    app.session_state["lb_nut_free"] = True
    app.session_state["lb_age"] = 7
    app.run()
    app.button[0].click().run()

    assert not app.exception, [str(e.value) for e in app.exception]
    assert not app.error, [str(e.value) for e in app.error]

    payload = app.session_state["lb_last_run"]
    menus = payload["result"]["final_recommendations"]
    assert menus, "the rule-based arm returned nothing for a single milk allergy"

    index = recipes_by_id()
    for menu in menus:
        allergens = {a.lower() for a in index[menu["recipe_id"]]["allergens_present"]}
        assert not (allergens & {"milk", "nuts", "peanut"}), (
            f"{menu['menu_name']} carries {allergens & {'milk', 'nuts', 'peanut'}}"
        )
