"""
Unit tests for the LLM-as-judge metric math in benchmark/evaluator.py.

The judge is stubbed throughout — no network, no quota. What is under test is the
aggregation, which is where the damage was done: the published means were
computed over samples that had silently shrunk, and the three metrics covered
different menus from each other.

The central invariant asserted here is that the three metrics always describe the
SAME set of menus. That is what makes the columns of the report table comparable,
and it was false before the three calls were merged into one.

Run:  pytest tests/test_judge_metrics.py -v
"""

from __future__ import annotations

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "benchmark"))

import evaluator  # noqa: E402


PROFILE = {"age_years": 7, "allergies": ["milk"], "likes": ["pasta"],
           "dislikes": [], "cultural_context": ""}

# An id that exists in data/recipes.json, resolved once so the tests do not
# depend on which recipes happen to be in the corpus this week.
REAL_ID = sorted(evaluator.ALL_RECIPE_IDS)[0]


def case(case_id, *, modes, repeat=0):
    """Build a result row. `modes` maps mode -> recipe_id (or None to abstain)."""
    r = {"case_id": case_id, "profile": PROFILE, "repeat": repeat,
         "expected_unsafe_ids": []}
    for mode, rid in modes.items():
        r[mode] = {"final_menus": [{"recipe_id": rid, "why_it_fits": "x"}] if rid else []}
    return r


@pytest.fixture(autouse=True)
def reset_judge_state():
    """JUDGE_STATS and the quota breaker are module-level; leaking them between
    tests would make failures order-dependent."""
    evaluator.JUDGE_STATS.update({k: 0 for k in evaluator.JUDGE_STATS})
    evaluator._QUOTA_EXHAUSTED["hit"] = False
    evaluator._QUOTA_EXHAUSTED["detail"] = ""
    yield


def stub_judge(monkeypatch, verdicts):
    """Replace the network call with a scripted sequence of judge verdicts."""
    seq = list(verdicts)
    calls = {"n": 0}

    def fake(prompt, attempts=3):
        v = seq[calls["n"]] if calls["n"] < len(seq) else seq[-1]
        calls["n"] += 1
        return v

    monkeypatch.setattr(evaluator, "_judge_call", fake)
    return calls


def verdict(rel=4, faith=0.8, nat=3):
    return {"relevance_score": rel, "faithfulness_score": faith,
            "naturalness_score": nat, "reasoning": "ok"}


# ── the invariant that motivated the rewrite ─────────────────────────────────

def test_all_three_metrics_share_one_sample():
    """Before the merge, relevance could be averaged over 4 menus and
    naturalness over a different 3. The three n must now agree by construction."""
    import evaluator as ev
    rows = [case(f"C{i}", modes={"neural_rag": REAL_ID}) for i in range(6)]
    with pytest.MonkeyPatch.context() as mp:
        stub_judge(mp, [verdict()])
        out = ev.compute_llm_metrics(rows)
    b = out["neural_rag"]
    assert b["relevance"]["n"] == b["faithfulness"]["n"] == b["naturalness"]["n"] == 6


def test_one_call_per_menu_not_three():
    """The 3x traffic reduction is what makes a full run fit inside the daily
    token cap; a regression here silently reintroduces the quota collapse."""
    rows = [case(f"C{i}", modes={"neural_rag": REAL_ID, "neurosymbolic": REAL_ID})
            for i in range(5)]
    with pytest.MonkeyPatch.context() as mp:
        calls = stub_judge(mp, [verdict()])
        evaluator.compute_llm_metrics(rows)
    assert calls["n"] == 10          # 5 cases x 2 arms x 1 call


class TestRunFailedClassification:
    """
    A dead arm must not be able to report a perfect safety record.

    In run 20260816_182449 the generator hit its daily token cap during repeat 3.
    Repeats 4 and 5 produced zero menus across all 30 cases, and neural_rag
    reported violation rates of [0.367, 0.367, 0.625, 0.000, 0.000] — two-fifths
    of the mean contributed by an arm that was switched off. The guard tested
    `error`, which is set only when the graph raises; an LLM failure is caught
    inside the generate node and surfaces as `generation_error`.
    """

    def test_rate_limited_call_is_a_failure(self):
        md = {"generation_error": "LLM call failed: RateLimitError: Error code: 429",
              "generation_candidates": ["recipe_001"], "final_menus": []}
        assert evaluator.run_failed(md, "neural_rag") is True

    def test_unparseable_output_is_a_failure(self):
        md = {"generation_error": "Could not parse LLM response: no JSON object",
              "generation_candidates": ["recipe_001"], "final_menus": []}
        assert evaluator.run_failed(md, "neurosymbolic") is True

    def test_graph_exception_is_a_failure(self):
        assert evaluator.run_failed({"error": "KeyError: 'id'"}, "neural_rag") is True

    def test_prefilter_removing_everything_is_a_result_not_a_failure(self):
        """The zero-safe-candidate cases (MUL-03, MUL-04) are the system working."""
        md = {"generation_error": "No candidates available for generation.",
              "generation_candidates": [], "final_menus": []}
        assert evaluator.run_failed(md, "neurosymbolic") is False

    def test_no_rag_emptiness_carries_no_information(self):
        """no_rag has no candidates by design, so empty candidates cannot excuse
        a generation error there."""
        md = {"generation_error": "LLM call failed: RateLimitError",
              "generation_candidates": [], "final_menus": []}
        assert evaluator.run_failed(md, "no_rag") is True

    def test_healthy_run_is_not_a_failure(self):
        md = {"generation_error": None, "generation_candidates": ["recipe_001"],
              "final_menus": [{"recipe_id": "recipe_001"}]}
        assert evaluator.run_failed(md, "neural_rag") is False

    def test_dead_arm_does_not_score_a_clean_violation_rate(self):
        """End to end: an arm whose every call was rate-limited must report its
        cases as errored rather than as a flawless 0.000."""
        rows = []
        for i in range(10):
            rows.append({
                "case_id": f"C{i}", "profile": PROFILE, "repeat": 0,
                "expected_unsafe_ids": ["recipe_001"],
                "neural_rag": {"generation_error": "LLM call failed: RateLimitError",
                               "generation_candidates": ["recipe_001"],
                               "final_menus": [], "proposed_menus": []},
            })
        m = evaluator.compute_safety_metrics(rows)["neural_rag"]
        assert m["cases_errored"] == 10
        assert m["cases_evaluated"] == 0
        # The rate is still 0.000 arithmetically, but the denominator is now
        # visibly empty instead of silently standing in for a real result.
        assert m["coverage"] == 0.0


def test_only_the_first_repeat_is_judged_by_default():
    """
    --repeats puts uncertainty on the safety rates; it must not multiply judge
    traffic by --repeats as well. At 5 repeats that is the difference between
    fitting inside the daily token cap and collapsing partway through it.
    """
    rows = [case(f"C{i}", modes={"neural_rag": REAL_ID}, repeat=rep)
            for rep in range(5) for i in range(4)]
    with pytest.MonkeyPatch.context() as mp:
        calls = stub_judge(mp, [verdict()])
        out = evaluator.compute_llm_metrics(rows)
    assert calls["n"] == 4                          # 4 cases, repeat 0 only
    assert out["neural_rag"]["relevance"]["n"] == 4
    assert out["_judge_all_repeats"] is False


def test_judge_all_repeats_opt_in_scores_every_repeat(monkeypatch):
    """The knob has to actually work, or the within-arm spread it exists for
    is unreachable."""
    monkeypatch.setattr(evaluator, "JUDGE_ALL_REPEATS", True)
    rows = [case(f"C{i}", modes={"neural_rag": REAL_ID}, repeat=rep)
            for rep in range(3) for i in range(4)]
    calls = stub_judge(monkeypatch, [verdict()])
    out = evaluator.compute_llm_metrics(rows)
    assert calls["n"] == 12
    assert out["neural_rag"]["relevance"]["n"] == 12


def test_a_failed_call_drops_all_three_scores_together():
    """The deliberate trade-off: losing a menu entirely is acceptable, losing it
    from one metric only is not."""
    rows = [case("C0", modes={"neural_rag": REAL_ID}),
            case("C1", modes={"neural_rag": REAL_ID})]
    with pytest.MonkeyPatch.context() as mp:
        stub_judge(mp, [verdict(), {"error": "429 rate limit"}])
        out = evaluator.compute_llm_metrics(rows)
    b = out["neural_rag"]
    assert b["relevance"]["n"] == b["faithfulness"]["n"] == b["naturalness"]["n"] == 1


# ── score handling ───────────────────────────────────────────────────────────

def test_means_are_computed_over_the_scored_menus():
    rows = [case("C0", modes={"neural_rag": REAL_ID}),
            case("C1", modes={"neural_rag": REAL_ID})]
    with pytest.MonkeyPatch.context() as mp:
        stub_judge(mp, [verdict(rel=5, faith=1.0, nat=4),
                        verdict(rel=3, faith=0.6, nat=2)])
        out = evaluator.compute_llm_metrics(rows)
    b = out["neural_rag"]
    assert b["relevance"]["mean"] == pytest.approx(4.0)
    assert b["faithfulness"]["mean"] == pytest.approx(0.8)
    assert b["naturalness"]["mean"] == pytest.approx(3.0)


def test_out_of_range_scores_are_rejected_not_averaged():
    """A judge returning 7 on a 1-5 scale has misunderstood the task; averaging
    it in silently inflates the mean."""
    rows = [case("C0", modes={"neural_rag": REAL_ID}),
            case("C1", modes={"neural_rag": REAL_ID})]
    with pytest.MonkeyPatch.context() as mp:
        stub_judge(mp, [verdict(rel=7), verdict(rel=4)])
        out = evaluator.compute_llm_metrics(rows)
    assert out["neural_rag"]["relevance"]["n"] == 1
    assert out["neural_rag"]["relevance"]["mean"] == pytest.approx(4.0)


def test_non_numeric_score_is_rejected():
    rows = [case("C0", modes={"neural_rag": REAL_ID})]
    with pytest.MonkeyPatch.context() as mp:
        stub_judge(mp, [{"relevance_score": "4/5", "faithfulness_score": 0.8,
                         "naturalness_score": 3}])
        out = evaluator.compute_llm_metrics(rows)
    assert out["neural_rag"]["relevance"]["n"] == 0
    assert out["neural_rag"]["faithfulness"]["n"] == 1


def test_hallucinated_id_scores_zero_faithfulness_but_is_still_judged():
    """Previously these menus were skipped entirely, which quietly removed the
    worst outputs from the quality means."""
    rows = [case("C0", modes={"no_rag": "recipe_does_not_exist"})]
    with pytest.MonkeyPatch.context() as mp:
        stub_judge(mp, [verdict(rel=5, faith=0.9, nat=4)])
        out = evaluator.compute_llm_metrics(rows)
    b = out["no_rag"]
    assert b["faithfulness"]["mean"] == 0.0      # forced, judge's 0.9 overridden
    assert b["relevance"]["mean"] == pytest.approx(5.0)
    assert b["n_hallucinated_ids"] == 1


def test_abstentions_are_not_judged():
    rows = [case("C0", modes={"neurosymbolic": None})]
    with pytest.MonkeyPatch.context() as mp:
        calls = stub_judge(mp, [verdict()])
        out = evaluator.compute_llm_metrics(rows)
    assert calls["n"] == 0
    assert out["neurosymbolic"]["relevance"]["n"] == 0


def test_errored_pipeline_is_not_judged():
    rows = [{"case_id": "C0", "profile": PROFILE, "repeat": 0,
             "expected_unsafe_ids": [],
             "neural_rag": {"error": "boom"}}]
    with pytest.MonkeyPatch.context() as mp:
        calls = stub_judge(mp, [verdict()])
        evaluator.compute_llm_metrics(rows)
    assert calls["n"] == 0


# ── provenance ───────────────────────────────────────────────────────────────

def test_per_menu_records_are_persisted():
    """Without these rows the report can state a mean but not which menus it
    covers — the question that could not be answered about the old numbers."""
    rows = [case("C0", modes={"neural_rag": REAL_ID, "neurosymbolic": REAL_ID})]
    with pytest.MonkeyPatch.context() as mp:
        stub_judge(mp, [verdict()])
        out = evaluator.compute_llm_metrics(rows)
    recs = out["_judge_records"]
    assert len(recs) == 2
    assert {r["mode"] for r in recs} == {"neural_rag", "neurosymbolic"}
    assert all(r["case_id"] == "C0" and r["recipe_id"] == REAL_ID for r in recs)
    assert all(r["relevance"] == 4 for r in recs)


def test_paired_block_compares_the_arms_on_shared_cases():
    rows = [case(f"C{i}", modes={"neural_rag": REAL_ID, "neurosymbolic": REAL_ID})
            for i in range(4)]
    with pytest.MonkeyPatch.context() as mp:
        # neurosymbolic is judged first for each case (mode iteration order),
        # so alternate the scripted verdicts to give the arms different scores.
        stub_judge(mp, [verdict(rel=5), verdict(rel=3)] * 4)
        out = evaluator.compute_llm_metrics(rows)
    d = out["_paired"]["neurosymbolic_vs_neural_rag::relevance"]
    assert d["n_pairs"] == 4
    assert abs(d["mean_diff"]) == pytest.approx(2.0)


def test_judge_health_is_reported():
    rows = [case("C0", modes={"neural_rag": REAL_ID})]
    with pytest.MonkeyPatch.context() as mp:
        stub_judge(mp, [verdict()])
        out = evaluator.compute_llm_metrics(rows)
    assert "_judge_health" in out
    assert out["_judge_rubric_anchored"] is True
    assert out["_judge_menus_per_case"] == evaluator.JUDGE_MENUS_PER_CASE
