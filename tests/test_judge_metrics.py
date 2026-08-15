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
