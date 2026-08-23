"""
Unit tests for benchmark/stats.py.

A significance test that is silently wrong is worse than no test at all — it
launders noise into a claim. The p-values below are checked against hand-computed
exact binomial values rather than against the implementation's own output.

Run:  pytest tests/ -v
"""

from __future__ import annotations

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "benchmark"))

from stats import (  # noqa: E402
    _binom_sf_inclusive,
    bootstrap_ci,
    case_violated,
    mcnemar,
    outcomes_by_case,
    paired_score_diff,
    repeat_variance,
)


def row(case_id, *, unsafe_ids=("recipe_bad",), repeat=0, **modes):
    """Build a result row. Each mode kwarg is the recipe_id it recommended."""
    r = {"case_id": case_id, "expected_unsafe_ids": list(unsafe_ids), "repeat": repeat}
    for mode, rid in modes.items():
        r[mode] = {"final_menus": [{"recipe_id": rid}]} if rid else {"final_menus": []}
    return r


# ── binomial helper ──────────────────────────────────────────────────────────

def test_binom_cdf_matches_hand_computed():
    # P(X <= 0 | n=10, p=0.5) = 1/1024
    assert _binom_sf_inclusive(0, 10) == pytest.approx(1 / 1024)
    # P(X <= 5 | n=10, p=0.5) = 638/1024
    assert _binom_sf_inclusive(5, 10) == pytest.approx(638 / 1024)
    # Whole distribution sums to 1
    assert _binom_sf_inclusive(10, 10) == pytest.approx(1.0)


def test_binom_cdf_zero_trials():
    assert _binom_sf_inclusive(0, 0) == 1.0


# ── per-case outcomes ────────────────────────────────────────────────────────

def test_case_violated_detects_unsafe_recommendation():
    assert case_violated(row("C1", neural_rag="recipe_bad"), "neural_rag") is True
    assert case_violated(row("C1", neural_rag="recipe_ok"), "neural_rag") is False


def test_errored_case_is_none_not_safe():
    """An errored pipeline has no safety outcome; counting it as safe flatters it."""
    r = {"case_id": "C1", "expected_unsafe_ids": ["recipe_bad"],
         "neural_rag": {"error": "429 rate limit"}}
    assert case_violated(r, "neural_rag") is None


def test_abstention_is_not_a_violation():
    assert case_violated(row("C1", neurosymbolic=None), "neurosymbolic") is False


def test_repeats_collapse_conservatively():
    """Violating in any repeat marks the case violated, not majority-vote."""
    rows = [
        row("C1", repeat=0, neural_rag="recipe_ok"),
        row("C1", repeat=1, neural_rag="recipe_ok"),
        row("C1", repeat=2, neural_rag="recipe_bad"),   # one leak in three
    ]
    assert outcomes_by_case(rows, "neural_rag") == {"C1": True}


# ── McNemar ──────────────────────────────────────────────────────────────────

def test_mcnemar_perfect_separation():
    """
    10 cases: neurosymbolic safe on all, neural_rag unsafe on all.
    b=10, c=0 -> p = 2 * P(X<=0 | n=10) = 2/1024 = 0.001953
    """
    rows = [row(f"C{i}", neurosymbolic="recipe_ok", neural_rag="recipe_bad")
            for i in range(10)]
    res = mcnemar(rows, "neurosymbolic", "neural_rag")
    assert res["a_safe_b_unsafe"] == 10
    assert res["a_unsafe_b_safe"] == 0
    assert res["p_value"] == pytest.approx(2 / 1024, abs=1e-6)
    assert res["significant_at_0_05"] is True


def test_mcnemar_no_difference_is_not_significant():
    """5 discordant each way -> p capped at 1.0."""
    rows = [row(f"A{i}", neurosymbolic="recipe_ok", neural_rag="recipe_bad") for i in range(5)]
    rows += [row(f"B{i}", neurosymbolic="recipe_bad", neural_rag="recipe_ok") for i in range(5)]
    res = mcnemar(rows, "neurosymbolic", "neural_rag")
    assert res["n_discordant"] == 10
    assert res["p_value"] == 1.0
    assert res["significant_at_0_05"] is False


def test_mcnemar_known_asymmetric_case():
    """b=8, c=1, n=9 -> p = 2 * (C(9,0)+C(9,1))/2^9 = 20/512 = 0.0390625"""
    rows = [row(f"A{i}", neurosymbolic="recipe_ok", neural_rag="recipe_bad") for i in range(8)]
    rows += [row("B0", neurosymbolic="recipe_bad", neural_rag="recipe_ok")]
    res = mcnemar(rows, "neurosymbolic", "neural_rag")
    assert res["p_value"] == pytest.approx(20 / 512, abs=1e-6)
    assert res["significant_at_0_05"] is True


def test_mcnemar_concordant_cases_are_ignored():
    """
    Cases where both arms agree carry no information about which is better.
    Adding 100 agreeing cases must not change the p-value.
    """
    disc = [row(f"A{i}", neurosymbolic="recipe_ok", neural_rag="recipe_bad") for i in range(8)]
    disc += [row("B0", neurosymbolic="recipe_bad", neural_rag="recipe_ok")]
    base = mcnemar(disc, "neurosymbolic", "neural_rag")["p_value"]

    padded = disc + [row(f"S{i}", neurosymbolic="recipe_ok", neural_rag="recipe_ok")
                     for i in range(100)]
    assert mcnemar(padded, "neurosymbolic", "neural_rag")["p_value"] == base


def test_mcnemar_no_discordant_pairs():
    rows = [row(f"C{i}", neurosymbolic="recipe_ok", neural_rag="recipe_ok") for i in range(5)]
    res = mcnemar(rows, "neurosymbolic", "neural_rag")
    assert res["n_discordant"] == 0
    assert res["p_value"] == 1.0
    assert res["significant_at_0_05"] is False


# ── repeat variance ──────────────────────────────────────────────────────────

def test_variance_reports_nothing_for_single_pass():
    rows = [row("C1", neural_rag="recipe_ok")]
    out = repeat_variance(rows, lambda rs: {})
    assert out["n_repeats"] == 1
    assert "note" in out


def test_variance_mean_and_sd_across_repeats():
    rows = [row("C1", repeat=0, neural_rag="recipe_bad"),
            row("C1", repeat=1, neural_rag="recipe_ok")]

    def fake_metrics(rs):
        # 1.0 on repeat 0, 0.0 on repeat 1
        viol = 1.0 if any(r["neural_rag"]["final_menus"][0]["recipe_id"] == "recipe_bad"
                          for r in rs) else 0.0
        blk = {"allergen_violation_rate": viol,
               "allergen_violation_rate_over_all_cases": viol,
               "coverage": 1.0, "safe_and_useful_rate": 1.0 - viol}
        return {m: blk for m in ["no_llm", "neural_rag", "neurosymbolic", "no_rag"]}

    out = repeat_variance(rows, fake_metrics)
    assert out["n_repeats"] == 2
    vr = out["neural_rag"]["allergen_violation_rate"]
    assert vr["mean"] == pytest.approx(0.5)
    assert vr["sd"] == pytest.approx(0.7071, abs=1e-3)   # sample SD of [1, 0]
    assert sorted(vr["per_repeat"]) == [0.0, 1.0]


def test_variance_drops_repeats_where_every_run_of_an_arm_failed():
    """
    A repeat in which the whole arm was rate-limited scores 0 violations over an
    empty denominator. Averaging it in makes an arm look *safer* the more of it
    died — which is exactly what run 20260816_182449 did, reporting neural_rag
    violations of [0.367, 0.367, 0.625, 0.000, 0.000] after the generator hit
    its daily token cap on repeats 4 and 5.
    """
    rows = [row("C1", repeat=rep, neural_rag="recipe_bad") for rep in range(4)]

    def fake_metrics(rs):
        rep = rs[0]["repeat"]
        alive = rep < 2
        blk = {"allergen_violation_rate": 0.8 if alive else 0.0,
               "allergen_violation_rate_over_all_cases": 0.8 if alive else 0.0,
               "coverage": 1.0 if alive else 0.0,
               "safe_and_useful_rate": 0.2 if alive else 0.0,
               "cases_evaluated": 30 if alive else 0}
        return {m: blk for m in ["no_llm", "neural_rag", "neurosymbolic", "no_rag"]}

    out = repeat_variance(rows, fake_metrics)
    blk = out["neural_rag"]
    assert blk["n_repeats_scored"] == 2
    assert blk["repeats_dropped_all_runs_failed"] == [2, 3]
    vr = blk["allergen_violation_rate"]
    assert vr["per_repeat"] == [0.8, 0.8]
    # The mean is the truth about the runs that happened, not diluted to 0.4 by
    # two repeats that never produced an answer.
    assert vr["mean"] == pytest.approx(0.8)
    assert vr["sd"] == pytest.approx(0.0)


# ── bootstrap CI ─────────────────────────────────────────────────────────────

def test_bootstrap_ci_brackets_the_mean():
    xs = [3, 4, 4, 5, 3, 4, 5, 4, 3, 4]
    ci = bootstrap_ci(xs)
    assert ci["n"] == 10
    assert ci["mean"] == pytest.approx(3.9, abs=1e-9)
    assert ci["lo"] <= ci["mean"] <= ci["hi"]


def test_bootstrap_ci_is_deterministic():
    """A CI that moves between runs of the same data cannot be cited."""
    xs = [1, 2, 3, 4, 5, 4, 3, 2, 5, 1, 3]
    assert bootstrap_ci(xs) == bootstrap_ci(xs)


def test_bootstrap_ci_of_constant_data_has_zero_width():
    ci = bootstrap_ci([4.0] * 8)
    assert (ci["mean"], ci["lo"], ci["hi"]) == (4.0, 4.0, 4.0)


def test_bootstrap_ci_stays_inside_the_scale():
    """The reason for bootstrapping rather than a normal approximation: a mean
    of 4.9 on a 1-5 scale must not produce an upper bound above 5."""
    ci = bootstrap_ci([5, 5, 5, 5, 5, 5, 5, 5, 5, 4])
    assert ci["hi"] <= 5.0
    assert ci["lo"] >= 1.0


def test_bootstrap_ci_refuses_tiny_samples():
    assert bootstrap_ci([4.0, 5.0])["lo"] is None
    assert bootstrap_ci([])["n"] == 0
    assert bootstrap_ci([])["mean"] is None


def test_wider_spread_gives_wider_interval():
    tight = bootstrap_ci([4, 4, 4, 4, 4, 4, 4, 4, 3, 5])
    loose = bootstrap_ci([1, 5, 1, 5, 1, 5, 1, 5, 1, 5])
    assert (loose["hi"] - loose["lo"]) > (tight["hi"] - tight["lo"])


# ── paired judge-score differences ───────────────────────────────────────────

def rec(case_id, mode, *, relevance=None, faithfulness=None, repeat=0):
    return {"case_id": case_id, "mode": mode, "repeat": repeat,
            "relevance": relevance, "faithfulness": faithfulness}


def test_paired_diff_only_uses_cases_both_arms_scored():
    records = [
        rec("A", "neurosymbolic", relevance=4), rec("A", "neural_rag", relevance=3),
        rec("B", "neurosymbolic", relevance=5), rec("B", "neural_rag", relevance=4),
        # C scored for one arm only — must not contribute a phantom pair.
        rec("C", "neurosymbolic", relevance=1),
    ]
    d = paired_score_diff(records, "neurosymbolic", "neural_rag", "relevance")
    assert d["n_pairs"] == 2
    assert d["mean_diff"] == pytest.approx(1.0)


def test_paired_diff_skips_unscored_metric():
    """A menu whose relevance failed to parse is not a pair, even though the
    same menu's faithfulness scored fine."""
    records = [
        rec("A", "neurosymbolic", relevance=None, faithfulness=0.9),
        rec("A", "neural_rag", relevance=4, faithfulness=0.5),
        rec("B", "neurosymbolic", relevance=4, faithfulness=0.8),
        rec("B", "neural_rag", relevance=4, faithfulness=0.4),
    ]
    assert paired_score_diff(records, "neurosymbolic", "neural_rag",
                             "relevance")["n_pairs"] == 1
    assert paired_score_diff(records, "neurosymbolic", "neural_rag",
                             "faithfulness")["n_pairs"] == 2


def test_paired_diff_pairs_within_repeat_not_across():
    """Repeat 0 of one arm must pair with repeat 0 of the other, not repeat 1."""
    records = [
        rec("A", "neurosymbolic", relevance=5, repeat=0),
        rec("A", "neural_rag", relevance=4, repeat=0),
        rec("A", "neurosymbolic", relevance=2, repeat=1),
        rec("A", "neural_rag", relevance=1, repeat=1),
    ]
    d = paired_score_diff(records, "neurosymbolic", "neural_rag", "relevance")
    assert d["n_pairs"] == 2
    assert d["mean_diff"] == pytest.approx(1.0)


def test_paired_diff_detects_a_real_difference():
    records = []
    for i in range(12):
        records.append(rec(f"C{i}", "neurosymbolic", relevance=5))
        records.append(rec(f"C{i}", "neural_rag", relevance=3))
    d = paired_score_diff(records, "neurosymbolic", "neural_rag", "relevance")
    assert d["excludes_zero"] is True
    assert d["mean_diff"] == pytest.approx(2.0)


def test_paired_diff_reports_no_difference_when_arms_agree():
    """The claim 'the symbolic layer does not degrade quality' rests on this
    case producing an interval that spans zero."""
    scores = [4, 3, 5, 4, 4, 3, 5, 5, 4, 3, 4, 4]
    records = []
    for i, s in enumerate(scores):
        records.append(rec(f"C{i}", "neurosymbolic", relevance=s))
        records.append(rec(f"C{i}", "neural_rag", relevance=s))
    d = paired_score_diff(records, "neurosymbolic", "neural_rag", "relevance")
    assert d["excludes_zero"] is False
    assert d["mean_diff"] == pytest.approx(0.0)


def test_paired_diff_with_no_overlap_is_undecidable_not_negative():
    """excludes_zero must be None, not False: 'no pairs' is not evidence of
    equivalence."""
    records = [rec("A", "neurosymbolic", relevance=4),
               rec("B", "neural_rag", relevance=2)]
    d = paired_score_diff(records, "neurosymbolic", "neural_rag", "relevance")
    assert d["n_pairs"] == 0
    assert d["excludes_zero"] is None


# ── failed case-runs must not be paired ──────────────────────────────────────
#
# The 2026-08-18 run is the case in point: the generator's daily token budget ran
# out at case 19, so `neural_rag` produced a scored outcome for 17 of 30 cases.
# `case_violated` tested only `error`, which the runner sets when the *graph*
# raises — a dead LLM call is caught inside the generate node and surfaces as
# `generation_error`. All 13 dead runs were therefore paired as safe refusals,
# and McNemar reported 30 pairs when it had 17.

def _row(case_id, **modes):
    row = {"case_id": case_id, "expected_unsafe_ids": ["recipe_bad"]}
    row.update(modes)
    return row


ANSWERED_SAFE = {"final_menus": [{"recipe_id": "recipe_ok"}],
                 "generation_candidates": [{"id": "recipe_ok"}]}
ANSWERED_UNSAFE = {"final_menus": [{"recipe_id": "recipe_bad"}],
                   "generation_candidates": [{"id": "recipe_bad"}]}
QUOTA_DEAD = {"final_menus": [], "generation_candidates": [{"id": "recipe_ok"}],
              "generation_error": "LLM daily quota exhausted: TPD"}
DELIBERATE_REFUSAL = {"final_menus": [], "generation_candidates": [],
                      "generation_error": "No candidates available for generation."}


def test_a_dead_llm_call_has_no_safety_outcome():
    assert case_violated(_row("X", neural_rag=QUOTA_DEAD), "neural_rag") is None


def test_a_deliberate_refusal_is_a_scored_safe_outcome():
    """The pre-filter removing every candidate is the system working, not failing —
    it is the behaviour the whole neuro-symbolic claim rests on, so it must stay
    in the sample rather than being discarded alongside the outages."""
    assert case_violated(_row("X", neurosymbolic=DELIBERATE_REFUSAL), "neurosymbolic") is False


def test_a_raised_graph_error_has_no_safety_outcome():
    assert case_violated(_row("X", neural_rag={"error": "Timeout"}), "neural_rag") is None


def test_mcnemar_pairs_only_the_cases_both_arms_scored():
    results = [
        _row("C1", neurosymbolic=ANSWERED_SAFE, no_rag=ANSWERED_UNSAFE),
        _row("C2", neurosymbolic=ANSWERED_SAFE, no_rag=ANSWERED_UNSAFE),
        # no_rag died here; the pair does not exist and must not be invented.
        _row("C3", neurosymbolic=ANSWERED_SAFE, no_rag=QUOTA_DEAD),
    ]
    m = mcnemar(results, "neurosymbolic", "no_rag")

    assert m["n_paired_cases"] == 2, "the dead case is not a pair"
    assert m["n_cases_a"] == 3 and m["n_cases_b"] == 2
    assert m["a_safe_b_unsafe"] == 2 and m["a_unsafe_b_safe"] == 0


def test_paired_violation_counts_are_taken_over_the_shared_set():
    """The report prints these next to the p-value, so they have to be counted
    over the same cases the test used — not over each arm's own total."""
    results = [
        _row("C1", neurosymbolic=ANSWERED_SAFE, no_rag=ANSWERED_UNSAFE),
        _row("C2", neurosymbolic=QUOTA_DEAD, no_rag=ANSWERED_UNSAFE),
    ]
    m = mcnemar(results, "neurosymbolic", "no_rag")

    assert m["b_violations"] == 2, "no_rag was unsafe in both cases it scored"
    assert m["b_violations_paired"] == 1, "but only one of those was a pair"
