"""
Agreement metrics for the human validation study (src/reward/agreement.py).

These are the numbers that would be reported as "the reward is validated", so
each has to be shown behaving correctly at both ends: 1.0 for a rater who agrees
perfectly, and ~0 for one answering at random. A metric that only ever returns a
comfortable number is worse than none, because it launders noise into evidence.

The case that carries the most weight is tie handling. A pair the annotator
could not separate, and a pair the reward itself scored level, are both excluded
from the accuracy — folding either into the numerator would let indecision read
as agreement, and on a 60-pair study that shift is larger than the effect being
measured.

Run:  pytest tests/test_agreement.py -v
"""

from __future__ import annotations

import os
import random
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from reward.agreement import (judge_vs_human, kappa, pairwise_accuracy,  # noqa: E402
                              rater_reliability, reward_vs_human, spearman)


# -- kappa and rho ------------------------------------------------------------

def test_kappa_is_one_for_a_rater_who_agrees_exactly():
    out = kappa([1, 2, 3, 4, 5, 1, 5], [1, 2, 3, 4, 5, 1, 5])
    assert out["value"] == 1.0
    assert out["n"] == 7


def test_kappa_is_near_zero_for_a_coin_flip():
    rng = random.Random(11)
    a = [rng.randint(1, 5) for _ in range(300)]
    b = [rng.randint(1, 5) for _ in range(300)]
    assert abs(kappa(a, b)["value"]) < 0.2


def test_quadratic_weighting_treats_a_near_miss_as_near():
    """5-vs-4 is a smaller disagreement than 5-vs-1; unweighted kappa cannot say so."""
    near = kappa([5, 5, 4, 4, 1, 1, 2, 2], [4, 5, 5, 4, 1, 2, 2, 1])["value"]
    far = kappa([5, 5, 4, 4, 1, 1, 2, 2], [1, 2, 1, 2, 5, 4, 5, 4])["value"]
    assert near > far


def test_kappa_is_undefined_rather_than_zero_when_both_raters_are_constant():
    """
    0/0. Reporting 0.0 would read as "no better than chance" when the truth is
    that the sample says nothing about agreement at all.
    """
    assert kappa([3, 3, 3, 3], [3, 3, 3, 3])["value"] is None


def test_intervals_are_reported_and_reproducible():
    a = kappa([1, 2, 3, 4, 5, 2, 4], [1, 2, 3, 5, 5, 2, 3])
    b = kappa([1, 2, 3, 4, 5, 2, 4], [1, 2, 3, 5, 5, 2, 3])
    assert a["lo"] is not None and a["hi"] is not None
    assert (a["lo"], a["hi"]) == (b["lo"], b["hi"]), "seed is not fixed"


def test_spearman_tracks_rank_not_value():
    assert spearman([1, 2, 3, 4, 5], [10, 20, 30, 40, 50])["value"] == 1.0
    assert spearman([1, 2, 3, 4, 5], [50, 40, 30, 20, 10])["value"] == -1.0


def test_a_sample_too_small_for_an_interval_says_so():
    out = kappa([1, 2], [1, 2])
    assert out["lo"] is None and out["n"] == 2


# -- pairwise accuracy --------------------------------------------------------

def _keymap(spec):
    return {pid: {"stratum": stratum, "higher_reward": higher}
            for pid, stratum, higher in spec}


def test_ties_are_excluded_from_both_numerator_and_denominator():
    """
    10 pairs: 5 agree, 3 disagree, 1 human tie, 1 reward tie.
    Correct answer is 5/8 = 0.625 — not 5/10, and not 7/10 counting ties as hits.
    """
    km = _keymap([("P%d" % i, "random", "left") for i in range(10)])
    prefs = ([{"pair_id": "P%d" % i, "winner": "left"} for i in range(5)]
             + [{"pair_id": "P%d" % i, "winner": "right"} for i in range(5, 8)]
             + [{"pair_id": "P8", "winner": "tie"}])
    km["P9"]["higher_reward"] = "tie"
    prefs.append({"pair_id": "P9", "winner": "left"})

    out = pairwise_accuracy(km, prefs)
    assert out["accuracy"] == 0.625
    assert out["n_scored"] == 8
    assert out["human_ties"] == 1 and out["reward_ties"] == 1


def test_accuracy_is_reported_per_stratum():
    km = _keymap([("P1", "reranked", "left"), ("P2", "reranked", "left"),
                  ("P3", "safety_contrast", "right")])
    prefs = [{"pair_id": "P1", "winner": "left"},
             {"pair_id": "P2", "winner": "right"},
             {"pair_id": "P3", "winner": "right"}]
    out = pairwise_accuracy(km, prefs)
    assert out["by_stratum"]["reranked"]["accuracy"] == 0.5
    assert out["by_stratum"]["safety_contrast"]["accuracy"] == 1.0


def test_a_preference_for_an_unknown_pair_is_ignored():
    out = pairwise_accuracy(_keymap([("P1", "random", "left")]),
                            [{"pair_id": "GHOST", "winner": "left"}])
    assert out["n_scored"] == 0


def test_no_separable_pairs_reports_none_not_zero():
    km = _keymap([("P1", "random", "tie")])
    out = pairwise_accuracy(km, [{"pair_id": "P1", "winner": "left"}])
    assert out["accuracy"] is None


# -- joins against the other instruments --------------------------------------

KEYMAP_ITEMS = {
    "IT-001": {"case_id": "STD-01", "repeat": 0, "mode": "neurosymbolic", "menu_index": 0},
    "IT-002": {"case_id": "STD-02", "repeat": 0, "mode": "no_rag", "menu_index": 0},
    "IT-R01": {"case_id": "STD-01", "repeat": 0, "mode": "neurosymbolic",
               "menu_index": 0, "repeat_of": "IT-001"},
}


def test_repeated_items_do_not_enter_the_judge_correlation_twice():
    """
    A repeat is the same menu. Counting it again would narrow the interval on no
    new information.
    """
    ratings = [{"item_id": "IT-001", "relevance": 5, "faithfulness": 0.9, "naturalness": 4},
               {"item_id": "IT-R01", "relevance": 5, "faithfulness": 0.9, "naturalness": 4},
               {"item_id": "IT-002", "relevance": 2, "faithfulness": 0.1, "naturalness": 3}]
    judge = [{"case_id": "STD-01", "repeat": 0, "mode": "neurosymbolic",
              "relevance": 5, "faithfulness": 0.9, "naturalness": 4},
             {"case_id": "STD-02", "repeat": 0, "mode": "no_rag",
              "relevance": 2, "faithfulness": 0.1, "naturalness": 3}]
    out = judge_vs_human(ratings, KEYMAP_ITEMS, judge)
    assert out["relevance"]["n"] == 2


def test_reward_correlation_is_reported_per_component():
    ratings = [{"item_id": "IT-001", "relevance": 5, "faithfulness": 0.9, "naturalness": 5},
               {"item_id": "IT-002", "relevance": 1, "faithfulness": 0.1, "naturalness": 1}]
    records = [{"case_id": "STD-01", "repeat": 0, "mode": "neurosymbolic", "menu_index": 0,
                "reward": 0.95, "components": {"correctness": {"score": 1.0}}},
               {"case_id": "STD-02", "repeat": 0, "mode": "no_rag", "menu_index": 0,
                "reward": 0.10, "components": {"correctness": {"score": 0.0}}}]
    out = reward_vs_human(ratings, KEYMAP_ITEMS, records, ["correctness"])
    assert "correctness" in out["by_component"]
    assert out["overall"]["n"] == 2


def test_intra_rater_agreement_uses_the_repeats():
    responses = {"alice": [
        {"item_id": "IT-001", "relevance": 5, "naturalness": 4},
        {"item_id": "IT-R01", "relevance": 5, "naturalness": 4},
    ]}
    out = rater_reliability(responses, KEYMAP_ITEMS)
    assert out["intra"]["alice"]["n"] == 2


def test_inter_rater_agreement_uses_only_shared_items():
    responses = {
        "alice": [{"item_id": "IT-001", "relevance": 5, "naturalness": 4},
                  {"item_id": "IT-002", "relevance": 2, "naturalness": 2}],
        "bob": [{"item_id": "IT-001", "relevance": 5, "naturalness": 4}],
    }
    out = rater_reliability(responses, KEYMAP_ITEMS)
    assert out["inter"]["alice vs bob"]["n"] == 2      # one shared item, two scales


def test_a_single_annotator_reports_no_inter_rater_figure():
    out = rater_reliability({"solo": []}, KEYMAP_ITEMS)
    assert out["inter"] == {}
