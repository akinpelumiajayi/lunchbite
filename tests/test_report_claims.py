"""
Unit tests for the §3.1 quality verdict in report/generate_report.py.

The report is the artifact the research claims are read from, so a sentence it
prints is as much a result as a number in a table. The regression guarded here
is a real one: the verdict was gated on relevance alone, so run 20260816_160852
published "the symbolic constraint layer does **not** degrade recommendation
quality" directly beneath its own table showing faithfulness at
-0.260 [-0.420, -0.100] and naturalness at -0.600 [-1.040, -0.120].

Run:  pytest tests/test_report_claims.py -v
"""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "report"))

from generate_report import quality_verdict  # noqa: E402

NO_DEGRADE = "does **not** degrade"


def metric(mean_diff, lo, hi, n_pairs=25):
    """One entry of the `_paired` block, in the shape stats.paired_score_diff emits."""
    return {"n_pairs": n_pairs, "mean_diff": mean_diff, "lo": lo, "hi": hi,
            "excludes_zero": (None if lo is None else bool(lo > 0 or hi < 0))}


def paired(**metrics):
    return {f"neurosymbolic_vs_neural_rag::{k}": v for k, v in metrics.items()}


def test_all_intervals_span_zero_claims_no_degradation():
    v = quality_verdict(paired(
        relevance=metric(-0.20, -0.80, 0.40),
        faithfulness=metric(-0.02, -0.15, 0.11),
        naturalness=metric(0.10, -0.30, 0.50),
    ))
    assert NO_DEGRADE in v
    assert "measurably lower" not in v


def test_run_20260816_shape_does_not_claim_no_degradation():
    """The exact regression: relevance flat, the other two significantly worse."""
    v = quality_verdict(paired(
        relevance=metric(-0.200, -0.800, 0.400),
        faithfulness=metric(-0.260, -0.420, -0.100),
        naturalness=metric(-0.600, -1.040, -0.120),
    ))
    assert NO_DEGRADE not in v
    assert "measurably lower" in v
    # Both degraded metrics are named, and the one that held is not miscounted.
    assert "faithfulness" in v.lower() and "naturalness" in v.lower()
    assert "interval for relevance spans zero" in v


def test_single_degraded_metric_uses_singular_verb():
    v = quality_verdict(paired(
        relevance=metric(-0.10, -0.50, 0.30),
        faithfulness=metric(-0.26, -0.42, -0.10),
        naturalness=metric(0.05, -0.20, 0.40),
    ))
    assert "**Faithfulness is measurably lower" in v
    assert NO_DEGRADE not in v


def test_improvement_is_not_reported_as_degradation():
    v = quality_verdict(paired(
        relevance=metric(0.40, 0.10, 0.70),
        faithfulness=metric(0.30, 0.05, 0.55),
        naturalness=metric(0.20, 0.02, 0.38),
    ))
    assert "higher under the symbolic layer" in v
    assert "measurably lower" not in v


def test_mixed_directions_lead_with_the_loss():
    v = quality_verdict(paired(
        relevance=metric(0.50, 0.20, 0.80),
        faithfulness=metric(-0.30, -0.50, -0.10),
        naturalness=metric(0.01, -0.30, 0.32),
    ))
    assert "**Faithfulness is measurably lower" in v
    assert "Relevance is higher" in v
    assert NO_DEGRADE not in v


def test_unbounded_metric_is_excluded_not_treated_as_agreement():
    """
    lo is None means the sample was too small to bootstrap. "Not shown to
    differ" is not "shown not to differ", so an unbounded metric must not be
    counted towards a no-degradation claim.
    """
    v = quality_verdict(paired(
        relevance=metric(-0.20, None, None, n_pairs=2),
        faithfulness=metric(-0.26, -0.42, -0.10),
        naturalness=metric(-0.60, -1.04, -0.12),
    ))
    assert NO_DEGRADE not in v
    assert "relevance" not in v.lower().split("faithfulness")[0]


def test_no_bounded_metrics_draws_no_verdict():
    v = quality_verdict(paired(
        relevance=metric(-0.20, None, None, n_pairs=1),
        faithfulness=metric(-0.26, None, None, n_pairs=1),
        naturalness=metric(-0.60, None, None, n_pairs=1),
    ))
    assert "No verdict is drawn" in v


def test_empty_paired_block_draws_no_verdict():
    assert "No verdict is drawn" in quality_verdict({})
