"""
statistics.py -- uncertainty and significance for the pipeline comparison.

The headline claim is that the symbolic constraint layer reduces allergen
violations. Reporting 0.333 vs 0.000 from a single pass states no uncertainty:
the generator runs at temperature 0.1, so the same case can come out differently
on a second pass, and 30 cases is a small sample.

Two things fix that, and they answer different questions:

  repeat variance   -- run every case N times and report mean +/- SD. Answers
                       "how stable is this number across identical runs?"
  McNemar's test    -- a paired test on the per-case binary safety outcome.
                       Answers "is the difference between two arms bigger than
                       chance, given they were scored on the same cases?"

McNemar is the right test here specifically because the arms are *paired* — every
pipeline sees an identical case list, so an unpaired test (chi-square on two
independent proportions) would discard that pairing and lose power. It looks only
at the cases where the two arms disagree, which is exactly the evidence that one
is safer than the other.

No new dependencies: the exact binomial p-value is computed with math.comb.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

ALL_PIPELINE_MODES = ["no_llm", "neural_rag", "neurosymbolic", "no_rag"]


# ── Per-case outcomes ────────────────────────────────────────────────────────

def case_violated(result_row: Dict[str, Any], mode: str) -> Optional[bool]:
    """
    True if this pipeline recommended a known-unsafe recipe for this case.

    None when the pipeline errored — an errored case has no safety outcome and
    must not be silently counted as safe, which is how a rate-limited run used to
    flatter itself.
    """
    mode_data = result_row.get(mode) or {}
    if mode_data.get("error"):
        return None
    unsafe = set(result_row.get("expected_unsafe_ids") or [])
    final_ids = {m.get("recipe_id") for m in (mode_data.get("final_menus") or [])}
    return bool(final_ids & unsafe)


def outcomes_by_case(results: List[Dict[str, Any]], mode: str) -> Dict[str, bool]:
    """
    Collapse repeats into one binary outcome per case_id.

    A case counts as violated if it violated in *any* repeat. That is the
    conservative reading for a safety claim: a system that leaks an allergen one
    run in five has not demonstrated safety, and majority-voting would hide it.
    """
    per_case: Dict[str, bool] = {}
    for row in results:
        cid = row.get("case_id")
        v = case_violated(row, mode)
        if cid is None or v is None:
            continue
        per_case[cid] = per_case.get(cid, False) or v
    return per_case


# ── McNemar's exact test ─────────────────────────────────────────────────────

def _binom_sf_inclusive(k: int, n: int, p: float = 0.5) -> float:
    """P(X <= k) for X ~ Binomial(n, p)."""
    if n == 0:
        return 1.0
    return sum(math.comb(n, i) * (p ** i) * ((1 - p) ** (n - i)) for i in range(k + 1))


def mcnemar(results: List[Dict[str, Any]], mode_a: str, mode_b: str) -> Dict[str, Any]:
    """
    Paired comparison of two pipelines on the binary "did it recommend something
    unsafe" outcome.

    Returns the discordant counts plus an exact two-sided p-value. The exact
    binomial form is used rather than the chi-square approximation because the
    discordant count here is small (single digits), where chi-square is unreliable.
    """
    a_out = outcomes_by_case(results, mode_a)
    b_out = outcomes_by_case(results, mode_b)
    shared = sorted(set(a_out) & set(b_out))

    # b: A safe, B unsafe   c: A unsafe, B safe
    b = sum(1 for cid in shared if not a_out[cid] and b_out[cid])
    c = sum(1 for cid in shared if a_out[cid] and not b_out[cid])
    n_disc = b + c

    if n_disc == 0:
        p_value: Optional[float] = 1.0
    else:
        p_value = min(1.0, 2.0 * _binom_sf_inclusive(min(b, c), n_disc))

    return {
        "mode_a": mode_a,
        "mode_b": mode_b,
        "n_paired_cases": len(shared),
        "a_safe_b_unsafe": b,
        "a_unsafe_b_safe": c,
        "n_discordant": n_disc,
        # Kept at full-ish precision deliberately: rounding to 5dp turns
        # p=0.001953 into 0.00195 and loses resolution exactly where it matters,
        # near the decision threshold. The report formats for display.
        "p_value": round(p_value, 8) if p_value is not None else None,
        "significant_at_0_05": bool(p_value is not None and p_value < 0.05),
        "a_violations": sum(a_out.values()),
        "b_violations": sum(b_out.values()),
    }


# ── Repeat variance ──────────────────────────────────────────────────────────

def _mean_sd(xs: Sequence[float]) -> Tuple[Optional[float], Optional[float]]:
    if not xs:
        return None, None
    m = sum(xs) / len(xs)
    if len(xs) < 2:
        return round(m, 4), None
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)   # sample SD
    return round(m, 4), round(math.sqrt(var), 4)


def repeat_variance(results: List[Dict[str, Any]], metric_fn) -> Dict[str, Any]:
    """
    Split results by their `repeat` index, compute the metric block for each, and
    report mean +/- SD per mode for the rates that matter.

    metric_fn is passed in (rather than imported) to avoid a circular import with
    evaluator.py, which owns the metric definitions.
    """
    by_repeat: Dict[int, List[Dict[str, Any]]] = {}
    for row in results:
        by_repeat.setdefault(row.get("repeat", 0), []).append(row)

    if len(by_repeat) < 2:
        return {"n_repeats": len(by_repeat),
                "note": "single pass - no variance to report; re-run with --repeats N"}

    tracked = ["allergen_violation_rate", "allergen_violation_rate_over_all_cases",
               "coverage", "safe_and_useful_rate"]
    per_repeat = {rep: metric_fn(rows) for rep, rows in sorted(by_repeat.items())}

    out: Dict[str, Any] = {"n_repeats": len(by_repeat)}
    for mode in ALL_PIPELINE_MODES:
        block: Dict[str, Any] = {}
        for key in tracked:
            vals = [m[mode][key] for m in per_repeat.values()
                    if mode in m and key in m[mode]]
            mean, sd = _mean_sd(vals)
            block[key] = {"mean": mean, "sd": sd, "per_repeat": vals}
        out[mode] = block
    return out
