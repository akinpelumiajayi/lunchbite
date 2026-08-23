"""
statistics.py -- uncertainty and significance for the pipeline comparison.

The headline claim is that the symbolic constraint layer reduces allergen
violations. Reporting 0.333 vs 0.000 from a single pass states no uncertainty:
the generator runs at temperature 0.1, so the same case can come out differently
on a second pass, and 30 cases is a small sample.

Four things fix that, and they answer different questions:

  repeat variance   -- run every case N times and report mean +/- SD. Answers
                       "how stable is this number across identical runs?"
  McNemar's test    -- a paired test on the per-case binary safety outcome.
                       Answers "is the difference between two arms bigger than
                       chance, given they were scored on the same cases?"
  bootstrap CI      -- an interval around a judge-score mean. Answers "how
                       precisely is this mean pinned down by n menus?"
  paired score diff -- the same pairing logic applied to judge scores rather
                       than the binary outcome. Answers "does the symbolic layer
                       change the score for the same child?"

McNemar is the right test here specifically because the arms are *paired* — every
pipeline is put to the same case list, so an unpaired test (chi-square on two
independent proportions) would discard that pairing and lose power. It looks only
at the cases where the two arms disagree, which is exactly the evidence that one
is safer than the other.

Pairing is on the cases both arms actually scored. That is the whole list when a
run completes, and less than it when an arm loses cases to a failed API call —
which is a reason to finish the run rather than to read the p-value as though it
had. `n_paired_cases` reports how many pairs each test really had.

No new dependencies: the exact binomial p-value is computed with math.comb.
"""

from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

ALL_PIPELINE_MODES = ["no_llm", "neural_rag", "neurosymbolic", "no_rag",
                      "reward_ranked"]


# ── Per-case outcomes ────────────────────────────────────────────────────────

def run_failed(mode_data: Dict[str, Any], mode: str) -> bool:
    """
    True when this case-run did not produce a usable answer for reasons that are
    the harness's fault rather than the pipeline's decision.

    The distinction matters more than it looks. An empty `final_menus` is
    ambiguous: it is what a correct refusal looks like *and* what a
    rate-limited API call looks like. Scoring the second as the first is how a
    dead arm reports a perfect safety record — the violation rate divides by
    the cases a pipeline answered, so an arm that answered nothing scores 0.000.

    That is not hypothetical. In run 20260816_182449 the generator hit its daily
    token cap during repeat 3; repeats 4 and 5 produced zero menus across all 30
    cases, and `neural_rag` duly reported violation rates of [0.367, 0.367,
    0.625, 0.000, 0.000]. Two-fifths of the published mean came from an arm that
    was switched off. The guard existed but tested `error`, which runner.py sets
    only when the *graph* raises; an LLM failure is caught inside the generate
    node and surfaces as `generation_error`, so it never fired.

    The classification is structural rather than string-matched:
      * `error`            -- the graph itself raised. Always a failure.
      * `generation_error` with no generation candidates, in a retrieval mode --
        the symbolic pre-filter removed everything, which is the system
        deliberately declining to answer. That is a *result*, not a failure.
      * `generation_error` otherwise -- the model was called and the call or its
        output failed. A failure.

    `no_rag` is excluded from the middle case because it has no candidates by
    design, so emptiness there carries no information.
    """
    if mode_data.get("error"):
        return True
    if not mode_data.get("generation_error"):
        return False
    if mode != "no_rag" and not (mode_data.get("generation_candidates") or []):
        return False
    return True


def case_violated(result_row: Dict[str, Any], mode: str) -> Optional[bool]:
    """
    True if this pipeline recommended a known-unsafe recipe for this case.

    None when the pipeline errored — an errored case has no safety outcome and
    must not be silently counted as safe, which is how a rate-limited run used to
    flatter itself.

    The test is `run_failed`, not a bare `error` check. This function tested
    `error` alone until it was found scoring the 2026-08-18 outage as thirteen
    safe refusals: the runner sets `error` only when the *graph* raises, and a
    dead LLM call is caught inside the generate node and surfaces as
    `generation_error`. compute_safety_metrics had already been fixed; this had
    not, so the safety table excluded those runs while the McNemar test that sits
    directly beneath it counted them as evidence of safety.
    """
    mode_data = result_row.get(mode) or {}
    if run_failed(mode_data, mode):
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


def min_detectable_discordant(alpha: float = 0.05) -> int:
    """
    Fewest one-way discordant cases that could ever reach significance.

    Why a null result needs this printed beside it
    ----------------------------------------------
    McNemar's exact test looks only at the cases where the two arms disagree.
    With every disagreement pointing the same way, the two-sided p-value is
    2 * 0.5^b for b discordant cases, so b=5 gives 0.0625 and b=6 gives 0.031:
    **below six, no result is attainable at alpha=0.05 however real the effect.**

    Run 20260819_222156 produced exactly one discordant case and reported
    p = 1.0000. Read alone that looks like evidence of no difference. It is not:
    it is a design that could not have detected one. Reporting the floor turns
    "no difference found" into the honest "this could not have found less than X".
    """
    b = 1
    while 2 * (0.5 ** b) > alpha:
        b += 1
    return b


def power_note(n_paired: int, alpha: float = 0.05) -> Dict[str, Any]:
    """The floor above expressed against a specific benchmark size."""
    b = min_detectable_discordant(alpha)
    return {
        "min_discordant_for_significance": b,
        "n_paired_cases": n_paired,
        "min_detectable_gap": round(b / n_paired, 3) if n_paired else None,
        "alpha": alpha,
    }


def mcnemar(results: List[Dict[str, Any]], mode_a: str, mode_b: str) -> Dict[str, Any]:
    """
    Paired comparison of two pipelines on the binary "did it recommend something
    unsafe" outcome.

    Returns the discordant counts plus an exact two-sided p-value. The exact
    binomial form is used rather than the chi-square approximation because the
    discordant count here is small (single digits), where chi-square is unreliable.

    Pairing is on the *intersection* of the two arms' scored cases. When a run
    completes, that intersection is every case and the distinction is idle; when
    one arm loses cases to a failed API call it is not, and on 2026-08-18 the
    three LLM arms finished with 17, 18 and 17 scored cases. `a_violations` and
    `b_violations` count over each arm's own case set — the numbers quoted
    elsewhere in the report — while `*_violations_paired` count over the shared
    set the test actually used. Callers that display the two side by side should
    quote the paired pair, or the reader sees a p-value computed from counts
    that are not the ones printed next to it.
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
        "a_violations_paired": sum(1 for cid in shared if a_out[cid]),
        "b_violations_paired": sum(1 for cid in shared if b_out[cid]),
        # How much of the run this test actually rests on. Equal to the case
        # count only when both arms completed every case.
        "n_cases_a": len(a_out),
        "n_cases_b": len(b_out),
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


def bootstrap_ci(xs: Sequence[float], confidence: float = 0.95,
                 n_boot: int = 2000, seed: int = 20260815) -> Dict[str, Any]:
    """
    Percentile bootstrap CI for a mean.

    Bootstrap rather than a t-interval because these are 1-5 ordinal judge scores
    and 0-1 faithfulness ratios over n=20-30 — bounded, discrete, and visibly
    non-normal, which is exactly where the normal approximation puts a CI bound
    outside the scale's own range. The seed is fixed so the reported interval is
    reproducible; no scipy/numpy needed.
    """
    xs = list(xs)
    n = len(xs)
    if n == 0:
        return {"mean": None, "lo": None, "hi": None, "n": 0}
    mean = sum(xs) / n
    if n < 3:
        # An interval from one or two points is not information.
        return {"mean": round(mean, 3), "lo": None, "hi": None, "n": n,
                "note": "n<3, no interval"}

    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        means.append(sum(xs[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    alpha = (1.0 - confidence) / 2.0
    lo = means[int(alpha * n_boot)]
    hi = means[min(n_boot - 1, int((1.0 - alpha) * n_boot))]
    return {"mean": round(mean, 3), "lo": round(lo, 3), "hi": round(hi, 3), "n": n}


def paired_score_diff(records: List[Dict[str, Any]], mode_a: str, mode_b: str,
                      metric: str) -> Dict[str, Any]:
    """
    Paired difference (a - b) in a judge metric, over cases both arms scored.

    Comparing two independent means answers "are these two numbers different?".
    Pairing on case_id answers the question actually being asked — "does swapping
    the symbolic layer in change the score *for the same child*?" — and removes
    between-case variance, which here is the dominant term: some profiles are
    simply easier to serve well than others.

    A CI spanning 0 is the evidence for "no meaningful difference". That is a
    real result and much stronger than the report's previous rule of thumb, which
    declared no degradation whenever two unpaired means fell within 0.2.
    """
    def by_case(mode: str) -> Dict[Tuple[Any, Any], float]:
        out: Dict[Tuple[Any, Any], float] = {}
        for rec in records:
            if rec.get("mode") != mode:
                continue
            v = rec.get(metric)
            if v is None:
                continue
            out[(rec.get("case_id"), rec.get("repeat"))] = float(v)
        return out

    a, b = by_case(mode_a), by_case(mode_b)
    shared = sorted(set(a) & set(b), key=lambda k: (str(k[0]), str(k[1])))
    diffs = [a[k] - b[k] for k in shared]
    ci = bootstrap_ci(diffs)
    return {
        "mode_a": mode_a, "mode_b": mode_b, "metric": metric,
        "n_pairs": len(diffs),
        "mean_diff": ci["mean"], "lo": ci["lo"], "hi": ci["hi"],
        # None (rather than False) when there are too few pairs to bound the
        # difference at all — "not shown to differ" and "shown not to differ"
        # are different findings and must not collapse into one flag.
        "excludes_zero": (None if ci["lo"] is None
                          else bool(ci["lo"] > 0 or ci["hi"] < 0)),
    }


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
        # A repeat in which every run of this mode failed contributes no
        # evidence about the mode, only about the API. Its rates are all 0.0 by
        # construction (0 violations over an empty denominator), and averaging
        # those in pulls the mean toward zero — i.e. makes an arm look *safer*
        # the more of it died. Run 20260816_182449 lost repeats 4 and 5 of
        # neural_rag to the daily token cap and reported violations of
        # [0.367, 0.367, 0.625, 0.000, 0.000], mean 0.272, on that basis.
        present = {rep: m for rep, m in per_repeat.items() if mode in m}
        # Default 1, not 0: a metric_fn that does not report cases_evaluated is
        # unknown, not empty, and must not have every repeat silently discarded.
        scored = {rep: m for rep, m in present.items()
                  if m[mode].get("cases_evaluated", 1) > 0}
        dropped = sorted(set(present) - set(scored))
        for key in tracked:
            vals = [m[mode][key] for m in scored.values() if key in m[mode]]
            mean, sd = _mean_sd(vals)
            block[key] = {"mean": mean, "sd": sd, "per_repeat": vals}
        block["n_repeats_scored"] = len(scored)
        if dropped:
            block["repeats_dropped_all_runs_failed"] = dropped
        out[mode] = block
    return out
