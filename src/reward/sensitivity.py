"""
sensitivity.py -- Is the reward conclusion an artefact of the weights?

The six weights in `model.DEFAULT_WEIGHTS` were chosen by argument, not fitted
to data. That is the most attackable line in the reward: "why 0.35 for
correctness?" has no answer in the code beyond prose. Fitting them would need a
few hundred human preference pairs that do not exist.

This module answers the question a different way. Rather than defend one
weighting, it re-aggregates the same component scores under several deliberately
different ones -- including a hostile weighting that almost ignores safety --
and reports whether the ordering of the arms changes. If it does not, the
conclusion does not rest on the weights, and their exact values stop mattering.

That is a stronger claim than a fitted number would earn. A Bradley-Terry fit on
200 pairs says "these weights are what annotators implied"; this says "the
result holds whatever you weight it".

Nothing here recomputes a check. Component scores are a property of the menu and
the corpus and do not depend on weights at all, so re-aggregation reads the
scores already recorded and redoes only the arithmetic -- which is itself the
point being demonstrated.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from .model import DEFAULT_WEIGHTS, load_weights

# Weightings to test, each chosen to stress a different assumption.
#
# `hostile` is the important one. It weights correctness at a seventeenth of the
# default, which is as close as this can get to asking "what if safety barely
# counted?" without removing the gate. A conclusion that survives it is not
# resting on the weighting.
WEIGHTINGS: Dict[str, Dict[str, float]] = {
    "default": dict(DEFAULT_WEIGHTS),
    "equal": {k: 1.0 for k in DEFAULT_WEIGHTS},
    "safety_heavy": {**DEFAULT_WEIGHTS, "correctness": 0.60},
    "grounding_heavy": {**DEFAULT_WEIGHTS, "groundedness": 0.50},
    "traceability_heavy": {**DEFAULT_WEIGHTS, "citation_accuracy": 0.45,
                           "retrieval_accuracy": 0.25},
    "hostile": {**DEFAULT_WEIGHTS, "correctness": 0.02},
}


def reaggregate(record: Dict[str, Any], weights: Mapping[str, float],
                gate_on_correctness: bool = True) -> Optional[float]:
    """
    Recompute one reward from component scores already on disk.

    Returns None when no component applied, matching how `score_menu` treats an
    empty weight mass rather than reporting a 0.0 that would be
    indistinguishable from a menu which failed everything.
    """
    components = record.get("components") or {}

    if gate_on_correctness:
        # Checked before the weighted mean, not after. The gate is a veto, not a
        # term -- that is what stops a reward-maximising policy buying a safety
        # violation with fluent prose, and it must not be reachable by lowering
        # the weight on correctness.
        if (components.get("correctness") or {}).get("score") == 0.0:
            return 0.0

    applicable = [(name, c["score"]) for name, c in components.items()
                  if c.get("score") is not None and name in weights]
    mass = sum(weights[n] for n, _ in applicable)
    if mass <= 0:
        return None
    return sum(weights[n] * float(s) for n, s in applicable) / mass


def _mean(xs: Iterable[Optional[float]]) -> Optional[float]:
    vals = [x for x in xs if x is not None]
    return round(sum(vals) / len(vals), 6) if vals else None


def _ranking(means: Mapping[str, Optional[float]]) -> List[str]:
    """Arms best-first. An arm with no score is omitted rather than ranked last."""
    scored = [(m, v) for m, v in means.items() if v is not None]
    scored.sort(key=lambda t: (-t[1], t[0]))
    return [m for m, _ in scored]


def sensitivity(scored: Dict[str, Any],
                weightings: Optional[Mapping[str, Mapping[str, float]]] = None,
                gate_on_correctness: bool = True) -> Dict[str, Any]:
    """
    Re-aggregate a scored run under each weighting and compare the orderings.

    `scored` is the payload written by `reward.scoring.score_run`. It takes that
    rather than a results file because recorded component scores are all this
    needs, and re-deriving them from the corpus would obscure the point that the
    weighting changes nothing except the arithmetic.
    """
    weightings = weightings or WEIGHTINGS
    records = scored.get("records") or []

    modes: List[str] = []
    for r in records:
        if r["mode"] not in modes:
            modes.append(r["mode"])

    per_weighting: Dict[str, Any] = {}
    rankings: Dict[str, List[str]] = {}

    for label, raw in weightings.items():
        # Normalised through load_weights so an unknown or negative weight is
        # refused here exactly as it would be in a real scoring run.
        w = load_weights(raw)
        means = {m: _mean(reaggregate(r, w, gate_on_correctness)
                          for r in records if r["mode"] == m)
                 for m in modes}
        ranking = _ranking(means)
        rankings[label] = ranking
        per_weighting[label] = {"weights": w, "mean_reward": means, "ranking": ranking}

    orderings = sorted({tuple(r) for r in rankings.values()})

    # Where each arm landed across every weighting. An arm holding one position
    # throughout is unaffected by the weighting; a spread names the arms whose
    # rank the choice of weights actually decides, which is the honest place to
    # put a caveat.
    rank_range: Dict[str, Dict[str, int]] = {}
    for m in modes:
        positions = [r.index(m) + 1 for r in rankings.values() if m in r]
        if positions:
            rank_range[m] = {"best": min(positions), "worst": max(positions)}

    return {
        "weightings_tested": list(weightings),
        "gate_on_correctness": gate_on_correctness,
        "ordering_stable": len(orderings) == 1,
        "distinct_orderings": [list(o) for o in orderings],
        "rank_range": rank_range,
        "per_weighting": per_weighting,
    }


def verify_reaggregation(scored: Dict[str, Any]) -> Tuple[bool, float]:
    """
    Re-aggregating under the weights a run was scored with must reproduce it.

    `reaggregate` is a second implementation of what `score_menu` already does,
    and two implementations that disagree would make every sensitivity number
    below meaningless. Returns (agrees, largest absolute difference seen).
    """
    meta = scored.get("metadata") or {}
    weights = meta.get("weights") or DEFAULT_WEIGHTS
    gate = meta.get("gate_on_correctness", True)

    worst = 0.0
    for r in scored.get("records") or []:
        again = reaggregate(r, weights, gate)
        if again is None:
            continue
        worst = max(worst, abs(again - float(r["reward"])))
    return worst <= 1e-6, worst
