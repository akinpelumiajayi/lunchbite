"""
agreement.py -- Does the reward, or the judge, match a human?

Three questions, three measures, and they are not interchangeable:

  judge vs human    quadratic-weighted Cohen's kappa on the ordinal 1-5 scales.
                    Quadratic because 5-vs-4 is a smaller disagreement than
                    5-vs-1, and unweighted kappa treats them alike.
  reward vs human   PAIRWISE ACCURACY: of the pairs a person could separate, how
                    often did the higher-reward menu win. This is the headline
                    for a reward model, and the only figure that breaks the
                    circularity in the best-of-N result -- there the reward both
                    chose the menu and scored the improvement.
  human vs human    inter-rater kappa on the overlap, intra-rater on the repeats.
                    Without these the "human standard" is one person's taste.

Ties are reported, never dropped into either column. "The annotator could not
separate them" is a real answer about a reward asserting they differ, and
counting a tie as agreement would inflate every accuracy here.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / "benchmark"))

from stats import bootstrap_ci  # noqa: E402

# Scales the study collects, and how each is compared against the judge.
ORDINAL_SCALES = ("relevance", "naturalness")   # 1-5, kappa
CONTINUOUS_SCALES = ("faithfulness",)           # 0-1, Spearman

_SEED = 20260822


def _bootstrap_statistic(pairs: Sequence[Tuple[Any, Any]],
                         stat: Callable[[List[Any], List[Any]], Optional[float]],
                         confidence: float = 0.95, n_boot: int = 2000,
                         seed: int = _SEED) -> Dict[str, Any]:
    """
    Percentile interval for a paired statistic that is not a mean.

    `stats.bootstrap_ci` is deliberately mean-only, so kappa and rho need this
    rather than a second copy of it. Same fixed-seed convention, so a reported
    interval is reproducible.
    """
    pairs = [p for p in pairs if p[0] is not None and p[1] is not None]
    n = len(pairs)
    point = stat([p[0] for p in pairs], [p[1] for p in pairs]) if n else None
    if n < 3 or point is None:
        return {"value": None if point is None else round(point, 3),
                "lo": None, "hi": None, "n": n}

    rng = random.Random(seed)
    draws = []
    for _ in range(n_boot):
        sample = [pairs[rng.randrange(n)] for _ in range(n)]
        v = stat([p[0] for p in sample], [p[1] for p in sample])
        if v is not None and v == v:                 # drop NaN from degenerate draws
            draws.append(v)
    if not draws:
        return {"value": round(point, 3), "lo": None, "hi": None, "n": n}
    draws.sort()
    lo = draws[int((1 - confidence) / 2 * len(draws))]
    hi = draws[min(len(draws) - 1, int((1 + confidence) / 2 * len(draws)))]
    return {"value": round(point, 3), "lo": round(lo, 3), "hi": round(hi, 3), "n": n}


def _kappa(a: List[Any], b: List[Any]) -> Optional[float]:
    from sklearn.metrics import cohen_kappa_score
    if len(set(a)) < 2 and len(set(b)) < 2:
        # Both raters constant: kappa is undefined (0/0), not zero. Returning
        # 0.0 would read as "no better than chance" when the truth is that the
        # sample carries no information about agreement at all.
        return None
    return float(cohen_kappa_score(a, b, weights="quadratic"))


def _spearman(a: List[Any], b: List[Any]) -> Optional[float]:
    from scipy.stats import spearmanr
    if len(a) < 3 or len(set(a)) < 2 or len(set(b)) < 2:
        return None
    rho = spearmanr(a, b).statistic
    return None if rho != rho else float(rho)


def kappa(a: Sequence[Any], b: Sequence[Any]) -> Dict[str, Any]:
    """Quadratic-weighted Cohen's kappa with a bootstrap interval."""
    return _bootstrap_statistic(list(zip(a, b)), _kappa)


def spearman(a: Sequence[Any], b: Sequence[Any]) -> Dict[str, Any]:
    """Spearman rho with a bootstrap interval."""
    return _bootstrap_statistic(list(zip(a, b)), _spearman)


def pairwise_accuracy(keymap_pairs: Dict[str, Dict[str, Any]],
                      preferences: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """
    How often the higher-reward menu is the one the annotator chose.

    Denominator is pairs BOTH sides could separate: a human "no preference", and
    a pair the reward itself scored level, are excluded and counted separately.
    Folding either into the numerator would let indecision look like agreement.
    """
    agree = disagree = human_tie = reward_tie = 0
    per_stratum: Dict[str, List[float]] = {}
    indicators: List[float] = []

    for pref in preferences:
        km = keymap_pairs.get(pref.get("pair_id"))
        if km is None:
            continue
        choice = pref.get("winner")
        if choice == "tie":
            human_tie += 1
            continue
        if km["higher_reward"] == "tie":
            reward_tie += 1
            continue
        hit = 1.0 if choice == km["higher_reward"] else 0.0
        indicators.append(hit)
        per_stratum.setdefault(km["stratum"], []).append(hit)
        if hit:
            agree += 1
        else:
            disagree += 1

    scored = agree + disagree
    out = {
        "accuracy": round(agree / scored, 3) if scored else None,
        "n_scored": scored, "agree": agree, "disagree": disagree,
        "human_ties": human_tie, "reward_ties": reward_tie,
        "ci": bootstrap_ci(indicators) if indicators else
              {"mean": None, "lo": None, "hi": None, "n": 0},
        "by_stratum": {},
    }
    for s, hits in sorted(per_stratum.items()):
        out["by_stratum"][s] = {
            "accuracy": round(sum(hits) / len(hits), 3),
            "n": len(hits),
            "ci": bootstrap_ci(hits),
        }
    return out


def judge_vs_human(ratings: Sequence[Dict[str, Any]],
                   keymap_items: Dict[str, Dict[str, Any]],
                   judge_records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Agreement between the LLM judge and a human on the same anchored scales.

    This is what `plan.md` §4.3 has been owed since before the reward existed:
    `evaluator._RUBRIC` was anchored per scale point precisely so a human could
    apply the identical definitions, and the study was never run.

    Repeats are excluded -- rating the same menu twice would enter it into the
    correlation twice and narrow the interval on no new information.
    """
    judge = {(r["case_id"], r.get("repeat", 0), r["mode"]): r
             for r in judge_records if not r.get("error")}
    out: Dict[str, Any] = {}

    for scale in ORDINAL_SCALES + CONTINUOUS_SCALES:
        human_vals, judge_vals = [], []
        for rating in ratings:
            km = keymap_items.get(rating.get("item_id"))
            if km is None or km.get("repeat_of"):
                continue
            j = judge.get((km["case_id"], km["repeat"], km["mode"]))
            if not j or j.get(scale) is None or rating.get(scale) is None:
                continue
            human_vals.append(rating[scale])
            judge_vals.append(j[scale])

        if scale in ORDINAL_SCALES:
            out[scale] = {"measure": "cohen_kappa_quadratic",
                          **kappa([int(round(v)) for v in human_vals],
                                  [int(round(v)) for v in judge_vals])}
        else:
            # Faithfulness is a continuous ratio, so kappa would need an
            # arbitrary binning that the rubric does not define.
            out[scale] = {"measure": "spearman_rho",
                          **spearman(human_vals, judge_vals)}
    return out


def reward_vs_human(ratings: Sequence[Dict[str, Any]],
                    keymap_items: Dict[str, Dict[str, Any]],
                    reward_records: Sequence[Dict[str, Any]],
                    components: Sequence[str] = ()) -> Dict[str, Any]:
    """
    Correlation between the reward and a human rating, overall and per component.

    Per component matters because they do not all need validating. `correctness`,
    `citation_accuracy` and `retrieval_accuracy` are corpus facts -- a human can
    audit them but cannot meaningfully disagree -- and they carry 0.60 of the
    weight. Reporting only an aggregate lets those mask `relevance`, which is a
    token-overlap proxy and the component most likely to be wrong.

    The human side is the mean of the two 1-5 scales, normalised to 0-1, since
    the reward is a single scalar and no single rated scale is its counterpart.
    """
    by_key = {(r["case_id"], r.get("repeat", 0), r["mode"], r.get("menu_index", 0)): r
              for r in reward_records}

    human, reward, per_comp = [], [], {c: ([], []) for c in components}
    for rating in ratings:
        km = keymap_items.get(rating.get("item_id"))
        if km is None or km.get("repeat_of"):
            continue
        rec = by_key.get((km["case_id"], km["repeat"], km["mode"], km["menu_index"]))
        if rec is None:
            continue
        scales = [rating.get(s) for s in ORDINAL_SCALES]
        if any(v is None for v in scales):
            continue
        h = (sum(scales) / len(scales) - 1.0) / 4.0        # 1-5 -> 0-1
        human.append(h)
        reward.append(rec["reward"])
        for c in components:
            score = (rec.get("components", {}).get(c) or {}).get("score")
            if score is not None:
                per_comp[c][0].append(h)
                per_comp[c][1].append(score)

    return {
        "overall": {"measure": "spearman_rho", **spearman(human, reward)},
        "by_component": {c: {"measure": "spearman_rho", **spearman(h, s)}
                         for c, (h, s) in per_comp.items()},
    }


def rater_reliability(responses_by_annotator: Dict[str, List[Dict[str, Any]]],
                      keymap_items: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Is the human standard stable enough to validate anything against?

    intra: an annotator against themselves on the repeated items.
    inter: two annotators against each other on the items both rated.
    """
    out: Dict[str, Any] = {"intra": {}, "inter": {}}

    for annotator, rows in responses_by_annotator.items():
        by_item = {r["item_id"]: r for r in rows}
        first, second = [], []
        for item_id, rating in by_item.items():
            origin = (keymap_items.get(item_id) or {}).get("repeat_of")
            if origin and origin in by_item:
                for scale in ORDINAL_SCALES:
                    a, b = by_item[origin].get(scale), rating.get(scale)
                    if a is not None and b is not None:
                        first.append(int(round(a)))
                        second.append(int(round(b)))
        out["intra"][annotator] = {"measure": "cohen_kappa_quadratic",
                                   **kappa(first, second)}

    names = sorted(responses_by_annotator)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            ra = {r["item_id"]: r for r in responses_by_annotator[a]}
            rb = {r["item_id"]: r for r in responses_by_annotator[b]}
            xs, ys = [], []
            for item_id in sorted(set(ra) & set(rb)):
                for scale in ORDINAL_SCALES:
                    va, vb = ra[item_id].get(scale), rb[item_id].get(scale)
                    if va is not None and vb is not None:
                        xs.append(int(round(va)))
                        ys.append(int(round(vb)))
            out["inter"][f"{a} vs {b}"] = {"measure": "cohen_kappa_quadratic",
                                           **kappa(xs, ys)}
    return out
