"""
scoring.py -- Applies the verifiable reward to benchmark results already on disk.

No pipeline re-run and no LLM call. Every input the reward needs is already
persisted by `benchmark/runner.py`: the clean profile, the ranked candidate id
lists from each retrieval stage, the menus the generator proposed, and the
menus that survived the gates. So a reward for a run recorded weeks ago is
computable today, and recomputable by anyone holding the same two files.

The headline output is `best_of_n`: for each case, the reward of the menu the
pipeline actually returned first, against the reward of the menu the reward
model would have promoted. That difference is the policy-improvement step --
best-of-N reranking over menus the generator produced anyway, at zero
additional token cost.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

from .checks import RewardContext
from .model import REWARD_VERSION, RewardResult, load_weights, score_menu

# Arms as the runner records them. Kept local rather than imported from
# benchmark.evaluator so that scoring a results file needs nothing from the
# benchmark package on sys.path.
KNOWN_MODES = ("no_llm", "neural_rag", "neurosymbolic", "no_rag", "reward_ranked")


def context_for(row: Dict[str, Any], mode: str) -> RewardContext:
    """
    Build the scoring context for one arm of one case-run.

    `row["profile"]` is the clean case profile: `run_single_case` applies the
    adversarial injection to a copy held inside the graph state, so what lands
    in the results file was never injected. That is what the reward must see --
    scoring against injected text would let an attacker raise their own reward.
    """
    data = row.get(mode) or {}
    return RewardContext(
        profile=row.get("profile") or {},
        expected_unsafe_ids=row.get("expected_unsafe_ids") or [],
        expected_nutrition_unsafe_ids=row.get("expected_nutrition_unsafe_ids") or [],
        expected_safe_ids=row.get("expected_safe_ids") or [],
        generation_candidates=data.get("generation_candidates") or [],
        reranked_candidates=data.get("reranked_candidates") or [],
        fused_candidates=data.get("fused_candidates") or [],
        mode=mode,
    )


def menus_for(row: Dict[str, Any], mode: str,
              source: str = "proposed") -> Tuple[List[Dict[str, Any]], str]:
    """
    The menus to score, and which field they came from.

    `proposed` measures the model: it is what the generator returned before the
    symbolic post-filter repaired citations or dropped unsafe menus. `final`
    measures the system as shipped.

    `no_llm` never proposes anything -- it selects deterministically straight
    into `final_menus` -- so asking for `proposed` on that arm falls back to
    `final` rather than dropping the baseline out of the comparison entirely.
    """
    data = row.get(mode) or {}
    if source == "final":
        return list(data.get("final_menus") or []), "final"
    proposed = list(data.get("proposed_menus") or [])
    if proposed:
        return proposed, "proposed"
    final = list(data.get("final_menus") or [])
    return final, ("final_fallback" if final else "proposed")


def _record(row: Dict[str, Any], mode: str, index: int, menu: Dict[str, Any],
            result: RewardResult, menu_source: str,
            keep_evidence: bool) -> Dict[str, Any]:
    components: Dict[str, Any] = {}
    for c in result.checks:
        entry: Dict[str, Any] = {"score": c.score, "verifiable": c.verifiable}
        if keep_evidence:
            entry["evidence"] = list(c.evidence)
            entry["detail"] = dict(c.detail)
        components[c.name] = entry
    return {
        "case_id": row.get("case_id"),
        "repeat": row.get("repeat", 0),
        "mode": mode,
        "menu_index": index,
        "menu_source": menu_source,
        "recipe_id": menu.get("recipe_id"),
        "reward": round(result.reward, 6),
        "weighted_score": round(result.weighted_score, 6),
        "gated": result.gated,
        "verifiable_fraction": round(result.verifiable_fraction, 6),
        "components": components,
        "input_digest": result.input_digest,
        # Each record commits to the weights it was scored under, not just the
        # file header. Without this, editing `metadata.weights` after the fact
        # is invisible for any menu whose reward the reweighting cannot move --
        # a menu scoring 1.0 on every applicable check stays at 1.0 under any
        # weighting, and a gated menu stays at 0.0, so the two most common
        # records in a run are exactly the ones a header-only claim cannot
        # protect.
        "weights_digest": result.weights_digest,
    }


def _mean(xs: Iterable[float]) -> Optional[float]:
    xs = list(xs)
    return round(sum(xs) / len(xs), 6) if xs else None


def score_run(payload: Dict[str, Any], *, source: str = "proposed",
              weights: Optional[Dict[str, float]] = None,
              gate_on_correctness: bool = True,
              keep_evidence: bool = True,
              modes: Iterable[str] = KNOWN_MODES) -> Dict[str, Any]:
    """
    Score every menu in a loaded `run_<ts>.json` payload.

    Returns the per-menu records, per-arm aggregates, and the best-of-N
    comparison. Nothing is written; the caller decides where it goes.
    """
    w = load_weights(weights)
    results = payload.get("results") or []
    records: List[Dict[str, Any]] = []
    # (mode, case_id, repeat) -> rewards in generator order, for best-of-N
    by_case: Dict[Tuple[str, str, int], List[float]] = {}

    for row in results:
        for mode in modes:
            data = row.get(mode)
            if not data or data.get("error"):
                continue
            ctx = context_for(row, mode)
            menus, menu_source = menus_for(row, mode, source)
            if not menus:
                continue
            key = (mode, row.get("case_id"), row.get("repeat", 0))
            for i, menu in enumerate(menus):
                res = score_menu(menu, ctx, w, gate_on_correctness)
                records.append(_record(row, mode, i, menu, res, menu_source, keep_evidence))
                by_case.setdefault(key, []).append(res.reward)

    per_mode: Dict[str, Any] = {}
    best_of_n: Dict[str, Any] = {}
    for mode in modes:
        rows = [r for r in records if r["mode"] == mode]
        if not rows:
            continue
        per_mode[mode] = {
            "n_menus": len(rows),
            "mean_reward": _mean(r["reward"] for r in rows),
            "mean_weighted_score": _mean(r["weighted_score"] for r in rows),
            "gated_count": sum(1 for r in rows if r["gated"]),
            "mean_verifiable_fraction": _mean(r["verifiable_fraction"] for r in rows),
            "components": {
                name: {
                    "mean": _mean(r["components"][name]["score"] for r in rows
                                  if r["components"][name]["score"] is not None),
                    "n": sum(1 for r in rows if r["components"][name]["score"] is not None),
                }
                for name in w
            },
        }

        # Best-of-N: what reranking on the reward would have changed.
        cases = {k: v for k, v in by_case.items() if k[0] == mode}
        first = [v[0] for v in cases.values() if v]
        best = [max(v) for v in cases.values() if v]
        improved = sum(1 for v in cases.values() if v and max(v) > v[0] + 1e-9)
        multi = sum(1 for v in cases.values() if len(v) > 1)
        best_of_n[mode] = {
            "n_cases": len(cases),
            "n_cases_with_choice": multi,
            "mean_reward_first_menu": _mean(first),
            "mean_reward_best_menu": _mean(best),
            "delta": (round(sum(best) / len(best) - sum(first) / len(first), 6)
                      if first else None),
            "cases_reranked": improved,
        }

    return {
        "metadata": {
            "reward_version": REWARD_VERSION,
            "weights": w,
            "menu_source": source,
            "gate_on_correctness": gate_on_correctness,
            "evidence_retained": keep_evidence,
            "source_run": (payload.get("metadata") or {}).get("timestamp"),
            "source_git_sha": (payload.get("metadata") or {}).get("git_sha"),
            "source_model": (payload.get("metadata") or {}).get("model"),
            "n_result_rows": len(results),
        },
        "per_mode": per_mode,
        "best_of_n": best_of_n,
        "records": records,
    }
