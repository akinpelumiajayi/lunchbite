"""
model.py -- Aggregates the six verifiable checks into one scalar reward.

The reward is a weighted mean over the checks that apply to a given menu,
renormalised so a sparse profile is not punished for the checks it cannot
answer, and then gated on correctness.

The gate is the part that matters for this project. A menu containing an
allergen the child reacts to scores zero however well it cites, however
fluently it explains, however precisely it quotes the protein figure. Without
the gate a reward-maximising policy can buy a safety violation with five good
paragraphs -- which is precisely the failure mode the neuro-symbolic thesis is
about, reproduced inside the reward function that is supposed to detect it.

Every result carries `reward_version`, a digest of the weights, and a digest of
the exact inputs it was computed from, so `reward.verify` can re-derive it and
prove the number was not hand-tuned after the fact.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

from .checks import ALL_CHECKS, Check, RewardContext

# Bump when any check changes what it returns for the same input. Rewards from
# different versions are not poolable, for the same reason the judge rubric
# carries `_judge_rubric_version`.
REWARD_VERSION = 1

# Correctness carries the largest single share because it is the claim this
# artifact exists to support. Groundedness and citation together outweigh it,
# though, so a policy cannot chase safety alone by refusing to say anything
# substantive. Completeness is deliberately small: it is the easiest dimension
# to game by padding text.
DEFAULT_WEIGHTS: Dict[str, float] = {
    "correctness": 0.35,
    "groundedness": 0.25,
    "citation_accuracy": 0.15,
    "retrieval_accuracy": 0.10,
    "relevance": 0.10,
    "completeness": 0.05,
}


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def _digest(obj: Any) -> str:
    return hashlib.sha256(_canonical(obj).encode("utf-8")).hexdigest()[:16]


def load_weights(overrides: Optional[Mapping[str, float]] = None) -> Dict[str, float]:
    """
    Weights for this scoring run, in precedence order: explicit argument, then
    REWARD_WEIGHTS (a JSON object in the environment), then the defaults.

    Read at call time rather than import time so a test can set the env var
    without reloading the module, matching how `guardrails.nutrition_gate` and
    the rest of this codebase handle configuration.
    """
    weights = dict(DEFAULT_WEIGHTS)
    env = os.environ.get("REWARD_WEIGHTS")
    if env:
        try:
            parsed = json.loads(env)
        except (TypeError, ValueError) as exc:
            raise ValueError("REWARD_WEIGHTS is not valid JSON: %s" % exc) from exc
        unknown = sorted(set(parsed) - set(DEFAULT_WEIGHTS))
        if unknown:
            # Silently ignoring a typo would leave the intended reweighting
            # inert and the run would look like it had been applied.
            raise ValueError("REWARD_WEIGHTS names unknown checks: %s" % ", ".join(unknown))
        weights.update({k: float(v) for k, v in parsed.items()})
    if overrides:
        unknown = sorted(set(overrides) - set(DEFAULT_WEIGHTS))
        if unknown:
            raise ValueError("weight overrides name unknown checks: %s" % ", ".join(unknown))
        weights.update({k: float(v) for k, v in overrides.items()})
    if any(v < 0 for v in weights.values()):
        raise ValueError("reward weights must be non-negative")
    if sum(weights.values()) <= 0:
        raise ValueError("reward weights must not sum to zero")
    return weights


@dataclass
class RewardResult:
    """One scored menu, with everything needed to re-derive the number."""
    reward: float                      # post-gate, [0, 1]
    weighted_score: float              # pre-gate weighted mean over applicable checks
    gated: bool                        # True when correctness zeroed the total
    checks: List[Check] = field(default_factory=list)
    weights: Dict[str, float] = field(default_factory=dict)
    applicable_weight: float = 0.0     # weight mass the score was normalised over
    verifiable_fraction: float = 1.0   # share of that mass derived without a model
    reward_version: int = REWARD_VERSION
    input_digest: str = ""
    weights_digest: str = ""

    def component_scores(self) -> Dict[str, Optional[float]]:
        return {c.name: c.score for c in self.checks}

    def as_dict(self) -> Dict[str, Any]:
        return {
            "reward": round(self.reward, 6),
            "weighted_score": round(self.weighted_score, 6),
            "gated": self.gated,
            "components": {c.name: c.as_dict() for c in self.checks},
            "weights": self.weights,
            "applicable_weight": round(self.applicable_weight, 6),
            "verifiable_fraction": round(self.verifiable_fraction, 6),
            "reward_version": self.reward_version,
            "input_digest": self.input_digest,
            "weights_digest": self.weights_digest,
        }


def context_fingerprint(ctx: RewardContext) -> Dict[str, Any]:
    """The parts of the context a check can actually read, for the input digest."""
    return {
        "profile": ctx.profile,
        "expected_unsafe_ids": list(ctx.expected_unsafe_ids),
        "expected_nutrition_unsafe_ids": list(ctx.expected_nutrition_unsafe_ids),
        "expected_safe_ids": list(ctx.expected_safe_ids),
        "generation_candidates": list(ctx.generation_candidates),
        "reranked_candidates": list(ctx.reranked_candidates),
        "fused_candidates": list(ctx.fused_candidates),
        "mode": ctx.mode,
        "trust_free_text": ctx.trust_free_text,
    }


def score_menu(menu: Dict[str, Any], ctx: RewardContext,
               weights: Optional[Mapping[str, float]] = None,
               gate_on_correctness: bool = True) -> RewardResult:
    """
    Score one generated menu.

    Pure: the same menu and context always produce the same number, with no
    network call and no model. That is what `verify` relies on.
    """
    w = load_weights(weights)
    checks = [fn(menu, ctx) for fn in ALL_CHECKS]

    applicable = [c for c in checks if c.score is not None]
    mass = sum(w[c.name] for c in applicable)
    if mass <= 0:
        # No check applied at all. Reporting 0.0 would be indistinguishable
        # from a menu that failed every check, so this is surfaced instead.
        weighted = 0.0
    else:
        weighted = sum(w[c.name] * float(c.score) for c in applicable) / mass

    verifiable_mass = sum(w[c.name] for c in applicable if c.verifiable)
    verifiable_fraction = (verifiable_mass / mass) if mass > 0 else 1.0

    correctness = next((c for c in checks if c.name == "correctness"), None)
    gated = bool(gate_on_correctness and correctness is not None and correctness.score == 0.0)

    return RewardResult(
        reward=0.0 if gated else weighted,
        weighted_score=weighted,
        gated=gated,
        checks=checks,
        weights=w,
        applicable_weight=mass,
        verifiable_fraction=verifiable_fraction,
        reward_version=REWARD_VERSION,
        input_digest=_digest({"menu": menu, "ctx": context_fingerprint(ctx),
                              "version": REWARD_VERSION}),
        weights_digest=_digest(w),
    )


def rank_menus(menus: List[Dict[str, Any]], ctx: RewardContext,
               weights: Optional[Mapping[str, float]] = None,
               gate_on_correctness: bool = True) -> List[Dict[str, Any]]:
    """
    Best-of-N over menus the generator has already produced.

    This is the policy-improvement step, and it costs no extra tokens: the
    generator is asked for up to three menus per case anyway, so reranking them
    against the reward is free where resampling would need N times the daily
    budget the free tier allows.

    Ties break on the original generator order, so a reward that cannot separate
    two menus leaves the pipeline behaving exactly as it did before.
    """
    scored = [(i, m, score_menu(m, ctx, weights, gate_on_correctness))
              for i, m in enumerate(menus)]
    scored.sort(key=lambda t: (-t[2].reward, t[0]))
    return [{"menu": m, "reward": r, "original_rank": i + 1, "new_rank": n + 1}
            for n, (i, m, r) in enumerate(scored)]
