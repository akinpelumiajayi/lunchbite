"""
reward -- A verifiable reward model for generated lunch menus.

This package supplies the reward signal an RLHF-style loop needs, built so that
every number it produces can be re-derived by a third party from files already
on disk. Nothing here calls a language model.

  checks    the six components: correctness, groundedness, completeness,
            relevance, citation accuracy, retrieval accuracy
  model     weighted aggregation into one scalar, gated on correctness
  scoring   applies the reward to a benchmark results file, offline
  verify    recomputes a scored file and proves the numbers hold
  sensitivity  re-aggregates under several weightings and reports whether the
            conclusion depends on the weights at all

Why verifiable rather than learned. The obvious way to build a reward model is
to fit one on human preference judgements. That is defensible, but it produces
a number no examiner can check: disagree with it and there is nothing to
inspect but weights. Every component here instead resolves to a fact in
data/recipes.json or a hand-labelled field on the benchmark case, and every
score ships with the evidence it was derived from.

That also makes the reward robust to the attack this project studies. A reward
learned from text can be moved by text -- flattering prose, an injected
instruction, a confident tone. A reward that reads `allergens_present` from the
corpus cannot be argued with.
"""

from .checks import (ALL_CHECKS, Check, RewardContext, check_citation,
                     check_completeness, check_correctness, check_groundedness,
                     check_relevance, check_retrieval)
from .model import (DEFAULT_WEIGHTS, REWARD_VERSION, RewardResult, load_weights,
                    rank_menus, score_menu)
from .scoring import context_for, menus_for, score_run
from .sensitivity import (WEIGHTINGS, reaggregate, sensitivity,
                          verify_reaggregation)
from .verify import VerificationReport, load_and_verify, verify_determinism, verify_scored_run

__all__ = [
    "ALL_CHECKS", "Check", "RewardContext",
    "check_correctness", "check_groundedness", "check_completeness",
    "check_relevance", "check_citation", "check_retrieval",
    "DEFAULT_WEIGHTS", "REWARD_VERSION", "RewardResult", "load_weights",
    "score_menu", "rank_menus",
    "context_for", "menus_for", "score_run",
    "WEIGHTINGS", "sensitivity", "reaggregate", "verify_reaggregation",
    "VerificationReport", "verify_scored_run", "verify_determinism", "load_and_verify",
]
