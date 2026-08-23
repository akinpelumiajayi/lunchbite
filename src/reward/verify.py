"""
verify.py -- Proves a recorded reward is the reward the checks actually produce.

A reward is only verifiable if someone other than its author can re-derive it.
This module is that someone. Given a scored-run file and the results file it
was computed from, it recomputes every record from scratch and reports any
number that does not come back identical.

Three distinct properties are checked, because they fail for different reasons:

  reproducible  recomputing from the same inputs gives the same reward
  deterministic scoring the same menu twice in one process agrees with itself
  model-free    every component was derived without calling a model

The third is what separates this from the LLM judge in `benchmark/evaluator.py`.
A judge score cannot be re-derived -- rerun it and the number moves, and at
temperature 0 the gpt-oss judge was already observed to vary. A verifiable
reward that quietly grew a model-backed component would lose the only property
that makes it worth having, so the fraction is asserted rather than assumed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .model import REWARD_VERSION, score_menu
from .scoring import context_for, menus_for

# Rewards are rounded to 6 dp on the way to disk, so equality is checked at the
# precision that was actually recorded rather than at float identity.
TOLERANCE = 1e-6


@dataclass
class Mismatch:
    case_id: str
    mode: str
    menu_index: int
    field: str
    recorded: Any
    recomputed: Any

    def __str__(self) -> str:
        return ("%s/%s menu %d: %s recorded %r, recomputed %r"
                % (self.case_id, self.mode, self.menu_index, self.field,
                   self.recorded, self.recomputed))


@dataclass
class VerificationReport:
    n_records: int = 0
    n_checked: int = 0
    mismatches: List[Mismatch] = field(default_factory=list)
    version_recorded: Optional[int] = None
    version_current: int = REWARD_VERSION
    non_verifiable_components: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    @property
    def reproducible(self) -> bool:
        return not self.mismatches

    @property
    def model_free(self) -> bool:
        return not self.non_verifiable_components

    @property
    def version_matches(self) -> bool:
        return self.version_recorded == self.version_current

    @property
    def passed(self) -> bool:
        return self.reproducible and self.model_free and self.version_matches

    def summary(self) -> str:
        lines = [
            "records checked      : %d of %d" % (self.n_checked, self.n_records),
            "reproducible         : %s" % ("yes" if self.reproducible else
                                           "NO (%d mismatches)" % len(self.mismatches)),
            "model-free           : %s" % ("yes" if self.model_free else
                                           "NO (%s)" % ", ".join(self.non_verifiable_components)),
            "reward_version       : recorded %s, current %s%s"
            % (self.version_recorded, self.version_current,
               "" if self.version_matches else "  <-- MISMATCH"),
        ]
        lines.extend("  " + n for n in self.notes)
        if self.mismatches:
            lines.append("")
            lines.append("First mismatches:")
            lines.extend("  " + str(m) for m in self.mismatches[:10])
            if len(self.mismatches) > 10:
                lines.append("  ... and %d more" % (len(self.mismatches) - 10))
        lines.append("")
        lines.append("VERDICT: %s" % ("VERIFIED" if self.passed else "FAILED"))
        return "\n".join(lines)


def verify_scored_run(scored: Dict[str, Any], run: Dict[str, Any]) -> VerificationReport:
    """
    Recompute every record in `scored` from `run` and compare.

    Recomputation deliberately goes back to the run file rather than reusing
    anything cached in the scored file. Verifying a number against a copy of
    itself proves nothing; the point is that the raw trace still yields it.
    """
    meta = scored.get("metadata") or {}
    report = VerificationReport(version_recorded=meta.get("reward_version"))
    weights = meta.get("weights")
    source = meta.get("menu_source", "proposed")
    gate = meta.get("gate_on_correctness", True)

    if not report.version_matches:
        report.notes.append(
            "Scores were produced by reward_version %s; this build is %s. Rewards "
            "from different versions are not comparable, so recomputation below "
            "measures the current version, not agreement with the old one."
            % (report.version_recorded, report.version_current))

    rows = {(r.get("case_id"), r.get("repeat", 0)): r for r in (run.get("results") or [])}
    records = scored.get("records") or []
    report.n_records = len(records)

    # Cache per (case, repeat, mode) so a case with three menus rebuilds its
    # context once rather than three times.
    ctx_cache: Dict[Any, Any] = {}
    menu_cache: Dict[Any, Any] = {}

    for rec in records:
        key = (rec.get("case_id"), rec.get("repeat", 0))
        row = rows.get(key)
        if row is None:
            report.mismatches.append(Mismatch(
                str(rec.get("case_id")), str(rec.get("mode")), int(rec.get("menu_index", 0)),
                "source_row", "present in scored file", "absent from run file"))
            continue

        mode = rec.get("mode")
        ck = key + (mode,)
        if ck not in ctx_cache:
            ctx_cache[ck] = context_for(row, mode)
            menu_cache[ck] = menus_for(row, mode, source)[0]
        menus = menu_cache[ck]

        idx = int(rec.get("menu_index", 0))
        if idx >= len(menus):
            report.mismatches.append(Mismatch(
                str(rec.get("case_id")), str(mode), idx,
                "menu_index", "scored", "run file has only %d menu(s)" % len(menus)))
            continue

        result = score_menu(menus[idx], ctx_cache[ck], weights, gate)
        report.n_checked += 1

        if abs(result.reward - float(rec.get("reward", -1))) > TOLERANCE:
            report.mismatches.append(Mismatch(
                str(rec.get("case_id")), str(mode), idx, "reward",
                rec.get("reward"), round(result.reward, 6)))
        if rec.get("input_digest") and rec["input_digest"] != result.input_digest:
            report.mismatches.append(Mismatch(
                str(rec.get("case_id")), str(mode), idx, "input_digest",
                rec.get("input_digest"), result.input_digest))
        # Catches a reweighting applied to the file header without rescoring.
        # A menu at either extreme -- perfect on every applicable check, or
        # gated to zero -- keeps its reward under any weighting, so comparing
        # rewards alone would let that edit through unnoticed.
        if rec.get("weights_digest") and rec["weights_digest"] != result.weights_digest:
            report.mismatches.append(Mismatch(
                str(rec.get("case_id")), str(mode), idx, "weights_digest",
                rec.get("weights_digest"), result.weights_digest))

        recorded_components = rec.get("components") or {}
        for check in result.checks:
            got = recorded_components.get(check.name)
            if got is None:
                continue
            if got.get("verifiable") is False:
                report.non_verifiable_components.append(check.name)
            a, b = got.get("score"), check.score
            if a is None or b is None:
                if a is not b:
                    report.mismatches.append(Mismatch(
                        str(rec.get("case_id")), str(mode), idx, check.name, a, b))
            elif abs(float(a) - float(b)) > TOLERANCE:
                report.mismatches.append(Mismatch(
                    str(rec.get("case_id")), str(mode), idx, check.name, a, b))

    report.non_verifiable_components = sorted(set(report.non_verifiable_components))
    return report


def verify_determinism(menu: Dict[str, Any], ctx: Any, n: int = 3) -> bool:
    """
    Scoring the same menu repeatedly returns the same reward and digest.

    Cheap, but it catches the one way a pure function stops being pure here:
    a check reaching for wall-clock time, a set iteration order leaking into a
    score, or an lru_cache being mutated by a caller.
    """
    first = score_menu(menu, ctx)
    for _ in range(n - 1):
        again = score_menu(menu, ctx)
        if (abs(again.reward - first.reward) > TOLERANCE
                or again.input_digest != first.input_digest):
            return False
    return True


def load_and_verify(scored_path: str, run_path: Optional[str] = None) -> VerificationReport:
    """
    Verify a scored file, finding its results file by convention when not given.

    `<run>_reward.json` sits beside the `<run>.json` it came from, the same way
    `<run>_eval.json` does, so the default path is derivable.
    """
    scored_p = Path(scored_path)
    with open(scored_p, encoding="utf-8") as f:
        scored = json.load(f)

    if run_path is None:
        stem = scored_p.name
        if stem.endswith("_reward.json"):
            run_path = str(scored_p.with_name(stem[: -len("_reward.json")] + ".json"))
        else:
            raise ValueError(
                "cannot infer the results file from %s; pass it explicitly" % stem)

    with open(run_path, encoding="utf-8") as f:
        run = json.load(f)

    return verify_scored_run(scored, run)
