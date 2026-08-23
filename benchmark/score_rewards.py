"""
score_rewards.py -- Scores a benchmark results file with the verifiable reward.

    python benchmark/score_rewards.py benchmark/results/run_<ts>.json
    python benchmark/score_rewards.py benchmark/results/run_<ts>.json --verify
    python benchmark/score_rewards.py --verify-only benchmark/results/run_<ts>_reward.json

Writes `<run>_reward.json` beside the results file, following the same naming
convention as `<run>_eval.json`.

Costs nothing to run. No model is called, no network request is made, and the
run being scored is never re-executed -- which is what makes it usable against
the free-tier daily token budget that already constrains this project, and what
lets a run recorded weeks ago be scored today.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from console import enable_utf8_stdout  # noqa: E402
from reward.scoring import score_run  # noqa: E402
from reward.sensitivity import sensitivity, verify_reaggregation  # noqa: E402
from reward.verify import verify_scored_run  # noqa: E402

enable_utf8_stdout()


def reward_path_for(results_path: str) -> Path:
    p = Path(results_path)
    return p.with_name(p.stem + "_reward.json")


def _print_summary(scored: Dict[str, Any]) -> None:
    meta = scored["metadata"]
    print("\nVerifiable reward -- version %s, menus scored from %r"
          % (meta["reward_version"], meta["menu_source"]))
    print("Weights: " + ", ".join("%s=%.2f" % (k, v) for k, v in meta["weights"].items()))
    print()

    names = list(meta["weights"].keys())
    header = "%-15s %6s %8s %7s" % ("arm", "menus", "reward", "gated")
    header += "".join("%12s" % n[:11] for n in names)
    print(header)
    print("-" * len(header))
    for mode, agg in scored["per_mode"].items():
        row = "%-15s %6d %8.3f %7d" % (mode, agg["n_menus"], agg["mean_reward"] or 0.0,
                                       agg["gated_count"])
        for n in names:
            m = agg["components"][n]["mean"]
            row += "%12s" % ("--" if m is None else "%.3f" % m)
        print(row)

    print("\nBest-of-N reranking (menus the generator already produced):")
    print("%-15s %6s %10s %10s %8s %10s"
          % ("arm", "cases", "first", "best", "delta", "reranked"))
    print("-" * 62)
    for mode, b in scored["best_of_n"].items():
        print("%-15s %6d %10s %10s %8s %10s"
              % (mode, b["n_cases"],
                 "--" if b["mean_reward_first_menu"] is None else "%.3f" % b["mean_reward_first_menu"],
                 "--" if b["mean_reward_best_menu"] is None else "%.3f" % b["mean_reward_best_menu"],
                 "--" if b["delta"] is None else "%+.3f" % b["delta"],
                 "%d/%d" % (b["cases_reranked"], b["n_cases_with_choice"])))

    frac = [a["mean_verifiable_fraction"] for a in scored["per_mode"].values()
            if a["mean_verifiable_fraction"] is not None]
    if frac:
        print("\nVerifiable fraction of reward mass: %.3f (1.000 means no model "
              "was consulted anywhere)" % (sum(frac) / len(frac)))


def _print_sensitivity(block: Dict[str, Any]) -> None:
    """
    Does the arm ordering depend on the weights?

    The weights were chosen by argument rather than fitted, so the honest
    defence is not to justify them but to show the conclusion survives changing
    them -- including `hostile`, which weights correctness at a fraction of its
    default. An ordering that holds throughout means the exact values stop
    mattering.
    """
    per = block["per_weighting"]
    modes = list(next(iter(per.values()))["mean_reward"].keys())

    print("\nWeight sensitivity -- mean reward under each weighting")
    header = "%-20s" % "weighting" + "".join("%15s" % m[:14] for m in modes)
    print(header)
    print("-" * len(header))
    for label, blk in per.items():
        row = "%-20s" % label
        for m in modes:
            v = blk["mean_reward"][m]
            row += "%15s" % ("--" if v is None else "%.3f" % v)
        print(row)

    print()
    if block["ordering_stable"]:
        print("  Ordering STABLE across all %d weightings:"
              % len(block["weightings_tested"]))
        print("    " + " > ".join(block["distinct_orderings"][0]))
        print("  The conclusion does not rest on the weighting.")
    else:
        print("  Ordering CHANGES with the weighting -- %d distinct orderings:"
              % len(block["distinct_orderings"]))
        for o in block["distinct_orderings"]:
            print("    " + " > ".join(o))
        moved = [m for m, r in block["rank_range"].items() if r["best"] != r["worst"]]
        print("  Arms whose rank the weighting decides: " + ", ".join(moved))


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", nargs="?", help="benchmark/results/run_<ts>.json")
    ap.add_argument("--out", default=None, help="defaults to <run>_reward.json")
    ap.add_argument("--menus", choices=["proposed", "final"], default="proposed",
                    help="proposed measures the model before the gates repaired "
                         "it; final measures the system as shipped. Default: proposed.")
    ap.add_argument("--no-gate", action="store_true",
                    help="do not zero the reward when correctness fails. For "
                         "showing what an ungated reward would have paid out; "
                         "not for reporting a system score.")
    ap.add_argument("--slim", action="store_true",
                    help="drop per-check evidence from the output file. Smaller, "
                         "but no longer self-contained as an audit trail.")
    ap.add_argument("--sensitivity", action="store_true",
                    help="re-aggregate under several weightings and report "
                         "whether the ordering of the arms changes. Answers "
                         "\"why these weights\" without needing preference data.")
    ap.add_argument("--verify", action="store_true",
                    help="recompute every record after writing and report whether "
                         "the numbers hold")
    ap.add_argument("--verify-only", default=None, metavar="REWARD.JSON",
                    help="verify an existing scored file and exit")
    ap.add_argument("--run", default=None, metavar="RESULTS.JSON",
                    help="the results file to verify against. Only needed with "
                         "--verify-only when the scored file has been renamed "
                         "or moved away from the run it came from; otherwise it "
                         "is derived from the filename.")
    args = ap.parse_args(argv)

    if args.verify_only:
        from reward.verify import load_and_verify
        report = load_and_verify(args.verify_only, args.run)
        print(report.summary())
        return 0 if report.passed else 1

    if not args.results:
        ap.error("a results file is required unless --verify-only is given")

    with open(args.results, encoding="utf-8") as f:
        payload = json.load(f)

    meta = payload.get("metadata") or {}
    if meta.get("synthetic"):
        print("!" * 70)
        print("SYNTHETIC RUN: these results came from the mock LLM, not a real model.")
        print("Rewards computed from them describe the mock, and are not evidence.")
        print("!" * 70)

    scored = score_run(payload, source=args.menus,
                       gate_on_correctness=not args.no_gate,
                       keep_evidence=not args.slim)
    scored["metadata"]["synthetic"] = bool(meta.get("synthetic"))
    scored["metadata"]["source_results_file"] = str(Path(args.results).name)

    out = Path(args.out) if args.out else reward_path_for(args.results)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(scored, f, indent=2)

    _print_summary(scored)

    if args.sensitivity:
        # Self-check first. `reaggregate` is a second implementation of the
        # aggregation in `model.score_menu`, and two implementations that
        # disagree would make every number below meaningless.
        agrees, worst = verify_reaggregation(scored)
        if not agrees:
            print("\n  WARNING: re-aggregation disagrees with the recorded rewards "
                  "by up to %.2e. The sensitivity numbers below are not "
                  "trustworthy." % worst)
        block = sensitivity(scored)
        scored["sensitivity"] = block
        with open(out, "w", encoding="utf-8") as f:
            json.dump(scored, f, indent=2)
        _print_sensitivity(block)

    print("\nRewards written to: %s" % out)

    if args.verify:
        report = verify_scored_run(scored, payload)
        print()
        print(report.summary())
        return 0 if report.passed else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
