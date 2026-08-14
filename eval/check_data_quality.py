"""
check_data_quality.py -- Plausibility checks on data/recipes.json.

Why this exists
---------------
The nutrition guardrail rejects ~72% of the corpus at age 8 before any allergy is
considered, and every one of those rejections is on sugar. Investigating the
per-lunch fraction showed the fraction is not the cause:

  * salt never binds -- at fractions from 30% to 80% it rejects 0-1 recipes
  * raising the fraction from 40% to 80% only moves coverage from 8/29 to 12/29
  * the UK Gov recipes (001-009) have a median 4.4 g sugars per serving; the
    PACK-IT recipes (010-029) have a median 28.5 g, while median energy is
    comparable (351 vs 405 kcal)

A 6.5x difference in sugar with no matching difference in energy is a data
problem, not a dietary one. Savoury dishes carry implausible values (a turkey and
cheese tortilla roll-up at 44 g), and several single-serving entries repeat
suspiciously round numbers.

There is a second, independent issue: `sugars_g` is total sugars, but the
guideline field is `free_sugars_g_day_max`. Naturally occurring lactose and
fructose count toward the former and not the latter, so the comparison overstates
free sugars even where the data is right.

This script flags the suspect records. It does NOT rewrite them -- correcting
nutrition figures in a child-safety system requires the authoritative source, not
an estimate.

Usage:
  python eval/check_data_quality.py
  python eval/check_data_quality.py --json out.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from console import enable_utf8_stdout          # noqa: E402
from document_loader import _load_json          # noqa: E402

enable_utf8_stdout()

# 4 kcal per gram of sugar; sugar alone should not dominate a dish's energy.
KCAL_PER_G_SUGAR = 4.0
IMPLAUSIBLE_SUGAR_ENERGY_SHARE = 0.45
SAVOURY_MARKERS = ("chicken", "turkey", "beef", "pork", "tuna", "salmon", "egg",
                   "cheese", "spaghetti", "noodle", "salad", "burrito", "soup")
SAVOURY_SUGAR_CEILING_G = 15.0


def check(recipes: List[Dict[str, Any]]) -> Dict[str, Any]:
    findings: List[Dict[str, Any]] = []

    def flag(recipe, issue, detail):
        findings.append({"id": recipe["id"], "name": recipe["name"],
                         "issue": issue, "detail": detail})

    sugar_counts = Counter(r["nutrition_per_serving"].get("sugars_g") for r in recipes)

    for r in recipes:
        n = r.get("nutrition_per_serving", {})
        sugars = n.get("sugars_g")
        carbs = n.get("carbohydrate_g")
        kcal = n.get("energy_kcal")
        if sugars is None:
            flag(r, "missing_sugars", "no sugars_g field")
            continue

        if carbs is not None and sugars > carbs:
            flag(r, "sugars_exceed_carbs", f"sugars {sugars}g > carbohydrate {carbs}g")

        if kcal:
            share = (sugars * KCAL_PER_G_SUGAR) / kcal
            if share > IMPLAUSIBLE_SUGAR_ENERGY_SHARE:
                flag(r, "sugar_dominates_energy",
                     f"{sugars}g sugars = {share:.0%} of {kcal} kcal")

        text = (r["name"] + " " + " ".join(r.get("ingredients", []))).lower()
        if sugars > SAVOURY_SUGAR_CEILING_G and any(m in text for m in SAVOURY_MARKERS):
            flag(r, "savoury_dish_high_sugar",
                 f"{sugars}g sugars in a savoury dish")

        if sugar_counts[sugars] > 1:
            others = [x["id"] for x in recipes
                      if x is not r and x["nutrition_per_serving"].get("sugars_g") == sugars]
            flag(r, "repeated_sugar_value",
                 f"{sugars}g also on {', '.join(others)} — looks like a placeholder")

    by_issue = Counter(f["issue"] for f in findings)
    affected = sorted({f["id"] for f in findings})
    return {"n_recipes": len(recipes), "n_findings": len(findings),
            "n_affected_recipes": len(affected), "by_issue": dict(by_issue),
            "affected_recipes": affected, "findings": findings}


def print_report(result: Dict[str, Any]) -> None:
    print(f"Checked {result['n_recipes']} recipes\n")
    if not result["findings"]:
        print("No plausibility issues found.")
        return

    for issue, count in sorted(result["by_issue"].items(), key=lambda kv: -kv[1]):
        print(f"  {issue:<28} {count:>3}")
    print(f"\n{result['n_affected_recipes']} of {result['n_recipes']} recipes affected\n")

    print(f"{'recipe':<13}{'issue':<28}detail")
    print("-" * 96)
    for f in result["findings"]:
        print(f"{f['id']:<13}{f['issue']:<28}{f['detail'][:52]}")

    print("\nRecommended action: re-derive nutrition for the affected recipes from")
    print("the published source, and record whether sugars_g is TOTAL or FREE sugars.")
    print("Do not compensate by raising LUNCH_NUTRITION_FRACTION -- that weakens a")
    print("real safety limit to work around a data defect.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", default=None, help="also write findings to this path")
    args = parser.parse_args()

    result = check(_load_json("recipes.json"))
    print_report(result)

    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"\nWrote {args.json}")

    sys.exit(1 if result["findings"] else 0)
