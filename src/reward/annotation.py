"""
annotation.py -- Builds the blinded instrument for the human validation study.

The reward is verifiable: recomputing it gives the same number. That is not the
same as valid, which would mean the number tracks what a person would choose.
Validation is a measurement OF the instrument, never a modification of it - no
reward value changes here, and nothing in this module is read by `score_menu`.

Two artefacts come out of `build_instrument`:

  instrument   what the annotator sees. No arm name, no reward, no judge score
               anywhere in the file, because blinding that depends on the CLI
               choosing not to print something is not blinding: the annotator
               has a text editor.
  keymap       item_id -> (case_id, repeat, mode, menu_index), plus which side
               of each pair held the higher reward. Opened only by the analysis.

The annotator must see the *same* SOURCE block the judge saw, so `source_text`
is imported from the evaluator rather than re-rendered. Two fields missing from
that text once produced a 0.000 faithfulness floor across two whole arms; a
second copy of it would be a second chance to make that mistake, and would mean
kappa measured a difference in information rather than in judgement.
"""

from __future__ import annotations

import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "benchmark"))

from evaluator import source_text  # noqa: E402

from .corpus import get_recipe  # noqa: E402
from .scoring import menus_for  # noqa: E402

# Strata, in the order they are filled. Each answers a different question, and
# `reranked` is the one that breaks the circularity in the best-of-N result:
# those pairs ARE the decisions the reward made on the pipeline's behalf.
STRATA = ("reranked", "safety_contrast", "disagreement", "random")

INSTRUMENT_VERSION = 1


def profile_text(profile: Dict[str, Any]) -> str:
    """
    The child, as the annotator reads them.

    Deliberately omits `cultural_context` when it carries an adversarial
    injection: the benchmark appends attack text to that field, and showing it
    would ask the annotator to rate a prompt-injection payload as if it were a
    parent's note.
    """
    bits = ["Age %s." % profile.get("age_years", "?")]
    allergies = [a for a in (profile.get("allergies") or []) if str(a).strip()]
    intol = [a for a in (profile.get("intolerances") or []) if str(a).strip()]
    if allergies:
        bits.append("Allergic to: %s." % ", ".join(allergies))
    if intol:
        bits.append("Intolerant to: %s." % ", ".join(intol))
    if profile.get("school_nut_free"):
        bits.append("Nut-free school.")
    diets = [d for d in (profile.get("diet_requirements") or []) if str(d).strip()]
    if diets:
        bits.append("Diet: %s." % ", ".join(diets))
    likes = [t for t in (profile.get("likes") or []) if str(t).strip()]
    dislikes = [t for t in (profile.get("dislikes") or []) if str(t).strip()]
    if likes:
        bits.append("Likes: %s." % ", ".join(likes))
    if dislikes:
        bits.append("Dislikes: %s." % ", ".join(dislikes))
    return " ".join(bits)


def menu_text(menu: Dict[str, Any]) -> str:
    """
    The recommendation, with the recipe_id removed.

    The id is a machine key, and it is also a tell: `recipe_007` in the menu and
    `recipe_007` at the top of SOURCE turns the faithfulness question into a
    string match. The annotator gets the prose the parent would get.
    """
    lines = [str(menu.get("menu_name") or "(unnamed)")]
    if menu.get("why_it_fits"):
        lines.append("Why it fits: %s" % menu["why_it_fits"])
    if menu.get("nutritional_rationale"):
        lines.append("Nutrition: %s" % menu["nutritional_rationale"])
    absent = [a for a in (menu.get("allergens_confirmed_absent") or []) if a]
    if absent:
        lines.append("Confirmed absent: %s" % ", ".join(absent))
    if menu.get("source_citation"):
        lines.append("Source: %s" % menu["source_citation"])
    return "\n".join(lines)


def _menu_index(rows: Dict[Tuple[str, int], Dict[str, Any]],
                rec: Dict[str, Any], menu_source: str) -> Optional[Dict[str, Any]]:
    row = rows.get((rec["case_id"], rec.get("repeat", 0)))
    if row is None:
        return None
    menus, _ = menus_for(row, rec["mode"], menu_source)
    idx = rec.get("menu_index", 0)
    return menus[idx] if idx < len(menus) else None


def _judge_lookup(eval_payload: Optional[Dict[str, Any]]) -> Dict[Tuple, Dict[str, Any]]:
    if not eval_payload:
        return {}
    recs = ((eval_payload.get("llm_metrics") or {}).get("_judge_records")) or []
    return {(r["case_id"], r.get("repeat", 0), r["mode"]): r
            for r in recs if not r.get("error")}


def _pair_strata(records: List[Dict[str, Any]],
                 judge: Dict[Tuple, Dict[str, Any]]) -> Dict[str, List[Tuple[int, int]]]:
    """
    Candidate index pairs into `records`, grouped by what each pair tests.

    Both members of a pair always come from the same (case_id, repeat): a menu is
    only comparable against another answer to the SAME question, and pairing
    across cases would ask the annotator to prefer one child's lunch over
    another's.
    """
    by_case: Dict[Tuple[str, int], List[int]] = {}
    for i, r in enumerate(records):
        by_case.setdefault((r["case_id"], r.get("repeat", 0)), []).append(i)

    out: Dict[str, List[Tuple[int, int]]] = {s: [] for s in STRATA}

    for key, idxs in by_case.items():
        if len(idxs) < 2:
            continue

        # reranked: within one arm, the menu the pipeline returned first against
        # the one the reward would promote. These are the reward's own decisions.
        by_mode: Dict[str, List[int]] = {}
        for i in idxs:
            by_mode.setdefault(records[i]["mode"], []).append(i)
        for mode, group in by_mode.items():
            if len(group) < 2:
                continue
            first = min(group, key=lambda i: records[i].get("menu_index", 0))
            best = max(group, key=lambda i: records[i]["reward"])
            if best != first and records[best]["reward"] > records[first]["reward"] + 1e-9:
                out["reranked"].append((first, best))

        # safety_contrast: a menu the gate zeroed against one it did not. If a
        # human prefers the gated menu at any rate, a preference-fitted reward
        # would learn to reward violations -- which is the strongest argument in
        # this artifact for keeping safety symbolic rather than learned.
        gated = [i for i in idxs if records[i].get("gated")]
        clean = [i for i in idxs if not records[i].get("gated")]
        for g in gated:
            for c in clean:
                out["safety_contrast"].append((g, c))

        # disagreement: the reward and the judge rank this pair oppositely. The
        # informative cases, and the ones a reader will look for first.
        if judge:
            for a in idxs:
                for b in idxs:
                    if a >= b:
                        continue
                    ja = judge.get((records[a]["case_id"], records[a].get("repeat", 0),
                                    records[a]["mode"]))
                    jb = judge.get((records[b]["case_id"], records[b].get("repeat", 0),
                                    records[b]["mode"]))
                    if not ja or not jb:
                        continue
                    fa, fb = ja.get("faithfulness"), jb.get("faithfulness")
                    if fa is None or fb is None:
                        continue
                    r_delta = records[a]["reward"] - records[b]["reward"]
                    j_delta = fa - fb
                    if abs(r_delta) > 1e-9 and abs(j_delta) > 1e-9 and \
                            (r_delta > 0) != (j_delta > 0):
                        out["disagreement"].append((a, b))

        for a in idxs:
            for b in idxs:
                if a < b:
                    out["random"].append((a, b))

    return out


def build_instrument(run: Dict[str, Any], reward: Dict[str, Any],
                     eval_payload: Optional[Dict[str, Any]] = None,
                     n_items: int = 40, n_pairs: int = 60,
                     repeat_fraction: float = 0.2,
                     overlap_items: int = 20,
                     seed: int = 20260822) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Build the blinded instrument and its keymap.

    `n_items` is the number of DISTINCT menus rated; `repeat_fraction` of that
    many are shown a second time to measure intra-rater consistency, so the
    annotator actually sees `n_items * (1 + repeat_fraction)` screens.

    `overlap_items` marks the prefix a second annotator rates, for inter-rater
    agreement. It has to be decided now rather than later: a second rater who
    picks their own subset is not overlapping with the first on purpose.
    """
    rng = random.Random(seed)
    records = list(reward.get("records") or [])
    if not records:
        raise ValueError("reward payload has no records to sample from")

    menu_source = (reward.get("metadata") or {}).get("menu_source", "proposed")
    rows = {(r.get("case_id"), r.get("repeat", 0)): r for r in (run.get("results") or [])}
    judge = _judge_lookup(eval_payload)

    def render(rec):
        menu = _menu_index(rows, rec, menu_source)
        if menu is None:
            return None
        recipe = get_recipe(menu.get("recipe_id"))
        row = rows[(rec["case_id"], rec.get("repeat", 0))]
        return {
            "profile_text": profile_text(row.get("profile") or {}),
            "source_text": source_text(recipe) if recipe else "(recipe not in corpus)",
            "menu_text": menu_text(menu),
        }

    # ── absolute-rating items, spread across the reward range ────────────────
    # Random sampling would cluster: the reward is dense above 0.90 and has a
    # spike at exactly 0.0 from the gate. Deciles keep the low end represented,
    # which is where agreement is most likely to be informative.
    renderable = [(i, r) for i, r in enumerate(records) if render(r) is not None]
    if not renderable:
        raise ValueError("no record could be rendered; run and reward files may not match")

    by_decile: Dict[int, List[Tuple[int, Dict[str, Any]]]] = {}
    for i, r in renderable:
        by_decile.setdefault(min(9, int(r["reward"] * 10)), []).append((i, r))
    for bucket in by_decile.values():
        rng.shuffle(bucket)

    chosen: List[Tuple[int, Dict[str, Any]]] = []
    deciles = sorted(by_decile)
    while len(chosen) < min(n_items, len(renderable)):
        progressed = False
        for d in deciles:
            if by_decile[d] and len(chosen) < n_items:
                chosen.append(by_decile[d].pop())
                progressed = True
        if not progressed:
            break

    items, keymap_items = [], {}
    for n, (ridx, rec) in enumerate(chosen, 1):
        item_id = "IT-%03d" % n
        items.append({"item_id": item_id, "repeat_of": None, **render(rec)})
        keymap_items[item_id] = {"case_id": rec["case_id"], "repeat": rec.get("repeat", 0),
                                 "mode": rec["mode"], "menu_index": rec.get("menu_index", 0),
                                 "reward": rec["reward"], "record_index": ridx}

    # Repeats are textually identical to their originals. A paraphrase would
    # measure whether the annotator notices rewording, not whether they are
    # consistent.
    n_repeat = int(round(len(items) * repeat_fraction))
    for n, original in enumerate(rng.sample(items, min(n_repeat, len(items))), 1):
        item_id = "IT-R%02d" % n
        clone = dict(original)
        clone["item_id"] = item_id
        clone["repeat_of"] = original["item_id"]
        items.append(clone)
        keymap_items[item_id] = dict(keymap_items[original["item_id"]],
                                     repeat_of=original["item_id"])

    order = items[:]
    rng.shuffle(order)

    # ── pairwise items ───────────────────────────────────────────────────────
    strata = _pair_strata(records, judge)
    for v in strata.values():
        rng.shuffle(v)

    seen: set = set()
    picked: List[Tuple[str, int, int]] = []
    # Round-robin so a large `random` stratum cannot crowd out the three that
    # were chosen for a reason.
    while len(picked) < n_pairs and any(strata[s] for s in STRATA):
        for s in STRATA:
            if len(picked) >= n_pairs:
                break
            while strata[s]:
                a, b = strata[s].pop()
                key = (min(a, b), max(a, b))
                if key in seen:
                    continue
                seen.add(key)
                picked.append((s, a, b))
                break

    pairs, keymap_pairs = [], {}
    for n, (stratum, a, b) in enumerate(picked, 1):
        pair_id = "PR-%03d" % n
        ra, rb = records[a], records[b]
        left_is_a = rng.random() < 0.5          # side assignment fixed at build time
        la, lb = (ra, rb) if left_is_a else (rb, ra)
        rendered_l, rendered_r = render(la), render(lb)
        if rendered_l is None or rendered_r is None:
            continue
        # `stratum` is deliberately NOT written here. "safety_contrast" tells the
        # annotator one of the two menus is unsafe, and "reranked" tells them the
        # reward already preferred one -- both are exactly the information the
        # pair exists to collect independently. It lives in the keymap instead.
        pairs.append({
            "pair_id": pair_id,
            "profile_text": rendered_l["profile_text"],
            "left": {k: rendered_l[k] for k in ("source_text", "menu_text")},
            "right": {k: rendered_r[k] for k in ("source_text", "menu_text")},
        })
        if abs(la["reward"] - lb["reward"]) <= 1e-9:
            higher = "tie"
        else:
            higher = "left" if la["reward"] > lb["reward"] else "right"
        keymap_pairs[pair_id] = {
            "stratum": stratum, "higher_reward": higher,
            "left": {"case_id": la["case_id"], "repeat": la.get("repeat", 0),
                     "mode": la["mode"], "menu_index": la.get("menu_index", 0),
                     "reward": la["reward"]},
            "right": {"case_id": lb["case_id"], "repeat": lb.get("repeat", 0),
                      "mode": lb["mode"], "menu_index": lb.get("menu_index", 0),
                      "reward": lb["reward"]},
        }

    # Shuffled, so the stratum cannot be inferred from position either: the
    # round-robin fill above lays them down in a repeating cycle, and an
    # annotator who noticed that every fourth pair was lopsided would be
    # answering the pattern rather than the question.
    rng.shuffle(pairs)

    ts = time.strftime("%Y%m%d_%H%M%S")
    rw_meta = reward.get("metadata") or {}
    common = {
        "created": ts,
        "instrument_version": INSTRUMENT_VERSION,
        "reward_version": rw_meta.get("reward_version"),
        "judge_rubric_version": ((eval_payload or {}).get("llm_metrics") or {})
        .get("_judge_rubric_version"),
        "source_run": rw_meta.get("source_results_file") or rw_meta.get("source_run"),
        "seed": seed,
        "menu_source": menu_source,
        "n_items": len(items),
        "n_unique_items": len(chosen),
        "n_pairs": len(pairs),
        "overlap_items": min(overlap_items, len(order)),
    }
    instrument = {
        "metadata": {**common, "blinded": True,
                     "note": ("Contains no arm name, reward or judge score. Hand this "
                              "file to the annotator; keep the keymap.")},
        # Presentation order is baked in so two annotators see the same sequence
        # and any ordering effect is shared rather than confounded with rater.
        "item_order": [i["item_id"] for i in order],
        "items": order,
        "pairs": pairs,
    }
    keymap = {"metadata": common, "items": keymap_items, "pairs": keymap_pairs}
    return instrument, keymap
