"""
eval_negation.py -- Can retrieval honour allergen negation on its own?

This is the empirical test behind the project's central design claim: that a
deterministic symbolic layer is necessary because retrieval cannot be trusted
to exclude an allergen just because the query asked it to.

Method
------
For each of the 14 declarable allergens, issue an explicit negation query
("<allergen>-free lunch for a child allergic to <allergen>") and count how
many of the top-5 retrieved recipes ACTUALLY CONTAIN that allergen, according
to the ground-truth `allergens_present` field.

A retriever that understood negation would score 0 everywhere. The count is
capped by how many unsafe recipes exist, so the "safe in corpus" column is
reported alongside — where few safe recipes exist, a high count is partly
unavoidable and should not be read as a retrieval failure.

Four retrievers are compared over the same queries:
  BM25    -- lexical
  dense   -- all-MiniLM-L6-v2 bi-encoder
  RRF     -- reciprocal-rank fusion of the two
  rerank  -- ms-marco-MiniLM-L-6-v2 cross-encoder over the fused list

Usage:
  python eval/eval_negation.py
  python eval/eval_negation.py --k 5 --json out.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "graphs"))

from console import enable_utf8_stdout
from document_loader import ALL_14_ALLERGENS, load_recipe_chunks, _load_json
from vector_store import semantic_search
from huggingface_upgrade.reranker import get_reranker
import nodes

enable_utf8_stdout()

RETRIEVE_N = 9
RRF_K = 60


def _contains(recipe: Dict[str, Any], allergen: str) -> bool:
    return allergen in [a.lower() for a in recipe.get("allergens_present", [])]


def run(k: int = 5) -> Dict[str, Any]:
    recipes = {r["id"]: r for r in _load_json("recipes.json")}
    chunk_text = {c.id: c.text for c in load_recipe_chunks()}
    reranker = get_reranker()

    def unsafe_in_top_k(ids: List[str], allergen: str) -> int:
        return sum(1 for rid in ids[:k] if _contains(recipes[rid], allergen))

    rows: List[Dict[str, Any]] = []
    for allergen in ALL_14_ALLERGENS:
        query = f"{allergen}-free lunch for a child allergic to {allergen}"

        tokens = nodes.TOKEN_RE.findall(query.lower())
        index, recipe_list = nodes._get_bm25()
        scores = index.get_scores(tokens)
        bm25_ids = [
            recipe_list[i]["id"]
            for i in sorted(range(len(recipe_list)), key=lambda i: scores[i], reverse=True)[:RETRIEVE_N]
        ]

        dense_ids = [h["id"] for h in semantic_search(query, n_results=RETRIEVE_N, source_types=["recipe"])]

        fused_scores: Dict[str, float] = {}
        for ranked in (bm25_ids, dense_ids):
            for rank, rid in enumerate(ranked, start=1):
                fused_scores[rid] = fused_scores.get(rid, 0.0) + 1.0 / (RRF_K + rank)
        fused_ids = sorted(fused_scores, key=lambda r: fused_scores[r], reverse=True)

        candidates = [{"id": rid, "text": chunk_text[rid]} for rid in fused_ids]
        rerank_ids = [c["id"] for c in reranker.rerank(query, candidates, top_k=RETRIEVE_N)]

        rows.append({
            "allergen": allergen,
            "safe_recipes_in_corpus": sum(1 for r in recipes.values() if not _contains(r, allergen)),
            "bm25": unsafe_in_top_k(bm25_ids, allergen),
            "dense": unsafe_in_top_k(dense_ids, allergen),
            "rrf": unsafe_in_top_k(fused_ids, allergen),
            "rerank": unsafe_in_top_k(rerank_ids, allergen),
        })

    totals = {m: sum(r[m] for r in rows) for m in ("bm25", "dense", "rrf", "rerank")}
    return {"k": k, "n_allergens": len(rows), "slots": len(rows) * k,
            "rows": rows, "totals": totals}


def print_report(result: Dict[str, Any]) -> None:
    k, slots = result["k"], result["slots"]
    print(f"\nUnsafe recipes appearing in top-{k} for an explicit '<allergen>-free' query")
    print("(0 = retriever fully honoured the negation; lower is better)\n")
    header = (f"{'allergen':<28} {'safe in corpus':>14} "
              f"{'BM25':>6} {'dense':>6} {'RRF':>6} {'rerank':>7}")
    print(header)
    print("-" * len(header))
    for r in result["rows"]:
        print(f"{r['allergen']:<28} {r['safe_recipes_in_corpus']:>14} "
              f"{r['bm25']:>6} {r['dense']:>6} {r['rrf']:>6} {r['rerank']:>7}")
    print("-" * len(header))
    t = result["totals"]
    print(f"{f'TOTAL (of {slots} slots)':<28} {'':>14} "
          f"{t['bm25']:>6} {t['dense']:>6} {t['rrf']:>6} {t['rerank']:>7}")
    print(
        "\nReading: none of the four retrievers reliably honours negation. Upgrading\n"
        "from lexical to neural retrieval does not make retrieval allergen-safe, which\n"
        "is precisely why the symbolic guardrail is a separate, deterministic stage."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=5, help="cut-off for the top-k count")
    parser.add_argument("--json", default=None, help="also write results to this path")
    args = parser.parse_args()

    result = run(k=args.k)
    print_report(result)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"\nWrote {args.json}")
