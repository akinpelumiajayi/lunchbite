#!/usr/bin/env python3
"""
eval_compare_retrievers.py

Compares the four retrieval configurations the pipeline can be built from,
against the hand-labelled ground truth in eval_dataset.py:

  bm25            lexical only (rank_bm25, k1=1.0 b=0.4)
  semantic        dense only (all-MiniLM-L6-v2 sentence embeddings)
  hybrid_rrf      both, fused with Reciprocal Rank Fusion (k=60)
  hybrid_rerank   hybrid_rrf, then re-ordered by a cross-encoder

This is what justifies the architecture: the pipeline pays for two retrievers, a
fusion step and a cross-encoder, and that cost is only defensible if the numbers
move. If hybrid does not beat its parts, the honest conclusion is to simplify.

SCOPE: recipe queries only. The BM25 index is built over recipe chunks
(nodes._get_bm25 reads data/recipes.json), so it cannot answer the allergen-rule
or nutrition-guideline queries at all. Including those would not be a comparison
of retrievers, it would be a comparison of index coverage — the dense retriever
would win every one by default. eval_retrieval.py covers all 17 queries against
the dense retriever; this file compares methods on the 7 they can all serve.

Usage:
    python eval/eval_compare_retrievers.py
    python eval/eval_compare_retrievers.py --k 5 --json out.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Callable, Dict, List

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "src", "graphs"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from console import enable_utf8_stdout  # noqa: E402

enable_utf8_stdout()

from eval_dataset import RECIPE_QUERIES, EvalQuery  # noqa: E402
from eval_retrieval import (  # noqa: E402
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from vector_store import semantic_search  # noqa: E402

# Reuse the pipeline's own retrieval pieces so this measures the shipped system
# rather than a reimplementation that could drift from it.
from graphs.nodes import RRF_K, TOKEN_RE, _get_bm25, _recipe_chunk_texts  # noqa: E402

DEPTH = 20          # candidates each base retriever contributes before fusion


# ── The four retrievers ──────────────────────────────────────────────────────

def retrieve_bm25(query: str, n: int) -> List[str]:
    index, recipes = _get_bm25()
    scores = index.get_scores(TOKEN_RE.findall(query.lower()))
    ranked = sorted(zip(recipes, scores), key=lambda p: p[1], reverse=True)
    return [r["id"] for r, s in ranked[:n]]


def retrieve_semantic(query: str, n: int) -> List[str]:
    return [h["id"] for h in semantic_search(query, n_results=n, source_types=["recipe"])]


def _rrf(rankings: List[List[str]], k: int = RRF_K) -> List[str]:
    """Reciprocal Rank Fusion: score = sum over lists of 1/(k + rank)."""
    scores: Dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return [d for d, _ in sorted(scores.items(), key=lambda p: p[1], reverse=True)]


def retrieve_hybrid_rrf(query: str, n: int) -> List[str]:
    return _rrf([retrieve_bm25(query, DEPTH), retrieve_semantic(query, DEPTH)])[:n]


def retrieve_hybrid_rerank(query: str, n: int) -> List[str]:
    from huggingface_upgrade.reranker import get_reranker
    fused = retrieve_hybrid_rrf(query, DEPTH)
    texts = _recipe_chunk_texts()
    candidates = [{"id": rid, "text": texts.get(rid, "")} for rid in fused]
    ranked = get_reranker().rerank(query, candidates, top_k=n,
                                   text_of=lambda c: c.get("text", ""))
    return [c["id"] for c in ranked]


RETRIEVERS: Dict[str, Callable[[str, int], List[str]]] = {
    "bm25": retrieve_bm25,
    "semantic": retrieve_semantic,
    "hybrid_rrf": retrieve_hybrid_rrf,
    "hybrid_rerank": retrieve_hybrid_rerank,
}


# ── Scoring ──────────────────────────────────────────────────────────────────

def score_retriever(fn: Callable[[str, int], List[str]], queries: List[EvalQuery],
                    k: int) -> Dict[str, Any]:
    rows = []
    for q in queries:
        retrieved = fn(q.query, max(k, 8))
        rows.append({
            "query": q.query,
            "n_relevant": len(q.relevant_ids),
            "retrieved": retrieved[:k],
            "precision": precision_at_k(retrieved, q.relevant_ids, k),
            "recall": recall_at_k(retrieved, q.relevant_ids, k),
            "mrr": reciprocal_rank(retrieved, q.relevant_ids),
            "ndcg": ndcg_at_k(retrieved, q.relevant_ids, k),
            "hit_rate": hit_rate_at_k(retrieved, q.relevant_ids, k),
        })

    def mean(key):
        return round(sum(r[key] for r in rows) / len(rows), 3) if rows else 0.0

    return {
        "precision_at_k": mean("precision"),
        "recall_at_k": mean("recall"),
        "mrr": mean("mrr"),
        "ndcg_at_k": mean("ndcg"),
        "hit_rate_at_k": mean("hit_rate"),
        "n_queries": len(rows),
        "per_query": rows,
    }


def compare(k: int = 5) -> Dict[str, Any]:
    out: Dict[str, Any] = {"k": k, "n_queries": len(RECIPE_QUERIES), "retrievers": {}}
    for name, fn in RETRIEVERS.items():
        print(f"  scoring {name} ...")
        try:
            out["retrievers"][name] = score_retriever(fn, RECIPE_QUERIES, k)
        except Exception as e:                                   # noqa: BLE001
            # A missing cross-encoder download should not lose the other three.
            print(f"    SKIPPED ({type(e).__name__}: {e})")
            out["retrievers"][name] = {"error": f"{type(e).__name__}: {e}"}
    return out


def print_table(results: Dict[str, Any]) -> None:
    k = results["k"]
    print("\n" + "=" * 78)
    print(f"RETRIEVER COMPARISON  (K={k}, {results['n_queries']} recipe queries)")
    print("=" * 78)
    print(f"{'Retriever':<16}{'P@K':>9}{'R@K':>9}{'MRR':>9}{'NDCG@K':>10}{'HitRate':>10}")
    print("-" * 78)
    for name, m in results["retrievers"].items():
        if "error" in m:
            print(f"{name:<16}{'— unavailable —':>47}")
            continue
        print(f"{name:<16}{m['precision_at_k']:>9.3f}{m['recall_at_k']:>9.3f}"
              f"{m['mrr']:>9.3f}{m['ndcg_at_k']:>10.3f}{m['hit_rate_at_k']:>10.3f}")
    print("-" * 78)

    ok = {n: m for n, m in results["retrievers"].items() if "error" not in m}
    if ok:
        best = max(ok, key=lambda n: ok[n]["ndcg_at_k"])
        print(f"Best by NDCG@{k}: {best} ({ok[best]['ndcg_at_k']:.3f})")
    print("\nNote: recipe queries only — the BM25 index covers recipe chunks, so the")
    print("guideline and allergen-rule queries cannot be served by every method.")
    print("Absence queries (e.g. 'gluten-free') are the hard case for every dense")
    print("method here: a gluten-free recipe does not say 'gluten-free', it simply")
    print("lacks wheat, and an embedding cannot represent an absent ingredient.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--json", default=None, help="Write full results to this path")
    args = p.parse_args()

    res = compare(k=args.k)
    print_table(res)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
        print(f"\nFull results written to: {args.json}")
