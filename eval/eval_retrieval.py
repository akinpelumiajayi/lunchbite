"""
eval_retrieval.py

Computes standard information-retrieval metrics for the vector_store's
semantic_search() against the hand-labeled ground truth in eval_dataset.py.

Metrics implemented (all standard IR metrics, computed from scratch -- no
external eval library, so the math is auditable):

  - Precision@K:  of the top-K results returned, what fraction are relevant?
  - Recall@K:     of all relevant items that exist, what fraction appear in the top-K?
  - MRR (Mean Reciprocal Rank): 1/rank of the FIRST relevant result, averaged
                  across queries. Rewards getting a relevant result near the top.
  - NDCG@K (Normalized Discounted Cumulative Gain): rewards relevant results
                  near the top more than relevant results further down,
                  normalized against the ideal ranking.
  - Hit Rate@K:   fraction of queries where AT LEAST ONE relevant result
                  appears in the top-K (a coarser, easier-to-interpret signal).

These are computed per-query and then averaged, and reported overall and
broken down by query category (recipe / allergen_rule / nutrition_guideline)
since retrieval quality could plausibly differ by content type.
"""

import math
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from vector_store import semantic_search
from eval_dataset import EvalQuery, RECIPE_QUERIES, ALLERGEN_QUERIES, GUIDELINE_QUERIES, ALL_QUERIES


def precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if k == 0:
        return 0.0
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for rid in top_k if rid in relevant_ids)
    return hits / len(top_k)


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        # No relevant items exist for this query (the "trick" gluten-free
        # case). Recall is undefined/vacuous here -- by convention we treat
        # "nothing relevant exists, nothing relevant retrieved" as perfect
        # recall (1.0), since there was nothing to miss.
        return 1.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for rid in top_k if rid in relevant_ids)
    return hits / len(relevant_ids)


def reciprocal_rank(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    for rank, rid in enumerate(retrieved_ids, start=1):
        if rid in relevant_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Binary relevance NDCG (each relevant doc has gain=1, irrelevant gain=0)."""
    top_k = retrieved_ids[:k]
    dcg = 0.0
    for i, rid in enumerate(top_k):
        rel = 1.0 if rid in relevant_ids else 0.0
        dcg += rel / math.log2(i + 2)  # i+2 because rank starts at 1, log2(1+1)=1

    # Ideal DCG: all relevant docs (up to k) placed at the top
    ideal_hits = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))

    if idcg == 0:
        # No relevant docs exist for this query at all
        return 1.0 if dcg == 0 else 0.0
    return dcg / idcg


def hit_rate_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        # Vacuous case (the deliberate no-relevant-results query): there is no
        # relevant document to hit, so the metric is undefined and scored 1.0 by
        # convention. The previous form was `... or True`, which made the
        # surrounding condition dead code and obscured that this is a convention
        # rather than a measurement.
        return 1.0
    return 1.0 if any(rid in relevant_ids for rid in retrieved_ids[:k]) else 0.0


def evaluate_query(eq: EvalQuery, k: int = 5) -> dict:
    hits = semantic_search(eq.query, n_results=max(k, 8), source_types=eq.source_types)
    retrieved_ids = [h["id"] for h in hits]

    return {
        "query": eq.query,
        "retrieved_ids": retrieved_ids[:k],
        "relevant_ids": eq.relevant_ids,
        "precision@k": precision_at_k(retrieved_ids, eq.relevant_ids, k),
        "recall@k": recall_at_k(retrieved_ids, eq.relevant_ids, k),
        "mrr": reciprocal_rank(retrieved_ids, eq.relevant_ids),
        "ndcg@k": ndcg_at_k(retrieved_ids, eq.relevant_ids, k),
        "hit_rate@k": hit_rate_at_k(retrieved_ids, eq.relevant_ids, k),
        "notes": eq.notes,
    }


def evaluate_query_set(queries: list[EvalQuery], k: int = 5) -> dict:
    results = [evaluate_query(eq, k=k) for eq in queries]
    n = len(results)
    avg = lambda key: sum(r[key] for r in results) / n if n else 0.0
    return {
        "n_queries": n,
        "avg_precision@k": avg("precision@k"),
        "avg_recall@k": avg("recall@k"),
        "mrr": avg("mrr"),
        "avg_ndcg@k": avg("ndcg@k"),
        "avg_hit_rate@k": avg("hit_rate@k"),
        "per_query_results": results,
    }


def print_report(label: str, summary: dict, show_per_query: bool = True):
    print(f"\n{'=' * 70}\n{label}  (n={summary['n_queries']} queries)\n{'=' * 70}")
    print(f"  Precision@K : {summary['avg_precision@k']:.3f}")
    print(f"  Recall@K    : {summary['avg_recall@k']:.3f}")
    print(f"  MRR         : {summary['mrr']:.3f}")
    print(f"  NDCG@K      : {summary['avg_ndcg@k']:.3f}")
    print(f"  Hit Rate@K  : {summary['avg_hit_rate@k']:.3f}")

    if show_per_query:
        print(f"\n  Per-query breakdown:")
        for r in summary["per_query_results"]:
            flag = "OK" if r["hit_rate@k"] == 1.0 or not r["relevant_ids"] else "MISS"
            print(f"   [{flag}] \"{r['query']}\"")
            print(f"        retrieved: {r['retrieved_ids']}")
            print(f"        relevant : {sorted(r['relevant_ids']) if r['relevant_ids'] else '(none expected)'}")
            print(f"        P@K={r['precision@k']:.2f}  R@K={r['recall@k']:.2f}  MRR={r['mrr']:.2f}  NDCG@K={r['ndcg@k']:.2f}")


if __name__ == "__main__":
    K = 5

    recipe_summary = evaluate_query_set(RECIPE_QUERIES, k=K)
    allergen_summary = evaluate_query_set(ALLERGEN_QUERIES, k=K)
    guideline_summary = evaluate_query_set(GUIDELINE_QUERIES, k=K)
    overall_summary = evaluate_query_set(ALL_QUERIES, k=K)

    print_report("RECIPE RETRIEVAL", recipe_summary)
    print_report("ALLERGEN-RULE RETRIEVAL", allergen_summary)
    print_report("NUTRITION-GUIDELINE RETRIEVAL", guideline_summary)
    print_report("OVERALL (all categories combined)", overall_summary, show_per_query=False)

    print(f"\n{'=' * 70}\nPRECISION@K ACROSS K VALUES (overall, all 17 queries)\n{'=' * 70}")
    print("Note: most queries here have exactly 1 ground-truth relevant chunk.")
    print("Precision@K is mathematically capped at 1/K when only 1 relevant")
    print("item exists, even with perfect ranking -- so Precision@K shrinking")
    print("as K grows is expected here and is NOT evidence of worse retrieval.")
    print("MRR and NDCG are the more meaningful metrics for this dataset shape.\n")
    for k_val in [1, 3, 5, 8]:
        s = evaluate_query_set(ALL_QUERIES, k=k_val)
        print(f"  K={k_val}:  Precision@K={s['avg_precision@k']:.3f}  Recall@K={s['avg_recall@k']:.3f}  "
              f"NDCG@K={s['avg_ndcg@k']:.3f}  HitRate@K={s['avg_hit_rate@k']:.3f}")
