"""
setup_database.py

Builds both retrieval indexes:
  1. ChromaDB vector store (neural sentence embeddings, all-MiniLM-L6-v2)
  2. BM25 lexical index with empirically tuned k1/b

Also warms the cross-encoder reranker so its model download happens here
rather than partway through a benchmark run.

Run once on first setup, and again whenever data/*.json changes.

Usage:
  python3 src/setup_database.py
"""

from __future__ import annotations
import os
import pickle
import re
import sys
from pathlib import Path
from typing import List, Dict, Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from console import enable_utf8_stdout
from vector_store import build_collection
from document_loader import load_all_chunks, _load_json
from llm_provider import use_cross_encoder_reranker
from huggingface_upgrade.reranker import get_reranker

enable_utf8_stdout()

TOKEN_RE = re.compile(r"[a-z0-9]+")


def build_bm25_index(k1: float = 1.0, b: float = 0.4) -> None:
    """Build and persist the BM25 index using tuned hyperparameters."""
    from rank_bm25 import BM25Okapi

    recipes: List[Dict[str, Any]] = _load_json("recipes.json")
    corpus = [
        TOKEN_RE.findall(
            (r["name"] + " " + " ".join(r["ingredients"]) + " " + r.get("description", "")).lower()
        )
        for r in recipes
    ]
    index = BM25Okapi(corpus, k1=k1, b=b)

    vectordb_dir = Path(__file__).resolve().parent.parent / "vectordb"
    vectordb_dir.mkdir(parents=True, exist_ok=True)
    index_path = vectordb_dir / "bm25_index.pkl"

    with open(index_path, "wb") as f:
        pickle.dump({"bm25": index, "recipes": recipes, "k1": k1, "b": b}, f)
    print(f"BM25 index built (k1={k1}, b={b}) with {len(recipes)} recipes → {index_path}")


def main() -> None:
    print("Loading knowledge base documents...")
    chunks = load_all_chunks()
    by_type: Dict[str, int] = {}
    for c in chunks:
        t = c.metadata.get("source_type", "unknown")
        by_type[t] = by_type.get(t, 0) + 1
    for t, n in by_type.items():
        print(f"  {t}: {n} chunks")

    print("\nBuilding ChromaDB vector store (neural sentence embeddings)...")
    print("  First run downloads the model (~90 MB) and caches it locally.")
    collection = build_collection(reset=True)
    print(f"Vector store ready: {collection.count()} chunks in '{collection.name}'")

    print("\nBuilding BM25 lexical index (k1=1.0, b=0.4)...")
    build_bm25_index(k1=1.0, b=0.4)

    # Pay the cross-encoder download cost here rather than mid-benchmark.
    if use_cross_encoder_reranker():
        print("\nWarming up the cross-encoder reranker...")
        reranker = get_reranker()
        reranker.warm_up()
        print(f"  Reranker ready: {reranker.model_name}")
    else:
        print("\nCross-encoder reranker disabled (USE_CROSS_ENCODER_RERANKER=false).")

    print("\nSetup complete. Both indexes are ready.")
    print("Run: python run_all.py --mock   (to test without an API key)")
    print("Run: python run_all.py          (full live run; reads GROQ_API_KEY from .env)")


if __name__ == "__main__":
    main()
