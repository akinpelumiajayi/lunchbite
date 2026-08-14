"""
vector_store.py

Wraps ChromaDB (a local, persistent vector database) with neural sentence
embeddings from huggingface_upgrade.huggingface_embeddings.

Provides build_collection() and semantic_search(), used by the LangGraph nodes
and by the retrieval evaluation harness.

The client, collection, and embedding model are all process-level singletons —
each is expensive to construct and was previously rebuilt on every query.
"""

from __future__ import annotations

import os
import threading
from typing import Any, Dict, List, Optional

import chromadb

from document_loader import Chunk, load_all_chunks
from huggingface_upgrade.huggingface_embeddings import (
    EmbeddingModelUnavailable,
    HuggingFaceEmbeddingFunction,
    get_embedding_function as _get_embedding_function,
)

PERSIST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vectordb")
COLLECTION_NAME = "lunch_rag_kb"

# Cosine is the right metric for L2-normalised sentence embeddings, and it must
# be identical on the create and the read path or the index is queried wrongly.
COLLECTION_METADATA = {"hnsw:space": "cosine"}

_client_singleton: Optional[chromadb.PersistentClient] = None
_collection_singleton: Optional[chromadb.Collection] = None

# Reentrant: get_collection() calls get_client() while already holding this.
_lock = threading.RLock()


# ── Handles ───────────────────────────────────────────────────────────────────

def get_embedding_function() -> HuggingFaceEmbeddingFunction:
    return _get_embedding_function()


def get_client() -> chromadb.PersistentClient:
    """Cached PersistentClient — opening the store per query was a real cost."""
    global _client_singleton
    if _client_singleton is None:
        with _lock:
            if _client_singleton is None:
                os.makedirs(PERSIST_DIR, exist_ok=True)
                _client_singleton = chromadb.PersistentClient(path=PERSIST_DIR)
    return _client_singleton


def _reset_caches() -> None:
    """Drop cached handles — used after a rebuild changes the collection."""
    global _collection_singleton
    _collection_singleton = None


def get_collection() -> chromadb.Collection:
    """
    Cached handle to the knowledge-base collection.

    Passes the same metadata as build_collection so that if the collection does
    not yet exist, it is created with cosine space rather than Chroma's default
    L2 — previously this path silently produced a mismatched empty index.
    """
    global _collection_singleton
    if _collection_singleton is None:
        with _lock:
            if _collection_singleton is None:
                _collection_singleton = get_client().get_or_create_collection(
                    name=COLLECTION_NAME,
                    embedding_function=get_embedding_function(),
                    metadata=COLLECTION_METADATA,
                )
    return _collection_singleton


# ── Build ─────────────────────────────────────────────────────────────────────

def _flatten_metadata(md: Dict[str, Any]) -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    for k, v in md.items():
        if isinstance(v, list):
            flat[k] = "|".join(str(x) for x in v)
        elif isinstance(v, (str, int, float, bool)) or v is None:
            flat[k] = v if v is not None else ""
        else:
            flat[k] = str(v)
    return flat


def build_collection(reset: bool = True) -> chromadb.Collection:
    """Build (or rebuild) the vector collection from the knowledge base JSON files."""
    client = get_client()
    embed_fn = get_embedding_function()

    # Load the model before touching the store, so a download failure does not
    # leave a half-deleted collection behind.
    embed_fn.warm_up()

    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass  # collection did not exist — nothing to reset
        _reset_caches()

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn,
        metadata=COLLECTION_METADATA,
    )

    chunks: List[Chunk] = load_all_chunks()
    if not chunks:
        raise RuntimeError("load_all_chunks() returned no chunks — check data/*.json")

    collection.add(
        ids=[c.id for c in chunks],
        documents=[c.text for c in chunks],
        metadatas=[_flatten_metadata(c.metadata) for c in chunks],
    )

    global _collection_singleton
    _collection_singleton = collection

    print(
        f"Built collection '{COLLECTION_NAME}' with {collection.count()} chunks "
        f"using {embed_fn.name()} ({embed_fn.dimension()}-dim)."
    )
    return collection


def health_check() -> Dict[str, Any]:
    """
    Verifies the index is usable before a benchmark run.

    Catches the failure mode where the collection is missing or empty and
    semantic retrieval silently degrades to BM25-only with no error.
    """
    embed_fn = get_embedding_function()
    collection = get_collection()
    count = collection.count()
    if count == 0:
        raise RuntimeError(
            f"Vector collection '{COLLECTION_NAME}' is empty.\n"
            "  Fix: python src/setup_database.py"
        )
    return {
        "collection": COLLECTION_NAME,
        "chunks": count,
        "embedding_model": embed_fn.name(),
        "dimension": embed_fn.dimension(),
        "space": COLLECTION_METADATA["hnsw:space"],
    }


# ── Query ─────────────────────────────────────────────────────────────────────

def semantic_search(
    query: str,
    n_results: int = 8,
    where: Optional[Dict[str, Any]] = None,
    source_types: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Dense semantic search with optional metadata filtering."""
    collection = get_collection()

    final_where: Dict[str, Any] = dict(where) if where else {}
    if source_types:
        if len(source_types) == 1:
            final_where["source_type"] = source_types[0]
        else:
            final_where["source_type"] = {"$in": source_types}

    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        where=final_where if final_where else None,
    )

    ids = results["ids"][0]
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]
    return [
        {"id": ids[i], "text": docs[i], "metadata": metas[i], "distance": dists[i]}
        for i in range(len(ids))
    ]
