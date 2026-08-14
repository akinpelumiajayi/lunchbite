"""
reranker.py -- Cross-encoder reranking stage.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2. Override with CROSS_ENCODER_MODEL.

WHY A CROSS-ENCODER, ON TOP OF NEURAL EMBEDDINGS:
  The bi-encoder in huggingface_embeddings.py encodes query and document
  independently and compares them with cosine similarity. That is fast enough
  to index the whole corpus, but it cannot model fine-grained interaction
  between query terms and document terms — negation in particular:
    "milk-free lunch" — the model has seen "milk" and "free" separately but
    does not reliably encode that "free" negates "milk".

  A cross-encoder attends to query and document together in one forward pass,
  so it can distinguish:
    "milk-free" vs "contains milk"  → low score
    "milk-free" vs "free from milk" → high score

  It is far too slow to run over a whole corpus, so it runs only over the
  fused top-N candidates from RRF.

Toggle with USE_CROSS_ENCODER_RERANKER in .env (default: true).
Disabling it skips this stage entirely and keeps RRF fusion order.

FAILURE POLICY:
  When the reranker is enabled but its model cannot be loaded, this raises
  rather than silently returning the unranked list — a run that quietly drops
  a retrieval stage would report numbers for a system that never ran.
"""

from __future__ import annotations

import os
import threading
from typing import Any, Callable, Dict, List, Optional

DEFAULT_CE_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_RERANK_TOP_K = 9


def _model_name() -> str:
    return os.environ.get("CROSS_ENCODER_MODEL", DEFAULT_CE_MODEL).strip() or DEFAULT_CE_MODEL


def rerank_top_k() -> int:
    try:
        return int(os.environ.get("RERANK_TOP_K", str(DEFAULT_RERANK_TOP_K)))
    except ValueError:
        return DEFAULT_RERANK_TOP_K


class RerankerUnavailable(RuntimeError):
    """Raised when the cross-encoder model cannot be loaded."""


def _default_text_of(candidate: Dict[str, Any]) -> str:
    return candidate.get("text") or candidate.get("name") or candidate.get("id", "")


class CrossEncoderReranker:
    """Re-scores retrieval candidates with a cross-encoder. Loaded lazily."""

    def __init__(self, model_name: Optional[str] = None) -> None:
        self._model_name = model_name or _model_name()
        self._model: Any = None
        self._lock = threading.Lock()

    def _ensure_model(self) -> Any:
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is not None:
                return self._model
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as e:
                raise RerankerUnavailable(
                    "sentence-transformers is not installed.\n"
                    "  Fix: pip install -r requirements.txt\n"
                    "  Or set USE_CROSS_ENCODER_RERANKER=false in .env to skip reranking."
                ) from e
            try:
                model = CrossEncoder(self._model_name)
            except Exception as e:
                raise RerankerUnavailable(
                    f"Could not load cross-encoder '{self._model_name}': "
                    f"{type(e).__name__}: {e}\n"
                    "  The first run needs network access to download the model\n"
                    "  (~90 MB, cached afterwards in ~/.cache/huggingface/).\n"
                    "  Or set USE_CROSS_ENCODER_RERANKER=false in .env to skip reranking."
                ) from e
            self._model = model
            return self._model

    def warm_up(self) -> None:
        self._ensure_model()

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: Optional[int] = None,
        text_of: Optional[Callable[[Dict[str, Any]], str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Returns candidates sorted by cross-encoder score, highest first, each
        annotated with a `reranker_score`. Candidates are not mutated.

        Args:
            query:      the retrieval query string
            candidates: dicts carrying at least 'id' and 'text'
            top_k:      how many to keep; None keeps all
            text_of:    how to derive the document side of each pair
        """
        if not candidates:
            return []

        model = self._ensure_model()
        get_text = text_of or _default_text_of
        pairs = [(query, get_text(c)) for c in candidates]
        scores = model.predict(pairs, show_progress_bar=False)

        scored = []
        for cand, score in zip(candidates, scores):
            enriched = dict(cand)
            enriched["reranker_score"] = float(score)
            scored.append(enriched)

        scored.sort(key=lambda c: c["reranker_score"], reverse=True)
        return scored[:top_k] if top_k else scored


# ── Module-level singleton ────────────────────────────────────────────────────

_singleton: Optional[CrossEncoderReranker] = None
_singleton_lock = threading.Lock()


def get_reranker() -> CrossEncoderReranker:
    """Shared instance — the model is loaded once per process."""
    global _singleton
    if _singleton is None:
        with _singleton_lock:
            if _singleton is None:
                _singleton = CrossEncoderReranker()
    return _singleton
