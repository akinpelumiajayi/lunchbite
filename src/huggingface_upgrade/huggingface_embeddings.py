"""
huggingface_embeddings.py -- Neural sentence embeddings for the vector store.

This is now the ONLY embedding backend. The previous offline TF-IDF
implementation (local_embeddings.LocalTfidfEmbeddingFunction) has been removed.

Model: sentence-transformers/all-MiniLM-L6-v2 (384-dim, normalised).
Override with HF_EMBEDDING_MODEL in .env.

The model is downloaded once (~90 MB) to ~/.cache/huggingface/ and loaded from
cache on every subsequent run. The first run therefore needs network access.

WHY NEURAL EMBEDDINGS MATTER HERE:
  TF-IDF scored lexical overlap, so it could not represent negation:
    "milk-free lunch" and "contains milk" share the token "milk" and scored
    as similar. In an allergen-safety system that is exactly the wrong failure.
  all-MiniLM-L6-v2 encodes meaning, so "milk-free lunch" sits close to
  "dairy-free" and far from "cheesy pasta".
  reranker.CrossEncoderReranker then re-scores the fused top-N with full
  cross-attention, which resolves negation more precisely still.

FAILURE POLICY:
  If the model cannot be loaded, this raises. It does NOT silently fall back
  to a weaker retriever — a run that quietly swaps out the retriever produces
  benchmark numbers attributed to the wrong system.
"""

from __future__ import annotations

import os
import threading
from typing import Any, List, Optional

DEFAULT_HF_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_BATCH_SIZE = 32


def _model_name() -> str:
    return os.environ.get("HF_EMBEDDING_MODEL", DEFAULT_HF_MODEL).strip() or DEFAULT_HF_MODEL


class EmbeddingModelUnavailable(RuntimeError):
    """Raised when the sentence-transformers model cannot be loaded."""


class HuggingFaceEmbeddingFunction:
    """
    ChromaDB-compatible embedding function backed by sentence-transformers.

    Chroma calls `__call__(input=[...])`; LangChain-style callers can use
    `embed_documents` / `embed_query`. The model is loaded lazily on first
    use so that importing this module does not pull in torch.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        self._model_name = model_name or _model_name()
        self._batch_size = batch_size
        self._model: Any = None
        self._lock = threading.Lock()

    # ── Model loading ─────────────────────────────────────────────────────────

    def _ensure_model(self) -> Any:
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is not None:  # another thread won the race
                return self._model
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:
                raise EmbeddingModelUnavailable(
                    "sentence-transformers is not installed.\n"
                    "  Fix: pip install -r requirements.txt"
                ) from e
            try:
                model = SentenceTransformer(self._model_name)
            except Exception as e:
                raise EmbeddingModelUnavailable(
                    f"Could not load embedding model '{self._model_name}': "
                    f"{type(e).__name__}: {e}\n"
                    "  The first run needs network access to download the model\n"
                    "  (~90 MB, cached afterwards in ~/.cache/huggingface/).\n"
                    "  Override the model with HF_EMBEDDING_MODEL in .env."
                ) from e
            self._model = model
            return self._model

    def warm_up(self) -> None:
        """Force the model to load now, so download cost is paid up front."""
        self._ensure_model()

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def dimension(self) -> int:
        model = self._ensure_model()
        # Renamed in sentence-transformers 5.x; keep the old name as fallback.
        getter = getattr(model, "get_embedding_dimension", None) or \
            getattr(model, "get_sentence_embedding_dimension")
        return int(getter())

    # ── Embedding ─────────────────────────────────────────────────────────────

    def _encode(self, texts: List[str]) -> List[List[float]]:
        model = self._ensure_model()
        embeddings = model.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return embeddings.tolist()

    # Chroma's embedding-function protocol: documents go through __call__ and
    # queries through embed_query, both as `embed(input=[...]) -> [vector, ...]`.
    # Note this is NOT the LangChain convention, where embed_query takes a
    # single string and returns a single vector. Chroma is the caller here.

    def __call__(self, input: List[str]) -> List[List[float]]:
        if isinstance(input, str):  # tolerate a bare string
            input = [input]
        if not input:
            return []
        return self._encode(list(input))

    def embed_documents(self, input: List[str]) -> List[List[float]]:
        return self(input)

    def embed_query(self, input: List[str]) -> List[List[float]]:
        return self(input)

    def embed_one(self, text: str) -> List[float]:
        """Single string in, single vector out — for callers that want that shape."""
        if not isinstance(text, str):
            raise TypeError(f"embed_one expects a string, got {type(text).__name__}")
        return self._encode([text])[0]

    def name(self) -> str:
        return f"huggingface/{self._model_name}"


# ── Module-level singleton ────────────────────────────────────────────────────

_singleton: Optional[HuggingFaceEmbeddingFunction] = None
_singleton_lock = threading.Lock()


def get_embedding_function() -> HuggingFaceEmbeddingFunction:
    """Shared instance — loading the model once per process is the whole point."""
    global _singleton
    if _singleton is None:
        with _singleton_lock:
            if _singleton is None:
                _singleton = HuggingFaceEmbeddingFunction()
    return _singleton
