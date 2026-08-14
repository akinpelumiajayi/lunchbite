"""
llm_provider.py -- Single source of truth for all LLM configuration.

Provider resolution (automatic, reads .env):
  1. Groq  -- if GROQ_API_KEY is set
  2. Ollama -- if Groq absent and Ollama is reachable at OLLAMA_BASE_URL

TWO SEPARATE MODELS are used:
  Generator model  -- fast, lower-cost (GROQ_MODEL / OLLAMA_MODEL)
                      used for lunch recommendation generation
  Judge model      -- larger, higher-quality (GROQ_JUDGE_MODEL / OLLAMA_JUDGE_MODEL)
                      used for LLM-as-judge evaluation metrics
  Using DIFFERENT models for generator and judge prevents self-preferencing
  bias in evaluation (a well-documented LLM evaluation problem).

No Anthropic dependency anywhere in this project.
"""

from __future__ import annotations
import os
import sys
import socket
import urllib.parse
from pathlib import Path
from typing import Any, Optional, Tuple

_ROOT = Path(__file__).resolve().parent.parent
_ENV_PATH = _ROOT / ".env"


# ── Load .env ─────────────────────────────────────────────────────────────────

def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=str(_ENV_PATH), override=False)
    except ImportError:
        if _ENV_PATH.exists():
            with open(_ENV_PATH, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, _, value = line.partition("=")
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    if key and key not in os.environ:
                        os.environ[key] = value


_load_dotenv()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_temperature(role: str = "generator") -> float:
    """Judge scoring defaults to 0.0 — a scorer should not sample."""
    if role == "judge":
        var, default = "JUDGE_TEMPERATURE", "0.0"
    else:
        var, default = "LLM_TEMPERATURE", "0.1"
    try:
        return float(os.environ.get(var, default))
    except ValueError:
        return float(default)


def _get_max_tokens(role: str = "generator") -> int:
    """
    Sized per role, because Groq bills the *reserved* max_tokens against the
    rate-limit budget, not the tokens actually generated.

    The judge shared the generator's 2000 — a value sized for multi-menu JSON —
    while its prompts ask for one score and one sentence (~30 tokens). Every
    judge call therefore cost ~2081 tokens against the 100,000 tokens/day cap
    on llama-3.3-70b-versatile, capping the judge at ~48 calls a day. A 30-case
    run needs ~321, so it exhausted the quota after 42 and recorded 279
    consecutive 429s; the published means rested on the surviving handful.
    """
    if role == "judge":
        var, default = "JUDGE_MAX_TOKENS", "200"
    else:
        var, default = "LLM_MAX_TOKENS", "2000"
    try:
        return int(os.environ.get(var, default))
    except ValueError:
        return int(default)


def _ollama_reachable() -> bool:
    base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    parsed = urllib.parse.urlparse(base_url)
    host = parsed.hostname or "localhost"
    port = parsed.port or 11434
    try:
        s = socket.create_connection((host, port), timeout=2)
        s.close()
        return True
    except (socket.timeout, ConnectionRefusedError, OSError):
        return False


# ── Provider builders ─────────────────────────────────────────────────────────

def _get_timeout() -> float:
    try:
        return float(os.environ.get("LLM_TIMEOUT_SECS", "60"))
    except ValueError:
        return 60.0


def _get_max_retries() -> int:
    try:
        return int(os.environ.get("LLM_MAX_RETRIES", "4"))
    except ValueError:
        return 4


def provider_available() -> bool:
    """True when some provider could serve a request. No client is constructed."""
    return bool(os.environ.get("GROQ_API_KEY", "").strip()) or _ollama_reachable()


def _try_groq(model: str, role: str = "generator") -> Optional[Any]:
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if not key:
        return None
    try:
        from langchain_groq import ChatGroq
        # Retries and a timeout matter here: an unretried 429 mid-benchmark used
        # to be recorded as a pipeline error and then dropped from the metrics,
        # quietly shrinking the sample a run was scored on.
        return ChatGroq(
            api_key=key,
            model=model,
            temperature=_get_temperature(role),
            max_tokens=_get_max_tokens(role),
            timeout=_get_timeout(),
            max_retries=_get_max_retries(),
        )
    except ImportError:
        raise RuntimeError("langchain-groq not installed. Run: pip install langchain-groq")


def _try_ollama(model: str, role: str = "generator") -> Optional[Any]:
    if not _ollama_reachable():
        return None
    base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    try:
        from langchain_ollama import ChatOllama
    except ImportError:
        try:
            from langchain_community.chat_models import ChatOllama  # type: ignore
        except ImportError:
            raise RuntimeError("langchain-ollama not installed. Run: pip install langchain-ollama")
    return ChatOllama(
        base_url=base_url,
        model=model,
        temperature=_get_temperature(role),
        num_predict=_get_max_tokens(role),
    )


# ── Public API ────────────────────────────────────────────────────────────────

def get_llm(prefer: Optional[str] = None) -> Tuple[Any, str]:
    """
    Returns (llm, provider_name) for the GENERATOR model.
    Generator: fast, lower-cost model for lunch recommendation text.
    """
    groq_model = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant").strip()
    ollama_model = os.environ.get("OLLAMA_MODEL", "llama3.2").strip()
    return _resolve(groq_model, ollama_model, prefer, role="generator")


def get_judge_llm(prefer: Optional[str] = None) -> Tuple[Any, str]:
    """
    Returns (llm, provider_name) for the JUDGE model.
    Judge: larger, higher-quality model for LLM-as-judge evaluation.
    MUST be a different model from the generator to avoid self-preferencing bias.
    """
    groq_model = os.environ.get("GROQ_JUDGE_MODEL", "llama-3.3-70b-versatile").strip()
    ollama_model = os.environ.get("OLLAMA_JUDGE_MODEL", "llama3.1").strip()
    return _resolve(groq_model, ollama_model, prefer, role="judge")


def _resolve(
    groq_model: str,
    ollama_model: str,
    prefer: Optional[str],
    role: str,
) -> Tuple[Any, str]:
    tried = []

    if prefer in (None, "groq"):
        llm = _try_groq(groq_model, role)
        if llm is not None:
            return llm, f"groq/{groq_model}"
        tried.append(f"groq (GROQ_API_KEY not set)")

    if prefer in (None, "ollama"):
        llm = _try_ollama(ollama_model, role)
        if llm is not None:
            base = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
            return llm, f"ollama/{ollama_model}@{base}"
        tried.append("ollama (not reachable at OLLAMA_BASE_URL)")

    raise RuntimeError(
        f"No LLM provider available for {role} model. Tried: {', '.join(tried)}\n\n"
        "Options:\n"
        "  A) Groq (cloud, free):  Add GROQ_API_KEY=gsk_... to .env\n"
        "                          Get key at https://console.groq.com\n"
        "  B) Ollama (local):      Install from https://ollama.com\n"
        "                          Run: ollama pull llama3.2 && ollama serve\n"
        "  C) Mock (testing only): python3 run_all.py --mock"
    )


def configure_langsmith(project_name: Optional[str] = None) -> bool:
    key = os.environ.get("LANGSMITH_API_KEY", "").strip()
    if not key:
        return False
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = key
    os.environ["LANGCHAIN_PROJECT"] = (
        project_name or os.environ.get("LANGCHAIN_PROJECT", "lunch-rag-benchmark")
    )
    return True


def use_cross_encoder_reranker() -> bool:
    """
    Cross-encoder reranking is on by default. Set USE_CROSS_ENCODER_RERANKER=false
    to skip the stage and keep RRF fusion order.

    Note there is no corresponding embeddings flag: neural embeddings are the
    only backend. The former USE_HUGGINGFACE_EMBEDDINGS switch was inert — it
    was read only for the status printout while retrieval always used TF-IDF,
    so runs reported a retriever they were not using.
    """
    return os.environ.get("USE_CROSS_ENCODER_RERANKER", "true").lower() == "true"


def print_provider_status() -> None:
    groq_key = os.environ.get("GROQ_API_KEY", "").strip()
    groq_model = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")
    groq_judge = os.environ.get("GROQ_JUDGE_MODEL", "llama-3.3-70b-versatile")
    ollama_model = os.environ.get("OLLAMA_MODEL", "llama3.2")
    ollama_judge = os.environ.get("OLLAMA_JUDGE_MODEL", "llama3.1")
    langsmith_key = os.environ.get("LANGSMITH_API_KEY", "").strip()
    reranker = use_cross_encoder_reranker()

    print("LLM provider configuration:")
    if groq_key:
        print(f"  Groq generator : CONFIGURED  model={groq_model}")
        print(f"  Groq judge     : CONFIGURED  model={groq_judge}  (separate from generator)")
    else:
        print("  Groq           : not configured  (set GROQ_API_KEY in .env)")

    if _ollama_reachable():
        base = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        print(f"  Ollama generator: RUNNING  model={ollama_model} @ {base}")
        print(f"  Ollama judge    : RUNNING  model={ollama_judge} @ {base}")
    else:
        print("  Ollama         : not running  (start: ollama serve)")

    if langsmith_key:
        project = os.environ.get("LANGCHAIN_PROJECT", "lunch-rag-benchmark")
        print(f"  LangSmith      : CONFIGURED  project={project}")
    else:
        print("  LangSmith      : not configured  (optional, set LANGSMITH_API_KEY in .env)")

    # Retrieval status. Reports configured models and the real state of the
    # index, without loading the models (that would trigger a download here).
    from huggingface_upgrade.huggingface_embeddings import DEFAULT_HF_MODEL
    from huggingface_upgrade.reranker import DEFAULT_CE_MODEL, rerank_top_k

    embed_model = os.environ.get("HF_EMBEDDING_MODEL", DEFAULT_HF_MODEL)
    ce_model = os.environ.get("CROSS_ENCODER_MODEL", DEFAULT_CE_MODEL)

    print(f"  Embeddings     : {embed_model}  (neural, sentence-transformers)")
    if reranker:
        print(f"  Cross-encoder  : ENABLED  model={ce_model}  top_k={rerank_top_k()}")
    else:
        print("  Cross-encoder  : disabled  (USE_CROSS_ENCODER_RERANKER=false)")

    try:
        from vector_store import get_collection
        print(f"  Vector index   : {get_collection().count()} chunks")
    except Exception as e:
        print(f"  Vector index   : unavailable ({type(e).__name__}) — run: python src/setup_database.py")
