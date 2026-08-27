"""
service.py -- The only module in the app that touches the pipeline.

Two rules hold here and nowhere else in the app:

1. **The display contract is not reimplemented.** A terminal state becomes a
   result through `main.shape_result`, the same function `recommend_lunches`
   ends in. `src/main.py`'s own docstring records what happened last time two
   implementations of this mapping existed -- the CLI and the benchmark
   measured different systems, so no reported number described what a user
   actually got. The dashboard is not going to be the third.

2. **Graphs are cached per arm.** Compiling one loads MiniLM and a
   cross-encoder, ~90 MB each, so an uncached build would pay that on every
   Streamlit rerun.
"""

from __future__ import annotations

import time
from dataclasses import asdict
from typing import Any, Callable, Dict, List, Optional, Tuple

import streamlit as st

from lunchbite import bootstrap  # noqa: F401  (path setup must precede src imports)

from document_loader import recipes_by_id
from guardrails import ChildProfile
from main import shape_result

# The order nodes run in, for progress display and for ordering the latency
# chart by pipeline position rather than by magnitude.
NODE_SEQUENCE: List[str] = [
    "build_query", "bm25_retrieve", "semantic_retrieve", "rrf_fuse", "rerank",
    "symbolic_prefilter", "passthrough_candidates", "skip_retrieval",
    "no_llm_select", "generate", "symbolic_postfilter", "passthrough_menus",
    "reward_rank", "widen_query",
]

NODE_LABELS: Dict[str, str] = {
    "build_query": "Building the query",
    "bm25_retrieve": "Lexical retrieval (BM25)",
    "semantic_retrieve": "Semantic retrieval (embeddings)",
    "rrf_fuse": "Fusing the two rankings (RRF)",
    "rerank": "Cross-encoder rerank",
    "symbolic_prefilter": "Symbolic guardrail",
    "passthrough_candidates": "Skipping the guardrail (no symbolic gate)",
    "skip_retrieval": "Skipping retrieval",
    "no_llm_select": "Rule-based selection",
    "generate": "Asking the model",
    "symbolic_postfilter": "Verifying the model's claims",
    "passthrough_menus": "Accepting the model's claims unchecked",
    "reward_rank": "Reward reranking",
    "widen_query": "Widening the search",
}


class PipelineUnavailable(RuntimeError):
    """Raised when the pipeline cannot run at all -- no LLM, or no index."""

    def __init__(self, message: str, remedy: str = "") -> None:
        super().__init__(message)
        self.remedy = remedy


# -- Corpus -------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def recipe_index() -> Dict[str, Dict[str, Any]]:
    """The recipe corpus by id. Cached; treat the returned dicts as read-only."""
    return recipes_by_id()


def recipe_for(menu: Dict[str, Any]) -> Dict[str, Any]:
    """
    The corpus record behind a MenuOption.

    A MenuOption carries six keys and none of them are ingredients, nutrition,
    cost or a source URL -- all of that comes from this join. Returns {} when
    the id is unknown, which is itself worth showing: in the `no_rag` arm the
    model has no recipe list and can return an id that does not exist.
    """
    return recipe_index().get(menu.get("recipe_id", ""), {})


# -- Environment checks -------------------------------------------------------

@st.cache_data(ttl=30, show_spinner=False)
def provider_status() -> Dict[str, Any]:
    """Which LLM provider is configured, without ever reading a key's value."""
    import os

    from llm_provider import provider_available

    groq_key = bool(os.environ.get("GROQ_API_KEY", "").strip())
    status: Dict[str, Any] = {
        "available": False,
        "provider": None,
        "groq_key_present": groq_key,
        "model": os.environ.get("GROQ_MODEL") if groq_key else os.environ.get("OLLAMA_MODEL"),
        "error": None,
    }
    try:
        status["available"] = provider_available()
    except Exception as exc:
        status["error"] = str(exc)
    if status["available"]:
        try:
            from llm_provider import get_llm

            _, name = get_llm()
            status["provider"] = name
        except Exception as exc:
            status["available"] = False
            status["error"] = str(exc)
    return status


@st.cache_resource(show_spinner=False)
def ensure_index() -> Dict[str, Any]:
    """
    Builds the vector index on first use if it is missing or empty.

    Locally this never fires -- `python src/setup_database.py` has already run
    and vectordb/ is on disk. It exists for the deployed app, where it has to:
    vectordb/ is gitignored (it is generated, and regenerating it is the point
    of setup_database.py), so a cloud container starts with no index and no
    shell to build one from.

    Building has to happen here rather than being skipped, because
    `get_collection` calls `get_or_create_collection` -- with no index the app
    would not fail, it would come up holding an empty collection and quietly
    serve BM25-only retrieval while presenting itself as the full pipeline.
    That is the failure `health_check` was written to catch, and the deployed
    app is exactly where nobody would be watching a console to catch it.

    Cheap: the corpus is the three JSON files under data/, a few hundred chunks,
    and the embedding model it needs is the same one the first query loads
    anyway. Cached as a resource, so it is once per server, not once per rerun.
    """
    from vector_store import build_collection, get_collection

    if get_collection().count() > 0:
        return {"built": False, "chunks": get_collection().count()}

    collection = build_collection(reset=True)
    return {"built": True, "chunks": collection.count()}


@st.cache_data(ttl=30, show_spinner=False)
def index_status() -> Dict[str, Any]:
    """ChromaDB collection health. `ok=False` means the index needs building."""
    try:
        from vector_store import health_check

        ensure_index()
        return {"ok": True, "detail": health_check(), "error": None}
    except Exception as exc:
        return {"ok": False, "detail": {}, "error": str(exc)}


def shadow_warnings() -> List[str]:
    """
    Process env vars that override the .env value for the same key.

    `llm_provider` writes this to stderr, which under Streamlit goes to a
    terminal nobody is looking at. It burned a real run once; the dashboard
    surfaces it where the run is actually being started.
    """
    import os

    from llm_provider import _DOTENV_SHADOW_WATCH, _load_dotenv

    _load_dotenv()
    out: List[str] = []
    for key in _DOTENV_SHADOW_WATCH:
        env_value = os.environ.get(key)
        if not env_value:
            continue
        file_value = _dotenv_value(key)
        if file_value and file_value != env_value:
            out.append(
                f"{key} is set in this shell and differs from the value in .env. "
                f"The shell value wins -- the run will not use the .env key."
            )
    return out


def _dotenv_value(key: str) -> Optional[str]:
    """Reads one key straight from .env without mutating the environment."""
    path = bootstrap.ROOT / ".env"
    if not path.exists():
        return None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        if name.strip() == key:
            return value.strip().strip("'\"")
    return None


# -- Graphs -------------------------------------------------------------------

@st.cache_resource(show_spinner=False, max_entries=5)
def compiled_graph(arm: str):
    """
    The compiled LangGraph for one arm. Cached -- compiling loads two models.

    `no_llm` is built without touching `get_llm`, so it stays available when no
    key is configured. That matters beyond convenience: the key rotation in
    plan.md is still open, and an app that dies without a key would be
    undemonstrable for the one arm that needs no key at all.
    """
    from graphs.build_graphs import (build_neural_rag_graph, build_neurosymbolic_graph,
                                     build_no_llm_graph, build_no_rag_graph,
                                     build_reward_ranked_graph)

    if arm == "no_llm":
        return build_no_llm_graph()

    builders: Dict[str, Callable[..., Any]] = {
        "neurosymbolic": build_neurosymbolic_graph,
        "neural_rag": build_neural_rag_graph,
        "reward_ranked": build_reward_ranked_graph,
        "no_rag": build_no_rag_graph,
    }
    if arm not in builders:
        raise PipelineUnavailable(f"Unknown pipeline arm: {arm!r}")

    from llm_provider import get_llm

    try:
        llm, _ = get_llm()
    except RuntimeError as exc:
        raise PipelineUnavailable(
            str(exc),
            remedy="Set GROQ_API_KEY in .env, or run Ollama locally, or switch "
                   "to the 'No LLM (rule-based)' arm, which needs neither.",
        ) from exc
    return builders[arm](llm)


def _initial_state(profile: ChildProfile, arm: str, run_id: str) -> Dict[str, Any]:
    """
    Mirrors `main._initial_state`, plus the keys the other arms read.

    `refine_count` / `retrieve_top_k` / `refine_log` / `reward_log` are declared
    on PipelineState but `main._initial_state` omits them because only the
    neurosymbolic graph runs there. `reward_ranked` reads reward_log and the
    refine loop reads refine_count, so an app offering those arms seeds them.
    """
    return {
        "profile": asdict(profile),
        "pipeline_mode": arm,
        "run_id": run_id,
        "query": "", "bm25_candidates": [], "semantic_candidates": [],
        "fused_candidates": [], "reranked_candidates": [],
        "symbolic_pre_filter_log": [],
        "generation_candidates": [], "llm_raw_output": "",
        "proposed_menus": [], "generation_error": None,
        "symbolic_post_filter_log": [], "final_menus": [],
        "refine_count": 0, "retrieve_top_k": 0, "refine_log": [],
        "reward_log": [],
        "error": None, "latency_ms": {},
    }


def run(profile: ChildProfile, arm: str = "neurosymbolic",
        on_node: Optional[Callable[[str, int], None]] = None) -> Dict[str, Any]:
    """
    Runs one arm and returns {"result", "state", "arm", "wall_ms"}.

    Streams rather than invokes so `on_node` can report progress as each node
    finishes -- the generate node alone can take several seconds, and a bare
    spinner gives no sign of whether the retrieval stages even got that far.
    The accumulated state is equivalent to what `.invoke()` returns: the
    "updates" stream yields exactly the partial dicts each node returned, which
    is what the reducer would have merged anyway.

    `run_id` is prefixed `app-` so LangSmith traces from the dashboard stay
    distinguishable from benchmark and CLI runs.
    """
    from uuid import uuid4

    graph = compiled_graph(arm)
    state = _initial_state(profile, arm, f"app-{uuid4().hex[:8]}")

    started = time.perf_counter()
    seen = 0
    for update in graph.stream(state, stream_mode="updates"):
        for _node_name, partial in (update or {}).items():
            if isinstance(partial, dict):
                state.update(partial)
            seen += 1
            if on_node is not None:
                on_node(_node_name, seen)
    wall_ms = (time.perf_counter() - started) * 1000.0

    return {
        "result": shape_result(state, profile),
        "state": state,
        "arm": arm,
        "wall_ms": wall_ms,
    }


# -- Session ------------------------------------------------------------------

LAST_RUN_KEY = "lb_last_run"


def store_run(run_payload: Dict[str, Any]) -> None:
    st.session_state[LAST_RUN_KEY] = run_payload


def last_run() -> Optional[Dict[str, Any]]:
    """The most recent run, so the diagnostic pages never re-invoke the LLM."""
    return st.session_state.get(LAST_RUN_KEY)


def require_last_run() -> Optional[Dict[str, Any]]:
    """Shared empty state for the pages that read a run they did not start."""
    payload = last_run()
    if payload is None:
        st.info(
            "No run yet. Open **LunchBite** in the sidebar, build a profile and "
            "press **Find lunches** -- this page reads that run without "
            "repeating it."
        )
        return None
    return payload


# -- Result helpers -----------------------------------------------------------

def fatal_error_kind(message: Optional[str]) -> Optional[str]:
    """
    Classifies a generation_error as 'quota', 'auth' or None.

    A rotated key and an exhausted daily quota both stop the run and need
    opposite responses -- wait, versus go and get a new key -- so the app tells
    them apart using the prefixes the nodes already mark them with.
    """
    if not message:
        return None
    from graphs.nodes import AUTH_FAILED_PREFIX, QUOTA_EXHAUSTED_PREFIX

    if message.startswith(QUOTA_EXHAUSTED_PREFIX):
        return "quota"
    if message.startswith(AUTH_FAILED_PREFIX):
        return "auth"
    return None


def funnel_counts(state: Dict[str, Any], result: Dict[str, Any]) -> List[Tuple[str, int]]:
    """Stage-by-stage survivor counts, in pipeline order."""
    pre_log = state.get("symbolic_pre_filter_log") or []
    return [
        ("Retrieved", len(state.get("fused_candidates") or [])),
        ("Checked by the guardrail", len(pre_log)),
        ("Passed the guardrail", len(state.get("generation_candidates") or [])),
        ("Proposed by the model", len(state.get("proposed_menus") or [])),
        ("Survived verification", len(result.get("final_recommendations") or [])),
    ]


def warnings_by_recipe(state: Dict[str, Any],
                       include_profile_level: bool = False) -> Dict[str, List[str]]:
    """
    Guardrail warnings keyed by the recipe they were raised against.

    Profile-level warnings are dropped by default. The guardrail attaches them
    to every candidate it checks, so keeping them would repeat one sentence on
    every card -- and they are about the profile, which the form already reports
    before the run. Pass `include_profile_level=True` for the raw log.
    """
    drop = set() if include_profile_level else set(profile_warnings(state))
    out: Dict[str, List[str]] = {}
    for entry in state.get("symbolic_pre_filter_log") or []:
        warnings = [w for w in (entry.get("warnings") or []) if w not in drop]
        if warnings:
            out[entry.get("recipe_id", "")] = warnings
    return out


def profile_warnings(state: Dict[str, Any]) -> List[str]:
    """
    Warnings that are about the profile rather than about one recipe.

    The guardrail raises two kinds of warning through one list. Some describe
    the profile -- an allergy term outside the 14-allergen vocabulary, a diet
    that can only be checked as an ingredient exclusion -- and those repeat
    identically on every candidate because they depend on nothing else. Others
    describe a single recipe: a side item carrying a restricted allergen, a
    sugar figure over the band.

    Telling them apart by *which recipes carry them* rather than by matching
    their wording keeps this from breaking the next time a message is reworded,
    which plan.md §1.4 records as having happened before. A warning on every
    checked candidate is a statement about the profile; anything less is about
    the recipes that carry it.

    Needs at least two candidates to distinguish the two, so with fewer it
    reports nothing and lets the per-recipe view carry everything.
    """
    log = state.get("symbolic_pre_filter_log") or []
    if len(log) < 2:
        return []

    counts: Dict[str, int] = {}
    order: List[str] = []
    for entry in log:
        for warning in set(entry.get("warnings") or []):
            if warning not in counts:
                order.append(warning)
            counts[warning] = counts.get(warning, 0) + 1
    return [w for w in order if counts[w] == len(log)]
