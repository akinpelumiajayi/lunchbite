"""
build_graphs.py -- Builds all four LangGraph StateGraphs.

  build_no_llm_graph()            -> CompiledGraph  (primary baseline, Aim 1)
  build_neural_rag_graph(llm)     -> CompiledGraph  (main system, Aim 1)
  build_neurosymbolic_graph(llm)  -> CompiledGraph  (comparator, Aim 2)
  build_no_rag_graph(llm)         -> CompiledGraph  (secondary reference)

Why four pipelines:
  The research question requires isolating the effect of the symbolic
  constraint layer from all other variables. Having identical LLM and
  corpus across systems means any safety difference is attributable
  only to the presence or absence of the guardrail gates.

  no_llm  vs  neural_rag  → effect of adding an LLM (vs rule-only baseline)
  neural_rag vs neurosymbolic → effect of adding the symbolic constraint layer
  no_rag  (secondary) → effect of retrieval itself (LLM with no context)
"""

from __future__ import annotations
import sys
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langgraph.graph import StateGraph, END
from llm_provider import configure_langsmith
from state import PipelineState
from nodes import (
    build_query, bm25_retrieve, semantic_retrieve, rrf_fuse, rerank,
    symbolic_prefilter, passthrough_candidates,
    no_llm_select, skip_retrieval,
    make_generate_node,
    symbolic_postfilter, passthrough_menus,
)


def _add_retrieval_chain(graph) -> str:
    """
    Adds the shared retrieval stages and returns the name of the last node.

    All three retrieval pipelines use an identical front half — BM25 + dense
    semantic search, RRF fusion, then cross-encoder reranking — so the only
    difference between arms remains the symbolic layer, which is the whole
    point of the comparison.
    """
    graph.add_node("build_query", build_query)
    graph.add_node("bm25_retrieve", bm25_retrieve)
    graph.add_node("semantic_retrieve", semantic_retrieve)
    graph.add_node("rrf_fuse", rrf_fuse)
    graph.add_node("rerank", rerank)

    graph.set_entry_point("build_query")
    graph.add_edge("build_query", "bm25_retrieve")
    graph.add_edge("bm25_retrieve", "semantic_retrieve")
    graph.add_edge("semantic_retrieve", "rrf_fuse")
    graph.add_edge("rrf_fuse", "rerank")
    return "rerank"


def build_no_llm_graph(project_name: Optional[str] = None):
    """
    Primary baseline (Aim 1): purely deterministic.
    Retrieval → guardrail filter → top-1 safe candidate returned as menu.
    No LLM is ever called. Acts as the floor: zero hallucination, zero
    naturalness, but also zero LLM-induced safety violation.
    """
    configure_langsmith(project_name)
    graph = StateGraph(PipelineState)
    last = _add_retrieval_chain(graph)
    graph.add_node("no_llm_select", no_llm_select)

    graph.add_edge(last, "no_llm_select")
    graph.add_edge("no_llm_select", END)
    return graph.compile()


def build_neural_rag_graph(llm: Any, project_name: Optional[str] = None):
    """
    Neural-only RAG (Aim 1 main comparison system):
    Retrieval → RRF fusion → LLM (constraints as prompt text only).
    No deterministic filtering — all safety depends on LLM following instructions.
    """
    configure_langsmith(project_name)
    generate_node = make_generate_node(llm)
    graph = StateGraph(PipelineState)
    last = _add_retrieval_chain(graph)
    graph.add_node("passthrough_candidates", passthrough_candidates)
    graph.add_node("generate", generate_node)
    graph.add_node("passthrough_menus", passthrough_menus)

    graph.add_edge(last, "passthrough_candidates")
    graph.add_edge("passthrough_candidates", "generate")
    graph.add_edge("generate", "passthrough_menus")
    graph.add_edge("passthrough_menus", END)
    return graph.compile()


def build_neurosymbolic_graph(llm: Any, project_name: Optional[str] = None):
    """
    Neuro-symbolic RAG (Aim 2): retrieval → symbolic pre-filter
    → LLM (safe candidates only) → symbolic post-filter re-verification.
    The gates are pure Python, outside the LLM — injection-resistant.
    """
    configure_langsmith(project_name)
    generate_node = make_generate_node(llm)
    graph = StateGraph(PipelineState)
    last = _add_retrieval_chain(graph)
    graph.add_node("symbolic_prefilter", symbolic_prefilter)
    graph.add_node("generate", generate_node)
    graph.add_node("symbolic_postfilter", symbolic_postfilter)

    graph.add_edge(last, "symbolic_prefilter")
    graph.add_edge("symbolic_prefilter", "generate")
    graph.add_edge("generate", "symbolic_postfilter")
    graph.add_edge("symbolic_postfilter", END)
    return graph.compile()


def build_no_rag_graph(llm: Any, project_name: Optional[str] = None):
    """
    No-RAG control (secondary reference only):
    LLM with profile only, no retrieval context.
    Used to isolate the contribution of retrieval — NOT a fair comparison
    for safety since the LLM has no recipe data to ground claims in.
    """
    configure_langsmith(project_name)
    generate_node = make_generate_node(llm)
    graph = StateGraph(PipelineState)
    graph.add_node("skip_retrieval", skip_retrieval)
    graph.add_node("generate", generate_node)
    graph.add_node("passthrough_menus", passthrough_menus)

    graph.set_entry_point("skip_retrieval")
    graph.add_edge("skip_retrieval", "generate")
    graph.add_edge("generate", "passthrough_menus")
    graph.add_edge("passthrough_menus", END)
    return graph.compile()


# ── Backward compatibility aliases ────────────────────────────────────────────
# Tests and scripts that reference the old two-graph API still work.
build_baseline_graph = build_neural_rag_graph
