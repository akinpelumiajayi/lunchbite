"""
state.py -- Shared LangGraph state schema for every pipeline arm.

Pipeline modes (pipeline_mode field):
  "no_llm"         -- Aim 1 primary baseline: rule-based only, no LLM at all
  "neural_rag"     -- Aim 1 main system: retrieval + LLM, constraints as prompt text
  "neurosymbolic"  -- Aim 2: retrieval + symbolic pre-filter + LLM + symbolic post-filter
  "no_rag"         -- secondary reference control: LLM with profile only, no retrieval
  "reward_ranked"  -- neurosymbolic + best-of-N reranking on the verifiable reward

The symbolic gates (pre-filter + post-filter) only run in neurosymbolic mode.
The no_llm baseline never calls an LLM.
The no_rag control sends only the profile to the LLM without any retrieved context.

Every arm shares this schema so LangSmith traces are directly comparable
and the benchmark runner can use a single result record format.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict


class ChildProfileDict(TypedDict, total=False):
    age_years: int
    allergies: List[str]
    intolerances: List[str]
    likes: List[str]
    dislikes: List[str]
    school_nut_free: bool
    cultural_context: str
    # Diets the symbolic layer enforces (vegetarian / vegan / pescatarian /
    # halal / kosher). Previously carried only as prose in cultural_context,
    # which reached the generator's prompt and no gate.
    diet_requirements: List[str]


class RetrievedCandidate(TypedDict, total=False):
    id: str
    name: str
    score: float           # RRF fusion score
    text: str
    metadata: Dict[str, Any]
    raw_recipe: Dict[str, Any]
    citation: str          # source citation string for use in generated output
    reranker_score: float  # cross-encoder score; absent when reranking is off


class SymbolicFilterDecision(TypedDict):
    recipe_id: str
    recipe_name: str
    passed: bool
    reasons: List[str]
    warnings: List[str]


class MenuOption(TypedDict):
    recipe_id: str
    menu_name: str
    why_it_fits: str
    nutritional_rationale: str
    allergens_confirmed_absent: List[str]
    source_citation: str       # e.g. "Farris, A. (n.d.). PACK-IT Cookbook. Virginia Cooperative Extension."


class PostFilterResult(TypedDict):
    recipe_id: str
    menu_name: str
    survived: bool
    rejection_reason: Optional[str]
    symbolic_verified: bool


class PipelineState(TypedDict):
    # Input
    profile: ChildProfileDict
    pipeline_mode: str    # "no_llm" | "neural_rag" | "neurosymbolic" | "no_rag" | "reward_ranked"
    run_id: str

    # Retrieval (empty in no_rag mode)
    query: str
    bm25_candidates: List[RetrievedCandidate]
    semantic_candidates: List[RetrievedCandidate]
    fused_candidates: List[RetrievedCandidate]

    # Cross-encoder rerank of fused_candidates. Empty when reranking is disabled.
    # fused_candidates keeps its RRF order so the two rankings stay comparable.
    reranked_candidates: List[RetrievedCandidate]

    # Symbolic pre-filter (neurosymbolic only; [] in all others)
    symbolic_pre_filter_log: List[SymbolicFilterDecision]
    generation_candidates: List[RetrievedCandidate]

    # Generation (empty in no_llm mode)
    llm_raw_output: str
    proposed_menus: List[MenuOption]
    generation_error: Optional[str]

    # Symbolic post-filter (neurosymbolic only; [] in all others)
    symbolic_post_filter_log: List[PostFilterResult]
    final_menus: List[MenuOption]

    # Iterative retrieve-and-refine. `refine_count` is the number of EXTRA
    # retrieval passes taken beyond the first, so 0 means the pipeline answered
    # on its opening attempt. `retrieve_top_k` widens with each pass; it is part
    # of the state rather than a constant so a refined pass can look further
    # down the ranking than the pass that came up short.
    refine_count: int
    retrieve_top_k: int
    refine_log: List[Dict[str, Any]]

    # Verifiable-reward reranking (reward_ranked only; [] in all others).
    # Records the reward each surviving menu scored and whether the ordering
    # changed, so an arm that reranked nothing is distinguishable from one that
    # never had two menus to choose between.
    reward_log: List[Dict[str, Any]]

    # Eval metadata
    error: Optional[str]
    latency_ms: Dict[str, float]
