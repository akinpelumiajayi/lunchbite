"""
nodes.py -- All LangGraph node functions for all four pipelines.

Pipeline modes:
  no_llm       -- rule-based only: guardrail pre-filter → deterministic menu selection
  neural_rag   -- BM25+semantic retrieval → RRF fusion → LLM (constraints as prompt text)
  neurosymbolic-- retrieval → symbolic pre-filter → LLM → symbolic post-filter
  no_rag       -- LLM with profile only, no retrieval context (secondary reference)

Source citations from data_sources.json are embedded in every candidate
and carried through to generated menus so recommendations are traceable.
"""

from __future__ import annotations
import json
import os
import pickle
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_SRC_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SRC_DIR))

from functools import lru_cache

from rank_bm25 import BM25Okapi
from state import PipelineState, RetrievedCandidate, MenuOption
from document_loader import ALL_14_ALLERGENS, _load_json, load_recipe_chunks
from guardrails import ChildProfile, check_recipe_against_profile
from json_parsing import parse_menu_response
from llm_provider import use_cross_encoder_reranker
from vector_store import semantic_search as _vector_semantic_search
from huggingface_upgrade.reranker import get_reranker, rerank_top_k

_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
_VECTORDB_DIR = Path(__file__).resolve().parent.parent.parent / "vectordb"
_BM25_INDEX_PATH = _VECTORDB_DIR / "bm25_index.pkl"

TOKEN_RE = re.compile(r"[a-z0-9]+")
RRF_K = 60
RETRIEVE_TOP_K = 9
_bm25_cache: Dict[str, Any] = {}


# ── Cached corpus lookups ─────────────────────────────────────────────────────
# These were re-read from disk on every node invocation, so a single benchmark
# case re-parsed recipes.json roughly a dozen times.

@lru_cache(maxsize=1)
def _recipes_by_id() -> Dict[str, Dict[str, Any]]:
    return {r["id"]: r for r in _load_json("recipes.json")}


@lru_cache(maxsize=1)
def _recipe_chunk_texts() -> Dict[str, str]:
    """
    Full indexed chunk text per recipe id.

    Used as the document side of every cross-encoder pair so reranking sees
    identical text regardless of which retriever surfaced the candidate. It
    also carries the explicit "free from ..." sentence, which is exactly the
    negation signal the cross-encoder is here to resolve.
    """
    return {c.id: c.text for c in load_recipe_chunks()}


# ── BM25 ──────────────────────────────────────────────────────────────────────

def _get_bm25() -> Tuple[BM25Okapi, List[Dict[str, Any]]]:
    if "index" in _bm25_cache:
        return _bm25_cache["index"], _bm25_cache["recipes"]
    if _BM25_INDEX_PATH.exists():
        with open(_BM25_INDEX_PATH, "rb") as f:
            data = pickle.load(f)
        _bm25_cache["index"] = data["bm25"]
        _bm25_cache["recipes"] = data["recipes"]
        return _bm25_cache["index"], _bm25_cache["recipes"]
    recipes: List[Dict[str, Any]] = _load_json("recipes.json")
    corpus = [
        TOKEN_RE.findall((r["name"] + " " + " ".join(r["ingredients"]) + " " + r.get("description", "")).lower())
        for r in recipes
    ]
    index = BM25Okapi(corpus, k1=1.0, b=0.4)
    _bm25_cache["index"] = index
    _bm25_cache["recipes"] = recipes
    return index, recipes


def _profile_from_dict(pd: Dict[str, Any]) -> ChildProfile:
    return ChildProfile(
        age_years=pd["age_years"],
        allergies=pd.get("allergies", []),
        intolerances=pd.get("intolerances", []),
        likes=pd.get("likes", []),
        dislikes=pd.get("dislikes", []),
        school_nut_free=pd.get("school_nut_free", False),
        cultural_context=pd.get("cultural_context", ""),
    )


def _merge_latency(state: Dict[str, Any], key: str, t0: float) -> Dict[str, float]:
    existing: Dict[str, float] = dict(state.get("latency_ms") or {})
    existing[key] = (time.perf_counter() - t0) * 1000
    return existing


def _recipe_to_candidate(r: Dict[str, Any], score: float, text: str = "") -> RetrievedCandidate:
    citation = r.get("citation", r.get("source", ""))
    source_url = r.get("source_url", "")
    return {
        "id": r["id"],
        "name": r["name"],
        "score": score,
        "text": text or r.get("description", ""),
        "metadata": {
            "source_type": "recipe",
            "name": r["name"],
            "allergens_present": "|".join(r.get("allergens_present", [])),
            "source_id": r.get("source_id", ""),
            "source_url": source_url,
        },
        "raw_recipe": r,
        "citation": citation,
    }


# ── Node 1: build_query ───────────────────────────────────────────────────────

def build_query(state: PipelineState) -> Dict[str, Any]:
    t0 = time.perf_counter()
    p = state["profile"]
    parts = [f"age {p['age_years']}"]
    if p.get("allergies"):
        parts.append(f"allergic to {', '.join(p['allergies'])}")
    if p.get("intolerances"):
        parts.append(f"{', '.join(p['intolerances'])} intolerant")
    if p.get("school_nut_free"):
        parts.append("nut-free school")
    if p.get("likes"):
        parts.append(f"likes {', '.join(p['likes'])}")
    if p.get("dislikes"):
        parts.append(f"dislikes {', '.join(p['dislikes'])}")
    if p.get("cultural_context"):
        parts.append(p["cultural_context"])
    return {"query": ", ".join(parts), "latency_ms": _merge_latency(state, "build_query", t0)}


# ── Node 2: bm25_retrieve ─────────────────────────────────────────────────────

def bm25_retrieve(state: PipelineState) -> Dict[str, Any]:
    t0 = time.perf_counter()
    tokens = TOKEN_RE.findall(state["query"].lower())
    index, recipes = _get_bm25()
    scores = index.get_scores(tokens)
    ranked = sorted(range(len(recipes)), key=lambda i: scores[i], reverse=True)[:RETRIEVE_TOP_K]
    candidates = [_recipe_to_candidate(recipes[i], float(scores[i])) for i in ranked]
    return {"bm25_candidates": candidates, "latency_ms": _merge_latency(state, "bm25_retrieve", t0)}


# ── Node 3: semantic_retrieve ─────────────────────────────────────────────────

def semantic_retrieve(state: PipelineState) -> Dict[str, Any]:
    t0 = time.perf_counter()
    hits = _vector_semantic_search(
        state["query"], n_results=RETRIEVE_TOP_K, source_types=["recipe"]
    )
    raw_recipes = _recipes_by_id()
    candidates = []
    for h in hits:
        recipe = raw_recipes.get(h["id"])
        if recipe is None:
            # Index/corpus drift: an id in the vector store that no longer
            # exists in recipes.json. Previously this raised KeyError inside
            # _recipe_to_candidate and killed the whole run.
            print(f"    WARNING [semantic_retrieve]: '{h['id']}' is in the vector "
                  f"index but not in recipes.json — skipping. Rebuild with: "
                  f"python src/setup_database.py")
            continue
        cand = _recipe_to_candidate(recipe, max(0.0, 1.0 - float(h.get("distance", 1.0))), h["text"])
        cand["metadata"].update(h["metadata"])
        candidates.append(cand)
    return {"semantic_candidates": candidates, "latency_ms": _merge_latency(state, "semantic_retrieve", t0)}


# ── Node 4: rrf_fuse ──────────────────────────────────────────────────────────

def rrf_fuse(state: PipelineState) -> Dict[str, Any]:
    t0 = time.perf_counter()
    ranked_lists = [state.get("bm25_candidates") or [], state.get("semantic_candidates") or []]
    scores: Dict[str, float] = {}
    payloads: Dict[str, Any] = {}
    for ranked_list in ranked_lists:
        for rank, doc in enumerate(ranked_list, start=1):
            did = doc["id"]
            scores[did] = scores.get(did, 0.0) + 1.0 / (RRF_K + rank)
            if did not in payloads:
                payloads[did] = doc
    fused = sorted(payloads.values(), key=lambda d: scores[d["id"]], reverse=True)
    for d in fused:
        d["score"] = scores[d["id"]]
    return {"fused_candidates": fused, "latency_ms": _merge_latency(state, "rrf_fuse", t0)}


# ── Node 4b: rerank [cross-encoder] ───────────────────────────────────────────

def rerank(state: PipelineState) -> Dict[str, Any]:
    """
    Re-scores the RRF-fused candidates with a cross-encoder.

    RRF order is preserved in `fused_candidates` so the two rankings stay
    comparable in the results JSON; downstream nodes consume the reranked list
    via _generation_pool().

    Skipped (leaving reranked_candidates empty) when
    USE_CROSS_ENCODER_RERANKER=false.
    """
    t0 = time.perf_counter()
    fused = state.get("fused_candidates") or []

    if not use_cross_encoder_reranker() or not fused:
        return {"reranked_candidates": [], "latency_ms": _merge_latency(state, "rerank", t0)}

    chunk_texts = _recipe_chunk_texts()
    reranked = get_reranker().rerank(
        query=state["query"],
        candidates=fused,
        top_k=rerank_top_k(),
        text_of=lambda c: chunk_texts.get(c["id"], c.get("text", "")),
    )
    return {"reranked_candidates": reranked, "latency_ms": _merge_latency(state, "rerank", t0)}


def _generation_pool(state: PipelineState) -> List[RetrievedCandidate]:
    """Reranked candidates when the stage ran, otherwise the RRF-fused list."""
    return list(state.get("reranked_candidates") or state.get("fused_candidates") or [])


def _rank_basis(cand: Dict[str, Any]) -> str:
    """Describes which score put this candidate on top, for the no-LLM rationale."""
    if "reranker_score" in cand:
        return f"cross-encoder score: {cand['reranker_score']:.4f}"
    return f"RRF score: {cand.get('score', 0.0):.4f}"


# ── Node 5a: symbolic_prefilter [neurosymbolic] ───────────────────────────────

def _apply_guardrail(
    candidates: List[RetrievedCandidate], profile: ChildProfile
) -> Tuple[List[Dict[str, Any]], List[RetrievedCandidate]]:
    """Runs the deterministic gate over candidates. Returns (log, approved)."""
    log, approved = [], []
    for cand in candidates:
        result = check_recipe_against_profile(cand.get("raw_recipe", {}), profile)
        log.append({"recipe_id": cand["id"], "recipe_name": cand["name"],
                    "passed": result.passed, "reasons": result.reasons_for_rejection,
                    "warnings": result.warnings})
        if result.passed:
            approved.append(cand)
    return log, approved


def symbolic_prefilter(state: PipelineState) -> Dict[str, Any]:
    t0 = time.perf_counter()
    profile = _profile_from_dict(state["profile"])
    log, approved = _apply_guardrail(_generation_pool(state), profile)
    return {"symbolic_pre_filter_log": log, "generation_candidates": approved,
            "latency_ms": _merge_latency(state, "symbolic_prefilter", t0)}


# ── Node 5b: passthrough_candidates [neural_rag] ──────────────────────────────

def passthrough_candidates(state: PipelineState) -> Dict[str, Any]:
    return {"symbolic_pre_filter_log": [], "generation_candidates": _generation_pool(state)}


# ── Node 5c: no_llm_select [no_llm baseline] ──────────────────────────────────

def no_llm_select(state: PipelineState) -> Dict[str, Any]:
    """
    No-LLM baseline: applies the deterministic guardrail filter then selects
    the top safe candidate(s) by RRF score — no LLM involved at any step.
    This is the primary baseline Aim 1: purely rule-based safety + retrieval.
    """
    t0 = time.perf_counter()
    profile = _profile_from_dict(state["profile"])
    log, approved = _apply_guardrail(_generation_pool(state), profile)

    # Select top-1 by retrieval rank (most relevant safe candidate)
    menus: List[MenuOption] = []
    for cand in approved[:1]:
        r = cand.get("raw_recipe", {})
        n = r.get("nutrition_per_serving", {})
        menus.append({
            "recipe_id": cand["id"],
            "menu_name": r.get("name", cand["id"]),
            "why_it_fits": (
                f"Highest-scoring safe recipe ({_rank_basis(cand)}). "
                f"Meets all allergen and nutrition constraints for age {profile.age_years}. "
                f"Category: {r.get('meal_category', 'unknown')}."
            ),
            "nutritional_rationale": (
                f"{n.get('energy_kcal', '?')} kcal, {n.get('protein_g', '?')}g protein, "
                f"{n.get('sugars_g', '?')}g sugars, {n.get('salt_g', '?')}g salt per serving. "
                f"Selected by deterministic rule-based scoring — no LLM involved."
            ),
            "allergens_confirmed_absent": sorted(
                set(ALL_14_ALLERGENS) - set(a.lower() for a in r.get("allergens_present", []))
            ),
            "source_citation": cand.get("citation", r.get("source", "")),
        })

    return {
        "symbolic_pre_filter_log": log,
        "generation_candidates": approved,
        "llm_raw_output": "[No-LLM baseline: no LLM was called]",
        "proposed_menus": menus,
        "generation_error": None,
        "symbolic_post_filter_log": [],
        "final_menus": menus,
        "latency_ms": _merge_latency(state, "no_llm_select", t0),
    }


# ── Node 6: generate (LLM call) ───────────────────────────────────────────────

def make_generate_node(llm: Any):
    def generate(state: PipelineState) -> Dict[str, Any]:
        t0 = time.perf_counter()
        candidates = state.get("generation_candidates") or []
        mode = state.get("pipeline_mode", "neural_rag")
        p = state["profile"]

        # no_rag: build a synthetic single candidate from profile only
        if mode == "no_rag":
            candidates = []

        if not candidates and mode != "no_rag":
            return {
                "llm_raw_output": "",
                "proposed_menus": [],
                "generation_error": "No candidates available for generation.",
                "latency_ms": _merge_latency(state, "generate", t0),
            }

        restricted = sorted(set(p.get("allergies", []) + p.get("intolerances", [])))

        if mode == "no_rag":
            recipe_block = (
                "No recipe database available. Suggest healthy, allergen-safe lunch ideas "
                "based on the profile only. Do not invent specific recipes with exact nutrition data."
            )
            constraint_note = (
                f"Restrictions to avoid: {', '.join(restricted) if restricted else 'none'}. "
                f"School nut-free: {'yes' if p.get('school_nut_free') else 'no'}. "
                "WARNING: No recipe database is being used — you must check allergens from memory."
            )
        else:
            recipe_lines = []
            for c in candidates:
                r = c.get("raw_recipe", {})
                n = r.get("nutrition_per_serving", {})
                citation = c.get("citation", r.get("source", ""))
                recipe_lines.append(
                    f"- {r.get('name', '?')} (id: {r.get('id', '?')}) "
                    f"[Source: {citation}]\n"
                    f"  Ingredients: {', '.join(r.get('ingredients', []))}\n"
                    f"  Nutrition: {n.get('energy_kcal', '?')} kcal, "
                    f"{n.get('sugars_g', '?')}g sugars, {n.get('salt_g', '?')}g salt, "
                    f"{n.get('protein_g', '?')}g protein\n"
                    f"  Allergens: {', '.join(r.get('allergens_present', [])) or 'none declared'}\n"
                    f"  Tags: {', '.join(r.get('diet_tags', []))}"
                )
            recipe_block = "\n\n".join(recipe_lines)

            if mode == "neurosymbolic":
                constraint_note = (
                    "These recipes have been pre-verified safe by a deterministic rule-based system. "
                    f"Restrictions for reference: {', '.join(restricted) if restricted else 'none'}."
                )
            else:  # neural_rag
                constraint_note = (
                    "IMPORTANT: You MUST NOT recommend any recipe containing: "
                    f"{', '.join(restricted) if restricted else 'none stated'}. "
                    f"School nut-free: {'yes' if p.get('school_nut_free') else 'no'}. "
                    "Check the allergen list for each recipe carefully."
                )

        prompt = (
            "You are a children's school lunch recommendation assistant.\n\n"
            f"CHILD PROFILE:\n"
            f"- Age: {p['age_years']} years\n"
            f"- Allergies/intolerances: {', '.join(restricted) if restricted else 'none'}\n"
            f"- Likes: {', '.join(p.get('likes', [])) or 'not specified'}\n"
            f"- Dislikes: {', '.join(p.get('dislikes', [])) or 'not specified'}\n"
            f"- Cultural context: {p.get('cultural_context') or 'not specified'}\n\n"
            f"{constraint_note}\n\n"
            "AVAILABLE RECIPES:\n"
            f"{recipe_block}\n\n"
            "Recommend 1 to 3 lunch options. For each, explain why it suits this child, "
            "give a brief nutritional rationale with actual numbers, and cite the source.\n\n"
            "Respond ONLY with valid JSON (no markdown fences):\n"
            '{"menu_options": [{"recipe_id": "recipe_001", "menu_name": "...", '
            '"why_it_fits": "...", "nutritional_rationale": "...", '
            '"allergens_confirmed_absent": ["..."], "source_citation": "..."}]}\n\n'
            'If you cannot safely recommend any recipe, return {"menu_options": []}.'
        )

        try:
            from langchain_core.messages import HumanMessage
            response = llm.invoke([HumanMessage(content=prompt)])
            raw = response.content.strip()
        except Exception as e:
            return {
                "llm_raw_output": "",
                "proposed_menus": [],
                "generation_error": f"LLM call failed: {type(e).__name__}: {e}",
                "latency_ms": _merge_latency(state, "generate", t0),
            }

        # A parse failure is recorded, not silently turned into an empty list.
        # `menus = []` is exactly what a correct refusal looks like, so the old
        # handler let a run of malformed responses score as a cautious system
        # instead of a broken one.
        menus, parse_error = parse_menu_response(raw)

        return {
            "llm_raw_output": raw,
            "proposed_menus": menus,
            "generation_error": (f"Could not parse LLM response: {parse_error}"
                                 if parse_error else None),
            "latency_ms": _merge_latency(state, "generate", t0),
        }

    return generate


# ── Node 7a: symbolic_postfilter [neurosymbolic] ──────────────────────────────

def symbolic_postfilter(state: PipelineState) -> Dict[str, Any]:
    t0 = time.perf_counter()
    profile = _profile_from_dict(state["profile"])
    raw_recipes = _recipes_by_id()
    log, final = [], []
    for menu in state.get("proposed_menus") or []:
        rid = menu.get("recipe_id")
        recipe = raw_recipes.get(rid)
        if recipe is None:
            log.append({"recipe_id": rid or "unknown", "menu_name": menu.get("menu_name", ""),
                        "survived": False, "rejection_reason": f"'{rid}' not in database (hallucinated ID)",
                        "symbolic_verified": False})
            continue
        result = check_recipe_against_profile(recipe, profile)

        # The model's own allergen claim is checked against the recipe record.
        # Re-running the guardrail catches a recipe that is unsafe *for this
        # profile*; it does not catch the model asserting "contains no milk"
        # about a recipe whose allergens_present says otherwise. That assertion
        # is what a reader would rely on, so a contradiction is a rejection.
        claimed_absent = {a.lower() for a in (menu.get("allergens_confirmed_absent") or [])}
        actually_present = {a.lower() for a in (recipe.get("allergens_present") or [])}
        contradictions = sorted(claimed_absent & actually_present)

        if not result.passed:
            log.append({"recipe_id": rid, "menu_name": menu.get("menu_name", ""), "survived": False,
                        "rejection_reason": "; ".join(result.reasons_for_rejection),
                        "symbolic_verified": False})
            continue

        if contradictions:
            log.append({"recipe_id": rid, "menu_name": menu.get("menu_name", ""), "survived": False,
                        "rejection_reason": (
                            f"Model claimed absent but recipe lists as present: "
                            f"{', '.join(contradictions)}"),
                        "symbolic_verified": False,
                        "false_allergen_claim": contradictions})
            continue

        # Traceability: the citation the model hands back is not evidence until
        # it matches the citation attached to that recipe. A mismatch is repaired
        # rather than rejected — the recipe itself is safe, and silently shipping
        # a fabricated source is the actual harm. The flag makes fabrication
        # measurable instead of invisible.
        entry = {"recipe_id": rid, "menu_name": menu.get("menu_name", ""), "survived": True,
                 "rejection_reason": None, "symbolic_verified": True}
        expected_citation = recipe.get("citation", recipe.get("source", "")) or ""
        returned_citation = (menu.get("source_citation") or "").strip()
        if expected_citation and returned_citation != expected_citation:
            menu = {**menu, "source_citation": expected_citation}
            entry["citation_corrected"] = True
            entry["citation_returned"] = returned_citation
        log.append(entry)
        final.append(menu)
    return {"symbolic_post_filter_log": log, "final_menus": final,
            "latency_ms": _merge_latency(state, "symbolic_postfilter", t0)}


# ── Node 7b: passthrough_menus [neural_rag / no_rag] ─────────────────────────

def passthrough_menus(state: PipelineState) -> Dict[str, Any]:
    return {"symbolic_post_filter_log": [],
            "final_menus": list(state.get("proposed_menus") or [])}


# ── Node for no_rag: skip retrieval ──────────────────────────────────────────

def skip_retrieval(state: PipelineState) -> Dict[str, Any]:
    """No-RAG control: passes empty retrieval results straight to generation."""
    return {
        "query": "",
        "bm25_candidates": [],
        "semantic_candidates": [],
        "fused_candidates": [],
        "reranked_candidates": [],
        "symbolic_pre_filter_log": [],
        "generation_candidates": [],
    }
