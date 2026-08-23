"""
nodes.py -- All LangGraph node functions for every pipeline arm.

Pipeline modes:
  no_llm       -- rule-based only: guardrail pre-filter → deterministic menu selection
  neural_rag   -- BM25+semantic retrieval → RRF fusion → LLM (constraints as prompt text)
  neurosymbolic-- retrieval → symbolic pre-filter → LLM → symbolic post-filter
  no_rag       -- LLM with profile only, no retrieval context (secondary reference)
  reward_ranked-- neurosymbolic + best-of-N reranking on the verifiable reward

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
from typing import Any, Callable, Dict, List, Optional, Tuple

_SRC_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SRC_DIR))

from functools import lru_cache

from rank_bm25 import BM25Okapi
from state import PipelineState, RetrievedCandidate, MenuOption
from document_loader import ALL_14_ALLERGENS, _load_json, load_recipe_chunks
from guardrails import ChildProfile, check_recipe_against_profile
from json_parsing import parse_menu_response
from rate_limit import (AUTH_FAILED, GENERATOR_QUOTA, is_auth_failure,
                        is_daily_quota, retry_after_secs)
from llm_provider import use_cross_encoder_reranker
from reward.checks import RewardContext
from reward.model import REWARD_VERSION, rank_menus
from vector_store import semantic_search as _vector_semantic_search
from huggingface_upgrade.reranker import get_reranker, rerank_top_k

_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
_VECTORDB_DIR = Path(__file__).resolve().parent.parent.parent / "vectordb"
_BM25_INDEX_PATH = _VECTORDB_DIR / "bm25_index.pkl"

# A generation "error" that is really a retrieval shortfall: the pre-filter
# rejected everything this pass surfaced. Named so should_refine can tell it
# apart from a dead key or an unparseable response, which re-retrieving cannot fix.
NO_CANDIDATES_ERROR = "No candidates available for generation."

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
        diet_requirements=pd.get("diet_requirements", []),
        # Carried through, not dropped. These two were absent here, so a ceiling
        # set on a ChildProfile reached the guardrail only when it was called
        # directly — every graph rebuilt the profile from its dict and lost
        # them, leaving a documented per-child override inert in the pipeline.
        max_sugar_g_override=pd.get("max_sugar_g_override"),
        max_salt_g_override=pd.get("max_salt_g_override"),
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

def _top_k(state: PipelineState) -> int:
    """Window for this pass. Widens on each refine; RETRIEVE_TOP_K on the first."""
    return int(state.get("retrieve_top_k") or RETRIEVE_TOP_K)


def bm25_retrieve(state: PipelineState) -> Dict[str, Any]:
    t0 = time.perf_counter()
    tokens = TOKEN_RE.findall(state["query"].lower())
    index, recipes = _get_bm25()
    scores = index.get_scores(tokens)
    ranked = sorted(range(len(recipes)), key=lambda i: scores[i], reverse=True)[:_top_k(state)]
    candidates = [_recipe_to_candidate(recipes[i], float(scores[i])) for i in ranked]
    return {"bm25_candidates": candidates, "latency_ms": _merge_latency(state, "bm25_retrieve", t0)}


# ── Node 3: semantic_retrieve ─────────────────────────────────────────────────

def semantic_retrieve(state: PipelineState) -> Dict[str, Any]:
    t0 = time.perf_counter()
    hits = _vector_semantic_search(
        state["query"], n_results=_top_k(state), source_types=["recipe"]
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


def _nutrition_note(cand: Dict[str, Any]) -> str:
    """
    One compact line naming which nutrient is over its per-lunch guideline.

    Deliberately terse. The figures themselves are already on the Nutrition line
    above it, and the prompt is charged against a 200k tokens/day budget that a
    full 30-case run very nearly fills — spelling the breach out in prose for
    every candidate costs a few hundred tokens a call to say nothing new.

    Empty when nothing is flagged, so the common case adds no tokens at all.
    """
    nutrients = sorted({
        "sugars" if "Sugar" in flag else "salt"
        for flag in (cand.get("nutrition_flags") or [])
    })
    if not nutrients:
        return ""
    return "\n  Above per-lunch guideline: " + ", ".join(nutrients)


def _nutrition_rank_key(cand: Dict[str, Any]) -> int:
    """0 for a candidate within every guideline, 1 for one that is over."""
    return 1 if cand.get("nutrition_flags") else 0


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
                    "warnings": result.warnings,
                    "nutrition_flags": result.nutrition_flags})
        if result.passed:
            # Advisories ride along with the candidate so the generator can
            # prefer the cleaner option, rather than being dropped here as they
            # were when a breach simply removed the recipe.
            cand = dict(cand)
            cand["nutrition_flags"] = result.nutrition_flags
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

    # Select top-1 by retrieval rank among the candidates that are within every
    # per-lunch nutrition guideline, falling back to a flagged one only when
    # nothing clean survived. Nutrition breaches stopped removing candidates
    # when the band gate became advisory (see guardrails.nutrition_gate), so
    # without this the deterministic baseline would return whatever ranked
    # highest, up to and including a 53 g-of-sugar recipe.
    approved = sorted(approved, key=_nutrition_rank_key)

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

# Marks a generation_error the runner should treat as fatal rather than as one
# more failed case. Matched as a prefix, so the provider's own message survives.
QUOTA_EXHAUSTED_PREFIX = "LLM daily quota exhausted"

# Kept distinct from the quota prefix so the runner can tell a run worth resuming
# after the reset from one that needs a new key before it is worth anything.
AUTH_FAILED_PREFIX = "LLM credentials rejected"

GENERATE_ATTEMPTS = 3

# Appended to the prompt when a response arrives whole but unreadable. An
# identical re-ask at temperature 0.1 tends to reproduce the same malformed
# output, so the retry names the defect rather than hoping for a better roll.
_MALFORMED_RETRY_NOTE = (
    "\n\nYour previous response could not be parsed as JSON ({error}). "
    "Return the same recommendation again as raw JSON only -- no prose and no "
    "markdown fences, every key a quoted string followed by a colon, and every "
    "member separated by a comma."
)


def _invoke_with_retry(
    llm: Any,
    prompt: str,
    parse: Optional[Callable[[str], Tuple[Any, Optional[str]]]] = None,
) -> Tuple[str, Optional[str]]:
    """
    One generation call, retrying the rate limits that will actually clear and,
    when `parse` is supplied, the responses that arrive whole but unreadable.

    Returns (raw_text, error). The error is None on success; `raw_text` is
    populated whenever the model actually said something, including on a final
    parse failure, so the run record keeps the evidence.

    A per-minute 429 is slept off for the interval Groq names and retried, which
    is what the raw `llm.invoke` here never did: `max_retries` on the ChatGroq
    client retries on its own schedule, not the provider's, and gives up while
    the window is still closed.

    A per-day 429 is different in kind. It latches GENERATOR_QUOTA so that every
    later case skips its call instead of rediscovering the same dead budget --
    on 2026-08-18 that rediscovery cost 39 calls and 13 lost case-runs -- and
    returns an error the runner recognises as fatal.

    A rejected credential is fatal in the same way and for longer: no wait fixes
    it. It was already not retried here, but only as a side effect of carrying no
    `try again in` hint -- so the run went on to spend 89 more calls proving the
    key was still expired. It now latches AUTH_FAILED explicitly.

    A response that arrives successfully but cannot be read used to end the
    case-run on its first try: the transport had succeeded, so this loop
    returned and `generate` recorded the parse failure as a dead run. That is a
    whole case lost to one bad roll of the decoder. In run 20260821_114840 the
    generator dropped a key from its second menu object --
    `{"recipe_id": "recipe_002", "Chicken and Hummus Veggie Wrap", ...}`, a bare
    value where `"menu_name":` belonged -- and ADV-11/no_rag was excluded from
    every rate in the report. The judge in evaluator.py has always re-asked on a
    parse failure; the generator now matches it.

    What is deliberately *not* retried is a valid refusal. `{"menu_options": []}`
    parses, so it returns as the answer it is -- re-asking there would badger the
    model out of a safe abstention, which is the one outcome this project most
    wants to preserve. Nor is the malformed text repaired: guessing which key a
    bare string was missing would put invented data behind an allergen claim.
    """
    from langchain_core.messages import HumanMessage

    if AUTH_FAILED.hit:
        return "", f"{AUTH_FAILED_PREFIX}: {AUTH_FAILED.detail}"

    if GENERATOR_QUOTA.hit:
        return "", f"{QUOTA_EXHAUSTED_PREFIX}: {GENERATOR_QUOTA.detail}"

    attempt_prompt = prompt
    for attempt in range(GENERATE_ATTEMPTS):
        try:
            response = llm.invoke([HumanMessage(content=attempt_prompt)])
            raw = (response.content or "").strip()
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            wait = retry_after_secs(msg)

            # Tested before the quota branch: an auth error carries no retry hint,
            # so it would fall through to the generic failure below and be
            # rediscovered by every later case.
            if is_auth_failure(msg):
                AUTH_FAILED.record(msg)
                return "", f"{AUTH_FAILED_PREFIX}: {msg}"

            if is_daily_quota(msg, wait):
                GENERATOR_QUOTA.record(msg, wait)
                return "", f"{QUOTA_EXHAUSTED_PREFIX}: {msg}"

            if wait is None or attempt == GENERATE_ATTEMPTS - 1:
                return "", f"LLM call failed: {msg}"

            # The provider knows when its own window reopens; the margin covers
            # clock skew between its clock and ours.
            time.sleep(wait + 0.5)
            continue

        if parse is None:
            return raw, None

        _, parse_error = parse(raw)
        if parse_error is None:
            return raw, None
        if attempt == GENERATE_ATTEMPTS - 1:
            return raw, f"Could not parse LLM response: {parse_error}"

        # No sleep: nothing about this failure is time-dependent.
        attempt_prompt = prompt + _MALFORMED_RETRY_NOTE.format(error=parse_error)

    return "", "LLM call failed: exhausted retries"



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
                "generation_error": NO_CANDIDATES_ERROR,
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
                    + _nutrition_note(c)
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

        # `parse_menu_response` runs inside the retry loop, so a response that
        # comes back unreadable is re-asked rather than ending the case-run.
        raw, gen_error = _invoke_with_retry(llm, prompt, parse=parse_menu_response)
        if gen_error is not None:
            # A parse failure is recorded, not silently turned into an empty
            # list. `menus = []` is exactly what a correct refusal looks like,
            # so the old handler let a run of malformed responses score as a
            # cautious system instead of a broken one.
            return {
                "llm_raw_output": raw,
                "proposed_menus": [],
                "generation_error": gen_error,
                "latency_ms": _merge_latency(state, "generate", t0),
            }

        menus, _ = parse_menu_response(raw)

        return {
            "llm_raw_output": raw,
            "proposed_menus": menus,
            "generation_error": None,
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


# ── Node 8: reward_rank [reward_ranked] ──────────────────────────────────────

def reward_rank(state: PipelineState) -> Dict[str, Any]:
    """
    Reorder the surviving menus by the verifiable reward, best first.

    This is the policy-improvement half of the RLHF loop. Classic RLHF updates
    the weights of the generator; that is not available here -- the generator is
    a hosted API model with no weight access -- so the improvement is applied at
    inference instead, as best-of-N over the menus the generator was already
    asked to produce. The prompt requests up to three, so ranking them costs no
    extra tokens where resampling would cost N times a daily budget the free
    tier does not have.

    Two properties this node must preserve, both load-bearing for the thesis:

    It reorders and never admits. The list it receives has already passed the
    symbolic post-filter, so every menu in it is safe before the reward sees
    it. The reward chooses among safe options; it cannot promote an unsafe one,
    because an unsafe one is not here. Safety stays deterministic and the
    learned-preference layer sits strictly outside it.

    It does not read attacker-controlled text. `trust_free_text=False` because
    at inference `profile["cultural_context"]` is whatever the caller sent, and
    in the benchmark that is where the adversarial injection is planted.
    """
    t0 = time.perf_counter()
    menus = list(state.get("final_menus") or [])

    ctx = RewardContext(
        profile=state.get("profile") or {},
        generation_candidates=[c["id"] for c in (state.get("generation_candidates") or [])],
        reranked_candidates=[c["id"] for c in (state.get("reranked_candidates") or [])],
        fused_candidates=[c["id"] for c in (state.get("fused_candidates") or [])],
        mode=state.get("pipeline_mode", ""),
        # The benchmark's hand-labelled unsafe ids are ground truth for scoring
        # a finished run, not something the pipeline can consult about itself.
        # Left empty here, so the in-graph reward rests on the guardrail and the
        # corpus alone -- the two authorities that exist at inference time.
        trust_free_text=False,
    )

    if len(menus) < 2:
        # Nothing to choose between. Recorded rather than skipped silently, so a
        # run where reranking never had an opportunity is distinguishable from
        # one where it ran and changed nothing.
        return {
            "final_menus": menus,
            "reward_log": [{"n_menus": len(menus), "reranked": False,
                            "reason": "fewer than two menus survived the gates"}],
            "latency_ms": _merge_latency(state, "reward_rank", t0),
        }

    ranked = rank_menus(menus, ctx)
    log = [{
        "recipe_id": r["menu"].get("recipe_id"),
        "reward": round(r["reward"].reward, 6),
        "weighted_score": round(r["reward"].weighted_score, 6),
        "gated": r["reward"].gated,
        "components": r["reward"].component_scores(),
        "original_rank": r["original_rank"],
        "new_rank": r["new_rank"],
    } for r in ranked]

    return {
        "final_menus": [r["menu"] for r in ranked],
        "reward_log": [{"n_menus": len(menus), "reranked": log[0]["original_rank"] != 1,
                        "reward_version": REWARD_VERSION, "ranking": log}],
        "latency_ms": _merge_latency(state, "reward_rank", t0),
    }


# ── Iterative retrieve-and-refine ────────────────────────────────────────────
#
# A single retrieval pass answers with whatever the first ranking happened to
# surface. When that pass comes up short -- the guardrail rejected most of the
# candidates, or the model returned fewer menus than asked for -- the pipeline
# used to simply return the short answer. Refining re-retrieves with a relaxed
# query and a wider window instead.
#
# What gets relaxed matters. Preference terms (likes, dislikes) are dropped
# first because they are soft: a child who would rather have pasta is better
# served an unloved safe lunch than no lunch. Allergy, intolerance, nut-free and
# diet terms are NEVER dropped -- widening the search must not widen what counts
# as acceptable, and the symbolic gates re-run on every pass regardless.

REFINE_MAX_PASSES = 2      # extra passes beyond the first attempt
REFINE_TARGET_MENUS = 3    # a pass returning fewer than this is "short"
REFINE_TOP_K_STEP = 6      # additional candidates admitted per pass


def _refine_query(profile: Dict[str, Any], refine_count: int) -> str:
    """Query for pass N: hard constraints always, preferences shed as N rises."""
    parts = [f"age {profile['age_years']}"]
    if profile.get("allergies"):
        parts.append(f"allergic to {', '.join(profile['allergies'])}")
    if profile.get("intolerances"):
        parts.append(f"{', '.join(profile['intolerances'])} intolerant")
    if profile.get("school_nut_free"):
        parts.append("nut-free school")
    if profile.get("diet_requirements"):
        parts.append(f"{', '.join(profile['diet_requirements'])} diet")
    if profile.get("cultural_context"):
        parts.append(profile["cultural_context"])
    # Pass 1 keeps likes and drops dislikes; pass 2 drops both.
    if refine_count < 2 and profile.get("likes"):
        parts.append(f"likes {', '.join(profile['likes'])}")
    return ", ".join(parts)


def widen_query(state: PipelineState) -> Dict[str, Any]:
    """One extra retrieval pass: relax the query, look further down the ranking."""
    t0 = time.perf_counter()
    n = int(state.get("refine_count") or 0) + 1
    k = int(state.get("retrieve_top_k") or RETRIEVE_TOP_K) + REFINE_TOP_K_STEP
    query = _refine_query(state["profile"], n)

    log = list(state.get("refine_log") or [])
    log.append({
        "pass": n,
        "reason": f"{len(state.get('final_menus') or [])} usable menu(s) "
                  f"< target {REFINE_TARGET_MENUS}",
        "new_query": query,
        "new_top_k": k,
    })
    return {
        "refine_count": n,
        "retrieve_top_k": k,
        "query": query,
        "refine_log": log,
        "latency_ms": _merge_latency(state, f"widen_query_{n}", t0),
    }


def should_refine(state: PipelineState) -> str:
    """
    Conditional edge: "widen_query" to take another pass, "done" to finish.

    Deliberately keyed on `final_menus` -- what the pipeline would actually
    return -- rather than on the model's raw proposal count. In the neuro-symbolic
    arm the post-filter runs before this, so a pass whose menus were all rejected
    as unsafe is correctly seen as short and retried, which is the case the loop
    exists for.
    """
    if int(state.get("refine_count") or 0) >= REFINE_MAX_PASSES:
        return "done"
    # A generation error is generally not a retrieval problem: re-retrieving
    # cannot fix a dead API key, an exhausted quota or an unparseable response,
    # and retrying would multiply the spend on a call that will fail again.
    #
    # NO_CANDIDATES_ERROR is the exception, and treating it as terminal defeated
    # the loop in the arm that needs it most. When the pre-filter rejects every
    # candidate this pass surfaced, a wider pass is exactly the remedy -- that is
    # a retrieval shortfall wearing a generation error's clothing.
    gen_error = state.get("generation_error")
    if state.get("error") or (gen_error and gen_error != NO_CANDIDATES_ERROR):
        return "done"
    menus = len(state.get("final_menus") or [])
    if menus >= REFINE_TARGET_MENUS:
        return "done"
    # Nothing left to find: the last pass already saw the whole corpus.
    if int(state.get("retrieve_top_k") or RETRIEVE_TOP_K) >= len(_recipes_by_id()):
        return "done"

    # Refine only when RETRIEVAL was the bottleneck.
    #
    # A short answer has two very different causes. If the gates rejected
    # everything the pass surfaced, a wider search may find something safe --
    # worth another pass. But if the model was handed an ample pool and chose to
    # return one option anyway, fetching more candidates cannot change that; it
    # re-runs BM25, dense retrieval, fusion and the cross-encoder to hand the
    # model options it already declined.
    #
    # Not a hypothetical: the mock generator returns exactly one menu whatever it
    # is given, so gating on menu count alone sent every case to the pass cap and
    # roughly tripled the run.
    if menus == 0:
        return "widen_query"
    if len(state.get("generation_candidates") or []) >= REFINE_TARGET_MENUS:
        return "done"
    return "widen_query"


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
