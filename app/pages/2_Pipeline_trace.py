"""
Pipeline trace -- the retrieval stages, the timings, and the model's raw reply.

The engineering view of the run the previous page summarised. Its main job is
the ranking table: BM25 and the embedding retriever disagree, RRF fuses them,
and the cross-encoder reorders the result. Seeing which recipe moved where is
the difference between "hybrid retrieval with reranking" as a phrase and as
something that visibly did work.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lunchbite import bootstrap  # noqa: F401
from lunchbite import components, service, theme

st.set_page_config(page_title="Pipeline trace - LunchBite",
                   page_icon="\N{LEFT-POINTING MAGNIFYING GLASS}", layout="wide")
components.inject_css()

st.title("Pipeline trace")

payload = service.require_last_run()
if payload is None:
    st.stop()

state: Dict[str, Any] = payload["state"]
result: Dict[str, Any] = payload["result"]
arm: str = payload["arm"]

st.caption(f"**{theme.ARMS[arm]['label']}** &middot; run id `{state.get('run_id', '?')}`")

# -- The query ----------------------------------------------------------------

st.markdown("### The query the profile became")
if state.get("query"):
    st.code(state["query"], language="text")
    st.caption(
        "Built from the profile by the first node. Note what it cannot express: "
        "an *absence*. Retrieval scores 0.00 NDCG@5 on gluten-free and milk-free "
        "queries, which is precisely why allergens are enforced symbolically "
        "instead of being trusted to the ranking."
    )
else:
    st.caption("This arm skips retrieval, so no query was built.")

# -- Ranking ------------------------------------------------------------------

def _rank_map(candidates: List[Dict[str, Any]]) -> Dict[str, int]:
    return {c.get("id", ""): i + 1 for i, c in enumerate(candidates or [])}


bm25 = state.get("bm25_candidates") or []
dense = state.get("semantic_candidates") or []
fused = state.get("fused_candidates") or []
reranked = state.get("reranked_candidates") or []

if fused:
    st.markdown("### How each retriever ranked the recipes")

    bm25_rank, dense_rank = _rank_map(bm25), _rank_map(dense)
    fused_rank, rerank_rank = _rank_map(fused), _rank_map(reranked)
    # `reranked_candidates` is empty when USE_CROSS_ENCODER_RERANKER is off, and
    # the generation pool then falls back to the fused list.
    reranking_ran = bool(reranked)

    final_order = reranked if reranking_ran else fused
    approved_ids = {c.get("id") for c in (state.get("generation_candidates") or [])}
    chosen_ids = {m.get("recipe_id") for m in (result.get("final_recommendations") or [])}

    rows = []
    for candidate in final_order:
        recipe_id = candidate.get("id", "")
        row = {
            "Recipe": candidate.get("name", recipe_id),
            "BM25": bm25_rank.get(recipe_id),
            "Embeddings": dense_rank.get(recipe_id),
            "Fused (RRF)": fused_rank.get(recipe_id),
        }
        if reranking_ran:
            row["Reranked"] = rerank_rank.get(recipe_id)
            row["Moved"] = (fused_rank.get(recipe_id, 0) - rerank_rank.get(recipe_id, 0)
                            if recipe_id in fused_rank and recipe_id in rerank_rank else None)
        row["Passed guardrail"] = recipe_id in approved_ids
        row["Recommended"] = recipe_id in chosen_ids
        rows.append(row)

    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    if reranking_ran:
        st.caption(
            "A blank cell means that retriever never surfaced the recipe at all. "
            "**Moved** is positions gained by the cross-encoder relative to the "
            "fused order: positive means it was promoted."
        )
    else:
        st.caption(
            "Cross-encoder reranking is switched off (USE_CROSS_ENCODER_RERANKER), "
            "so the fused order is what reached the guardrail. Measured across "
            "the eval set reranking lifts NDCG@5 from 0.622 to 0.727, so this is "
            "not a free saving."
        )
elif arm == "no_rag":
    st.info(
        "**No RAG.** Retrieval was skipped entirely -- the model was given the "
        "profile and nothing else. Anything it named came from its own weights, "
        "not from the corpus, which is why an id it returns may not exist."
    )

# -- Refine loop --------------------------------------------------------------

if state.get("refine_count"):
    st.markdown("### The search was widened")
    st.caption(
        f"The first pass returned fewer than the target number of menus, so "
        f"retrieval ran {state['refine_count']} more time(s) with a wider "
        f"candidate pool (now top-{state.get('retrieve_top_k', '?')})."
    )
    if state.get("refine_log"):
        st.dataframe(pd.DataFrame(state["refine_log"]), width="stretch",
                     hide_index=True)

# -- Reward ranking -----------------------------------------------------------

if state.get("reward_log"):
    st.markdown("### Verifiable-reward reranking")
    st.dataframe(pd.DataFrame(state["reward_log"]), width="stretch", hide_index=True)
    st.caption(
        "Each surviving menu scored against the verifiable reward, and whether "
        "the ordering changed. An arm that reranked nothing is not the same as "
        "one that never had two menus to choose between."
    )

# -- Timings ------------------------------------------------------------------

latency: Dict[str, float] = state.get("latency_ms") or {}
if latency:
    st.markdown("### Where the time went")
    # Ordered by pipeline position. Sorting these by magnitude would put the
    # fastest node first and destroy the only thing the sequence means.
    ordered = [(service.NODE_LABELS.get(node, node), round(latency[node], 1))
               for node in service.NODE_SEQUENCE if node in latency]
    ordered += [(service.NODE_LABELS.get(node, node), round(value, 1))
                for node, value in latency.items() if node not in service.NODE_SEQUENCE]

    frame = pd.DataFrame(ordered, columns=["Stage", "Milliseconds"])
    components.bar_chart(frame, "Stage", "Milliseconds",
                         "Time per stage, in pipeline order",
                         sort_by_value=False, value_format=".1f")

    measured = sum(latency.values())
    st.caption(
        f"Measured stages total {measured / 1000:.2f}s of a {payload['wall_ms'] / 1000:.2f}s "
        f"wall-clock run. The difference is graph overhead and any stage that "
        f"does not record a timing."
    )

# -- Raw output ---------------------------------------------------------------

st.markdown("### What the model actually returned")
raw: Optional[str] = state.get("llm_raw_output")
if not raw:
    st.caption("No model was called in this arm.")
else:
    proposed = state.get("proposed_menus") or []
    st.caption(
        f"{len(proposed)} menu(s) parsed out of this response. The generator is "
        f"asked for raw JSON; when it returns something unparseable the node "
        f"retries with the parse error quoted back to it."
    )
    with st.expander("Raw response"):
        st.code(raw, language="json")

if result.get("generation_error"):
    st.error(result["generation_error"])

with st.expander("Full terminal state (every key on PipelineState)"):
    st.caption(
        "Large lists are summarised by length. The profile is shown as stored, "
        "which is the dict form the graph receives."
    )
    summary = {}
    for key, value in state.items():
        if isinstance(value, list) and len(value) > 3:
            summary[key] = f"<{len(value)} items>"
        elif isinstance(value, str) and len(value) > 300:
            summary[key] = value[:300] + " …"
        else:
            summary[key] = value
    st.json(summary, expanded=False)
