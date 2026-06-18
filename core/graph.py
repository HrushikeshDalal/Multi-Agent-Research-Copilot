# app/core/graph.py

from __future__ import annotations

import logging
from typing import Any, Dict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from app.core.nodes import (
    critic_node,
    planner_node,
    retriever_node,
    summariser_node,
)
from app.core.schemas import AgentState

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


def _router(state: AgentState) -> str:
    """
    Reads ``state.next_node`` to decide which node to visit next.

    Safety valve: if ``retry_count`` exceeds 3 the graph is forced to END
    regardless of what any node requested, preventing infinite revision loops.
    """
    if state.retry_count > 3:
        logger.warning(
            "[router] retry_count=%d exceeded threshold — forcing END.", state.retry_count
        )
        return END

    target = state.next_node
    if target in ("retriever", "summariser", "critic", "END", END):
        return target if target != "END" else END

    logger.warning("[router] Unknown next_node=%r — defaulting to END.", target)
    return END


# ---------------------------------------------------------------------------
# Graph factory
# ---------------------------------------------------------------------------


def build_graph() -> Any:
    """
    Constructs and compiles the LangGraph ``StateGraph`` for the Multi-Agent
    Research Copilot.

    Returns a compiled graph instance that exposes ``.ainvoke()`` and
    ``.astream()`` methods compatible with the FastAPI streaming endpoint.
    """
    checkpointer = MemorySaver()

    graph = StateGraph(AgentState)

    # ── Register nodes ──────────────────────────────────────────────────────
    graph.add_node("planner", planner_node)
    graph.add_node("retriever", retriever_node)
    graph.add_node("summariser", summariser_node)
    graph.add_node("critic", critic_node)

    # ── Entry point ─────────────────────────────────────────────────────────
    graph.set_entry_point("planner")

    # ── Edges ───────────────────────────────────────────────────────────────
    # After planner runs the router decides where to go (normally → retriever)
    graph.add_conditional_edges("planner", _router)

    # After retriever runs the router decides (normally → summariser)
    graph.add_conditional_edges("retriever", _router)

    # After summariser runs the router decides (normally → critic)
    graph.add_conditional_edges("summariser", _router)

    # After critic the router may loop back to summariser or terminate
    graph.add_conditional_edges("critic", _router)

    # ── Compile with in-memory checkpointer ─────────────────────────────────
    compiled = graph.compile(checkpointer=checkpointer)
    logger.info("[graph] Research Copilot graph compiled successfully.")
    return compiled


# Module-level singleton — imported by the FastAPI routes
research_graph = build_graph()