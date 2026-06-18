from __future__ import annotations

import logging
from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from app.core.nodes import critic_node, planner_node, retriever_node, summariser_node
from app.core.schemas import AgentState

logger = logging.getLogger(__name__)


def _router(state: AgentState) -> str:
    if state.retry_count > 3:
        logger.warning("[router] retry_count=%d exceeded — forcing END.", state.retry_count)
        return END
    target = state.next_node
    if target in ("retriever", "summariser", "critic"):
        return target
    return END


def build_graph() -> Any:
    checkpointer = MemorySaver()
    graph = StateGraph(AgentState)
    graph.add_node("planner", planner_node)
    graph.add_node("retriever", retriever_node)
    graph.add_node("summariser", summariser_node)
    graph.add_node("critic", critic_node)
    graph.set_entry_point("planner")
    graph.add_conditional_edges("planner", _router)
    graph.add_conditional_edges("retriever", _router)
    graph.add_conditional_edges("summariser", _router)
    graph.add_conditional_edges("critic", _router)
    compiled = graph.compile(checkpointer=checkpointer)
    logger.info("[graph] Research Copilot graph compiled successfully.")
    return compiled


research_graph = build_graph()