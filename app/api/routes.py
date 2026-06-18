# app/api/routes.py

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import AsyncGenerator

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.core.graph import research_graph
from app.core.schemas import AgentState

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["research"])


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------


class ResearchRequest(BaseModel):
    query: str
    session_id: str | None = None


# ---------------------------------------------------------------------------
# SSE helper
# ---------------------------------------------------------------------------


def _sse_event(event_type: str, data: dict) -> str:
    """Format a single Server-Sent Event frame."""
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event_type}\ndata: {payload}\n\n"


# ---------------------------------------------------------------------------
# Streaming endpoint
# ---------------------------------------------------------------------------


@router.post("/research/stream")
async def research_stream(request: ResearchRequest) -> StreamingResponse:
    """
    Accepts a research *query* and streams back Server-Sent Events (SSE) for
    every node transition and the final markdown report.

    Event types
    -----------
    ``node_start``   – emitted just before a node begins execution
    ``step``         – emitted after each node completes, carrying the latest
                       step message and partial state
    ``result``       – emitted once with the final markdown summary
    ``error``        – emitted on unhandled exceptions
    ``done``         – terminal event signalling stream completion
    """
    session_id = request.session_id or str(uuid.uuid4())
    logger.info("[routes] /research/stream  session=%s  query=%r", session_id, request.query)

    initial_state = AgentState(query=request.query)

    async def event_generator() -> AsyncGenerator[str, None]:
        config = {"configurable": {"thread_id": session_id}}

        try:
            async for chunk in research_graph.astream(
                initial_state.model_dump(),
                config=config,
                stream_mode="updates",
            ):
                # chunk is a dict: { node_name: updated_state_dict }
                for node_name, state_update in chunk.items():
                    if node_name == "__end__":
                        continue

                    # ── node_start ────────────────────────────────────────
                    yield _sse_event(
                        "node_start",
                        {"node": node_name, "session_id": session_id},
                    )
                    # Small breathing room so the client can paint before data
                    await asyncio.sleep(0)

                    # ── step ─────────────────────────────────────────────
                    steps: list[str] = state_update.get("steps", [])
                    latest_step = steps[-1] if steps else f"{node_name} completed."
                    retry_count = state_update.get("retry_count", 0)

                    yield _sse_event(
                        "step",
                        {
                            "node": node_name,
                            "message": latest_step,
                            "retry_count": retry_count,
                            "session_id": session_id,
                        },
                    )
                    await asyncio.sleep(0)

                    # ── result (critic accepted / forced END) ─────────────
                    next_node = state_update.get("next_node", "")
                    if node_name == "critic" and next_node in ("END", ""):
                        summary: str = state_update.get("current_summary", "")
                        yield _sse_event(
                            "result",
                            {
                                "summary": summary,
                                "steps": steps,
                                "session_id": session_id,
                            },
                        )
                        await asyncio.sleep(0)

            yield _sse_event("done", {"session_id": session_id})

        except Exception as exc:  # noqa: BLE001
            logger.exception("[routes] Unhandled error in stream: %s", exc)
            yield _sse_event("error", {"detail": str(exc), "session_id": session_id})
            yield _sse_event("done", {"session_id": session_id})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


@router.get("/health")
async def health() -> dict:
    return {"status": "ok"}