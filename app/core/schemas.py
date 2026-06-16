# app/core/schemas.py

from __future__ import annotations

from typing import Annotated, List
from pydantic import BaseModel, Field


class PlanStep(BaseModel):
    """A single discrete sub-query produced by the Planner agent."""

    index: int = Field(..., description="1-based ordinal position of this step.")
    sub_query: str = Field(..., description="The concrete sub-query to research.")


class PlanModel(BaseModel):
    """Structured output from the Planner node."""

    steps: List[PlanStep] = Field(
        ...,
        description="Ordered list of sub-queries that collectively answer the original query.",
    )


class AgentState(BaseModel):
    """
    Shared mutable state propagated across every node in the LangGraph pipeline.

    All fields are optional at construction time so the graph can initialise the
    state with only a `query` and let each node incrementally enrich it.
    """

    query: str = Field(..., description="The original user research question.")
    steps: List[str] = Field(
        default_factory=list,
        description="Human-readable audit trail of actions taken so far.",
    )
    plan: List[PlanStep] = Field(
        default_factory=list,
        description="Structured sub-queries emitted by the Planner.",
    )
    retrieved_documents: List[dict] = Field(
        default_factory=list,
        description="Raw document chunks returned by the Retriever.",
    )
    current_summary: str = Field(
        default="",
        description="Running synthesis produced by the Summariser.",
    )
    critic_feedback: str = Field(
        default="",
        description="Critique and improvement instructions from the Critic.",
    )
    retry_count: int = Field(
        default=0,
        description="Number of structural self-healing retries consumed so far.",
    )
    next_node: str = Field(
        default="",
        description="Routing hint written by each node to direct the conditional edge.",
    )

    class Config:
        # Allow extra keys so LangGraph can attach its own bookkeeping fields
        # without raising validation errors.
        extra = "allow"