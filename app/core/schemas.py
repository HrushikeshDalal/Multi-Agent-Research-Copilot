from __future__ import annotations
from typing import List
from pydantic import BaseModel, Field


class PlanStep(BaseModel):
    index: int = Field(..., description="1-based ordinal position of this step.")
    sub_query: str = Field(..., description="The concrete sub-query to research.")


class PlanModel(BaseModel):
    steps: List[PlanStep] = Field(..., description="Ordered list of sub-queries.")


class AgentState(BaseModel):
    query: str = Field(..., description="The original user research question.")
    steps: List[str] = Field(default_factory=list)
    plan: List[PlanStep] = Field(default_factory=list)
    retrieved_documents: List[dict] = Field(default_factory=list)
    current_summary: str = Field(default="")
    critic_feedback: str = Field(default="")
    retry_count: int = Field(default=0)
    next_node: str = Field(default="")

    class Config:
        extra = "allow"