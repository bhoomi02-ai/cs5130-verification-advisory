"""Structured outputs exchanged between agents and scored by the checker."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Trade(BaseModel):
    ticker: str
    action: Literal["buy", "sell", "hold"]
    amount_usd: float = Field(ge=0)
    rationale: str = ""


class Claim(BaseModel):
    """A factual or numeric assertion the advisor makes, e.g. a projected balance."""

    statement: str
    value: float | None = None
    unit: str | None = None


class Recommendation(BaseModel):
    task_id: str
    trades: list[Trade] = Field(default_factory=list)
    claims: list[Claim] = Field(default_factory=list)
    summary: str = ""
