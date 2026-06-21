"""Pydantic v2 request/response models. See blueprint Bước 6."""

from __future__ import annotations

from pydantic import BaseModel


class PredictRequest(BaseModel):
    """Application features input."""


class PredictResponse(BaseModel):
    probability_of_default: float
    decision: str
    risk_tier: str
    recommended_interest_rate: str


class Factor(BaseModel):
    feature: str
    impact: float
    direction: str


class ExplainResponse(BaseModel):
    top_factors: list[Factor]
    summary: str
