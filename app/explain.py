"""SHAP explanation logic for the /explain endpoint. See blueprint Bước 6."""

from __future__ import annotations


def top_factors(features, k: int = 5):
    """Return the top-k SHAP factors driving the decision."""
    raise NotImplementedError
