"""Model evaluation.

Imbalanced-aware metrics: ROC-AUC, PR-AUC, F1. Threshold tuning for
business-cost-sensitive decisions. See blueprint Bước 3.
"""

from __future__ import annotations


def evaluate(model, X, y):
    """Return dict of evaluation metrics."""
    raise NotImplementedError
