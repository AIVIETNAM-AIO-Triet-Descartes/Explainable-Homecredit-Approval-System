"""Inference + business rule layer.

PD < 0.15 -> approve; 0.15-0.40 -> manual_review; > 0.40 -> reject.
Maps PD to risk tier + risk-based pricing. See blueprint Bước 6.
"""

from __future__ import annotations


def decide(pd: float) -> str:
    """Map probability of default to an approval decision."""
    if pd < 0.15:
        return "approve"
    if pd <= 0.40:
        return "manual_review"
    return "reject"
