"""Tests: API endpoints (status codes, response schema) + business logic. Blueprint Bước 8."""

from app.predict import decide


def test_decide_thresholds():
    assert decide(0.10) == "approve"
    assert decide(0.25) == "manual_review"
    assert decide(0.50) == "reject"
