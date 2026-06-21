"""FastAPI app — Loan Approval service.

Endpoints:
  POST /predict  -> probability_of_default, decision, risk_tier, rate
  POST /explain  -> top_factors, summary
See blueprint Bước 6.
"""

from __future__ import annotations

from fastapi import FastAPI

app = FastAPI(title="Credit Risk & Loan Approval API")


@app.get("/health")
def health():
    return {"status": "ok"}
