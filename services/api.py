"""
FastAPI Microservice — Loan Approval REST API

Endpoints:
  POST /loan/apply            — submit and immediately process a loan application
  GET  /loan/status/{case_id} — look up a previously processed case

Applications are processed synchronously inside the POST handler: the caller
receives the full decision (risk score, confidence, explanation) in the
response body.  The GET endpoint serves as a lookup backed by an in-memory
registry; swap for Redis/PostgreSQL in production.
"""

from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import uvicorn
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from config.settings import settings
from models.schemas import (
    LoanApplicationRequest,
    LoanDecision,
    LoanStatusResponse,
)

logger = logging.getLogger(__name__)

# In-memory case registry: case_id → final LangGraph state dict
# Replace with a persistent store (Redis, Postgres) in production.
_case_registry: Dict[str, Dict[str, Any]] = {}


# ─── Lifecycle ────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loan Approval API starting")
    yield
    logger.info("Loan Approval API shutting down")


app = FastAPI(
    title="Loan Approval Agent API",
    description=(
        "Multi-agent loan approval system powered by Claude and LangGraph. "
        "Submit an application and receive an AI-driven decision with full audit trail."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Allow Streamlit UI (any origin in dev; restrict in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _state_to_response(case_id: str, state: Dict[str, Any]) -> LoanStatusResponse:
    """
    Converts a LangGraph final state dict into a LoanStatusResponse.

    Handles partial state gracefully — if the pipeline short-circuited on
    an error, decision/compliance fields may be absent.
    """
    decision_data: Dict[str, Any] = state.get("decision") or {}
    compliance_data: Dict[str, Any] = state.get("compliance") or {}

    # Timestamp from compliance record, or current time as fallback
    raw_ts = compliance_data.get("timestamp")
    try:
        ts = datetime.fromisoformat(raw_ts) if raw_ts else datetime.now(timezone.utc)
    except (ValueError, TypeError):
        ts = datetime.now(timezone.utc)

    # Map decision string to enum; None if pipeline failed before decision node
    decision_str = decision_data.get("decision")
    try:
        decision_enum = LoanDecision(decision_str) if decision_str else None
    except ValueError:
        decision_enum = None

    return LoanStatusResponse(
        case_id=case_id,
        status=state.get("status", "error"),
        decision=decision_enum,
        risk_score=decision_data.get("risk_score"),
        confidence_level=decision_data.get("confidence_level"),
        explanation=decision_data.get("explanation"),
        key_factors=decision_data.get("key_factors"),
        timestamp=ts,
        error_detail=state.get("error"),
    )


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.post(
    "/loan/apply",
    response_model=LoanStatusResponse,
    status_code=status.HTTP_200_OK,
    summary="Submit a loan application",
    description=(
        "Runs the full 4-agent pipeline synchronously. Returns the complete "
        "decision including risk score, confidence, explanation, and case ID."
    ),
)
async def apply_for_loan(application: LoanApplicationRequest) -> LoanStatusResponse:
    """
    POST /loan/apply

    Validates via Pydantic, runs the LangGraph orchestrator, caches the result,
    and returns the full decision to the caller.

    Raises:
        422: request body fails validation
        500: orchestration pipeline raises an unexpected error
    """
    logger.info("New application: %s | $%.0f", application.applicant_name, application.loan_amount)

    # Late import avoids circular imports at module load time
    from orchestration.orchestrator import process_application

    try:
        final_state = await process_application(application)
    except Exception as exc:
        logger.error("Orchestration error: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pipeline error: {exc}",
        )

    case_id = final_state.get("case_id") or f"ERR-{int(datetime.now().timestamp())}"
    _case_registry[case_id] = final_state

    response = _state_to_response(case_id, final_state)
    logger.info("Processed %s → %s (case %s)", application.applicant_name, response.decision, case_id)
    return response


@app.get(
    "/loan/status/{case_id}",
    response_model=LoanStatusResponse,
    summary="Get decision by case ID",
    description="Look up the stored decision for a previously submitted application.",
)
async def get_loan_status(case_id: str) -> LoanStatusResponse:
    """
    GET /loan/status/{case_id}

    Raises:
        404: no application with this case_id has been processed
    """
    if case_id not in _case_registry:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Case '{case_id}' not found. Submit the application first.",
        )
    return _state_to_response(case_id, _case_registry[case_id])


@app.get("/health", summary="Health check")
async def health() -> Dict[str, str]:
    """Used by load balancers and monitoring; no auth required."""
    return {"status": "healthy", "service": "loan-approval-api"}


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=settings.log_level)
    uvicorn.run(
        "services.api:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
    )
