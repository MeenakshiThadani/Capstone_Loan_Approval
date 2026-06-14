"""
Layer 5 Tests: FastAPI Service

Tests all REST endpoints using FastAPI's TestClient (synchronous HTTPX-based
test transport — no real server needed).

Coverage:
  POST /loan/apply
    - valid application → 200 with full decision
    - invalid body (missing fields, bad values) → 422
    - orchestration error → 500
    - various decision outcomes (approved, rejected, manual_review)

  GET /loan/status/{case_id}
    - known case_id → 200 with stored decision
    - unknown case_id → 404

  GET /health
    - always 200
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from models.schemas import (
    ApplicantProfile,
    ComplianceRecord,
    FinancialRiskAssessment,
    LoanDecision,
    LoanDecisionOutput,
    RiskLevel,
)

# ─── App import (must happen before client creation) ──────────────────────────
from services.api import app, _case_registry

client = TestClient(app)

# ─── Helpers ──────────────────────────────────────────────────────────────────

VALID_PAYLOAD = dict(
    applicant_name="Jane Smith",
    annual_income=80000,
    monthly_debt=1200,
    loan_amount=20000,
    loan_purpose="Home renovation",
    credit_score=720,
    employment_type="full_time",
    years_employed=4.0,
    existing_loans=1,
    assets_value=50000,
)


def _approved_state(case_id: str = "LOAN-20260604-TESTAPPR") -> Dict[str, Any]:
    """Build a complete LangGraph state dict for an approved application."""
    return {
        "application": VALID_PAYLOAD,
        "profile": {
            "income_stability_score": 82.5,
            "employment_risk": "low",
            "credit_history_summary": "Good",
            "completeness_flags": [],
        },
        "risk_assessment": {
            "debt_to_income_ratio": 0.18,
            "credit_risk_level": "low",
            "loan_amount_risk": "low",
            "anomalies": [],
            "risk_reasoning": "Low risk.",
        },
        "decision": {
            "decision": "approved",
            "risk_score": 22.0,
            "confidence_level": 91.0,
            "key_factors": ["Low DTI", "Prime credit"],
            "explanation": "Approved based on strong credit profile.",
        },
        "compliance": {
            "case_id": case_id,
            "action_taken": "Loan approved",
            "notification_sent": True,
            "notification_type": "email",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "audit_summary": "Full audit record.",
        },
        "case_id": case_id,
        "error": None,
        "status": "completed",
    }


def _rejected_state(case_id: str = "LOAN-20260604-TESTREJ") -> Dict[str, Any]:
    state = _approved_state(case_id)
    state["decision"]["decision"] = "rejected"
    state["decision"]["risk_score"] = 78.0
    state["compliance"]["case_id"] = case_id
    state["case_id"] = case_id
    return state


def _review_state(case_id: str = "LOAN-20260604-TESTREV") -> Dict[str, Any]:
    state = _approved_state(case_id)
    state["decision"]["decision"] = "manual_review"
    state["decision"]["risk_score"] = 55.0
    state["compliance"]["case_id"] = case_id
    state["case_id"] = case_id
    return state


# ─── POST /loan/apply ─────────────────────────────────────────────────────────

class TestApplyEndpoint:

    def test_valid_application_returns_200(self):
        """A complete, valid application should receive a 200 response."""
        with patch("orchestration.orchestrator.process_application",
                   new=AsyncMock(return_value=_approved_state())):
            response = client.post("/loan/apply", json=VALID_PAYLOAD)

        assert response.status_code == 200

    def test_response_contains_case_id(self):
        with patch("orchestration.orchestrator.process_application",
                   new=AsyncMock(return_value=_approved_state())):
            response = client.post("/loan/apply", json=VALID_PAYLOAD)

        data = response.json()
        assert "case_id" in data
        assert data["case_id"].startswith("LOAN-")

    def test_approved_decision_in_response(self):
        with patch("orchestration.orchestrator.process_application",
                   new=AsyncMock(return_value=_approved_state())):
            response = client.post("/loan/apply", json=VALID_PAYLOAD)

        data = response.json()
        assert data["decision"] == "approved"

    def test_rejected_decision_in_response(self):
        with patch("orchestration.orchestrator.process_application",
                   new=AsyncMock(return_value=_rejected_state())):
            response = client.post("/loan/apply", json=VALID_PAYLOAD)

        data = response.json()
        assert data["decision"] == "rejected"
        assert data["risk_score"] == 78.0

    def test_manual_review_decision_in_response(self):
        with patch("orchestration.orchestrator.process_application",
                   new=AsyncMock(return_value=_review_state())):
            response = client.post("/loan/apply", json=VALID_PAYLOAD)

        data = response.json()
        assert data["decision"] == "manual_review"

    def test_response_contains_risk_score_and_confidence(self):
        with patch("orchestration.orchestrator.process_application",
                   new=AsyncMock(return_value=_approved_state())):
            response = client.post("/loan/apply", json=VALID_PAYLOAD)

        data = response.json()
        assert "risk_score" in data
        assert "confidence_level" in data
        assert 0 <= data["risk_score"] <= 100
        assert 0 <= data["confidence_level"] <= 100

    def test_response_contains_explanation(self):
        with patch("orchestration.orchestrator.process_application",
                   new=AsyncMock(return_value=_approved_state())):
            response = client.post("/loan/apply", json=VALID_PAYLOAD)

        data = response.json()
        assert "explanation" in data
        assert len(data["explanation"]) > 0

    def test_response_contains_key_factors(self):
        with patch("orchestration.orchestrator.process_application",
                   new=AsyncMock(return_value=_approved_state())):
            response = client.post("/loan/apply", json=VALID_PAYLOAD)

        data = response.json()
        assert "key_factors" in data
        assert isinstance(data["key_factors"], list)

    def test_missing_required_field_returns_422(self):
        """Omitting applicant_name should trigger Pydantic validation → 422."""
        payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "applicant_name"}
        response = client.post("/loan/apply", json=payload)
        assert response.status_code == 422

    def test_invalid_credit_score_returns_422(self):
        """Credit score of 900 is outside FICO range → 422."""
        payload = {**VALID_PAYLOAD, "credit_score": 900}
        response = client.post("/loan/apply", json=payload)
        assert response.status_code == 422

    def test_negative_income_returns_422(self):
        payload = {**VALID_PAYLOAD, "annual_income": -5000}
        response = client.post("/loan/apply", json=payload)
        assert response.status_code == 422

    def test_zero_loan_amount_returns_422(self):
        payload = {**VALID_PAYLOAD, "loan_amount": 0}
        response = client.post("/loan/apply", json=payload)
        assert response.status_code == 422

    def test_orchestration_failure_returns_500(self):
        """An unhandled exception in the pipeline should return 500."""
        with patch("orchestration.orchestrator.process_application",
                   new=AsyncMock(side_effect=RuntimeError("MCP servers down"))):
            response = client.post("/loan/apply", json=VALID_PAYLOAD)

        assert response.status_code == 500
        assert "Pipeline error" in response.json()["detail"]

    def test_processed_case_is_stored_in_registry(self):
        """After a successful POST, the case must be retrievable via GET."""
        state = _approved_state("LOAN-20260604-STORED01")
        with patch("orchestration.orchestrator.process_application",
                   new=AsyncMock(return_value=state)):
            post_resp = client.post("/loan/apply", json=VALID_PAYLOAD)

        assert post_resp.status_code == 200
        case_id = post_resp.json()["case_id"]

        get_resp = client.get(f"/loan/status/{case_id}")
        assert get_resp.status_code == 200
        assert get_resp.json()["case_id"] == case_id

    def test_empty_body_returns_422(self):
        response = client.post("/loan/apply", json={})
        assert response.status_code == 422

    def test_loan_amount_exceeds_10m_returns_422(self):
        payload = {**VALID_PAYLOAD, "loan_amount": 10_000_001}
        response = client.post("/loan/apply", json=payload)
        assert response.status_code == 422


# ─── GET /loan/status/{case_id} ───────────────────────────────────────────────

class TestStatusEndpoint:

    def setup_method(self):
        """Inject a known case directly into the registry before each test."""
        _case_registry["LOAN-KNOWN-001"] = _approved_state("LOAN-KNOWN-001")

    def test_known_case_id_returns_200(self):
        response = client.get("/loan/status/LOAN-KNOWN-001")
        assert response.status_code == 200

    def test_known_case_returns_correct_data(self):
        response = client.get("/loan/status/LOAN-KNOWN-001")
        data = response.json()
        assert data["case_id"] == "LOAN-KNOWN-001"
        assert data["decision"] == "approved"

    def test_unknown_case_id_returns_404(self):
        response = client.get("/loan/status/LOAN-DOES-NOT-EXIST")
        assert response.status_code == 404

    def test_404_detail_message_is_helpful(self):
        response = client.get("/loan/status/LOAN-MISSING")
        assert "not found" in response.json()["detail"].lower()

    def test_status_response_has_required_fields(self):
        response = client.get("/loan/status/LOAN-KNOWN-001")
        data = response.json()
        for field in ["case_id", "status", "decision", "risk_score", "confidence_level"]:
            assert field in data, f"Missing field: {field}"


# ─── GET /health ──────────────────────────────────────────────────────────────

class TestHealthEndpoint:

    def test_health_returns_200(self):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_returns_healthy_status(self):
        response = client.get("/health")
        data = response.json()
        assert data["status"] == "healthy"

    def test_health_identifies_service(self):
        response = client.get("/health")
        assert "loan-approval-api" in response.json().get("service", "")
