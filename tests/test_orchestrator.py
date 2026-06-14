"""
Layer 4 Tests: LangGraph Orchestrator

Tests the orchestration engine in isolation by mocking all four agent entry
points.  This lets us verify:
  - Profile and Risk agents are invoked concurrently (parallel_analysis_node)
  - State is correctly passed between nodes
  - Conditional routing works (error handling, happy path)
  - The graph short-circuits when either parallel agent fails
  - The complete pipeline populates all state fields
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from models.schemas import (
    ApplicantProfile,
    ComplianceRecord,
    FinancialRiskAssessment,
    LoanApplicationRequest,
    LoanDecision,
    LoanDecisionOutput,
    RiskLevel,
)


# ─── Shared test data ─────────────────────────────────────────────────────────

def _make_application(**overrides) -> LoanApplicationRequest:
    base = dict(
        applicant_name="Jane Smith",
        annual_income=80000,
        monthly_debt=1200,
        loan_amount=20000,
        loan_purpose="Home renovation",
        credit_score=720,
        employment_type="full_time",
        years_employed=4,
        existing_loans=1,
        assets_value=50000,
    )
    base.update(overrides)
    return LoanApplicationRequest(**base)


def _mock_profile() -> ApplicantProfile:
    return ApplicantProfile(
        income_stability_score=82.5,
        employment_risk=RiskLevel.LOW,
        credit_history_summary="Good — 720 FICO",
        completeness_flags=[],
    )


def _mock_risk() -> FinancialRiskAssessment:
    return FinancialRiskAssessment(
        debt_to_income_ratio=0.18,
        credit_risk_level=RiskLevel.LOW,
        loan_amount_risk=RiskLevel.LOW,
        anomalies=[],
        risk_reasoning="Low risk across all metrics.",
    )


def _mock_decision() -> LoanDecisionOutput:
    return LoanDecisionOutput(
        decision=LoanDecision.APPROVED,
        risk_score=22.0,
        confidence_level=91.0,
        key_factors=["Low DTI", "Prime credit"],
        explanation="Approved based on strong credit profile.",
    )


def _mock_compliance(case_id: str = "LOAN-20260604-TEST0001") -> ComplianceRecord:
    return ComplianceRecord(
        case_id=case_id,
        action_taken="Loan approved; applicant notified via email",
        notification_sent=True,
        notification_type="email",
        timestamp=datetime.now(timezone.utc),
        audit_summary="Full audit record created.",
    )


# ─── Parallel Analysis Node Tests ─────────────────────────────────────────────

class TestParallelAnalysisNode:
    """Tests for orchestration/orchestrator.py::parallel_analysis_node()"""

    @pytest.mark.asyncio
    async def test_returns_both_profile_and_risk(self):
        """Both agents run and both results land in the returned state dict."""
        from orchestration.orchestrator import parallel_analysis_node

        with patch("agents.applicant_profile_agent.analyze", new=AsyncMock(return_value=_mock_profile())):
            with patch("agents.financial_risk_agent.analyze", new=AsyncMock(return_value=_mock_risk())):
                result = await parallel_analysis_node({"application": _make_application().model_dump()})

        assert "profile" in result
        assert "risk_assessment" in result
        assert result["profile"]["income_stability_score"] == 82.5
        assert result["risk_assessment"]["debt_to_income_ratio"] == pytest.approx(0.18)
        assert "error" not in result

    @pytest.mark.asyncio
    async def test_both_agents_are_called(self):
        """asyncio.gather fires both agents; verify both are invoked."""
        from orchestration.orchestrator import parallel_analysis_node

        profile_spy = AsyncMock(return_value=_mock_profile())
        risk_spy = AsyncMock(return_value=_mock_risk())

        with patch("agents.applicant_profile_agent.analyze", new=profile_spy):
            with patch("agents.financial_risk_agent.analyze", new=risk_spy):
                await parallel_analysis_node({"application": _make_application().model_dump()})

        profile_spy.assert_called_once()
        risk_spy.assert_called_once()

    @pytest.mark.asyncio
    async def test_profile_failure_returns_error(self):
        """If Profile Agent raises, the node returns error state."""
        from orchestration.orchestrator import parallel_analysis_node

        with patch("agents.applicant_profile_agent.analyze", new=AsyncMock(side_effect=RuntimeError("MCP down"))):
            with patch("agents.financial_risk_agent.analyze", new=AsyncMock(return_value=_mock_risk())):
                result = await parallel_analysis_node({"application": _make_application().model_dump()})

        assert result.get("status") == "error"
        assert "Profile agent failed" in result.get("error", "")
        assert "profile" not in result

    @pytest.mark.asyncio
    async def test_risk_failure_returns_error(self):
        """If Risk Agent raises, the node returns error state."""
        from orchestration.orchestrator import parallel_analysis_node

        with patch("agents.applicant_profile_agent.analyze", new=AsyncMock(return_value=_mock_profile())):
            with patch("agents.financial_risk_agent.analyze", new=AsyncMock(side_effect=RuntimeError("timeout"))):
                result = await parallel_analysis_node({"application": _make_application().model_dump()})

        assert result.get("status") == "error"
        assert "Risk agent failed" in result.get("error", "")
        assert "risk_assessment" not in result

    @pytest.mark.asyncio
    async def test_risk_agent_still_called_when_profile_fails(self):
        """Because both run concurrently, Risk Agent is called even if Profile fails."""
        from orchestration.orchestrator import parallel_analysis_node

        risk_spy = AsyncMock(return_value=_mock_risk())

        with patch("agents.applicant_profile_agent.analyze", new=AsyncMock(side_effect=RuntimeError("MCP down"))):
            with patch("agents.financial_risk_agent.analyze", new=risk_spy):
                await parallel_analysis_node({"application": _make_application().model_dump()})

        risk_spy.assert_called_once()


# ─── Decision Node Tests ───────────────────────────────────────────────────────

class TestDecisionNode:
    """Tests for orchestration/orchestrator.py::decision_node()"""

    @pytest.mark.asyncio
    async def test_decision_node_returns_decision(self):
        from orchestration.orchestrator import decision_node

        mock_llm_response = MagicMock()
        mock_llm_response.content = [MagicMock(text="Approved — strong profile.")]

        state = {
            "application": _make_application().model_dump(),
            "profile": _mock_profile().model_dump(),
            "risk_assessment": _mock_risk().model_dump(),
        }

        with patch("agents.loan_decision_agent.decide", new=AsyncMock(return_value=_mock_decision())):
            with patch("anthropic.AsyncAnthropic") as mock_client_class:
                mock_client = AsyncMock()
                mock_client.messages.create = AsyncMock(return_value=mock_llm_response)
                mock_client_class.return_value = mock_client
                result = await decision_node(state)

        assert "decision" in result
        assert result["decision"]["decision"] == "approved"
        assert result["decision"]["risk_score"] == 22.0

    @pytest.mark.asyncio
    async def test_decision_node_sets_error_on_exception(self):
        from orchestration.orchestrator import decision_node

        state = {
            "application": _make_application().model_dump(),
            "profile": _mock_profile().model_dump(),
            "risk_assessment": _mock_risk().model_dump(),
        }

        with patch("agents.loan_decision_agent.decide", new=AsyncMock(side_effect=Exception("API failure"))):
            result = await decision_node(state)

        assert result.get("status") == "error"


# ─── Compliance Node Tests ─────────────────────────────────────────────────────

class TestComplianceNode:
    """Tests for orchestration/orchestrator.py::compliance_node()"""

    @pytest.mark.asyncio
    async def test_compliance_node_returns_record(self):
        from orchestration.orchestrator import compliance_node

        state = {
            "application": _make_application().model_dump(),
            "profile": _mock_profile().model_dump(),
            "risk_assessment": _mock_risk().model_dump(),
            "decision": _mock_decision().model_dump(),
        }

        with patch("agents.compliance_agent.record_and_notify", new=AsyncMock(return_value=_mock_compliance())):
            result = await compliance_node(state)

        assert "compliance" in result
        assert result["case_id"] == "LOAN-20260604-TEST0001"
        assert result["status"] == "completed"

    @pytest.mark.asyncio
    async def test_compliance_node_sets_error_on_exception(self):
        from orchestration.orchestrator import compliance_node

        state = {
            "application": _make_application().model_dump(),
            "profile": _mock_profile().model_dump(),
            "risk_assessment": _mock_risk().model_dump(),
            "decision": _mock_decision().model_dump(),
        }

        with patch("agents.compliance_agent.record_and_notify", new=AsyncMock(side_effect=Exception("notify failed"))):
            result = await compliance_node(state)

        assert result.get("status") == "error"


# ─── Routing Function Tests ────────────────────────────────────────────────────

class TestRoutingFunctions:
    """Tests for the conditional edge routing function."""

    def test_route_after_parallel_ok_routes_to_decision(self):
        from orchestration.orchestrator import route_after_parallel
        state = {
            "profile": {"income_stability_score": 80},
            "risk_assessment": {"dti": 0.25},
            "error": None,
        }
        assert route_after_parallel(state) == "decision_agent"

    def test_route_after_parallel_error_routes_to_end(self):
        from orchestration.orchestrator import route_after_parallel
        state = {"profile": None, "risk_assessment": None, "error": "Agent failed"}
        assert route_after_parallel(state) == "__end__"

    def test_route_after_parallel_missing_profile_routes_to_end(self):
        from orchestration.orchestrator import route_after_parallel
        state = {"profile": None, "risk_assessment": {"dti": 0.25}, "error": None}
        assert route_after_parallel(state) == "__end__"

    def test_route_after_parallel_missing_risk_routes_to_end(self):
        from orchestration.orchestrator import route_after_parallel
        state = {"profile": {"income_stability_score": 80}, "risk_assessment": None, "error": None}
        assert route_after_parallel(state) == "__end__"


# ─── Full Pipeline Integration Tests ──────────────────────────────────────────

class TestFullPipeline:
    """End-to-end orchestrator tests mocking all four agents."""

    @pytest.mark.asyncio
    async def test_successful_approval_pipeline(self):
        """
        Happy path: all agents succeed → state contains profile, risk,
        decision, compliance, and status='completed'.
        """
        from orchestration.orchestrator import process_application

        mock_llm_response = MagicMock()
        mock_llm_response.content = [MagicMock(text="Approved based on strong profile.")]

        with patch("agents.applicant_profile_agent.analyze", new=AsyncMock(return_value=_mock_profile())):
            with patch("agents.financial_risk_agent.analyze", new=AsyncMock(return_value=_mock_risk())):
                with patch("agents.loan_decision_agent.decide", new=AsyncMock(return_value=_mock_decision())):
                    with patch("agents.compliance_agent.record_and_notify", new=AsyncMock(return_value=_mock_compliance())):
                        with patch("anthropic.AsyncAnthropic") as mock_cls:
                            mock_client = AsyncMock()
                            mock_client.messages.create = AsyncMock(return_value=mock_llm_response)
                            mock_cls.return_value = mock_client
                            result = await process_application(_make_application())

        assert result["status"] == "completed"
        assert result["profile"] is not None
        assert result["risk_assessment"] is not None
        assert result["decision"]["decision"] == "approved"
        assert result["case_id"] == "LOAN-20260604-TEST0001"

    @pytest.mark.asyncio
    async def test_pipeline_short_circuits_on_profile_failure(self):
        """
        When the Profile Agent fails, the pipeline halts after parallel_analysis.
        Because agents run concurrently, the Risk Agent IS still called.
        Decision and Compliance nodes must NOT be reached.
        """
        from orchestration.orchestrator import process_application

        risk_spy = AsyncMock(return_value=_mock_risk())
        decision_spy = AsyncMock(return_value=_mock_decision())

        with patch("agents.applicant_profile_agent.analyze", new=AsyncMock(side_effect=RuntimeError("MCP down"))):
            with patch("agents.financial_risk_agent.analyze", new=risk_spy):
                with patch("agents.loan_decision_agent.decide", new=decision_spy):
                    result = await process_application(_make_application())

        # Risk runs concurrently — it IS called even though profile failed
        risk_spy.assert_called_once()
        # Decision and Compliance must not be reached
        decision_spy.assert_not_called()
        assert result.get("decision") is None
        assert result.get("compliance") is None

    @pytest.mark.asyncio
    async def test_pipeline_short_circuits_on_risk_failure(self):
        """
        When the Risk Agent fails, the pipeline halts after parallel_analysis.
        Profile Agent IS still called (concurrent). Decision and Compliance must not run.
        """
        from orchestration.orchestrator import process_application

        profile_spy = AsyncMock(return_value=_mock_profile())
        decision_spy = AsyncMock(return_value=_mock_decision())

        with patch("agents.applicant_profile_agent.analyze", new=profile_spy):
            with patch("agents.financial_risk_agent.analyze", new=AsyncMock(side_effect=RuntimeError("Risk MCP down"))):
                with patch("agents.loan_decision_agent.decide", new=decision_spy):
                    result = await process_application(_make_application())

        profile_spy.assert_called_once()
        decision_spy.assert_not_called()
        assert result.get("decision") is None
        assert result.get("compliance") is None

    @pytest.mark.asyncio
    async def test_pipeline_populates_all_state_fields_on_success(self):
        """All five key state fields must be populated after a successful run."""
        from orchestration.orchestrator import process_application

        mock_llm = MagicMock()
        mock_llm.content = [MagicMock(text="Approved.")]

        with patch("agents.applicant_profile_agent.analyze", new=AsyncMock(return_value=_mock_profile())):
            with patch("agents.financial_risk_agent.analyze", new=AsyncMock(return_value=_mock_risk())):
                with patch("agents.loan_decision_agent.decide", new=AsyncMock(return_value=_mock_decision())):
                    with patch("agents.compliance_agent.record_and_notify", new=AsyncMock(return_value=_mock_compliance())):
                        with patch("anthropic.AsyncAnthropic") as mock_cls:
                            mock_cls.return_value.messages.create = AsyncMock(return_value=mock_llm)
                            result = await process_application(_make_application())

        assert result.get("profile") is not None
        assert result.get("risk_assessment") is not None
        assert result.get("decision") is not None
        assert result.get("compliance") is not None
        assert result.get("case_id") is not None

    @pytest.mark.asyncio
    async def test_rejected_application_completes_full_pipeline(self):
        """A rejected application should still flow through all nodes."""
        from orchestration.orchestrator import process_application

        rejected = LoanDecisionOutput(
            decision=LoanDecision.REJECTED,
            risk_score=78.0,
            confidence_level=85.0,
            key_factors=["High DTI", "Poor credit"],
            explanation="Declined.",
        )

        mock_llm = MagicMock()
        mock_llm.content = [MagicMock(text="Rejected.")]

        with patch("agents.applicant_profile_agent.analyze", new=AsyncMock(return_value=_mock_profile())):
            with patch("agents.financial_risk_agent.analyze", new=AsyncMock(return_value=_mock_risk())):
                with patch("agents.loan_decision_agent.decide", new=AsyncMock(return_value=rejected)):
                    with patch("agents.compliance_agent.record_and_notify", new=AsyncMock(return_value=_mock_compliance())):
                        with patch("anthropic.AsyncAnthropic") as mock_cls:
                            mock_cls.return_value.messages.create = AsyncMock(return_value=mock_llm)
                            result = await process_application(_make_application())

        assert result["decision"]["decision"] == "rejected"
        assert result["status"] == "completed"
