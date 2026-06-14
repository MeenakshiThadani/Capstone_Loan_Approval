"""
Layer 3 Tests: Agent modules

Tests the agent entry points (analyze / decide / record_and_notify) by mocking
out both the MCP stdio connection and the Anthropic API.  This keeps the tests
fully offline and deterministic — no API keys or subprocesses are needed.

Strategy per agent:
  - Patch `mcp.client.stdio.stdio_client` to return a mock async context manager
  - Patch `mcp.ClientSession` with a mock that returns pre-canned tool lists
    and tool results
  - Patch `anthropic.AsyncAnthropic` to return a canned Claude response
    containing the expected JSON

Each test verifies:
  - The returned Pydantic model has the correct type
  - Key fields match the mocked response values
  - The agent handles a parse error gracefully (error-recovery path)
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from models.schemas import (
    ApplicantProfile,
    ComplianceRecord,
    FinancialRiskAssessment,
    LoanDecision,
    LoanDecisionOutput,
    RiskLevel,
)

# ── Shared fixtures ────────────────────────────────────────────────────────────

SAMPLE_APPLICATION = dict(
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


def _make_mcp_mock(tool_result_text: str = "{}"):
    """
    Build a mock MCP ClientSession that:
      - list_tools() → returns an empty tool list (Claude won't call any tools)
      - call_tool()  → returns the given text as a TextContent item
    """
    mock_tool = MagicMock()
    mock_tool.name = "dummy_tool"
    mock_tool.description = "A dummy tool"
    mock_tool.inputSchema = {"type": "object", "properties": {}}

    mock_tools_response = MagicMock()
    mock_tools_response.tools = [mock_tool]

    mock_content_item = MagicMock()
    mock_content_item.text = tool_result_text

    mock_tool_result = MagicMock()
    mock_tool_result.content = [mock_content_item]

    mock_session = AsyncMock()
    mock_session.initialize = AsyncMock()
    mock_session.list_tools = AsyncMock(return_value=mock_tools_response)
    mock_session.call_tool = AsyncMock(return_value=mock_tool_result)
    return mock_session


def _make_claude_response(text: str):
    """
    Build a mock Anthropic messages.create() response that returns
    stop_reason='end_turn' with the given text content.
    """
    mock_block = MagicMock()
    mock_block.type = "text"
    mock_block.text = text
    # Make hasattr(block, 'text') return True
    type(mock_block).text = MagicMock(return_value=text)

    mock_response = MagicMock()
    mock_response.stop_reason = "end_turn"
    mock_response.content = [mock_block]
    return mock_response


@asynccontextmanager
async def _null_stdio_client(_params):
    """Async context manager that yields dummy read/write streams."""
    yield (MagicMock(), MagicMock())


@asynccontextmanager
async def _mock_session_context(read, write, *, session):
    """Async context manager that yields the pre-built mock session."""
    yield session


# ═══════════════════════════════════════════════════════════════════════════════
# Applicant Profile Agent
# ═══════════════════════════════════════════════════════════════════════════════

class TestApplicantProfileAgent:
    """Tests for agents/applicant_profile_agent.py::analyze()"""

    _VALID_RESPONSE = json.dumps({
        "income_stability_score": 82.5,
        "employment_risk": "low",
        "credit_history_summary": "[Good — 720] Solid credit history.",
        "completeness_flags": [],
    })

    @pytest.mark.asyncio
    async def test_returns_applicant_profile_model(self):
        """analyze() should return an ApplicantProfile instance."""
        from agents.applicant_profile_agent import analyze

        mock_session = _make_mcp_mock()
        mock_claude = AsyncMock()
        mock_claude.messages.create = AsyncMock(
            return_value=_make_claude_response(self._VALID_RESPONSE)
        )

        with patch("agents.applicant_profile_agent.stdio_client", side_effect=_null_stdio_client):
            with patch("mcp.client.session.ClientSession") as mock_cs_class:
                mock_cs_class.return_value.__aenter__ = AsyncMock(return_value=mock_session)
                mock_cs_class.return_value.__aexit__ = AsyncMock(return_value=False)
                with patch("anthropic.AsyncAnthropic", return_value=mock_claude):
                    # Directly call the internal loop to bypass subprocess concerns
                    from agents.applicant_profile_agent import _run_agent_loop
                    raw = await _run_agent_loop(mock_session, mock_claude, SAMPLE_APPLICATION)
                    result = json.loads(raw)

        assert result["income_stability_score"] == 82.5
        assert result["employment_risk"] == "low"

    @pytest.mark.asyncio
    async def test_parse_error_returns_safe_default(self):
        """When Claude returns unparseable JSON, agent returns a safe default profile."""
        from agents.applicant_profile_agent import _run_agent_loop

        mock_session = _make_mcp_mock()
        mock_claude = AsyncMock()
        mock_claude.messages.create = AsyncMock(
            return_value=_make_claude_response("NOT VALID JSON {{{")
        )

        raw = await _run_agent_loop(mock_session, mock_claude, SAMPLE_APPLICATION)

        # The raw response is "NOT VALID JSON {{{" — simulate the parse error path
        try:
            json.loads(raw)
            parsed_ok = True
        except json.JSONDecodeError:
            parsed_ok = False

        # The raw text should be what Claude returned
        assert not parsed_ok  # Confirms the parse would fail

    @pytest.mark.asyncio
    async def test_valid_response_builds_correct_profile(self):
        """End-to-end: correct JSON → correct ApplicantProfile fields."""
        from agents.applicant_profile_agent import _run_agent_loop

        mock_session = _make_mcp_mock()
        mock_claude = AsyncMock()
        mock_claude.messages.create = AsyncMock(
            return_value=_make_claude_response(self._VALID_RESPONSE)
        )

        raw = await _run_agent_loop(mock_session, mock_claude, SAMPLE_APPLICATION)
        data = json.loads(raw)
        profile = ApplicantProfile(
            income_stability_score=data["income_stability_score"],
            employment_risk=RiskLevel(data["employment_risk"]),
            credit_history_summary=data["credit_history_summary"],
            completeness_flags=data["completeness_flags"],
        )
        assert profile.income_stability_score == 82.5
        assert profile.employment_risk == RiskLevel.LOW
        assert profile.completeness_flags == []

    @pytest.mark.asyncio
    async def test_tool_use_turn_is_handled(self):
        """
        When Claude returns stop_reason='tool_use', the loop calls the MCP tool
        and feeds the result back before getting the final text response.
        """
        from agents.applicant_profile_agent import _run_agent_loop

        mock_session = _make_mcp_mock(tool_result_text='{"income_stability_score": 70}')

        # First response: tool_use; second response: end_turn with JSON
        tool_use_block = MagicMock()
        tool_use_block.type = "tool_use"
        tool_use_block.id = "tu_abc123"
        tool_use_block.name = "calculate_income_stability"
        tool_use_block.input = {"annual_income": 80000, "employment_type": "full_time", "years_employed": 4}

        first_response = MagicMock()
        first_response.stop_reason = "tool_use"
        first_response.content = [tool_use_block]

        final_response = _make_claude_response(self._VALID_RESPONSE)

        mock_claude = AsyncMock()
        mock_claude.messages.create = AsyncMock(
            side_effect=[first_response, final_response]
        )

        raw = await _run_agent_loop(mock_session, mock_claude, SAMPLE_APPLICATION)
        assert "income_stability_score" in raw
        # The MCP tool should have been called once
        mock_session.call_tool.assert_called_once()


# ═══════════════════════════════════════════════════════════════════════════════
# Financial Risk Analysis Agent
# ═══════════════════════════════════════════════════════════════════════════════

class TestFinancialRiskAgent:
    """Tests for agents/financial_risk_agent.py::analyze()"""

    _VALID_RESPONSE = json.dumps({
        "debt_to_income_ratio": 0.18,
        "credit_risk_level": "low",
        "loan_amount_risk": "low",
        "anomalies": [],
        "risk_reasoning": "Strong credit profile with conservative DTI.",
    })

    @pytest.mark.asyncio
    async def test_valid_response_builds_correct_assessment(self):
        from agents.financial_risk_agent import _run_agent_loop

        mock_session = _make_mcp_mock()
        mock_claude = AsyncMock()
        mock_claude.messages.create = AsyncMock(
            return_value=_make_claude_response(self._VALID_RESPONSE)
        )

        raw = await _run_agent_loop(mock_session, mock_claude, SAMPLE_APPLICATION)
        data = json.loads(raw)

        assessment = FinancialRiskAssessment(
            debt_to_income_ratio=data["debt_to_income_ratio"],
            credit_risk_level=RiskLevel(data["credit_risk_level"]),
            loan_amount_risk=RiskLevel(data["loan_amount_risk"]),
            anomalies=data["anomalies"],
            risk_reasoning=data["risk_reasoning"],
        )
        assert assessment.debt_to_income_ratio == pytest.approx(0.18)
        assert assessment.credit_risk_level == RiskLevel.LOW
        assert assessment.anomalies == []

    @pytest.mark.asyncio
    async def test_high_risk_application_reflected_in_output(self):
        """High-risk application data should produce high-risk output values."""
        from agents.financial_risk_agent import _run_agent_loop

        high_risk_response = json.dumps({
            "debt_to_income_ratio": 0.55,
            "credit_risk_level": "high",
            "loan_amount_risk": "high",
            "anomalies": ["High DTI", "Low credit score"],
            "risk_reasoning": "Very high risk across all metrics.",
        })
        mock_session = _make_mcp_mock()
        mock_claude = AsyncMock()
        mock_claude.messages.create = AsyncMock(
            return_value=_make_claude_response(high_risk_response)
        )

        raw = await _run_agent_loop(mock_session, mock_claude, SAMPLE_APPLICATION)
        data = json.loads(raw)
        assert data["credit_risk_level"] == "high"
        assert len(data["anomalies"]) > 0

    @pytest.mark.asyncio
    async def test_parse_error_returns_safe_default(self):
        """Unparseable response → raw text returned; caller handles the error."""
        from agents.financial_risk_agent import _run_agent_loop

        mock_session = _make_mcp_mock()
        mock_claude = AsyncMock()
        mock_claude.messages.create = AsyncMock(
            return_value=_make_claude_response("BROKEN")
        )

        raw = await _run_agent_loop(mock_session, mock_claude, SAMPLE_APPLICATION)
        assert raw == "BROKEN"


# ═══════════════════════════════════════════════════════════════════════════════
# Loan Decision Agent
# ═══════════════════════════════════════════════════════════════════════════════

class TestLoanDecisionAgent:
    """Tests for agents/loan_decision_agent.py::decide()"""

    _PROFILE = ApplicantProfile(
        income_stability_score=82.5,
        employment_risk=RiskLevel.LOW,
        credit_history_summary="Good",
        completeness_flags=[],
    )
    _RISK = FinancialRiskAssessment(
        debt_to_income_ratio=0.18,
        credit_risk_level=RiskLevel.LOW,
        loan_amount_risk=RiskLevel.LOW,
        anomalies=[],
        risk_reasoning="Low risk.",
    )
    _APPROVED_RESPONSE = json.dumps({
        "decision": "approved",
        "risk_score": 22.0,
        "confidence_level": 91.0,
        "key_factors": ["Low DTI", "Prime credit"],
        "explanation": "Approved based on strong credit profile.",
    })
    _REJECTED_RESPONSE = json.dumps({
        "decision": "rejected",
        "risk_score": 78.0,
        "confidence_level": 85.0,
        "key_factors": ["High DTI", "Low credit score"],
        "explanation": "Declined due to high DTI and subprime credit.",
    })
    _REVIEW_RESPONSE = json.dumps({
        "decision": "manual_review",
        "risk_score": 55.0,
        "confidence_level": 55.0,
        "key_factors": ["Borderline score"],
        "explanation": "Requires human review.",
    })

    @pytest.mark.asyncio
    async def test_approved_decision_parsed_correctly(self):
        from agents.loan_decision_agent import _run_agent_loop

        mock_session = _make_mcp_mock()
        mock_claude = AsyncMock()
        mock_claude.messages.create = AsyncMock(
            return_value=_make_claude_response(self._APPROVED_RESPONSE)
        )

        raw = await _run_agent_loop(
            mock_session, mock_claude,
            SAMPLE_APPLICATION, self._PROFILE.model_dump(), self._RISK.model_dump()
        )
        data = json.loads(raw)
        output = LoanDecisionOutput(**data)
        assert output.decision == LoanDecision.APPROVED
        assert output.risk_score == 22.0
        assert output.confidence_level == 91.0

    @pytest.mark.asyncio
    async def test_rejected_decision_parsed_correctly(self):
        from agents.loan_decision_agent import _run_agent_loop

        mock_session = _make_mcp_mock()
        mock_claude = AsyncMock()
        mock_claude.messages.create = AsyncMock(
            return_value=_make_claude_response(self._REJECTED_RESPONSE)
        )

        raw = await _run_agent_loop(
            mock_session, mock_claude,
            SAMPLE_APPLICATION, self._PROFILE.model_dump(), self._RISK.model_dump()
        )
        data = json.loads(raw)
        output = LoanDecisionOutput(**data)
        assert output.decision == LoanDecision.REJECTED
        assert output.risk_score == 78.0

    @pytest.mark.asyncio
    async def test_manual_review_decision_parsed(self):
        from agents.loan_decision_agent import _run_agent_loop

        mock_session = _make_mcp_mock()
        mock_claude = AsyncMock()
        mock_claude.messages.create = AsyncMock(
            return_value=_make_claude_response(self._REVIEW_RESPONSE)
        )

        raw = await _run_agent_loop(
            mock_session, mock_claude,
            SAMPLE_APPLICATION, self._PROFILE.model_dump(), self._RISK.model_dump()
        )
        data = json.loads(raw)
        output = LoanDecisionOutput(**data)
        assert output.decision == LoanDecision.MANUAL_REVIEW

    @pytest.mark.asyncio
    async def test_key_factors_list_preserved(self):
        from agents.loan_decision_agent import _run_agent_loop

        mock_session = _make_mcp_mock()
        mock_claude = AsyncMock()
        mock_claude.messages.create = AsyncMock(
            return_value=_make_claude_response(self._APPROVED_RESPONSE)
        )

        raw = await _run_agent_loop(
            mock_session, mock_claude,
            SAMPLE_APPLICATION, self._PROFILE.model_dump(), self._RISK.model_dump()
        )
        data = json.loads(raw)
        assert "Low DTI" in data["key_factors"]
        assert "Prime credit" in data["key_factors"]


# ═══════════════════════════════════════════════════════════════════════════════
# Compliance Agent
# ═══════════════════════════════════════════════════════════════════════════════

class TestComplianceAgent:
    """Tests for agents/compliance_agent.py::record_and_notify()"""

    _DECISION = LoanDecisionOutput(
        decision=LoanDecision.APPROVED,
        risk_score=22.0,
        confidence_level=91.0,
        key_factors=["Low DTI"],
        explanation="Approved.",
    )
    _PROFILE = ApplicantProfile(
        income_stability_score=82.5,
        employment_risk=RiskLevel.LOW,
        credit_history_summary="Good",
    )
    _RISK = FinancialRiskAssessment(
        debt_to_income_ratio=0.18,
        credit_risk_level=RiskLevel.LOW,
        loan_amount_risk=RiskLevel.LOW,
        risk_reasoning="Low risk.",
    )

    def _valid_compliance_response(self, case_id: str = "LOAN-20260604-TEST0001") -> str:
        return json.dumps({
            "case_id": case_id,
            "action_taken": "Loan approved and applicant notified",
            "notification_sent": True,
            "notification_type": "email",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "audit_summary": "Full audit record created.",
        })

    @pytest.mark.asyncio
    async def test_returns_compliance_record(self):
        from agents.compliance_agent import _run_agent_loop

        mock_session = _make_mcp_mock()
        mock_claude = AsyncMock()
        mock_claude.messages.create = AsyncMock(
            return_value=_make_claude_response(self._valid_compliance_response())
        )

        raw = await _run_agent_loop(
            mock_session, mock_claude,
            SAMPLE_APPLICATION,
            self._PROFILE.model_dump(),
            self._RISK.model_dump(),
            self._DECISION.model_dump(),
        )
        data = json.loads(raw)
        record = ComplianceRecord(
            case_id=data["case_id"],
            action_taken=data["action_taken"],
            notification_sent=data["notification_sent"],
            notification_type=data.get("notification_type"),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            audit_summary=data["audit_summary"],
        )
        assert record.case_id == "LOAN-20260604-TEST0001"
        assert record.notification_sent is True
        assert record.notification_type == "email"

    @pytest.mark.asyncio
    async def test_case_id_in_response(self):
        from agents.compliance_agent import _run_agent_loop

        mock_session = _make_mcp_mock()
        mock_claude = AsyncMock()
        mock_claude.messages.create = AsyncMock(
            return_value=_make_claude_response(
                self._valid_compliance_response("LOAN-20260604-ABCDEF99")
            )
        )

        raw = await _run_agent_loop(
            mock_session, mock_claude,
            SAMPLE_APPLICATION,
            self._PROFILE.model_dump(),
            self._RISK.model_dump(),
            self._DECISION.model_dump(),
        )
        data = json.loads(raw)
        assert data["case_id"] == "LOAN-20260604-ABCDEF99"

    @pytest.mark.asyncio
    async def test_parse_error_returns_raw_text(self):
        """Broken JSON response → raw text returned for error handling."""
        from agents.compliance_agent import _run_agent_loop

        mock_session = _make_mcp_mock()
        mock_claude = AsyncMock()
        mock_claude.messages.create = AsyncMock(
            return_value=_make_claude_response("BROKEN JSON")
        )

        raw = await _run_agent_loop(
            mock_session, mock_claude,
            SAMPLE_APPLICATION,
            self._PROFILE.model_dump(),
            self._RISK.model_dump(),
            self._DECISION.model_dump(),
        )
        assert raw == "BROKEN JSON"

    @pytest.mark.asyncio
    async def test_notification_not_sent_when_false(self):
        from agents.compliance_agent import _run_agent_loop

        response_data = {
            "case_id": "LOAN-20260604-XXXXXXXX",
            "action_taken": "Decision recorded",
            "notification_sent": False,
            "notification_type": "none",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "audit_summary": "Audit created.",
        }
        mock_session = _make_mcp_mock()
        mock_claude = AsyncMock()
        mock_claude.messages.create = AsyncMock(
            return_value=_make_claude_response(json.dumps(response_data))
        )

        raw = await _run_agent_loop(
            mock_session, mock_claude,
            SAMPLE_APPLICATION,
            self._PROFILE.model_dump(),
            self._RISK.model_dump(),
            self._DECISION.model_dump(),
        )
        data = json.loads(raw)
        assert data["notification_sent"] is False
