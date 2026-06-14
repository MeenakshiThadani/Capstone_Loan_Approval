"""
Layer 1 Tests: models/schemas.py

Validates that every Pydantic model enforces its field constraints and
business-rule validators correctly.

Coverage:
  - LoanApplicationRequest: valid input, invalid fields, cross-field validators
  - LoanStatusResponse / LoanApplicationResponse: happy path
  - Agent output models: type coercion and field ranges
  - LoanState: default values and optional fields
"""

import pytest
from pydantic import ValidationError

from models.schemas import (
    ApplicantProfile,
    ComplianceRecord,
    EmploymentType,
    FinancialRiskAssessment,
    LoanApplicationRequest,
    LoanApplicationResponse,
    LoanDecision,
    LoanDecisionOutput,
    LoanState,
    LoanStatusResponse,
    RiskLevel,
)
from datetime import datetime, timezone


# ─── Helpers ──────────────────────────────────────────────────────────────────

def valid_application(**overrides) -> dict:
    """Return a baseline valid application dict with optional field overrides."""
    base = dict(
        applicant_name="Jane Smith",
        annual_income=80000,
        monthly_debt=1200,
        loan_amount=20000,
        loan_purpose="Home renovation",
        credit_score=720,
        employment_type="full_time",
        years_employed=3,
        existing_loans=1,
        assets_value=50000,
    )
    base.update(overrides)
    return base


# ─── LoanApplicationRequest ────────────────────────────────────────────────────

class TestLoanApplicationRequest:
    """Tests for the primary API request model."""

    def test_valid_application_parses_successfully(self):
        """Happy path: a complete, valid application should parse without error."""
        app = LoanApplicationRequest(**valid_application())
        assert app.applicant_name == "Jane Smith"
        assert app.annual_income == 80000
        assert app.credit_score == 720
        assert app.employment_type == EmploymentType.FULL_TIME

    def test_all_employment_types_are_accepted(self):
        """Each EmploymentType enum value must be accepted by the model."""
        for emp_type in EmploymentType:
            app = LoanApplicationRequest(**valid_application(employment_type=emp_type.value))
            assert app.employment_type == emp_type

    def test_credit_score_minimum_boundary_300(self):
        """Credit score of exactly 300 (FICO minimum) should be valid."""
        app = LoanApplicationRequest(**valid_application(credit_score=300))
        assert app.credit_score == 300

    def test_credit_score_maximum_boundary_850(self):
        """Credit score of exactly 850 (FICO maximum) should be valid."""
        app = LoanApplicationRequest(**valid_application(credit_score=850))
        assert app.credit_score == 850

    def test_credit_score_below_300_raises(self):
        """Credit score below 300 is outside the FICO range and must be rejected."""
        with pytest.raises(ValidationError) as exc_info:
            LoanApplicationRequest(**valid_application(credit_score=299))
        assert "credit_score" in str(exc_info.value)

    def test_credit_score_above_850_raises(self):
        """Credit score above 850 is outside the FICO range and must be rejected."""
        with pytest.raises(ValidationError):
            LoanApplicationRequest(**valid_application(credit_score=851))

    def test_zero_annual_income_raises(self):
        """annual_income must be strictly positive (gt=0)."""
        with pytest.raises(ValidationError) as exc_info:
            LoanApplicationRequest(**valid_application(annual_income=0))
        assert "annual_income" in str(exc_info.value)

    def test_negative_annual_income_raises(self):
        """Negative income is not physically meaningful."""
        with pytest.raises(ValidationError):
            LoanApplicationRequest(**valid_application(annual_income=-1000))

    def test_negative_monthly_debt_raises(self):
        """Debt cannot be negative."""
        with pytest.raises(ValidationError):
            LoanApplicationRequest(**valid_application(monthly_debt=-1))

    def test_zero_monthly_debt_is_valid(self):
        """Zero monthly debt (no existing obligations) is a valid state."""
        app = LoanApplicationRequest(**valid_application(monthly_debt=0, existing_loans=0))
        assert app.monthly_debt == 0

    def test_zero_loan_amount_raises(self):
        """Loan amount must be strictly positive."""
        with pytest.raises(ValidationError):
            LoanApplicationRequest(**valid_application(loan_amount=0))

    def test_loan_amount_above_10_million_raises(self):
        """Loan amounts above $10M exceed the platform limit."""
        with pytest.raises(ValidationError) as exc_info:
            LoanApplicationRequest(**valid_application(loan_amount=10_000_001))
        assert "10,000,000" in str(exc_info.value)

    def test_loan_amount_exactly_10_million_is_valid(self):
        """Boundary: exactly $10M should be accepted."""
        app = LoanApplicationRequest(**valid_application(loan_amount=10_000_000))
        assert app.loan_amount == 10_000_000

    def test_name_too_short_raises(self):
        """Applicant name must be at least 2 characters."""
        with pytest.raises(ValidationError):
            LoanApplicationRequest(**valid_application(applicant_name="X"))

    def test_negative_years_employed_raises(self):
        """Negative tenure makes no sense."""
        with pytest.raises(ValidationError):
            LoanApplicationRequest(**valid_application(years_employed=-1))

    def test_monthly_debt_exceeds_150pct_of_income_raises(self):
        """Cross-field validator: monthly_debt > 1.5 × monthly_income is flagged."""
        monthly_income = 80000 / 12
        excessive_debt = monthly_income * 1.6
        with pytest.raises(ValidationError) as exc_info:
            LoanApplicationRequest(**valid_application(monthly_debt=excessive_debt))
        assert "150%" in str(exc_info.value)

    def test_optional_applicant_id_defaults_to_none(self):
        """applicant_id is optional and should default to None when omitted."""
        app = LoanApplicationRequest(**valid_application())
        assert app.applicant_id is None

    def test_default_existing_loans_is_zero(self):
        """existing_loans defaults to 0 when not provided."""
        data = valid_application()
        data.pop("existing_loans", None)
        app = LoanApplicationRequest(**data)
        assert app.existing_loans == 0

    def test_default_assets_value_is_zero(self):
        """assets_value defaults to 0.0 when not provided."""
        data = valid_application()
        data.pop("assets_value", None)
        app = LoanApplicationRequest(**data)
        assert app.assets_value == 0.0


# ─── LoanApplicationResponse ──────────────────────────────────────────────────

class TestLoanApplicationResponse:
    def test_valid_response(self):
        resp = LoanApplicationResponse(
            case_id="LOAN-20260604-ABCD1234",
            status="processing",
            message="Application received",
        )
        assert resp.case_id == "LOAN-20260604-ABCD1234"
        assert resp.status == "processing"

    def test_invalid_status_raises(self):
        """Status must be one of the three Literal values."""
        with pytest.raises(ValidationError):
            LoanApplicationResponse(case_id="X", status="pending", message="")


# ─── LoanStatusResponse ───────────────────────────────────────────────────────

class TestLoanStatusResponse:
    def test_minimal_status_response(self):
        """A processing response with only required fields should be valid."""
        resp = LoanStatusResponse(case_id="LOAN-123", status="processing")
        assert resp.decision is None
        assert resp.risk_score is None

    def test_full_completed_response(self):
        resp = LoanStatusResponse(
            case_id="LOAN-123",
            status="completed",
            decision=LoanDecision.APPROVED,
            risk_score=25.5,
            confidence_level=88.0,
            explanation="Approved based on strong credit profile.",
            key_factors=["Low DTI", "High credit score"],
            timestamp=datetime.now(timezone.utc),
        )
        assert resp.decision == LoanDecision.APPROVED
        assert resp.risk_score == 25.5

    def test_risk_score_cannot_exceed_100(self):
        with pytest.raises(ValidationError):
            LoanStatusResponse(
                case_id="X", status="completed", risk_score=101.0
            )

    def test_risk_score_cannot_be_negative(self):
        with pytest.raises(ValidationError):
            LoanStatusResponse(case_id="X", status="completed", risk_score=-1)


# ─── Agent Output Models ───────────────────────────────────────────────────────

class TestApplicantProfile:
    def test_valid_profile(self):
        profile = ApplicantProfile(
            income_stability_score=75.0,
            employment_risk=RiskLevel.LOW,
            credit_history_summary="Good credit — 720 FICO",
            completeness_flags=[],
        )
        assert profile.income_stability_score == 75.0
        assert profile.employment_risk == RiskLevel.LOW

    def test_stability_score_below_zero_raises(self):
        with pytest.raises(ValidationError):
            ApplicantProfile(
                income_stability_score=-1,
                employment_risk=RiskLevel.LOW,
                credit_history_summary="",
            )

    def test_stability_score_above_100_raises(self):
        with pytest.raises(ValidationError):
            ApplicantProfile(
                income_stability_score=101,
                employment_risk=RiskLevel.LOW,
                credit_history_summary="",
            )

    def test_completeness_flags_defaults_to_empty_list(self):
        profile = ApplicantProfile(
            income_stability_score=50,
            employment_risk=RiskLevel.MEDIUM,
            credit_history_summary="Fair",
        )
        assert profile.completeness_flags == []


class TestFinancialRiskAssessment:
    def test_valid_assessment(self):
        assessment = FinancialRiskAssessment(
            debt_to_income_ratio=0.32,
            credit_risk_level=RiskLevel.MEDIUM,
            loan_amount_risk=RiskLevel.LOW,
            anomalies=[],
            risk_reasoning="Moderate DTI; good credit score.",
        )
        assert assessment.debt_to_income_ratio == 0.32

    def test_anomalies_defaults_to_empty_list(self):
        assessment = FinancialRiskAssessment(
            debt_to_income_ratio=0.2,
            credit_risk_level=RiskLevel.LOW,
            loan_amount_risk=RiskLevel.LOW,
            risk_reasoning="Low risk across all metrics.",
        )
        assert assessment.anomalies == []


class TestLoanDecisionOutput:
    def test_approved_decision(self):
        output = LoanDecisionOutput(
            decision=LoanDecision.APPROVED,
            risk_score=22.0,
            confidence_level=91.5,
            key_factors=["Low DTI", "Prime credit"],
            explanation="Application approved.",
        )
        assert output.decision == LoanDecision.APPROVED
        assert output.risk_score == 22.0

    def test_risk_score_boundaries(self):
        """Risk score must be in [0, 100]."""
        with pytest.raises(ValidationError):
            LoanDecisionOutput(
                decision=LoanDecision.APPROVED,
                risk_score=100.1,
                confidence_level=90,
                key_factors=[],
                explanation="",
            )
        with pytest.raises(ValidationError):
            LoanDecisionOutput(
                decision=LoanDecision.APPROVED,
                risk_score=-0.1,
                confidence_level=90,
                key_factors=[],
                explanation="",
            )

    def test_all_three_decision_values(self):
        """Each LoanDecision variant must be accepted."""
        for decision in LoanDecision:
            output = LoanDecisionOutput(
                decision=decision,
                risk_score=50.0,
                confidence_level=60.0,
                key_factors=["test"],
                explanation="test",
            )
            assert output.decision == decision


# ─── LoanState ────────────────────────────────────────────────────────────────

class TestLoanState:
    def test_default_state_values(self):
        """A freshly created state should have default status and empty fields."""
        state = LoanState(application={"applicant_name": "Test"})
        assert state.status == "processing"
        assert state.profile is None
        assert state.risk_assessment is None
        assert state.decision is None
        assert state.compliance is None
        assert state.case_id is None
        assert state.error is None

    def test_state_accepts_populated_fields(self):
        state = LoanState(
            application={"applicant_name": "Test"},
            status="completed",
            case_id="LOAN-123",
            error=None,
        )
        assert state.case_id == "LOAN-123"
        assert state.status == "completed"
