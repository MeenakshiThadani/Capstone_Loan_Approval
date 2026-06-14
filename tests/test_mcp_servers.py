"""
Layer 2 Tests: MCP Server Tool Logic

Tests the pure Python business-logic functions inside each MCP server
directly (without starting a subprocess or using the MCP protocol).
This keeps the tests fast and deterministic.

Each server module exports private functions prefixed with `_`; we import
and test those directly.  Each test covers:
  - Happy path (expected inputs → expected outputs)
  - Edge cases (boundary values, zero inputs, max inputs)
  - Negative cases (inputs that should return high-risk / flag conditions)
"""

import sys
import importlib

import pytest

# ── Import the internal functions from each server module ─────────────────────
# Servers live in mcp/ which is NOT a Python package (no __init__.py),
# so we load them via importlib with an explicit path.

from pathlib import Path

_SERVER_DIR = Path(__file__).parent.parent / "mcp"


def _load_server(name: str):
    """Load a server module from the mcp/ directory by filename."""
    spec = importlib.util.spec_from_file_location(name, _SERVER_DIR / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Load all four servers once at module level
applicant_db = _load_server("applicant_db_server")
risk_rules_db = _load_server("risk_rules_db_server")
decision_synthesis = _load_server("decision_synthesis_server")
notification_system = _load_server("notification_system_server")


# ═══════════════════════════════════════════════════════════════════════════════
# ApplicantDB Server
# ═══════════════════════════════════════════════════════════════════════════════

class TestCalculateIncomeStability:
    """Tests for applicant_db_server._calculate_income_stability"""

    fn = staticmethod(applicant_db._calculate_income_stability)

    def test_unemployed_always_returns_zero(self):
        """Unemployed status must produce a score of 0 regardless of other inputs."""
        assert self.fn(100_000, "unemployed", 10) == 0.0

    def test_full_time_senior_employee_scores_high(self):
        """Full-time employee with 10 years tenure and high income should score >= 90."""
        score = self.fn(100_000, "full_time", 10)
        assert score >= 90.0

    def test_self_employed_short_tenure_scores_low(self):
        """Self-employed with 1 year is inherently risky — score should be < 60."""
        score = self.fn(30_000, "self_employed", 1)
        assert score < 60.0

    def test_score_capped_at_100(self):
        """Score must never exceed 100 even for perfect inputs."""
        score = self.fn(500_000, "full_time", 50)
        assert score <= 100.0

    def test_score_is_non_negative(self):
        """Score must always be >= 0."""
        for emp in ["full_time", "part_time", "contract", "self_employed"]:
            assert self.fn(0.01, emp, 0) >= 0.0

    def test_high_income_bonus_applied(self):
        """Income > $80k should yield a higher score than income < $40k, all else equal."""
        low = self.fn(30_000, "full_time", 3)
        high = self.fn(90_000, "full_time", 3)
        assert high > low

    def test_tenure_bonus_capped_at_20_pts(self):
        """Tenure bonus caps at 20 pts (10 years × 2 pts/yr = 20)."""
        score_10yr = self.fn(50_000, "full_time", 10)
        score_20yr = self.fn(50_000, "full_time", 20)
        # Both should score the same once tenure cap is reached
        assert score_10yr == score_20yr


class TestAssessEmploymentRisk:
    """Tests for applicant_db_server._assess_employment_risk"""

    fn = staticmethod(applicant_db._assess_employment_risk)

    def test_unemployed_always_high(self):
        assert self.fn("unemployed", 0) == "high"
        assert self.fn("unemployed", 100) == "high"

    def test_full_time_3_years_is_low(self):
        assert self.fn("full_time", 3) == "low"

    def test_full_time_less_than_1_year_is_high(self):
        assert self.fn("full_time", 0.5) == "high"

    def test_full_time_1_to_3_years_is_medium(self):
        assert self.fn("full_time", 2) == "medium"

    def test_self_employed_8_years_is_low(self):
        assert self.fn("self_employed", 8) == "low"

    def test_self_employed_2_years_is_high(self):
        assert self.fn("self_employed", 2) == "high"

    def test_contract_6_years_is_low(self):
        assert self.fn("contract", 6) == "low"

    def test_returns_only_valid_risk_levels(self):
        """All return values must be one of the three valid risk levels."""
        valid = {"low", "medium", "high"}
        for emp in ["full_time", "part_time", "contract", "self_employed", "unemployed"]:
            for years in [0, 1, 3, 6, 10]:
                result = self.fn(emp, years)
                assert result in valid, f"Got {result} for {emp}/{years}"


class TestGetCreditHistorySummary:
    """Tests for applicant_db_server._get_credit_history_summary"""

    fn = staticmethod(applicant_db._get_credit_history_summary)

    def test_exceptional_score_band(self):
        summary = self.fn(820, 0)
        assert "Exceptional" in summary
        assert "820" in summary

    def test_poor_score_band(self):
        summary = self.fn(520, 2)
        assert "Poor" in summary

    def test_good_score_band_boundary(self):
        summary = self.fn(670, 0)
        assert "Good" in summary

    def test_existing_loans_mentioned(self):
        summary = self.fn(700, 3)
        assert "3" in summary

    def test_no_existing_loans_message(self):
        summary = self.fn(700, 0)
        assert "No existing" in summary


class TestCheckApplicationCompleteness:
    """Tests for applicant_db_server._check_application_completeness"""

    fn = staticmethod(applicant_db._check_application_completeness)

    def test_complete_application_returns_empty_list(self):
        app = dict(
            applicant_name="Jane",
            annual_income=80000,
            monthly_debt=1000,
            loan_amount=25000,
            credit_score=720,
            employment_type="full_time",
            years_employed=3,
            loan_purpose="Home",
        )
        assert self.fn(app) == []

    def test_missing_required_field_flagged(self):
        flags = self.fn({"applicant_name": "Jane"})
        assert any("annual_income" in f for f in flags)

    def test_unemployed_with_years_flagged(self):
        app = dict(
            applicant_name="Jane", annual_income=0, monthly_debt=0,
            loan_amount=10000, credit_score=600, employment_type="unemployed",
            years_employed=5, loan_purpose="Car",
        )
        flags = self.fn(app)
        assert any("unemployed" in f.lower() or "inconsistency" in f.lower() for f in flags)

    def test_existing_loans_with_zero_debt_flagged(self):
        app = dict(
            applicant_name="Jane", annual_income=80000, monthly_debt=0,
            loan_amount=10000, credit_score=700, employment_type="full_time",
            years_employed=3, loan_purpose="Car", existing_loans=3,
        )
        flags = self.fn(app)
        assert any("debt" in f.lower() or "loan" in f.lower() for f in flags)


# ═══════════════════════════════════════════════════════════════════════════════
# RiskRulesDB Server
# ═══════════════════════════════════════════════════════════════════════════════

class TestCalculateDtiRatio:
    """Tests for risk_rules_db_server._calculate_dti_ratio"""

    fn = staticmethod(risk_rules_db._calculate_dti_ratio)

    def test_low_dti_classified_as_low_risk(self):
        result = self.fn(annual_income=120_000, monthly_debt=1_000)
        # DTI = 1000 / 10000 = 0.10 → low
        assert result["risk_level"] == "low"
        assert result["dti_ratio"] == pytest.approx(0.10, abs=0.01)

    def test_high_dti_classified_as_high_risk(self):
        result = self.fn(annual_income=36_000, monthly_debt=2_000)
        # DTI = 2000 / 3000 = 0.667 → high
        assert result["risk_level"] == "high"
        assert result["dti_ratio"] > 0.43

    def test_dti_boundary_28pct_is_low(self):
        # DTI just below 28% should be low
        result = self.fn(annual_income=120_000, monthly_debt=2_700)
        assert result["risk_level"] == "low"

    def test_zero_income_returns_high_risk(self):
        result = self.fn(annual_income=0, monthly_debt=500)
        assert result["risk_level"] == "high"

    def test_zero_debt_returns_zero_dti(self):
        result = self.fn(annual_income=60_000, monthly_debt=0)
        assert result["dti_ratio"] == 0.0
        assert result["risk_level"] == "low"

    def test_dti_ratio_is_decimal_not_percentage(self):
        """DTI should be expressed as a decimal (e.g. 0.35), not 35."""
        result = self.fn(annual_income=60_000, monthly_debt=1_750)
        assert result["dti_ratio"] < 1.0


class TestClassifyCreditScoreRisk:
    """Tests for risk_rules_db_server._classify_credit_score_risk"""

    fn = staticmethod(risk_rules_db._classify_credit_score_risk)

    def test_750_plus_is_low(self):
        assert self.fn(750)["risk_level"] == "low"
        assert self.fn(820)["risk_level"] == "low"

    def test_650_to_749_is_medium(self):
        assert self.fn(650)["risk_level"] == "medium"
        assert self.fn(700)["risk_level"] == "medium"
        assert self.fn(749)["risk_level"] == "medium"

    def test_580_to_649_is_high(self):
        assert self.fn(580)["risk_level"] == "high"
        assert self.fn(620)["risk_level"] == "high"

    def test_below_580_is_high_deep_subprime(self):
        result = self.fn(520)
        assert result["risk_level"] == "high"
        assert "Deep" in result["band"] or "Subprime" in result["band"]

    def test_boundary_750_is_low_not_medium(self):
        assert self.fn(750)["risk_level"] == "low"
        assert self.fn(749)["risk_level"] == "medium"


class TestEvaluateLoanAmountRisk:
    """Tests for risk_rules_db_server._evaluate_loan_amount_risk"""

    fn = staticmethod(risk_rules_db._evaluate_loan_amount_risk)

    def test_low_lti_is_low_risk(self):
        # LTI = 20000 / 80000 = 0.25 → well below 3× threshold
        result = self.fn(20_000, 80_000)
        assert result["risk_level"] == "low"

    def test_high_lti_is_high_risk(self):
        # LTI = 400000 / 50000 = 8× → high
        result = self.fn(400_000, 50_000)
        assert result["risk_level"] == "high"

    def test_assets_reduce_effective_risk(self):
        """Significant assets should reduce adjusted LTI and potentially the risk level."""
        without_assets = self.fn(250_000, 50_000, assets_value=0)
        with_assets = self.fn(250_000, 50_000, assets_value=400_000)
        # Assets should bring effective risk down
        risk_order = {"low": 0, "medium": 1, "high": 2}
        assert risk_order[with_assets["risk_level"]] <= risk_order[without_assets["risk_level"]]

    def test_zero_income_returns_high_risk(self):
        result = self.fn(10_000, 0)
        assert result["risk_level"] == "high"


class TestDetectAnomalies:
    """Tests for risk_rules_db_server._detect_anomalies"""

    fn = staticmethod(risk_rules_db._detect_anomalies)

    def test_clean_application_has_no_anomalies(self):
        app = dict(
            annual_income=80_000, monthly_debt=1_200, loan_amount=25_000,
            credit_score=720, existing_loans=1, years_employed=4, assets_value=50_000,
        )
        assert self.fn(app) == []

    def test_high_income_with_low_credit_score_flagged(self):
        app = dict(
            annual_income=200_000, credit_score=550, monthly_debt=500,
            loan_amount=50_000, existing_loans=0, years_employed=5, assets_value=0,
        )
        flags = self.fn(app)
        assert len(flags) > 0
        assert any("income" in f.lower() or "credit" in f.lower() for f in flags)

    def test_extreme_loan_to_income_ratio_flagged(self):
        app = dict(
            annual_income=30_000, monthly_debt=500, loan_amount=3_000_000,
            credit_score=700, existing_loans=0, years_employed=5, assets_value=0,
        )
        flags = self.fn(app)
        assert any("loan" in f.lower() or "income" in f.lower() for f in flags)

    def test_many_loans_with_zero_debt_flagged(self):
        app = dict(
            annual_income=80_000, monthly_debt=0, loan_amount=20_000,
            credit_score=700, existing_loans=5, years_employed=3, assets_value=0,
        )
        flags = self.fn(app)
        assert len(flags) > 0

    def test_implausibly_long_tenure_flagged(self):
        app = dict(
            annual_income=60_000, monthly_debt=1_000, loan_amount=20_000,
            credit_score=700, existing_loans=0, years_employed=50, assets_value=0,
        )
        flags = self.fn(app)
        assert any("years" in f.lower() or "employ" in f.lower() for f in flags)


# ═══════════════════════════════════════════════════════════════════════════════
# DecisionSynthesis Server
# ═══════════════════════════════════════════════════════════════════════════════

class TestComputeRiskScore:
    """Tests for decision_synthesis_server._compute_risk_score"""

    fn = staticmethod(decision_synthesis._compute_risk_score)

    def test_perfect_applicant_scores_near_zero(self):
        """An ideal applicant should score close to 0 (very safe)."""
        result = self.fn(
            credit_score=850, dti_ratio=0.05, employment_risk="low",
            loan_amount_risk="low", income_stability_score=100,
            anomaly_count=0, completeness_flag_count=0,
        )
        assert result["risk_score"] < 20.0

    def test_terrible_applicant_scores_near_100(self):
        """A worst-case applicant should score close to 100."""
        result = self.fn(
            credit_score=300, dti_ratio=0.6, employment_risk="high",
            loan_amount_risk="high", income_stability_score=0,
            anomaly_count=5, completeness_flag_count=3,
        )
        assert result["risk_score"] > 80.0

    def test_score_is_within_0_to_100(self):
        """Score must always be in [0, 100]."""
        result = self.fn(
            credit_score=700, dti_ratio=0.3, employment_risk="medium",
            loan_amount_risk="medium", income_stability_score=60,
        )
        assert 0 <= result["risk_score"] <= 100

    def test_anomaly_penalty_increases_score(self):
        """Adding anomalies should increase (worsen) the risk score."""
        base = self.fn(
            credit_score=700, dti_ratio=0.3, employment_risk="low",
            loan_amount_risk="low", income_stability_score=75, anomaly_count=0,
        )
        with_anomalies = self.fn(
            credit_score=700, dti_ratio=0.3, employment_risk="low",
            loan_amount_risk="low", income_stability_score=75, anomaly_count=5,
        )
        assert with_anomalies["risk_score"] > base["risk_score"]

    def test_components_dict_is_returned(self):
        result = self.fn(
            credit_score=700, dti_ratio=0.3, employment_risk="low",
            loan_amount_risk="low", income_stability_score=70,
        )
        assert "components" in result
        assert "credit_score" in result["components"]


class TestCalculateConfidence:
    """Tests for decision_synthesis_server._calculate_confidence"""

    fn = staticmethod(decision_synthesis._calculate_confidence)

    def test_very_low_risk_score_yields_high_confidence(self):
        """A risk score of 10 (far from the 40 approval boundary) → high confidence."""
        result = self.fn(risk_score=10, completeness_flag_count=0, anomaly_count=0)
        assert result["confidence_level"] >= 70.0

    def test_borderline_risk_score_yields_lower_confidence(self):
        """A risk score right on the boundary (e.g. 39 or 41) → lower confidence."""
        clear = self.fn(risk_score=10, completeness_flag_count=0, anomaly_count=0)
        borderline = self.fn(risk_score=39, completeness_flag_count=0, anomaly_count=0)
        assert clear["confidence_level"] > borderline["confidence_level"]

    def test_data_quality_penalty_reduces_confidence(self):
        clean = self.fn(risk_score=20, completeness_flag_count=0, anomaly_count=0)
        dirty = self.fn(risk_score=20, completeness_flag_count=3, anomaly_count=2)
        assert dirty["confidence_level"] < clean["confidence_level"]

    def test_confidence_minimum_is_30(self):
        """Even the worst data quality should not drop confidence below 30%."""
        result = self.fn(risk_score=50, completeness_flag_count=20, anomaly_count=10)
        assert result["confidence_level"] >= 30.0

    def test_confidence_maximum_is_99(self):
        result = self.fn(risk_score=0, completeness_flag_count=0, anomaly_count=0)
        assert result["confidence_level"] <= 99.0


class TestDetermineDecision:
    """Tests for decision_synthesis_server._determine_decision"""

    fn = staticmethod(decision_synthesis._determine_decision)

    def test_low_risk_score_approves(self):
        result = self.fn(risk_score=25, credit_score=750, dti_ratio=0.25)
        assert result["decision"] == "approved"

    def test_high_risk_score_rejects(self):
        result = self.fn(risk_score=75, credit_score=650, dti_ratio=0.35)
        assert result["decision"] == "rejected"

    def test_borderline_score_manual_review(self):
        result = self.fn(risk_score=55, credit_score=650, dti_ratio=0.35)
        assert result["decision"] == "manual_review"

    def test_unemployed_hard_rejects(self):
        """Unemployed is a hard disqualifier regardless of risk score."""
        result = self.fn(risk_score=10, credit_score=800, dti_ratio=0.1, employment_type="unemployed")
        assert result["decision"] == "rejected"

    def test_very_low_credit_score_hard_rejects(self):
        """Credit score below 550 is a hard disqualifier."""
        result = self.fn(risk_score=10, credit_score=540, dti_ratio=0.2)
        assert result["decision"] == "rejected"

    def test_excessive_dti_hard_rejects(self):
        """DTI > 55% triggers a hard rejection."""
        result = self.fn(risk_score=20, credit_score=750, dti_ratio=0.56)
        assert result["decision"] == "rejected"

    def test_three_anomalies_trigger_manual_review(self):
        """Three or more anomalies should route to manual review, not outright rejection."""
        result = self.fn(risk_score=20, credit_score=750, dti_ratio=0.2, anomaly_count=3)
        assert result["decision"] == "manual_review"

    def test_approval_boundary_exactly_39_point_9(self):
        """Score just below 40 should be approved."""
        result = self.fn(risk_score=39.9, credit_score=700, dti_ratio=0.3)
        assert result["decision"] == "approved"

    def test_rejection_boundary_exactly_70(self):
        """Score at exactly 70 should be rejected."""
        result = self.fn(risk_score=70, credit_score=650, dti_ratio=0.35)
        assert result["decision"] == "rejected"


class TestGenerateExplanation:
    """Tests for decision_synthesis_server._generate_explanation"""

    fn = staticmethod(decision_synthesis._generate_explanation)

    def test_explanation_contains_applicant_name(self):
        text = self.fn("approved", 25.0, 90.0, ["Low DTI"], "Jane Smith", 20000)
        assert "Jane Smith" in text

    def test_explanation_contains_loan_amount(self):
        text = self.fn("approved", 25.0, 90.0, ["Low DTI"], "Jane Smith", 20000)
        assert "20,000" in text

    def test_approved_explanation_mentions_approved(self):
        text = self.fn("approved", 25.0, 90.0, ["Low DTI"], "Jane", 10000)
        assert "APPROVED" in text.upper()

    def test_rejected_explanation_mentions_declined(self):
        text = self.fn("rejected", 80.0, 85.0, ["High DTI"], "Bob", 50000)
        assert "DECLINED" in text.upper() or "REJECTED" in text.upper()

    def test_manual_review_explanation_mentions_review(self):
        text = self.fn("manual_review", 55.0, 50.0, ["Borderline score"], "Alice", 30000)
        assert "REVIEW" in text.upper()

    def test_key_factors_appear_in_explanation(self):
        factors = ["Low DTI", "Prime credit score"]
        text = self.fn("approved", 25.0, 90.0, factors, "Jane", 20000)
        for factor in factors:
            assert factor in text


# ═══════════════════════════════════════════════════════════════════════════════
# NotificationSystem Server
# ═══════════════════════════════════════════════════════════════════════════════

class TestGenerateCaseId:
    """Tests for notification_system_server._generate_case_id"""

    fn = staticmethod(notification_system._generate_case_id)

    def test_case_id_starts_with_loan_prefix(self):
        assert self.fn().startswith("LOAN-")

    def test_case_id_contains_date(self):
        from datetime import datetime, timezone
        today = datetime.now(timezone.utc).strftime("%Y%m%d")
        assert today in self.fn()

    def test_case_id_is_unique_on_each_call(self):
        ids = {self.fn() for _ in range(50)}
        assert len(ids) == 50, "Case IDs must be globally unique"

    def test_case_id_format(self):
        """Format: LOAN-YYYYMMDD-XXXXXXXX (8 uppercase hex chars)."""
        case_id = self.fn("Test Applicant")
        parts = case_id.split("-")
        assert len(parts) == 3
        assert parts[0] == "LOAN"
        assert len(parts[1]) == 8  # YYYYMMDD
        assert len(parts[2]) == 8  # hex suffix


class TestRecordAction:
    """Tests for notification_system_server._record_action"""

    fn = staticmethod(notification_system._record_action)

    def test_records_action_successfully(self):
        result = self.fn("LOAN-TEST-001", "loan_approved", "approved", 25.0, 90.0)
        assert result is True

    def test_records_rejected_action(self):
        result = self.fn("LOAN-TEST-002", "loan_rejected", "rejected", 75.0, 85.0)
        assert result is True


class TestSendNotification:
    """Tests for notification_system_server._send_notification"""

    fn = staticmethod(notification_system._send_notification)

    def test_email_notification_sent(self):
        result = self.fn("LOAN-123", "Jane Smith", "approved", "email")
        assert result["notification_sent"] is True
        assert "email" in result["channels"]

    def test_sms_notification_sent(self):
        result = self.fn("LOAN-123", "Bob Jones", "rejected", "sms")
        assert "sms" in result["channels"]

    def test_both_channels_notification(self):
        result = self.fn("LOAN-123", "Alice", "manual_review", "both")
        assert "email" in result["channels"]
        assert "sms" in result["channels"]

    def test_none_notification_type(self):
        result = self.fn("LOAN-123", "Alice", "approved", "none")
        assert result["notification_sent"] is False

    def test_message_preview_contains_case_id(self):
        result = self.fn("LOAN-UNIQUE-99", "Alice", "approved", "email")
        assert "LOAN-UNIQUE-99" in result["message_preview"]

    def test_approved_message_is_positive(self):
        result = self.fn("LOAN-123", "Jane", "approved", "email")
        assert "approved" in result["message_preview"].lower() or "congratulations" in result["message_preview"].lower()

    def test_rejected_message_is_informative(self):
        result = self.fn("LOAN-123", "Bob", "rejected", "email")
        assert "not been approved" in result["message_preview"].lower() or "declined" in result["message_preview"].lower()


class TestCreateAuditRecord:
    """Tests for notification_system_server._create_audit_record"""

    fn = staticmethod(notification_system._create_audit_record)

    def _sample_inputs(self):
        return dict(
            case_id="LOAN-20260604-ABCD1234",
            application=dict(
                applicant_name="Jane Smith", loan_amount=25000, loan_purpose="Home"
            ),
            profile=dict(
                income_stability_score=75, employment_risk="low", completeness_flags=[]
            ),
            risk_assessment=dict(
                debt_to_income_ratio=0.25, credit_risk_level="low", anomalies=[]
            ),
            decision_output=dict(
                decision="approved", risk_score=22.0, confidence_level=91.0,
                key_factors=["Low DTI", "Good credit"],
            ),
        )

    def test_audit_record_contains_case_id(self):
        record = self.fn(**self._sample_inputs())
        assert "LOAN-20260604-ABCD1234" in record

    def test_audit_record_contains_applicant_name(self):
        record = self.fn(**self._sample_inputs())
        assert "Jane Smith" in record

    def test_audit_record_contains_decision(self):
        record = self.fn(**self._sample_inputs())
        assert "approved" in record.lower() or "APPROVED" in record

    def test_audit_record_contains_risk_score(self):
        record = self.fn(**self._sample_inputs())
        assert "22.0" in record or "22" in record

    def test_audit_record_is_non_empty_string(self):
        record = self.fn(**self._sample_inputs())
        assert isinstance(record, str)
        assert len(record) > 100
