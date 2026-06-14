"""
Shared Pydantic models (schemas) for the Loan Approval Agent system.

Every layer — API, agents, orchestrator, UI — imports from here so that
data contracts are defined in exactly one place.  Changing a field here
propagates to all consumers automatically.
"""

from __future__ import annotations

import logging
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)


# ─── Enumerations ─────────────────────────────────────────────────────────────


class EmploymentType(str, Enum):
    """
    Classifies how the applicant is employed.

    Affects income-stability scoring and employment-risk assessment:
    FULL_TIME is lowest risk; UNEMPLOYED triggers automatic rejection.
    """

    FULL_TIME = "full_time"
    PART_TIME = "part_time"
    CONTRACT = "contract"
    SELF_EMPLOYED = "self_employed"
    UNEMPLOYED = "unemployed"


class LoanDecision(str, Enum):
    """
    The three possible outcomes produced by the Loan Decision Agent.

    APPROVED        → risk score < 40 and credit criteria met
    REJECTED        → risk score ≥ 70 or hard disqualifiers present
    MANUAL_REVIEW   → borderline case requiring human underwriter
    """

    APPROVED = "approved"
    REJECTED = "rejected"
    MANUAL_REVIEW = "manual_review"


class RiskLevel(str, Enum):
    """Generic three-tier risk classification reused across multiple agents."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# ─── API Request / Response Models ────────────────────────────────────────────


class LoanApplicationRequest(BaseModel):
    """
    Incoming loan application submitted to POST /loan/apply.

    All monetary values are in USD.  Validation rules encode business
    constraints (e.g., credit scores must be valid FICO range 300–850).
    """

    # Applicant identification
    applicant_name: str = Field(
        ...,
        min_length=2,
        max_length=100,
        description="Full legal name of the applicant",
    )
    applicant_id: Optional[str] = Field(
        default=None,
        description="Optional existing customer ID; generated if absent",
    )

    # Financial details
    annual_income: float = Field(
        ..., gt=0, description="Gross annual income in USD"
    )
    monthly_debt: float = Field(
        ..., ge=0, description="Total existing monthly debt payments in USD"
    )
    loan_amount: float = Field(
        ..., gt=0, description="Requested loan principal in USD"
    )
    loan_purpose: str = Field(
        ..., min_length=3, max_length=200, description="Purpose of the loan"
    )

    # Credit profile
    credit_score: int = Field(
        ..., ge=300, le=850, description="FICO credit score"
    )

    # Employment
    employment_type: EmploymentType
    years_employed: float = Field(
        ..., ge=0, description="Years at current employer or self-employed"
    )

    # Optional enrichment
    existing_loans: int = Field(
        default=0, ge=0, description="Number of open loan accounts"
    )
    assets_value: float = Field(
        default=0.0, ge=0, description="Total value of declared assets in USD"
    )

    @field_validator("loan_amount")
    @classmethod
    def loan_must_not_exceed_sanity_limit(cls, v: float) -> float:
        """Reject absurdly large loan requests early (> $10 million)."""
        if v > 10_000_000:
            raise ValueError("Loan amount exceeds the maximum supported value of $10,000,000")
        return v

    @model_validator(mode="after")
    def monthly_debt_cannot_exceed_monthly_income(self) -> "LoanApplicationRequest":
        """
        Debt cannot exceed income — this would be a data-entry error or fraud.
        We flag it here rather than silently passing it to the risk agent.
        """
        monthly_income = self.annual_income / 12
        if self.monthly_debt > monthly_income * 1.5:
            raise ValueError(
                "monthly_debt exceeds 150% of monthly income — please verify the data"
            )
        return self


class LoanApplicationResponse(BaseModel):
    """
    Immediate response returned by POST /loan/apply.

    The application is processed asynchronously; the caller polls
    GET /loan/status/{case_id} for the final decision.
    """

    case_id: str = Field(..., description="Unique identifier for this application")
    status: Literal["processing", "completed", "error"]
    message: str


class LoanStatusResponse(BaseModel):
    """
    Full decision status returned by GET /loan/status/{case_id}.

    Populated progressively; fields are None until that processing stage
    is complete.
    """

    case_id: str
    status: Literal["processing", "completed", "error"]
    decision: Optional[LoanDecision] = None
    risk_score: Optional[float] = Field(None, ge=0, le=100)
    confidence_level: Optional[float] = Field(None, ge=0, le=100)
    explanation: Optional[str] = None
    key_factors: Optional[List[str]] = None
    timestamp: Optional[datetime] = None
    error_detail: Optional[str] = None


# ─── Agent Output Models ───────────────────────────────────────────────────────


class ApplicantProfile(BaseModel):
    """
    Structured output produced by the Applicant Profile Agent.

    income_stability_score  — 0-100, higher means more stable income
    employment_risk         — derived from employment type + tenure
    credit_history_summary  — human-readable digest of credit profile
    completeness_flags      — list of missing / suspicious fields found
    """

    income_stability_score: float = Field(..., ge=0, le=100)
    employment_risk: RiskLevel
    credit_history_summary: str
    completeness_flags: List[str] = Field(default_factory=list)


class FinancialRiskAssessment(BaseModel):
    """
    Structured output produced by the Financial Risk Analysis Agent.

    debt_to_income_ratio  — monthly_debt / (annual_income/12), expressed as a decimal
    credit_risk_level     — based on FICO score bands
    loan_amount_risk      — loan vs income multiples
    anomalies             — unusual patterns detected in the application
    risk_reasoning        — narrative explanation of the overall risk posture
    """

    debt_to_income_ratio: float = Field(..., ge=0, description="DTI as decimal, e.g. 0.35")
    credit_risk_level: RiskLevel
    loan_amount_risk: RiskLevel
    anomalies: List[str] = Field(default_factory=list)
    risk_reasoning: str


class LoanDecisionOutput(BaseModel):
    """
    Structured output produced by the Loan Decision Agent.

    risk_score      — composite 0–100 score (lower = safer)
    confidence_level — 0–100 % expressing how certain the model is
    key_factors     — ordered list of the most influential decision drivers
    explanation     — human-readable, audit-ready narrative
    """

    decision: LoanDecision
    risk_score: float = Field(..., ge=0, le=100)
    confidence_level: float = Field(..., ge=0, le=100)
    key_factors: List[str]
    explanation: str


class ComplianceRecord(BaseModel):
    """
    Structured output produced by the Compliance & Action Orchestrator Agent.

    Serves as the immutable audit trail entry for every application.
    """

    case_id: str
    action_taken: str
    notification_sent: bool
    notification_type: Optional[str] = Field(
        None, description="email | sms | none"
    )
    timestamp: datetime
    audit_summary: str


# ─── LangGraph Orchestrator State ─────────────────────────────────────────────


class LoanState(BaseModel):
    """
    State object threaded through every node in the LangGraph workflow.

    LangGraph merges the dict returned by each node into this state, so
    every field is Optional — nodes only populate what they own.

    Fields are intentionally Dict[str, Any] rather than typed models so that
    LangGraph's JSON serialisation round-trips cleanly without Pydantic
    discriminator complexity.
    """

    # ── Raw input ─────────────────────────────────────────────────────────────
    application: Dict[str, Any] = Field(
        default_factory=dict,
        description="LoanApplicationRequest serialised to dict",
    )

    # ── Agent outputs (populated sequentially) ────────────────────────────────
    profile: Optional[Dict[str, Any]] = Field(
        default=None, description="ApplicantProfile serialised to dict"
    )
    risk_assessment: Optional[Dict[str, Any]] = Field(
        default=None, description="FinancialRiskAssessment serialised to dict"
    )
    decision: Optional[Dict[str, Any]] = Field(
        default=None, description="LoanDecisionOutput serialised to dict"
    )
    compliance: Optional[Dict[str, Any]] = Field(
        default=None, description="ComplianceRecord serialised to dict"
    )

    # ── Workflow metadata ──────────────────────────────────────────────────────
    case_id: Optional[str] = None
    error: Optional[str] = None
    status: str = "processing"

    class Config:
        # Allow arbitrary types so LangGraph can attach its own metadata
        arbitrary_types_allowed = True
