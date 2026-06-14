"""
MCP Server: DecisionSynthesis

Exposes tools consumed exclusively by the Loan Decision Agent.
Responsibilities modelled here:
  - Composite risk score computation
  - Confidence level calculation
  - Decision threshold evaluation (approve/reject/review)
  - Human-readable explanation generation

Run standalone:
    python mcp/decision_synthesis_server.py
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any, Dict, List

from mcp.server import NotificationOptions, Server
from mcp.server.models import InitializationOptions
import mcp.server.stdio as stdio_server
import mcp.types as types

logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger("decision_synthesis_server")

server = Server("DecisionSynthesis")


# ─── Tool definitions ──────────────────────────────────────────────────────────


@server.list_tools()
async def list_tools() -> List[types.Tool]:
    """Advertise all decision-synthesis tools to the Loan Decision Agent."""
    return [
        types.Tool(
            name="compute_risk_score",
            description=(
                "Calculates a composite risk score (0-100, lower is safer) "
                "by weighting inputs from the profile and risk-assessment agents. "
                "Returns the score plus a breakdown of each contributing component."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "credit_score": {"type": "integer"},
                    "dti_ratio": {"type": "number"},
                    "employment_risk": {
                        "type": "string",
                        "enum": ["low", "medium", "high"],
                    },
                    "loan_amount_risk": {
                        "type": "string",
                        "enum": ["low", "medium", "high"],
                    },
                    "income_stability_score": {"type": "number"},
                    "anomaly_count": {
                        "type": "integer",
                        "description": "Number of anomalies detected",
                        "default": 0,
                    },
                    "completeness_flag_count": {
                        "type": "integer",
                        "description": "Number of completeness flags",
                        "default": 0,
                    },
                },
                "required": [
                    "credit_score",
                    "dti_ratio",
                    "employment_risk",
                    "loan_amount_risk",
                    "income_stability_score",
                ],
            },
        ),
        types.Tool(
            name="calculate_confidence",
            description=(
                "Calculates model confidence (0-100%) based on data completeness "
                "and the spread between the risk score and nearest decision boundary."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "risk_score": {"type": "number"},
                    "completeness_flag_count": {"type": "integer"},
                    "anomaly_count": {"type": "integer"},
                },
                "required": ["risk_score", "completeness_flag_count", "anomaly_count"],
            },
        ),
        types.Tool(
            name="determine_decision",
            description=(
                "Applies decision thresholds to produce a final classification: "
                "approved | rejected | manual_review."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "risk_score": {"type": "number"},
                    "credit_score": {"type": "integer"},
                    "dti_ratio": {"type": "number"},
                    "employment_type": {"type": "string"},
                    "anomaly_count": {"type": "integer", "default": 0},
                },
                "required": ["risk_score", "credit_score", "dti_ratio"],
            },
        ),
        types.Tool(
            name="generate_explanation",
            description=(
                "Generates a human-readable, audit-ready explanation of the "
                "decision, citing the key factors in order of importance."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "decision": {
                        "type": "string",
                        "enum": ["approved", "rejected", "manual_review"],
                    },
                    "risk_score": {"type": "number"},
                    "confidence_level": {"type": "number"},
                    "key_factors": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Ordered list of decision drivers",
                    },
                    "applicant_name": {"type": "string"},
                    "loan_amount": {"type": "number"},
                },
                "required": [
                    "decision",
                    "risk_score",
                    "confidence_level",
                    "key_factors",
                    "applicant_name",
                    "loan_amount",
                ],
            },
        ),
    ]


# ─── Tool implementations ──────────────────────────────────────────────────────

# Risk level numeric mappings used in score computation
_RISK_LEVEL_MAP = {"low": 0, "medium": 1, "high": 2}


def _compute_risk_score(
    credit_score: int,
    dti_ratio: float,
    employment_risk: str,
    loan_amount_risk: str,
    income_stability_score: float,
    anomaly_count: int = 0,
    completeness_flag_count: int = 0,
) -> Dict[str, Any]:
    """
    Weighted composite risk score (0–100).

    Component weights (sum = 100):
      Credit score component      : 35 pts
      DTI component               : 25 pts
      Employment risk component   : 20 pts
      Loan amount risk component  : 10 pts
      Income stability (inverse)  :  5 pts
      Anomaly penalty             :  5 pts (up to 5 anomalies × 1 pt)

    Each component scores 0 (best) to its maximum weight (worst).
    """
    components: Dict[str, float] = {}

    # ── Credit score component (35 pts, lower score → more risk) ──────────────
    # Map 300-850 range to 0-35 risk points inversely
    # 850 → 0 pts of risk, 300 → 35 pts of risk
    credit_risk_pts = (850 - credit_score) / (850 - 300) * 35
    components["credit_score"] = round(credit_risk_pts, 2)

    # ── DTI component (25 pts) ─────────────────────────────────────────────────
    # DTI 0 → 0 risk, DTI ≥ 0.5 → 25 risk
    dti_risk_pts = min(dti_ratio / 0.5, 1.0) * 25
    components["dti_ratio"] = round(dti_risk_pts, 2)

    # ── Employment risk (20 pts) ───────────────────────────────────────────────
    emp_map = {"low": 0, "medium": 10, "high": 20}
    components["employment_risk"] = float(emp_map.get(employment_risk, 20))

    # ── Loan amount risk (10 pts) ──────────────────────────────────────────────
    loan_map = {"low": 0, "medium": 5, "high": 10}
    components["loan_amount_risk"] = float(loan_map.get(loan_amount_risk, 10))

    # ── Income stability inverse (5 pts) ──────────────────────────────────────
    # High stability (100) → 0 risk pts; low stability (0) → 5 risk pts
    income_risk_pts = (100 - income_stability_score) / 100 * 5
    components["income_stability"] = round(income_risk_pts, 2)

    # ── Anomaly penalty (5 pts) ───────────────────────────────────────────────
    anomaly_pts = min(anomaly_count, 5) * 1.0
    components["anomalies"] = anomaly_pts

    # ── Completeness penalty ───────────────────────────────────────────────────
    completeness_pts = min(completeness_flag_count * 2, 10)
    components["completeness_flags"] = completeness_pts

    total = sum(components.values())
    # Cap at 100
    final_score = round(min(total, 100.0), 2)

    return {
        "risk_score": final_score,
        "components": components,
    }


def _calculate_confidence(
    risk_score: float,
    completeness_flag_count: int,
    anomaly_count: int,
) -> Dict[str, Any]:
    """
    Confidence = distance from nearest decision boundary × data quality factor.

    Decision boundaries: approve < 40, review 40-69, reject ≥ 70.
    A score of 20 (far from boundary 40) yields high confidence.
    A score of 38 (near boundary 40) yields lower confidence.

    Data quality penalty: −5% per flag, −10% per anomaly, min 30%.
    """
    # Distance-based base confidence
    approve_boundary = 40.0
    reject_boundary = 70.0

    if risk_score < approve_boundary:
        # Distance from approve boundary (higher distance = more confident)
        distance = approve_boundary - risk_score
        base_confidence = 60 + (distance / approve_boundary) * 40
    elif risk_score >= reject_boundary:
        distance = risk_score - reject_boundary
        base_confidence = 60 + (distance / (100 - reject_boundary)) * 40
    else:
        # In manual-review zone — confidence is lower by definition
        mid = (approve_boundary + reject_boundary) / 2
        distance_from_mid = abs(risk_score - mid)
        base_confidence = 40 + (distance_from_mid / (mid - approve_boundary)) * 20

    # Data quality penalty
    flag_penalty = completeness_flag_count * 5
    anomaly_penalty = anomaly_count * 10
    total_penalty = flag_penalty + anomaly_penalty

    confidence = max(base_confidence - total_penalty, 30.0)

    return {
        "confidence_level": round(min(confidence, 99.0), 2),
        "base_confidence": round(base_confidence, 2),
        "penalty_applied": total_penalty,
    }


def _determine_decision(
    risk_score: float,
    credit_score: int,
    dti_ratio: float,
    employment_type: str = "full_time",
    anomaly_count: int = 0,
) -> Dict[str, Any]:
    """
    Decision thresholds with hard-override rules.

    Hard rejections (override risk score):
      - credit_score < 550
      - employment_type == "unemployed"
      - dti_ratio > 0.55
      - anomaly_count >= 3

    Thresholds (after hard rules):
      risk_score < 40  → approved
      risk_score ≥ 70  → rejected
      else             → manual_review
    """
    key_factors: List[str] = []

    # ── Hard rejection rules ───────────────────────────────────────────────────
    if employment_type == "unemployed":
        key_factors.append("Applicant is unemployed (hard disqualifier)")
        return {
            "decision": "rejected",
            "reason": "hard_rule",
            "key_factors": key_factors,
        }

    if credit_score < 550:
        key_factors.append(f"Credit score {credit_score} below minimum threshold of 550")
        return {
            "decision": "rejected",
            "reason": "hard_rule",
            "key_factors": key_factors,
        }

    if dti_ratio > 0.55:
        key_factors.append(f"DTI ratio {dti_ratio:.1%} exceeds maximum of 55%")
        return {
            "decision": "rejected",
            "reason": "hard_rule",
            "key_factors": key_factors,
        }

    if anomaly_count >= 3:
        key_factors.append(f"{anomaly_count} data anomalies detected — requires human review")
        return {
            "decision": "manual_review",
            "reason": "anomaly_threshold",
            "key_factors": key_factors,
        }

    # ── Score-based decision ───────────────────────────────────────────────────
    if risk_score < 40:
        decision = "approved"
        key_factors.append(f"Risk score {risk_score:.1f} is below approval threshold of 40")
    elif risk_score >= 70:
        decision = "rejected"
        key_factors.append(f"Risk score {risk_score:.1f} exceeds rejection threshold of 70")
    else:
        decision = "manual_review"
        key_factors.append(f"Risk score {risk_score:.1f} falls in manual-review band (40–70)")

    # Contributing factors for transparency
    if credit_score >= 750:
        key_factors.append(f"Excellent credit score ({credit_score})")
    elif credit_score < 650:
        key_factors.append(f"Below-average credit score ({credit_score})")

    if dti_ratio < 0.28:
        key_factors.append(f"Low DTI ratio ({dti_ratio:.1%})")
    elif dti_ratio > 0.43:
        key_factors.append(f"High DTI ratio ({dti_ratio:.1%})")

    return {"decision": decision, "reason": "score_threshold", "key_factors": key_factors}


def _generate_explanation(
    decision: str,
    risk_score: float,
    confidence_level: float,
    key_factors: List[str],
    applicant_name: str,
    loan_amount: float,
) -> str:
    """
    Produces a plain-English, audit-ready explanation of the decision.

    Designed to be readable by both applicants and compliance reviewers.
    """
    decision_phrases = {
        "approved": "has been APPROVED",
        "rejected": "has been DECLINED",
        "manual_review": "requires MANUAL REVIEW by an underwriter",
    }
    phrase = decision_phrases.get(decision, "has an UNDETERMINED outcome")

    factors_text = "\n".join(f"  • {f}" for f in key_factors)

    explanation = (
        f"Loan application for {applicant_name} (amount: ${loan_amount:,.2f}) "
        f"{phrase}.\n\n"
        f"Risk Score: {risk_score:.1f}/100  |  Confidence: {confidence_level:.1f}%\n\n"
        f"Key Decision Factors:\n{factors_text}\n\n"
    )

    if decision == "approved":
        explanation += (
            "The applicant meets all underwriting criteria. "
            "Proceed with standard loan documentation and disbursement."
        )
    elif decision == "rejected":
        explanation += (
            "The application does not meet minimum lending criteria. "
            "The applicant may reapply after addressing the factors above."
        )
    else:
        explanation += (
            "This case has been flagged for human underwriter review due to "
            "borderline risk indicators or data quality concerns. "
            "A decision will be communicated within 2 business days."
        )

    return explanation


# ─── Tool dispatcher ───────────────────────────────────────────────────────────


@server.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> List[types.TextContent]:
    """Dispatch incoming tool requests to implementation functions."""
    logger.info("Tool called: %s", name)

    if name == "compute_risk_score":
        result = _compute_risk_score(
            credit_score=arguments["credit_score"],
            dti_ratio=arguments["dti_ratio"],
            employment_risk=arguments["employment_risk"],
            loan_amount_risk=arguments["loan_amount_risk"],
            income_stability_score=arguments["income_stability_score"],
            anomaly_count=arguments.get("anomaly_count", 0),
            completeness_flag_count=arguments.get("completeness_flag_count", 0),
        )
        return [types.TextContent(type="text", text=json.dumps(result))]

    elif name == "calculate_confidence":
        result = _calculate_confidence(
            risk_score=arguments["risk_score"],
            completeness_flag_count=arguments["completeness_flag_count"],
            anomaly_count=arguments["anomaly_count"],
        )
        return [types.TextContent(type="text", text=json.dumps(result))]

    elif name == "determine_decision":
        result = _determine_decision(
            risk_score=arguments["risk_score"],
            credit_score=arguments["credit_score"],
            dti_ratio=arguments["dti_ratio"],
            employment_type=arguments.get("employment_type", "full_time"),
            anomaly_count=arguments.get("anomaly_count", 0),
        )
        return [types.TextContent(type="text", text=json.dumps(result))]

    elif name == "generate_explanation":
        result = _generate_explanation(
            decision=arguments["decision"],
            risk_score=arguments["risk_score"],
            confidence_level=arguments["confidence_level"],
            key_factors=arguments["key_factors"],
            applicant_name=arguments["applicant_name"],
            loan_amount=arguments["loan_amount"],
        )
        return [types.TextContent(type="text", text=json.dumps({"explanation": result}))]

    else:
        raise ValueError(f"Unknown tool: {name}")


# ─── Entry point ───────────────────────────────────────────────────────────────


async def main() -> None:
    """Launch the DecisionSynthesis MCP server over stdio transport."""
    async with stdio_server.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="DecisionSynthesis",
                server_version="1.0.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
