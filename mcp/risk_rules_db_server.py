"""
MCP Server: RiskRulesDB

Exposes tools consumed exclusively by the Financial Risk Analysis Agent.
Responsibilities modelled here:
  - Debt-to-Income ratio computation
  - Credit score risk classification
  - Loan amount risk evaluation
  - Anomaly detection in application data

Run standalone:
    python mcp/risk_rules_db_server.py
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
logger = logging.getLogger("risk_rules_db_server")

server = Server("RiskRulesDB")


# ─── Tool definitions ──────────────────────────────────────────────────────────


@server.list_tools()
async def list_tools() -> List[types.Tool]:
    """Advertise all risk-analysis tools to the Financial Risk Agent."""
    return [
        types.Tool(
            name="calculate_dti_ratio",
            description=(
                "Computes the Debt-to-Income (DTI) ratio as "
                "monthly_debt / (annual_income / 12).  Returns the ratio as "
                "a decimal (e.g. 0.35 = 35% DTI) and a risk classification."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "annual_income": {"type": "number"},
                    "monthly_debt": {"type": "number"},
                },
                "required": ["annual_income", "monthly_debt"],
            },
        ),
        types.Tool(
            name="classify_credit_score_risk",
            description=(
                "Returns a risk level (low/medium/high) for a given FICO score "
                "using standard underwriting thresholds."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "credit_score": {"type": "integer", "minimum": 300, "maximum": 850}
                },
                "required": ["credit_score"],
            },
        ),
        types.Tool(
            name="evaluate_loan_amount_risk",
            description=(
                "Assesses whether the requested loan amount is reasonable "
                "relative to annual income.  Returns low/medium/high risk "
                "and an explanation."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "loan_amount": {"type": "number"},
                    "annual_income": {"type": "number"},
                    "assets_value": {
                        "type": "number",
                        "description": "Total declared assets (optional)",
                        "default": 0,
                    },
                },
                "required": ["loan_amount", "annual_income"],
            },
        ),
        types.Tool(
            name="detect_anomalies",
            description=(
                "Scans the full application for statistical anomalies and "
                "potential fraud indicators.  Returns a list of anomaly "
                "descriptions; empty list means no anomalies detected."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "application": {
                        "type": "object",
                        "description": "Full loan application as JSON",
                    }
                },
                "required": ["application"],
            },
        ),
    ]


# ─── Tool implementations ──────────────────────────────────────────────────────


def _calculate_dti_ratio(annual_income: float, monthly_debt: float) -> Dict[str, Any]:
    """
    DTI = monthly_debt / monthly_income.

    Standard industry thresholds:
      < 0.28  → low risk (front-end guideline for housing)
      0.28-0.36 → medium
      0.36-0.43 → elevated (back-end limit for QM loans)
      > 0.43  → high risk
    """
    monthly_income = annual_income / 12
    # Guard against zero income (should have been caught by validator, but be safe)
    if monthly_income <= 0:
        return {"dti_ratio": 1.0, "risk_level": "high", "explanation": "Zero income"}

    dti = monthly_debt / monthly_income

    if dti < 0.28:
        risk_level = "low"
        explanation = f"DTI of {dti:.1%} is well within conservative underwriting limits."
    elif dti < 0.36:
        risk_level = "medium"
        explanation = f"DTI of {dti:.1%} is moderate; within standard back-end limits."
    elif dti < 0.43:
        risk_level = "medium"
        explanation = (
            f"DTI of {dti:.1%} approaches the QM back-end limit of 43%; elevated risk."
        )
    else:
        risk_level = "high"
        explanation = (
            f"DTI of {dti:.1%} exceeds the 43% QM threshold; high default probability."
        )

    return {"dti_ratio": round(dti, 4), "risk_level": risk_level, "explanation": explanation}


def _classify_credit_score_risk(credit_score: int) -> Dict[str, str]:
    """
    FICO-based risk tier mapping used by most US lenders.

    750+  → low     (prime / super-prime)
    650-749 → medium (near-prime)
    580-649 → high   (subprime)
    <580   → high   (deep subprime — automatic rejection trigger)
    """
    if credit_score >= 750:
        return {
            "risk_level": "low",
            "band": "Prime/Super-Prime",
            "note": "Qualifies for best available rates.",
        }
    elif credit_score >= 650:
        return {
            "risk_level": "medium",
            "band": "Near-Prime",
            "note": "Standard rates; may require compensating factors.",
        }
    elif credit_score >= 580:
        return {
            "risk_level": "high",
            "band": "Subprime",
            "note": "High risk; requires additional collateral or co-signer.",
        }
    else:
        return {
            "risk_level": "high",
            "band": "Deep Subprime",
            "note": "Below minimum threshold; likely rejection.",
        }


def _evaluate_loan_amount_risk(
    loan_amount: float, annual_income: float, assets_value: float = 0.0
) -> Dict[str, Any]:
    """
    Loan-to-income (LTI) ratio assessment.

    LTI = loan_amount / annual_income
    Adjusts slightly if assets_value provides meaningful collateral (LTV consideration).

    Thresholds:
      LTI < 3   → low
      LTI 3-5   → medium
      LTI > 5   → high
    """
    if annual_income <= 0:
        return {"risk_level": "high", "lti_ratio": 999, "explanation": "Zero income"}

    lti = loan_amount / annual_income

    # Assets partially offset LTI risk — each dollar of assets covers 0.2 of loan
    effective_loan = max(loan_amount - assets_value * 0.2, loan_amount * 0.5)
    adjusted_lti = effective_loan / annual_income

    if adjusted_lti < 3:
        risk_level = "low"
        explanation = f"Loan-to-income ratio of {lti:.1f}x is conservative."
    elif adjusted_lti < 5:
        risk_level = "medium"
        explanation = f"Loan-to-income ratio of {lti:.1f}x is moderate."
    else:
        risk_level = "high"
        explanation = (
            f"Loan-to-income ratio of {lti:.1f}x is high; repayment capacity is strained."
        )

    return {
        "risk_level": risk_level,
        "lti_ratio": round(lti, 2),
        "adjusted_lti": round(adjusted_lti, 2),
        "explanation": explanation,
    }


def _detect_anomalies(application: Dict[str, Any]) -> List[str]:
    """
    Heuristic anomaly detection rules.

    These rules flag patterns that are statistically unusual or indicative
    of data manipulation.  Each anomaly is a string describing the concern.
    """
    anomalies: List[str] = []

    income = application.get("annual_income", 0)
    debt = application.get("monthly_debt", 0)
    loan = application.get("loan_amount", 0)
    score = application.get("credit_score", 700)
    loans = application.get("existing_loans", 0)
    years = application.get("years_employed", 0)
    assets = application.get("assets_value", 0)

    # Rule 1: Very high income with very low credit score — unusual combination
    if income > 150_000 and score < 600:
        anomalies.append(
            "High income (>$150k) paired with low credit score (<600) — inconsistent profile"
        )

    # Rule 2: Loan amount much larger than income — possible straw-buyer pattern
    if income > 0 and loan / income > 10:
        anomalies.append(
            f"Loan amount is {loan/income:.0f}x annual income — extremely high ratio"
        )

    # Rule 3: Zero debt with many existing loans — may indicate undisclosed obligations
    if loans >= 3 and debt < 100:
        anomalies.append(
            f"Has {loans} existing loans but reports < $100/month debt — likely incomplete"
        )

    # Rule 4: Declared assets far exceed income with no explanation
    if assets > income * 20 and income < 50_000:
        anomalies.append(
            "Asset value is >20x annual income for a low-income applicant — verify source"
        )

    # Rule 5: Years employed is implausibly long for younger applicant proxy
    # (We don't have age, but > 45 years is typically impossible in a working career)
    if years > 45:
        anomalies.append(f"years_employed of {years} is implausibly high")

    # Rule 6: Perfect credit score with high debt load
    if score >= 800 and debt / (income / 12) > 0.45:
        anomalies.append(
            "Perfect credit score alongside very high DTI is statistically unusual"
        )

    return anomalies


# ─── Tool dispatcher ───────────────────────────────────────────────────────────


@server.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> List[types.TextContent]:
    """Dispatch incoming tool requests to implementation functions."""
    logger.info("Tool called: %s", name)

    if name == "calculate_dti_ratio":
        result = _calculate_dti_ratio(
            annual_income=arguments["annual_income"],
            monthly_debt=arguments["monthly_debt"],
        )
        return [types.TextContent(type="text", text=json.dumps(result))]

    elif name == "classify_credit_score_risk":
        result = _classify_credit_score_risk(credit_score=arguments["credit_score"])
        return [types.TextContent(type="text", text=json.dumps(result))]

    elif name == "evaluate_loan_amount_risk":
        result = _evaluate_loan_amount_risk(
            loan_amount=arguments["loan_amount"],
            annual_income=arguments["annual_income"],
            assets_value=arguments.get("assets_value", 0.0),
        )
        return [types.TextContent(type="text", text=json.dumps(result))]

    elif name == "detect_anomalies":
        result = _detect_anomalies(application=arguments["application"])
        return [types.TextContent(type="text", text=json.dumps({"anomalies": result}))]

    else:
        raise ValueError(f"Unknown tool: {name}")


# ─── Entry point ───────────────────────────────────────────────────────────────


async def main() -> None:
    """Launch the RiskRulesDB MCP server over stdio transport."""
    async with stdio_server.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="RiskRulesDB",
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