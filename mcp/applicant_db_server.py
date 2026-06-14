"""
MCP Server: ApplicantDB

Exposes tools consumed exclusively by the Applicant Profile Agent.
Responsibilities modelled here:
  - Income stability scoring
  - Employment risk classification
  - Credit history summarisation
  - Application completeness checking

Run standalone:
    python mcp/applicant_db_server.py
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any, Dict, List

# MCP server primitives (mcp >= 1.0)
from mcp.server import NotificationOptions, Server
from mcp.server.models import InitializationOptions
import mcp.server.stdio as stdio_server
import mcp.types as types

logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger("applicant_db_server")

# ── Server instance ────────────────────────────────────────────────────────────
server = Server("ApplicantDB")


# ─── Tool definitions ─────────────────────────────────────────────────────────


@server.list_tools()
async def list_tools() -> List[types.Tool]:
    """Advertise all tools this MCP server exposes to agents."""
    return [
        types.Tool(
            name="calculate_income_stability",
            description=(
                "Calculates an income stability score (0-100) based on the "
                "applicant's annual income, employment type, and tenure. "
                "Higher scores indicate more predictable, reliable income."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "annual_income": {
                        "type": "number",
                        "description": "Gross annual income in USD",
                    },
                    "employment_type": {
                        "type": "string",
                        "enum": [
                            "full_time",
                            "part_time",
                            "contract",
                            "self_employed",
                            "unemployed",
                        ],
                    },
                    "years_employed": {
                        "type": "number",
                        "description": "Years at current employer",
                    },
                },
                "required": ["annual_income", "employment_type", "years_employed"],
            },
        ),
        types.Tool(
            name="assess_employment_risk",
            description=(
                "Classifies employment risk as low/medium/high based on "
                "employment type and tenure.  Contract and self-employed "
                "applicants require higher tenure to achieve low risk."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "employment_type": {"type": "string"},
                    "years_employed": {"type": "number"},
                },
                "required": ["employment_type", "years_employed"],
            },
        ),
        types.Tool(
            name="get_credit_history_summary",
            description=(
                "Returns a human-readable summary of the applicant's credit "
                "profile, derived from FICO score bands and existing loan count."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "credit_score": {
                        "type": "integer",
                        "description": "FICO credit score 300-850",
                    },
                    "existing_loans": {
                        "type": "integer",
                        "description": "Number of open loan accounts",
                    },
                },
                "required": ["credit_score", "existing_loans"],
            },
        ),
        types.Tool(
            name="check_application_completeness",
            description=(
                "Inspects the application for missing, suspicious, or "
                "inconsistent fields.  Returns a list of flag strings; "
                "an empty list means the application is complete."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "application": {
                        "type": "object",
                        "description": "Full loan application as a JSON object",
                    }
                },
                "required": ["application"],
            },
        ),
    ]


# ─── Tool implementations ──────────────────────────────────────────────────────


def _calculate_income_stability(
    annual_income: float, employment_type: str, years_employed: float
) -> float:
    """
    Income stability score algorithm (0–100).

    Base score is derived from employment type reliability:
      full_time    → 70 base
      part_time    → 50 base
      contract     → 45 base
      self_employed→ 40 base
      unemployed   → 0  (immediate zero)

    Tenure bonus: +2 pts per year, capped at +20.
    Income tier bonus: +10 for income > $80k, +5 for > $40k.
    """
    if employment_type == "unemployed":
        return 0.0

    base_scores = {
        "full_time": 70,
        "part_time": 50,
        "contract": 45,
        "self_employed": 40,
    }
    score = float(base_scores.get(employment_type, 40))

    # Tenure bonus — longer tenure signals stability
    tenure_bonus = min(years_employed * 2, 20)
    score += tenure_bonus

    # Income tier bonus — higher earners have more buffer
    if annual_income > 80_000:
        score += 10
    elif annual_income > 40_000:
        score += 5

    return round(min(score, 100.0), 2)


def _assess_employment_risk(employment_type: str, years_employed: float) -> str:
    """
    Employment risk classification logic.

    Risk thresholds by employment type (years required for each tier):
      full_time:    <1 yr → high, 1-3 → medium, ≥3 → low
      part_time:    <2 yr → high, 2-5 → medium, ≥5 → low
      contract:     <3 yr → high, 3-6 → medium, ≥6 → low
      self_employed:<5 yr → high, 5-8 → medium, ≥8 → low
      unemployed:   always high
    """
    thresholds: Dict[str, tuple] = {
        # (medium threshold, low threshold)
        "full_time": (1, 3),
        "part_time": (2, 5),
        "contract": (3, 6),
        "self_employed": (5, 8),
    }

    if employment_type == "unemployed":
        return "high"

    medium_t, low_t = thresholds.get(employment_type, (3, 6))

    if years_employed >= low_t:
        return "low"
    elif years_employed >= medium_t:
        return "medium"
    else:
        return "high"


def _get_credit_history_summary(credit_score: int, existing_loans: int) -> str:
    """
    Maps FICO score bands and loan count to a human-readable summary.

    FICO bands:
      800-850 → Exceptional
      740-799 → Very Good
      670-739 → Good
      580-669 → Fair
      300-579 → Poor
    """
    if credit_score >= 800:
        band = "Exceptional"
        narrative = "Demonstrates outstanding credit management with minimal risk."
    elif credit_score >= 740:
        band = "Very Good"
        narrative = "Strong credit history with very few derogatory marks."
    elif credit_score >= 670:
        band = "Good"
        narrative = "Solid credit history; eligible for most standard products."
    elif credit_score >= 580:
        band = "Fair"
        narrative = "Some negative history present; higher interest rates likely."
    else:
        band = "Poor"
        narrative = "Significant derogatory history; high default risk."

    loan_note = (
        f"Has {existing_loans} existing open loan account(s)."
        if existing_loans > 0
        else "No existing open loans."
    )

    return f"[{band} — {credit_score}] {narrative} {loan_note}"


def _check_application_completeness(application: Dict[str, Any]) -> List[str]:
    """
    Scans the application for data quality issues.

    Returns a list of flag strings.  Each flag describes a specific concern
    so that the agent can include it in the risk narrative.
    """
    flags: List[str] = []

    # Required fields that must be present and truthy
    required = [
        "applicant_name",
        "annual_income",
        "monthly_debt",
        "loan_amount",
        "credit_score",
        "employment_type",
        "years_employed",
        "loan_purpose",
    ]
    for field in required:
        if field not in application or application[field] is None:
            flags.append(f"Missing required field: {field}")

    # Business rule: unemployed applicants with positive years_employed is inconsistent
    if application.get("employment_type") == "unemployed" and application.get("years_employed", 0) > 0:
        flags.append("Inconsistency: unemployed status but positive years_employed")

    # Suspiciously round loan amounts may indicate estimation rather than real need
    loan_amount = application.get("loan_amount", 0)
    if isinstance(loan_amount, (int, float)) and loan_amount > 0:
        if loan_amount % 100_000 == 0 and loan_amount > 500_000:
            flags.append("Loan amount is a very round number — verify accuracy")

    # Zero monthly debt with existing loans is suspicious
    if application.get("existing_loans", 0) > 0 and application.get("monthly_debt", 0) == 0:
        flags.append("Has existing loans but reports zero monthly debt — verify")

    return flags


# ─── Tool dispatcher ───────────────────────────────────────────────────────────


@server.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> List[types.TextContent]:
    """
    Routes incoming tool calls to the appropriate implementation function.

    Returns results wrapped in TextContent so the MCP protocol can
    transport them back to the calling agent.
    """
    logger.info("Tool called: %s with args: %s", name, arguments)

    if name == "calculate_income_stability":
        result = _calculate_income_stability(
            annual_income=arguments["annual_income"],
            employment_type=arguments["employment_type"],
            years_employed=arguments["years_employed"],
        )
        return [types.TextContent(type="text", text=json.dumps({"income_stability_score": result}))]

    elif name == "assess_employment_risk":
        result = _assess_employment_risk(
            employment_type=arguments["employment_type"],
            years_employed=arguments["years_employed"],
        )
        return [types.TextContent(type="text", text=json.dumps({"employment_risk": result}))]

    elif name == "get_credit_history_summary":
        result = _get_credit_history_summary(
            credit_score=arguments["credit_score"],
            existing_loans=arguments.get("existing_loans", 0),
        )
        return [types.TextContent(type="text", text=json.dumps({"credit_history_summary": result}))]

    elif name == "check_application_completeness":
        result = _check_application_completeness(application=arguments["application"])
        return [types.TextContent(type="text", text=json.dumps({"completeness_flags": result}))]

    else:
        raise ValueError(f"Unknown tool: {name}")


# ─── Entry point ───────────────────────────────────────────────────────────────


async def main() -> None:
    """Launch the ApplicantDB MCP server over stdio transport."""
    async with stdio_server.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="ApplicantDB",
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
