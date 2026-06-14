"""
MCP Server: NotificationSystem

Exposes tools consumed exclusively by the Compliance & Action Orchestrator Agent.
Responsibilities modelled here:
  - Unique case ID generation
  - Action recording (persists to an in-memory log — swap for DB in production)
  - Notification dispatch mock (email / SMS)
  - Audit trail creation

Run standalone:
    python mcp/notification_system_server.py
"""

from __future__ import annotations

import json
import logging
import sys
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List

from mcp.server import NotificationOptions, Server
from mcp.server.models import InitializationOptions
import mcp.server.stdio as stdio_server
import mcp.types as types

logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger("notification_system_server")

server = Server("NotificationSystem")

# ── In-memory action log (replace with persistent storage in production) ───────
_action_log: List[Dict[str, Any]] = []


# ─── Tool definitions ──────────────────────────────────────────────────────────


@server.list_tools()
async def list_tools() -> List[types.Tool]:
    """Advertise all compliance and notification tools."""
    return [
        types.Tool(
            name="generate_case_id",
            description=(
                "Generates a globally unique case identifier for this loan "
                "application.  Format: LOAN-<YYYYMMDD>-<8 hex chars>."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "applicant_name": {
                        "type": "string",
                        "description": "Used for logging only; not part of ID",
                    }
                },
                "required": [],
            },
        ),
        types.Tool(
            name="record_action",
            description=(
                "Persists a record of the action taken on this case to the "
                "audit log.  Returns True on success."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "case_id": {"type": "string"},
                    "action": {
                        "type": "string",
                        "description": "e.g. 'loan_approved', 'loan_rejected', 'sent_to_review'",
                    },
                    "decision": {
                        "type": "string",
                        "enum": ["approved", "rejected", "manual_review"],
                    },
                    "risk_score": {"type": "number"},
                    "confidence_level": {"type": "number"},
                },
                "required": ["case_id", "action", "decision"],
            },
        ),
        types.Tool(
            name="send_notification",
            description=(
                "Mocks sending an email and/or SMS notification to the applicant. "
                "In production, replace the mock bodies with calls to SendGrid / Twilio."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "case_id": {"type": "string"},
                    "applicant_name": {"type": "string"},
                    "decision": {
                        "type": "string",
                        "enum": ["approved", "rejected", "manual_review"],
                    },
                    "notification_type": {
                        "type": "string",
                        "enum": ["email", "sms", "both", "none"],
                        "default": "email",
                    },
                    "contact_email": {
                        "type": "string",
                        "description": "Applicant email address (optional)",
                        "default": "",
                    },
                },
                "required": ["case_id", "applicant_name", "decision"],
            },
        ),
        types.Tool(
            name="create_audit_record",
            description=(
                "Assembles a complete audit trail entry that includes all "
                "decision inputs, outputs, timestamps, and agent chain. "
                "Returns a formatted audit summary string."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "case_id": {"type": "string"},
                    "application": {
                        "type": "object",
                        "description": "Original loan application",
                    },
                    "profile": {
                        "type": "object",
                        "description": "Applicant Profile Agent output",
                    },
                    "risk_assessment": {
                        "type": "object",
                        "description": "Financial Risk Agent output",
                    },
                    "decision_output": {
                        "type": "object",
                        "description": "Loan Decision Agent output",
                    },
                },
                "required": [
                    "case_id",
                    "application",
                    "profile",
                    "risk_assessment",
                    "decision_output",
                ],
            },
        ),
    ]


# ─── Tool implementations ──────────────────────────────────────────────────────


def _generate_case_id(applicant_name: str = "") -> str:
    """
    Case ID format: LOAN-YYYYMMDD-XXXXXXXX

    The date prefix aids human identification; the UUID suffix guarantees
    global uniqueness even under concurrent submissions.
    """
    today = datetime.now(timezone.utc).strftime("%Y%m%d")
    unique_suffix = uuid.uuid4().hex[:8].upper()
    case_id = f"LOAN-{today}-{unique_suffix}"
    logger.info("Generated case_id: %s for applicant: %s", case_id, applicant_name)
    return case_id


def _record_action(
    case_id: str,
    action: str,
    decision: str,
    risk_score: float = 0.0,
    confidence_level: float = 0.0,
) -> bool:
    """
    Appends an action record to the in-memory audit log.

    In production this would write to a relational DB or audit table
    with appropriate ACID guarantees.
    """
    record = {
        "case_id": case_id,
        "action": action,
        "decision": decision,
        "risk_score": risk_score,
        "confidence_level": confidence_level,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "agent": "ComplianceAgent",
    }
    _action_log.append(record)
    logger.info("Action recorded: %s", json.dumps(record))
    return True


def _send_notification(
    case_id: str,
    applicant_name: str,
    decision: str,
    notification_type: str = "email",
    contact_email: str = "",
) -> Dict[str, Any]:
    """
    Mocks sending notifications.  Logs what would be sent in production.

    Returns a record of which channels were triggered and their mock status.
    """
    decision_messages = {
        "approved": (
            f"Congratulations {applicant_name}! Your loan application "
            f"(Case ID: {case_id}) has been approved. "
            "Please log in to your account to complete the next steps."
        ),
        "rejected": (
            f"Dear {applicant_name}, after careful review your loan application "
            f"(Case ID: {case_id}) has not been approved at this time. "
            "You may request a full explanation of the decision."
        ),
        "manual_review": (
            f"Dear {applicant_name}, your loan application "
            f"(Case ID: {case_id}) is under additional review. "
            "You will receive a final decision within 2 business days."
        ),
    }

    message = decision_messages.get(
        decision,
        f"Your application {case_id} has been processed.",
    )

    channels_sent: List[str] = []

    if notification_type in ("email", "both"):
        target = contact_email or f"{applicant_name.replace(' ', '.').lower()}@example.com"
        logger.info("[MOCK EMAIL] To: %s | Subject: Loan Decision | Body: %s", target, message)
        channels_sent.append("email")

    if notification_type in ("sms", "both"):
        logger.info("[MOCK SMS] To: applicant | Message: %s", message[:160])
        channels_sent.append("sms")

    return {
        "notification_sent": len(channels_sent) > 0,
        "channels": channels_sent,
        "message_preview": message[:200],
    }


def _create_audit_record(
    case_id: str,
    application: Dict[str, Any],
    profile: Dict[str, Any],
    risk_assessment: Dict[str, Any],
    decision_output: Dict[str, Any],
) -> str:
    """
    Assembles a structured audit trail entry.

    The audit summary is a human-readable but machine-parseable record
    of every agent's output.  It should be immutable once written.
    """
    timestamp = datetime.now(timezone.utc).isoformat()

    applicant_name = application.get("applicant_name", "Unknown")
    loan_amount = application.get("loan_amount", 0)
    decision = decision_output.get("decision", "unknown")
    risk_score = decision_output.get("risk_score", 0)
    confidence = decision_output.get("confidence_level", 0)

    summary_lines = [
        "=" * 70,
        f"LOAN APPLICATION AUDIT RECORD",
        f"Case ID   : {case_id}",
        f"Timestamp : {timestamp}",
        "=" * 70,
        f"Applicant : {applicant_name}",
        f"Amount    : ${loan_amount:,.2f}",
        f"Purpose   : {application.get('loan_purpose', 'N/A')}",
        "-" * 70,
        "DECISION SUMMARY",
        f"  Decision         : {decision.upper()}",
        f"  Risk Score       : {risk_score:.1f}/100",
        f"  Confidence       : {confidence:.1f}%",
        "-" * 70,
        "PROFILE AGENT OUTPUT",
        f"  Income Stability : {profile.get('income_stability_score', 'N/A')}",
        f"  Employment Risk  : {profile.get('employment_risk', 'N/A')}",
        f"  Completeness Flags: {profile.get('completeness_flags', [])}",
        "-" * 70,
        "RISK ASSESSMENT OUTPUT",
        f"  DTI Ratio        : {risk_assessment.get('debt_to_income_ratio', 'N/A')}",
        f"  Credit Risk      : {risk_assessment.get('credit_risk_level', 'N/A')}",
        f"  Anomalies        : {risk_assessment.get('anomalies', [])}",
        "-" * 70,
        "KEY DECISION FACTORS",
    ]

    for factor in decision_output.get("key_factors", []):
        summary_lines.append(f"  • {factor}")

    summary_lines += [
        "=" * 70,
        f"System: LoanApprovalAgent v1.0 | Model: claude-sonnet-4-6",
        "=" * 70,
    ]

    return "\n".join(summary_lines)


# ─── Tool dispatcher ───────────────────────────────────────────────────────────


@server.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> List[types.TextContent]:
    """Dispatch incoming tool requests to implementation functions."""
    logger.info("Tool called: %s", name)

    if name == "generate_case_id":
        result = _generate_case_id(applicant_name=arguments.get("applicant_name", ""))
        return [types.TextContent(type="text", text=json.dumps({"case_id": result}))]

    elif name == "record_action":
        result = _record_action(
            case_id=arguments["case_id"],
            action=arguments["action"],
            decision=arguments["decision"],
            risk_score=arguments.get("risk_score", 0.0),
            confidence_level=arguments.get("confidence_level", 0.0),
        )
        return [types.TextContent(type="text", text=json.dumps({"success": result}))]

    elif name == "send_notification":
        result = _send_notification(
            case_id=arguments["case_id"],
            applicant_name=arguments["applicant_name"],
            decision=arguments["decision"],
            notification_type=arguments.get("notification_type", "email"),
            contact_email=arguments.get("contact_email", ""),
        )
        return [types.TextContent(type="text", text=json.dumps(result))]

    elif name == "create_audit_record":
        result = _create_audit_record(
            case_id=arguments["case_id"],
            application=arguments["application"],
            profile=arguments["profile"],
            risk_assessment=arguments["risk_assessment"],
            decision_output=arguments["decision_output"],
        )
        return [types.TextContent(type="text", text=json.dumps({"audit_summary": result}))]

    else:
        raise ValueError(f"Unknown tool: {name}")


# ─── Entry point ───────────────────────────────────────────────────────────────


async def main() -> None:
    """Launch the NotificationSystem MCP server over stdio transport."""
    async with stdio_server.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="NotificationSystem",
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
