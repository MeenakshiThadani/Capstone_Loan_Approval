"""
Compliance & Action Orchestrator Agent

The final agent in the pipeline.  Uses the NotificationSystem MCP server to:
  1. Generate a unique Case ID
  2. Record the action to the audit log
  3. Send a mock notification to the applicant (email / SMS)
  4. Produce a complete, immutable audit trail entry
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import anthropic
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from agents.utils import create_message_with_retry

_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from config.settings import settings
from models.schemas import (
    ApplicantProfile,
    ComplianceRecord,
    FinancialRiskAssessment,
    LoanDecisionOutput,
)

logger = logging.getLogger(__name__)

CLAUDE_MODEL = settings.claude_model
MCP_SERVER_PATH = str(_PROJECT_ROOT / "mcp" / "notification_system_server.py")

_SYSTEM_PROMPT = """You are the Compliance & Action Orchestrator Agent for a loan approval system.

Finalise the loan case by calling ALL four tools:
1. generate_case_id — create a unique case identifier
2. record_action — persist the decision to the audit log
3. send_notification — dispatch the applicant notification (use "email" type)
4. create_audit_record — produce the full audit trail entry

Respond with ONLY a valid JSON object:
{
  "case_id": <string>,
  "action_taken": <string>,
  "notification_sent": <true|false>,
  "notification_type": <"email"|"sms"|"both"|"none">,
  "timestamp": <ISO 8601 UTC string>,
  "audit_summary": <audit summary text from create_audit_record>
}
"""


async def _run_agent_loop(
    session: ClientSession,
    client: anthropic.AsyncAnthropic,
    application: Dict[str, Any],
    profile: Dict[str, Any],
    risk_assessment: Dict[str, Any],
    decision_output: Dict[str, Any],
) -> str:
    """Agentic loop that passes all upstream outputs to Claude."""
    tools_response = await session.list_tools()
    tools = [
        {
            "name": t.name,
            "description": t.description or "",
            "input_schema": t.inputSchema,
        }
        for t in tools_response.tools
    ]

    context = {
        "application": application,
        "profile": profile,
        "risk_assessment": risk_assessment,
        "decision_output": decision_output,
    }

    messages: List[Dict[str, Any]] = [
        {
            "role": "user",
            "content": (
                "Finalise this loan case — record the decision, send the notification, "
                f"and create the audit record:\n\n{json.dumps(context, indent=2)}"
            ),
        }
    ]

    while True:
        response = await create_message_with_retry(
            client,
            model=CLAUDE_MODEL,
            max_tokens=4096,
            system=_SYSTEM_PROMPT,
            tools=tools,
            messages=messages,
        )

        if response.stop_reason == "end_turn":
            for block in response.content:
                if hasattr(block, "text"):
                    return block.text
            return "{}"

        if response.stop_reason != "tool_use":
            for block in response.content:
                if hasattr(block, "text"):
                    return block.text
            return "{}"

        tool_results: List[Dict[str, Any]] = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            logger.info("Calling MCP tool: %s(%s)", block.name, block.input)
            mcp_result = await session.call_tool(block.name, block.input)
            result_text = "\n".join(
                item.text for item in mcp_result.content if hasattr(item, "text")
            )
            tool_results.append(
                {"type": "tool_result", "tool_use_id": block.id, "content": result_text}
            )

        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": tool_results})


async def record_and_notify(
    application: Dict[str, Any],
    profile: ApplicantProfile,
    risk_assessment: FinancialRiskAssessment,
    decision_output: LoanDecisionOutput,
    case_id: str | None = None,
) -> ComplianceRecord:
    """
    Entry point for the Compliance & Action Orchestrator Agent.

    Args:
        application:     Original loan application dict.
        profile:         Applicant Profile Agent output.
        risk_assessment: Financial Risk Agent output.
        decision_output: Loan Decision Agent output.
        case_id:         Pre-generated case ID (optional).

    Returns:
        ComplianceRecord with case_id, notification status, and audit summary.
    """
    logger.info(
        "Compliance Agent started for: %s | decision: %s",
        application.get("applicant_name"),
        decision_output.decision,
    )

    server_params = StdioServerParameters(
        command=sys.executable,
        args=[MCP_SERVER_PATH],
        env={**os.environ, "PYTHONPATH": str(_PROJECT_ROOT)},
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            client = anthropic.AsyncAnthropic(
                api_key=settings.anthropic_api_key.get_secret_value(),
                base_url=settings.anthropic_base_url or None,
            )

            app_ctx = dict(application)
            if case_id:
                app_ctx["existing_case_id"] = case_id

            raw = await _run_agent_loop(
                session, client,
                app_ctx,
                profile.model_dump(),
                risk_assessment.model_dump(),
                decision_output.model_dump(),
            )

            try:
                import re as _re
                _m = _re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, _re.DOTALL)
                if _m:
                    clean = _m.group(1)
                else:
                    _m2 = _re.search(r"\{.*\}", raw, _re.DOTALL)
                    clean = _m2.group(0) if _m2 else raw.strip()
                data = json.loads(clean)

                raw_ts = data.get("timestamp")
                try:
                    ts = datetime.fromisoformat(raw_ts) if raw_ts else datetime.now(timezone.utc)
                except ValueError:
                    ts = datetime.now(timezone.utc)

                record = ComplianceRecord(
                    case_id=data.get("case_id", case_id or "UNKNOWN"),
                    action_taken=data.get("action_taken", "Decision recorded"),
                    notification_sent=data.get("notification_sent", False),
                    notification_type=data.get("notification_type"),
                    timestamp=ts,
                    audit_summary=data.get("audit_summary", "No audit summary"),
                )
                logger.info(
                    "Compliance record created — case_id: %s, notified: %s",
                    record.case_id,
                    record.notification_sent,
                )
                return record
            except (json.JSONDecodeError, ValueError) as exc:
                logger.error("Parse error in Compliance Agent: %s", exc)
                return ComplianceRecord(
                    case_id=case_id or "ERROR",
                    action_taken="Compliance recording failed",
                    notification_sent=False,
                    notification_type=None,
                    timestamp=datetime.now(timezone.utc),
                    audit_summary=f"Compliance agent error: {exc}",
                )
