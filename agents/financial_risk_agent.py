"""
Financial Risk Analysis Agent

Performs quantitative risk analysis using tools from the RiskRulesDB MCP server.
Produces:
  1. Debt-to-Income (DTI) Ratio with risk classification
  2. Credit Score Risk Level (low / medium / high)
  3. Loan Amount Risk relative to income
  4. Anomaly Detection
  5. Structured risk reasoning narrative
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import anthropic
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from agents.utils import create_message_with_retry

_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from config.settings import settings
from models.schemas import FinancialRiskAssessment, RiskLevel

logger = logging.getLogger(__name__)

CLAUDE_MODEL = settings.claude_model
MCP_SERVER_PATH = str(_PROJECT_ROOT / "mcp" / "risk_rules_db_server.py")

_SYSTEM_PROMPT = """You are the Financial Risk Analysis Agent for a loan approval system.

Perform a thorough quantitative risk analysis using ALL four tools:
1. calculate_dti_ratio — compute the debt-to-income ratio
2. classify_credit_score_risk — classify the FICO score
3. evaluate_loan_amount_risk — assess loan amount vs income
4. detect_anomalies — check for unusual patterns

Respond with ONLY a valid JSON object:
{
  "debt_to_income_ratio": <float>,
  "credit_risk_level": <"low"|"medium"|"high">,
  "loan_amount_risk": <"low"|"medium"|"high">,
  "anomalies": [<string>, ...],
  "risk_reasoning": <paragraph summarising overall risk posture>
}
"""


async def _run_agent_loop(
    session: ClientSession,
    client: anthropic.AsyncAnthropic,
    application: Dict[str, Any],
) -> str:
    """Agentic loop for the Financial Risk Agent."""
    tools_response = await session.list_tools()
    tools = [
        {
            "name": t.name,
            "description": t.description or "",
            "input_schema": t.inputSchema,
        }
        for t in tools_response.tools
    ]

    messages: List[Dict[str, Any]] = [
        {
            "role": "user",
            "content": (
                f"Perform a complete financial risk analysis on this loan "
                f"application:\n\n{json.dumps(application, indent=2)}"
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


async def analyze(application: Dict[str, Any]) -> FinancialRiskAssessment:
    """
    Entry point for the Financial Risk Analysis Agent.

    Args:
        application: Loan application data as a plain dict.

    Returns:
        FinancialRiskAssessment with DTI, credit risk, loan risk, and anomalies.
    """
    logger.info("Financial Risk Agent started for: %s", application.get("applicant_name"))

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
            raw = await _run_agent_loop(session, client, application)

            try:
                import re as _re
                _m = _re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, _re.DOTALL)
                if _m:
                    clean = _m.group(1)
                else:
                    _m2 = _re.search(r"\{.*\}", raw, _re.DOTALL)
                    clean = _m2.group(0) if _m2 else raw.strip()
                data = json.loads(clean)
                assessment = FinancialRiskAssessment(
                    debt_to_income_ratio=data.get("debt_to_income_ratio", 0.0),
                    credit_risk_level=RiskLevel(data.get("credit_risk_level", "medium")),
                    loan_amount_risk=RiskLevel(data.get("loan_amount_risk", "medium")),
                    anomalies=data.get("anomalies", []),
                    risk_reasoning=data.get("risk_reasoning", "Unable to generate reasoning"),
                )
                logger.info(
                    "Risk assessment complete — DTI: %.2f, credit risk: %s",
                    assessment.debt_to_income_ratio,
                    assessment.credit_risk_level,
                )
                return assessment
            except (json.JSONDecodeError, ValueError) as exc:
                logger.error("Parse error in Risk Agent: %s", exc)
                return FinancialRiskAssessment(
                    debt_to_income_ratio=0.0,
                    credit_risk_level=RiskLevel.HIGH,
                    loan_amount_risk=RiskLevel.HIGH,
                    anomalies=[f"Agent parse error: {exc}"],
                    risk_reasoning="Risk assessment failed; defaulting to high risk",
                )
