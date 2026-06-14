"""
Loan Decision Agent

Synthesises profile and risk outputs into a final loan decision using tools
from the DecisionSynthesis MCP server.  Outputs:
  - Decision: approved | rejected | manual_review
  - Risk Score: 0-100 composite score
  - Confidence Level: 0-100%
  - Key Decision Factors (ordered list)
  - Human-readable audit explanation
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
from models.schemas import (
    ApplicantProfile,
    FinancialRiskAssessment,
    LoanDecision,
    LoanDecisionOutput,
)

logger = logging.getLogger(__name__)

CLAUDE_MODEL = settings.claude_model
MCP_SERVER_PATH = str(_PROJECT_ROOT / "mcp" / "decision_synthesis_server.py")

_SYSTEM_PROMPT = """You are the Loan Decision Agent for an automated loan approval system.

You receive outputs from the Applicant Profile Agent and Financial Risk Agent,
and must produce a final, well-reasoned loan decision.

Call tools in this order:
1. compute_risk_score — calculate the composite risk score from all inputs
2. calculate_confidence — determine model confidence
3. determine_decision — apply threshold rules to get approve/reject/review
4. generate_explanation — produce the human-readable audit explanation

Respond with ONLY a valid JSON object:
{
  "decision": <"approved"|"rejected"|"manual_review">,
  "risk_score": <float 0-100>,
  "confidence_level": <float 0-100>,
  "key_factors": [<string>, ...],
  "explanation": <string>
}
"""


async def _run_agent_loop(
    session: ClientSession,
    client: anthropic.AsyncAnthropic,
    application: Dict[str, Any],
    profile: Dict[str, Any],
    risk_assessment: Dict[str, Any],
) -> str:
    """Agentic loop that passes all upstream context to Claude."""
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
        "applicant_profile": profile,
        "financial_risk_assessment": risk_assessment,
    }

    messages: List[Dict[str, Any]] = [
        {
            "role": "user",
            "content": (
                "Based on the following data, make a final loan decision:\n\n"
                f"{json.dumps(context, indent=2)}"
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


async def decide(
    application: Dict[str, Any],
    profile: ApplicantProfile,
    risk_assessment: FinancialRiskAssessment,
) -> LoanDecisionOutput:
    """
    Entry point for the Loan Decision Agent.

    Args:
        application:     Original loan application dict.
        profile:         Output of the Applicant Profile Agent.
        risk_assessment: Output of the Financial Risk Analysis Agent.

    Returns:
        LoanDecisionOutput with decision, scores, factors, and explanation.
    """
    logger.info("Loan Decision Agent started for: %s", application.get("applicant_name"))

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
            raw = await _run_agent_loop(
                session, client,
                application,
                profile.model_dump(),
                risk_assessment.model_dump(),
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
                output = LoanDecisionOutput(
                    decision=LoanDecision(data.get("decision", "manual_review")),
                    risk_score=data.get("risk_score", 50.0),
                    confidence_level=data.get("confidence_level", 50.0),
                    key_factors=data.get("key_factors", []),
                    explanation=data.get("explanation", "No explanation generated"),
                )
                logger.info(
                    "Decision: %s | Risk: %.1f | Confidence: %.1f%%",
                    output.decision,
                    output.risk_score,
                    output.confidence_level,
                )
                return output
            except (json.JSONDecodeError, ValueError) as exc:
                logger.error("Parse error in Decision Agent: %s", exc)
                return LoanDecisionOutput(
                    decision=LoanDecision.MANUAL_REVIEW,
                    risk_score=50.0,
                    confidence_level=0.0,
                    key_factors=["Decision engine error"],
                    explanation=f"Decision agent failed: {exc}. Sent to manual review.",
                )
