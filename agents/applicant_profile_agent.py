"""
Applicant Profile Agent

Builds a structured profile of the loan applicant using tools from the
ApplicantDB MCP server.  Responsibilities:
  1. Calculate Income Stability Score (0-100)
  2. Assess Employment Risk (low / medium / high)
  3. Summarise Credit History
  4. Flag Application Completeness issues
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
from models.schemas import ApplicantProfile, RiskLevel

logger = logging.getLogger(__name__)

CLAUDE_MODEL = settings.claude_model
MCP_SERVER_PATH = str(_PROJECT_ROOT / "mcp" / "applicant_db_server.py")

_SYSTEM_PROMPT = """You are the Applicant Profile Agent for a loan approval system.

Your job is to analyse the loan application and produce a structured applicant
profile using the tools available to you.

Call tools in this order:
1. check_application_completeness — identify missing or suspicious fields
2. calculate_income_stability — compute the income stability score
3. assess_employment_risk — classify employment risk
4. get_credit_history_summary — summarise the credit profile

Respond with ONLY a valid JSON object:
{
  "income_stability_score": <float 0-100>,
  "employment_risk": <"low"|"medium"|"high">,
  "credit_history_summary": <string>,
  "completeness_flags": [<string>, ...]
}
"""


async def _run_agent_loop(
    session: ClientSession,
    client: anthropic.AsyncAnthropic,
    application: Dict[str, Any],
) -> str:
    """
    Core agentic loop: sends prompt to Claude, handles tool-use responses,
    and iterates until Claude produces a final text response.
    """
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
                f"Analyse this loan application and produce a complete "
                f"applicant profile:\n\n{json.dumps(application, indent=2)}"
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

        # Forward each tool call to the MCP server and collect results
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


async def analyze(application: Dict[str, Any]) -> ApplicantProfile:
    """
    Entry point for the Applicant Profile Agent.

    Launches the ApplicantDB MCP server as a subprocess, connects via stdio,
    and runs the agentic loop with Claude.

    Args:
        application: Loan application data as a plain dict.

    Returns:
        ApplicantProfile with all fields populated.
    """
    logger.info("Applicant Profile Agent started for: %s", application.get("applicant_name"))

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
                profile = ApplicantProfile(
                    income_stability_score=data.get("income_stability_score", 50.0),
                    employment_risk=RiskLevel(data.get("employment_risk", "medium")),
                    credit_history_summary=data.get("credit_history_summary", ""),
                    completeness_flags=data.get("completeness_flags", []),
                )
                logger.info(
                    "Profile complete — stability: %.1f, risk: %s",
                    profile.income_stability_score,
                    profile.employment_risk,
                )
                return profile
            except (json.JSONDecodeError, ValueError) as exc:
                logger.error("Parse error in Profile Agent: %s", exc)
                return ApplicantProfile(
                    income_stability_score=50.0,
                    employment_risk=RiskLevel.MEDIUM,
                    credit_history_summary="Unable to determine",
                    completeness_flags=[f"Agent parse error: {exc}"],
                )
