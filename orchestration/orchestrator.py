"""
LangGraph Orchestration Engine

Coordinates the four domain agents in a directed acyclic graph:

  [parallel_analysis] → [decision_node] → [compliance_node] → END
        ├── profile_agent ┐  (run concurrently via asyncio.gather)
        └── risk_agent   ─┘

Profile Agent and Financial Risk Agent run concurrently because neither
depends on the other's output.  The Decision Agent waits for both before
synthesising a final decision.

Conditional edges route the pipeline to END early on hard errors.

State transitions:
  parallel_analysis → (either failed?) → END
                    → (both ok)        → decision_node
  decision_node     → always           → compliance_node
  compliance_node   → always           → END
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional, TypedDict

_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import anthropic
from langgraph.graph import StateGraph, END

from agents.utils import create_message_with_retry
from config.settings import settings

from models.schemas import (
    ApplicantProfile,
    ComplianceRecord,
    FinancialRiskAssessment,
    LoanApplicationRequest,
    LoanDecisionOutput,
    LoanDecision,
)


# LangGraph requires a TypedDict (not a plain dict or Pydantic model) for state.
# Using total=False makes all keys optional so nodes can return partial updates
# that LangGraph merges into the accumulated state.
class LoanGraphState(TypedDict, total=False):
    application: Dict[str, Any]
    profile: Optional[Dict[str, Any]]
    risk_assessment: Optional[Dict[str, Any]]
    decision: Optional[Dict[str, Any]]
    compliance: Optional[Dict[str, Any]]
    case_id: Optional[str]
    error: Optional[str]
    status: str

logger = logging.getLogger(__name__)

CLAUDE_MODEL = settings.claude_model


# ─── LangGraph Node Functions ──────────────────────────────────────────────────


async def parallel_analysis_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Node 1: Profile Agent + Risk Agent (concurrent)

    Input state keys used:  application
    Output state keys set:  profile, risk_assessment, error (on failure)

    Fires both agents simultaneously with asyncio.gather().  If either agent
    raises, the node returns an error state and the pipeline short-circuits
    before reaching the Decision Agent.
    """
    logger.info(
        "==> parallel_analysis_node started — "
        "running Profile Agent and Risk Agent concurrently"
    )
    from agents.applicant_profile_agent import analyze as profile_analyze
    from agents.financial_risk_agent import analyze as risk_analyze

    application = state["application"]

    try:
        profile_result, risk_result = await asyncio.gather(
            profile_analyze(application),
            risk_analyze(application),
            return_exceptions=True,
        )

        if isinstance(profile_result, Exception):
            logger.error("Profile Agent failed: %s", profile_result, exc_info=profile_result)
            return {"error": f"Profile agent failed: {profile_result}", "status": "error"}

        if isinstance(risk_result, Exception):
            logger.error("Risk Agent failed: %s", risk_result, exc_info=risk_result)
            return {"error": f"Risk agent failed: {risk_result}", "status": "error"}

        logger.info(
            "parallel_analysis_node complete — stability: %.1f | DTI: %.2f, credit risk: %s",
            profile_result.income_stability_score,
            risk_result.debt_to_income_ratio,
            risk_result.credit_risk_level,
        )
        return {
            "profile": profile_result.model_dump(),
            "risk_assessment": risk_result.model_dump(),
        }

    except Exception as exc:
        logger.error("parallel_analysis_node failed: %s", exc, exc_info=True)
        return {"error": f"Parallel analysis failed: {exc}", "status": "error"}


async def decision_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Node 2: Loan Decision Agent

    Input state keys used: application, profile, risk_assessment
    Output state keys set: decision, error (on failure)

    The decision agent uses Claude + DecisionSynthesis MCP to compute a
    composite risk score, confidence level, and final approve/reject/review.
    Additionally, this node invokes Claude directly for final LLM reasoning
    that synthesises all agent outputs into a coherent decision narrative.
    """
    logger.info("==> decision_node started")
    from agents.loan_decision_agent import decide

    try:
        application = state["application"]
        profile = ApplicantProfile(**state["profile"])
        risk_assessment = FinancialRiskAssessment(**state["risk_assessment"])

        decision: LoanDecisionOutput = await decide(application, profile, risk_assessment)

        # Final LLM reasoning: Claude enriches the explanation using all outputs
        enriched_explanation = await _final_llm_reasoning(
            application, profile, risk_assessment, decision
        )
        decision.explanation = enriched_explanation

        logger.info(
            "decision_node complete — %s (risk: %.1f, confidence: %.1f%%)",
            decision.decision,
            decision.risk_score,
            decision.confidence_level,
        )
        return {"decision": decision.model_dump()}
    except Exception as exc:
        logger.error("decision_node failed: %s", exc, exc_info=True)
        return {"error": f"Decision agent failed: {exc}", "status": "error"}


async def compliance_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Node 3: Compliance & Action Orchestrator Agent

    Input state keys used: application, profile, risk_assessment, decision
    Output state keys set: compliance, case_id, status

    Records the decision, sends mock notifications, and creates the
    immutable audit trail entry via the NotificationSystem MCP server.
    """
    logger.info("==> compliance_node started")
    from agents.compliance_agent import record_and_notify

    try:
        application = state["application"]
        profile = ApplicantProfile(**state["profile"])
        risk_assessment = FinancialRiskAssessment(**state["risk_assessment"])
        decision = LoanDecisionOutput(**state["decision"])

        compliance: ComplianceRecord = await record_and_notify(
            application, profile, risk_assessment, decision
        )
        logger.info("compliance_node complete — case_id: %s", compliance.case_id)
        return {
            "compliance": compliance.model_dump(mode="json"),
            "case_id": compliance.case_id,
            "status": "completed",
        }
    except Exception as exc:
        logger.error("compliance_node failed: %s", exc, exc_info=True)
        return {"error": f"Compliance agent failed: {exc}", "status": "error"}


async def _final_llm_reasoning(
    application: Dict[str, Any],
    profile: ApplicantProfile,
    risk_assessment: FinancialRiskAssessment,
    decision: LoanDecisionOutput,
) -> str:
    """
    Final LLM reasoning step using Claude directly (no tools).

    Claude reads all four agent outputs and produces a single, coherent
    explanation that is more nuanced than any individual agent can provide.
    This enriched explanation replaces the raw decision agent explanation.
    """
    client = anthropic.AsyncAnthropic(
        api_key=settings.anthropic_api_key.get_secret_value(),
        base_url=settings.anthropic_base_url or None,
    )

    prompt = f"""You are the final reasoning layer of a loan approval system.
Review all agent outputs and write a clear, concise, audit-ready explanation
of the decision in 3-5 sentences.  Cite specific numbers.  Be factual.

Application: {json.dumps(application, indent=2)}
Applicant Profile: {json.dumps(profile.model_dump(), indent=2)}
Financial Risk Assessment: {json.dumps(risk_assessment.model_dump(), indent=2)}
Decision Agent Output:
  - Decision: {decision.decision}
  - Risk Score: {decision.risk_score}
  - Confidence: {decision.confidence_level}%
  - Key Factors: {decision.key_factors}

Write only the explanation text, no JSON, no preamble."""

    try:
        response = await create_message_with_retry(
            client,
            model=CLAUDE_MODEL,
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text if response.content else decision.explanation
    except Exception as exc:
        logger.warning("Final LLM reasoning failed, using original explanation: %s", exc)
        return decision.explanation


# ─── Routing Functions ─────────────────────────────────────────────────────────


def route_after_parallel(state: Dict[str, Any]) -> str:
    """
    After parallel_analysis_node, check both outputs before proceeding.

    Routes to:
      "decision_agent" — both profile and risk_assessment populated successfully
      "__end__"        — either agent failed, halt with error state
    """
    if state.get("error"):
        logger.warning("Routing to END due to parallel analysis error: %s", state["error"])
        return "__end__"
    if not state.get("profile") or not state.get("risk_assessment"):
        return "__end__"
    return "decision_agent"


# ─── Graph Construction ────────────────────────────────────────────────────────


def build_graph() -> Any:
    """
    Assembles and compiles the LangGraph workflow.

    Graph topology:
      parallel_analysis → (conditional) → decision_agent → compliance_agent → END

    Profile Agent and Risk Agent run concurrently inside parallel_analysis_node.
    The conditional edge halts the pipeline if either agent failed.
    """
    workflow = StateGraph(LoanGraphState)

    workflow.add_node("parallel_analysis", parallel_analysis_node)
    workflow.add_node("decision_agent", decision_node)
    workflow.add_node("compliance_agent", compliance_node)

    workflow.set_entry_point("parallel_analysis")

    workflow.add_conditional_edges(
        "parallel_analysis",
        route_after_parallel,
        {
            "decision_agent": "decision_agent",
            "__end__": END,
        },
    )

    workflow.add_edge("decision_agent", "compliance_agent")
    workflow.add_edge("compliance_agent", END)

    return workflow.compile()


# ─── Public Entry Point ────────────────────────────────────────────────────────


async def process_application(application: LoanApplicationRequest) -> Dict[str, Any]:
    """
    Main entry point for the orchestration layer.

    Accepts a validated LoanApplicationRequest, runs it through the full
    LangGraph pipeline, and returns the final state dict containing all
    agent outputs and the compliance record.

    Args:
        application: Validated Pydantic model from the API layer.

    Returns:
        Final LangGraph state dict with keys:
          application, profile, risk_assessment, decision, compliance,
          case_id, status, error (if any).
    """
    logger.info(
        "Processing application for: %s | loan: $%.0f",
        application.applicant_name,
        application.loan_amount,
    )

    initial_state: LoanGraphState = {
        "application": application.model_dump(),
        "profile": None,
        "risk_assessment": None,
        "decision": None,
        "compliance": None,
        "case_id": None,
        "error": None,
        "status": "processing",
    }

    graph = build_graph()
    final_state = await graph.ainvoke(initial_state)

    logger.info(
        "Pipeline complete for %s — status: %s, case_id: %s",
        application.applicant_name,
        final_state.get("status"),
        final_state.get("case_id"),
    )

    return final_state


# ─── CLI test runner ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import asyncio as _asyncio

    logging.basicConfig(level=logging.INFO)

    sample = LoanApplicationRequest(
        applicant_name="Jane Smith",
        annual_income=85000,
        monthly_debt=1200,
        loan_amount=25000,
        loan_purpose="Home renovation",
        credit_score=720,
        employment_type="full_time",
        years_employed=4,
        existing_loans=1,
        assets_value=50000,
    )

    result = _asyncio.run(process_application(sample))
    print(json.dumps(result, indent=2, default=str))
