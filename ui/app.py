"""
Streamlit Chatbot UI — Loan Approval Agent

A conversational interface for loan application submission and decision display.

User flow:
  1. User fills out the loan application form in the sidebar
  2. Submits → real-time status messages appear in the chat while processing
  3. Final decision is rendered with colour-coded risk score, confidence,
     key factors, and full audit explanation
  4. Chat history persists in session state for the current session

Run:
    streamlit run ui/app.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import httpx
import streamlit as st

_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from config.settings import settings

# ── Configuration ─────────────────────────────────────────────────────────────
API_BASE_URL = settings.api_base_url

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Loan Approval Agent",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stDeployButton { display: none; }
    [data-testid="stDeployButton"] { display: none; }
    [data-testid="stToolbar"] { display: none; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Session state initialisation ──────────────────────────────────────────────
# chat_messages holds the conversation history displayed in the main panel.
# Each entry: {"role": "user"|"assistant"|"system", "content": str}
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = [
        {
            "role": "assistant",
            "content": (
                "Hello! I'm the **Loan Approval Agent**. 👋\n\n"
                "Fill out the application form in the sidebar and click "
                "**Submit Application** to receive an AI-powered loan decision "
                "complete with risk score, confidence level, and a full explanation."
            ),
        }
    ]

# last_result stores the most recent LoanStatusResponse dict for display
if "last_result" not in st.session_state:
    st.session_state.last_result = None

# processing flag prevents double-submission
if "processing" not in st.session_state:
    st.session_state.processing = False


# ─── Helper Functions ──────────────────────────────────────────────────────────

def _decision_colour(decision: Optional[str]) -> str:
    """Map a decision string to a Streamlit-compatible colour name."""
    return {
        "approved": "green",
        "rejected": "red",
        "manual_review": "orange",
    }.get(decision or "", "gray")


def _risk_score_label(score: Optional[float]) -> str:
    """Convert numeric risk score to a human label for display."""
    if score is None:
        return "Unknown"
    if score < 40:
        return f"🟢 Low Risk ({score:.1f})"
    if score < 70:
        return f"🟡 Medium Risk ({score:.1f})"
    return f"🔴 High Risk ({score:.1f})"


def _submit_application(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    POST the application to the FastAPI service and return the response JSON.

    Raises:
        httpx.HTTPStatusError: on 4xx/5xx responses
        httpx.ConnectError: if the API is not reachable
    """
    with httpx.Client(timeout=300.0) as client:  # 5-minute timeout for full pipeline
        response = client.post(f"{API_BASE_URL}/loan/apply", json=payload)
        response.raise_for_status()
        return response.json()


def _add_message(role: str, content: str) -> None:
    """Append a message to the chat history in session state."""
    st.session_state.chat_messages.append({"role": role, "content": content})


# ─── Sidebar: Application Form ────────────────────────────────────────────────

with st.sidebar:
    st.title("📋 Loan Application")
    st.markdown("---")

    # ── Applicant info ────────────────────────────────────────────────────────
    st.subheader("Applicant Information")
    applicant_name = st.text_input(
        "Full Name *",
        placeholder="Jane Smith",
        help="Legal name as it appears on official documents",
    )
    _LOAN_REASONS = [
        "Home Purchase",
        "Home Renovation",
        "Car Purchase",
        "Education",
        "Medical Expenses",
        "Business Investment",
        "Debt Consolidation",
        "Personal / Travel",
        "Wedding",
        "Other",
    ]
    loan_purpose_choice = st.selectbox(
        "Loan Purpose *",
        options=_LOAN_REASONS,
        index=None,
        placeholder="Select a reason…",
    )
    if loan_purpose_choice == "Other":
        loan_purpose = st.text_input(
            "Please specify *",
            placeholder="Describe your loan purpose…",
        )
    else:
        loan_purpose = loan_purpose_choice or ""

    # ── Financial details ─────────────────────────────────────────────────────
    st.subheader("Financial Details")
    col1, col2 = st.columns(2)
    with col1:
        annual_income = st.number_input(
            "Annual Income ($) *",
            min_value=1,
            max_value=10_000_000,
            value=80_000,
            step=1_000,
            help="Gross annual income before tax",
        )
        loan_amount = st.number_input(
            "Loan Amount ($) *",
            min_value=1,
            max_value=10_000_000,
            value=25_000,
            step=1_000,
        )
    with col2:
        monthly_debt = st.number_input(
            "Monthly Debt ($) *",
            min_value=0,
            max_value=500_000,
            value=1_200,
            step=100,
            help="Total existing monthly debt payments",
        )
        assets_value = st.number_input(
            "Total Assets ($)",
            min_value=0,
            max_value=100_000_000,
            value=50_000,
            step=5_000,
        )

    # ── Credit profile ────────────────────────────────────────────────────────
    st.subheader("Credit Profile")
    credit_score = st.slider(
        "Credit Score (FICO) *",
        min_value=300,
        max_value=850,
        value=720,
        help="300 = Poor, 850 = Exceptional",
    )
    # Visual credit band indicator
    if credit_score >= 800:
        st.caption("✨ Exceptional")
    elif credit_score >= 740:
        st.caption("⭐ Very Good")
    elif credit_score >= 670:
        st.caption("👍 Good")
    elif credit_score >= 580:
        st.caption("⚠️ Fair")
    else:
        st.caption("🚨 Poor")

    existing_loans = st.number_input(
        "Existing Open Loans",
        min_value=0,
        max_value=50,
        value=1,
    )

    # ── Employment ────────────────────────────────────────────────────────────
    st.subheader("Employment")
    employment_type = st.selectbox(
        "Employment Type *",
        options=["full_time", "part_time", "contract", "self_employed", "unemployed"],
        format_func=lambda x: x.replace("_", " ").title(),
    )
    years_employed = st.number_input(
        "Years at Current Employer",
        min_value=0.0,
        max_value=50.0,
        value=3.0,
        step=0.5,
    )

    st.markdown("---")

    # ── Submit button ─────────────────────────────────────────────────────────
    submit_disabled = st.session_state.processing
    submitted = st.button(
        "🚀 Submit Application",
        type="primary",
        disabled=submit_disabled,
        use_container_width=True,
    )


# ─── Main Panel: Chat Interface ────────────────────────────────────────────────

st.title("🏦 Loan Approval Agent")

# Render chat history
chat_container = st.container()
with chat_container:
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# ─── Decision Display ──────────────────────────────────────────────────────────

if st.session_state.last_result:
    result = st.session_state.last_result
    decision = result.get("decision")
    colour = _decision_colour(decision)

    st.markdown("---")
    st.subheader("📊 Decision Summary")

    # Top-level metric cards
    col1, col2, col3 = st.columns(3)
    with col1:
        decision_display = (decision or "unknown").upper().replace("_", " ")
        st.metric("Decision", decision_display)
    with col2:
        risk = result.get("risk_score")
        st.metric("Risk Score", _risk_score_label(risk))
    with col3:
        conf = result.get("confidence_level")
        st.metric("Confidence", f"{conf:.1f}%" if conf is not None else "N/A")

    # Key factors
    key_factors = result.get("key_factors") or []
    if key_factors:
        st.subheader("🔑 Key Decision Factors")
        for factor in key_factors:
            st.markdown(f"- {factor}")

    # Full explanation
    explanation = result.get("explanation")
    if explanation:
        st.subheader("📄 Full Explanation")
        st.info(explanation)

    # Case ID and timestamp for audit trail
    case_id = result.get("case_id", "")
    timestamp = result.get("timestamp", "")
    st.caption(f"Case ID: `{case_id}` · Processed: {timestamp}")


# ─── Submission Handler ────────────────────────────────────────────────────────

if submitted and not st.session_state.processing:
    # Validate required text fields before calling API
    if not applicant_name.strip():
        st.sidebar.error("Please enter the applicant's full name.")
    elif not loan_purpose.strip():
        st.sidebar.error("Please specify the loan purpose.")
    else:
        st.session_state.processing = True

        # Add user's application summary to chat
        user_summary = (
            f"**Loan Application Submitted**\n\n"
            f"- **Applicant:** {applicant_name}\n"
            f"- **Amount:** ${loan_amount:,}\n"
            f"- **Purpose:** {loan_purpose}\n"
            f"- **Income:** ${annual_income:,}/yr\n"
            f"- **Credit Score:** {credit_score}\n"
            f"- **Employment:** {employment_type.replace('_', ' ').title()}"
        )
        _add_message("user", user_summary)

        # Show real-time processing status in the assistant turn
        with st.chat_message("assistant"):
            status_placeholder = st.empty()

            steps = [
                "🔍 Analysing applicant profile…",
                "📊 Running financial risk assessment…",
                "⚖️ Computing loan decision…",
                "📋 Recording compliance and audit trail…",
            ]

            # Animate through the pipeline stages
            for step in steps:
                status_placeholder.markdown(step)
                time.sleep(0.4)

            status_placeholder.markdown("⏳ Awaiting final decision from AI agents…")

            try:
                payload = dict(
                    applicant_name=applicant_name,
                    annual_income=float(annual_income),
                    monthly_debt=float(monthly_debt),
                    loan_amount=float(loan_amount),
                    loan_purpose=loan_purpose,
                    credit_score=int(credit_score),
                    employment_type=employment_type,
                    years_employed=float(years_employed),
                    existing_loans=int(existing_loans),
                    assets_value=float(assets_value),
                )

                result = _submit_application(payload)
                st.session_state.last_result = result

                # Build the assistant's decision message
                decision = result.get("decision", "unknown")
                risk_score = result.get("risk_score")
                confidence = result.get("confidence_level")
                case_id = result.get("case_id", "N/A")

                decision_emoji = {"approved": "✅", "rejected": "❌", "manual_review": "🔄"}.get(decision, "❓")
                assistant_reply = (
                    f"{decision_emoji} **Decision: {decision.upper().replace('_', ' ')}**\n\n"
                    f"- **Risk Score:** {_risk_score_label(risk_score)}\n"
                    f"- **Confidence:** {confidence:.1f}%\n"
                    f"- **Case ID:** `{case_id}`\n\n"
                    "See the **Decision Summary** panel below for full details."
                )
                status_placeholder.markdown(assistant_reply)
                _add_message("assistant", assistant_reply)

            except httpx.ConnectError:
                msg = (
                    "❌ **Could not connect to the API.** "
                    f"Is the FastAPI service running at `{API_BASE_URL}`?\n\n"
                    "Start it with: `python services/api.py`"
                )
                status_placeholder.error(msg)
                _add_message("assistant", msg)

            except httpx.HTTPStatusError as exc:
                msg = f"❌ **API Error {exc.response.status_code}:** {exc.response.text}"
                status_placeholder.error(msg)
                _add_message("assistant", msg)

            except Exception as exc:
                msg = f"❌ **Unexpected error:** {exc}"
                status_placeholder.error(msg)
                _add_message("assistant", msg)

            finally:
                st.session_state.processing = False

        st.rerun()
