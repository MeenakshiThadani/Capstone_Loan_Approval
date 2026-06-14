# Loan Approval Agent

An end-to-end multi-agent loan approval system built with Claude, LangGraph, and MCP.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Streamlit UI (ui/app.py)                │
│              Chatbot interface · decision display           │
└──────────────────────────┬──────────────────────────────────┘
                           │ HTTP POST /loan/apply
┌──────────────────────────▼──────────────────────────────────┐
│                  FastAPI Service (services/api.py)          │
│         POST /loan/apply  ·  GET /loan/status/{id}         │
└──────────────────────────┬──────────────────────────────────┘
                           │ await process_application()
┌──────────────────────────▼──────────────────────────────────┐
│           LangGraph Orchestrator (orchestration/)           │
│                                                             │
│  profile_agent → risk_analysis → decision_agent            │
│                                       ↓                     │
│                              compliance_agent → END         │
└──────┬──────────────┬──────────────┬──────────────┬─────────┘
       │              │              │              │
  MCP stdio      MCP stdio      MCP stdio      MCP stdio
       │              │              │              │
┌──────▼──────┐ ┌─────▼──────┐ ┌───▼────────┐ ┌───▼──────────┐
│ ApplicantDB │ │RiskRulesDB │ │ Decision   │ │Notification  │
│   Server    │ │  Server    │ │ Synthesis  │ │  System      │
│  (mcp/)     │ │  (mcp/)    │ │  (mcp/)   │ │  (mcp/)      │
└─────────────┘ └────────────┘ └────────────┘ └──────────────┘
```

## Agent Pipeline

| # | Agent | MCP Server | Output |
|---|-------|-----------|--------|
| 1 | Applicant Profile Agent | ApplicantDB | Income stability, employment risk, credit summary, completeness flags |
| 2 | Financial Risk Agent | RiskRulesDB | DTI ratio, credit risk, loan amount risk, anomalies |
| 3 | Loan Decision Agent | DecisionSynthesis | Decision (approved/rejected/review), risk score 0-100, confidence %, explanation |
| 4 | Compliance Agent | NotificationSystem | Case ID, notification, audit trail |

## Decision Thresholds

| Risk Score | Decision |
|-----------|---------|
| < 40 | Approved |
| 40 – 69 | Manual Review |
| ≥ 70 | Rejected |

**Hard rejection rules** (override score): unemployed, credit score < 550, DTI > 55%, ≥ 3 anomalies.

## Setup

### 1. Clone and create environment

```bash
cd loan_approval_agent
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and set ANTHROPIC_API_KEY=sk-ant-...
```

### 3. Run the FastAPI service

```bash
python services/api.py
# Listening on http://localhost:8000
# Docs at  http://localhost:8000/docs
```

### 4. Run the Streamlit UI

```bash
streamlit run ui/app.py
# Opens http://localhost:8501
```

## Testing

```bash
# Full suite (179 tests, ~4 seconds, no API key needed)
pytest

# Individual layers
pytest tests/test_schemas.py       # Layer 1: Data models
pytest tests/test_mcp_servers.py   # Layer 2: MCP business logic
pytest tests/test_agents.py        # Layer 3: Agent agentic loops
pytest tests/test_orchestrator.py  # Layer 4: LangGraph pipeline
pytest tests/test_api.py           # Layer 5: REST endpoints

# With coverage
pytest --cov=. --cov-report=term-missing
```

## Project Structure

```
loan_approval_agent/
├── ui/
│   └── app.py                    # Streamlit chatbot UI
├── services/
│   └── api.py                    # FastAPI REST microservice
├── orchestration/
│   └── orchestrator.py           # LangGraph pipeline
├── agents/
│   ├── applicant_profile_agent.py
│   ├── financial_risk_agent.py
│   ├── loan_decision_agent.py
│   └── compliance_agent.py
├── mcp/                          # MCP server scripts (not a Python package)
│   ├── applicant_db_server.py
│   ├── risk_rules_db_server.py
│   ├── decision_synthesis_server.py
│   └── notification_system_server.py
├── models/
│   └── schemas.py                # Shared Pydantic models
├── tests/                        # pytest test suite
├── .env.example
├── requirements.txt
└── README.md
```

## API Reference

### POST /loan/apply

Submit a loan application. Runs the full agent pipeline synchronously.

```json
{
  "applicant_name": "Jane Smith",
  "annual_income": 80000,
  "monthly_debt": 1200,
  "loan_amount": 25000,
  "loan_purpose": "Home renovation",
  "credit_score": 720,
  "employment_type": "full_time",
  "years_employed": 4.0,
  "existing_loans": 1,
  "assets_value": 50000
}
```

**Response** (`LoanStatusResponse`):
```json
{
  "case_id": "LOAN-20260604-A3F9B21C",
  "status": "completed",
  "decision": "approved",
  "risk_score": 22.5,
  "confidence_level": 91.0,
  "explanation": "Loan approved. Strong credit profile with low DTI...",
  "key_factors": ["Low DTI (18%)", "Prime credit score (720)"],
  "timestamp": "2026-06-04T10:23:45Z"
}
```

### GET /loan/status/{case_id}

Retrieve a previously processed decision by case ID.

### GET /health

Health check. Returns `{"status": "healthy"}`.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | — | **Required.** Your Anthropic API key |
| `CLAUDE_MODEL` | `claude-sonnet-4-6` | Claude model for all agents |
| `API_HOST` | `0.0.0.0` | FastAPI bind address |
| `API_PORT` | `8000` | FastAPI port |
| `API_BASE_URL` | `http://localhost:8000` | URL Streamlit uses to call the API |
| `LOG_LEVEL` | `INFO` | Python logging level |
