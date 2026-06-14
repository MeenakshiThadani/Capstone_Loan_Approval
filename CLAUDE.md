# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Setup:**
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # then set ANTHROPIC_API_KEY
```

**Run services:**
```bash
python services/api.py          # FastAPI on http://localhost:8000
streamlit run ui/app.py         # Streamlit UI on http://localhost:8501
```

**Tests:**
```bash
pytest                                       # full suite (179 tests, ~4s, no API key needed)
pytest tests/test_schemas.py                 # single module
pytest --cov=. --cov-report=term-missing     # with coverage
```

`pytest.ini` sets `asyncio_mode=auto` and `pythonpath=.`.

## Architecture

Four Claude agents run sequentially through a LangGraph pipeline:

```
LoanApplicationRequest
  → Profile Agent (analyze applicant)
  → Risk Agent (quantify financial risk)
  → Decision Agent (synthesize + final LLM reasoning)
  → Compliance Agent (generate case ID, audit trail, notifications)
  → LoanStatusResponse
```

**Entry points:**
- `services/api.py` — FastAPI REST layer; `POST /loan/apply` blocks until the full pipeline completes, result stored in an in-memory `_case_registry` for `GET /loan/status/{case_id}`
- `orchestration/orchestrator.py` — `process_application()` builds a LangGraph `StateGraph` and calls `graph.ainvoke()`
- `ui/app.py` — Streamlit chatbot that POSTs to the API

**Agent pattern (same for all four agents):**
Each agent spawns its MCP server as a subprocess (stdio), opens a `ClientSession`, runs an agentic tool-use loop until `stop_reason == "end_turn"`, then parses the final JSON reply into a Pydantic model.

**MCP servers** (`mcp/`) hold all domain business logic as callable tools; agents never implement rules directly:
| Server | Key tools |
|---|---|
| `applicant_db_server.py` | income stability, employment risk, credit summary, completeness |
| `risk_rules_db_server.py` | DTI ratio, credit score classification, loan-amount risk, anomaly detection |
| `decision_synthesis_server.py` | composite risk score (0–100), confidence, decision, explanation |
| `notification_system_server.py` | case ID generation, audit record, mock notifications |

**Decision thresholds** (in `decision_synthesis_server.py`):
- Risk score < 40 → Approved; 40–69 → Manual Review; ≥ 70 → Rejected
- Hard disqualifiers override the score: unemployed, FICO < 550, DTI > 55%, ≥ 3 anomalies

**LangGraph state** uses `TypedDict` (`LoanState` in `orchestration/orchestrator.py`), not Pydantic. Conditional edges after each node route to `END` on error, short-circuiting the pipeline.

**Shared data models** (`models/schemas.py`) — Pydantic v2, used at API boundary and between orchestrator/agents:
- Input: `LoanApplicationRequest`
- Per-agent outputs: `ApplicantProfile`, `FinancialRiskAssessment`, `LoanDecisionOutput`, `ComplianceRecord`
- API response: `LoanStatusResponse`

## Environment Variables

| Variable | Default | Notes |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | Required for live runs |
| `CLAUDE_MODEL` | `claude-sonnet-4-6` | Used by all agents |
| `API_HOST` / `API_PORT` | `0.0.0.0` / `8000` | FastAPI bind |
| `API_BASE_URL` | `http://localhost:8000` | Streamlit → API |
| `LOG_LEVEL` | `INFO` | |
