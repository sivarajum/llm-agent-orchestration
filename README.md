# LLM Agent Orchestration

Multi-agent system with LangGraph — researcher/writer/reviewer pipeline with iterative refinement. Offline fallback mode available. FastAPI API + Streamlit dashboard.

## What It Does

- **3 Specialized Agents**: Researcher (web search + summarization), Writer (article generation), Reviewer (quality scoring + feedback)
- **LangGraph StateGraph**: Agents wired into a state machine with conditional edges — reviewer loops back to writer if score < 7/10
- **Iterative Refinement**: Write → review → revise loop terminates when approved or after max iterations
- **Fallback Mode**: Runs without API keys using DuckDuckGo search + template-based writing + heuristic scoring
- **REST API**: FastAPI endpoints for pipeline execution and status
- **Dashboard**: Streamlit UI for topic input and pipeline visualization

## Architecture

```
src/state.py              # AgentState TypedDict (shared state between nodes)
src/orchestrator.py       # LangGraph StateGraph definition + run_pipeline()
src/llm.py                # LLM factory (ChatOpenAI / ChatAnthropic / None)
src/tools.py              # Web search tool (DuckDuckGo)
src/agents/
  researcher.py           # Research node — web search + LLM summarization
  writer.py               # Writer node — article generation from research
  reviewer.py             # Reviewer node — quality scoring + feedback
src/api.py                # FastAPI REST API
src/ui.py                 # Streamlit dashboard
```

## Quick Start

```bash
pip install -r requirements.txt

# Optional: configure LLM API key
cp .env.example .env      # Add OPENAI_API_KEY or ANTHROPIC_API_KEY

python main.py run "Benefits of Python"   # Run pipeline directly
python main.py api                        # API on :8009
python main.py ui                         # Dashboard on :8501
```

## Testing

```bash
pytest                     # 59 tests
```

## LLM Configuration

| Key | Model |
|-----|-------|
| `OPENAI_API_KEY` | gpt-4o-mini (preferred) |
| `ANTHROPIC_API_KEY` | claude-sonnet-4-20250514 |
| Neither | Fallback mode (DuckDuckGo + templates + heuristics) |

## Docker

```bash
docker compose up --build
```

See [RUNNING.md](RUNNING.md) for full build, test, and deployment instructions.
