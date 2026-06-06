# POC-05: LLM Agent Orchestration

This is a working multi-agent system using LangGraph. API key required for full LLM output; the system also runs in offline fallback mode without one.

Three specialized agents — Researcher, Writer, and Reviewer — are wired into a LangGraph `StateGraph`. The graph runs as a loop: Researcher gathers information, Writer drafts an article, Reviewer scores it, and if the score is below 7/10 the Writer revises. The loop terminates when the draft is approved or after 3 iterations.

---

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure API key
cp .env.example .env
# Edit .env — add OPENAI_API_KEY or ANTHROPIC_API_KEY (not both needed)

# 3. Run
python main.py
```

The `.env.example` file shows all supported variables.

---

## API key

| Key | Model used |
|-----|-----------|
| `OPENAI_API_KEY` | `gpt-4o-mini` (preferred, cheaper) |
| `ANTHROPIC_API_KEY` | `claude-sonnet-4-20250514` (alternative) |

Only one key is needed. If `OPENAI_API_KEY` is set, it takes precedence.

---

## Offline fallback mode

If neither key is set, the system runs in fallback mode:

- **Researcher**: Runs web search queries via DuckDuckGo. If all searches fail (no network), uses a canned research template.
- **Writer**: Generates a structured article from the research text using string templates — no LLM call.
- **Reviewer**: Applies heuristic scoring (word count, heading count, conclusion presence) — no LLM call.

The graph still runs the full researcher → writer → reviewer loop. Output quality is lower without an LLM but the state machine executes correctly, which makes fallback mode useful for testing the orchestration logic.

---

## Project structure

```
POC-05-LLM-Agent-Orchestration/
├── main.py
├── requirements.txt
├── .env.example
├── src/
│   ├── state.py          # AgentState TypedDict — shared state passed between nodes
│   ├── orchestrator.py   # LangGraph StateGraph definition and run_pipeline()
│   ├── llm.py            # get_llm() factory — returns ChatOpenAI or ChatAnthropic or None
│   ├── tools.py          # web_search tool (DuckDuckGo)
│   ├── agents/
│   │   ├── researcher.py # research() node — web search + LLM summarization
│   │   ├── writer.py     # write() node — article generation from research
│   │   └── reviewer.py   # review() node — quality scoring + feedback
│   ├── api.py            # FastAPI endpoints
│   └── ui.py             # Streamlit UI
└── docs/
    └── architecture.md   # State machine diagram and LangGraph pattern explanation
```

---

## Running the pipeline directly

```python
from src.orchestrator import run_pipeline

result = run_pipeline("Kubernetes networking internals")
print(result["draft"])          # the final article
print(result["review_feedback"])  # reviewer score and feedback
print(result["iteration"])      # how many write-review cycles ran
```

---

## Architecture

See `docs/architecture.md` for the state machine diagram, conditional edge logic,
and an explanation of the LangGraph `StateGraph` code pattern.
