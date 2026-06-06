# Architecture: LangGraph Multi-Agent State Machine

## State machine diagram

```
                    START
                      |
                      v
              +----------------+
              |   researcher   |  research() node
              |                |  - runs 3 DuckDuckGo queries
              |                |  - LLM summarises results into a research brief
              +----------------+
                      |
                      v
              +----------------+
              |    writer      |  write() node
              |                |  - LLM drafts article from research brief
              |                |  - incorporates review_feedback if this is
              |                |    a revision iteration
              +----------------+
                      |
                      v
              +----------------+
              |   reviewer     |  review() node
              |                |  - LLM scores draft 1-10
              |                |  - returns {"score": N, "feedback": "...",
              |                |             "approved": true/false}
              +----------------+
                      |
           _should_revise() conditional edge
                      |
          +-----------+-------------+
          |                         |
     score < 7                 score >= 7
     AND iteration < 3         OR iteration >= 3
          |                         |
          v                         v
      writer (loop)               END
```

The graph is compiled once via `build_graph()` and then `invoke()`-ed with an
initial `AgentState`. LangGraph handles the loop internally — the Python process
does not use recursion or explicit loops; the graph executor drives state
transitions.

---

## Conditional edge: when does the loop terminate?

The conditional edge is defined in `src/orchestrator.py`:

```python
def _should_revise(state: AgentState) -> str:
    """Conditional edge: route back to writer if not approved, else finish."""
    if state["status"] == "complete":
        return "end"
    return "writer"
```

The `reviewer` node sets `state["status"]` to one of two values:

| Condition | `status` set by reviewer | `_should_revise` returns | Next node |
|-----------|--------------------------|--------------------------|-----------|
| score >= 7 | `"complete"` | `"end"` | END (graph finishes) |
| score < 7 AND iteration < 3 | `"writing"` | `"writer"` | writer (revision) |
| iteration >= 3 (any score) | `"complete"` | `"end"` | END (forced termination) |

The forced-termination rule (iteration >= 3) prevents infinite loops when the
LLM reviewer repeatedly rejects drafts. This is a standard pattern for
production agent loops.

---

## AgentState: shared memory across nodes

```python
# src/state.py
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]  # chat history (accumulated)
    topic: str                               # user input — never mutated
    research: str                            # populated by researcher node
    draft: str                               # populated/updated by writer node
    review_feedback: str                     # populated by reviewer node
    iteration: int                           # incremented by reviewer each cycle
    status: str                              # "researching"|"writing"|"reviewing"|"complete"
```

Every node receives the full state dict and returns a partial dict with only the
fields it modifies. LangGraph merges returned fields into the shared state
before passing it to the next node. The `messages` field uses the `add_messages`
reducer which appends rather than replaces — all other fields replace on update.

---

## LangGraph StateGraph code pattern

```python
# src/orchestrator.py — annotated

from langgraph.graph import StateGraph, END
from src.state import AgentState

graph = StateGraph(AgentState)      # declare state schema

# 1. Add nodes — each is a callable(state: dict) -> dict
graph.add_node("researcher", research)
graph.add_node("writer", write)
graph.add_node("reviewer", review)

# 2. Set the entry point
graph.set_entry_point("researcher")

# 3. Add deterministic edges (always follow this transition)
graph.add_edge("researcher", "writer")
graph.add_edge("writer", "reviewer")

# 4. Add a conditional edge (router function picks the next node)
graph.add_conditional_edges(
    "reviewer",           # source node
    _should_revise,       # router: receives state, returns a key from the map below
    {
        "writer": "writer",  # if router returns "writer" -> go to writer node
        "end": END,          # if router returns "end"   -> terminate graph
    },
)

app = graph.compile()     # compile to an executable graph

# 5. Run — invoke() blocks until END is reached
final_state = app.invoke(initial_state)
```

Key LangGraph concepts demonstrated here:

- `StateGraph(Schema)` — typed state container; all nodes share one state object
- `add_node(name, callable)` — nodes are plain Python functions
- `set_entry_point(name)` — equivalent to `add_edge(START, name)`
- `add_edge(a, b)` — unconditional transition
- `add_conditional_edges(source, router_fn, mapping)` — branching; `router_fn` returns a string key
- `graph.compile()` — returns a `CompiledGraph` that implements `.invoke()` and `.stream()`

---

## Fallback mode behaviour

When no LLM API key is present, `get_llm()` returns `None`. Each agent checks
for `None` and falls back:

| Agent | Fallback behaviour |
|-------|--------------------|
| Researcher | Formats raw DuckDuckGo results as a plain-text brief; if network is unavailable, returns a canned template |
| Writer | Builds a structured markdown article from the top 8 lines of research text using string templates |
| Reviewer | Heuristic scoring: +1 for word count > 200, +1 for >= 3 headings, +1 for conclusion section, +1 for >= 4 paragraphs; score >= 7 → approved |

The state machine graph, state transitions, conditional edges, and iteration
counting all function identically in fallback mode. This makes fallback mode
reliable for testing orchestration logic without incurring API costs.
