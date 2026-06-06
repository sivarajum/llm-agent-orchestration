# Multi-Agent Orchestration with LangGraph - Complete End-to-End Guide

This guide walks you through every piece of the multi-agent orchestration system, from a user-provided topic to a fully reviewed article. Each section explains **what** the code does, **why** it does it, and **how** to run it yourself.

---

## Table of Contents

1. [What Is This Project?](#1-what-is-this-project)
2. [Project Structure](#2-project-structure)
3. [Prerequisites - Setting Up Your Environment](#3-prerequisites---setting-up-your-environment)
4. [Core Concept: What Is LangGraph?](#4-core-concept-what-is-langgraph)
5. [Step 1: Shared State (src/state.py)](#5-step-1-shared-state)
6. [Step 2: Tools (src/tools.py)](#6-step-2-tools)
7. [Step 3: LLM Factory (src/llm.py)](#7-step-3-llm-factory)
8. [Step 4: Researcher Agent (src/agents/researcher.py)](#8-step-4-researcher-agent)
9. [Step 5: Writer Agent (src/agents/writer.py)](#9-step-5-writer-agent)
10. [Step 6: Reviewer Agent (src/agents/reviewer.py)](#10-step-6-reviewer-agent)
11. [Step 7: The Orchestrator (src/orchestrator.py)](#11-step-7-the-orchestrator)
12. [Step 8: REST API (src/api.py)](#12-step-8-rest-api)
13. [Step 9: Streamlit Dashboard (src/ui.py)](#13-step-9-streamlit-dashboard)
14. [Step 10: Entry Point (main.py)](#14-step-10-entry-point)
15. [Running the Full System](#15-running-the-full-system)
16. [Running with Docker](#16-running-with-docker)
17. [Testing the API Manually](#17-testing-the-api-manually)
18. [How Data Flows Through the System](#18-how-data-flows-through-the-system)
19. [Key Concepts Explained](#19-key-concepts-explained)
20. [Troubleshooting](#20-troubleshooting)

---

## 1. What Is This Project?

This project builds a **multi-agent system** where three AI agents collaborate to produce a researched article on any topic. The agents are:

1. **Researcher** -- searches the web and gathers information
2. **Writer** -- turns research into a structured markdown article
3. **Reviewer** -- scores the article and either approves it or sends it back for revision

The agents communicate through a **shared state** managed by LangGraph, which acts as a state machine controlling the flow between agents.

```
User provides a topic (e.g., "Quantum Computing")
       |
       v
RESEARCHER agent
  - Runs 3 DuckDuckGo searches
  - Summarizes findings into a research brief
       |
       v
WRITER agent
  - Reads the research brief
  - Produces a structured markdown article
       |
       v
REVIEWER agent
  - Scores the article 1-10
  - If score >= 7 or 3 iterations done --> DONE
  - If score < 7 --> sends feedback, loops back to WRITER
       |
       v
Final article + research + review feedback returned
```

The system works in two modes:

- **LLM-enhanced mode**: If you provide an OpenAI or Anthropic API key, agents use a real LLM for summarization, writing, and reviewing.
- **Fallback mode**: If no API keys are provided, agents use heuristic-based logic (template articles, word-count scoring). This means the project runs out of the box with zero configuration.

---

## 2. Project Structure

```
POC-05-LLM-Agent-Orchestration/
|
|-- main.py                  # Entry point - run everything from here
|-- requirements.txt         # Python packages needed
|-- .env.example             # Optional API key configuration
|-- Dockerfile               # Container build instructions
|-- docker-compose.yml       # Multi-container orchestration
|
|-- src/
|   |-- __init__.py          # Makes src/ a Python package
|   |-- state.py             # Step 1: AgentState TypedDict (shared memory)
|   |-- tools.py             # Step 2: Web search and calculator tools
|   |-- llm.py               # Step 3: LLM factory (OpenAI / Anthropic / None)
|   |-- orchestrator.py      # Step 7: LangGraph state machine wiring
|   |-- api.py               # Step 8: FastAPI REST endpoints
|   |-- ui.py                # Step 9: Streamlit visual dashboard
|   |
|   |-- agents/
|       |-- __init__.py      # Makes agents/ a Python package
|       |-- researcher.py    # Step 4: Web search + summarization
|       |-- writer.py        # Step 5: Article generation
|       |-- reviewer.py      # Step 6: Quality scoring + feedback
```

---

## 3. Prerequisites - Setting Up Your Environment

### Step 3.1: Make sure you have Python 3.10+

```bash
python --version
# Should print Python 3.10.x or higher
```

### Step 3.2: Navigate to the project directory

```bash
cd POCs/POC-05-LLM-Agent-Orchestration
```

### Step 3.3: (Recommended) Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate    # macOS/Linux
# venv\Scripts\activate     # Windows
```

### Step 3.4: Install dependencies

```bash
pip install langgraph langchain langchain-core langchain-openai langchain-anthropic \
    duckduckgo-search fastapi uvicorn streamlit httpx python-dotenv
```

This installs:

| Package | What It Does |
|---------|-------------|
| `langgraph` | State machine framework for orchestrating AI agents |
| `langchain` | Foundation library for LLM application building blocks |
| `langchain-core` | Core abstractions (tools, messages, prompts) |
| `langchain-openai` | OpenAI LLM integration (optional, for LLM mode) |
| `langchain-anthropic` | Anthropic LLM integration (optional, for LLM mode) |
| `duckduckgo-search` | Free web search API (no API key needed) |
| `fastapi` | Web framework for the REST API |
| `uvicorn` | ASGI server that runs FastAPI |
| `streamlit` | Web framework for the dashboard UI |
| `httpx` | Async-capable HTTP client (UI calls the API with this) |
| `python-dotenv` | Loads `.env` files into environment variables |

### Step 3.5: (Optional) Configure API keys

Copy the example environment file and add your keys:

```bash
cp .env.example .env
```

Edit `.env`:

```
# OpenAI (preferred)
OPENAI_API_KEY=sk-your-key-here

# OR Anthropic (alternative)
ANTHROPIC_API_KEY=sk-ant-your-key-here
```

**You do NOT need API keys to run this project.** Without keys, the system operates in fallback mode using heuristics instead of an LLM. The pipeline still runs end-to-end -- you get a template-based article instead of an LLM-generated one.

---

## 4. Core Concept: What Is LangGraph?

Before diving into the code, let us understand the framework that ties everything together.

### The problem LangGraph solves

Traditional LLM applications are linear: prompt in, response out. But real-world tasks often need **multiple steps**, **branching logic**, and **loops**. For example:

```
Simple LLM call:
  User question --> LLM --> Answer

Multi-agent pipeline (what we need):
  Topic --> Research --> Write --> Review --+--> Done
                                  ^        |
                                  |        | (revision needed)
                                  +--------+
```

LangGraph gives us a way to define this as a **state machine** -- a graph where:

- **Nodes** are functions (our agents)
- **Edges** define the flow between nodes
- **State** is a shared dictionary passed between all nodes
- **Conditional edges** allow branching (e.g., "if approved, stop; otherwise, revise")

### State machine basics

A state machine has:

1. **States** -- the possible situations (researching, writing, reviewing, complete)
2. **Transitions** -- rules for moving between states
3. **A shared memory** -- data that accumulates as the machine runs

```
                    +-------------------------------------------+
                    |           SHARED STATE (AgentState)        |
                    |                                           |
                    |  topic: "Quantum Computing"               |
                    |  research: "..."  (filled by Researcher)  |
                    |  draft: "..."     (filled by Writer)      |
                    |  review_feedback: "..." (filled by Rev.)  |
                    |  iteration: 0, 1, 2, or 3                 |
                    |  status: researching|writing|reviewing|... |
                    +-------------------------------------------+
                          ^        ^        ^
                          |        |        |
                    +-----+--+  +--+---+  +-+------+
                    |Research |  |Writer|  |Reviewer|
                    |  Node  |  | Node |  |  Node  |
                    +--------+  +------+  +--------+
```

Each node reads from the state, does its work, and writes its results back. LangGraph handles the plumbing.

### How LangGraph differs from a simple for-loop

You could write this with `if/else` and `while` loops. LangGraph adds:

- **Checkpointing** -- save and resume pipeline state
- **Streaming** -- watch agent outputs arrive in real time
- **Visualization** -- auto-generate flow diagrams
- **Parallelism** -- run independent nodes concurrently (not used here, but available)
- **Type safety** -- TypedDict state catches bugs at development time

---

## 5. Step 1: Shared State

**File:** `src/state.py`

### What it does

Defines the single data structure that every agent reads from and writes to. This is the "shared memory" of the entire pipeline.

### The AgentState TypedDict

```python
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    topic: str
    research: str
    draft: str
    review_feedback: str
    iteration: int
    status: str
```

### Field-by-field breakdown

| Field | Type | Set By | Purpose |
|-------|------|--------|---------|
| `messages` | list | LangGraph | Chat message history; accumulated via the `add_messages` reducer |
| `topic` | str | User | The subject to research (e.g., "Quantum Computing") |
| `research` | str | Researcher | Web search results, optionally summarized by LLM |
| `draft` | str | Writer | The markdown article produced from research |
| `review_feedback` | str | Reviewer | Score and feedback (e.g., "Score: 8/10\nGood structure...") |
| `iteration` | int | Reviewer | Current revision count (starts at 0, increments each review) |
| `status` | str | All agents | Current pipeline stage: `researching`, `writing`, `reviewing`, or `complete` |

### What is `Annotated[list, add_messages]`?

This is a LangGraph **reducer**. When a node returns `{"messages": [new_message]}`, LangGraph does not replace the list -- it appends the new message to the existing list. This is how chat history accumulates without each node needing to know about previous messages.

```
Node 1 returns: {"messages": [msg_A]}
State after:    {"messages": [msg_A]}

Node 2 returns: {"messages": [msg_B]}
State after:    {"messages": [msg_A, msg_B]}   <-- appended, not replaced
```

For all other fields (topic, research, draft, etc.), the default behavior is **replace** -- the latest value wins.

### Why TypedDict?

TypedDict gives us:

1. **Type checking** -- your editor can warn if you misspell a field name
2. **Documentation** -- every field and its type is visible in one place
3. **LangGraph integration** -- LangGraph reads the TypedDict to know the state schema

---

## 6. Step 2: Tools

**File:** `src/tools.py`

### What it does

Defines two tools that agents can use: a web search tool and a calculator. These are decorated with LangChain's `@tool` decorator, which makes them callable by agents and LLMs.

### Tool 1: web_search

```python
@tool
def web_search(query: str) -> str:
    """Search the web using DuckDuckGo and return the top 3 results."""
```

**How it works, step by step:**

1. Receives a search query string
2. Tries to import `DDGS` from `ddgs` (newer package name), falls back to `duckduckgo_search`
3. Calls `ddgs.text(query, max_results=3)` to get the top 3 results
4. Formats each result as a numbered entry with title, body, and source URL
5. If search fails (no internet, rate-limited, etc.), returns an error string instead of crashing

**Example output:**

```
[1] Quantum Computing Explained - IBM Research
    Quantum computing leverages quantum mechanical phenomena...
    Source: https://research.ibm.com/quantum-computing

[2] Latest Breakthroughs in Quantum Tech
    In 2025, researchers achieved a 1000-qubit processor...
    Source: https://example.com/quantum-breakthroughs

[3] Quantum Computing Facts & Statistics
    The quantum computing market is projected to reach $125B...
    Source: https://example.com/quantum-stats
```

**Why DuckDuckGo?** It requires no API key, no account, and no payment. The `duckduckgo-search` Python package wraps DuckDuckGo's instant answer API.

### Tool 2: calculate

```python
@tool
def calculate(expression: str) -> str:
    """Safely evaluate a math expression."""
```

**How it works, step by step:**

1. Receives a math expression as a string (e.g., `"2 + 3 * 4"`)
2. Validates that only allowed characters are present: digits, operators (`+-*/.`), parentheses, and spaces
3. Uses Python's `eval()` with an empty `__builtins__` dict -- this prevents access to any Python functions, making it safe
4. Returns the result as a string

**Security note:** The `{"__builtins__": {}}` argument to `eval()` strips all built-in functions. You cannot call `os.system()`, `open()`, or any other dangerous function. Only arithmetic operators work.

### What does `@tool` do?

The `@tool` decorator from LangChain converts a regular Python function into a LangChain `Tool` object. This adds:

- A `.invoke()` method (the standard way LangChain calls tools)
- Metadata (name, description, argument schema) that an LLM can read to decide which tool to use
- Serialization support for passing tools to LLMs

In this project, the Researcher agent calls `web_search.invoke(query)` directly rather than having the LLM choose a tool. The `@tool` decorator still helps because it provides the `.invoke()` interface and handles error wrapping.

---

## 7. Step 3: LLM Factory

**File:** `src/llm.py`

### What it does

Provides a single function, `get_llm()`, that returns the best available LLM -- or `None` if no API keys are configured. Every agent calls this function to decide whether to use LLM-enhanced mode or fallback mode.

### The priority chain

```python
def get_llm(temperature: float = 0):
    if os.getenv("OPENAI_API_KEY"):
        return ChatOpenAI(model="gpt-4o-mini", temperature=temperature)
    if os.getenv("ANTHROPIC_API_KEY"):
        return ChatAnthropic(model="claude-sonnet-4-20250514", temperature=temperature)
    return None
```

```
Check OPENAI_API_KEY environment variable
       |
       +-- Found? --> Return ChatOpenAI(gpt-4o-mini)
       |
       +-- Not found? --> Check ANTHROPIC_API_KEY
                              |
                              +-- Found? --> Return ChatAnthropic(claude-sonnet)
                              |
                              +-- Not found? --> Return None (fallback mode)
```

### Why lazy imports?

Notice that `from langchain_openai import ChatOpenAI` happens inside the `if` block, not at the top of the file. This is intentional:

- If a user does not have `langchain-openai` installed (because they only use Anthropic), the import would crash at startup
- Lazy importing means the package is only loaded if actually needed
- This makes the project more flexible about which optional dependencies are installed

### What is `temperature`?

Temperature controls randomness in LLM output:

| Temperature | Behavior | Used By |
|-------------|----------|---------|
| 0.0 | Deterministic -- same input always gives same output | Researcher (wants accurate summaries), Reviewer (wants consistent scoring) |
| 0.7 | Creative -- varied phrasing and structure | Writer (wants engaging, diverse articles) |
| 1.0+ | Very random -- may produce incoherent output | Not used |

The Researcher and Reviewer call `get_llm()` (defaults to temperature=0). The Writer calls `get_llm(temperature=0.7)` for more creative output.

---

## 8. Step 4: Researcher Agent

**File:** `src/agents/researcher.py`

### What it does

Takes a topic, runs three web searches, and produces a research brief. This is the first agent in the pipeline -- it gathers the raw material that the Writer will turn into an article.

### The search strategy

The Researcher does not run just one search. It runs three, each targeting different types of information:

```
Topic: "Quantum Computing"

Search 1: "Quantum Computing"
  --> General overview results

Search 2: "Quantum Computing latest developments"
  --> Recent news and breakthroughs

Search 3: "Quantum Computing key facts and statistics"
  --> Numbers, market data, adoption metrics
```

This gives the Writer a well-rounded research base: background context, recent developments, and hard data.

### The _search_topic helper

```
_search_topic(topic)
       |
       v
  Run 3 web searches (web_search.invoke for each query)
       |
       v
  Combine results with "---" separators
       |
       v
  If ALL searches failed? --> Return offline fallback text
  Otherwise?              --> Return combined results
```

The offline fallback is a short placeholder paragraph. This ensures the pipeline never crashes, even without internet access.

### The research function (LangGraph node)

```
research(state) -- the function LangGraph calls
       |
       v
  Read state["topic"]
       |
       v
  Call _search_topic(topic) --> raw search results
       |
       v
  Call get_llm()
       |
       +-- LLM available?
       |     |
       |     v
       |   Send raw results to LLM with prompt:
       |   "Summarize these search results into a
       |    structured research brief with bullet points"
       |     |
       |     v
       |   research_output = LLM's summary
       |
       +-- No LLM?
             |
             v
           Format raw results with a header:
           "Research Brief: {topic}"
           "(Fallback mode - raw search results)"
             |
             v
           research_output = formatted raw text
       |
       v
  Return updated state:
    research = research_output
    status = "writing"
    iteration = (unchanged)
```

### What the agent returns

Each LangGraph node returns a **partial state update** -- a dict with only the fields that changed. LangGraph merges this into the full state:

```python
return {
    "research": research_output,    # New research text
    "status": "writing",            # Signal that Writer should run next
    "iteration": state.get("iteration", 0),  # Pass through unchanged
}
```

The `status` field is key. By setting it to `"writing"`, the Researcher signals that it is done and the Writer should go next. The orchestrator uses edges (not status) for this particular transition, but the status field helps track pipeline progress for the UI.

---

## 9. Step 5: Writer Agent

**File:** `src/agents/writer.py`

### What it does

Takes the research brief from the Researcher and produces a structured markdown article. If this is a revision (iteration > 0), it also incorporates the Reviewer's feedback.

### LLM-enhanced writing

When an LLM is available, the Writer sends this prompt:

```
"You are an expert content writer. Write a well-structured article
about '{topic}' using the research below. Include an introduction,
key findings, analysis, and conclusion. Use markdown formatting.

Research:
{research}

Previous review feedback to address:    <-- only if revision
{feedback}
```

The LLM generates a full article with headings, paragraphs, and proper markdown.

### Fallback writing (_fallback_article)

Without an LLM, the Writer builds a template article from the raw research:

```
Step 1: Split research text into lines
Step 2: Keep lines longer than 20 characters (filter noise)
Step 3: Take the first 8 substantial lines as key points

Build article:
  # {Topic}

  ## Introduction
  This report covers key findings on the topic of {topic}...

  ## Key Findings
  1. {first substantial line from research}
  2. {second substantial line}
  ...up to 8 points

  ## Analysis
  Based on the research gathered, {topic} is a multifaceted subject...

  ## Conclusion
  The research highlights important aspects of {topic}...

  ---
  *Revision note: incorporated feedback from review iteration.*
  (only if feedback exists)
```

This produces a well-structured article even without any LLM. It will not be as eloquent, but it demonstrates the pipeline mechanics.

### The write function (LangGraph node)

```
write(state) -- the function LangGraph calls
       |
       v
  Read state["topic"], state["research"], state["review_feedback"]
       |
       v
  Call get_llm(temperature=0.7)    <-- higher temp for creativity
       |
       +-- LLM available?
       |     |
       |     v
       |   Build prompt with research + any feedback
       |   Call llm.invoke(prompt)
       |   draft = response.content
       |
       +-- No LLM?
             |
             v
           Call _fallback_article(topic, research, feedback)
           draft = template article
       |
       v
  Return updated state:
    draft = article text
    status = "reviewing"
```

### Why temperature 0.7?

The Writer is the one agent where creativity matters. At temperature 0, the LLM would produce the same article every time. At 0.7, it varies sentence structure, word choice, and paragraph flow -- producing more natural-sounding text. The Researcher and Reviewer use temperature 0 because accuracy and consistency matter more for those tasks.

---

## 10. Step 6: Reviewer Agent

**File:** `src/agents/reviewer.py`

### What it does

Evaluates the Writer's draft and either approves it (pipeline ends) or sends it back for revision (pipeline loops). This is the **decision point** that creates the revision loop -- the most interesting pattern in the system.

### LLM-enhanced reviewing

When an LLM is available, the Reviewer sends this prompt:

```
"You are a content reviewer. Evaluate this article draft.
Respond with valid JSON only:
{"score": <1-10>, "feedback": "<detailed feedback>", "approved": <true/false>}
Score 7+ means approved.

Draft:
{draft}"
```

The LLM returns a JSON object. If parsing fails (the LLM sometimes adds extra text around JSON), the Reviewer falls back to heuristic review.

### Fallback reviewing (_fallback_review)

Without an LLM, the Reviewer uses a scoring system based on measurable qualities:

```
Start with base score: 5

Check 1: Word count
  - More than 200 words?  --> score += 1, "Good length (N words)."
  - 200 or fewer?         --> no change, "Short article. Consider expanding."

Check 2: Headings
  - 3+ "#" characters?    --> score += 1, "Good use of section headings."
  - Fewer?                --> no change, "Add more section headings."

Check 3: Conclusion
  - "conclusion" found?   --> score += 1, "Conclusion section present."
  - Not found?            --> no change, "Missing conclusion section."

Check 4: Paragraph structure
  - 4+ paragraphs (30+ chars each)?  --> score += 1, "Well-structured."
  - Fewer?                            --> no change, "Add more detailed paragraphs."

Final score: 5 + (0 to 4 bonus points) = range 5 to 9
Capped at 10.
```

A well-structured fallback article from the Writer typically scores 8 or 9 (all four checks pass), so it gets approved on the first try.

### The decision logic

```
review(state) -- the function LangGraph calls
       |
       v
  Read state["draft"]
  iteration = state["iteration"] + 1    <-- increment on every review
       |
       v
  Get review (LLM or fallback) --> {score, feedback, approved}
       |
       v
  Check: iteration >= 3?
       |
       +-- Yes --> Force approved = True   (safety valve: max 3 revisions)
       |
       +-- No --> Keep the LLM's/heuristic's decision
       |
       v
  If approved:
    status = "complete"     --> pipeline ends
  Else:
    status = "writing"      --> pipeline loops back to Writer
       |
       v
  Return updated state:
    review_feedback = "Score: {score}/10\n{feedback}"
    iteration = incremented count
    status = "complete" or "writing"
```

### The three-iteration safety valve

The pipeline allows at most 3 revision cycles. This prevents infinite loops if the LLM keeps scoring below 7:

```
Iteration 1: Researcher --> Writer --> Reviewer (score: 5, needs revision)
Iteration 2:               Writer --> Reviewer (score: 6, needs revision)
Iteration 3:               Writer --> Reviewer (score: 6, but forced approval)
                                                         ^^^^^^^^^^^^^^^^^
                                               Max iterations reached, stop.
```

### Score thresholds

| Score | Meaning | Action |
|-------|---------|--------|
| 1-6 | Below quality bar | Send back to Writer with feedback (unless iteration >= 3) |
| 7-10 | Meets quality bar | Approve, set status to "complete" |

---

## 11. Step 7: The Orchestrator

**File:** `src/orchestrator.py`

### What it does

This is the **brain** of the system. It wires the three agents into a LangGraph state machine, defining the flow, the conditional loop, and the entry/exit points.

### Building the graph (build_graph)

```python
graph = StateGraph(AgentState)

# Add agent nodes
graph.add_node("researcher", research)
graph.add_node("writer", write)
graph.add_node("reviewer", review)

# Linear flow
graph.set_entry_point("researcher")
graph.add_edge("researcher", "writer")
graph.add_edge("writer", "reviewer")

# Conditional loop
graph.add_conditional_edges(
    "reviewer",
    _should_revise,
    {"writer": "writer", "end": END},
)

return graph.compile()
```

### The graph visualized

```
    +----------+       +--------+       +----------+
    |          |       |        |       |          |
--->|RESEARCHER|------>| WRITER |------>| REVIEWER |
    |          |       |        |       |          |
    +----------+       +---^----+       +-----+----+
                           |                  |
                           |   score < 7      |
                           |   AND iter < 3   |
                           +------------------+
                                              |
                                              | score >= 7
                                              | OR iter >= 3
                                              |
                                              v
                                           [END]
```

### Understanding each piece

**`StateGraph(AgentState)`** -- Creates a new graph using our AgentState schema. LangGraph validates that nodes return dicts matching the state fields.

**`add_node("researcher", research)`** -- Registers the `research` function as a node named "researcher". When LangGraph reaches this node, it calls `research(state)`.

**`set_entry_point("researcher")`** -- Tells LangGraph which node to run first when the pipeline starts.

**`add_edge("researcher", "writer")`** -- Unconditional edge: after "researcher" finishes, always go to "writer".

**`add_edge("writer", "reviewer")`** -- Unconditional edge: after "writer" finishes, always go to "reviewer".

**`add_conditional_edges("reviewer", _should_revise, ...)`** -- This is the key piece. After "reviewer" finishes, LangGraph calls `_should_revise(state)` which returns either `"writer"` or `"end"`:

```python
def _should_revise(state: AgentState) -> str:
    if state["status"] == "complete":
        return "end"
    return "writer"
```

The mapping `{"writer": "writer", "end": END}` translates the return value to actual node targets. `END` is a special LangGraph constant meaning "stop the pipeline."

**`graph.compile()`** -- Validates the graph (no disconnected nodes, all edges valid) and returns a runnable object.

### Running the pipeline (run_pipeline)

```python
def run_pipeline(topic: str) -> dict:
    app = build_graph()

    initial_state: AgentState = {
        "messages": [],
        "topic": topic,
        "research": "",
        "draft": "",
        "review_feedback": "",
        "iteration": 0,
        "status": "researching",
    }

    final_state = app.invoke(initial_state)
    return dict(final_state)
```

**Step by step:**

1. `build_graph()` compiles the state machine
2. An initial state is created with the user's topic and empty fields
3. `app.invoke(initial_state)` runs the graph from START to END, passing state through each node
4. The final state contains all accumulated data: research, draft, feedback, iteration count, and status

### What `invoke` does internally

```
invoke(initial_state)
  |
  v
Run "researcher" node
  - Input:  {topic: "Quantum Computing", research: "", ...}
  - Output: {research: "...", status: "writing"}
  - State:  {topic: "Quantum Computing", research: "...", status: "writing", ...}
  |
  v
Follow edge: researcher --> writer
  |
  v
Run "writer" node
  - Input:  {topic: "Quantum Computing", research: "...", ...}
  - Output: {draft: "# Quantum Computing\n...", status: "reviewing"}
  - State:  {topic: "...", research: "...", draft: "...", status: "reviewing"}
  |
  v
Follow edge: writer --> reviewer
  |
  v
Run "reviewer" node
  - Input:  {topic: "...", research: "...", draft: "...", ...}
  - Output: {review_feedback: "Score: 8/10\n...", iteration: 1, status: "complete"}
  - State:  {topic: "...", ..., review_feedback: "...", iteration: 1, status: "complete"}
  |
  v
Evaluate conditional edge: _should_revise(state) --> "end"
  |
  v
END --> return final_state
```

If the score had been below 7:

```
Run "reviewer" node
  - Output: {review_feedback: "Score: 5/10\n...", iteration: 1, status: "writing"}
  |
  v
Evaluate conditional edge: _should_revise(state) --> "writer"
  |
  v
Run "writer" node AGAIN (with feedback in state)
  - Output: {draft: "# Quantum Computing (revised)\n...", status: "reviewing"}
  |
  v
Run "reviewer" node AGAIN
  ... and so on up to 3 iterations
```

---

## 12. Step 8: REST API

**File:** `src/api.py`

### What it does

Wraps the agent pipeline in a web API using FastAPI. External clients (including the Streamlit dashboard) call these HTTP endpoints to run the pipeline.

### Application setup

```python
app = FastAPI(title="Agent Orchestration API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
```

CORS (Cross-Origin Resource Sharing) middleware is added with `allow_origins=["*"]` so the Streamlit dashboard (running on port 8501) can call the API (running on port 8000) without browser security blocks.

### Request and response models

```python
class RunRequest(BaseModel):
    topic: str

class RunResponse(BaseModel):
    topic: str
    research: str
    draft: str
    review_feedback: str
    iteration: int
    status: str
    elapsed_seconds: float
```

Pydantic models validate incoming requests and document the API automatically. If someone sends `{"topik": "..."}` (typo), FastAPI returns a 422 error with a clear message.

### Endpoints

#### `GET /health`

Health check. Returns the service name and status.

```json
{"status": "ok", "service": "agent-orchestration"}
```

Used by Docker health checks and the UI to verify the API is running.

#### `POST /run`

The main endpoint. Accepts a topic, runs the full pipeline, and returns all outputs.

**Request:**

```json
{"topic": "Quantum Computing breakthroughs in 2025"}
```

**What happens internally:**

```
1. Record start time
2. Call run_pipeline(topic)
   --> Researcher --> Writer --> Reviewer (with possible loops)
3. Record end time
4. Return all state fields + elapsed time
```

**Response (success):**

```json
{
  "topic": "Quantum Computing breakthroughs in 2025",
  "research": "Research Brief: Quantum Computing...\n...",
  "draft": "# Quantum Computing Breakthroughs in 2025\n\n## Introduction\n...",
  "review_feedback": "Score: 8/10\n- Good length (342 words).\n- Good use of section headings.\n...",
  "iteration": 1,
  "status": "complete",
  "elapsed_seconds": 12.34
}
```

**Response (error):**

If the pipeline crashes, the API catches the exception and returns an error response instead of a 500:

```json
{
  "topic": "...",
  "research": "",
  "draft": "",
  "review_feedback": "Pipeline error: ConnectionError...\n<traceback>",
  "iteration": 0,
  "status": "error",
  "elapsed_seconds": 2.1
}
```

This error handling is important because the pipeline involves network calls (web search) that can fail. The API never crashes -- it always returns a structured response.

### Run the API

```bash
python main.py api
```

The API starts at `http://localhost:8000`. FastAPI auto-generates interactive docs at `http://localhost:8000/docs`.

---

## 13. Step 9: Streamlit Dashboard

**File:** `src/ui.py`

### What it does

A visual web dashboard that lets users enter a topic, run the pipeline, and see results in an interactive layout.

### How it connects to the API

```python
API_URL = os.getenv("API_URL", "http://localhost:8000")
```

- When running locally: calls `http://localhost:8000`
- When running in Docker: the `API_URL` environment variable overrides to `http://api:8000` (Docker service name)

### The dashboard layout

```
+--------------------------------------------------------------+
|  Multi-Agent Orchestration System                             |
|  Researcher -> Writer -> Reviewer pipeline powered by         |
|  LangGraph                                                    |
+--------------------------------------------------------------+
|                                                               |
|  SIDEBAR:                 MAIN AREA:                          |
|  +-----------------+      +----------------------------------+|
|  | How It Works    |      | Enter a topic to research:       ||
|  |                 |      | [________________________________]||
|  | 1. Researcher   |      |                                  ||
|  |    searches web |      | [Run Pipeline]                   ||
|  | 2. Writer makes |      |                                  ||
|  |    article      |      | Status: COMPLETE | Iterations: 1 ||
|  | 3. Reviewer     |      | Score: 8/10     | Time: 12.3s   ||
|  |    scores 1-10  |      |                                  ||
|  | 4. If < 7,      |      | Stage 1: Research                ||
|  |    revise       |      | [v] View research results        ||
|  |                 |      |   (expandable section)            ||
|  | Mode: Fallback  |      |                                  ||
|  | (no API keys)   |      | Stage 2: Draft Article           ||
|  +-----------------+      | [v] View written draft            ||
|                           |   (expandable section)            ||
|                           |                                  ||
|                           | Stage 3: Review Feedback          ||
|                           | [v] View review feedback          ||
|                           |   (expandable section)            ||
|                           |                                  ||
|                           | Agent Flow                        ||
|                           | [Research] [Writer] [Reviewer]    ||
|                           |   [Done]                          ||
|                           +----------------------------------+|
+--------------------------------------------------------------+
```

### User interaction flow

```
User types topic and clicks "Run Pipeline"
       |
       v
Progress bar appears: "Starting pipeline..."
       |
       v
httpx POST to {API_URL}/run with {"topic": "..."}
       |
       +-- Connection refused? --> Show error: "Cannot connect to API"
       |
       +-- Success? --> Continue
       |
       v
Progress bar: "Pipeline complete!"
       |
       v
Display 4 metric columns:
  - Status (COMPLETE / ERROR)
  - Iterations (1, 2, or 3)
  - Review Score (extracted from feedback text)
  - Time in seconds
       |
       v
Display 3 expandable sections:
  - Research results (markdown rendered)
  - Written draft (markdown rendered)
  - Review feedback (markdown rendered)
       |
       v
Display agent flow visualization (4 columns showing each stage)
```

### Why httpx instead of requests?

The UI uses `httpx` with a 120-second timeout. The pipeline can take 15-30 seconds with LLM calls, and `httpx` handles long-running requests well. The extended timeout prevents premature disconnection.

---

## 14. Step 10: Entry Point

**File:** `main.py`

### What it does

A unified command-line interface for running any part of the system.

### Commands reference

| Command | What It Does |
|---------|-------------|
| `python main.py api` | Start the FastAPI server on port 8000 (default if no argument) |
| `python main.py ui` | Start the Streamlit dashboard on port 8501 |
| `python main.py all` | Start both API and UI together |
| `python main.py run <topic>` | Run the pipeline directly in the terminal (no server) |

### How `main.py all` works

```
python main.py all
       |
       v
Start API as a subprocess (Popen -- non-blocking)
  uvicorn src.api:app --host 0.0.0.0 --port 8000
       |
       v
Start UI as a foreground process (run -- blocking)
  streamlit run src/ui.py --server.port 8501
       |
       v
When UI exits (user presses Ctrl+C):
  api_proc.terminate()   <-- clean up the API subprocess
```

The `try/finally` ensures the API process is killed even if the UI crashes.

### How `main.py run` works

This mode skips the API and UI entirely. It calls `run_pipeline()` directly and prints results to the terminal:

```bash
python main.py run Quantum Computing
```

**Output:**

```
============================================================
  Topic: Quantum Computing
  Status: complete
  Iterations: 1
============================================================

--- Research ---
Research Brief: Quantum Computing
========================================
(Fallback mode - raw search results)
...

--- Draft ---
# Quantum Computing

## Introduction
This report covers key findings on the topic of Quantum Computing...
...

--- Review ---
Score: 8/10
- Good length (245 words).
- Good use of section headings.
- Conclusion section present.
- Well-structured (5 paragraphs).
```

This is useful for quick testing without starting any servers.

---

## 15. Running the Full System

### Option A: Quick start (one command)

```bash
cd POCs/POC-05-LLM-Agent-Orchestration
pip install langgraph langchain langchain-core duckduckgo-search fastapi uvicorn streamlit httpx
python main.py all
```

This starts:
- API server on port 8000
- Streamlit dashboard on port 8501

Open `http://localhost:8501` in your browser.

### Option B: Step by step

```bash
# Step 1: Install dependencies
pip install langgraph langchain langchain-core duckduckgo-search fastapi uvicorn streamlit httpx

# Step 2: (Optional) Configure API keys for LLM-enhanced mode
cp .env.example .env
# Edit .env with your API key

# Step 3: Start the API (leave this running)
python main.py api

# Step 4: In a NEW terminal, start the dashboard
python main.py ui
```

### Option C: Terminal-only (no servers)

```bash
# Run the pipeline directly and see output in the terminal
python main.py run "Quantum Computing breakthroughs"
```

### Option D: Individual modules

```bash
# Start just the API with auto-reload for development
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload

# Start just the Streamlit UI
streamlit run src/ui.py --server.port 8501
```

---

## 16. Running with Docker

If you have Docker and Docker Compose installed, you can run everything in containers.

### Build and start

```bash
cd POCs/POC-05-LLM-Agent-Orchestration
docker-compose up --build
```

### What happens during build

1. Docker pulls `python:3.11-slim` base image
2. Installs all packages from `requirements.txt`
3. Copies `src/` into the container
4. Two containers start:
   - **api** (port 8000): FastAPI server with health check
   - **ui** (port 8501): Streamlit dashboard, depends on API being healthy

### Docker Compose architecture

```
+---------------------------------+    +---------------------------------+
|  api container                  |    |  ui container                   |
|                                 |    |                                 |
|  uvicorn src.api:app            |    |  streamlit run src/ui.py        |
|  --host 0.0.0.0 --port 8000    |    |  --server.port 8501             |
|                                 |    |                                 |
|  Health check:                  |    |  API_URL=http://api:8000        |
|  httpx.get(localhost:8000/      |    |  (Docker internal networking)   |
|    health) every 10s            |    |                                 |
|                                 |    |  depends_on: api                |
+-----------+---------------------+    +-----------+---------------------+
            |                                      |
            | port 8000                            | port 8501
            v                                      v
   http://localhost:8000                  http://localhost:8501
```

### Passing API keys to Docker

```bash
# Option 1: Environment variables
OPENAI_API_KEY=sk-... docker-compose up --build

# Option 2: .env file (docker-compose reads it automatically)
echo "OPENAI_API_KEY=sk-..." > .env
docker-compose up --build
```

### Access the services

| Service | URL |
|---------|-----|
| API | http://localhost:8000 |
| API Docs | http://localhost:8000/docs |
| Dashboard | http://localhost:8501 |

### Stop everything

```bash
docker-compose down
```

---

## 17. Testing the API Manually

### Using the interactive docs

Open `http://localhost:8000/docs` in your browser. FastAPI generates a Swagger UI where you can click "Try it out" on any endpoint.

### Using curl

```bash
# Health check
curl http://localhost:8000/health

# Expected response:
# {"status":"ok","service":"agent-orchestration"}

# Run the full pipeline
curl -X POST http://localhost:8000/run \
  -H "Content-Type: application/json" \
  -d '{"topic": "Quantum Computing breakthroughs in 2025"}'

# Expected response (abbreviated):
# {
#   "topic": "Quantum Computing breakthroughs in 2025",
#   "research": "Research Brief: ...",
#   "draft": "# Quantum Computing...",
#   "review_feedback": "Score: 8/10\n...",
#   "iteration": 1,
#   "status": "complete",
#   "elapsed_seconds": 15.23
# }
```

### Using Python

```python
import httpx

# Run the pipeline
response = httpx.post(
    "http://localhost:8000/run",
    json={"topic": "Quantum Computing breakthroughs in 2025"},
    timeout=120.0,  # Pipeline can take 15-30 seconds
)
data = response.json()

print(f"Status: {data['status']}")
print(f"Iterations: {data['iteration']}")
print(f"Time: {data['elapsed_seconds']}s")
print(f"\n--- Draft ---\n{data['draft'][:500]}...")
```

### Using the terminal (no server needed)

```bash
python main.py run "The future of renewable energy"
```

---

## 18. How Data Flows Through the System

### Complete pipeline flow (single iteration, approved on first review)

```
User input: "Quantum Computing"
       |
       v
run_pipeline("Quantum Computing")
       |
       v
Initial state created:
  {
    messages: [],
    topic: "Quantum Computing",
    research: "",
    draft: "",
    review_feedback: "",
    iteration: 0,
    status: "researching"
  }
       |
       v
+----------------------------------------------+
| RESEARCHER NODE                              |
|                                              |
|  1. Read topic from state                    |
|  2. web_search("Quantum Computing")          |
|     --> 3 results with titles and snippets   |
|  3. web_search("Quantum Computing latest     |
|     developments")                           |
|     --> 3 more results                       |
|  4. web_search("Quantum Computing key facts  |
|     and statistics")                         |
|     --> 3 more results                       |
|  5. Combine all 9 results into raw text      |
|  6. If LLM: summarize into bullet points    |
|     If no LLM: format with header            |
|  7. Return {research: "...", status:          |
|     "writing"}                               |
+----------------------------------------------+
       |
       | State now has: research = "Research Brief: ..."
       v
+----------------------------------------------+
| WRITER NODE                                  |
|                                              |
|  1. Read topic, research, feedback from      |
|     state                                    |
|  2. feedback is empty (first iteration)      |
|  3. If LLM: generate article from research   |
|     If no LLM: build template article        |
|       - Introduction from topic              |
|       - Key Findings from research lines     |
|       - Analysis paragraph                   |
|       - Conclusion paragraph                 |
|  4. Return {draft: "# Quantum Computing\n    |
|     ...", status: "reviewing"}               |
+----------------------------------------------+
       |
       | State now has: draft = "# Quantum Computing\n..."
       v
+----------------------------------------------+
| REVIEWER NODE                                |
|                                              |
|  1. Read draft from state                    |
|  2. iteration = 0 + 1 = 1                   |
|  3. If LLM: ask for JSON review              |
|     If no LLM: heuristic scoring             |
|       - Word count > 200? (+1)               |
|       - 3+ headings? (+1)                    |
|       - "conclusion" present? (+1)           |
|       - 4+ paragraphs? (+1)                  |
|       - Base score: 5 + bonuses = 8          |
|  4. Score 8 >= 7, so approved = True         |
|  5. status = "complete"                      |
|  6. Return {review_feedback: "Score: 8/10    |
|     \n...", iteration: 1, status: "complete"}|
+----------------------------------------------+
       |
       | State now has: status = "complete"
       v
Conditional edge: _should_revise(state)
  state["status"] == "complete" --> return "end"
       |
       v
END --> return final state to caller
```

### Pipeline flow with revision (score too low)

```
       ... (Researcher and Writer run as above) ...
       |
       v
+----------------------------------------------+
| REVIEWER NODE (iteration 1)                  |
|  Score: 5/10                                 |
|  Feedback: "Short article. Add more          |
|    headings. Missing conclusion."            |
|  approved = False (5 < 7)                    |
|  status = "writing"                          |
+----------------------------------------------+
       |
       | _should_revise --> "writer" (not complete)
       v
+----------------------------------------------+
| WRITER NODE (revision)                       |
|                                              |
|  1. Read feedback: "Short article..."        |
|  2. If LLM: prompt includes "Previous        |
|     review feedback to address: ..."         |
|     If no LLM: adds revision note to article |
|  3. Return {draft: revised article}          |
+----------------------------------------------+
       |
       v
+----------------------------------------------+
| REVIEWER NODE (iteration 2)                  |
|  Score: 8/10                                 |
|  approved = True                             |
|  status = "complete"                         |
+----------------------------------------------+
       |
       v
END
```

### API and UI flow

```
User types topic in Streamlit UI
       |
       v
Streamlit sends POST /run to FastAPI
  httpx.post("http://localhost:8000/run", json={"topic": "..."})
       |
       v
FastAPI calls run_pipeline(topic)
  (entire Researcher -> Writer -> Reviewer flow runs)
       |
       v
FastAPI returns JSON with all fields + elapsed_seconds
       |
       v
Streamlit receives response
       |
       v
Streamlit renders:
  - 4 metric cards (status, iterations, score, time)
  - 3 expandable sections (research, draft, feedback)
  - Agent flow visualization
```

---

## 19. Key Concepts Explained

### What is an "agent" in this system?

In AI, an "agent" is software that can:
1. Perceive its environment (read state)
2. Make decisions (choose actions based on data)
3. Act (modify state, call tools)

Our agents are Python functions that read shared state, do work (search, write, review), and return updated state. They are "agents" because they make autonomous decisions -- the Researcher decides which queries to run, the Writer decides how to structure the article, and the Reviewer decides whether quality is sufficient.

### What is a state machine?

A state machine is a model of computation with:

```
+-------------+    event    +-------------+    event    +-------------+
|   State A   |------------>|   State B   |------------>|   State C   |
+-------------+             +-------------+             +-------------+
                                   |
                                   | event (loop)
                                   v
                            +-------------+
                            |   State B   |
                            +-------------+
```

In our system:
- **States**: researching, writing, reviewing, complete
- **Transitions**: researcher finishes (researching --> writing), writer finishes (writing --> reviewing), reviewer approves (reviewing --> complete), reviewer rejects (reviewing --> writing)
- **Shared data**: the AgentState TypedDict accumulates information at each step

### How conditional edges work

A conditional edge is an edge where the next node depends on the current state. In LangGraph:

```python
graph.add_conditional_edges(
    "reviewer",                         # Source node
    _should_revise,                     # Decision function
    {"writer": "writer", "end": END},   # Mapping: return value -> target node
)
```

After the "reviewer" node runs, LangGraph calls `_should_revise(state)`. This function reads `state["status"]` and returns either `"writer"` or `"end"`. The mapping dict translates these strings to actual graph targets.

```
_should_revise returns "writer"  -->  go to "writer" node
_should_revise returns "end"     -->  go to END (stop pipeline)
```

This is how the revision loop works. Without conditional edges, the graph would be linear (researcher --> writer --> reviewer --> done) with no possibility of looping.

### The fallback mode pattern

Every agent follows the same pattern:

```python
llm = get_llm()
if llm:
    # LLM-enhanced mode: use the model
    response = llm.invoke(prompt)
    result = response.content
else:
    # Fallback mode: use heuristics
    result = _fallback_function(inputs)
```

This design has several benefits:

1. **Zero-config startup**: Users can run the project immediately without obtaining API keys
2. **Graceful degradation**: If an API key expires or the LLM service is down, the system still works
3. **Cost control**: Fallback mode costs nothing (no API calls)
4. **Testing**: You can test the pipeline mechanics without spending money on LLM calls

### What is CORS and why does the API need it?

CORS (Cross-Origin Resource Sharing) is a browser security feature. When the Streamlit UI (running on `localhost:8501`) makes an HTTP request to the API (running on `localhost:8000`), the browser considers this a "cross-origin" request because the ports differ.

Without CORS middleware, the browser would block the request. The middleware adds headers to the API's responses telling the browser "yes, requests from other origins are allowed."

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],     # Allow any origin
    allow_methods=["*"],     # Allow any HTTP method
    allow_headers=["*"],     # Allow any headers
)
```

In production, you would restrict `allow_origins` to your specific frontend domain.

### How the Researcher's triple-search strategy improves results

A single search query often returns a narrow view. By running three searches with different angles, we get broader coverage:

```
Search 1: "{topic}"
  --> Wikipedia-style overviews, general articles
  --> Provides: definitions, background, context

Search 2: "{topic} latest developments"
  --> News articles, recent announcements
  --> Provides: what is happening right now

Search 3: "{topic} key facts and statistics"
  --> Data-heavy articles, reports
  --> Provides: numbers, market size, adoption rates
```

The Writer then has three categories of source material to work with, producing a more comprehensive article than a single search could support.

### How the fallback reviewer scores

The heuristic scoring system measures four qualities of a well-structured article:

```
                          Points
                          Possible   How Measured
Base score:               5          (everyone starts here)
Length:                    +1         word_count > 200
Structure (headings):     +1         3+ "#" characters in draft
Conclusion:               +1         "conclusion" appears in text (case-insensitive)
Paragraph depth:          +1         4+ paragraphs with 30+ characters each
                          ---
Maximum possible:         9          (capped at 10)
Minimum possible:         5          (base score, no bonuses)
```

Since the fallback Writer always produces an article with headings, a conclusion section, and multiple paragraphs, a well-functioning fallback pipeline typically scores 8-9 and gets approved on the first iteration.

### What is `add_messages` and why is it special?

Most state fields use "replace" semantics -- when a node returns `{"draft": "new text"}`, the old draft is overwritten. But `messages` uses "append" semantics via the `add_messages` reducer:

```python
messages: Annotated[list, add_messages]
```

This means:
- Node A returns `{"messages": [HumanMessage("hello")]}`
- State becomes `{"messages": [HumanMessage("hello")]}`
- Node B returns `{"messages": [AIMessage("hi back")]}`
- State becomes `{"messages": [HumanMessage("hello"), AIMessage("hi back")]}`

The message history grows over time, which is useful for maintaining conversation context. In this project, the agents do not heavily use the messages field (they communicate via the dedicated research/draft/feedback fields), but the infrastructure is there for future enhancements like conversational refinement.

---

## 20. Troubleshooting

### "ModuleNotFoundError: No module named 'langgraph'"

You need to install the dependencies:

```bash
pip install langgraph langchain langchain-core duckduckgo-search fastapi uvicorn streamlit httpx
```

If you want LLM support:

```bash
pip install langchain-openai langchain-anthropic
```

### "ModuleNotFoundError: No module named 'src'"

You must run commands from the project root directory:

```bash
cd POCs/POC-05-LLM-Agent-Orchestration
python main.py run "AI agents"    # correct
```

Not from inside `src/`:

```bash
cd src
python orchestrator.py            # WRONG - will fail
```

### "Cannot connect to API" in the Streamlit dashboard

The API must be running before you start the dashboard:

```bash
# Terminal 1
python main.py api        # Start this first, wait until you see "Uvicorn running"

# Terminal 2
python main.py ui         # Then start this
```

Or use `python main.py all` to start both.

### "Search failed (RatelimitException): ..."

DuckDuckGo rate-limits frequent requests. This is not a crash -- the Researcher catches the error and includes it in the results. Possible fixes:

1. Wait a minute and try again
2. The pipeline still works -- the Writer uses whatever results were successful
3. If all searches fail, the offline fallback provides placeholder content

### Pipeline returns "error" status

Check the `review_feedback` field in the response -- it contains the full traceback:

```bash
curl -X POST http://localhost:8000/run \
  -H "Content-Type: application/json" \
  -d '{"topic": "test"}' | python -m json.tool
```

Look at the `review_feedback` field for the error message and traceback.

### LLM mode is not working despite having an API key

Make sure the key is in your environment:

```bash
# Check if the variable is set
echo $OPENAI_API_KEY

# If empty, export it
export OPENAI_API_KEY=sk-your-key-here

# Then restart the server
python main.py api
```

If using a `.env` file, make sure you have `python-dotenv` installed and the file is in the project root.

### Port already in use

If port 8000 or 8501 is busy:

```bash
# Find what is using the port
lsof -i :8000

# Kill it (replace PID with the actual number)
kill <PID>
```

Or use a different port:

```bash
uvicorn src.api:app --port 8001
API_URL=http://localhost:8001 streamlit run src/ui.py --server.port 8502
```

### Docker build fails

Make sure Docker is running, then:

```bash
docker-compose down        # Clean up
docker-compose up --build  # Rebuild from scratch
```

### The pipeline takes too long

Typical timings:

| Mode | Expected Time |
|------|--------------|
| Fallback (no LLM) | 3-10 seconds (mostly web search time) |
| LLM-enhanced, 1 iteration | 15-30 seconds |
| LLM-enhanced, 3 iterations | 30-90 seconds |

If it takes longer, the web search may be slow. The UI has a 120-second timeout, which should be sufficient.

### Review always scores 5 in fallback mode

The fallback reviewer starts at score 5 and adds points for specific qualities. If the Writer's fallback article is missing headings, a conclusion, or has too few paragraphs, the score stays low. This triggers the revision loop, but since the fallback Writer produces a consistent template, the second draft is usually identical to the first. After 3 iterations, the pipeline forces approval.

This is expected behavior in fallback mode. For more meaningful reviews, add an LLM API key.

---

## 21. Interview Questions

*Situation-based and technical questions from AI Engineer, ML Platform, and Generative AI interviews. Sourced from LinkedIn posts, Glassdoor reports, and engineering community discussions at AI-first companies.*

---

### Situational / Behavioral Questions

**Q: "Your multi-agent pipeline produces inconsistent review quality — sometimes the Reviewer approves weak articles, sometimes it rejects good ones. How do you make agent behavior more deterministic?"**

A: LLM non-determinism has three sources: temperature, prompt variability, and model version drift. Fix each layer: (1) **Temperature discipline** — set `temperature=0.0` for the Reviewer (scoring requires consistency, not creativity) and `temperature=0.3` for the Writer (some creative variance is acceptable). The Researcher can also use `temperature=0.0` since it's summarizing, not generating. (2) **Structured output** — instead of parsing a score from free-text review feedback, use JSON mode or a Pydantic model: `{"score": 8, "approved": true, "strengths": ["..."], "issues": ["..."]}`. The conditional edge reads `state["score"]` directly rather than using regex on `review_feedback`. (3) **Explicit rubric in the Reviewer's prompt** — rather than "review the article," provide a scoring grid: "Award 1 point each for: 200+ words, 3+ section headings, quantitative data present, conclusion section, clear thesis. Score = sum of points. Approve if score >= 4." This makes the scoring rubric explicit and reproducible. (4) **Golden test set in CI** — maintain 20 sample articles with known expected scores. Run these in CI on every prompt change. Alert if any expected score changes by more than 1 point.

**Q: "A customer-facing agent started giving incorrect financial advice due to hallucinations. Walk through how you'd add guardrails to this LangGraph pipeline."**

A: Four guardrail layers integrated into the LangGraph graph: (1) **Input classifier node** — before the Researcher, add a `classify_query` node. A lightweight classifier (rule-based or small LLM call) categorizes the query. If it matches "financial advice" patterns, route to a compliance-reviewed FAQ lookup instead of the research pipeline. Add this as a conditional edge from START. (2) **Context-grounded generation** — modify the Writer's prompt to require citations: "Every factual claim must be supported by information from the research brief. Do not introduce facts from general knowledge." Add a post-write `fact_check_draft` node that uses another LLM call to verify each sentence in the draft is supported by a sentence in the research brief. (3) **Output guardrail node** — add a `safety_check` node between the Writer and Reviewer. Use a classifier (Perspective API, or a fine-tuned model) to detect advice patterns, regulatory red flags, or harmful content. If flagged, return a structured refusal rather than the draft. (4) **Human escalation path** — add `requires_human_review: bool` to `AgentState`. The Reviewer sets it `True` for any output above a risk threshold. The pipeline pauses at a `human_review_gate` node that writes to a review queue and waits for approval before proceeding to END.

**Q: "Your LangGraph pipeline sometimes runs 20+ iterations and times out in production. How do you debug this?"**

A: Step 1 — instrument the state. Add `logger.info(f"iteration={state['iteration']}, status={state['status']}, score={state.get('score', 'N/A')}")` after every node. Run the pipeline on the failing input and inspect the state transition log. You'll immediately see whether the Reviewer keeps returning `status="writing"` instead of `status="complete"`. Step 2 — identify the root cause. Common culprits: (a) The conditional edge function `_should_revise` has a bug — `state["status"] == "complete"` never matches due to a whitespace or case difference. Verify with `assert state["status"] in {"complete", "writing"}`. (b) In LLM mode, the Reviewer's prompt is miscalibrated — it consistently scores articles 5/10 even for good quality. Log the raw LLM response before parsing. (c) The max iterations guard (`if iteration >= 3: return "end"`) is in the wrong place — the iteration counter isn't being incremented correctly. Step 3 — hardcode the safety cap. Add `if state["iteration"] >= 5: force_status = "complete"` as the first line of the `_should_revise` function regardless of the actual status. This is a belt-and-suspenders guard against any future logic errors.

---

### Technical Deep-Dive Questions

**Q: "Why use LangGraph's state machine vs a simple while loop with LLM calls? What concrete problems does it solve?"**

A: A while loop solves the happy path. LangGraph's value is in production resilience: (1) **Checkpointing** — LangGraph persists state after each node to a database (SQLite, PostgreSQL, or Redis). If the pipeline crashes mid-run (LLM timeout at the Writer step), it resumes from the last checkpoint — the Researcher's work is not lost. A while loop starts over. For pipelines with expensive steps (web search + LLM summarization = 30 seconds), this matters. (2) **Streaming** — `app.stream(initial_state)` yields state updates after each node completes. The UI can show "Researcher is gathering information..." before the Writer starts, then "Writer is drafting..." in real-time. A while loop blocks until the final result. (3) **Parallel execution** — LangGraph supports fan-out via the `Send` API: dispatch the Researcher, a fact-checker, and a citation generator to run concurrently, then merge results. A sequential while loop can't parallelize without custom threading. (4) **Observability** — LangGraph integrates with LangSmith for distributed tracing: every node's input state, output state, and latency is recorded. Debugging a while loop requires custom logging that reinvents this wheel.

**Q: "How does TypedDict-based AgentState prevent production bugs in multi-agent systems?"**

A: Without TypedDict, nodes return arbitrary dicts and downstream nodes hope the keys match. Classic bug: the Researcher writes `state["research_brief"]` but the Writer reads `state["research"]` — silent KeyError or wrong value consumed. TypedDict catches this at development time: `AgentState = TypedDict("AgentState", {"research": str, "draft": str, ...})`. Your IDE flags `state["research_brief"]` as an invalid key before any test runs. The `Annotated[list, add_messages]` pattern goes further — it declares not just the type but the merge semantics. LangGraph reads the `add_messages` annotation to know this field should be appended, not replaced. Other fields use replace semantics by default. Without this annotation, two nodes returning `{"messages": [...]}` would clobber each other's messages. Making merge semantics explicit in the type system is what allows multi-agent state sharing without silent data loss.

**Q: "How would you add parallel execution to this pipeline — having the Researcher run 3 searches concurrently instead of sequentially?"**

A: LangGraph's `Send` API enables fan-out. Refactor: (1) Add three `search_node_N` functions, each running one search query. Each returns `{"search_result_N": "..."}` as a partial state update. (2) In the orchestrator, replace the sequential search loop with: `graph.add_conditional_edges("dispatcher", lambda s: [Send("search_1", s), Send("search_2", s), Send("search_3", s)])`. LangGraph executes all three nodes concurrently. (3) Add a `merge_search_results` node that reads `state["search_result_1"]`, `state["search_result_2"]`, `state["search_result_3"]` and combines them into `state["research"]`. (4) For the underlying HTTP requests within each search node, use `async def search_node(state)` with `asyncio.gather()` to make sub-requests concurrent too. Result: the three 5-second searches that currently take 15 seconds sequentially complete in ~6 seconds concurrently. This pattern scales to any number of parallel information-gathering steps.

---

### System Design Questions

**Q: "Design a multi-agent code review system where agents review PRs for security vulnerabilities, performance issues, and style violations."**

A: Five-node LangGraph pipeline: (1) **Dispatcher node** — receives the PR diff, uses an LLM to split it into logical hunks (by function or class) and classifies each hunk: "this touches authentication → send to Security Agent," "this has nested loops → send to Performance Agent," "all hunks → Style Agent." Returns a routing plan. (2) **Specialist agents** (Security, Performance, Style) — run in parallel via `Send` API on their classified hunks. Each returns `List[ReviewComment]` with fields: `file`, `line`, `severity` (error/warning/info), `message`, `suggested_fix`. (3) **Aggregator node** — merges all comments, deduplicates overlapping feedback (Security and Style both flagging the same import), ranks by severity. (4) **Summary node** — generates a PR review summary: blocker count, must-fix vs. nice-to-have, overall risk assessment. (5) **Publisher node** — posts the structured review to GitHub via the GitHub API as a function tool. State: `AgentState = TypedDict(diff, security_comments, perf_comments, style_comments, aggregated_comments, final_summary, github_pr_url)`. Hard time limit: if any specialist agent exceeds 60 seconds (LLM timeout), return partial results with a `timed_out_agents` list.

**Q: "How would you make this pipeline resilient when the primary LLM API is unavailable during a critical workflow?"**

A: Defense in depth with four layers: (1) **Model cascade** — exactly as this POC implements: try GPT-4 → Claude Sonnet → extractive/heuristic fallback. Each node tries the cascade independently. If OpenAI is down for the Researcher, it doesn't block the Writer from trying GPT-4 on its own step. (2) **Circuit breaker pattern** — after 3 consecutive HTTP errors from OpenAI, open the circuit for 60 seconds and immediately route to the next model in the cascade. Prevents cascading timeouts when a service is clearly unavailable. Implement with `pybreaker` or a simple counter + `time.monotonic()` timestamp. (3) **Async timeout** — wrap every LLM call: `await asyncio.wait_for(llm_call(), timeout=30.0)`. A hung LLM call doesn't block the entire pipeline. Timeout counts as a failure and triggers the cascade. (4) **Checkpoint-based resume** — with LangGraph's SQLite checkpointer, a pipeline that fails at the Writer step (due to an LLM outage) can be resumed once the service recovers. The Researcher's expensive web search work is preserved. Add a `resume_from_checkpoint(run_id)` endpoint to the API so operators can recover in-progress pipelines without restarting from scratch.
