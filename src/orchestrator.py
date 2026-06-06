"""LangGraph orchestrator: wires Researcher -> Writer -> Reviewer into a state machine."""

import logging
from typing import Any

from langgraph.graph import END, StateGraph

from src.agents.researcher import research
from src.agents.writer import write
from src.agents.reviewer import review
from src.state import AgentState

logger = logging.getLogger(__name__)


def _should_revise(state: AgentState) -> str:
    """Conditional edge: route back to writer if not approved, else finish.

    Args:
        state: Current agent state with 'status' field.

    Returns:
        'end' if the pipeline is complete, 'writer' to revise.
    """
    decision = "end" if state["status"] == "complete" else "writer"
    logger.info("Routing decision after review: %s", decision)
    return decision


def build_graph() -> Any:
    """Build and compile the agent orchestration graph.

    Graph structure:
        START -> researcher -> writer -> reviewer
        reviewer -> writer   (if not approved and iterations remain)
        reviewer -> END      (if approved or max iterations reached)

    Returns:
        A compiled LangGraph that accepts AgentState.
    """
    logger.debug("Building orchestration graph")
    graph = StateGraph(AgentState)

    # Add agent nodes
    graph.add_node("researcher", research)
    graph.add_node("writer", write)
    graph.add_node("reviewer", review)

    # Edges: linear flow with a review loop
    graph.set_entry_point("researcher")
    graph.add_edge("researcher", "writer")
    graph.add_edge("writer", "reviewer")
    graph.add_conditional_edges(
        "reviewer",
        _should_revise,
        {"writer": "writer", "end": END},
    )

    compiled = graph.compile()
    logger.debug("Orchestration graph compiled successfully")
    return compiled


def run_pipeline(topic: str) -> dict:
    """Run the full agent pipeline for a given topic.

    Args:
        topic: The subject to research, write about, and review.

    Returns:
        Final AgentState dict with all intermediate outputs.
    """
    logger.info("Pipeline started for topic: '%s'", topic)
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
    result = dict(final_state)

    logger.info(
        "Pipeline completed: topic='%s', status='%s', iterations=%d",
        result.get("topic"),
        result.get("status"),
        result.get("iteration", 0),
    )

    return result
