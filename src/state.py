"""Shared agent state for the LangGraph orchestration pipeline."""

import logging
from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    """State passed between agents in the orchestration graph.

    Fields:
        messages: Chat message history (accumulated via add_messages reducer).
        topic: The user-provided topic to research and write about.
        research: Raw research findings from the Researcher agent.
        draft: The written article/report from the Writer agent.
        review_feedback: Feedback from the Reviewer agent.
        iteration: Current revision iteration (starts at 0).
        status: Pipeline stage: researching | writing | reviewing | complete.
    """

    messages: Annotated[list, add_messages]
    topic: str
    research: str
    draft: str
    review_feedback: str
    iteration: int
    status: str
