"""Shared fixtures for LLM-Agent-Orchestration tests."""

import os
import sys
from pathlib import Path

import pytest

# Ensure project root is on sys.path so `src` imports work
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Force offline mode: clear LLM API keys so all agents use fallback paths
os.environ.pop("OPENAI_API_KEY", None)
os.environ.pop("ANTHROPIC_API_KEY", None)

# Configure logging for test runs
from src.logging_config import setup_logging  # noqa: E402

setup_logging()


@pytest.fixture()
def sample_topic() -> str:
    """A simple topic string for pipeline tests."""
    return "Artificial Intelligence"


@pytest.fixture()
def initial_state(sample_topic: str) -> dict:
    """A fully-initialized AgentState dict ready for the pipeline."""
    return {
        "messages": [],
        "topic": sample_topic,
        "research": "",
        "draft": "",
        "review_feedback": "",
        "iteration": 0,
        "status": "researching",
    }


@pytest.fixture()
def researched_state(initial_state: dict) -> dict:
    """State after the researcher has run (simulated)."""
    return {
        **initial_state,
        "research": (
            "Research Brief: Artificial Intelligence\n"
            "========================================\n"
            "(Fallback mode - raw search results)\n\n"
            "[Offline mode] Artificial Intelligence is a rapidly evolving field. "
            "Key areas include architecture patterns, best practices, "
            "and production considerations. Further research recommended "
            "when internet is available."
        ),
        "status": "writing",
    }


@pytest.fixture()
def written_state(researched_state: dict) -> dict:
    """State after the writer has run (simulated)."""
    return {
        **researched_state,
        "draft": (
            "# Artificial Intelligence\n\n"
            "## Introduction\n"
            "This report covers key findings on the topic of Artificial Intelligence, "
            "compiled from multiple web sources.\n\n"
            "## Key Findings\n"
            "1. AI is a rapidly evolving field.\n"
            "2. Key areas include architecture patterns and best practices.\n"
            "3. Production considerations are important.\n\n"
            "## Analysis\n"
            "Based on the research gathered, Artificial Intelligence is a multifaceted "
            "subject with several important dimensions worth exploring further.\n\n"
            "## Conclusion\n"
            "The research highlights important aspects of Artificial Intelligence that "
            "merit continued attention and deeper investigation."
        ),
        "status": "reviewing",
    }
