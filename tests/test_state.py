"""Tests for the AgentState TypedDict."""

from typing import get_type_hints

from src.state import AgentState


class TestAgentState:
    """Tests for the AgentState TypedDict structure."""

    def test_agent_state_has_expected_keys(self) -> None:
        """AgentState should declare all seven expected fields."""
        hints = get_type_hints(AgentState, include_extras=True)
        expected_keys = {
            "messages",
            "topic",
            "research",
            "draft",
            "review_feedback",
            "iteration",
            "status",
        }
        assert set(hints.keys()) == expected_keys

    def test_agent_state_messages_field_exists(self) -> None:
        """The 'messages' field should be present with an Annotated type."""
        hints = get_type_hints(AgentState, include_extras=True)
        assert "messages" in hints

    def test_agent_state_string_fields(self) -> None:
        """topic, research, draft, review_feedback, status should be str."""
        hints = get_type_hints(AgentState)
        string_fields = ["topic", "research", "draft", "review_feedback", "status"]
        for field in string_fields:
            assert hints[field] is str, f"{field} should be str, got {hints[field]}"

    def test_agent_state_iteration_is_int(self) -> None:
        """iteration should be int."""
        hints = get_type_hints(AgentState)
        assert hints["iteration"] is int

    def test_state_initialization(self) -> None:
        """AgentState can be instantiated as a regular dict."""
        state: AgentState = {
            "messages": [],
            "topic": "test topic",
            "research": "",
            "draft": "",
            "review_feedback": "",
            "iteration": 0,
            "status": "researching",
        }
        assert state["topic"] == "test topic"
        assert state["iteration"] == 0
        assert state["status"] == "researching"
        assert state["messages"] == []
        assert state["research"] == ""
        assert state["draft"] == ""
        assert state["review_feedback"] == ""

    def test_state_is_dict_subclass(self) -> None:
        """AgentState instances are plain dicts (TypedDict is a dict)."""
        state: AgentState = {
            "messages": [],
            "topic": "t",
            "research": "",
            "draft": "",
            "review_feedback": "",
            "iteration": 0,
            "status": "researching",
        }
        assert isinstance(state, dict)
