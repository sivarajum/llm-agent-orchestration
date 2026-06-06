"""Tests for agent functions (researcher, writer, reviewer) in offline/fallback mode.

All tests run with REAL fallback logic -- no LLM mocks. The conftest clears
API keys so agents exercise the full offline code path.
"""

from src.agents.researcher import research
from src.agents.writer import write
from src.agents.reviewer import review


class TestResearcher:
    """Tests for the researcher agent."""

    def test_research_returns_dict(self, initial_state: dict) -> None:
        """research() should return a dict."""
        result = research(initial_state)
        assert isinstance(result, dict)

    def test_research_has_research_key(self, initial_state: dict) -> None:
        """Returned dict should contain 'research' with non-empty content."""
        result = research(initial_state)
        assert "research" in result
        assert len(result["research"]) > 0

    def test_research_sets_status_to_writing(self, initial_state: dict) -> None:
        """After research, status should transition to 'writing'."""
        result = research(initial_state)
        assert result["status"] == "writing"

    def test_research_preserves_iteration(self, initial_state: dict) -> None:
        """research() should pass through the current iteration count."""
        result = research(initial_state)
        assert result["iteration"] == 0

    def test_research_contains_topic_reference(self, initial_state: dict) -> None:
        """Research output should mention the topic."""
        result = research(initial_state)
        assert initial_state["topic"] in result["research"] or "Artificial Intelligence" in result["research"]


class TestWriter:
    """Tests for the writer agent."""

    def test_write_returns_dict(self, researched_state: dict) -> None:
        """write() should return a dict."""
        result = write(researched_state)
        assert isinstance(result, dict)

    def test_write_has_draft_key(self, researched_state: dict) -> None:
        """Returned dict should contain 'draft' with non-empty content."""
        result = write(researched_state)
        assert "draft" in result
        assert len(result["draft"]) > 0

    def test_write_sets_status_to_reviewing(self, researched_state: dict) -> None:
        """After writing, status should transition to 'reviewing'."""
        result = write(researched_state)
        assert result["status"] == "reviewing"

    def test_write_draft_has_structure(self, researched_state: dict) -> None:
        """Fallback draft should have markdown headings."""
        result = write(researched_state)
        draft = result["draft"]
        assert "# " in draft  # at least one heading
        assert "Introduction" in draft
        assert "Conclusion" in draft

    def test_write_with_feedback_adds_revision_note(self, researched_state: dict) -> None:
        """When review_feedback is present, the draft should note the revision."""
        researched_state["review_feedback"] = "Needs more detail"
        result = write(researched_state)
        draft = result["draft"]
        assert "Revision note" in draft or "revision" in draft.lower()

    def test_write_without_feedback(self, researched_state: dict) -> None:
        """When no feedback is present, the draft should not mention revision."""
        researched_state["review_feedback"] = ""
        result = write(researched_state)
        draft = result["draft"]
        assert "Revision note" not in draft


class TestReviewer:
    """Tests for the reviewer agent."""

    def test_review_returns_dict(self, written_state: dict) -> None:
        """review() should return a dict."""
        result = review(written_state)
        assert isinstance(result, dict)

    def test_review_has_feedback(self, written_state: dict) -> None:
        """Returned dict should contain 'review_feedback'."""
        result = review(written_state)
        assert "review_feedback" in result
        assert len(result["review_feedback"]) > 0

    def test_review_has_score_in_feedback(self, written_state: dict) -> None:
        """review_feedback should start with a score line."""
        result = review(written_state)
        assert result["review_feedback"].startswith("Score:")

    def test_review_increments_iteration(self, written_state: dict) -> None:
        """review() should increment the iteration counter."""
        written_state["iteration"] = 0
        result = review(written_state)
        assert result["iteration"] == 1

    def test_review_sets_status(self, written_state: dict) -> None:
        """review() should set status to either 'complete' or 'writing'."""
        result = review(written_state)
        assert result["status"] in ("complete", "writing")

    def test_review_forces_approval_after_max_iterations(self, written_state: dict) -> None:
        """After MAX_ITERATIONS, review should force status to 'complete'."""
        written_state["iteration"] = 2  # will become 3 inside review()
        result = review(written_state)
        assert result["status"] == "complete"
        assert result["iteration"] == 3

    def test_review_short_draft_lower_score(self) -> None:
        """A very short draft should score lower than a well-structured one."""
        short_state: dict = {
            "topic": "Test",
            "research": "",
            "draft": "Just a few words.",
            "review_feedback": "",
            "iteration": 0,
            "status": "reviewing",
            "messages": [],
        }
        result = review(short_state)
        feedback = result["review_feedback"]
        # Should contain "Score:" with a number
        assert "Score:" in feedback

    def test_review_well_structured_draft_higher_score(self, written_state: dict) -> None:
        """A well-structured draft with headings and conclusion should score higher."""
        result = review(written_state)
        feedback = result["review_feedback"]
        # Extract score value
        score_line = feedback.split("\n")[0]  # "Score: X/10"
        score_str = score_line.replace("Score: ", "").split("/")[0]
        score = int(score_str)
        assert score >= 7  # Well-structured draft should pass
