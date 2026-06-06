"""Tests for the LangGraph orchestrator.

These tests run the REAL graph end-to-end in offline mode (no LLM mocks).
The conftest clears API keys so all agents exercise their fallback logic.
"""

from src.orchestrator import build_graph, run_pipeline
from src.settings import MAX_ITERATIONS


class TestBuildGraph:
    """Tests for graph construction."""

    def test_graph_compiles_successfully(self) -> None:
        """build_graph() should return a compiled graph without errors."""
        graph = build_graph()
        assert graph is not None

    def test_graph_has_expected_nodes(self) -> None:
        """The compiled graph should contain researcher, writer, and reviewer nodes."""
        graph = build_graph()
        graph_drawing = graph.get_graph()
        # Nodes may be objects with .id or plain strings depending on langgraph version
        nodes = graph_drawing.nodes
        if isinstance(nodes, dict):
            node_ids = set(nodes.keys())
        else:
            node_ids: set[str] = set()
            for node in nodes:
                node_ids.add(node.id if hasattr(node, "id") else str(node))
        assert "researcher" in node_ids
        assert "writer" in node_ids
        assert "reviewer" in node_ids

    def test_graph_entry_point_is_researcher(self) -> None:
        """The graph should start at the researcher node."""
        graph = build_graph()
        graph_drawing = graph.get_graph()
        # __start__ node should connect to researcher
        start_edges = [e for e in graph_drawing.edges if e.source == "__start__"]
        assert len(start_edges) == 1
        assert start_edges[0].target == "researcher"

    def test_graph_researcher_connects_to_writer(self) -> None:
        """researcher should have an edge to writer."""
        graph = build_graph()
        graph_drawing = graph.get_graph()
        researcher_edges = [e for e in graph_drawing.edges if e.source == "researcher"]
        targets = {e.target for e in researcher_edges}
        assert "writer" in targets

    def test_graph_writer_connects_to_reviewer(self) -> None:
        """writer should have an edge to reviewer."""
        graph = build_graph()
        graph_drawing = graph.get_graph()
        writer_edges = [e for e in graph_drawing.edges if e.source == "writer"]
        targets = {e.target for e in writer_edges}
        assert "reviewer" in targets


class TestRunPipeline:
    """Tests for end-to-end pipeline execution in offline mode.

    These run the REAL LangGraph state machine with real agent fallback
    logic -- no mocking of the orchestrator or agent functions.
    """

    def test_run_pipeline_returns_dict(self) -> None:
        """run_pipeline() should return a dict."""
        result = run_pipeline("test topic")
        assert isinstance(result, dict)

    def test_run_pipeline_has_all_state_keys(self) -> None:
        """The returned dict should have all AgentState keys."""
        result = run_pipeline("test topic")
        expected_keys = {"messages", "topic", "research", "draft", "review_feedback", "iteration", "status"}
        assert expected_keys.issubset(set(result.keys()))

    def test_run_pipeline_completes(self) -> None:
        """Pipeline should eventually reach 'complete' status."""
        result = run_pipeline("test topic")
        assert result["status"] == "complete"

    def test_run_pipeline_has_nonempty_outputs(self) -> None:
        """All main outputs should be populated."""
        result = run_pipeline("test topic")
        assert len(result["research"]) > 0
        assert len(result["draft"]) > 0
        assert len(result["review_feedback"]) > 0

    def test_run_pipeline_preserves_topic(self) -> None:
        """The topic should be preserved in the final state."""
        result = run_pipeline("Quantum Computing")
        assert result["topic"] == "Quantum Computing"

    def test_run_pipeline_iteration_at_least_one(self) -> None:
        """Pipeline should go through at least one review iteration."""
        result = run_pipeline("test topic")
        assert result["iteration"] >= 1

    def test_run_pipeline_max_iterations(self) -> None:
        """Pipeline should not exceed MAX_ITERATIONS review iterations."""
        result = run_pipeline("test topic")
        assert result["iteration"] <= MAX_ITERATIONS
