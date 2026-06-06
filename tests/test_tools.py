"""Tests for agent tools (web_search, calculate).

DuckDuckGo mocks are used here because web_search makes real network calls.
The calculate tool is tested with real evaluation -- no mocks needed.
"""

from unittest.mock import MagicMock, patch

from src.tools import calculate, web_search


class TestWebSearch:
    """Tests for the web_search tool."""

    def test_web_search_returns_string(self) -> None:
        """web_search.invoke() should always return a string."""
        result = web_search.invoke("test query")
        assert isinstance(result, str)

    def test_web_search_handles_failure_gracefully(self) -> None:
        """When search raises an exception, it should return an error string, not raise."""
        mock_ddgs_instance = MagicMock()
        mock_ddgs_instance.__enter__ = MagicMock(return_value=mock_ddgs_instance)
        mock_ddgs_instance.__exit__ = MagicMock(return_value=False)
        mock_ddgs_instance.text.side_effect = RuntimeError("network error")

        with patch("duckduckgo_search.DDGS", return_value=mock_ddgs_instance):
            result = web_search.invoke("test query")
            assert isinstance(result, str)
            assert "Search failed" in result or "error" in result.lower()

    def test_web_search_with_mocked_results(self) -> None:
        """With mocked search results, output should include title and body."""
        mock_results = [
            {"title": "Result 1", "body": "Body of result 1", "href": "https://example.com/1"},
            {"title": "Result 2", "body": "Body of result 2", "href": "https://example.com/2"},
        ]

        mock_ddgs_instance = MagicMock()
        mock_ddgs_instance.__enter__ = MagicMock(return_value=mock_ddgs_instance)
        mock_ddgs_instance.__exit__ = MagicMock(return_value=False)
        mock_ddgs_instance.text.return_value = mock_results

        with patch("duckduckgo_search.DDGS", return_value=mock_ddgs_instance):
            result = web_search.invoke("test query")
            assert "Result 1" in result
            assert "Result 2" in result
            assert "example.com" in result

    def test_web_search_no_results(self) -> None:
        """When search returns empty results, should indicate no results found."""
        mock_ddgs_instance = MagicMock()
        mock_ddgs_instance.__enter__ = MagicMock(return_value=mock_ddgs_instance)
        mock_ddgs_instance.__exit__ = MagicMock(return_value=False)
        mock_ddgs_instance.text.return_value = []

        with patch("duckduckgo_search.DDGS", return_value=mock_ddgs_instance):
            result = web_search.invoke("obscure query no results")
            assert "No results found" in result


class TestCalculate:
    """Tests for the calculate tool."""

    def test_calculate_basic_addition(self) -> None:
        """Should handle basic addition."""
        result = calculate.invoke("2 + 3")
        assert result == "5"

    def test_calculate_multiplication(self) -> None:
        """Should handle multiplication."""
        result = calculate.invoke("4 * 5")
        assert result == "20"

    def test_calculate_complex_expression(self) -> None:
        """Should handle compound expressions."""
        result = calculate.invoke("(10 + 5) * 2")
        assert result == "30"

    def test_calculate_division(self) -> None:
        """Should handle division."""
        result = calculate.invoke("10 / 4")
        assert result == "2.5"

    def test_calculate_invalid_characters(self) -> None:
        """Should reject expressions with invalid characters."""
        result = calculate.invoke("import os")
        assert "Invalid characters" in result

    def test_calculate_division_by_zero(self) -> None:
        """Should handle division by zero gracefully."""
        result = calculate.invoke("1 / 0")
        assert "error" in result.lower()
