"""Tools available to agents: web search and calculator."""

import logging

from langchain_core.tools import tool

from src.settings import SEARCH_MAX_RESULTS

logger = logging.getLogger(__name__)


@tool
def web_search(query: str) -> str:
    """Search the web using DuckDuckGo and return the top results.

    Args:
        query: The search query string.

    Returns:
        Formatted string with top search results.
    """
    logger.info("Web search query: %s (max_results=%d)", query, SEARCH_MAX_RESULTS)
    try:
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS

        results: list[str] = []
        with DDGS() as ddgs:
            for i, r in enumerate(ddgs.text(query, max_results=SEARCH_MAX_RESULTS)):
                results.append(
                    f"[{i + 1}] {r['title']}\n"
                    f"    {r['body']}\n"
                    f"    Source: {r['href']}"
                )
        if results:
            logger.info("Web search returned %d results for: %s", len(results), query)
            return "\n\n".join(results)
        logger.warning("No results found for: %s", query)
        return f"No results found for: {query}"
    except Exception as e:
        logger.error("Web search failed for '%s': %s: %s", query, type(e).__name__, e)
        return f"Search failed ({type(e).__name__}): {e}"


@tool
def calculate(expression: str) -> str:
    """Safely evaluate a math expression.

    Args:
        expression: A mathematical expression like '2 + 3 * 4'.

    Returns:
        The result as a string, or an error message.
    """
    logger.info("Calculate expression: %s", expression)
    allowed_chars = set("0123456789+-*/.() ")
    if not all(c in allowed_chars for c in expression):
        logger.warning("Invalid characters in expression: %s", expression)
        return f"Invalid characters in expression: {expression}"
    try:
        result = eval(expression, {"__builtins__": {}}, {})  # noqa: S307
        logger.info("Calculate result: %s = %s", expression, result)
        return str(result)
    except Exception as e:
        logger.error("Calculation error for '%s': %s", expression, e)
        return f"Calculation error: {e}"
