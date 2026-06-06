"""Researcher agent: gathers information on a topic via web search."""

import logging

from src.llm import get_llm
from src.tools import web_search

logger = logging.getLogger(__name__)


def _search_topic(topic: str) -> str:
    """Run multiple search queries and combine results.

    Args:
        topic: The subject to search for.

    Returns:
        Combined search results or an offline fallback message.
    """
    queries = [
        topic,
        f"{topic} latest developments",
        f"{topic} key facts and statistics",
    ]
    all_results: list[str] = []
    for q in queries:
        logger.debug("Running search query: %s", q)
        result = web_search.invoke(q)
        all_results.append(f"Query: {q}\n{result}")

    # Offline fallback: if every search failed, return canned data
    if all("Search failed" in r for r in all_results):
        logger.warning("All searches failed for topic '%s'; using offline fallback", topic)
        return (
            f"[Offline mode] {topic} is a rapidly evolving field. "
            "Key areas include architecture patterns, best practices, "
            "and production considerations. Further research recommended "
            "when internet is available."
        )

    return "\n\n---\n\n".join(all_results)


def research(state: dict) -> dict:
    """Researcher agent node: searches the web and summarizes findings.

    Args:
        state: Current AgentState with 'topic' populated.

    Returns:
        Updated state with 'research' and 'status' fields.
    """
    topic = state["topic"]
    logger.info("Researcher agent started for topic: %s", topic)

    raw_results = _search_topic(topic)

    llm = get_llm()
    if llm:
        logger.info("Summarizing research with LLM")
        prompt = (
            f"You are a research analyst. Summarize the following search results "
            f"about '{topic}' into a clear, structured research brief with key "
            f"findings, facts, and statistics. Use bullet points.\n\n"
            f"Search Results:\n{raw_results}"
        )
        response = llm.invoke(prompt)
        research_output: str = response.content
    else:
        logger.info("Using fallback mode for research formatting")
        research_output = (
            f"Research Brief: {topic}\n"
            f"{'=' * 40}\n"
            f"(Fallback mode - raw search results)\n\n"
            f"{raw_results}"
        )

    logger.info(
        "Researcher agent completed: output_length=%d, transitioning to 'writing'",
        len(research_output),
    )

    return {
        "research": research_output,
        "status": "writing",
        "iteration": state.get("iteration", 0),
    }
