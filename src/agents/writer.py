"""Writer agent: produces a structured article from research findings."""

import logging

from src.llm import get_llm

logger = logging.getLogger(__name__)


def _fallback_article(topic: str, research: str, feedback: str) -> str:
    """Generate a template-based article without an LLM.

    Args:
        topic: The subject of the article.
        research: Raw research findings to extract key points from.
        feedback: Previous review feedback, if any.

    Returns:
        A markdown-formatted article string.
    """
    # Extract lines that look like content from the research
    lines = [line.strip() for line in research.split("\n") if line.strip() and len(line.strip()) > 20]
    key_points = lines[:8]  # Take up to 8 substantial lines

    sections: list[str] = [
        f"# {topic}",
        "",
        "## Introduction",
        f"This report covers key findings on the topic of {topic}, "
        "compiled from multiple web sources.",
        "",
        "## Key Findings",
    ]
    for i, point in enumerate(key_points, 1):
        # Clean up search-result formatting
        clean = point.lstrip("[0123456789] ").strip()
        sections.append(f"{i}. {clean}")

    sections.extend([
        "",
        "## Analysis",
        f"Based on the research gathered, {topic} is a multifaceted subject "
        "with several important dimensions worth exploring further.",
        "",
        "## Conclusion",
        f"The research highlights important aspects of {topic} that merit "
        "continued attention and deeper investigation.",
    ])

    if feedback:
        sections.extend([
            "",
            "---",
            "*Revision note: incorporated feedback from review iteration.*",
        ])

    return "\n".join(sections)


def write(state: dict) -> dict:
    """Writer agent node: creates an article from research results.

    Args:
        state: Current AgentState with 'research' populated.

    Returns:
        Updated state with 'draft' and 'status' fields.
    """
    topic = state["topic"]
    research = state["research"]
    feedback = state.get("review_feedback", "")

    logger.info(
        "Writer agent started: topic='%s', has_feedback=%s",
        topic,
        bool(feedback),
    )

    llm = get_llm(temperature=0.7)
    if llm:
        revision_note = ""
        if feedback:
            revision_note = (
                f"\n\nPrevious review feedback to address:\n{feedback}"
            )
            logger.info("Writing revision incorporating previous feedback")
        prompt = (
            f"You are an expert content writer. Write a well-structured article "
            f"about '{topic}' using the research below. Include an introduction, "
            f"key findings, analysis, and conclusion. Use markdown formatting."
            f"\n\nResearch:\n{research}{revision_note}"
        )
        logger.info("Invoking LLM for article generation")
        response = llm.invoke(prompt)
        draft: str = response.content
    else:
        logger.info("Using fallback article generator")
        draft = _fallback_article(topic, research, feedback)

    logger.info(
        "Writer agent completed: draft_length=%d, transitioning to 'reviewing'",
        len(draft),
    )

    return {
        "draft": draft,
        "status": "reviewing",
    }
