"""Reviewer agent: evaluates draft quality and provides feedback."""

import json
import logging

from src.llm import get_llm
from src.settings import APPROVAL_THRESHOLD, MAX_ITERATIONS

logger = logging.getLogger(__name__)


def _fallback_review(draft: str) -> dict[str, int | str | bool]:
    """Basic heuristic review when no LLM is available.

    Args:
        draft: The article draft text to evaluate.

    Returns:
        Dict with 'score' (int), 'feedback' (str), and 'approved' (bool).
    """
    score = 5
    feedback_items: list[str] = []

    word_count = len(draft.split())
    if word_count > 200:
        score += 1
        feedback_items.append(f"Good length ({word_count} words).")
    else:
        feedback_items.append(f"Short article ({word_count} words). Consider expanding.")

    has_headings = draft.count("#") >= 3
    if has_headings:
        score += 1
        feedback_items.append("Good use of section headings.")
    else:
        feedback_items.append("Add more section headings for structure.")

    has_conclusion = "conclusion" in draft.lower()
    if has_conclusion:
        score += 1
        feedback_items.append("Conclusion section present.")
    else:
        feedback_items.append("Missing conclusion section.")

    paragraph_count = len([p for p in draft.split("\n\n") if len(p.strip()) > 30])
    if paragraph_count >= 4:
        score += 1
        feedback_items.append(f"Well-structured ({paragraph_count} paragraphs).")
    else:
        feedback_items.append("Add more detailed paragraphs.")

    # Cap score at 10
    score = min(score, 10)

    return {
        "score": score,
        "feedback": "\n".join(f"- {item}" for item in feedback_items),
        "approved": score >= APPROVAL_THRESHOLD,
    }


def review(state: dict) -> dict:
    """Reviewer agent node: evaluates the draft and returns feedback.

    Args:
        state: Current AgentState with 'draft' populated.

    Returns:
        Updated state with 'review_feedback', 'status', and 'iteration'.
    """
    draft = state["draft"]
    iteration: int = state.get("iteration", 0) + 1

    logger.info("Reviewer agent started: iteration=%d/%d", iteration, MAX_ITERATIONS)

    llm = get_llm()
    if llm:
        prompt = (
            "You are a content reviewer. Evaluate this article draft.\n"
            "Respond with valid JSON only: "
            '{"score": <1-10>, "feedback": "<detailed feedback>", '
            '"approved": <true/false>}\n'
            f"Score {APPROVAL_THRESHOLD}+ means approved.\n\n"
            f"Draft:\n{draft}"
        )
        logger.info("Invoking LLM for review evaluation")
        response = llm.invoke(prompt)
        try:
            result: dict = json.loads(response.content)
            logger.info("LLM review parsed successfully")
        except json.JSONDecodeError:
            logger.warning("LLM response was not valid JSON; falling back to heuristic review")
            result = _fallback_review(draft)
    else:
        logger.info("Using fallback heuristic review")
        result = _fallback_review(draft)

    score: int = result.get("score", 5)
    feedback: str = result.get("feedback", "No feedback.")
    approved: bool = result.get("approved", score >= APPROVAL_THRESHOLD)

    # After max iterations, force approval
    if iteration >= MAX_ITERATIONS:
        logger.info("Max iterations (%d) reached; forcing approval", MAX_ITERATIONS)
        approved = True

    if approved:
        next_status = "complete"
    else:
        next_status = "writing"

    logger.info(
        "Reviewer agent completed: score=%d/%d, approved=%s, next_status='%s'",
        score,
        APPROVAL_THRESHOLD,
        approved,
        next_status,
    )

    return {
        "review_feedback": f"Score: {score}/10\n{feedback}",
        "iteration": iteration,
        "status": next_status,
    }
