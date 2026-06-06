"""Shared LLM factory used by all agents."""

import logging
import os
from typing import Any

from src.settings import LLM_PROVIDER

logger = logging.getLogger(__name__)


def get_llm(temperature: float = 0) -> Any | None:
    """Try to create an LLM instance. Returns None if no API key.

    Provider selection order:
      1. If LLM_PROVIDER is set explicitly, use that provider.
      2. Otherwise, try OpenAI first, then Anthropic.
      3. Return None if no API keys are available (fallback mode).

    Args:
        temperature: Sampling temperature (0 = deterministic, higher = creative).

    Returns:
        A LangChain chat model instance, or None if no provider is available.
    """
    provider = LLM_PROVIDER.lower()

    if provider == "openai" or (not provider and os.getenv("OPENAI_API_KEY")):
        if os.getenv("OPENAI_API_KEY"):
            from langchain_openai import ChatOpenAI

            logger.info("Using OpenAI LLM (temperature=%.1f)", temperature)
            return ChatOpenAI(model="gpt-4o-mini", temperature=temperature)
        logger.warning("LLM_PROVIDER=openai but OPENAI_API_KEY not set")

    if provider == "anthropic" or (not provider and os.getenv("ANTHROPIC_API_KEY")):
        if os.getenv("ANTHROPIC_API_KEY"):
            from langchain_anthropic import ChatAnthropic

            logger.info("Using Anthropic LLM (temperature=%.1f)", temperature)
            return ChatAnthropic(model="claude-sonnet-4-20250514", temperature=temperature)
        logger.warning("LLM_PROVIDER=anthropic but ANTHROPIC_API_KEY not set")

    logger.info("No LLM provider configured; using fallback mode")
    return None
