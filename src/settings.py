"""Centralized configuration loaded from environment variables."""

import os

API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
API_PORT: int = int(os.getenv("API_PORT", "8000"))
UI_PORT: int = int(os.getenv("UI_PORT", "8501"))
CORS_ORIGINS: list[str] = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:8501").split(",")
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "")  # "openai", "anthropic", or "" for fallback
MAX_ITERATIONS: int = int(os.getenv("MAX_ITERATIONS", "3"))
APPROVAL_THRESHOLD: int = int(os.getenv("APPROVAL_THRESHOLD", "7"))
SEARCH_MAX_RESULTS: int = int(os.getenv("SEARCH_MAX_RESULTS", "5"))
