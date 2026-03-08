"""Hypothesis Engine Agents."""

from .base import BaseAgent
from .heuristic_agent import HeuristicScientist

# LLM agent is optional (requires openai package)
try:
    from .llm_agent import LLMScientist
except ImportError:
    LLMScientist = None

__all__ = ["BaseAgent", "HeuristicScientist", "LLMScientist"]
