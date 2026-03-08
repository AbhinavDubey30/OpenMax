"""
Base Agent Interface for the Hypothesis Engine.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple


class BaseAgent(ABC):
    """Abstract base class for Hypothesis Engine agents."""

    @abstractmethod
    def act(self, observation: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
        """
        Decide on an action given the current observation.

        Args:
            observation: The observation dict from the environment.

        Returns:
            Tuple of (action_dict, reasoning_string)
            action_dict: The action to take (e.g., {"action": "experiment", "inputs": {"x": 3}})
            reasoning_string: Human-readable explanation of why this action was chosen.
        """
        pass

    @abstractmethod
    def reset(self):
        """Reset the agent's internal state for a new episode."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """The agent's display name."""
        pass
