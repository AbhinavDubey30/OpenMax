"""
LLM Scientist Agent for the Hypothesis Engine.

Uses OpenAI-compatible APIs (GPT-4, etc.) to act as a scientist.
The LLM receives observations and must decide actions following
the scientific method.
"""

import json
import os
from typing import Dict, Any, Tuple, Optional
from .base import BaseAgent

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


SYSTEM_PROMPT = """You are a brilliant scientist investigating an unknown system. 
Your goal is to discover the hidden mathematical relationship between inputs and outputs 
through careful experimentation, hypothesis formation, and prediction.

## Your Investigation Tools

You can take these actions (respond with EXACTLY one JSON action per turn):

1. **Run an Experiment**: Test specific input values
   {"action": "experiment", "inputs": {"x": 3.0}}
   
2. **Submit a Hypothesis**: State your mathematical model
   {"action": "hypothesize", "expression": "2*x + 3"}
   Supported: +, -, *, /, **, sin(), cos(), exp(), log(), sqrt(), abs()
   Conditionals: "(a) if x > 0 else (b)" or "where(x > 0, a, b)"
   
3. **Submit Predictions**: Predict outputs for test cases (ENDS the episode)
   {"action": "predict", "predictions": [1.0, 2.0, 3.0, ...]}
   
4. **Get Hint**: Get a hint (small penalty)
   {"action": "get_hint"}

## Strategy Guidelines

1. START by running strategic experiments (probe key points like 0, 1, -1, etc.)
2. ANALYZE patterns: Is it linear? Quadratic? Conditional? Are there interactions?
3. HYPOTHESIZE: Form a mathematical expression that fits the data
4. TEST your hypothesis with a few more experiments
5. REFINE if needed
6. PREDICT only when confident

## Important Rules
- You have a LIMITED experiment budget — don't waste experiments
- For multi-variable systems, isolate variables (change one at a time)
- Watch for conditional/piecewise behavior (sudden changes)
- Noisy systems require repeated measurements to find the signal
- Respond with ONLY the JSON action, optionally preceded by a brief reasoning line

## Response Format
Think briefly, then respond with:
REASONING: <one line of reasoning>
ACTION: <json action>
"""


class LLMScientist(BaseAgent):
    """
    LLM-based scientist agent using OpenAI API.
    
    Requires OPENAI_API_KEY environment variable or explicit api_key.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        temperature: float = 0.3,
        max_history: int = 20,
    ):
        if not HAS_OPENAI:
            raise ImportError(
                "openai package required. Install with: pip install openai"
            )

        self.model = model
        self.temperature = temperature
        self.max_history = max_history
        self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        self.messages = []
        self.reset()

    @property
    def name(self) -> str:
        return f"Dr. LLM ({self.model})"

    def reset(self):
        """Reset conversation history for a new episode."""
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    def act(self, observation: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
        """
        Use the LLM to decide the next action.

        Args:
            observation: Current environment observation.

        Returns:
            Tuple of (action_dict, reasoning_string)
        """
        # Format observation for the LLM
        obs_text = self._format_observation(observation)

        self.messages.append({"role": "user", "content": obs_text})

        # Trim history if too long
        if len(self.messages) > self.max_history * 2 + 1:
            self.messages = [self.messages[0]] + self.messages[-(self.max_history * 2):]

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                temperature=self.temperature,
                max_tokens=500,
            )
            reply = response.choices[0].message.content.strip()
        except Exception as e:
            return (
                {"action": "get_status"},
                f"LLM API error: {e}. Requesting status...",
            )

        self.messages.append({"role": "assistant", "content": reply})

        # Parse the response
        action, reasoning = self._parse_response(reply, observation)
        return action, reasoning

    def _format_observation(self, obs: Dict[str, Any]) -> str:
        """Format observation as a clear text prompt for the LLM."""
        parts = []

        if obs.get("message"):
            parts.append(f"[System] {obs['message']}")

        if obs.get("world"):
            world = obs["world"]
            parts.append(f"\nWorld: {world.get('world_name', 'Unknown')}")
            parts.append(f"Description: {world.get('description', '')}")
            parts.append(f"Variables: {world.get('variables', [])}")
            parts.append(f"Ranges: {world.get('variable_ranges', {})}")
            if world.get("is_stateful"):
                parts.append("WARNING: This system has MEMORY -- order of experiments matters!")

        parts.append(f"\nExperiment Budget: {obs.get('experiments_remaining', '?')} remaining (used {obs.get('experiments_used', '?')})")

        if obs.get("last_experiment_result"):
            result = obs["last_experiment_result"]
            parts.append(f"\nLast Experiment: inputs={result.get('inputs')} → output={result.get('output')}")

        if obs.get("experiment_history"):
            parts.append("\nRecent Experiments:")
            for exp in obs["experiment_history"][-5:]:
                parts.append(f"  {exp.get('inputs')} → {exp.get('output')}")

        if obs.get("hypothesis_feedback"):
            fb = obs["hypothesis_feedback"]
            parts.append(f"\nHypothesis Feedback: {fb.get('quality', 'unknown')}")

        if obs.get("test_cases"):
            parts.append(f"\nTest Cases to Predict ({len(obs['test_cases'])} cases):")
            for i, case in enumerate(obs["test_cases"][:5]):
                parts.append(f"  Case {i+1}: {case}")
            if len(obs["test_cases"]) > 5:
                parts.append(f"  ... and {len(obs['test_cases']) - 5} more")

        parts.append("\nWhat is your next action? (respond with JSON)")

        return "\n".join(parts)

    def _parse_response(self, reply: str, observation: Dict) -> Tuple[Dict, str]:
        """Parse the LLM's response into an action and reasoning."""
        reasoning = ""
        action_str = reply

        # Extract reasoning and action
        if "REASONING:" in reply:
            parts = reply.split("ACTION:")
            if len(parts) >= 2:
                reasoning = parts[0].replace("REASONING:", "").strip()
                action_str = parts[1].strip()
            else:
                action_str = reply.split("REASONING:")[-1].strip()

        # Try to find JSON in the response
        action = self._extract_json(action_str)

        if action is None:
            # Try the full reply
            action = self._extract_json(reply)

        if action is None:
            return ({"action": "get_status"}, f"Could not parse LLM response: {reply[:100]}")

        # Validate action
        if "action" not in action:
            action["action"] = "get_status"

        if not reasoning:
            reasoning = reply[:100] if len(reply) > 100 else reply

        return action, reasoning

    @staticmethod
    def _extract_json(text: str) -> Optional[Dict]:
        """Try to extract a JSON object from text."""
        # Try direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try to find JSON in the text
        import re
        json_pattern = r'\{[^{}]*\}'
        matches = re.findall(json_pattern, text)
        for match in matches:
            try:
                obj = json.loads(match)
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                continue

        # Try nested JSON
        json_pattern = r'\{[^{}]*\{[^{}]*\}[^{}]*\}'
        matches = re.findall(json_pattern, text)
        for match in matches:
            try:
                obj = json.loads(match)
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                continue

        return None
