"""
Gymnasium-Compatible Wrapper for the Hypothesis Engine.

Provides a standard gymnasium.Env interface for seamless integration
with RL training frameworks (Stable-Baselines3, RLlib, TRL, etc.).

Usage:
    import gymnasium as gym

    # Register the environment
    from hypothesis_engine.gym_wrapper import HypothesisEngineGymEnv

    env = HypothesisEngineGymEnv(difficulty=3, experiment_budget=30)
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(action_text)
"""

import json
import numpy as np
from typing import Dict, Any, Optional, Tuple

try:
    import gymnasium as gym
    from gymnasium import spaces
    HAS_GYMNASIUM = True
except ImportError:
    HAS_GYMNASIUM = False

from .env import HypothesisEngine
from .worlds import WorldGenerator


class HypothesisEngineGymEnv:
    """
    Gymnasium-compatible wrapper for the Hypothesis Engine.

    This wrapper exposes the Hypothesis Engine as a standard RL environment
    with text-based observation and action spaces, designed for training
    LLMs via reinforcement learning (e.g., RLHF, PPO, GRPO).

    Observation Space:
        A text string containing the current state description, experiment
        history, and available actions.

    Action Space:
        A text string containing a JSON action (experiment, hypothesize,
        predict, get_hint, get_status).

    Reward:
        Float reward signal combining prediction accuracy, hypothesis quality,
        experiment efficiency, information gain, and progressive improvement.
    """

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(
        self,
        difficulty: int = 1,
        experiment_budget: int = 30,
        seed: Optional[int] = None,
        auto_curriculum: bool = False,
        advance_threshold: float = 65.0,
        max_steps: int = 50,
        render_mode: Optional[str] = None,
    ):
        """
        Initialize the Gymnasium-compatible Hypothesis Engine.

        Args:
            difficulty: Starting difficulty level (1-10).
            experiment_budget: Max experiments per episode.
            seed: Random seed for reproducibility.
            auto_curriculum: Enable automatic difficulty progression.
            advance_threshold: Score needed to advance to next level.
            max_steps: Maximum steps before truncation.
            render_mode: "human" for rich display, "ansi" for plain text, None for silent.
        """
        self.engine = HypothesisEngine(
            difficulty=difficulty,
            experiment_budget=experiment_budget,
            seed=seed,
            auto_curriculum=auto_curriculum,
            advance_threshold=advance_threshold,
        )
        self.max_steps = max_steps
        self.render_mode = render_mode
        self._step_count = 0
        self._last_obs_text = ""

        # If gymnasium is available, set up proper spaces
        if HAS_GYMNASIUM:
            # Text-based spaces for LLM training
            self.observation_space = spaces.Text(
                min_length=0,
                max_length=10000,
            )
            self.action_space = spaces.Text(
                min_length=1,
                max_length=2000,
            )

        # Display for render
        self._display = None
        if render_mode == "human":
            from .display import Display
            self._display = Display(slow_mode=False)

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Reset the environment and start a new episode.

        Args:
            seed: Optional random seed.
            options: Optional configuration overrides.

        Returns:
            Tuple of (observation_text, info_dict)
        """
        self._step_count = 0

        # Handle options
        if options and "difficulty" in options:
            self.engine.initial_difficulty = options["difficulty"]

        obs = self.engine.reset(seed=seed)

        obs_text = self._format_observation(obs)
        self._last_obs_text = obs_text

        info = {
            "raw_observation": obs,
            "difficulty": self.engine.world.difficulty if self.engine.world else 1,
            "world_name": self.engine.world.name if self.engine.world else "Unknown",
            "action_space_description": self.engine.get_action_space_description(),
        }

        if self.render_mode == "human" and self._display:
            world_briefing = obs.get("world", {})
            self._display.show_episode_start(
                self.engine.episode_count,
                self.engine.world.difficulty,
                world_briefing,
            )

        return obs_text, info

    def step(self, action_text: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        """
        Take a step in the environment.

        Args:
            action_text: A JSON string describing the action, or a natural language
                        string that will be parsed into an action.

        Returns:
            Tuple of (observation_text, reward, terminated, truncated, info)
        """
        self._step_count += 1

        # Parse action
        action = self._parse_action(action_text)

        # Execute in engine
        obs, reward, done, info = self.engine.step(action)

        # Format observation as text
        obs_text = self._format_observation(obs)
        self._last_obs_text = obs_text

        # Check truncation
        truncated = self._step_count >= self.max_steps and not done
        terminated = done

        # Add step metadata to info
        info["step"] = self._step_count
        info["raw_observation"] = obs

        # Render if needed
        if self.render_mode == "human" and self._display:
            action_type = action.get("action", "")
            if action_type == "experiment" and "experiment_result" in info:
                result = info["experiment_result"]
                self._display.show_experiment(
                    self._step_count,
                    result.get("inputs", {}),
                    result.get("output"),
                )
            elif action_type == "hypothesize" and "hypothesis_score" in info:
                score = info["hypothesis_score"]
                hint = "high" if score >= 0.8 else "medium" if score >= 0.5 else "low"
                self._display.show_hypothesis(
                    len(self.engine.hypothesis_history),
                    action.get("expression", "?"),
                    hint,
                )

        return obs_text, float(reward), terminated, truncated, info

    def render(self) -> Optional[str]:
        """Render the current state."""
        if self.render_mode == "ansi":
            return self._last_obs_text
        elif self.render_mode == "human":
            print(self._last_obs_text)
        return None

    def close(self):
        """Clean up resources."""
        pass

    # ── Observation Formatting ────────────────────────────────────────────

    def _format_observation(self, obs: Dict[str, Any]) -> str:
        """
        Format the raw observation dict into a structured text string
        suitable for LLM consumption.
        """
        parts = []

        # System message
        if obs.get("message"):
            parts.append(f"[System] {obs['message']}")

        # World description
        if obs.get("world"):
            world = obs["world"]
            parts.append(f"\n== World: {world.get('world_name', 'Unknown')} ==")
            parts.append(f"Difficulty: {world.get('difficulty', '?')}/10")
            parts.append(f"Category: {world.get('category', '?')}")
            parts.append(f"Description: {world.get('description', '')}")
            parts.append(f"Variables: {world.get('variables', [])}")

            ranges = world.get("variable_ranges", {})
            range_strs = [f"  {v}: [{r[0]}, {r[1]}]" for v, r in ranges.items()]
            parts.append("Ranges:\n" + "\n".join(range_strs))

            if world.get("is_stateful"):
                parts.append("WARNING: This system has MEMORY -- order matters!")

            if world.get("hints"):
                parts.append(f"Hint: {world['hints'][0]}")

        # Budget status
        remaining = obs.get("experiments_remaining", "?")
        used = obs.get("experiments_used", "?")
        parts.append(f"\nBudget: {remaining} experiments remaining ({used} used)")

        # Experiment history
        if obs.get("experiment_history"):
            parts.append("\nExperiment History (recent):")
            for i, exp in enumerate(obs["experiment_history"]):
                inputs = exp.get("inputs", {})
                output = exp.get("output", "ERROR")
                input_str = ", ".join(f"{k}={v}" for k, v in inputs.items())
                parts.append(f"  #{i+1}: [{input_str}] -> y = {output}")

        # Last experiment result (highlighted)
        if obs.get("last_experiment_result"):
            result = obs["last_experiment_result"]
            inputs = result.get("inputs", {})
            output = result.get("output", "ERROR")
            input_str = ", ".join(f"{k}={v}" for k, v in inputs.items())
            parts.append(f"\nLast Result: [{input_str}] -> y = {output}")

        # Hypothesis feedback
        if obs.get("hypothesis_feedback"):
            fb = obs["hypothesis_feedback"]
            parts.append(f"\nHypothesis Feedback:")
            parts.append(f"  Expression: y = {fb.get('expression', '?')}")
            parts.append(f"  Quality: {fb.get('quality', '?')}")
            parts.append(f"  Score Level: {fb.get('score_hint', '?')}")

        # Latest hypothesis
        if obs.get("latest_hypothesis"):
            parts.append(f"\nCurrent Hypothesis: y = {obs['latest_hypothesis']}")

        # Test cases (for prediction)
        if obs.get("test_cases"):
            cases = obs["test_cases"]
            parts.append(f"\nTest Cases ({len(cases)} predictions needed):")
            for i, case in enumerate(cases[:8]):
                case_str = ", ".join(f"{k}={v}" for k, v in case.items())
                parts.append(f"  Case {i+1}: [{case_str}] -> y = ?")
            if len(cases) > 8:
                parts.append(f"  ... and {len(cases) - 8} more cases")

        # Final results
        if obs.get("final_results"):
            results = obs["final_results"]
            parts.append("\n== Episode Complete ==")
            reward = results.get("reward_breakdown", {})
            parts.append(f"Total Score: {reward.get('total_reward', 0)}/100")
            parts.append(f"Passed: {results.get('passed', False)}")
            parts.append(f"Ground Truth: y = {results.get('ground_truth', '?')}")

        # Available actions reminder
        parts.append("\nAvailable Actions (respond with JSON):")
        parts.append('  {"action": "experiment", "inputs": {"x": VALUE}}')
        parts.append('  {"action": "hypothesize", "expression": "MATH_EXPR"}')
        parts.append('  {"action": "predict", "predictions": [v1, v2, ...]}')
        parts.append('  {"action": "get_hint"}')
        parts.append('  {"action": "get_status"}')

        return "\n".join(parts)

    # ── Action Parsing ────────────────────────────────────────────────────

    def _parse_action(self, action_text: str) -> Dict[str, Any]:
        """
        Parse an action string into an action dict.

        Handles:
            - Pure JSON: {"action": "experiment", "inputs": {"x": 3}}
            - JSON with reasoning prefix: "Let me try x=3\n{...}"
            - Natural language (fallback to get_status)
        """
        action_text = action_text.strip()

        # Try direct JSON parse
        try:
            action = json.loads(action_text)
            if isinstance(action, dict) and "action" in action:
                return action
        except json.JSONDecodeError:
            pass

        # Try to extract JSON from text
        import re
        # Nested JSON (e.g., {"action": "experiment", "inputs": {"x": 3}})
        patterns = [
            r'\{[^{}]*\{[^{}]*\}[^{}]*\}',  # nested
            r'\{[^{}]*\}',                      # simple
        ]
        for pattern in patterns:
            matches = re.findall(pattern, action_text)
            for match in matches:
                try:
                    action = json.loads(match)
                    if isinstance(action, dict) and "action" in action:
                        return action
                except json.JSONDecodeError:
                    continue

        # Fallback
        return {"action": "get_status"}

    # ── Utility ───────────────────────────────────────────────────────────

    def get_episode_summary(self) -> Dict[str, Any]:
        """Get summary of the current/last episode."""
        return self.engine.get_episode_summary()

    @property
    def difficulty(self) -> int:
        """Current difficulty level."""
        return self.engine.world.difficulty if self.engine.world else self.engine.initial_difficulty

    @property
    def world_name(self) -> str:
        """Current world name."""
        return self.engine.world.name if self.engine.world else "Not started"


def make_env(
    difficulty: int = 1,
    experiment_budget: int = 30,
    seed: Optional[int] = None,
    auto_curriculum: bool = False,
    render_mode: Optional[str] = None,
) -> HypothesisEngineGymEnv:
    """
    Factory function to create a Hypothesis Engine environment.

    This is the recommended way to create the environment:

        from hypothesis_engine.gym_wrapper import make_env

        env = make_env(difficulty=3, experiment_budget=30)
        obs, info = env.reset()
        obs, reward, terminated, truncated, info = env.step('{"action": "experiment", "inputs": {"x": 3}}')

    Args:
        difficulty: Starting difficulty (1-10).
        experiment_budget: Max experiments per episode.
        seed: Random seed.
        auto_curriculum: Auto-advance difficulty.
        render_mode: "human", "ansi", or None.

    Returns:
        HypothesisEngineGymEnv instance.
    """
    return HypothesisEngineGymEnv(
        difficulty=difficulty,
        experiment_budget=experiment_budget,
        seed=seed,
        auto_curriculum=auto_curriculum,
        render_mode=render_mode,
    )
