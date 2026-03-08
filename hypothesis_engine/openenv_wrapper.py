"""
OpenEnv Integration for Hypothesis Engine.

Wraps HypothesisEngine as an openenv-core Environment (v0.2.1)
for deployment on HuggingFace Spaces and use with TRL/Unsloth training.

Usage:
    from hypothesis_engine.openenv_wrapper import create_hypothesis_app
    app = create_hypothesis_app()
"""

import json
from typing import Any, Dict, List, Optional

from openenv.core import (
    Action,
    Environment,
    Observation,
    State,
    create_app,
)
from pydantic import Field

from .env import HypothesisEngine


# ---------------------------------------------------------------------------
# Custom types for OpenEnv
# ---------------------------------------------------------------------------

class HypothesisAction(Action):
    """Action for the Hypothesis Engine environment."""

    action: str = Field(
        description=(
            "Action type: 'experiment', 'hypothesize', 'predict', "
            "'get_status', or 'get_hint'"
        )
    )
    inputs: Optional[Dict[str, float]] = Field(
        default=None,
        description="Input values for experiment action, e.g. {'x': 3.0}",
    )
    mode: Optional[str] = Field(
        default=None,
        description="Experiment mode: 'observe' or 'intervene' (causal worlds only)",
    )
    expression: Optional[str] = Field(
        default=None,
        description="Mathematical expression for hypothesize action, e.g. '2*x + 3'",
    )
    predictions: Optional[List[float]] = Field(
        default=None,
        description="List of predicted values for predict action",
    )


class HypothesisObservation(Observation):
    """Observation returned by the Hypothesis Engine environment."""

    text: str = Field(
        default="",
        description="Natural language observation for LLM agents",
    )
    phase: str = Field(
        default="not_started",
        description="Current episode phase: exploration, prediction, or done",
    )
    world_name: str = Field(default="", description="Name of the current world")
    world_type: str = Field(default="", description="Type of the current world")
    experiments_remaining: int = Field(
        default=0, description="Experiments remaining in budget"
    )
    experiments_used: int = Field(
        default=0, description="Experiments used so far"
    )
    action_space: str = Field(
        default="", description="Description of available actions"
    )


class HypothesisState(State):
    """Internal state of the Hypothesis Engine environment."""

    difficulty: int = Field(default=1, description="Current difficulty level")
    world_name: str = Field(default="", description="Current world name")
    world_type: str = Field(default="", description="Current world type")
    phase: str = Field(default="not_started", description="Episode phase")
    experiments_used: int = Field(default=0, description="Experiments used")
    experiments_remaining: int = Field(default=0, description="Experiments left")
    hypothesis_count: int = Field(default=0, description="Hypotheses submitted")
    best_hypothesis_score: float = Field(
        default=0.0, description="Best hypothesis score so far"
    )


# ---------------------------------------------------------------------------
# Observation formatter -- turns raw dict into LLM-friendly text
# ---------------------------------------------------------------------------

def _format_observation_text(raw_obs: Dict[str, Any], action_desc: str) -> str:
    """Convert a raw HypothesisEngine observation dict into a natural-language string."""
    parts = []

    # Message
    if raw_obs.get("message"):
        parts.append(raw_obs["message"])

    # World info
    world = raw_obs.get("world", {})
    if world:
        parts.append(
            f"\n-- World: {world.get('world_name', '?')} "
            f"(type: {world.get('world_type', '?')}, "
            f"difficulty: {world.get('difficulty', '?')})"
        )
        parts.append(f"   Description: {world.get('description', '')}")
        parts.append(f"   Variables: {world.get('variables', [])}")
        if world.get('causal_mode'):
            parts.append(
                f"   Causal Mode: This world supports observe AND intervene experiments."
            )

    # Budget
    if "experiments_remaining" in raw_obs:
        parts.append(
            f"\n-- Budget: {raw_obs['experiments_remaining']} experiments remaining "
            f"(used {raw_obs.get('experiments_used', 0)})"
        )

    # Last experiment result
    if raw_obs.get("last_experiment_result"):
        r = raw_obs["last_experiment_result"]
        parts.append(f"\n-- Last Experiment: inputs={r.get('inputs')}, output={r.get('output')}")
        if r.get("mode"):
            parts.append(f"   Mode: {r['mode']}")

    # Hypothesis feedback
    if raw_obs.get("hypothesis_feedback"):
        hf = raw_obs["hypothesis_feedback"]
        parts.append(f"\n-- Hypothesis Feedback: {hf.get('quality', '')}")

    # Recent experiment history (last 5)
    hist = raw_obs.get("experiment_history", [])
    if hist:
        parts.append(f"\n-- Recent Experiments ({len(hist)} shown):")
        for i, exp in enumerate(hist[-5:], 1):
            parts.append(f"   {i}. inputs={exp.get('inputs')} -> output={exp.get('output')}")

    # Test cases
    tests = raw_obs.get("test_cases", [])
    if tests:
        parts.append(f"\n-- Test Cases to Predict ({len(tests)} total):")
        for i, tc in enumerate(tests[:5], 1):
            parts.append(f"   {i}. {tc}")
        if len(tests) > 5:
            parts.append(f"   ... and {len(tests) - 5} more")

    # Final results
    if raw_obs.get("final_results"):
        fr = raw_obs["final_results"]
        rb = fr.get("reward_breakdown", {})
        parts.append(f"\n-- FINAL RESULTS --")
        parts.append(f"   Total Reward: {rb.get('total_reward', 0):.1f}/100")
        parts.append(f"   Ground Truth: {fr.get('ground_truth', '?')}")
        parts.append(f"   Passed: {fr.get('passed', False)}")

    # Action space
    parts.append(f"\n-- Available Actions --\n{action_desc}")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# OpenEnv Environment
# ---------------------------------------------------------------------------

class HypothesisEngineOpenEnv(
    Environment[HypothesisAction, HypothesisObservation, HypothesisState]
):
    """
    OpenEnv-compatible wrapper for Hypothesis Engine.

    This wraps HypothesisEngine to work with openenv-core 0.2.1,
    enabling deployment on HuggingFace Spaces and training with TRL/Unsloth.
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(
        self,
        difficulty: int = 1,
        experiment_budget: int = 30,
        auto_curriculum: bool = True,
        use_self_play: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.difficulty = difficulty
        self.experiment_budget = experiment_budget
        self.auto_curriculum = auto_curriculum
        self.use_self_play = use_self_play
        self._env: Optional[HypothesisEngine] = None
        self._last_raw_obs: Dict[str, Any] = {}
        self._step_count = 0

    def _ensure_env(self, seed: Optional[int] = None) -> HypothesisEngine:
        """Create a new HypothesisEngine instance."""
        return HypothesisEngine(
            difficulty=self.difficulty,
            experiment_budget=self.experiment_budget,
            seed=seed,
            auto_curriculum=self.auto_curriculum,
            use_self_play=self.use_self_play,
        )

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs,
    ) -> HypothesisObservation:
        """Reset the environment and return initial observation."""
        self._env = self._ensure_env(seed=seed)
        self._step_count = 0
        raw_obs = self._env.reset(seed=seed)
        self._last_raw_obs = raw_obs

        action_desc = self._env.get_action_space_description()
        text = _format_observation_text(raw_obs, action_desc)

        world = raw_obs.get("world", {})
        obs = HypothesisObservation(
            text=text,
            done=False,
            reward=None,
            phase=raw_obs.get("phase", "exploration"),
            world_name=world.get("world_name", ""),
            world_type=world.get("world_type", ""),
            experiments_remaining=raw_obs.get("experiments_remaining", 0),
            experiments_used=raw_obs.get("experiments_used", 0),
            action_space=action_desc,
            metadata={"episode_id": episode_id or "", "raw": raw_obs},
        )

        return self._apply_transform(obs)

    def step(
        self,
        action: HypothesisAction,
        timeout_s: Optional[float] = None,
        **kwargs,
    ) -> HypothesisObservation:
        """Take a step in the environment."""
        if self._env is None:
            return HypothesisObservation(
                text="Environment not started. Call reset() first.",
                done=False,
                reward=-1.0,
                phase="not_started",
            )

        # Convert OpenEnv action to HypothesisEngine action dict
        action_dict: Dict[str, Any] = {"action": action.action}
        if action.inputs is not None:
            action_dict["inputs"] = action.inputs
        if action.mode is not None:
            action_dict["mode"] = action.mode
        if action.expression is not None:
            action_dict["expression"] = action.expression
        if action.predictions is not None:
            action_dict["predictions"] = action.predictions

        raw_obs, reward, done, info = self._env.step(action_dict)
        self._last_raw_obs = raw_obs
        self._step_count += 1

        action_desc = self._env.get_action_space_description()
        text = _format_observation_text(raw_obs, action_desc)

        world = raw_obs.get("world", {})
        obs = HypothesisObservation(
            text=text,
            done=done,
            reward=reward,
            phase=raw_obs.get("phase", ""),
            world_name=world.get("world_name", ""),
            world_type=world.get("world_type", ""),
            experiments_remaining=raw_obs.get("experiments_remaining", 0),
            experiments_used=raw_obs.get("experiments_used", 0),
            action_space=action_desc if not done else "",
            metadata={"info": info, "raw": raw_obs},
        )

        return self._apply_transform(obs)

    @property
    def state(self) -> HypothesisState:
        """Get the current environment state."""
        if self._env is None:
            return HypothesisState()

        summary = self._env.get_episode_summary()
        return HypothesisState(
            step_count=self._step_count,
            difficulty=summary.get("difficulty", self.difficulty),
            world_name=summary.get("world_name", ""),
            world_type=summary.get("world_type", ""),
            phase=summary.get("phase", "not_started"),
            experiments_used=summary.get("experiments_used", 0),
            experiments_remaining=summary.get("experiments_remaining", 0),
            hypothesis_count=summary.get("hypotheses_submitted", 0),
            best_hypothesis_score=summary.get("best_hypothesis_score", 0.0),
        )

    def get_metadata(self):
        """Return environment metadata."""
        from openenv.core.env_server.types import EnvironmentMetadata

        return EnvironmentMetadata(
            name="HypothesisEngine",
            description=(
                "A procedurally-generated RL environment for training LLMs on "
                "scientific reasoning through causal discovery, physics simulation, "
                "state machine reverse-engineering, and adversarial self-play."
            ),
            version="2.0.0",
            author="AbhinavDubey30",
            documentation_url="https://github.com/AbhinavDubey30/OpenMax",
        )

    def close(self) -> None:
        """Clean up resources."""
        self._env = None


# ---------------------------------------------------------------------------
# App factory for HuggingFace Spaces / local server
# ---------------------------------------------------------------------------

def create_hypothesis_app(
    difficulty: int = 1,
    experiment_budget: int = 30,
    auto_curriculum: bool = True,
    use_self_play: bool = False,
    max_concurrent_envs: int = 5,
):
    """
    Create a FastAPI app for serving HypothesisEngine on HF Spaces.

    Usage:
        # In app.py for HF Spaces:
        from hypothesis_engine.openenv_wrapper import create_hypothesis_app
        app = create_hypothesis_app()

        # Or run locally:
        # uvicorn hypothesis_engine.openenv_wrapper:app --reload
    """

    def env_factory():
        return HypothesisEngineOpenEnv(
            difficulty=difficulty,
            experiment_budget=experiment_budget,
            auto_curriculum=auto_curriculum,
            use_self_play=use_self_play,
        )

    return create_app(
        env=env_factory,
        action_cls=HypothesisAction,
        observation_cls=HypothesisObservation,
        env_name="HypothesisEngine",
        max_concurrent_envs=max_concurrent_envs,
    )


# Default app instance for uvicorn / HF Spaces
app = create_hypothesis_app()
