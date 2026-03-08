"""
Hypothesis Engine — Main RL Environment.

A gym-like reinforcement learning environment for training LLMs on
scientific reasoning and hypothesis testing through experimentation
with procedurally generated black-box worlds.

Interface:
    env = HypothesisEngine(difficulty=1, experiment_budget=30)
    obs = env.reset()
    obs, reward, done, info = env.step(action)

Actions:
    {"action": "experiment", "inputs": {"x": 3.0}}
    {"action": "hypothesize", "expression": "2*x + 3"}
    {"action": "predict", "predictions": [9.0, -5.0, ...]}
    {"action": "get_status"}
    {"action": "get_hint"}
"""

from typing import Dict, Any, Optional, List
from .worlds import WorldGenerator, World
from .verifier import HypothesisVerifier
from .rewards import RewardCalculator, EpisodeMetrics
from .curriculum import CurriculumController, EpisodeRecord


class HypothesisEngine:
    """
    The main RL environment for the Hypothesis Engine.
    
    Follows a gym-like interface: reset() -> step(action) -> (obs, reward, done, info)
    
    Episode Structure:
        1. EXPLORATION phase: Agent runs experiments and submits hypotheses
        2. PREDICTION phase: Agent submits predictions for held-out test cases
        3. Episode ends, final reward is computed
    """

    VALID_ACTIONS = {"experiment", "hypothesize", "predict", "get_status", "get_hint"}

    def __init__(
        self,
        difficulty: int = 1,
        experiment_budget: int = 30,
        seed: Optional[int] = None,
        auto_curriculum: bool = False,
        advance_threshold: float = 65.0,
    ):
        """
        Initialize the Hypothesis Engine environment.

        Args:
            difficulty: Starting difficulty level (1-10).
            experiment_budget: Max experiments per episode.
            seed: Random seed for reproducibility.
            auto_curriculum: Enable automatic difficulty progression.
            advance_threshold: Score needed to advance to next level.
        """
        self.initial_difficulty = difficulty
        self.experiment_budget = experiment_budget
        self.seed = seed
        self.auto_curriculum = auto_curriculum
        self._seed_counter = seed if seed is not None else 0

        self.verifier = HypothesisVerifier()
        self.reward_calc = RewardCalculator()

        self.curriculum = CurriculumController(
            start_difficulty=difficulty,
            advance_threshold=advance_threshold,
        ) if auto_curriculum else None

        # Episode state
        self.world: Optional[World] = None
        self.metrics: Optional[EpisodeMetrics] = None
        self.experiment_history: List[Dict] = []
        self.hypothesis_history: List[Dict] = []
        self.experiments_remaining: int = 0
        self.phase: str = "not_started"  # not_started | exploration | prediction | done
        self.episode_count: int = 0

    def reset(self, seed: Optional[int] = None) -> Dict[str, Any]:
        """
        Reset the environment and generate a new world.

        Args:
            seed: Optional seed override for this episode.

        Returns:
            Initial observation dict with world briefing.
        """
        # Determine difficulty
        if self.auto_curriculum and self.curriculum:
            difficulty = self.curriculum.get_next_difficulty()
        else:
            difficulty = self.initial_difficulty

        # Generate seed for this episode
        if seed is not None:
            ep_seed = seed
        else:
            ep_seed = self._seed_counter
            self._seed_counter += 1

        # Generate world
        self.world = WorldGenerator.generate(difficulty, seed=ep_seed)
        self.world.generate_test_cases(20)

        # Reset episode state
        self.metrics = EpisodeMetrics(total_budget=self.experiment_budget)
        self.experiment_history = []
        self.hypothesis_history = []
        self.experiments_remaining = self.experiment_budget
        self.phase = "exploration"
        self.episode_count += 1

        return self._get_observation(message="New world generated. Begin your investigation!")

    def step(self, action: Dict[str, Any]) -> tuple:
        """
        Take an action in the environment.

        Args:
            action: Dict with "action" key and action-specific parameters.

        Returns:
            Tuple of (observation, reward, done, info)
        """
        if self.phase == "not_started":
            return (
                {"error": "Environment not started. Call reset() first."},
                0.0,
                False,
                {},
            )

        if self.phase == "done":
            return (
                {"error": "Episode is finished. Call reset() to start a new one."},
                0.0,
                True,
                {},
            )

        action_type = action.get("action", "").lower()

        if action_type not in self.VALID_ACTIONS:
            return (
                {
                    "error": f"Invalid action '{action_type}'. Valid: {self.VALID_ACTIONS}"
                },
                -0.1,
                False,
                {},
            )

        if action_type == "experiment":
            return self._handle_experiment(action)
        elif action_type == "hypothesize":
            return self._handle_hypothesize(action)
        elif action_type == "predict":
            return self._handle_predict(action)
        elif action_type == "get_status":
            return self._handle_get_status()
        elif action_type == "get_hint":
            return self._handle_get_hint()

        return ({"error": "Unexpected error"}, 0.0, False, {})

    # ── Action Handlers ──────────────────────────────────────────────────

    def _handle_experiment(self, action: Dict) -> tuple:
        """Run an experiment."""
        if self.phase != "exploration":
            return (
                {"error": "Experiments can only be run during exploration phase."},
                -0.05,
                False,
                {},
            )

        if self.experiments_remaining <= 0:
            return (
                {
                    "error": "No experiments remaining! Submit your hypothesis or predictions.",
                    "experiments_remaining": 0,
                },
                -0.05,
                False,
                {},
            )

        inputs = action.get("inputs", {})
        if not inputs:
            return (
                {"error": "Experiment requires 'inputs' dict. E.g., {'inputs': {'x': 3.0}}"},
                -0.05,
                False,
                {},
            )

        result = self.world.run_experiment(inputs)
        self.experiments_remaining -= 1
        self.metrics.experiments_used += 1

        # Track for metrics
        self.experiment_history.append(result)
        if result.get("output") is not None:
            self.metrics.experiment_outputs.append(result["output"])

        # Dense reward
        reward = self.reward_calc.compute_step_reward("experiment", experiment_result=result)

        obs = self._get_observation(
            message=f"Experiment #{len(self.experiment_history)} complete.",
            last_result=result,
        )

        return (obs, reward, False, {"experiment_result": result})

    def _handle_hypothesize(self, action: Dict) -> tuple:
        """Submit a hypothesis."""
        if self.phase not in ("exploration", "prediction"):
            return ({"error": "Cannot hypothesize in current phase."}, -0.05, False, {})

        expression = action.get("expression", "").strip()
        if not expression:
            return (
                {"error": "Hypothesis requires 'expression' string. E.g., '2*x + 3'"},
                -0.05,
                False,
                {},
            )

        # Verify hypothesis against ground truth
        verification = self.verifier.verify(expression, self.world)
        score = verification["score"]

        # Track
        prev_score = self.hypothesis_history[-1]["score"] if self.hypothesis_history else None
        self.hypothesis_history.append({
            "expression": expression,
            "score": score,
            "verification": verification,
        })
        self.metrics.hypothesis_scores.append(score)
        self.metrics.final_hypothesis_score = score

        # Dense reward
        reward = self.reward_calc.compute_step_reward(
            "hypothesize",
            hypothesis_score=score,
            prev_hypothesis_score=prev_score,
        )

        # Provide feedback (without revealing ground truth)
        if score >= 0.95:
            feedback = "Excellent! Your hypothesis is very close to the true pattern."
        elif score >= 0.8:
            feedback = "Very good! Your hypothesis captures most of the pattern."
        elif score >= 0.6:
            feedback = "Good progress. Your hypothesis captures the general trend."
        elif score >= 0.3:
            feedback = "Partial match. Some aspects are right, but there are gaps."
        else:
            feedback = "The hypothesis doesn't match the pattern well. Keep experimenting."

        obs = self._get_observation(
            message=f"Hypothesis #{len(self.hypothesis_history)} submitted. {feedback}",
            hypothesis_feedback={
                "expression": expression,
                "quality": feedback,
                "score_hint": "high" if score >= 0.8 else "medium" if score >= 0.5 else "low",
            },
        )

        return (obs, reward, False, {"hypothesis_score": score, "verification": verification})

    def _handle_predict(self, action: Dict) -> tuple:
        """Submit predictions for test cases — ends the episode."""
        predictions = action.get("predictions", [])
        test_cases = self.world.test_cases

        if len(predictions) != len(test_cases):
            return (
                {
                    "error": f"Expected {len(test_cases)} predictions, got {len(predictions)}. "
                    f"Use 'get_status' to see the test cases."
                },
                -0.1,
                False,
                {},
            )

        # Score predictions
        actuals = self.world.get_test_answers()
        pred_results = self.verifier.score_predictions(predictions, actuals)

        # Update metrics
        self.metrics.prediction_accuracy = pred_results["accuracy"]
        self.metrics.prediction_r_squared = pred_results["r_squared"]

        # Compute final reward
        final_reward_info = self.reward_calc.compute_final_reward(self.metrics)

        # Episode done
        self.phase = "done"

        # Record for curriculum
        passed = final_reward_info["total_reward"] >= 60.0
        if self.auto_curriculum and self.curriculum:
            self.curriculum.record_episode(EpisodeRecord(
                difficulty=self.world.difficulty,
                total_reward=final_reward_info["total_reward"],
                prediction_accuracy=pred_results["accuracy"],
                hypothesis_score=self.metrics.final_hypothesis_score,
                experiments_used=self.metrics.experiments_used,
                passed=passed,
            ))

        obs = self._get_observation(
            message="Episode complete! Final results computed.",
            final_results={
                "prediction_results": pred_results,
                "reward_breakdown": final_reward_info,
                "ground_truth": self.world.ground_truth_expr,
                "passed": passed,
            },
        )

        return (
            obs,
            final_reward_info["total_reward"],
            True,
            {
                "final_reward": final_reward_info,
                "prediction_results": pred_results,
                "ground_truth_expr": self.world.ground_truth_expr,
                "passed": passed,
            },
        )

    def _handle_get_status(self) -> tuple:
        """Get current episode status."""
        obs = self._get_observation(message="Current status.")
        return (obs, 0.0, False, {})

    def _handle_get_hint(self) -> tuple:
        """Get a hint for the current world."""
        hints = self.world.hints
        hint_idx = min(len(self.experiment_history) // 5, len(hints) - 1)
        hint = hints[hint_idx] if hints else "No hints available."

        obs = self._get_observation(message=f"Hint: {hint}")
        return (obs, -0.02, False, {"hint": hint})

    # ── Observation Builder ──────────────────────────────────────────────

    def _get_observation(
        self,
        message: str = "",
        last_result: Optional[Dict] = None,
        hypothesis_feedback: Optional[Dict] = None,
        final_results: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Build the observation dict returned to the agent."""
        obs = {
            "phase": self.phase,
            "message": message,
            "episode": self.episode_count,
        }

        if self.world:
            obs["world"] = self.world.get_agent_briefing()
            obs["experiment_budget"] = self.experiment_budget
            obs["experiments_remaining"] = self.experiments_remaining
            obs["experiments_used"] = len(self.experiment_history)
            obs["experiment_history"] = self.experiment_history[-10:]  # Last 10
            obs["total_experiment_count"] = len(self.experiment_history)
            obs["hypothesis_count"] = len(self.hypothesis_history)

            if self.hypothesis_history:
                obs["latest_hypothesis"] = self.hypothesis_history[-1]["expression"]

            if self.phase in ("exploration", "prediction"):
                obs["test_cases"] = [
                    {v: case[v] for v in self.world.variables}
                    for case in self.world.test_cases
                ]

        if last_result:
            obs["last_experiment_result"] = last_result

        if hypothesis_feedback:
            obs["hypothesis_feedback"] = hypothesis_feedback

        if final_results:
            obs["final_results"] = final_results

        return obs

    # ── Utility ──────────────────────────────────────────────────────────

    def get_action_space_description(self) -> str:
        """Return a human-readable description of valid actions."""
        return """
Available Actions:
─────────────────
1. EXPERIMENT: Run an experiment with specific input values.
   Format: {"action": "experiment", "inputs": {"x": 3.0}}
   
2. HYPOTHESIZE: Submit a mathematical hypothesis about the system.
   Format: {"action": "hypothesize", "expression": "2*x + 3"}
   Supported: +, -, *, /, **, sin(), cos(), exp(), log(), sqrt(), abs()
   Conditionals: "value_if_true if x > 0 else value_if_false"
   Or: "where(x > 0, x**2, -x)"
   
3. PREDICT: Submit predictions for all test cases (ends episode).
   Format: {"action": "predict", "predictions": [9.0, -5.0, ...]}
   
4. GET_STATUS: View current experiment history and test cases.
   Format: {"action": "get_status"}
   
5. GET_HINT: Get a hint about the world (small reward penalty).
   Format: {"action": "get_hint"}
""".strip()

    def get_episode_summary(self) -> Dict[str, Any]:
        """Get a summary of the current/last episode."""
        return {
            "episode": self.episode_count,
            "difficulty": self.world.difficulty if self.world else None,
            "world_name": self.world.name if self.world else None,
            "phase": self.phase,
            "experiments_used": len(self.experiment_history),
            "experiments_remaining": self.experiments_remaining,
            "hypotheses_submitted": len(self.hypothesis_history),
            "best_hypothesis_score": (
                max((h["score"] for h in self.hypothesis_history), default=0.0)
            ),
            "latest_hypothesis": (
                self.hypothesis_history[-1]["expression"]
                if self.hypothesis_history
                else None
            ),
        }
