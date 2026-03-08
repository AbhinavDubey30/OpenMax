"""
Reward Calculator for the Hypothesis Engine.

Computes multi-component rewards for RL training:
    - Prediction Accuracy  (40%)  : How well the agent predicts unseen test cases
    - Hypothesis Quality   (25%)  : How close the hypothesis is to ground truth
    - Experiment Efficiency (15%)  : Fewer experiments = higher reward
    - Information Gain     (10%)  : Did experiments reduce uncertainty?
    - Progressive Improve. (10%)  : Did hypotheses improve over time?

All components are normalized to [0, 1] and combined with configurable weights.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import math


@dataclass
class RewardWeights:
    """Configurable weights for reward components."""
    prediction_accuracy: float = 0.40
    hypothesis_quality: float = 0.25
    experiment_efficiency: float = 0.15
    information_gain: float = 0.10
    progressive_improvement: float = 0.10

    def __post_init__(self):
        total = (
            self.prediction_accuracy
            + self.hypothesis_quality
            + self.experiment_efficiency
            + self.information_gain
            + self.progressive_improvement
        )
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Weights must sum to 1.0, got {total:.2f}")


@dataclass
class EpisodeMetrics:
    """Tracks all metrics during an episode for reward computation."""
    total_budget: int = 50
    experiments_used: int = 0
    hypothesis_scores: List[float] = field(default_factory=list)
    prediction_accuracy: float = 0.0
    prediction_r_squared: float = 0.0
    experiment_outputs: List[float] = field(default_factory=list)
    final_hypothesis_score: float = 0.0


class RewardCalculator:
    """
    Computes composite rewards for the Hypothesis Engine environment.
    
    Reward = weighted sum of 5 normalized components, scaled to [0, 100].
    """

    def __init__(self, weights: Optional[RewardWeights] = None):
        self.weights = weights or RewardWeights()

    def compute_final_reward(self, metrics: EpisodeMetrics) -> Dict[str, Any]:
        """
        Compute the final episode reward from collected metrics.

        Returns:
            Dict with total_reward (0-100) and breakdown of each component.
        """
        # 1. Prediction Accuracy (from R² and accuracy)
        pred_score = (
            0.6 * metrics.prediction_r_squared + 0.4 * metrics.prediction_accuracy
        )
        pred_score = max(0.0, min(1.0, pred_score))

        # 2. Hypothesis Quality (best hypothesis score)
        hyp_score = max(0.0, min(1.0, metrics.final_hypothesis_score))

        # 3. Experiment Efficiency
        if metrics.total_budget > 0:
            usage_ratio = metrics.experiments_used / metrics.total_budget
            # Sweet spot: using 30-70% of budget is optimal
            if usage_ratio <= 0.3:
                eff_score = 1.0  # Very efficient
            elif usage_ratio <= 0.7:
                eff_score = 1.0 - 0.5 * (usage_ratio - 0.3) / 0.4  # Gradual decrease
            else:
                eff_score = 0.5 - 0.5 * (usage_ratio - 0.7) / 0.3  # Steeper decrease
            eff_score = max(0.0, min(1.0, eff_score))
        else:
            eff_score = 0.0

        # 4. Information Gain (variance in experiment outputs indicates exploration)
        info_score = self._compute_info_gain(metrics.experiment_outputs)

        # 5. Progressive Improvement (are hypotheses getting better?)
        prog_score = self._compute_progressive_improvement(metrics.hypothesis_scores)

        # Weighted combination
        total = (
            self.weights.prediction_accuracy * pred_score
            + self.weights.hypothesis_quality * hyp_score
            + self.weights.experiment_efficiency * eff_score
            + self.weights.information_gain * info_score
            + self.weights.progressive_improvement * prog_score
        )

        # Scale to 0-100
        total_reward = round(total * 100, 1)

        return {
            "total_reward": total_reward,
            "max_reward": 100.0,
            "breakdown": {
                "prediction_accuracy": {
                    "score": round(pred_score, 4),
                    "weight": self.weights.prediction_accuracy,
                    "weighted": round(self.weights.prediction_accuracy * pred_score * 100, 1),
                    "max_points": round(self.weights.prediction_accuracy * 100, 1),
                },
                "hypothesis_quality": {
                    "score": round(hyp_score, 4),
                    "weight": self.weights.hypothesis_quality,
                    "weighted": round(self.weights.hypothesis_quality * hyp_score * 100, 1),
                    "max_points": round(self.weights.hypothesis_quality * 100, 1),
                },
                "experiment_efficiency": {
                    "score": round(eff_score, 4),
                    "weight": self.weights.experiment_efficiency,
                    "weighted": round(self.weights.experiment_efficiency * eff_score * 100, 1),
                    "max_points": round(self.weights.experiment_efficiency * 100, 1),
                },
                "information_gain": {
                    "score": round(info_score, 4),
                    "weight": self.weights.information_gain,
                    "weighted": round(self.weights.information_gain * info_score * 100, 1),
                    "max_points": round(self.weights.information_gain * 100, 1),
                },
                "progressive_improvement": {
                    "score": round(prog_score, 4),
                    "weight": self.weights.progressive_improvement,
                    "weighted": round(self.weights.progressive_improvement * prog_score * 100, 1),
                    "max_points": round(self.weights.progressive_improvement * 100, 1),
                },
            },
        }

    def compute_step_reward(
        self,
        action_type: str,
        experiment_result: Optional[Dict] = None,
        hypothesis_score: Optional[float] = None,
        prev_hypothesis_score: Optional[float] = None,
    ) -> float:
        """
        Compute intermediate (dense) reward for a single step.
        
        Provides small rewards during the episode to aid RL training:
            - Successful experiment: +0.1
            - Experiment with high info gain: +0.05 bonus
            - Hypothesis improvement: +0.5 * improvement
            - Hypothesis regression: -0.2 * regression
        """
        reward = 0.0

        if action_type == "experiment":
            if experiment_result and experiment_result.get("output") is not None:
                reward += 0.1  # Small reward for valid experiment

        elif action_type == "hypothesize":
            if hypothesis_score is not None:
                if prev_hypothesis_score is not None:
                    improvement = hypothesis_score - prev_hypothesis_score
                    if improvement > 0:
                        reward += 0.5 * improvement  # Reward improvement
                    else:
                        reward += 0.2 * improvement  # Penalize regression (less harshly)
                else:
                    reward += 0.3 * hypothesis_score  # First hypothesis reward

        return round(reward, 4)

    @staticmethod
    def _compute_info_gain(outputs: List[float]) -> float:
        """
        Estimate information gain from experiment diversity.
        
        High variance in outputs indicates the agent explored diverse inputs,
        which is better for hypothesis formation.
        """
        if len(outputs) < 3:
            return 0.0

        # Normalize by removing NaN/inf
        clean = [o for o in outputs if not (math.isnan(o) or math.isinf(o))]
        if len(clean) < 3:
            return 0.0

        mean = sum(clean) / len(clean)
        variance = sum((x - mean) ** 2 for x in clean) / len(clean)
        std = math.sqrt(variance)

        # Unique values ratio (did the agent explore, not repeat?)
        rounded = [round(o, 2) for o in clean]
        unique_ratio = len(set(rounded)) / len(rounded)

        # Combine: high std (relative to range) + high unique ratio
        output_range = max(clean) - min(clean)
        if output_range > 0:
            normalized_std = min(1.0, std / (output_range * 0.5))
        else:
            normalized_std = 0.0

        return 0.5 * normalized_std + 0.5 * unique_ratio

    @staticmethod
    def _compute_progressive_improvement(scores: List[float]) -> float:
        """
        Measure if hypotheses improved over the episode.
        
        Returns 1.0 if monotonically improving, 0.0 if no improvement.
        """
        if len(scores) < 2:
            return 0.5  # Neutral if only one hypothesis

        improvements = 0
        for i in range(1, len(scores)):
            if scores[i] >= scores[i - 1] - 0.01:  # Small tolerance
                improvements += 1

        # Also reward overall improvement (last vs first)
        overall_improvement = max(0.0, scores[-1] - scores[0])

        ratio = improvements / (len(scores) - 1)
        return 0.5 * ratio + 0.5 * min(1.0, overall_improvement / 0.5)
