"""
Auto-Curriculum Controller for the Hypothesis Engine.

Manages difficulty progression based on agent performance history.
Implements an adaptive curriculum that:
    - Advances difficulty when the agent consistently succeeds
    - Stays at the same level for more practice if struggling
    - Can optionally drop back a level if performance collapses
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class EpisodeRecord:
    """Record of a single episode's performance."""
    difficulty: int
    total_reward: float
    prediction_accuracy: float
    hypothesis_score: float
    experiments_used: int
    passed: bool


class CurriculumController:
    """
    Adaptive curriculum that controls difficulty progression.
    
    Advancement Criteria:
        - Score >= advance_threshold on current level
        - At least min_episodes at current level
        - Recent win rate >= required_win_rate
    
    Retreat Criteria:
        - Score < retreat_threshold for last N episodes
        - Only retreats one level at a time
    """

    MAX_DIFFICULTY = 10
    MIN_DIFFICULTY = 1

    def __init__(
        self,
        start_difficulty: int = 1,
        advance_threshold: float = 65.0,
        retreat_threshold: float = 25.0,
        min_episodes_per_level: int = 1,
        required_win_rate: float = 0.6,
        lookback_window: int = 3,
    ):
        self.current_difficulty = max(
            self.MIN_DIFFICULTY, min(self.MAX_DIFFICULTY, start_difficulty)
        )
        self.advance_threshold = advance_threshold
        self.retreat_threshold = retreat_threshold
        self.min_episodes_per_level = min_episodes_per_level
        self.required_win_rate = required_win_rate
        self.lookback_window = lookback_window
        self.history: List[EpisodeRecord] = []

    def record_episode(self, record: EpisodeRecord):
        """Record the result of a completed episode."""
        self.history.append(record)

    def get_next_difficulty(self) -> int:
        """Determine the next difficulty level based on performance history."""
        if not self.history:
            return self.current_difficulty

        # Get episodes at current difficulty
        current_episodes = [
            e for e in self.history if e.difficulty == self.current_difficulty
        ]

        if len(current_episodes) < self.min_episodes_per_level:
            return self.current_difficulty

        # Look at recent episodes at this level
        recent = current_episodes[-self.lookback_window:]

        # Compute win rate
        wins = sum(1 for e in recent if e.passed)
        win_rate = wins / len(recent)
        avg_reward = sum(e.total_reward for e in recent) / len(recent)

        # Advance?
        if (
            win_rate >= self.required_win_rate
            and avg_reward >= self.advance_threshold
            and self.current_difficulty < self.MAX_DIFFICULTY
        ):
            self.current_difficulty += 1
            return self.current_difficulty

        # Retreat?
        if (
            len(recent) >= self.lookback_window
            and avg_reward < self.retreat_threshold
            and self.current_difficulty > self.MIN_DIFFICULTY
        ):
            self.current_difficulty -= 1
            return self.current_difficulty

        # Stay
        return self.current_difficulty

    def get_progress_summary(self) -> Dict[str, Any]:
        """Get a summary of curriculum progress."""
        if not self.history:
            return {
                "current_difficulty": self.current_difficulty,
                "total_episodes": 0,
                "levels_completed": [],
                "overall_avg_reward": 0.0,
            }

        levels_completed = set()
        level_stats = {}

        for ep in self.history:
            d = ep.difficulty
            if d not in level_stats:
                level_stats[d] = {"attempts": 0, "wins": 0, "total_reward": 0.0}
            level_stats[d]["attempts"] += 1
            level_stats[d]["total_reward"] += ep.total_reward
            if ep.passed:
                level_stats[d]["wins"] += 1
                levels_completed.add(d)

        for d in level_stats:
            stats = level_stats[d]
            stats["avg_reward"] = round(stats["total_reward"] / stats["attempts"], 1)
            stats["win_rate"] = round(stats["wins"] / stats["attempts"], 2)

        overall_avg = sum(e.total_reward for e in self.history) / len(self.history)

        return {
            "current_difficulty": self.current_difficulty,
            "total_episodes": len(self.history),
            "levels_completed": sorted(levels_completed),
            "level_stats": level_stats,
            "overall_avg_reward": round(overall_avg, 1),
            "highest_level_reached": max(e.difficulty for e in self.history),
        }
