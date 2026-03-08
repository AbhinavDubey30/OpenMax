"""
Hypothesis Engine — Scientific Discovery RL Environment

A procedurally-generated RL environment for training LLMs on
scientific reasoning and hypothesis testing.

Quick Start:
    from hypothesis_engine import HypothesisEngine
    
    env = HypothesisEngine(difficulty=1, experiment_budget=30)
    obs = env.reset()
    
    # Run an experiment
    obs, reward, done, info = env.step({
        "action": "experiment",
        "inputs": {"x": 3.0}
    })
    
    # Submit a hypothesis
    obs, reward, done, info = env.step({
        "action": "hypothesize",
        "expression": "2*x + 3"
    })
    
    # Submit predictions (ends episode)
    obs, reward, done, info = env.step({
        "action": "predict",
        "predictions": [9.0, -5.0, ...]
    })
"""

__version__ = "1.0.0"
__author__ = "Hypothesis Engine Team"

from .env import HypothesisEngine
from .worlds import WorldGenerator, World
from .verifier import HypothesisVerifier, SafeMathEvaluator
from .rewards import RewardCalculator, RewardWeights, EpisodeMetrics
from .curriculum import CurriculumController
from .gym_wrapper import HypothesisEngineGymEnv, make_env

__all__ = [
    "HypothesisEngine",
    "HypothesisEngineGymEnv",
    "make_env",
    "WorldGenerator",
    "World",
    "HypothesisVerifier",
    "SafeMathEvaluator",
    "RewardCalculator",
    "RewardWeights",
    "EpisodeMetrics",
    "CurriculumController",
]
