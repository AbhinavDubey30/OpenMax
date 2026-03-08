#!/usr/bin/env python3
"""
Example: Using Hypothesis Engine for LLM RL Training

This script demonstrates how the Hypothesis Engine can be used in a standard
reinforcement learning training loop for LLMs.

Three integration patterns are shown:
    1. Basic RL loop (any framework)
    2. Gymnasium-compatible wrapper
    3. Multi-episode training with auto-curriculum

Usage:
    python examples/training_loop.py
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def example_basic_loop():
    """
    Example 1: Basic RL Training Loop

    This shows the simplest integration pattern using the core
    HypothesisEngine API directly.
    """
    print("=" * 60)
    print("  Example 1: Basic RL Training Loop")
    print("=" * 60)

    from hypothesis_engine import HypothesisEngine

    env = HypothesisEngine(difficulty=1, experiment_budget=20, seed=42)

    # Standard RL loop
    for episode in range(3):
        obs = env.reset()
        done = False
        total_reward = 0.0
        step = 0

        while not done and step < 40:
            # YOUR AGENT SELECTS AN ACTION HERE
            # In real training, this comes from your LLM policy network
            action = select_action(obs, step)

            obs, reward, done, info = env.step(action)
            total_reward += reward
            step += 1

            # YOUR AGENT LEARNS HERE
            # e.g., store transition in replay buffer, compute policy gradient, etc.

        summary = env.get_episode_summary()
        print(f"  Episode {episode + 1}: steps={step}, reward={total_reward:.1f}, "
              f"world='{summary['world_name']}'")

    print()


def example_gymnasium_wrapper():
    """
    Example 2: Gymnasium-Compatible Wrapper

    Uses the standard gymnasium interface for compatibility with
    Stable-Baselines3, RLlib, TRL, and other RL frameworks.
    """
    print("=" * 60)
    print("  Example 2: Gymnasium-Compatible Wrapper")
    print("=" * 60)

    from hypothesis_engine.gym_wrapper import make_env

    env = make_env(difficulty=2, experiment_budget=20, seed=42)

    for episode in range(3):
        obs_text, info = env.reset()
        terminated = False
        truncated = False
        total_reward = 0.0
        step = 0

        print(f"\n  Episode {episode + 1}: {info.get('world_name', '?')}")

        while not terminated and not truncated and step < 40:
            # In a real setup, your LLM generates this action text
            action_text = generate_action_text(obs_text, step)

            obs_text, reward, terminated, truncated, info = env.step(action_text)
            total_reward += reward
            step += 1

        print(f"    Steps: {step}, Total Reward: {total_reward:.1f}")

    env.close()
    print()


def example_curriculum_training():
    """
    Example 3: Multi-Episode Training with Auto-Curriculum

    Demonstrates how the auto-curriculum automatically advances
    difficulty as the agent improves, providing a self-improving
    training signal.
    """
    print("=" * 60)
    print("  Example 3: Auto-Curriculum Training")
    print("=" * 60)

    from hypothesis_engine import HypothesisEngine

    env = HypothesisEngine(
        difficulty=1,
        experiment_budget=25,
        auto_curriculum=True,
        advance_threshold=60.0,
    )

    episode_rewards = []

    for episode in range(10):
        obs = env.reset()
        done = False
        total_reward = 0.0
        step = 0

        while not done and step < 40:
            action = select_action(obs, step)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            step += 1

        episode_rewards.append(total_reward)
        difficulty = env.world.difficulty if env.world else "?"

        # Get curriculum progress
        if env.curriculum:
            progress = env.curriculum.get_progress_summary()
            current_level = progress.get("current_difficulty", "?")
        else:
            current_level = difficulty

        print(f"  Episode {episode + 1:>2d}: Level {current_level}, "
              f"Reward: {total_reward:>6.1f}, "
              f"Running Avg: {sum(episode_rewards[-5:]) / min(5, len(episode_rewards)):>6.1f}")

    print(f"\n  Final average reward (last 5): "
          f"{sum(episode_rewards[-5:]) / min(5, len(episode_rewards)):.1f}")
    print()


# ── Helper: Simple action selector (replace with your LLM) ───────────

def select_action(obs, step):
    """
    Dummy action selector for demonstration.

    In real training, replace this with your LLM policy:
        - Feed obs_text to LLM
        - Parse LLM output as JSON action
        - Return the action dict
    """
    world = obs.get("world", {})
    variables = world.get("variables", ["x"])
    ranges = world.get("variable_ranges", {"x": [-10, 10]})
    test_cases = obs.get("test_cases", [])
    remaining = obs.get("experiments_remaining", 0)

    # Simple strategy: probe, then predict
    if step < 8 and remaining > 2:
        # Run experiments at strategic points
        inputs = {}
        for var in variables:
            lo, hi = ranges.get(var, [-10, 10])
            # Probe different points
            probes = [0, 1, -1, 2, -2, 3, -3, 5]
            idx = step % len(probes)
            val = max(lo, min(hi, float(probes[idx])))
            inputs[var] = val
        return {"action": "experiment", "inputs": inputs}

    elif step == 8:
        # Simple hypothesis
        return {"action": "hypothesize", "expression": "2*x + 1"}

    else:
        # Predict (ends episode)
        predictions = [0.0] * len(test_cases) if test_cases else [0.0] * 20
        return {"action": "predict", "predictions": predictions}


def generate_action_text(obs_text, step):
    """
    Generate action text for the gymnasium wrapper.

    In real training, this is where your LLM generates text responses.
    """
    import json

    if step < 5:
        probes = [0, 1, -1, 2, -2]
        return json.dumps({
            "action": "experiment",
            "inputs": {"x": probes[step % len(probes)]}
        })
    elif step == 5:
        return json.dumps({
            "action": "hypothesize",
            "expression": "2*x + 1"
        })
    else:
        return json.dumps({
            "action": "predict",
            "predictions": [0.0] * 20
        })


# ── Integration Guide ─────────────────────────────────────────────────

INTEGRATION_GUIDE = """
== INTEGRATION WITH POPULAR RL FRAMEWORKS ==

1. Stable-Baselines3 (via gymnasium wrapper):
   
   from hypothesis_engine.gym_wrapper import make_env
   from stable_baselines3 import PPO
   
   env = make_env(difficulty=3, experiment_budget=30)
   model = PPO("MlpPolicy", env)  # Use custom text policy
   model.learn(total_timesteps=10000)

2. TRL (Transformer RL for LLM fine-tuning):
   
   from hypothesis_engine.gym_wrapper import HypothesisEngineGymEnv
   from trl import PPOTrainer
   
   env = HypothesisEngineGymEnv(difficulty=1, auto_curriculum=True)
   # Use env.reset() and env.step() in your TRL training loop
   # The text-based obs/action spaces work naturally with LLMs

3. Custom GRPO/RLHF Training:
   
   from hypothesis_engine import HypothesisEngine
   
   env = HypothesisEngine(difficulty=1, auto_curriculum=True)
   
   for episode in range(num_episodes):
       obs = env.reset()
       trajectory = []
       done = False
       
       while not done:
           # LLM generates action from observation text
           action = llm.generate(format_prompt(obs))
           obs, reward, done, info = env.step(parse_action(action))
           trajectory.append((obs, action, reward))
       
       # Use trajectory for GRPO/PPO update
       update_policy(trajectory)

4. Ray RLlib:
   
   from ray.rllib.algorithms.ppo import PPOConfig
   from hypothesis_engine.gym_wrapper import HypothesisEngineGymEnv
   
   config = PPOConfig().environment(
       env=HypothesisEngineGymEnv,
       env_config={"difficulty": 3, "experiment_budget": 30},
   )
"""


if __name__ == "__main__":
    print()
    print("Hypothesis Engine — RL Training Integration Examples")
    print("=" * 60)
    print()

    example_basic_loop()
    example_gymnasium_wrapper()
    example_curriculum_training()

    print(INTEGRATION_GUIDE)
