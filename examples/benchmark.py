#!/usr/bin/env python3
"""
Hypothesis Engine — Environment Validation Benchmark

Validates that the environment meets key RL environment requirements:
    1. Deterministic (same seed = same world)
    2. Proper reward signal (dense + sparse)
    3. Curriculum progression works
    4. All 10 difficulty levels are functional
    5. Observation space is rich and informative
    6. Episode termination is correct

This is useful for demonstrating environment quality to judges.

Usage:
    python examples/benchmark.py
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_determinism():
    """Test that the same seed produces the same world."""
    from hypothesis_engine import HypothesisEngine

    print("  [1/6] Determinism Test...")

    env1 = HypothesisEngine(difficulty=3, seed=42)
    obs1 = env1.reset()

    env2 = HypothesisEngine(difficulty=3, seed=42)
    obs2 = env2.reset()

    assert obs1["world"]["world_name"] == obs2["world"]["world_name"], "World names differ!"
    assert obs1["world"]["variables"] == obs2["world"]["variables"], "Variables differ!"

    # Run same experiment in both
    action = {"action": "experiment", "inputs": {v: 1.0 for v in obs1["world"]["variables"]}}
    r1 = env1.step(action)
    r2 = env2.step(action)

    assert r1[0]["last_experiment_result"]["output"] == r2[0]["last_experiment_result"]["output"], \
        "Experiment outputs differ!"

    print("        PASS - Same seed produces identical worlds and results")
    return True


def test_reward_signal():
    """Test that reward signals are properly shaped."""
    from hypothesis_engine import HypothesisEngine

    print("  [2/6] Reward Signal Test...")

    env = HypothesisEngine(difficulty=1, experiment_budget=20, seed=42)
    obs = env.reset()

    # Run experiments - should get small positive rewards
    dense_rewards = []
    for i in range(5):
        obs, reward, done, info = env.step({
            "action": "experiment",
            "inputs": {"x": float(i)}
        })
        dense_rewards.append(reward)

    assert all(r >= 0 for r in dense_rewards), f"Expected non-negative dense rewards, got {dense_rewards}"
    assert not done, "Episode should not end during experiments"

    # Submit hypothesis
    obs, reward, done, info = env.step({
        "action": "hypothesize",
        "expression": "2*x + 1"
    })
    assert not done, "Episode should not end after hypothesis"

    # Submit predictions - should end episode with final reward
    test_cases = obs.get("test_cases", [])
    predictions = [0.0] * len(test_cases)
    obs, reward, done, info = env.step({
        "action": "predict",
        "predictions": predictions
    })

    assert done, "Episode should end after predictions"
    assert "final_reward" in info, "Final reward info should be present"
    assert 0 <= info["final_reward"]["total_reward"] <= 100, \
        f"Total reward should be 0-100, got {info['final_reward']['total_reward']}"

    print(f"        PASS - Dense rewards: {dense_rewards[:3]}..., "
          f"Final: {info['final_reward']['total_reward']:.1f}/100")
    return True


def test_all_difficulty_levels():
    """Test that all 10 difficulty levels generate valid worlds."""
    from hypothesis_engine import WorldGenerator

    print("  [3/6] All Difficulty Levels Test...")

    for level in range(1, 11):
        world = WorldGenerator.generate(level, seed=42)
        assert world.name, f"Level {level}: Missing name"
        assert world.variables, f"Level {level}: Missing variables"
        assert world.ground_truth_expr, f"Level {level}: Missing ground truth"
        assert len(world.test_cases) > 0, f"Level {level}: No test cases"

        # Run a test experiment
        inputs = {v: 0.0 for v in world.variables}
        result = world.run_experiment(inputs)
        assert result.get("output") is not None or result.get("error"), \
            f"Level {level}: Experiment returned no output"

    print("        PASS - All 10 levels generate valid worlds with experiments")
    return True


def test_curriculum_progression():
    """Test that auto-curriculum advances difficulty."""
    from hypothesis_engine import HypothesisEngine
    from hypothesis_engine.curriculum import EpisodeRecord

    print("  [4/6] Curriculum Progression Test...")

    env = HypothesisEngine(
        difficulty=1,
        experiment_budget=20,
        auto_curriculum=True,
        advance_threshold=50.0,
    )

    assert env.curriculum is not None, "Curriculum should be enabled"
    assert env.curriculum.current_difficulty == 1, "Should start at level 1"

    # Simulate winning episodes
    for i in range(3):
        env.curriculum.record_episode(EpisodeRecord(
            difficulty=1,
            total_reward=80.0,
            prediction_accuracy=0.9,
            hypothesis_score=0.95,
            experiments_used=10,
            passed=True,
        ))

    next_diff = env.curriculum.get_next_difficulty()
    assert next_diff == 2, f"Should advance to level 2 after wins, got {next_diff}"

    print("        PASS - Curriculum advances from level 1 to 2 after consistent wins")
    return True


def test_observation_richness():
    """Test that observations contain all required information for LLM training."""
    from hypothesis_engine import HypothesisEngine

    print("  [5/6] Observation Richness Test...")

    env = HypothesisEngine(difficulty=3, experiment_budget=20, seed=42)
    obs = env.reset()

    # Check required fields
    required_fields = [
        "phase", "message", "episode", "world",
        "experiment_budget", "experiments_remaining",
        "test_cases",
    ]

    for field in required_fields:
        assert field in obs, f"Missing required field: {field}"

    # Check world info
    world = obs["world"]
    assert "world_name" in world, "Missing world_name"
    assert "description" in world, "Missing description"
    assert "variables" in world, "Missing variables"
    assert "variable_ranges" in world, "Missing variable_ranges"
    assert "hints" in world, "Missing hints"

    # Check that test cases have correct structure
    test_cases = obs.get("test_cases", [])
    assert len(test_cases) > 0, "No test cases"
    for var in world["variables"]:
        assert var in test_cases[0], f"Test case missing variable {var}"

    print(f"        PASS - Observation has {len(required_fields)} required fields, "
          f"{len(test_cases)} test cases, {len(world['variables'])} variables")
    return True


def test_gymnasium_wrapper():
    """Test the Gymnasium-compatible wrapper."""
    print("  [6/6] Gymnasium Wrapper Test...")

    from hypothesis_engine.gym_wrapper import make_env
    import json

    env = make_env(difficulty=1, experiment_budget=15, seed=42)

    # Test reset
    obs_text, info = env.reset()
    assert isinstance(obs_text, str), "Observation should be a string"
    assert len(obs_text) > 100, "Observation text should be substantial"
    assert "difficulty" in info, "Info should contain difficulty"

    # Test experiment step
    action = json.dumps({"action": "experiment", "inputs": {"x": 1.0}})
    obs_text, reward, terminated, truncated, info = env.step(action)
    assert isinstance(obs_text, str), "Observation should be a string"
    assert isinstance(reward, float), "Reward should be a float"
    assert not terminated, "Should not terminate after one experiment"

    # Test predict (ends episode)
    action = json.dumps({"action": "predict", "predictions": [0.0] * 20})
    obs_text, reward, terminated, truncated, info = env.step(action)
    assert terminated, "Should terminate after predictions"

    env.close()
    print("        PASS - Gymnasium wrapper: reset, step, terminate all work correctly")
    return True


def run_agent_benchmark():
    """Run the heuristic agent across all levels and show results."""
    from hypothesis_engine import HypothesisEngine
    from hypothesis_engine.agents.heuristic_agent import HeuristicScientist

    print("\n  Agent Performance Benchmark (Heuristic Scientist)")
    print("  " + "-" * 55)

    agent = HeuristicScientist()
    results = []

    for level in range(1, 11):
        budget = 20 + level * 3
        env = HypothesisEngine(difficulty=level, experiment_budget=budget, seed=42 + level)
        agent.reset()
        obs = env.reset()

        done = False
        steps = 0
        while not done and steps < budget + 10:
            action, _ = agent.act(obs)
            obs, reward, done, info = env.step(action)
            steps += 1

        score = info.get("final_reward", {}).get("total_reward", 0) if done else 0
        passed = info.get("passed", False) if done else False
        results.append({"level": level, "score": score, "passed": passed, "steps": steps})

        status = "PASS" if passed else "FAIL"
        bar_width = int(score / 5)
        bar = "#" * bar_width + "." * (20 - bar_width)
        print(f"    Level {level:>2d}: [{bar}] {score:>5.1f}/100  {status}")

    passed_count = sum(1 for r in results if r["passed"])
    avg_score = sum(r["score"] for r in results) / len(results)
    print(f"\n    Passed: {passed_count}/10 | Average Score: {avg_score:.1f}/100")

    return results


if __name__ == "__main__":
    # Fix Windows encoding
    if sys.platform == "win32":
        import os
        os.environ.setdefault("PYTHONIOENCODING", "utf-8")
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        except Exception:
            pass

    print()
    print("=" * 60)
    print("  HYPOTHESIS ENGINE -- Environment Validation Benchmark")
    print("=" * 60)
    print()

    tests = [
        ("Determinism", test_determinism),
        ("Reward Signal", test_reward_signal),
        ("Difficulty Levels", test_all_difficulty_levels),
        ("Curriculum", test_curriculum_progression),
        ("Observation", test_observation_richness),
        ("Gymnasium", test_gymnasium_wrapper),
    ]

    passed = 0
    failed = 0

    start = time.time()
    for name, test_fn in tests:
        try:
            if test_fn():
                passed += 1
        except Exception as e:
            print(f"        FAIL - {e}")
            failed += 1

    elapsed = time.time() - start

    print()
    print("  " + "=" * 55)
    print(f"  Results: {passed}/{passed + failed} tests passed in {elapsed:.2f}s")
    print("  " + "=" * 55)

    # Run agent benchmark
    run_agent_benchmark()

    print()
    print("  Benchmark complete.")
    print()
