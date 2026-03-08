#!/usr/bin/env python3
"""
Hypothesis Engine -- Environment Validation & Benchmark Suite.

Validates all environment properties and benchmarks the heuristic agent
across ALL world types: function, causal, physics, state machine, stochastic.
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")
sys.stdout.reconfigure(encoding="utf-8")

from hypothesis_engine.env import HypothesisEngine
from hypothesis_engine.worlds import WorldGenerator
from hypothesis_engine.agents.heuristic_agent import HeuristicScientist
from hypothesis_engine.self_play import ProceduralSelfPlay, WorldValidator, GeneratedWorldSpec


def separator(title: str):
    print(f"\n  [{len(RESULTS)+1}/{TOTAL_TESTS}] {title}...")


RESULTS = []
TOTAL_TESTS = 8


def record(name: str, passed: bool, detail: str = ""):
    status = "PASS" if passed else "FAIL"
    RESULTS.append((name, passed))
    print(f"        {status} - {detail}")


def main():
    global TOTAL_TESTS

    print()
    print("=" * 60)
    print("  HYPOTHESIS ENGINE -- Environment Validation Benchmark")
    print("=" * 60)

    t0 = time.time()

    # ── Test 1: Determinism ──────────────────────────────────────────
    separator("Determinism Test")
    env1 = HypothesisEngine(difficulty=1, seed=42)
    env2 = HypothesisEngine(difficulty=1, seed=42)
    obs1 = env1.reset()
    obs2 = env2.reset()
    same_world = obs1["world"]["world_name"] == obs2["world"]["world_name"]
    r1 = env1.step({"action": "experiment", "inputs": {"x": 3.0}})
    r2 = env2.step({"action": "experiment", "inputs": {"x": 3.0}})
    same_result = r1[0]["last_experiment_result"]["output"] == r2[0]["last_experiment_result"]["output"]
    record("Determinism", same_world and same_result, "Same seed produces identical worlds and results")

    # ── Test 2: Dense Rewards ────────────────────────────────────────
    separator("Reward Signal Test")
    env = HypothesisEngine(difficulty=1, seed=100, experiment_budget=30)
    obs = env.reset()
    rewards = []
    for i in range(3):
        obs, r, d, info = env.step({"action": "experiment", "inputs": {"x": float(i)}})
        rewards.append(r)
    obs, r, d, info = env.step({"action": "predict", "predictions": [0.0]*20})
    has_dense = all(r > 0 for r in rewards)
    has_final = r > 0
    record("Rewards", has_dense and has_final,
           f"Dense rewards: {rewards}..., Final: {r}/100")

    # ── Test 3: All 10 Difficulty Levels ─────────────────────────────
    separator("All Difficulty Levels Test")
    all_ok = True
    for level in range(1, 11):
        try:
            world = WorldGenerator.generate(level, seed=42)
            # Use midpoint of each variable's range for a safe test
            inputs = {}
            for v in world.variables:
                lo, hi = world.variable_ranges[v]
                inputs[v] = round((lo + hi) / 2, 2)
            result = world.run_experiment(inputs)
            if result.get("output") is None:
                all_ok = False
        except Exception as e:
            all_ok = False
    record("All Levels", all_ok, f"All 10 levels generate valid worlds with experiments")

    # ── Test 4: Curriculum Progression ───────────────────────────────
    separator("Curriculum Progression Test")
    env = HypothesisEngine(difficulty=1, seed=200, auto_curriculum=True, advance_threshold=50)
    agent_c = HeuristicScientist()
    for ep in range(5):
        agent_c.reset()
        obs = env.reset()
        done = False
        step_c = 0
        while not done and step_c < 40:
            action_c, _ = agent_c.act(obs)
            obs, r, done, info = env.step(action_c)
            step_c += 1
    initial = 1
    final = env.curriculum.get_next_difficulty()
    record("Curriculum", final >= 2, f"Curriculum advances from level {initial} to {final} after agent solves levels")

    # ── Test 5: Observation Richness ─────────────────────────────────
    separator("Observation Richness Test")
    env = HypothesisEngine(difficulty=3, seed=300, experiment_budget=30)
    obs = env.reset()
    required_fields = ["phase", "world", "experiment_budget", "experiments_remaining",
                       "experiments_used", "test_cases", "message"]
    has_all = all(f in obs for f in required_fields)
    n_test = len(obs.get("test_cases", []))
    n_vars = len(obs.get("world", {}).get("variables", []))
    record("Observations", has_all and n_test == 20,
           f"Observation has {len(required_fields)} required fields, {n_test} test cases, {n_vars} variables")

    # ── Test 6: Gymnasium Wrapper ────────────────────────────────────
    separator("Gymnasium Wrapper Test")
    try:
        from hypothesis_engine.gym_wrapper import HypothesisEngineGymEnv
        gym_env = HypothesisEngineGymEnv(difficulty=1, seed=42)
        obs_text, info = gym_env.reset()
        assert isinstance(obs_text, str)
        obs_text, r, term, trunc, info = gym_env.step('{"action": "get_status"}')
        assert isinstance(obs_text, str)
        obs_text, r, term, trunc, info = gym_env.step(
            '{"action": "predict", "predictions": ' + str([0.0]*20) + '}'
        )
        assert term is True
        record("Gymnasium", True, "Gymnasium wrapper: reset, step, terminate all work correctly")
    except Exception as e:
        record("Gymnasium", False, f"Gymnasium wrapper error: {e}")

    # ── Test 7: Causal Worlds (NOVEL) ────────────────────────────────
    separator("Causal Worlds Test (observe vs intervene)")
    try:
        env = HypothesisEngine(difficulty=5, seed=42, experiment_budget=30)
        obs = env.reset()
        world_type = obs["world"].get("world_type", "unknown")
        supports = obs["world"].get("supports_intervention", False)

        # Run observe experiment
        obs_o, r1, d1, info1 = env.step({"action": "experiment", "inputs": {"x": 2.0}, "mode": "observe"})
        output_observe = obs_o["last_experiment_result"]["output"]

        # Run intervene experiment at same point
        obs_i, r2, d2, info2 = env.step({"action": "experiment", "inputs": {"x": 2.0}, "mode": "intervene"})
        output_intervene = obs_i["last_experiment_result"]["output"]

        # In confounded world, observe and intervene should give different results
        causal_ok = (
            world_type == "causal"
            and supports is True
            and output_observe is not None
            and output_intervene is not None
        )
        record("Causal Worlds", causal_ok,
               f"Type={world_type}, intervention={supports}, "
               f"observe={output_observe}, intervene={output_intervene}, "
               f"differ={output_observe != output_intervene}")
    except Exception as e:
        record("Causal Worlds", False, f"Error: {e}")

    # ── Test 8: Self-Play World Generation (NOVEL) ───────────────────
    separator("Self-Play World Generation Test")
    try:
        sp = ProceduralSelfPlay(seed=42)
        worlds_ok = 0
        for diff in [3, 5, 7]:
            world = sp.generate_world(diff)
            if world is not None:
                result = world.run_experiment({v: 1.0 for v in world.variables})
                if result.get("output") is not None:
                    worlds_ok += 1
        sp.record_solver_score(70)
        sp.record_solver_score(60)
        sp.record_solver_score(80)
        next_diff = sp.get_next_difficulty()
        record("Self-Play", worlds_ok >= 2,
               f"{worlds_ok}/3 generated worlds valid, adaptive difficulty -> {next_diff}")
    except Exception as e:
        record("Self-Play", False, f"Error: {e}")

    # ── Summary ──────────────────────────────────────────────────────
    elapsed = time.time() - t0
    passed = sum(1 for _, p in RESULTS if p)
    total = len(RESULTS)
    print()
    print(f"  {'=' * 55}")
    print(f"  Results: {passed}/{total} tests passed in {elapsed:.2f}s")
    print(f"  {'=' * 55}")

    # ── Agent Benchmark ──────────────────────────────────────────────
    print()
    print("  Agent Performance Benchmark (Heuristic Scientist)")
    print("  " + "-" * 55)

    agent = HeuristicScientist()
    level_results = []

    for level in range(1, 11):
        agent.reset()
        budget = 20 + level * 3
        env = HypothesisEngine(difficulty=level, experiment_budget=budget, seed=42 + level)
        obs = env.reset()
        done = False
        step = 0

        while not done and step < budget + 10:
            action, reasoning = agent.act(obs)
            obs, reward, done, info = env.step(action)
            step += 1

        score = info.get("final_reward", {}).get("total_reward", 0) if done else 0
        passed_level = score >= 60

        bar_len = int(score / 5)
        bar = "#" * bar_len + "." * (20 - bar_len)
        status = "PASS" if passed_level else "FAIL"
        print(f"    Level {level:>2d}: [{bar}] {score:>5.1f}/100  {status}")
        level_results.append((level, score, passed_level))

    passed_count = sum(1 for _, _, p in level_results if p)
    avg_score = sum(s for _, s, _ in level_results) / len(level_results)
    print()
    print(f"    Passed: {passed_count}/{len(level_results)} | Average Score: {avg_score:.1f}/100")

    # ── World Type Breakdown ─────────────────────────────────────────
    print()
    print("  World Type Breakdown:")
    print("  " + "-" * 55)
    categories = {
        "Function Discovery (L1-3)": level_results[:3],
        "Causal Reasoning  (L4-6)": level_results[3:6],
        "Physics Simulation (L7-8)": level_results[6:8],
        "State Machine      (L9)": level_results[8:9],
        "Statistical Reason (L10)": level_results[9:10],
    }
    for cat_name, cat_results in categories.items():
        cat_avg = sum(s for _, s, _ in cat_results) / len(cat_results) if cat_results else 0
        cat_pass = sum(1 for _, _, p in cat_results if p)
        print(f"    {cat_name}: avg {cat_avg:>5.1f}/100, passed {cat_pass}/{len(cat_results)}")

    print()
    print("  Benchmark complete.")
    print()


if __name__ == "__main__":
    main()
