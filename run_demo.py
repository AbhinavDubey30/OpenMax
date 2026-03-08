#!/usr/bin/env python3
"""
Hypothesis Engine — Demo Runner

Run this script to experience the Scientific Discovery RL Environment.

Usage:
    python run_demo.py                  # Interactive menu
    python run_demo.py --auto           # Auto demo with heuristic agent
    python run_demo.py --interactive    # You play as the scientist
    python run_demo.py --llm            # Use LLM agent (needs OPENAI_API_KEY)
    python run_demo.py --benchmark      # Run all 10 levels
    python run_demo.py --quick          # Quick demo (levels 1-3)
    python run_demo.py --level 5        # Start at a specific level
"""

import sys
import os
import argparse
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hypothesis_engine.env import HypothesisEngine
from hypothesis_engine.display import Display
from hypothesis_engine.agents.heuristic_agent import HeuristicScientist


def run_episode_with_agent(env, agent, display, level, episode_num=1):
    """Run a single episode with an agent and display results."""
    agent.reset()
    obs = env.reset()

    world_briefing = obs.get("world", {})
    display.show_episode_start(episode_num, level, world_briefing)

    budget = obs.get("experiment_budget", 30)
    display.show_info(
        f"Experiment Budget: {budget} experiments | "
        f"Test Cases: {len(obs.get('test_cases', []))} predictions needed",
        style="bold bright_white"
    )
    display.show_phase_header("Phase 1: Exploration & Experimentation")

    # Agent loop
    max_steps = budget + 10  # Safety limit
    step = 0
    done = False

    while not done and step < max_steps:
        action, reasoning = agent.act(obs)
        action_type = action.get("action", "")

        if action_type == "experiment":
            inputs = action.get("inputs", {})
            display.show_agent_thinking(reasoning)
            obs, reward, done, info = env.step(action)

            result = info.get("experiment_result", {})
            output = result.get("output", "ERROR")
            exp_num = len(env.experiment_history)
            display.show_experiment(exp_num, inputs, output)

        elif action_type == "hypothesize":
            expression = action.get("expression", "?")
            display.show_phase_header("Phase 2: Hypothesis Formation")
            display.show_agent_thinking(reasoning)
            obs, reward, done, info = env.step(action)

            score_hint = obs.get("hypothesis_feedback", {}).get("score_hint", "low")
            hyp_num = len(env.hypothesis_history)
            display.show_hypothesis(hyp_num, expression, score_hint)

        elif action_type == "predict":
            display.show_phase_header("Phase 3: Prediction Challenge")
            display.show_agent_thinking(reasoning)
            obs, reward, done, info = env.step(action)

            if done:
                # Show prediction results
                pred_results = info.get("prediction_results", {})
                per_case = pred_results.get("per_case", [])
                variables = env.world.variables
                test_cases = env.world.test_cases

                display.show_prediction_results(per_case, variables, test_cases)

                # Show final score
                final_reward = info.get("final_reward", {})
                passed = info.get("passed", False)
                ground_truth = info.get("ground_truth_expr", "unknown")

                display.show_phase_header("Final Score")
                display.show_final_score(final_reward, passed, ground_truth)

                return {
                    "passed": passed,
                    "score": final_reward.get("total_reward", 0),
                    "ground_truth": ground_truth,
                    "hypothesis": env.hypothesis_history[-1]["expression"] if env.hypothesis_history else None,
                }

        elif action_type == "get_hint":
            obs, reward, done, info = env.step(action)
            hint = info.get("hint", "No hint available")
            display.show_info(f"Hint: {hint}", style="italic yellow")

        elif action_type == "get_status":
            obs, reward, done, info = env.step(action)

        step += 1

    return {"passed": False, "score": 0, "ground_truth": "timeout"}


def run_auto_demo(levels=None, start_level=1, seed_base=42):
    """Run the heuristic agent through specified levels."""
    display = Display(slow_mode=True, delay=0.15)
    display.show_banner()

    if levels is None:
        levels = list(range(start_level, 11))

    agent = HeuristicScientist()
    results = []

    display.show_info(
        f"Starting automated demo with {agent.name}",
        style="bold bright_cyan"
    )
    display.show_info(
        f"Levels: {levels[0]}-{levels[-1]} ({len(levels)} worlds to discover)",
        style="dim"
    )
    print()

    for i, level in enumerate(levels):
        budget = 20 + level * 3  # More budget for harder levels
        env = HypothesisEngine(
            difficulty=level,
            experiment_budget=budget,
            seed=seed_base + level,
        )

        result = run_episode_with_agent(env, agent, display, level, episode_num=i + 1)
        results.append({"level": level, **result})

        if result["passed"]:
            display.show_success(f"Level {level} PASSED! Score: {result['score']:.1f}/100")
        else:
            display.show_warning(f"Level {level} not passed. Score: {result['score']:.1f}/100")

        print()
        time.sleep(0.5)

    # Final summary
    show_summary(display, results)
    return results


def run_llm_demo(levels=None, start_level=1, model="gpt-4o-mini", seed_base=42):
    """Run the LLM agent through specified levels."""
    display = Display(slow_mode=True, delay=0.2)
    display.show_banner()

    try:
        from hypothesis_engine.agents.llm_agent import LLMScientist
        agent = LLMScientist(model=model)
    except ImportError:
        display.show_error("openai package not installed. Run: pip install openai")
        return
    except Exception as e:
        display.show_error(f"Failed to initialize LLM agent: {e}")
        display.show_info("Make sure OPENAI_API_KEY is set in your environment.")
        return

    if levels is None:
        levels = list(range(start_level, min(start_level + 3, 11)))

    results = []

    display.show_info(
        f"Starting LLM demo with {agent.name}",
        style="bold bright_yellow"
    )
    print()

    for i, level in enumerate(levels):
        budget = 25 + level * 3
        env = HypothesisEngine(
            difficulty=level,
            experiment_budget=budget,
            seed=seed_base + level,
        )

        result = run_episode_with_agent(env, agent, display, level, episode_num=i + 1)
        results.append({"level": level, **result})

        if result["passed"]:
            display.show_success(f"Level {level} PASSED! Score: {result['score']:.1f}/100")
        else:
            display.show_warning(f"Level {level} not passed. Score: {result['score']:.1f}/100")

        print()
        time.sleep(0.5)

    show_summary(display, results)
    return results


def run_interactive(start_level=1, seed_base=42):
    """Run in interactive mode where the user is the scientist."""
    display = Display(slow_mode=False)
    display.show_banner()

    display.show_info(
        "INTERACTIVE MODE — You are the scientist!",
        style="bold bright_green"
    )

    level = start_level
    while level <= 10:
        budget = 25 + level * 3
        env = HypothesisEngine(
            difficulty=level,
            experiment_budget=budget,
            seed=seed_base + level,
        )
        obs = env.reset()

        world_briefing = obs.get("world", {})
        display.show_episode_start(1, level, world_briefing)

        variables = world_briefing.get("variables", [])
        ranges = world_briefing.get("variable_ranges", {})

        display.show_info(
            f"Budget: {budget} experiments | "
            f"Test Cases: {len(obs.get('test_cases', []))} predictions needed",
            style="bold"
        )
        print()

        # Show action space
        print(env.get_action_space_description())
        print()

        # Exploration phase
        display.show_phase_header("Phase 1: Exploration")
        done = False

        while not done:
            remaining = obs.get("experiments_remaining", 0)
            display.show_info(f"Experiments remaining: {remaining}", style="bold bright_cyan")

            inputs = display.prompt_experiment(variables, ranges)

            if inputs is None:
                # Move to hypothesis phase
                break

            obs, reward, done, info = env.step({
                "action": "experiment",
                "inputs": inputs
            })

            result = info.get("experiment_result", {})
            output = result.get("output", result.get("error", "ERROR"))
            exp_num = len(env.experiment_history)
            display.show_experiment(exp_num, inputs, output)

        if not done:
            # Show experiment history
            if env.experiment_history:
                display.show_experiment_table(env.experiment_history, variables)

            # Hypothesis phase
            display.show_phase_header("Phase 2: Hypothesis")
            hypothesis = display.prompt_hypothesis()

            if hypothesis:
                obs, reward, done, info = env.step({
                    "action": "hypothesize",
                    "expression": hypothesis
                })
                score = info.get("hypothesis_score", 0)
                score_hint = "high" if score >= 0.8 else "medium" if score >= 0.5 else "low"
                display.show_hypothesis(1, hypothesis, score_hint)

        if not done:
            # Prediction phase
            display.show_phase_header("Phase 3: Prediction")
            test_cases = obs.get("test_cases", [])

            display.show_info(
                "You can enter predictions manually, or type 'auto' to use your hypothesis.",
                style="dim"
            )

            auto = input("  Enter 'auto' to predict from hypothesis, or 'manual': ").strip().lower()

            if auto == "auto" and hypothesis:
                from hypothesis_engine.verifier import SafeMathEvaluator
                evaluator = SafeMathEvaluator()
                predictions = []
                for case in test_cases:
                    try:
                        pred = evaluator.evaluate(hypothesis, case)
                        predictions.append(round(pred, 4))
                    except ValueError:
                        predictions.append(0.0)
            else:
                predictions = display.prompt_predictions(test_cases, variables)

            obs, reward, done, info = env.step({
                "action": "predict",
                "predictions": predictions
            })

            pred_results = info.get("prediction_results", {})
            per_case = pred_results.get("per_case", [])
            display.show_prediction_results(per_case, variables, test_cases)

            final_reward = info.get("final_reward", {})
            passed = info.get("passed", False)
            ground_truth = info.get("ground_truth_expr", "unknown")

            display.show_phase_header("Final Score")
            display.show_final_score(final_reward, passed, ground_truth)

            if passed:
                display.show_success("Congratulations! You passed this level!")
                cont = input("\n  Continue to next level? (y/n): ").strip().lower()
                if cont != "y":
                    break
                level += 1
            else:
                retry = input("\n  Retry this level? (y/n): ").strip().lower()
                if retry != "y":
                    break


def show_summary(display, results):
    """Show a summary of all results."""
    display.show_phase_header("Session Summary")

    try:
        from rich.table import Table
        from rich import box

        table = Table(
            title="Results Overview",
            box=box.HEAVY,
            border_style="bright_cyan",
            show_lines=True,
        )
        table.add_column("Level", style="bold", justify="center")
        table.add_column("Score", justify="center")
        table.add_column("Status", justify="center")
        table.add_column("Ground Truth", style="dim")
        table.add_column("Agent's Hypothesis", style="italic")

        total_score = 0
        passed_count = 0

        for r in results:
            score = r.get("score", 0)
            passed = r.get("passed", False)
            total_score += score
            if passed:
                passed_count += 1

            score_style = "bright_green" if score >= 60 else "bright_yellow" if score >= 30 else "bright_red"
            status = "PASS" if passed else "FAIL"

            from rich.text import Text
            table.add_row(
                str(r.get("level", "?")),
                Text(f"{score:.1f}", style=score_style),
                status,
                f"y = {r.get('ground_truth', '?')}",
                f"y = {r.get('hypothesis', '?')}",
            )

        display.console.print()
        display.console.print(table)

        avg_score = total_score / len(results) if results else 0
        display.show_info(
            f"\n  Levels Passed: {passed_count}/{len(results)} | "
            f"Average Score: {avg_score:.1f}/100 | "
            f"Total Score: {total_score:.1f}/{len(results) * 100}",
            style="bold bright_white"
        )

    except ImportError:
        print("\n  Summary:")
        for r in results:
            status = "PASS" if r.get("passed") else "FAIL"
            print(f"  Level {r.get('level', '?')}: {r.get('score', 0):.1f}/100 [{status}]")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Hypothesis Engine — Scientific Discovery RL Environment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_demo.py                  # Interactive menu
  python run_demo.py --auto           # Watch AI solve all 10 levels
  python run_demo.py --quick          # Quick demo (levels 1-3)
  python run_demo.py --interactive    # Play as the scientist
  python run_demo.py --level 5        # Start at level 5
  python run_demo.py --llm            # Use GPT-4 as the scientist
        """
    )
    parser.add_argument("--auto", action="store_true", help="Run heuristic agent automatically")
    parser.add_argument("--quick", action="store_true", help="Quick demo (levels 1-3)")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    parser.add_argument("--llm", action="store_true", help="Use LLM agent")
    parser.add_argument("--benchmark", action="store_true", help="Full benchmark (all 10 levels)")
    parser.add_argument("--level", type=int, default=1, help="Starting difficulty level (1-10)")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="LLM model to use")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    if args.auto:
        run_auto_demo(start_level=args.level, seed_base=args.seed)
    elif args.quick:
        run_auto_demo(levels=[1, 2, 3], seed_base=args.seed)
    elif args.interactive:
        run_interactive(start_level=args.level, seed_base=args.seed)
    elif args.llm:
        run_llm_demo(start_level=args.level, model=args.model, seed_base=args.seed)
    elif args.benchmark:
        run_auto_demo(levels=list(range(1, 11)), seed_base=args.seed)
    else:
        # Interactive menu
        display = Display()
        display.show_banner()
        choice = display.show_menu()

        if choice == "1":
            run_auto_demo(start_level=args.level, seed_base=args.seed)
        elif choice == "2":
            run_llm_demo(start_level=args.level, model=args.model, seed_base=args.seed)
        elif choice == "3":
            run_interactive(start_level=args.level, seed_base=args.seed)
        elif choice == "4":
            run_auto_demo(levels=list(range(1, 11)), seed_base=args.seed)
        elif choice == "5":
            run_auto_demo(levels=[1, 2, 3], seed_base=args.seed)
        else:
            print("Invalid choice. Running quick demo...")
            run_auto_demo(levels=[1, 2, 3], seed_base=args.seed)


if __name__ == "__main__":
    main()
