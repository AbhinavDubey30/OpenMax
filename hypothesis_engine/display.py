"""
Rich Terminal Display for the Hypothesis Engine.

Beautiful, polished terminal output using the Rich library for
hackathon demos and presentations.
"""

import sys
import os
import time
from typing import Dict, Any, List, Optional

# Fix Windows encoding for Unicode output
if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    from rich.columns import Columns
    from rich.rule import Rule
    from rich import box
    from rich.align import Align
    from rich.padding import Padding
    HAS_RICH = True
except ImportError:
    HAS_RICH = False


class Display:
    """Rich terminal display for the Hypothesis Engine."""

    def __init__(self, slow_mode: bool = False, delay: float = 0.3):
        if HAS_RICH:
            self.console = Console(force_terminal=True)
        else:
            self.console = None
        self.slow_mode = slow_mode
        self.delay = delay

    def _pause(self, extra: float = 0.0):
        if self.slow_mode:
            time.sleep(self.delay + extra)

    # ── Banner & Headers ─────────────────────────────────────────────────

    def show_banner(self):
        """Display the main title banner."""
        if not HAS_RICH:
            print("=" * 60)
            print("  HYPOTHESIS ENGINE")
            print("  Scientific Discovery RL Environment")
            print("=" * 60)
            return

        banner_text = Text()
        banner_text.append("  ╦ ╦╦ ╦╔═╗╔═╗╔╦╗╦ ╦╔═╗╔═╗╦╔═╗\n", style="bold bright_cyan")
        banner_text.append("  ╠═╣╚╦╝╠═╝║ ║ ║ ╠═╣║╣ ╚═╗║╚═╗\n", style="bold cyan")
        banner_text.append("  ╩ ╩ ╩ ╩  ╚═╝ ╩ ╩ ╩╚═╝╚═╝╩╚═╝\n", style="bold blue")
        banner_text.append("  ╔═╗╔╗╔╔═╗╦╔╗╔╔═╗\n", style="bold bright_magenta")
        banner_text.append("  ║╣ ║║║║ ╦║║║║║╣ \n", style="bold magenta")
        banner_text.append("  ╚═╝╝╚╝╚═╝╩╝╚╝╚═╝\n", style="bold purple")

        panel = Panel(
            banner_text,
            title="[bold white]v1.0[/]",
            subtitle="[dim]Scientific Discovery RL Environment[/]",
            border_style="bright_cyan",
            box=box.DOUBLE,
            padding=(1, 2),
        )
        self.console.print()
        self.console.print(panel)
        self.console.print()

    def show_menu(self) -> str:
        """Display the main menu and get user choice."""
        if not HAS_RICH:
            print("\nChoose a mode:")
            print("  [1] Watch AI Scientist (Heuristic Agent)")
            print("  [2] Watch LLM Scientist (Requires API Key)")
            print("  [3] Interactive Mode (You are the scientist!)")
            print("  [4] Run Benchmark (All 10 levels)")
            print("  [5] Quick Demo (Levels 1-3)")
            return input("\n> ").strip()

        menu_text = Text()
        menu_text.append("  [1] ", style="bold bright_green")
        menu_text.append("Watch AI Scientist ", style="bold white")
        menu_text.append("(Heuristic Agent)\n", style="dim")
        menu_text.append("  [2] ", style="bold bright_yellow")
        menu_text.append("Watch LLM Scientist ", style="bold white")
        menu_text.append("(Requires OPENAI_API_KEY)\n", style="dim")
        menu_text.append("  [3] ", style="bold bright_blue")
        menu_text.append("Interactive Mode ", style="bold white")
        menu_text.append("(You are the scientist!)\n", style="dim")
        menu_text.append("  [4] ", style="bold bright_magenta")
        menu_text.append("Full Benchmark ", style="bold white")
        menu_text.append("(All 10 difficulty levels)\n", style="dim")
        menu_text.append("  [5] ", style="bold bright_cyan")
        menu_text.append("Quick Demo ", style="bold white")
        menu_text.append("(Levels 1-3, great for presentations)\n", style="dim")

        panel = Panel(
            menu_text,
            title="[bold white]Choose Your Mode[/]",
            border_style="bright_white",
            padding=(1, 2),
        )
        self.console.print(panel)
        choice = input("  > ").strip()
        return choice

    # ── Episode Display ──────────────────────────────────────────────────

    def show_episode_start(self, episode: int, difficulty: int, world_briefing: Dict):
        """Display the start of a new episode."""
        if not HAS_RICH:
            print(f"\n{'='*60}")
            print(f"EPISODE {episode} | Level {difficulty}: {world_briefing.get('world_name', '???')}")
            print(f"{'='*60}")
            print(f"Description: {world_briefing.get('description', '')}")
            print(f"Variables: {world_briefing.get('variables', [])}")
            print(f"Ranges: {world_briefing.get('variable_ranges', {})}")
            return

        from .worlds import WorldGenerator
        level_name = WorldGenerator.DIFFICULTY_NAMES.get(difficulty, "Unknown")

        header = Text()
        header.append(f"EPISODE {episode}", style="bold bright_white")
        header.append(" | ", style="dim")
        header.append(f"Level {difficulty}", style="bold bright_yellow")
        header.append(f": {level_name}", style="bold yellow")

        self.console.print()
        self.console.print(Rule(style="bright_cyan"))
        self.console.print(Align.center(header))
        self.console.print(Rule(style="bright_cyan"))

        info_parts = []

        # World name
        world_name = world_briefing.get("world_name", "Unknown World")
        name_text = Text()
        name_text.append("World: ", style="bold")
        name_text.append(f'"{world_name}"', style="italic bright_cyan")
        info_parts.append(name_text)

        # Description
        desc = world_briefing.get("description", "No description.")
        desc_text = Text()
        desc_text.append("", style="bold")
        desc_text.append(desc, style="white")
        info_parts.append(desc_text)

        # Variables
        variables = world_briefing.get("variables", [])
        ranges = world_briefing.get("variable_ranges", {})
        var_text = Text()
        var_text.append("Variables: ", style="bold")
        for v in variables:
            lo, hi = ranges.get(v, [None, None])
            var_text.append(f"{v}", style="bold bright_green")
            var_text.append(f" [{lo}, {hi}]  ", style="dim")
        info_parts.append(var_text)

        # Hints
        if world_briefing.get("hints"):
            hint_text = Text()
            hint_text.append("Hint: ", style="bold yellow")
            hint_text.append(world_briefing["hints"][0], style="italic yellow")
            info_parts.append(hint_text)

        panel_content = Text("\n").join(info_parts)
        panel = Panel(
            panel_content,
            border_style="bright_blue",
            padding=(1, 2),
        )
        self.console.print(panel)
        self._pause(0.5)

    def show_experiment(self, exp_num: int, inputs: Dict, output: Any, reasoning: str = ""):
        """Display a single experiment result."""
        if not HAS_RICH:
            input_str = ", ".join(f"{k}={v}" for k, v in inputs.items())
            print(f"  Exp #{exp_num}: [{input_str}] → y = {output}  {reasoning}")
            return

        exp_text = Text()
        exp_text.append(f"  Experiment #{exp_num:>2d}  |  ", style="bold")

        for k, v in inputs.items():
            exp_text.append(f"{k}", style="bold bright_green")
            exp_text.append(f"={v:<8}  ", style="white")

        exp_text.append("→  ", style="dim")
        exp_text.append("y = ", style="bold")

        if output is not None:
            exp_text.append(f"{output}", style="bold bright_yellow")
        else:
            exp_text.append("ERROR", style="bold red")

        if reasoning:
            exp_text.append(f"  │  {reasoning}", style="italic dim")

        self.console.print(exp_text)
        self._pause()

    def show_experiment_table(self, experiments: List[Dict], variables: List[str]):
        """Display experiments as a formatted table."""
        if not HAS_RICH:
            for i, exp in enumerate(experiments):
                inputs = exp.get("inputs", {})
                output = exp.get("output", "N/A")
                input_str = ", ".join(f"{k}={v}" for k, v in inputs.items())
                print(f"  #{i+1}: [{input_str}] → y = {output}")
            return

        table = Table(
            title="Experiment Log",
            box=box.ROUNDED,
            border_style="bright_blue",
            show_lines=True,
            title_style="bold bright_cyan",
        )
        table.add_column("#", style="dim", width=4, justify="right")
        for v in variables:
            table.add_column(v, style="bright_green", justify="center")
        table.add_column("Output (y)", style="bold bright_yellow", justify="center")

        for i, exp in enumerate(experiments):
            inputs = exp.get("inputs", {})
            output = exp.get("output", "N/A")
            row = [str(i + 1)]
            for v in variables:
                row.append(str(inputs.get(v, "?")))
            row.append(str(output))
            table.add_row(*row)

        self.console.print()
        self.console.print(table)

    # ── Hypothesis Display ───────────────────────────────────────────────

    def show_hypothesis(self, hyp_num: int, expression: str, score_hint: str):
        """Display a hypothesis submission."""
        if not HAS_RICH:
            print(f"\n  Hypothesis #{hyp_num}: y = {expression}  [{score_hint}]")
            return

        color = "green" if score_hint == "high" else "yellow" if score_hint == "medium" else "red"

        hyp_text = Text()
        hyp_text.append(f"\n  Hypothesis #{hyp_num}: ", style="bold")
        hyp_text.append("y = ", style="dim")
        hyp_text.append(expression, style=f"bold {color}")

        quality_text = Text()
        if score_hint == "high":
            quality_text.append("     [PASS] Excellent match!", style="bold bright_green")
        elif score_hint == "medium":
            quality_text.append("     [PARTIAL] Partial match -- keep refining", style="bold yellow")
        else:
            quality_text.append("     [MISS] Poor match -- try more experiments", style="bold red")

        self.console.print(hyp_text)
        self.console.print(quality_text)
        self._pause(0.3)

    # ── Prediction Display ───────────────────────────────────────────────

    def show_prediction_results(self, per_case: List[Dict], variables: List[str], test_cases: List[Dict]):
        """Display prediction results in a table."""
        if not HAS_RICH:
            correct = sum(1 for c in per_case if c.get("correct"))
            total = len(per_case)
            print(f"\n  Predictions: {correct}/{total} correct")
            return

        table = Table(
            title="Prediction Results",
            box=box.ROUNDED,
            border_style="bright_magenta",
            show_lines=True,
            title_style="bold bright_magenta",
        )
        table.add_column("#", style="dim", width=4, justify="right")
        table.add_column("Inputs", style="bright_green", justify="center")
        table.add_column("Predicted", style="bright_yellow", justify="center")
        table.add_column("Actual", style="bright_cyan", justify="center")
        table.add_column("Error", justify="center")
        table.add_column("", width=3, justify="center")

        for i, case_result in enumerate(per_case[:15]):  # Show first 15
            inputs_str = ", ".join(
                f"{v}={test_cases[i].get(v, '?')}" for v in variables
            ) if i < len(test_cases) else "?"

            predicted = str(case_result.get("predicted", "N/A"))
            actual = str(case_result.get("actual", "N/A"))
            error = case_result.get("error", "N/A")
            correct = case_result.get("correct", False)

            error_str = f"{error:.3f}" if isinstance(error, (int, float)) else str(error)
            mark = "OK" if correct else "X"
            error_style = "green" if correct else "red"

            table.add_row(
                str(i + 1),
                inputs_str,
                predicted,
                actual,
                Text(error_str, style=error_style),
                mark,
            )

        if len(per_case) > 15:
            table.add_row("...", "...", "...", "...", "...", "...")

        self.console.print()
        self.console.print(table)

    # ── Score Display ────────────────────────────────────────────────────

    def show_final_score(self, reward_info: Dict, passed: bool, ground_truth: str):
        """Display the final score breakdown."""
        if not HAS_RICH:
            total = reward_info.get("total_reward", 0)
            print(f"\n{'='*50}")
            print(f"  FINAL SCORE: {total}/100  {'PASSED' if passed else 'FAILED'}")
            print(f"  Ground Truth: y = {ground_truth}")
            print(f"{'='*50}")
            return

        breakdown = reward_info.get("breakdown", {})

        score_table = Table(
            box=box.HEAVY,
            border_style="bright_yellow",
            show_lines=True,
            title_style="bold bright_yellow",
        )
        score_table.add_column("Component", style="bold white", min_width=25)
        score_table.add_column("Score", justify="center", min_width=10)
        score_table.add_column("Points", justify="center", min_width=12)
        score_table.add_column("Max", justify="center", min_width=8)
        score_table.add_column("Bar", min_width=20)

        for name, data in breakdown.items():
            display_name = name.replace("_", " ").title()
            score_val = data["score"]
            weighted = data["weighted"]
            max_pts = data["max_points"]

            bar_width = 15
            filled = int(score_val * bar_width)
            bar = "█" * filled + "░" * (bar_width - filled)

            if score_val >= 0.8:
                bar_style = "bright_green"
            elif score_val >= 0.5:
                bar_style = "bright_yellow"
            else:
                bar_style = "bright_red"

            score_table.add_row(
                display_name,
                f"{score_val:.0%}",
                f"{weighted:.1f}",
                f"{max_pts:.1f}",
                Text(bar, style=bar_style),
            )

        total = reward_info.get("total_reward", 0)
        total_style = "bold bright_green" if passed else "bold bright_red"
        status = "PASSED" if passed else "FAILED"

        score_table.add_section()
        score_table.add_row(
            Text("TOTAL", style="bold"),
            "",
            Text(f"{total:.1f}", style=total_style),
            "100.0",
            Text(status, style=total_style),
        )

        self.console.print()
        self.console.print(score_table)

        # Ground truth reveal
        gt_text = Text()
        gt_text.append("\n  Ground Truth: ", style="bold")
        gt_text.append("y = ", style="dim")
        gt_text.append(ground_truth, style="bold bright_cyan")
        self.console.print(gt_text)
        self.console.print()

    # ── Progress & Curriculum ────────────────────────────────────────────

    def show_curriculum_progress(self, progress: Dict):
        """Display curriculum progress overview."""
        if not HAS_RICH:
            print(f"\nProgress: Level {progress.get('current_difficulty', '?')}")
            print(f"Episodes: {progress.get('total_episodes', 0)}")
            return

        level_stats = progress.get("level_stats", {})
        current = progress.get("current_difficulty", 1)

        from .worlds import WorldGenerator

        progress_text = Text()
        for lvl in range(1, 11):
            name = WorldGenerator.DIFFICULTY_NAMES.get(lvl, "?")
            stats = level_stats.get(lvl, {})
            wins = stats.get("wins", 0)
            attempts = stats.get("attempts", 0)

            if lvl == current:
                marker = ">"
                style = "bold bright_yellow"
            elif wins > 0:
                marker = "+"
                style = "bold bright_green"
            elif attempts > 0:
                marker = "~"
                style = "bold yellow"
            else:
                marker = "-"
                style = "dim"

            line = f"  {marker} Level {lvl:>2d}: {name:<28s}"
            if attempts > 0:
                avg = stats.get("avg_reward", 0)
                line += f"  [{wins}/{attempts} passed, avg: {avg:.0f}]"

            progress_text.append(line + "\n", style=style)

        panel = Panel(
            progress_text,
            title="[bold white]Curriculum Progress[/]",
            border_style="bright_magenta",
            padding=(1, 2),
        )
        self.console.print()
        self.console.print(panel)

    # ── Agent Reasoning ──────────────────────────────────────────────────

    def show_agent_thinking(self, thought: str):
        """Display the agent's reasoning process."""
        if not HAS_RICH:
            print(f"  [think] {thought}")
            return

        thought_text = Text()
        thought_text.append("  [think] ", style="bold")
        thought_text.append(thought, style="italic bright_white")
        self.console.print(thought_text)
        self._pause()

    def show_phase_header(self, phase: str):
        """Display a phase transition header."""
        if not HAS_RICH:
            print(f"\n── {phase} ──")
            return

        self.console.print()
        self.console.print(Rule(f" {phase} ", style="bright_cyan"))
        self.console.print()
        self._pause(0.2)

    def show_info(self, message: str, style: str = "white"):
        """Display an informational message."""
        if not HAS_RICH:
            print(f"  [i] {message}")
            return

        self.console.print(f"  [i] {message}", style=style)

    def show_success(self, message: str):
        """Display a success message."""
        if not HAS_RICH:
            print(f"  [PASS] {message}")
            return

        self.console.print(f"  [PASS] {message}", style="bold bright_green")

    def show_warning(self, message: str):
        """Display a warning message."""
        if not HAS_RICH:
            print(f"  [WARN] {message}")
            return

        self.console.print(f"  [WARN] {message}", style="bold bright_yellow")

    def show_error(self, message: str):
        """Display an error message."""
        if not HAS_RICH:
            print(f"  [ERR] {message}")
            return

        self.console.print(f"  [ERR] {message}", style="bold bright_red")

    # ── Interactive Mode ─────────────────────────────────────────────────

    def prompt_experiment(self, variables: List[str], ranges: Dict) -> Dict[str, float]:
        """Prompt the user to enter experiment inputs."""
        if HAS_RICH:
            self.console.print()
            self.console.print(
                "  Enter input values (or 'done' to move to prediction phase):",
                style="bold bright_cyan",
            )
        else:
            print("\n  Enter input values (or 'done' to move to prediction phase):")

        inputs = {}
        for var in variables:
            lo, hi = ranges.get(var, [-10, 10])
            while True:
                val_str = input(f"    {var} [{lo}, {hi}]: ").strip()
                if val_str.lower() == "done":
                    return None  # Signal to move to next phase
                try:
                    val = float(val_str)
                    inputs[var] = val
                    break
                except ValueError:
                    print(f"    Invalid number. Please enter a value in [{lo}, {hi}]")

        return inputs

    def prompt_hypothesis(self) -> Optional[str]:
        """Prompt the user to enter a hypothesis."""
        if HAS_RICH:
            self.console.print(
                "\n  Enter your hypothesis (math expression, or 'skip'):",
                style="bold bright_yellow",
            )
        else:
            print("\n  Enter your hypothesis (math expression, or 'skip'):")

        expr = input("    y = ").strip()
        return expr if expr.lower() != "skip" else None

    def prompt_predictions(self, test_cases: List[Dict], variables: List[str]) -> List[float]:
        """Prompt the user to enter predictions."""
        if HAS_RICH:
            self.console.print(
                f"\n  Enter predictions for {len(test_cases)} test cases:",
                style="bold bright_magenta",
            )
        else:
            print(f"\n  Enter predictions for {len(test_cases)} test cases:")

        predictions = []
        for i, case in enumerate(test_cases):
            input_str = ", ".join(f"{v}={case[v]}" for v in variables)
            while True:
                val_str = input(f"    Test {i+1} [{input_str}] → y = ").strip()
                try:
                    val = float(val_str)
                    predictions.append(val)
                    break
                except ValueError:
                    print("    Invalid number. Please try again.")

        return predictions
