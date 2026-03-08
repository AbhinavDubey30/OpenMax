"""
Procedural World Generator for the Hypothesis Engine.

Generates black-box systems with known ground-truth rules at varying
difficulty levels. Each world is a self-contained scientific mystery
that an agent must unravel through experimentation.

Difficulty Levels:
    1  - Linear (single variable)
    2  - Polynomial (single variable)
    3  - Multi-variable Linear
    4  - Conditional / Piecewise
    5  - Interaction Effects
    6  - Trigonometric
    7  - Stochastic (noisy)
    8  - Hidden Variables
    9  - Dynamic / Stateful
    10 - Compositional
"""

import math
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Callable, Tuple, Any, Optional


@dataclass
class World:
    """A procedurally generated world with hidden rules to discover."""

    name: str
    description: str
    difficulty: int
    category: str
    variables: List[str]
    variable_ranges: Dict[str, Tuple[float, float]]
    ground_truth_fn: Callable[..., float]
    ground_truth_expr: str
    test_cases: List[Dict[str, float]] = field(default_factory=list)
    noise_std: float = 0.0
    hints: List[str] = field(default_factory=list)
    is_stateful: bool = False
    _rng: Optional[np.random.Generator] = field(default=None, repr=False)
    _state: Dict[str, Any] = field(default_factory=dict, repr=False)
    _experiment_count: int = field(default=0, repr=False)

    def run_experiment(self, inputs: Dict[str, float]) -> Dict[str, Any]:
        """Run a single experiment with given input values."""
        for var in self.variables:
            if var not in inputs:
                raise ValueError(
                    f"Missing input variable: '{var}'. Required: {self.variables}"
                )

        for var, val in inputs.items():
            if var in self.variable_ranges:
                lo, hi = self.variable_ranges[var]
                if val < lo or val > hi:
                    return {
                        "inputs": inputs,
                        "output": None,
                        "error": f"Variable '{var}'={val} out of range [{lo}, {hi}]",
                    }

        try:
            if self.is_stateful:
                output = self.ground_truth_fn(
                    _state=self._state,
                    _exp_num=self._experiment_count,
                    **{v: inputs[v] for v in self.variables},
                )
            else:
                output = self.ground_truth_fn(
                    **{v: inputs[v] for v in self.variables}
                )

            if self.noise_std > 0 and self._rng is not None:
                output += self._rng.normal(0, self.noise_std)

            output = float(output)
            if math.isnan(output) or math.isinf(output):
                return {"inputs": inputs, "output": None, "error": "Undefined result"}

        except Exception as e:
            return {"inputs": inputs, "output": None, "error": str(e)}

        self._experiment_count += 1
        return {"inputs": inputs, "output": round(output, 4)}

    def generate_test_cases(self, n: int = 20) -> List[Dict[str, float]]:
        """Generate n random test cases within variable ranges."""
        rng = self._rng if self._rng is not None else np.random.default_rng(42)
        cases = []
        for _ in range(n):
            case = {}
            for var in self.variables:
                lo, hi = self.variable_ranges[var]
                case[var] = round(float(rng.uniform(lo, hi)), 2)
            cases.append(case)
        self.test_cases = cases
        return cases

    def get_test_answers(self) -> List[Optional[float]]:
        """Get ground truth answers for the test cases (for scoring)."""
        saved_count = self._experiment_count
        saved_state = dict(self._state)
        answers = []
        for case in self.test_cases:
            result = self.run_experiment(case)
            answers.append(result.get("output"))
        self._experiment_count = saved_count
        self._state = saved_state
        return answers

    def reset_state(self):
        """Reset stateful world to initial conditions."""
        self._state = {}
        self._experiment_count = 0

    def get_agent_briefing(self) -> Dict[str, Any]:
        """Get the information packet shown to the agent at episode start."""
        return {
            "world_name": self.name,
            "description": self.description,
            "difficulty": self.difficulty,
            "category": self.category,
            "variables": self.variables,
            "variable_ranges": {
                v: list(r) for v, r in self.variable_ranges.items()
            },
            "hints": self.hints,
            "is_stateful": self.is_stateful,
        }


class WorldGenerator:
    """Procedurally generates scientific worlds at 10 difficulty levels."""

    DIFFICULTY_NAMES = {
        1: "Linear Discovery",
        2: "Polynomial Patterns",
        3: "Multi-Variable Linear",
        4: "Conditional Logic",
        5: "Interaction Effects",
        6: "Trigonometric Waves",
        7: "Signal in the Noise",
        8: "Hidden Variables",
        9: "Dynamic Systems",
        10: "Compositional Complexity",
    }

    @staticmethod
    def generate(difficulty: int, seed: Optional[int] = None) -> World:
        """Generate a world at the specified difficulty level (1-10)."""
        if difficulty < 1 or difficulty > 10:
            raise ValueError(f"Difficulty must be 1-10, got {difficulty}")

        rng = np.random.default_rng(seed)

        generators = {
            1: WorldGenerator._gen_linear_1var,
            2: WorldGenerator._gen_polynomial_1var,
            3: WorldGenerator._gen_linear_multivar,
            4: WorldGenerator._gen_conditional,
            5: WorldGenerator._gen_interaction,
            6: WorldGenerator._gen_trigonometric,
            7: WorldGenerator._gen_stochastic,
            8: WorldGenerator._gen_hidden_variable,
            9: WorldGenerator._gen_dynamic,
            10: WorldGenerator._gen_compositional,
        }

        world = generators[difficulty](rng)
        world._rng = rng
        world.generate_test_cases(20)
        return world

    @staticmethod
    def _nonzero_int(rng, lo, hi):
        """Generate a non-zero random integer in [lo, hi]."""
        val = 0
        while val == 0:
            val = int(rng.integers(lo, hi + 1))
        return val

    # ── Level 1: Linear single variable ─────────────────────────────────

    @staticmethod
    def _gen_linear_1var(rng: np.random.Generator) -> World:
        """y = a*x + b"""
        a = WorldGenerator._nonzero_int(rng, -5, 5)
        b = int(rng.integers(-10, 11))

        def fn(x):
            return a * x + b

        expr = f"{a}*x + ({b})" if b < 0 else f"{a}*x + {b}"

        return World(
            name="The Straight Line",
            description=(
                "This system accepts a single input variable (x) and produces "
                "a single output (y). The relationship appears to be smooth, "
                "continuous, and highly predictable. Simple patterns may "
                "reveal themselves with just a few experiments."
            ),
            difficulty=1,
            category="linear",
            variables=["x"],
            variable_ranges={"x": (-10.0, 10.0)},
            ground_truth_fn=fn,
            ground_truth_expr=expr,
            hints=[
                "Try varying x by equal steps and observe how the output changes.",
                "The output changes at a constant rate.",
            ],
        )

    # ── Level 2: Polynomial single variable ─────────────────────────────

    @staticmethod
    def _gen_polynomial_1var(rng: np.random.Generator) -> World:
        """y = a*x^2 + b*x + c"""
        a = WorldGenerator._nonzero_int(rng, -3, 3)
        b = int(rng.integers(-5, 6))
        c = int(rng.integers(-10, 11))

        def fn(x):
            return a * x ** 2 + b * x + c

        expr = f"{a}*x**2 + ({b})*x + ({c})"

        return World(
            name="The Parabola's Secret",
            description=(
                "This system accepts a single input (x) and produces an output (y). "
                "The relationship is smooth but may not be straight. The output "
                "could curve or bend in interesting ways. Pay attention to how "
                "the rate of change itself changes."
            ),
            difficulty=2,
            category="polynomial",
            variables=["x"],
            variable_ranges={"x": (-5.0, 5.0)},
            ground_truth_fn=fn,
            ground_truth_expr=expr,
            hints=[
                "Look at the second differences (differences of differences).",
                "Symmetry around a point may be a clue.",
            ],
        )

    # ── Level 3: Multi-variable linear ──────────────────────────────────

    @staticmethod
    def _gen_linear_multivar(rng: np.random.Generator) -> World:
        """y = a*x1 + b*x2 + c"""
        a = WorldGenerator._nonzero_int(rng, -5, 5)
        b = WorldGenerator._nonzero_int(rng, -5, 5)
        c = int(rng.integers(-10, 11))

        def fn(x1, x2):
            return a * x1 + b * x2 + c

        expr = f"{a}*x1 + ({b})*x2 + ({c})"

        return World(
            name="The Two-Body Problem",
            description=(
                "This system accepts two input variables (x1, x2) and produces "
                "a single output (y). Each variable may independently affect the "
                "output. Try isolating variables to understand individual effects, "
                "then check if they combine simply."
            ),
            difficulty=3,
            category="linear_multivar",
            variables=["x1", "x2"],
            variable_ranges={"x1": (-10.0, 10.0), "x2": (-10.0, 10.0)},
            ground_truth_fn=fn,
            ground_truth_expr=expr,
            hints=[
                "Hold one variable constant and vary the other.",
                "Each variable might contribute independently to the output.",
            ],
        )

    # ── Level 4: Conditional / Piecewise ────────────────────────────────

    @staticmethod
    def _gen_conditional(rng: np.random.Generator) -> World:
        """if x > threshold: y = a*x + b  else: y = c*x + d"""
        threshold = int(rng.choice([-2, -1, 0, 1, 2]))
        a = WorldGenerator._nonzero_int(rng, -4, 4)
        b = int(rng.integers(-5, 6))
        c = WorldGenerator._nonzero_int(rng, -4, 4)
        d = int(rng.integers(-5, 6))
        # Ensure different slopes so there IS a detectable boundary
        while c == a:
            c = WorldGenerator._nonzero_int(rng, -4, 4)

        def fn(x):
            if x > threshold:
                return a * x + b
            else:
                return c * x + d

        expr = f"({a}*x + {b}) if x > {threshold} else ({c}*x + {d})"

        return World(
            name="The Fork in the Road",
            description=(
                "This system accepts a single input (x) and produces an output (y). "
                "WARNING: The system may behave differently in different regions "
                "of the input space. There might be a critical threshold where "
                "the rules change. Scan across the full range carefully."
            ),
            difficulty=4,
            category="conditional",
            variables=["x"],
            variable_ranges={"x": (-10.0, 10.0)},
            ground_truth_fn=fn,
            ground_truth_expr=expr,
            hints=[
                "The system has distinct regimes — look for a breakpoint.",
                "Try testing many evenly-spaced values across the range.",
            ],
        )

    # ── Level 5: Interaction effects ────────────────────────────────────

    @staticmethod
    def _gen_interaction(rng: np.random.Generator) -> World:
        """y = a*x1*x2 + b*x1 + c*x2 + d"""
        a = WorldGenerator._nonzero_int(rng, -3, 3)
        b = int(rng.integers(-4, 5))
        c = int(rng.integers(-4, 5))
        d = int(rng.integers(-8, 9))

        def fn(x1, x2):
            return a * x1 * x2 + b * x1 + c * x2 + d

        expr = f"{a}*x1*x2 + ({b})*x1 + ({c})*x2 + ({d})"

        return World(
            name="The Entangled Variables",
            description=(
                "This system accepts two inputs (x1, x2) and produces an output (y). "
                "IMPORTANT: The variables may not act independently — changing one "
                "variable might alter the effect of the other. The whole may be "
                "greater than the sum of its parts."
            ),
            difficulty=5,
            category="interaction",
            variables=["x1", "x2"],
            variable_ranges={"x1": (-5.0, 5.0), "x2": (-5.0, 5.0)},
            ground_truth_fn=fn,
            ground_truth_expr=expr,
            hints=[
                "Test variables independently, then together.",
                "If the combined effect differs from individual effects, there's an interaction.",
            ],
        )

    # ── Level 6: Trigonometric ──────────────────────────────────────────

    @staticmethod
    def _gen_trigonometric(rng: np.random.Generator) -> World:
        """y = a*sin(b*x) + c"""
        a = WorldGenerator._nonzero_int(rng, -4, 4)
        b = int(rng.choice([1, 2, 3]))
        c = int(rng.integers(-5, 6))

        def fn(x):
            return a * math.sin(b * x) + c

        expr = f"{a}*sin({b}*x) + ({c})"

        return World(
            name="The Oscillator",
            description=(
                "This system accepts a single input (x) and produces an output (y). "
                "The output appears to oscillate or wave. It is bounded and periodic. "
                "Look for repeating patterns and try to determine the frequency "
                "and amplitude of the oscillation."
            ),
            difficulty=6,
            category="trigonometric",
            variables=["x"],
            variable_ranges={"x": (-6.28, 6.28)},
            ground_truth_fn=fn,
            ground_truth_expr=expr,
            hints=[
                "The output repeats — find the period.",
                "Check the maximum and minimum output values for amplitude.",
            ],
        )

    # ── Level 7: Stochastic (noisy) ────────────────────────────────────

    @staticmethod
    def _gen_stochastic(rng: np.random.Generator) -> World:
        """y = a*x^2 + b*x + c + noise"""
        a = WorldGenerator._nonzero_int(rng, -2, 2)
        b = int(rng.integers(-4, 5))
        c = int(rng.integers(-8, 9))
        noise_std = float(rng.choice([1.0, 1.5, 2.0]))

        def fn(x):
            return a * x ** 2 + b * x + c

        expr = f"{a}*x**2 + ({b})*x + ({c}) + N(0, {noise_std}^2)"

        return World(
            name="Through the Fog",
            description=(
                "This system accepts a single input (x) and produces an output (y). "
                "WARNING: The output contains random noise! Running the same "
                "experiment twice may yield slightly different results. You must "
                "find the underlying signal through the noise. Consider running "
                "repeated experiments at the same input values."
            ),
            difficulty=7,
            category="stochastic",
            variables=["x"],
            variable_ranges={"x": (-5.0, 5.0)},
            ground_truth_fn=fn,
            ground_truth_expr=expr,
            noise_std=noise_std,
            hints=[
                "Repeat experiments at the same x to average out noise.",
                "The underlying pattern is deterministic — only the noise is random.",
            ],
        )

    # ── Level 8: Hidden variable ────────────────────────────────────────

    @staticmethod
    def _gen_hidden_variable(rng: np.random.Generator) -> World:
        """y = a*x + b*h + c  where h cycles [1, 2, 3] each experiment"""
        a = WorldGenerator._nonzero_int(rng, -4, 4)
        b = WorldGenerator._nonzero_int(rng, -3, 3)
        c = int(rng.integers(-5, 6))
        cycle_len = int(rng.choice([2, 3, 4]))
        hidden_values = list(range(1, cycle_len + 1))

        def fn(x, _state=None, _exp_num=0):
            h = hidden_values[_exp_num % cycle_len]
            return a * x + b * h + c

        expr = (
            f"{a}*x + {b}*h + ({c}) where h cycles through "
            f"{hidden_values} every {cycle_len} experiments"
        )

        return World(
            name="The Invisible Hand",
            description=(
                "This system accepts a single input (x) and produces an output (y). "
                "MYSTERY: Something unseen is affecting the output! Even with the "
                "same input, the output may vary in a SYSTEMATIC (not random) way. "
                "There appears to be a hidden factor cycling through values. "
                "Can you uncover the hidden pattern?"
            ),
            difficulty=8,
            category="hidden_variable",
            variables=["x"],
            variable_ranges={"x": (-10.0, 10.0)},
            ground_truth_fn=fn,
            ground_truth_expr=expr,
            is_stateful=True,
            hints=[
                "Run the same input multiple times and see if the output cycles.",
                "The hidden variable follows a repeating pattern.",
            ],
        )

    # ── Level 9: Dynamic / Stateful ─────────────────────────────────────

    @staticmethod
    def _gen_dynamic(rng: np.random.Generator) -> World:
        """y_t = a*x_t + b*y_{t-1} + c   (output depends on previous output)"""
        a = WorldGenerator._nonzero_int(rng, -3, 3)
        b_val = float(rng.choice([0.25, 0.5, -0.25, -0.5]))
        c = int(rng.integers(-3, 4))

        def fn(x, _state=None, _exp_num=0):
            if _state is None:
                _state = {}
            prev_y = _state.get("prev_y", 0.0)
            y = a * x + b_val * prev_y + c
            _state["prev_y"] = y
            return y

        expr = f"y_t = {a}*x_t + {b_val}*y_{{t-1}} + ({c}), y_0 = 0"

        return World(
            name="The Time Machine",
            description=(
                "This system accepts a single input (x) and produces an output (y). "
                "CRITICAL: This system has MEMORY! The output depends not only on "
                "the current input but also on previous outputs. The system's "
                "history matters. Order of experiments affects results. "
                "Try to figure out how the past influences the present."
            ),
            difficulty=9,
            category="dynamic",
            variables=["x"],
            variable_ranges={"x": (-5.0, 5.0)},
            ground_truth_fn=fn,
            ground_truth_expr=expr,
            is_stateful=True,
            hints=[
                "Run the same experiment from a fresh state vs. after other experiments.",
                "The output depends on what happened before — track your history!",
            ],
        )

    # ── Level 10: Compositional ─────────────────────────────────────────

    @staticmethod
    def _gen_compositional(rng: np.random.Generator) -> World:
        """y = p * (a*x1 + b)^2 + q*x2 + r   (nested composition)"""
        a = WorldGenerator._nonzero_int(rng, -2, 2)
        b = int(rng.integers(-3, 4))
        p = WorldGenerator._nonzero_int(rng, -2, 2)
        q = WorldGenerator._nonzero_int(rng, -3, 3)
        r = int(rng.integers(-5, 6))

        def fn(x1, x2):
            z = a * x1 + b
            return p * z ** 2 + q * x2 + r

        expr = f"{p}*({a}*x1 + {b})**2 + ({q})*x2 + ({r})"

        return World(
            name="The Nested Puzzle",
            description=(
                "This system accepts two inputs (x1, x2) and produces an output (y). "
                "The relationship is COMPLEX. There may be intermediate quantities "
                "computed from one variable that then interact non-linearly. "
                "Think of it as a pipeline: one variable might go through a "
                "transformation before combining with the other."
            ),
            difficulty=10,
            category="compositional",
            variables=["x1", "x2"],
            variable_ranges={"x1": (-5.0, 5.0), "x2": (-5.0, 5.0)},
            ground_truth_fn=fn,
            ground_truth_expr=expr,
            hints=[
                "Hold x2 constant and vary x1 — the pattern may look like a transformed curve.",
                "The system might be decomposable into simpler sub-functions.",
            ],
        )
