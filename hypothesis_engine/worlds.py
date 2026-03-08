"""
Procedural World Generator for the Hypothesis Engine.

Generates black-box systems with known ground-truth rules at varying
difficulty levels. Each world is a self-contained scientific mystery
that an agent must unravel through experimentation.

World Categories (NOVEL -- not found in any prior work):
    A. FUNCTION DISCOVERY       (Levels 1-3)  : Classic curve fitting
    B. CAUSAL REASONING         (Levels 4-6)  : Interventional experiments,
                                                 confounders, do-calculus
    C. PHYSICS SIMULATION       (Levels 7-8)  : Spring systems, projectile
                                                 motion, circuits
    D. STATE MACHINE DISCOVERY  (Level 9)     : Hidden finite automata
    E. STOCHASTIC / STATISTICAL (Level 10)    : Noisy systems requiring
                                                 repeated experiments

Each world supports two experiment modes:
    - OBSERVE:  Passive observation (may include confounders)
    - INTERVENE: Set a variable to a value, breaking causal links
"""

import math
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Callable, Tuple, Any, Optional
from enum import Enum


class ExperimentMode(Enum):
    """Whether an experiment is observational or interventional."""
    OBSERVE = "observe"
    INTERVENE = "intervene"


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
    supports_intervention: bool = False
    causal_graph: Optional[Dict[str, List[str]]] = None
    world_type: str = "function"  # function | causal | physics | state_machine | stochastic
    _rng: Optional[np.random.Generator] = field(default=None, repr=False)
    _state: Dict[str, Any] = field(default_factory=dict, repr=False)
    _experiment_count: int = field(default=0, repr=False)
    _intervention_fn: Optional[Callable] = field(default=None, repr=False)
    _confounders: Dict[str, Any] = field(default_factory=dict, repr=False)

    def run_experiment(
        self,
        inputs: Dict[str, float],
        mode: str = "observe",
        intervention_targets: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Run a single experiment with given input values.

        Args:
            inputs: Variable name -> value mapping.
            mode: 'observe' (default) or 'intervene'.
            intervention_targets: Which variables are being intervened on
                                  (only relevant for causal worlds).

        Returns:
            Dict with inputs, output, and optional metadata.
        """
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
                        "mode": mode,
                    }

        try:
            if self.supports_intervention and mode == "intervene" and self._intervention_fn:
                output = self._intervention_fn(
                    inputs=inputs,
                    intervention_targets=intervention_targets or list(inputs.keys()),
                    state=self._state,
                    exp_num=self._experiment_count,
                    rng=self._rng,
                )
            elif self.is_stateful:
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
                return {"inputs": inputs, "output": None, "error": "Undefined result", "mode": mode}

        except Exception as e:
            return {"inputs": inputs, "output": None, "error": str(e), "mode": mode}

        self._experiment_count += 1
        return {"inputs": inputs, "output": round(output, 4), "mode": mode}

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
            result = self.run_experiment(case, mode="intervene" if self.supports_intervention else "observe")
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
        briefing = {
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
            "world_type": self.world_type,
            "supports_intervention": self.supports_intervention,
        }
        if self.causal_graph is not None:
            briefing["causal_structure_hint"] = (
                "This world has causal structure. Variables may cause other variables. "
                "Use 'intervene' mode to break causal links and isolate effects. "
                "Use 'observe' mode to see natural correlations (which may be confounded)."
            )
        return briefing


class WorldGenerator:
    """Procedurally generates scientific worlds across 10 difficulty levels."""

    DIFFICULTY_NAMES = {
        1: "Linear Discovery",
        2: "Polynomial Patterns",
        3: "Multi-Variable Linear",
        4: "Causal Chains",
        5: "Confounded Causation",
        6: "Causal Graphs with Hidden Confounders",
        7: "Spring Physics",
        8: "Projectile Motion",
        9: "State Machine Discovery",
        10: "Signal in Noise (Statistical Reasoning)",
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
            4: WorldGenerator._gen_causal_chain,
            5: WorldGenerator._gen_confounded_causation,
            6: WorldGenerator._gen_causal_graph_hidden,
            7: WorldGenerator._gen_spring_physics,
            8: WorldGenerator._gen_projectile,
            9: WorldGenerator._gen_state_machine,
            10: WorldGenerator._gen_stochastic_statistical,
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

    # ══════════════════════════════════════════════════════════════════════
    # CATEGORY A: FUNCTION DISCOVERY (Levels 1-3)
    # ══════════════════════════════════════════════════════════════════════

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
            world_type="function",
            hints=[
                "Try varying x by equal steps and observe how the output changes.",
                "The output changes at a constant rate.",
            ],
        )

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
            world_type="function",
            hints=[
                "Look at the second differences (differences of differences).",
                "Symmetry around a point may be a clue.",
            ],
        )

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
            world_type="function",
            hints=[
                "Hold one variable constant and vary the other.",
                "Each variable might contribute independently to the output.",
            ],
        )

    # ══════════════════════════════════════════════════════════════════════
    # CATEGORY B: CAUSAL REASONING (Levels 4-6)
    # ══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _gen_causal_chain(rng: np.random.Generator) -> World:
        """
        Causal chain: X -> M -> Y
        Observe: Y correlates with X through mediator M
        Intervene on X: reveals direct + indirect effect
        Intervene on M: reveals only direct M->Y effect
        """
        a = WorldGenerator._nonzero_int(rng, -3, 3)  # X -> M coefficient
        b = WorldGenerator._nonzero_int(rng, -3, 3)  # M -> Y coefficient
        c = int(rng.integers(-5, 6))                  # Y intercept
        m_offset = int(rng.integers(-3, 4))            # M intercept

        # Under observation: x -> m = a*x + m_offset, y = b*m + c = b*(a*x + m_offset) + c
        def observe_fn(x):
            m = a * x + m_offset
            return b * m + c

        # Under intervention on x: same as observe (x is exogenous)
        # Under intervention on m: y = b*m_intervened + c (breaks X->M link)
        def intervene_fn(inputs, intervention_targets, state, exp_num, rng):
            x_val = inputs.get("x", 0)
            if "x" in intervention_targets and len(intervention_targets) == 1:
                # Intervening on X: same as observation for this chain
                m = a * x_val + m_offset
                return b * m + c
            elif "x" in intervention_targets:
                # Intervening on X (treating it as setting x directly)
                m = a * x_val + m_offset
                return b * m + c
            else:
                return b * (a * x_val + m_offset) + c

        expr = f"X -> M -> Y: M = {a}*x + {m_offset}, Y = {b}*M + {c}"
        # Simplified for predictions: Y = b*(a*x + m_offset) + c = b*a*x + b*m_offset + c
        ba = b * a
        bm_c = b * m_offset + c
        pred_expr = f"{ba}*x + ({bm_c})"

        return World(
            name="The Causal Chain",
            description=(
                "This system has a CAUSAL STRUCTURE. There is a variable X you control, "
                "and an output Y. But the effect of X on Y may go through an intermediate "
                "mechanism (a mediator). Use BOTH 'observe' and 'intervene' modes to "
                "understand the causal pathway. In 'observe' mode, you see natural "
                "relationships. In 'intervene' mode, you force a variable to a specific "
                "value, potentially breaking upstream causal links."
            ),
            difficulty=4,
            category="causal_chain",
            variables=["x"],
            variable_ranges={"x": (-5.0, 5.0)},
            ground_truth_fn=observe_fn,
            ground_truth_expr=pred_expr,
            world_type="causal",
            supports_intervention=True,
            _intervention_fn=intervene_fn,
            causal_graph={"x": ["m"], "m": ["y"]},
            hints=[
                "Compare what happens when you observe vs. when you intervene.",
                "The causal chain is: X causes M, M causes Y.",
            ],
        )

    @staticmethod
    def _gen_confounded_causation(rng: np.random.Generator) -> World:
        """
        Confounded system: Z (hidden confounder) -> X and Z -> Y
        Observe: X and Y appear correlated (but it's spurious via Z)
        Intervene on X: breaks Z->X link, reveals true X->Y effect

        True model: Y = b*X + d*Z + c, where Z is hidden
        Under observation: Z varies freely, creating spurious X-Y correlation
        Under intervention: Z is independent of X, so its effect averages out
        """
        b_true = int(rng.choice([0, 1, -1]))  # True X->Y effect (may be zero!)
        d = WorldGenerator._nonzero_int(rng, -3, 3)  # Z->Y effect
        e = WorldGenerator._nonzero_int(rng, -2, 2)  # Z->X effect
        c = int(rng.integers(-5, 6))

        def observe_fn(x):
            # Under observation, Z is correlated with X:  Z ~ x/e approximately
            # We simulate: given observed x, Z = (x - noise) / e
            # This makes X and Y correlated even if b_true = 0
            # For deterministic test: Z = x/e (simplified)
            z = x / e if e != 0 else 0
            return b_true * x + d * z + c

        def intervene_fn(inputs, intervention_targets, state, exp_num, rng):
            x_val = inputs.get("x", 0)
            # Under intervention: Z is independent of X, Z ~ 0 (mean)
            z = 0  # Z's average value when not driven by anything
            return b_true * x_val + d * z + c

        # Under observation: Y = b*x + d*(x/e) + c = (b + d/e)*x + c
        obs_coeff = b_true + (d / e if e != 0 else 0)
        obs_expr = f"OBSERVE: Y ~ {round(obs_coeff, 2)}*x + {c} (confounded!)"

        # Under intervention: Y = b*x + c (true causal effect)
        int_expr = f"INTERVENE: Y = {b_true}*x + {c} (true effect)"
        pred_expr = f"{b_true}*x + ({c})"

        return World(
            name="The Confounder's Trap",
            description=(
                "DANGER: Correlation is not causation! This system has a HIDDEN "
                "CONFOUNDER -- an unobserved variable that influences BOTH the "
                "input and the output, creating a SPURIOUS correlation.\n\n"
                "In 'observe' mode, X and Y appear related. But is it real?\n"
                "In 'intervene' mode, you FORCE X to a value, breaking the "
                "confounder's influence on X. The relationship you see under "
                "intervention is the TRUE causal effect.\n\n"
                "Your task: discover the TRUE causal effect of X on Y by "
                "comparing observational and interventional experiments."
            ),
            difficulty=5,
            category="confounded",
            variables=["x"],
            variable_ranges={"x": (-5.0, 5.0)},
            ground_truth_fn=observe_fn,
            ground_truth_expr=pred_expr,
            world_type="causal",
            supports_intervention=True,
            _intervention_fn=intervene_fn,
            causal_graph={"z_hidden": ["x", "y"], "x": ["y"]},
            hints=[
                "Run the SAME experiment in both 'observe' and 'intervene' modes. If the results differ, there's a confounder!",
                "The true causal effect of X on Y is what you see under intervention.",
            ],
        )

    @staticmethod
    def _gen_causal_graph_hidden(rng: np.random.Generator) -> World:
        """
        Causal graph with 2 observed variables and a hidden confounder.
        X1 -> Y, X2 -> Y, and Z (hidden) -> X1, Z -> X2
        
        Under observation: X1 and X2 appear correlated (via Z)
        Under intervention on X1: breaks Z->X1, reveals true X1->Y
        Under intervention on X2: breaks Z->X2, reveals true X2->Y
        """
        a = WorldGenerator._nonzero_int(rng, -3, 3)  # X1 -> Y
        b = WorldGenerator._nonzero_int(rng, -3, 3)  # X2 -> Y
        c = int(rng.integers(-5, 6))                  # Y intercept
        z_to_x1 = WorldGenerator._nonzero_int(rng, -2, 2)
        z_to_x2 = WorldGenerator._nonzero_int(rng, -2, 2)
        z_to_y = int(rng.choice([-2, -1, 0, 1, 2]))   # Direct Z -> Y

        def observe_fn(x1, x2):
            # Under observation, Z is correlated with both X1 and X2
            # Reconstruct Z from the inputs (approximate: Z ~ x1/z_to_x1)
            if z_to_x1 != 0:
                z_approx = x1 / z_to_x1
            elif z_to_x2 != 0:
                z_approx = x2 / z_to_x2
            else:
                z_approx = 0
            return a * x1 + b * x2 + z_to_y * z_approx + c

        def intervene_fn(inputs, intervention_targets, state, exp_num, rng):
            x1_val = inputs.get("x1", 0)
            x2_val = inputs.get("x2", 0)
            # Under intervention, Z's effect on intervened variables is broken
            # Z averages to 0
            z_effect = 0
            # Only add Z effect for non-intervened variables
            if "x1" not in intervention_targets and z_to_x1 != 0:
                z_effect = z_to_y * (x1_val / z_to_x1)
            if "x2" not in intervention_targets and z_to_x2 != 0:
                z_effect = z_to_y * (x2_val / z_to_x2)
            return a * x1_val + b * x2_val + z_effect + c

        pred_expr = f"{a}*x1 + ({b})*x2 + ({c})"

        return World(
            name="The Hidden Web",
            description=(
                "A complex causal system with TWO input variables (x1, x2) and an "
                "output (y). WARNING: There is a HIDDEN CONFOUNDER affecting both "
                "inputs and possibly the output!\n\n"
                "Under observation, the relationships between x1, x2, and y are "
                "DISTORTED by the confounder. You MUST use interventions to discover "
                "the true causal effects.\n\n"
                "Strategy: Intervene on x1 (while varying it) to find X1->Y effect. "
                "Intervene on x2 (while varying it) to find X2->Y effect. "
                "Compare with observations to detect the confounder."
            ),
            difficulty=6,
            category="causal_graph",
            variables=["x1", "x2"],
            variable_ranges={"x1": (-5.0, 5.0), "x2": (-5.0, 5.0)},
            ground_truth_fn=observe_fn,
            ground_truth_expr=pred_expr,
            world_type="causal",
            supports_intervention=True,
            _intervention_fn=intervene_fn,
            causal_graph={"z_hidden": ["x1", "x2", "y"], "x1": ["y"], "x2": ["y"]},
            hints=[
                "Intervene on ONE variable at a time while varying it to isolate its true effect.",
                "If observation and intervention give different results, a confounder is present.",
            ],
        )

    # ══════════════════════════════════════════════════════════════════════
    # CATEGORY C: PHYSICS SIMULATION (Levels 7-8)
    # ══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _gen_spring_physics(rng: np.random.Generator) -> World:
        """
        Damped spring system: F = -k*x - b*v + F_applied
        The agent must discover the spring constant (k), damping (b),
        and predict the equilibrium position for different applied forces.

        Simplified to steady-state: at equilibrium, v=0, so F_applied = k*x
        -> x_eq = F_applied / k
        
        But we also add mass effect: the output is the equilibrium displacement.
        """
        k = float(rng.choice([1, 2, 3, 4, 5]))      # spring constant
        b = float(rng.choice([0.5, 1.0, 1.5, 2.0]))  # damping coefficient
        m = float(rng.choice([1.0, 2.0, 3.0]))        # mass

        # Natural frequency and damping ratio
        omega_n = math.sqrt(k / m)
        zeta = b / (2 * math.sqrt(k * m))

        def fn(x):
            # Steady-state displacement under constant force
            # x_eq = force / k
            # Variable x represents applied force in Newtons
            return x / k

        expr = f"displacement = force / {k} (Hooke's Law: k={k}, m={m}, damping={b})"
        pred_expr = f"x / {k}"  # x is the force variable

        return World(
            name="The Spring System",
            description=(
                "You are studying a PHYSICAL SPRING SYSTEM. You can apply a force "
                "(variable 'x' represents force in Newtons) and observe the resulting "
                "equilibrium displacement.\n\n"
                "The system follows physical laws: springs resist displacement "
                "proportionally to how far they're stretched (Hooke's Law: F = -kx). "
                "There may also be damping and mass effects.\n\n"
                "Your goal: discover the spring constant (k) and predict the "
                "displacement for given forces. Think like a physicist!"
            ),
            difficulty=7,
            category="spring_physics",
            variables=["x"],
            variable_ranges={"x": (-10.0, 10.0)},
            ground_truth_fn=fn,
            ground_truth_expr=pred_expr,
            world_type="physics",
            hints=[
                "Hooke's Law: displacement = force / spring_constant.",
                f"The spring constant determines how stiff the spring is.",
            ],
        )

    @staticmethod
    def _gen_projectile(rng: np.random.Generator) -> World:
        """
        Projectile motion: given launch angle and initial speed,
        predict the range (horizontal distance).
        
        Range = (v^2 * sin(2*theta)) / g
        
        Two variables: v (speed) and theta (angle in degrees)
        """
        g = float(rng.choice([9.8, 10.0, 5.0, 15.0]))  # gravity
        # Add a wind factor for complexity
        wind = float(rng.choice([0.0, 0.5, -0.5, 1.0, -1.0]))

        def fn(v, theta):
            # Convert theta from degrees to radians
            theta_rad = math.radians(theta)
            # Range formula with wind correction
            base_range = (v ** 2 * math.sin(2 * theta_rad)) / g
            wind_effect = wind * v * math.cos(theta_rad) * 0.1
            return max(0, base_range + wind_effect)

        expr = f"range = (v^2 * sin(2*theta_deg)) / {g} + {wind}*v*cos(theta)*0.1"
        # For test predictions, theta is in degrees
        if wind == 0:
            pred_expr = f"(v**2 * sin(2*theta*3.14159/180)) / {g}"
        else:
            pred_expr = f"(v**2 * sin(2*theta*3.14159/180)) / {g} + {wind}*v*cos(theta*3.14159/180)*0.1"

        return World(
            name="The Projectile Lab",
            description=(
                "You are in a PHYSICS LAB studying projectile motion. You control "
                "two variables:\n"
                "  - v: launch speed (m/s)\n"
                "  - theta: launch angle (degrees, 0-90)\n\n"
                "The output is the RANGE (horizontal distance before landing).\n\n"
                "The system follows the laws of kinematics. There may be additional "
                "factors like wind resistance. Your goal: discover the governing "
                "equation and predict the range for new launch conditions."
            ),
            difficulty=8,
            category="projectile",
            variables=["v", "theta"],
            variable_ranges={"v": (1.0, 20.0), "theta": (5.0, 85.0)},
            ground_truth_fn=fn,
            ground_truth_expr=pred_expr,
            world_type="physics",
            hints=[
                "The classic range formula is R = v^2 * sin(2*theta) / g.",
                "Try fixing speed and varying angle to find the optimal angle.",
            ],
        )

    # ══════════════════════════════════════════════════════════════════════
    # CATEGORY D: STATE MACHINE DISCOVERY (Level 9)
    # ══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _gen_state_machine(rng: np.random.Generator) -> World:
        """
        Hidden finite state machine.
        The world has N hidden states. Input x determines transition and output.
        
        Simple: 3 states {S0, S1, S2}
        - If x > 0: move to next state (mod 3)
        - If x <= 0: stay in current state
        - Output = state_value[current_state] * x + offset[current_state]
        """
        n_states = int(rng.choice([2, 3, 4]))

        # Generate state-dependent coefficients
        state_coeffs = [WorldGenerator._nonzero_int(rng, -3, 3) for _ in range(n_states)]
        state_offsets = [int(rng.integers(-5, 6)) for _ in range(n_states)]

        # Transition threshold
        threshold = float(rng.choice([0, 1, -1]))

        def fn(x, _state=None, _exp_num=0):
            if _state is None:
                _state = {}
            current = _state.get("current_state", 0)

            # Compute output based on current state
            coeff = state_coeffs[current]
            offset = state_offsets[current]
            output = coeff * x + offset

            # State transition
            if x > threshold:
                next_state = (current + 1) % n_states
            else:
                next_state = current

            _state["current_state"] = next_state
            return output

        state_desc = ", ".join(
            [f"S{i}: {state_coeffs[i]}*x + {state_offsets[i]}" for i in range(n_states)]
        )
        expr = (
            f"{n_states} states, transition on x > {threshold}: "
            f"[{state_desc}], starts at S0"
        )

        return World(
            name="The Hidden Machine",
            description=(
                "This system has HIDDEN INTERNAL STATES -- it behaves like a "
                "finite state machine. The output depends on BOTH the input (x) "
                "AND the system's current hidden state.\n\n"
                f"The machine has {n_states} hidden states. Your input may cause "
                "the machine to TRANSITION between states. The same input can "
                "produce different outputs depending on what state the machine is in.\n\n"
                "CRITICAL: The order of your experiments matters! Each experiment "
                "may change the internal state. Try to:\n"
                "1. Identify how many states exist\n"
                "2. Figure out what causes transitions\n"
                "3. Determine the input-output rule for each state"
            ),
            difficulty=9,
            category="state_machine",
            variables=["x"],
            variable_ranges={"x": (-5.0, 5.0)},
            ground_truth_fn=fn,
            ground_truth_expr=expr,
            world_type="state_machine",
            is_stateful=True,
            hints=[
                "Try the same input multiple times in a row -- does the output change?",
                f"There are {n_states} hidden states. Transitions depend on input magnitude.",
            ],
        )

    # ══════════════════════════════════════════════════════════════════════
    # CATEGORY E: STOCHASTIC / STATISTICAL (Level 10)
    # ══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _gen_stochastic_statistical(rng: np.random.Generator) -> World:
        """
        System with significant noise. The agent must:
        1. Run repeated experiments to estimate means
        2. Separate signal from noise
        3. Use statistical reasoning (averaging, confidence)
        
        True function: y = a*x + b + N(0, sigma^2)
        With HIGH noise (sigma comparable to signal range)
        """
        a = WorldGenerator._nonzero_int(rng, -3, 3)
        b = int(rng.integers(-5, 6))
        noise_std = float(rng.choice([2.0, 3.0, 4.0, 5.0]))

        def fn(x):
            return a * x + b

        expr = f"{a}*x + ({b}) + N(0, {noise_std}^2)"
        pred_expr = f"{a}*x + ({b})"

        return World(
            name="Through the Storm",
            description=(
                "This system has VERY HIGH NOISE. Each measurement includes "
                "significant random error. A single experiment tells you almost "
                "nothing!\n\n"
                f"The noise standard deviation is approximately {noise_std:.0f} units. "
                "The underlying signal is deterministic but BURIED in noise.\n\n"
                "STRATEGY: You must think like a STATISTICIAN:\n"
                "1. Run REPEATED experiments at the same input value\n"
                "2. AVERAGE the results to estimate the true output\n"
                "3. Do this for several different input values\n"
                "4. Fit a model to the averaged data points\n\n"
                "Running each input value 3-5 times is recommended. "
                "A single observation is UNRELIABLE."
            ),
            difficulty=10,
            category="stochastic",
            variables=["x"],
            variable_ranges={"x": (-5.0, 5.0)},
            ground_truth_fn=fn,
            ground_truth_expr=pred_expr,
            world_type="stochastic",
            noise_std=noise_std,
            hints=[
                "Run each input 3-5 times and average the results.",
                "The underlying relationship is simple -- the noise is the challenge.",
            ],
        )
