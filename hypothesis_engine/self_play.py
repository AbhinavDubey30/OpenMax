"""
Adversarial Self-Play World Generation.

This is the NOVEL CORE of the Hypothesis Engine -- no prior work has this.

Two roles:
    GENERATOR: An LLM that creates new scientific worlds (challenge design)
    SOLVER:    An LLM that tries to solve those worlds (scientific reasoning)

The Generator is rewarded for creating worlds that are:
    - SOLVABLE (not impossible)
    - CHALLENGING (hard enough that the solver doesn't trivially succeed)
    - DIVERSE (different from previously generated worlds)
    - VALID (mathematically well-defined)

This creates a self-improving curriculum where:
    1. Generator creates harder worlds as Solver improves
    2. Solver develops better scientific reasoning as worlds get harder
    3. Both agents improve through adversarial co-evolution

This directly addresses Track 4 (Self-Improvement):
    "environments where agents learn to generate new challenges,
     escalate difficulty, and improve through self-play"
"""

import math
import json
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from .worlds import World, WorldGenerator


@dataclass
class GeneratedWorldSpec:
    """Specification for a world created by the Generator agent."""
    name: str
    description: str
    expression: str           # e.g., "2*x**2 + sin(x) - 3"
    variables: List[str]
    variable_ranges: Dict[str, Tuple[float, float]]
    category: str             # function | causal | physics | stochastic
    estimated_difficulty: int  # 1-10
    noise_std: float = 0.0
    is_valid: bool = False
    validation_error: Optional[str] = None


@dataclass
class SelfPlayMetrics:
    """Tracks self-play performance over time."""
    generator_scores: List[float] = field(default_factory=list)
    solver_scores: List[float] = field(default_factory=list)
    world_diversity: List[float] = field(default_factory=list)
    difficulty_progression: List[int] = field(default_factory=list)
    rounds_played: int = 0


class WorldValidator:
    """Validates procedurally or LLM-generated world specifications."""

    @staticmethod
    def validate_spec(spec: GeneratedWorldSpec, n_test: int = 50) -> GeneratedWorldSpec:
        """
        Validate that a generated world spec is mathematically sound.
        
        Checks:
            1. Expression parses correctly
            2. Produces finite outputs for sample inputs
            3. Is not trivially constant
            4. Has sufficient variation (not degenerate)
        """
        from .verifier import SafeMathEvaluator
        evaluator = SafeMathEvaluator()
        rng = np.random.default_rng(42)

        outputs = []
        errors = 0

        for _ in range(n_test):
            point = {}
            for var in spec.variables:
                lo, hi = spec.variable_ranges.get(var, (-10, 10))
                point[var] = float(rng.uniform(lo, hi))

            try:
                val = evaluator.evaluate(spec.expression, point)
                if math.isnan(val) or math.isinf(val):
                    errors += 1
                else:
                    outputs.append(val)
            except Exception:
                errors += 1

        if errors > n_test * 0.3:
            spec.is_valid = False
            spec.validation_error = f"Expression produces errors for {errors}/{n_test} test points."
            return spec

        if len(outputs) < 10:
            spec.is_valid = False
            spec.validation_error = "Too few valid outputs."
            return spec

        # Check for trivially constant
        output_range = max(outputs) - min(outputs)
        if output_range < 0.01:
            spec.is_valid = False
            spec.validation_error = "Expression produces nearly constant output (degenerate)."
            return spec

        # Check for extreme values
        if max(abs(v) for v in outputs) > 1e8:
            spec.is_valid = False
            spec.validation_error = "Expression produces extreme values (overflow risk)."
            return spec

        spec.is_valid = True
        return spec

    @staticmethod
    def spec_to_world(spec: GeneratedWorldSpec, seed: int = 42) -> Optional[World]:
        """Convert a validated spec into a runnable World."""
        if not spec.is_valid:
            return None

        from .verifier import SafeMathEvaluator
        evaluator = SafeMathEvaluator()

        def fn(**kwargs):
            return evaluator.evaluate(spec.expression, kwargs)

        world = World(
            name=spec.name,
            description=spec.description,
            difficulty=spec.estimated_difficulty,
            category=spec.category,
            variables=spec.variables,
            variable_ranges=spec.variable_ranges,
            ground_truth_fn=fn,
            ground_truth_expr=spec.expression,
            world_type="generated",
            noise_std=spec.noise_std,
            hints=["This world was procedurally generated. No hints available."],
        )
        world._rng = np.random.default_rng(seed)
        world.generate_test_cases(20)
        return world


class SelfPlayOrchestrator:
    """
    Orchestrates adversarial self-play between Generator and Solver.
    
    This is the core self-improvement loop:
        1. Generator creates a world spec
        2. Validator checks the spec
        3. Solver attempts to solve the world
        4. Both are scored
        5. Difficulty adapts based on performance
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.metrics = SelfPlayMetrics()
        self.world_history: List[GeneratedWorldSpec] = []
        self.validator = WorldValidator()

    def generate_world_prompt(self, target_difficulty: int) -> str:
        """
        Create a prompt for the Generator LLM to produce a new world.
        
        Returns a structured prompt that guides the LLM to output
        a valid world specification in JSON format.
        """
        # Build context from recent worlds
        recent_worlds = self.world_history[-5:] if self.world_history else []
        recent_exprs = [w.expression for w in recent_worlds]

        prompt = f"""You are a WORLD GENERATOR for a scientific reasoning environment.

Your task: Create a NEW mathematical/scientific system for an AI agent to discover.

TARGET DIFFICULTY: {target_difficulty}/10

DIFFICULTY GUIDELINES:
- Level 1-3: Simple functions (linear, polynomial, multivariate linear)
- Level 4-5: Functions with conditionals, interaction effects, or trigonometric
- Level 6-7: Compositions of functions, systems with hidden patterns
- Level 8-10: Complex multi-variable systems, physics-inspired equations

RULES:
1. The expression MUST be a valid mathematical formula using these operations:
   +, -, *, /, **, sin(), cos(), exp(), log(), sqrt(), abs()
   Conditionals: where(condition, true_val, false_val)
2. Variables must be named x, x1, x2, etc.
3. The expression must produce FINITE outputs for all inputs in the given ranges
4. It must NOT be trivially constant
5. It MUST be DIFFERENT from these recent worlds: {recent_exprs}

OUTPUT FORMAT (strict JSON):
{{
    "name": "Creative name for the world",
    "description": "Description for the agent (do NOT reveal the formula!)",
    "expression": "mathematical_expression_here",
    "variables": ["x"],
    "variable_ranges": {{"x": [-10, 10]}},
    "category": "function",
    "estimated_difficulty": {target_difficulty},
    "noise_std": 0.0
}}

Generate a creative, well-defined world at difficulty {target_difficulty}:"""
        return prompt

    def parse_generator_output(self, llm_output: str) -> Optional[GeneratedWorldSpec]:
        """Parse the Generator LLM's output into a world spec."""
        try:
            # Try to extract JSON from the output
            start = llm_output.find("{")
            end = llm_output.rfind("}") + 1
            if start == -1 or end == 0:
                return None

            data = json.loads(llm_output[start:end])

            spec = GeneratedWorldSpec(
                name=data.get("name", "Generated World"),
                description=data.get("description", "A generated world."),
                expression=data.get("expression", ""),
                variables=data.get("variables", ["x"]),
                variable_ranges={
                    k: tuple(v) for k, v in data.get("variable_ranges", {"x": [-10, 10]}).items()
                },
                category=data.get("category", "function"),
                estimated_difficulty=data.get("estimated_difficulty", 5),
                noise_std=data.get("noise_std", 0.0),
            )

            return spec
        except (json.JSONDecodeError, KeyError, TypeError):
            return None

    def score_generator(
        self,
        spec: GeneratedWorldSpec,
        solver_score: float,
        target_difficulty: int,
    ) -> float:
        """
        Score the Generator based on the world it created.
        
        Optimal: world is challenging but solvable.
        Scoring:
            - Validity bonus: +20 if the world is valid
            - Challenge bonus: +30 * (1 - solver_score/100) -- harder is better
            - Solvability bonus: +20 if solver scores > 20 (not impossible)
            - Diversity bonus: +15 if different from recent worlds
            - Difficulty match: +15 if estimated difficulty matches actual
        """
        score = 0.0

        # Validity
        if spec.is_valid:
            score += 20.0

        # Challenge (higher is better when solver doesn't ace it)
        if solver_score < 90:
            score += 30.0 * (1.0 - solver_score / 100.0)
        else:
            score += 5.0  # Small bonus even if solver aces it

        # Solvability (must not be impossible)
        if solver_score > 20:
            score += 20.0
        elif solver_score > 0:
            score += 10.0

        # Diversity
        if self.world_history:
            recent_exprs = [w.expression for w in self.world_history[-10:]]
            if spec.expression not in recent_exprs:
                score += 15.0
            else:
                score += 0.0
        else:
            score += 15.0  # First world is always "diverse"

        # Difficulty match
        diff_error = abs(spec.estimated_difficulty - target_difficulty)
        score += max(0, 15.0 - diff_error * 5.0)

        return round(score, 1)

    def run_self_play_round(
        self,
        world_spec: GeneratedWorldSpec,
        solver_score: float,
        target_difficulty: int,
    ) -> Dict[str, Any]:
        """
        Complete one round of self-play scoring.
        
        Returns metrics for both Generator and Solver.
        """
        # Validate the spec
        spec = self.validator.validate_spec(world_spec)

        # Score the generator
        gen_score = self.score_generator(spec, solver_score, target_difficulty)

        # Update metrics
        self.metrics.generator_scores.append(gen_score)
        self.metrics.solver_scores.append(solver_score)
        self.metrics.difficulty_progression.append(target_difficulty)
        self.metrics.rounds_played += 1

        # Track world for diversity
        if spec.is_valid:
            self.world_history.append(spec)

        # Compute diversity metric
        if len(self.world_history) >= 2:
            recent = self.world_history[-10:]
            unique = len(set(w.expression for w in recent))
            diversity = unique / len(recent)
        else:
            diversity = 1.0
        self.metrics.world_diversity.append(diversity)

        return {
            "generator_score": gen_score,
            "solver_score": solver_score,
            "world_valid": spec.is_valid,
            "validation_error": spec.validation_error,
            "diversity": diversity,
            "round": self.metrics.rounds_played,
            "target_difficulty": target_difficulty,
        }

    def get_next_difficulty(self) -> int:
        """
        Adaptively determine next difficulty based on self-play history.
        
        If solver is winning too easily -> increase difficulty
        If solver is struggling too much -> decrease difficulty
        If generator is creating bad worlds -> stay at current level
        """
        if self.metrics.rounds_played < 3:
            return 5  # Start at medium difficulty

        recent_solver = self.metrics.solver_scores[-5:]
        recent_gen = self.metrics.generator_scores[-5:]
        current_diff = self.metrics.difficulty_progression[-1]

        avg_solver = sum(recent_solver) / len(recent_solver)
        avg_gen = sum(recent_gen) / len(recent_gen)

        if avg_solver > 70 and avg_gen > 50:
            # Both doing well -> increase difficulty
            return min(10, current_diff + 1)
        elif avg_solver < 30:
            # Solver struggling -> decrease difficulty
            return max(1, current_diff - 1)
        else:
            # Balanced -> stay
            return current_diff

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the self-play session."""
        if not self.metrics.rounds_played:
            return {"rounds": 0, "message": "No rounds played yet."}

        return {
            "rounds_played": self.metrics.rounds_played,
            "avg_generator_score": round(
                sum(self.metrics.generator_scores) / len(self.metrics.generator_scores), 1
            ),
            "avg_solver_score": round(
                sum(self.metrics.solver_scores) / len(self.metrics.solver_scores), 1
            ),
            "avg_diversity": round(
                sum(self.metrics.world_diversity) / len(self.metrics.world_diversity), 2
            ),
            "difficulty_range": (
                min(self.metrics.difficulty_progression),
                max(self.metrics.difficulty_progression),
            ),
            "worlds_generated": len(self.world_history),
            "valid_worlds": sum(1 for w in self.world_history if w.is_valid),
        }


class ProceduralSelfPlay:
    """
    Self-play using procedural generation (no LLM needed for Generator).
    
    The Generator is a procedural algorithm that creates increasingly
    complex worlds based on Solver performance. This allows self-play
    demos WITHOUT an API key.
    """

    TEMPLATES = [
        # (expression_template, variables, ranges, category, base_difficulty)
        ("{a}*x + ({b})", ["x"], {"x": (-10, 10)}, "linear", 1),
        ("{a}*x**2 + ({b})*x + ({c})", ["x"], {"x": (-5, 5)}, "polynomial", 2),
        ("{a}*x1 + ({b})*x2 + ({c})", ["x1", "x2"], {"x1": (-5, 5), "x2": (-5, 5)}, "multivar", 3),
        ("{a}*sin({b}*x) + ({c})", ["x"], {"x": (-6.28, 6.28)}, "trigonometric", 4),
        ("({a}*x + ({b})) if x > {t} else ({c}*x + ({d}))", ["x"], {"x": (-10, 10)}, "conditional", 5),
        ("{a}*x1*x2 + ({b})*x1 + ({c})*x2 + ({d})", ["x1", "x2"], {"x1": (-5, 5), "x2": (-5, 5)}, "interaction", 6),
        ("{a}*x**3 + ({b})*x**2 + ({c})*x + ({d})", ["x"], {"x": (-3, 3)}, "cubic", 7),
        ("{a}*exp({b}*x) + ({c})", ["x"], {"x": (-2, 2)}, "exponential", 8),
        ("{a}*sin({b}*x) + {c}*cos({d}*x) + ({e})", ["x"], {"x": (-6.28, 6.28)}, "fourier", 9),
        ("{a}*x1**2 + ({b})*x2**2 + ({c})*x1*x2 + ({d})*x1 + ({e})*x2 + ({f})", 
         ["x1", "x2"], {"x1": (-3, 3), "x2": (-3, 3)}, "quadratic_2d", 10),
    ]

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.round = 0
        self.solver_history: List[float] = []
        self.generated_expressions: List[str] = []

    def generate_world(self, target_difficulty: int) -> Optional[World]:
        """Generate a new world procedurally at the target difficulty."""
        # Pick a template near the target difficulty
        candidates = [t for t in self.TEMPLATES if abs(t[4] - target_difficulty) <= 2]
        if not candidates:
            candidates = self.TEMPLATES

        template = candidates[int(self.rng.integers(0, len(candidates)))]
        expr_template, variables, ranges, category, base_diff = template

        # Generate random coefficients
        params = {}
        for letter in "abcdef":
            params[letter] = WorldGenerator._nonzero_int(self.rng, -4, 4)
        params["t"] = int(self.rng.choice([-2, -1, 0, 1, 2]))

        # Fill in the template
        try:
            expression = expr_template.format(**params)
        except (KeyError, IndexError):
            expression = f"{params['a']}*x + ({params['b']})"

        # Validate
        spec = GeneratedWorldSpec(
            name=f"Generated World #{self.round + 1}",
            description=(
                f"A procedurally generated system (round {self.round + 1}). "
                f"Discover the hidden mathematical relationship."
            ),
            expression=expression,
            variables=variables,
            variable_ranges=ranges,
            category=category,
            estimated_difficulty=target_difficulty,
        )

        spec = WorldValidator.validate_spec(spec)
        if not spec.is_valid:
            # Fallback to simple world
            spec = GeneratedWorldSpec(
                name=f"Generated World #{self.round + 1}",
                description="Discover the hidden relationship.",
                expression=f"{params['a']}*x + ({params['b']})",
                variables=["x"],
                variable_ranges={"x": (-10, 10)},
                category="linear",
                estimated_difficulty=max(1, target_difficulty - 2),
            )
            spec = WorldValidator.validate_spec(spec)

        world = WorldValidator.spec_to_world(spec, seed=int(self.rng.integers(0, 10000)))
        if world:
            self.round += 1
            self.generated_expressions.append(expression)
        return world

    def record_solver_score(self, score: float):
        """Record the solver's performance for adaptive difficulty."""
        self.solver_history.append(score)

    def get_next_difficulty(self) -> int:
        """Adaptive difficulty based on solver history."""
        if len(self.solver_history) < 3:
            return 3  # Start easy-medium

        recent = self.solver_history[-5:]
        avg = sum(recent) / len(recent)

        if avg > 75:
            return min(10, max(d for d in [3, 4, 5, 6, 7, 8, 9, 10] if d > self._current_difficulty()))
        elif avg < 35:
            return max(1, min(d for d in [1, 2, 3, 4, 5, 6, 7] if d < self._current_difficulty()))
        else:
            return self._current_difficulty()

    def _current_difficulty(self) -> int:
        """Estimate current difficulty level."""
        if not self.solver_history:
            return 3
        recent = self.solver_history[-3:]
        avg = sum(recent) / len(recent)
        return max(1, min(10, int(avg / 10)))
