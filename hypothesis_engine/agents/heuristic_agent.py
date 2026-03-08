"""
Heuristic Scientist Agent for the Hypothesis Engine.

A rule-based agent that follows the scientific method:
    1. Probe: Run strategic baseline experiments
    2. Analyze: Detect patterns in the data
    3. Hypothesize: Form a mathematical hypothesis
    4. Test: Validate the hypothesis with targeted experiments
    5. Refine: Update the hypothesis if needed
    6. Predict: Use the hypothesis to predict test cases

Works well for difficulty levels 1-7. No API key required.
"""

import math
from typing import Dict, Any, List, Tuple, Optional
from .base import BaseAgent


class HeuristicScientist(BaseAgent):
    """
    A smart heuristic agent that follows systematic scientific method.
    Designed to reliably solve levels 1-7 and give reasonable results for 8-10.
    """

    def __init__(self):
        self.reset()

    @property
    def name(self) -> str:
        return "Dr. Heuristic (AI Scientist)"

    def reset(self):
        """Reset for a new episode."""
        self.phase = "probe"  # probe → analyze → hypothesize → test → predict
        self.experiments = []
        self.variables = []
        self.ranges = {}
        self.hypothesis = None
        self.hypothesis_confirmed = False
        self.probe_plan = []
        self.probe_index = 0
        self.test_plan = []
        self.test_index = 0
        self.world_category = None
        self.observations = {}
        self.total_budget = 30
        self.used = 0
        self.test_cases = []
        self.is_stateful = False

    def act(self, observation: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
        """Decide the next action based on observation and internal state."""
        # Extract info from observation
        world_info = observation.get("world", {})
        self.variables = world_info.get("variables", self.variables)
        self.ranges = world_info.get("variable_ranges", self.ranges)
        self.total_budget = observation.get("experiment_budget", self.total_budget)
        remaining = observation.get("experiments_remaining", self.total_budget - self.used)
        self.test_cases = observation.get("test_cases", self.test_cases)
        self.is_stateful = world_info.get("is_stateful", False)

        # Track experiment results
        last_result = observation.get("last_experiment_result")
        if last_result and last_result.get("output") is not None:
            self.experiments.append(last_result)

        # Track hypothesis feedback
        hyp_feedback = observation.get("hypothesis_feedback")
        if hyp_feedback:
            quality = hyp_feedback.get("score_hint", "low")
            if quality == "high":
                self.hypothesis_confirmed = True

        # State machine
        if self.phase == "probe":
            return self._probe_phase(remaining)
        elif self.phase == "analyze":
            return self._analyze_phase()
        elif self.phase == "hypothesize":
            return self._hypothesize_phase()
        elif self.phase == "test":
            return self._test_phase(remaining)
        elif self.phase == "refine":
            return self._refine_phase(remaining)
        elif self.phase == "predict":
            return self._predict_phase()

        return ({"action": "get_status"}, "Checking current status...")

    # ── Phase: Probe ─────────────────────────────────────────────────────

    def _probe_phase(self, remaining: int) -> Tuple[Dict, str]:
        """Run strategic baseline experiments."""
        if not self.probe_plan:
            self.probe_plan = self._create_probe_plan()
            self.probe_index = 0

        if self.probe_index < len(self.probe_plan) and remaining > 5:
            inputs = self.probe_plan[self.probe_index]
            self.probe_index += 1
            self.used += 1
            reasoning = self._probe_reasoning(inputs)
            return ({"action": "experiment", "inputs": inputs}, reasoning)

        # Done probing, move to analysis
        self.phase = "analyze"
        return self._analyze_phase()

    def _create_probe_plan(self) -> List[Dict[str, float]]:
        """Create a strategic probing plan based on variable count."""
        plan = []

        if len(self.variables) == 1:
            var = self.variables[0]
            lo, hi = self.ranges.get(var, (-10, 10))
            mid = (lo + hi) / 2

            # Strategic probe points
            probe_values = [
                mid,           # center
                mid + 1,       # center + 1
                mid - 1,       # center - 1
                mid + 2,       # center + 2
                lo,            # minimum
                hi,            # maximum
                mid + 0.5,     # non-integer
            ]

            # Add scan points for conditional detection
            step = (hi - lo) / 10
            for i in range(11):
                val = lo + i * step
                if val not in probe_values:
                    probe_values.append(round(val, 2))

            for v in probe_values[:15]:
                plan.append({var: round(v, 2)})

        elif len(self.variables) == 2:
            v1, v2 = self.variables
            lo1, hi1 = self.ranges.get(v1, (-5, 5))
            lo2, hi2 = self.ranges.get(v2, (-5, 5))
            m1 = (lo1 + hi1) / 2
            m2 = (lo2 + hi2) / 2

            # Baseline
            plan.append({v1: m1, v2: m2})

            # Vary v1 only (hold v2 constant)
            for offset in [1, -1, 2, -2, 3]:
                plan.append({v1: m1 + offset, v2: m2})

            # Vary v2 only (hold v1 constant)
            for offset in [1, -1, 2, -2, 3]:
                plan.append({v1: m1, v2: m2 + offset})

            # Vary both (interaction detection)
            plan.append({v1: m1 + 1, v2: m2 + 1})
            plan.append({v1: m1 + 1, v2: m2 - 1})
            plan.append({v1: m1 - 1, v2: m2 + 1})
            plan.append({v1: m1 + 2, v2: m2 + 2})
            plan.append({v1: lo1, v2: lo2})
            plan.append({v1: hi1, v2: hi2})

        return plan

    def _probe_reasoning(self, inputs: Dict) -> str:
        """Generate reasoning text for a probe experiment."""
        n = len(self.experiments) + 1
        if n == 1:
            return "Establishing baseline at center point"
        elif n <= 3:
            return "Testing unit changes to detect basic relationship"
        elif n <= 6:
            return "Scanning wider range for pattern confirmation"
        elif n <= 10:
            return "Checking for discontinuities or non-linearities"
        else:
            return "Gathering additional data points for refinement"

    # ── Phase: Analyze ───────────────────────────────────────────────────

    def _analyze_phase(self) -> Tuple[Dict, str]:
        """Analyze collected data to determine the pattern type."""
        if not self.experiments:
            self.phase = "probe"
            return self._probe_phase(self.total_budget)

        self.observations = self._analyze_data()
        self.phase = "hypothesize"
        return self._hypothesize_phase()

    def _analyze_data(self) -> Dict[str, Any]:
        """Analyze experiment data to detect pattern type."""
        obs = {"pattern": "unknown"}

        if len(self.variables) == 1:
            var = self.variables[0]
            points = []
            for exp in self.experiments:
                x = exp["inputs"].get(var)
                y = exp.get("output")
                if x is not None and y is not None:
                    points.append((x, y))

            points.sort(key=lambda p: p[0])

            if len(points) >= 3:
                obs["points"] = points
                obs["is_linear"] = self._check_linearity(points)
                obs["has_discontinuity"] = self._check_discontinuity(points)
                obs["coefficients"] = self._fit_polynomial(points)

        elif len(self.variables) == 2:
            v1, v2 = self.variables
            obs["v1_effect"] = self._isolate_variable_effect(v1, v2)
            obs["v2_effect"] = self._isolate_variable_effect(v2, v1)
            obs["has_interaction"] = self._check_interaction()

        return obs

    def _check_linearity(self, points: List[Tuple[float, float]]) -> bool:
        """Check if the data appears linear."""
        if len(points) < 3:
            return True

        # Check if first differences are approximately constant
        diffs = []
        for i in range(1, len(points)):
            dx = points[i][0] - points[i-1][0]
            dy = points[i][1] - points[i-1][1]
            if abs(dx) > 0.01:
                diffs.append(dy / dx)

        if len(diffs) < 2:
            return True

        mean_diff = sum(diffs) / len(diffs)
        max_deviation = max(abs(d - mean_diff) for d in diffs)
        return max_deviation < 0.5

    def _check_discontinuity(self, points: List[Tuple[float, float]]) -> Optional[float]:
        """Check for sudden jumps in the data."""
        if len(points) < 5:
            return None

        diffs = []
        for i in range(1, len(points)):
            dx = points[i][0] - points[i-1][0]
            dy = abs(points[i][1] - points[i-1][1])
            if abs(dx) > 0.01:
                diffs.append((points[i][0], dy / abs(dx)))

        if not diffs:
            return None

        mean_rate = sum(d[1] for d in diffs) / len(diffs)

        for x, rate in diffs:
            if rate > mean_rate * 3 and mean_rate > 0:
                return x

        return None

    def _fit_polynomial(self, points: List[Tuple[float, float]], max_degree: int = 3) -> Dict:
        """Fit polynomials of increasing degree and find best fit."""
        import numpy as np

        xs = np.array([p[0] for p in points])
        ys = np.array([p[1] for p in points])

        best = {"degree": 0, "coeffs": [float(np.mean(ys))], "residual": float("inf")}

        for deg in range(1, min(max_degree + 1, len(points))):
            try:
                coeffs = np.polyfit(xs, ys, deg)
                fitted = np.polyval(coeffs, xs)
                residual = float(np.sum((ys - fitted) ** 2))

                if residual < best["residual"] - 0.1:
                    best = {
                        "degree": deg,
                        "coeffs": [round(float(c), 2) for c in coeffs],
                        "residual": residual,
                    }
            except (np.linalg.LinAlgError, ValueError):
                continue

        return best

    def _isolate_variable_effect(self, target: str, other: str) -> Dict:
        """Analyze the effect of one variable while other is constant."""
        groups = {}
        for exp in self.experiments:
            other_val = round(exp["inputs"].get(other, 0), 1)
            if other_val not in groups:
                groups[other_val] = []
            groups[other_val].append((exp["inputs"].get(target, 0), exp.get("output", 0)))

        # Find the group with the most points
        best_group = max(groups.values(), key=len) if groups else []
        best_group.sort()

        if len(best_group) >= 2:
            slopes = []
            for i in range(1, len(best_group)):
                dx = best_group[i][0] - best_group[i-1][0]
                dy = best_group[i][1] - best_group[i-1][1]
                if abs(dx) > 0.01:
                    slopes.append(dy / dx)

            avg_slope = sum(slopes) / len(slopes) if slopes else 0
            return {"slope": round(avg_slope, 2), "points": best_group}

        return {"slope": 0, "points": best_group}

    def _check_interaction(self) -> bool:
        """Check if variables interact (non-additive effects)."""
        if len(self.variables) != 2 or len(self.experiments) < 6:
            return False

        v1, v2 = self.variables

        # Find experiments where we can check interaction
        baseline = None
        v1_changed = None
        v2_changed = None
        both_changed = None

        for exp in self.experiments:
            x1 = exp["inputs"].get(v1, 0)
            x2 = exp["inputs"].get(v2, 0)
            y = exp.get("output", 0)

            lo1, hi1 = self.ranges.get(v1, (-5, 5))
            lo2, hi2 = self.ranges.get(v2, (-5, 5))
            m1 = (lo1 + hi1) / 2
            m2 = (lo2 + hi2) / 2

            if abs(x1 - m1) < 0.1 and abs(x2 - m2) < 0.1:
                baseline = y
            elif abs(x1 - (m1 + 1)) < 0.1 and abs(x2 - m2) < 0.1:
                v1_changed = y
            elif abs(x1 - m1) < 0.1 and abs(x2 - (m2 + 1)) < 0.1:
                v2_changed = y
            elif abs(x1 - (m1 + 1)) < 0.1 and abs(x2 - (m2 + 1)) < 0.1:
                both_changed = y

        if all(v is not None for v in [baseline, v1_changed, v2_changed, both_changed]):
            expected_additive = (v1_changed - baseline) + (v2_changed - baseline) + baseline
            actual = both_changed
            return abs(actual - expected_additive) > 0.5

        return False

    # ── Phase: Hypothesize ───────────────────────────────────────────────

    def _hypothesize_phase(self) -> Tuple[Dict, str]:
        """Form a hypothesis based on analysis."""
        if not self.observations:
            self.phase = "probe"
            return self._probe_phase(self.total_budget - self.used)

        hypothesis = self._form_hypothesis()
        if hypothesis:
            self.hypothesis = hypothesis
            self.phase = "test"
            return (
                {"action": "hypothesize", "expression": hypothesis},
                f"Forming hypothesis: y = {hypothesis}"
            )

        # If no hypothesis, go to prediction with simple model
        self.phase = "predict"
        return self._predict_phase()

    def _form_hypothesis(self) -> Optional[str]:
        """Form a mathematical hypothesis from observations."""
        obs = self.observations

        if len(self.variables) == 1:
            var = self.variables[0]

            # Check for conditional
            discontinuity = obs.get("has_discontinuity")
            if discontinuity is not None:
                return self._form_conditional_hypothesis(var, discontinuity)

            # Polynomial fit
            fit = obs.get("coefficients", {})
            degree = fit.get("degree", 1)
            coeffs = fit.get("coeffs", [0, 0])

            if degree == 1 and len(coeffs) >= 2:
                a = round(coeffs[0], 1)
                b = round(coeffs[1], 1)
                # Clean up representation
                a_int = int(a) if a == int(a) else a
                b_int = int(b) if b == int(b) else b
                if b_int >= 0:
                    return f"{a_int}*{var} + {b_int}"
                else:
                    return f"{a_int}*{var} - {abs(b_int)}"

            elif degree == 2 and len(coeffs) >= 3:
                a = round(coeffs[0], 1)
                b = round(coeffs[1], 1)
                c = round(coeffs[2], 1)
                a_int = int(a) if a == int(a) else a
                b_int = int(b) if b == int(b) else b
                c_int = int(c) if c == int(c) else c
                return f"{a_int}*{var}**2 + ({b_int})*{var} + ({c_int})"

            elif degree == 3 and len(coeffs) >= 4:
                parts = []
                vars_str = [f"{var}**3", f"{var}**2", f"{var}", "1"]
                for coeff, v_str in zip(coeffs, vars_str):
                    c = round(coeff, 1)
                    c_int = int(c) if c == int(c) else c
                    if v_str == "1":
                        parts.append(f"({c_int})")
                    else:
                        parts.append(f"({c_int})*{v_str}")
                return " + ".join(parts)

        elif len(self.variables) == 2:
            v1, v2 = self.variables
            v1_eff = obs.get("v1_effect", {})
            v2_eff = obs.get("v2_effect", {})
            has_interaction = obs.get("has_interaction", False)

            a = v1_eff.get("slope", 0)
            b = v2_eff.get("slope", 0)

            # Get intercept from baseline
            baseline_output = None
            lo1, hi1 = self.ranges.get(v1, (-5, 5))
            lo2, hi2 = self.ranges.get(v2, (-5, 5))
            m1 = (lo1 + hi1) / 2
            m2 = (lo2 + hi2) / 2

            for exp in self.experiments:
                x1 = exp["inputs"].get(v1, 0)
                x2 = exp["inputs"].get(v2, 0)
                if abs(x1 - m1) < 0.1 and abs(x2 - m2) < 0.1:
                    baseline_output = exp.get("output", 0)
                    break

            if baseline_output is not None:
                c = round(baseline_output - a * m1 - b * m2, 1)
            else:
                c = 0

            a_int = int(a) if a == int(a) else round(a, 1)
            b_int = int(b) if b == int(b) else round(b, 1)
            c_int = int(c) if c == int(c) else round(c, 1)

            if has_interaction:
                # Estimate interaction coefficient
                interaction_coeff = self._estimate_interaction_coeff()
                i_int = int(interaction_coeff) if interaction_coeff == int(interaction_coeff) else round(interaction_coeff, 1)

                # Recalculate constant
                if baseline_output is not None:
                    c = round(baseline_output - a * m1 - b * m2 - interaction_coeff * m1 * m2, 1)
                    c_int = int(c) if c == int(c) else round(c, 1)

                return f"{i_int}*{v1}*{v2} + ({a_int})*{v1} + ({b_int})*{v2} + ({c_int})"
            else:
                return f"{a_int}*{v1} + ({b_int})*{v2} + ({c_int})"

        return None

    def _form_conditional_hypothesis(self, var: str, breakpoint: float) -> str:
        """Form a conditional/piecewise hypothesis."""
        import numpy as np

        points = self.observations.get("points", [])
        if not points:
            return f"0*{var}"

        above = [(x, y) for x, y in points if x > breakpoint]
        below = [(x, y) for x, y in points if x <= breakpoint]

        def fit_line(pts):
            if len(pts) < 2:
                return 0, 0
            xs = np.array([p[0] for p in pts])
            ys = np.array([p[1] for p in pts])
            coeffs = np.polyfit(xs, ys, 1)
            return round(float(coeffs[0]), 1), round(float(coeffs[1]), 1)

        a1, b1 = fit_line(above) if len(above) >= 2 else (0, 0)
        a2, b2 = fit_line(below) if len(below) >= 2 else (0, 0)

        a1_int = int(a1) if a1 == int(a1) else a1
        b1_int = int(b1) if b1 == int(b1) else b1
        a2_int = int(a2) if a2 == int(a2) else a2
        b2_int = int(b2) if b2 == int(b2) else b2

        bp_int = int(breakpoint) if breakpoint == int(breakpoint) else breakpoint

        return f"({a1_int}*{var} + ({b1_int})) if {var} > {bp_int} else ({a2_int}*{var} + ({b2_int}))"

    def _estimate_interaction_coeff(self) -> float:
        """Estimate the coefficient of the interaction term x1*x2."""
        if len(self.variables) != 2:
            return 0

        v1, v2 = self.variables
        lo1, hi1 = self.ranges.get(v1, (-5, 5))
        lo2, hi2 = self.ranges.get(v2, (-5, 5))
        m1 = (lo1 + hi1) / 2
        m2 = (lo2 + hi2) / 2

        baseline = v1_only = v2_only = both = None

        for exp in self.experiments:
            x1 = exp["inputs"].get(v1, 0)
            x2 = exp["inputs"].get(v2, 0)
            y = exp.get("output", 0)

            if abs(x1 - m1) < 0.1 and abs(x2 - m2) < 0.1:
                baseline = y
            elif abs(x1 - (m1 + 1)) < 0.1 and abs(x2 - m2) < 0.1:
                v1_only = y
            elif abs(x1 - m1) < 0.1 and abs(x2 - (m2 + 1)) < 0.1:
                v2_only = y
            elif abs(x1 - (m1 + 1)) < 0.1 and abs(x2 - (m2 + 1)) < 0.1:
                both = y

        if all(v is not None for v in [baseline, v1_only, v2_only, both]):
            interaction = both - v1_only - v2_only + baseline
            return round(interaction, 1)

        return 0

    # ── Phase: Test ──────────────────────────────────────────────────────

    def _test_phase(self, remaining: int) -> Tuple[Dict, str]:
        """Validate hypothesis with targeted experiments."""
        if self.hypothesis_confirmed:
            self.phase = "predict"
            return self._predict_phase()

        if not self.test_plan:
            self.test_plan = self._create_test_plan()
            self.test_index = 0

        if self.test_index < len(self.test_plan) and remaining > 2:
            inputs = self.test_plan[self.test_index]
            self.test_index += 1
            self.used += 1
            return (
                {"action": "experiment", "inputs": inputs},
                "Validating hypothesis with targeted test"
            )

        # Done testing, move to refine or predict
        self.phase = "refine"
        return self._refine_phase(remaining)

    def _create_test_plan(self) -> List[Dict[str, float]]:
        """Create experiments specifically to test the hypothesis."""
        plan = []
        for var in self.variables:
            lo, hi = self.ranges.get(var, (-10, 10))
            # Test at points we haven't tried yet
            test_vals = [lo + 0.7, hi - 0.7, (lo + hi) / 4, 3 * (lo + hi) / 4]
            for val in test_vals[:2]:
                inputs = {}
                for v in self.variables:
                    if v == var:
                        inputs[v] = round(val, 2)
                    else:
                        lo_v, hi_v = self.ranges.get(v, (-5, 5))
                        inputs[v] = round((lo_v + hi_v) / 2, 2)
                plan.append(inputs)
        return plan[:4]

    # ── Phase: Refine ────────────────────────────────────────────────────

    def _refine_phase(self, remaining: int) -> Tuple[Dict, str]:
        """Refine hypothesis based on test results."""
        # Re-analyze with all data
        self.observations = self._analyze_data()
        new_hypothesis = self._form_hypothesis()

        if new_hypothesis and new_hypothesis != self.hypothesis:
            self.hypothesis = new_hypothesis
            self.phase = "predict"
            return (
                {"action": "hypothesize", "expression": new_hypothesis},
                f"Refined hypothesis: y = {new_hypothesis}"
            )

        self.phase = "predict"
        return self._predict_phase()

    # ── Phase: Predict ───────────────────────────────────────────────────

    def _predict_phase(self) -> Tuple[Dict, str]:
        """Generate predictions for test cases."""
        predictions = self._make_predictions()
        return (
            {"action": "predict", "predictions": predictions},
            f"Submitting {len(predictions)} predictions based on hypothesis"
        )

    def _make_predictions(self) -> List[float]:
        """Generate predictions for all test cases."""
        from ..verifier import SafeMathEvaluator

        predictions = []
        evaluator = SafeMathEvaluator()

        for case in self.test_cases:
            if self.hypothesis:
                try:
                    pred = evaluator.evaluate(self.hypothesis, case)
                    predictions.append(round(pred, 4))
                    continue
                except (ValueError, Exception):
                    pass

            # Fallback: use nearest experiment
            pred = self._nearest_neighbor_predict(case)
            predictions.append(round(pred, 4))

        return predictions

    def _nearest_neighbor_predict(self, case: Dict[str, float]) -> float:
        """Fallback prediction using nearest neighbor from experiments."""
        if not self.experiments:
            return 0.0

        best_dist = float("inf")
        best_output = 0.0

        for exp in self.experiments:
            if exp.get("output") is None:
                continue
            dist = sum(
                (exp["inputs"].get(v, 0) - case.get(v, 0)) ** 2
                for v in self.variables
            )
            if dist < best_dist:
                best_dist = dist
                best_output = exp["output"]

        return best_output
