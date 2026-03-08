"""
Safe Expression Evaluator and Hypothesis Verifier.

Provides a secure AST-based mathematical expression evaluator (no eval/exec)
and a scoring system that compares agent hypotheses against ground truth
using numerical evaluation on test points.
"""

import ast
import math
import operator
from typing import Dict, List, Optional, Tuple, Any


class SafeMathEvaluator:
    """
    Safely evaluates mathematical expressions using Python's AST module.
    
    Supports:
        - Arithmetic: +, -, *, /, **, %
        - Functions: sin, cos, tan, exp, log, sqrt, abs, min, max
        - Constants: pi, e
        - Comparisons: >, <, >=, <=, ==, !=
        - Ternary: value_if_true if condition else value_if_false
        - Custom: where(condition, true_val, false_val)
    """

    SAFE_CONSTANTS = {
        "pi": math.pi,
        "e": math.e,
        "True": True,
        "False": False,
    }

    SAFE_FUNCTIONS = {
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "exp": math.exp,
        "log": math.log,
        "sqrt": math.sqrt,
        "abs": abs,
        "min": min,
        "max": max,
        "round": round,
    }

    OPERATORS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Pow: operator.pow,
        ast.Mod: operator.mod,
    }

    UNARY_OPS = {
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    COMPARE_OPS = {
        ast.Gt: operator.gt,
        ast.Lt: operator.lt,
        ast.GtE: operator.ge,
        ast.LtE: operator.le,
        ast.Eq: operator.eq,
        ast.NotEq: operator.ne,
    }

    def evaluate(self, expr_str: str, variables: Dict[str, float]) -> float:
        """
        Safely evaluate a mathematical expression string.

        Args:
            expr_str: Mathematical expression (e.g., "2*x + sin(y)")
            variables: Variable name -> value mapping

        Returns:
            The computed result as a float.

        Raises:
            ValueError: If the expression contains unsafe constructs.
        """
        try:
            tree = ast.parse(expr_str.strip(), mode="eval")
            result = self._eval_node(tree.body, variables)
            return float(result)
        except (ValueError, TypeError, ZeroDivisionError, OverflowError) as e:
            raise ValueError(f"Expression evaluation error: {e}")
        except Exception as e:
            raise ValueError(f"Cannot parse expression '{expr_str}': {e}")

    def _eval_node(self, node, variables: Dict[str, float]):
        """Recursively evaluate an AST node."""

        # ── Constants (numbers) ──
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError(f"Unsupported constant type: {type(node.value)}")

        # ── Variable names and built-in constants ──
        if isinstance(node, ast.Name):
            name = node.id
            if name in variables:
                return variables[name]
            if name in self.SAFE_CONSTANTS:
                return self.SAFE_CONSTANTS[name]
            raise ValueError(f"Unknown variable or constant: '{name}'")

        # ── Binary operations (+, -, *, /, **, %) ──
        if isinstance(node, ast.BinOp):
            left = self._eval_node(node.left, variables)
            right = self._eval_node(node.right, variables)
            op_fn = self.OPERATORS.get(type(node.op))
            if op_fn is None:
                raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
            try:
                result = op_fn(left, right)
                if isinstance(result, complex):
                    return float("nan")
                return result
            except (ZeroDivisionError, OverflowError, ValueError):
                return float("nan")

        # ── Unary operations (-, +) ──
        if isinstance(node, ast.UnaryOp):
            operand = self._eval_node(node.operand, variables)
            op_fn = self.UNARY_OPS.get(type(node.op))
            if op_fn is None:
                raise ValueError(f"Unsupported unary op: {type(node.op).__name__}")
            return op_fn(operand)

        # ── Function calls: sin(x), cos(x), where(cond, a, b), etc. ──
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("Only simple function calls are allowed")

            func_name = node.func.id

            # Special handling for 'where(condition, true_val, false_val)'
            if func_name == "where":
                if len(node.args) != 3:
                    raise ValueError("where() requires exactly 3 arguments")
                cond = self._eval_node(node.args[0], variables)
                true_val = self._eval_node(node.args[1], variables)
                false_val = self._eval_node(node.args[2], variables)
                return true_val if cond else false_val

            if func_name not in self.SAFE_FUNCTIONS:
                raise ValueError(f"Unknown function: '{func_name}'")

            args = [self._eval_node(arg, variables) for arg in node.args]
            try:
                return self.SAFE_FUNCTIONS[func_name](*args)
            except (ValueError, OverflowError):
                return float("nan")

        # ── Comparisons (>, <, >=, <=, ==, !=) ──
        if isinstance(node, ast.Compare):
            left = self._eval_node(node.left, variables)
            for op, comparator in zip(node.ops, node.comparators):
                right = self._eval_node(comparator, variables)
                op_fn = self.COMPARE_OPS.get(type(op))
                if op_fn is None:
                    raise ValueError(f"Unsupported comparison: {type(op).__name__}")
                if not op_fn(left, right):
                    return False
                left = right
            return True

        # ── Ternary (if-else expression) ──
        if isinstance(node, ast.IfExp):
            condition = self._eval_node(node.test, variables)
            if condition:
                return self._eval_node(node.body, variables)
            else:
                return self._eval_node(node.orelse, variables)

        # ── Boolean operations (and, or) ──
        if isinstance(node, ast.BoolOp):
            if isinstance(node.op, ast.And):
                result = True
                for value in node.values:
                    result = result and self._eval_node(value, variables)
                return result
            elif isinstance(node.op, ast.Or):
                result = False
                for value in node.values:
                    result = result or self._eval_node(value, variables)
                return result

        raise ValueError(f"Unsupported expression type: {type(node).__name__}")


class HypothesisVerifier:
    """
    Verifies agent hypotheses against ground truth worlds.
    
    Uses numerical evaluation on test points to compute similarity scores.
    """

    def __init__(self):
        self.evaluator = SafeMathEvaluator()

    def verify(
        self,
        hypothesis_expr: str,
        world,
        n_test_points: int = 200,
        seed: int = 99,
    ) -> Dict[str, Any]:
        """
        Verify a hypothesis against the world's ground truth.

        Args:
            hypothesis_expr: Mathematical expression string.
            world: The World object with ground truth.
            n_test_points: Number of test points for evaluation.
            seed: Random seed for test point generation.

        Returns:
            Dict with: score (0-1), r_squared, mae, details
        """
        import numpy as np

        rng = np.random.default_rng(seed)
        variables = world.variables
        ranges = world.variable_ranges

        # Generate dense test points
        test_points = []
        for _ in range(n_test_points):
            point = {}
            for var in variables:
                lo, hi = ranges[var]
                point[var] = float(rng.uniform(lo, hi))
            test_points.append(point)

        # Evaluate ground truth
        truth_values = []
        for point in test_points:
            result = world.run_experiment(point)
            val = result.get("output")
            if val is not None:
                truth_values.append(val)
            else:
                truth_values.append(float("nan"))

        # Evaluate hypothesis
        hyp_values = []
        parse_errors = 0
        for point in test_points:
            try:
                val = self.evaluator.evaluate(hypothesis_expr, point)
                hyp_values.append(val)
            except ValueError:
                hyp_values.append(float("nan"))
                parse_errors += 1

        # Filter out NaN pairs
        valid_pairs = [
            (t, h)
            for t, h in zip(truth_values, hyp_values)
            if not (math.isnan(t) or math.isnan(h) or math.isinf(t) or math.isinf(h))
        ]

        if len(valid_pairs) < 10:
            return {
                "score": 0.0,
                "r_squared": 0.0,
                "mae": float("inf"),
                "valid_points": len(valid_pairs),
                "total_points": n_test_points,
                "parse_errors": parse_errors,
                "details": "Too few valid evaluation points.",
            }

        truths = np.array([p[0] for p in valid_pairs])
        hyps = np.array([p[1] for p in valid_pairs])

        # Mean Absolute Error
        mae = float(np.mean(np.abs(truths - hyps)))

        # R-squared (coefficient of determination)
        ss_res = float(np.sum((truths - hyps) ** 2))
        ss_tot = float(np.sum((truths - np.mean(truths)) ** 2))

        if ss_tot == 0:
            r_squared = 1.0 if ss_res < 1e-6 else 0.0
        else:
            r_squared = max(0.0, 1.0 - ss_res / ss_tot)

        # Exact match ratio (within tolerance)
        tolerance = max(0.01, 0.01 * np.std(truths)) if np.std(truths) > 0 else 0.01
        exact_matches = float(np.mean(np.abs(truths - hyps) < tolerance))

        # Composite score
        score = 0.6 * r_squared + 0.4 * exact_matches

        return {
            "score": round(score, 4),
            "r_squared": round(r_squared, 4),
            "mae": round(mae, 4),
            "exact_match_ratio": round(exact_matches, 4),
            "valid_points": len(valid_pairs),
            "total_points": n_test_points,
            "parse_errors": parse_errors,
            "details": self._describe_score(score, r_squared, mae),
        }

    def score_predictions(
        self,
        predictions: List[Optional[float]],
        actuals: List[Optional[float]],
    ) -> Dict[str, Any]:
        """
        Score agent predictions against actual test case answers.

        Returns:
            Dict with: accuracy, mae, r_squared, per_case results
        """
        import numpy as np

        if len(predictions) != len(actuals):
            return {"accuracy": 0.0, "error": "Prediction count mismatch"}

        per_case = []
        valid_preds = []
        valid_actuals = []

        for i, (pred, actual) in enumerate(zip(predictions, actuals)):
            if pred is None or actual is None:
                per_case.append({"case": i, "predicted": pred, "actual": actual, "correct": False, "error": abs(0) })
                continue

            error = abs(pred - actual)
            tolerance = max(0.5, abs(actual) * 0.05)  # 5% or 0.5 absolute
            correct = error <= tolerance

            per_case.append({
                "case": i,
                "predicted": round(pred, 4),
                "actual": round(actual, 4),
                "error": round(error, 4),
                "correct": correct,
            })

            valid_preds.append(pred)
            valid_actuals.append(actual)

        if not valid_preds:
            return {"accuracy": 0.0, "mae": float("inf"), "r_squared": 0.0, "per_case": per_case}

        preds_arr = np.array(valid_preds)
        actuals_arr = np.array(valid_actuals)

        accuracy = sum(1 for c in per_case if c.get("correct", False)) / len(per_case)
        mae = float(np.mean(np.abs(preds_arr - actuals_arr)))

        ss_res = float(np.sum((actuals_arr - preds_arr) ** 2))
        ss_tot = float(np.sum((actuals_arr - np.mean(actuals_arr)) ** 2))
        r_squared = max(0.0, 1.0 - ss_res / ss_tot) if ss_tot > 0 else (1.0 if ss_res < 1e-6 else 0.0)

        return {
            "accuracy": round(accuracy, 4),
            "mae": round(mae, 4),
            "r_squared": round(r_squared, 4),
            "per_case": per_case,
        }

    @staticmethod
    def _describe_score(score: float, r_squared: float, mae: float) -> str:
        """Generate a human-readable description of the score."""
        if score >= 0.95:
            return "Excellent! Near-perfect match with ground truth."
        elif score >= 0.8:
            return "Very good! The hypothesis captures most of the pattern."
        elif score >= 0.6:
            return "Good. The hypothesis captures the general trend but misses details."
        elif score >= 0.3:
            return "Partial match. Some aspects are correct but significant errors remain."
        elif score > 0:
            return "Poor match. The hypothesis misses major aspects of the pattern."
        else:
            return "No match. The hypothesis does not describe the system."
