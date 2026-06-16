"""
ProjectFW Optimizer
Frank-Wolfe algorithm for Kelly Criterion portfolio optimization
"""
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from .models import Market, OptimizationStatus, PortfolioConstraints

logger = logging.getLogger(__name__)


class ProjectFWOptimizer:
    """
    Frank-Wolfe (Conditional Gradient) algorithm for Kelly-style portfolio optimization.

    Solves:
        max sum_i p_i * log(1 + x_i * (1 / q_i - 1)) + (1 - p_i) * log(1 - x_i)

    subject to linear budget, position, and category constraints.
    """

    def __init__(self, max_iterations: int = 1000, tolerance: float = 1e-6):
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.min_step_size = 1e-8
        self.history = []

    def kelly_objective(self, x: np.ndarray, markets: List[Market]) -> float:
        """Calculate expected log utility."""
        total = 0.0
        eps = 1e-10

        for i, market in enumerate(markets):
            if x[i] <= 0:
                continue

            q = np.clip(market.price, eps, 1 - eps)
            p = np.clip(market.probability, eps, 1 - eps)

            win_return = x[i] * (1 / q - 1)
            lose_return = -x[i]
            total += p * np.log1p(win_return) + (1 - p) * np.log1p(lose_return)

        return total

    def gradient(self, x: np.ndarray, markets: List[Market]) -> np.ndarray:
        """Compute gradient of the Kelly objective."""
        grad = np.zeros_like(x)
        eps = 1e-10

        for i, market in enumerate(markets):
            if x[i] >= 1 - eps:
                grad[i] = -1e10
                continue

            q = np.clip(market.price, eps, 1 - eps)
            p = np.clip(market.probability, eps, 1 - eps)

            win_factor = 1 / q - 1
            win_deriv = p * win_factor / (1 + x[i] * win_factor)
            lose_deriv = -(1 - p) / (1 - x[i])
            grad[i] = win_deriv + lose_deriv

        return grad

    def solve_linear_subproblem(
        self,
        grad: np.ndarray,
        markets: List[Market],
        constraints: PortfolioConstraints,
    ) -> np.ndarray:
        """
        Solve LP: max <grad, s> subject to the configured exposure constraints.

        The greedy implementation is exact enough for this simple constraint shape,
        but min-bet and category cutoffs can make the reported FW gap approximate.
        """
        n = len(markets)
        s = np.zeros(n)
        remaining = constraints.max_total_exposure

        sorted_indices = np.argsort(-grad)

        for i in sorted_indices:
            if remaining <= 0:
                break
            if grad[i] <= 0:
                continue

            category = markets[i].category
            category_limit = constraints.max_category_exposure.get(
                category,
                constraints.max_category_exposure.get("default", 0.25),
            )
            category_current = sum(s[j] for j in range(n) if markets[j].category == category)
            category_remaining = category_limit - category_current

            max_alloc = min(
                constraints.max_single_position,
                remaining,
                category_remaining,
            )

            if max_alloc >= constraints.min_bet_size:
                s[i] = max_alloc
                remaining -= max_alloc

        return s

    def line_search(self, x: np.ndarray, direction: np.ndarray, markets: List[Market]) -> float:
        """Backtracking line search for step size."""
        gamma = 1.0
        current_obj = self.kelly_objective(x, markets)
        grad = self.gradient(x, markets)

        for _ in range(20):
            x_new = (1 - gamma) * x + gamma * direction
            x_new = np.clip(x_new, 0, 1)
            new_obj = self.kelly_objective(x_new, markets)

            if new_obj > current_obj + 1e-4 * gamma * np.dot(grad, direction - x):
                return gamma

            gamma *= 0.5

        return gamma

    def optimize(
        self,
        markets: List[Market],
        constraints: PortfolioConstraints,
        initial_guess: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, OptimizationStatus, Dict]:
        """
        Main Frank-Wolfe optimization loop.

        Returns:
            x: optimal allocations
            status: convergence status
            info: iteration count, objective value, and gap diagnostics
        """
        n = len(markets)
        if n == 0:
            return np.array([]), OptimizationStatus.INFEASIBLE, {"error": "No markets provided"}

        x = initial_guess.copy() if initial_guess is not None else np.zeros(n)
        x = np.clip(x, 0, constraints.max_single_position)
        if np.sum(x) > constraints.max_total_exposure:
            x *= constraints.max_total_exposure / np.sum(x)

        self.history = []
        raw_gap = 0.0
        gap = 0.0

        previous_objective = self.kelly_objective(x, markets)

        for iteration in range(self.max_iterations):
            grad = self.gradient(x, markets)
            s = self.solve_linear_subproblem(grad, markets, constraints)

            raw_gap = float(np.dot(grad, s - x))
            gap = max(0.0, raw_gap)

            self.history.append({
                "iteration": iteration,
                "objective": self.kelly_objective(x, markets),
                "gap": gap,
                "raw_gap": raw_gap,
            })

            if gap < self.tolerance or np.allclose(s, x, atol=self.tolerance, rtol=0):
                logger.info("Converged at iteration %s", iteration)
                return x, OptimizationStatus.CONVERGED, {
                    "iterations": iteration,
                    "final_objective": self.kelly_objective(x, markets),
                    "fw_gap": gap,
                    "raw_fw_gap": raw_gap,
                }

            gamma = self.line_search(x, s, markets)
            if gamma < self.min_step_size:
                logger.info(
                    "Converged at iteration %s: line search step below %s",
                    iteration,
                    self.min_step_size,
                )
                return x, OptimizationStatus.CONVERGED, {
                    "iterations": iteration,
                    "final_objective": self.kelly_objective(x, markets),
                    "fw_gap": gap,
                    "raw_fw_gap": raw_gap,
                    "reason": "line_search_stalled",
                }

            previous_x = x.copy()
            x = (1 - gamma) * x + gamma * s
            current_objective = self.kelly_objective(x, markets)

            if abs(current_objective - previous_objective) < self.tolerance:
                logger.info("Converged at iteration %s: objective improvement below tolerance", iteration)
                return x, OptimizationStatus.CONVERGED, {
                    "iterations": iteration,
                    "final_objective": current_objective,
                    "fw_gap": gap,
                    "raw_fw_gap": raw_gap,
                    "reason": "objective_improvement_below_tolerance",
                }
            previous_objective = current_objective

            if np.max(np.abs(x - previous_x)) < self.tolerance:
                logger.info("Converged at iteration %s: allocation change below tolerance", iteration)
                return x, OptimizationStatus.CONVERGED, {
                    "iterations": iteration,
                    "final_objective": current_objective,
                    "fw_gap": gap,
                    "raw_fw_gap": raw_gap,
                    "reason": "allocation_change_below_tolerance",
                }

            if iteration % 50 == 0:
                logger.info(
                    "Iter %3d: obj=%.6f, gap=%.6e",
                    iteration,
                    current_objective,
                    gap,
                )

        logger.warning("Max iterations (%s) reached", self.max_iterations)
        return x, OptimizationStatus.MAX_ITER, {
            "iterations": self.max_iterations,
            "final_objective": self.kelly_objective(x, markets),
            "fw_gap": gap,
            "raw_fw_gap": raw_gap,
        }
