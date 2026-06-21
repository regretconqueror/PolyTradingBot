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

    def kelly_objective(self, x: np.ndarray, markets: List[Market], capital: float = 10000.0) -> float:
        """Calculate expected log utility with slippage awareness."""
        total = 0.0
        eps = 1e-10

        for i, market in enumerate(markets):
            if x[i] <= 0:
                continue

            q = np.clip(market.price, eps, 1 - eps)
            p = np.clip(market.probability, eps, 1 - eps)

            # Incorporate slippage into effective price
            liquidity = max(1.0, market.liquidity)
            size_usd = x[i] * capital
            slippage_pct = min(0.99, 0.1 * np.sqrt(size_usd / liquidity))
            q_eff = q * (1 + slippage_pct)

            win_return = x[i] * (1 / q_eff - 1)
            lose_return = -x[i]
            total += p * np.log1p(win_return) + (1 - p) * np.log1p(lose_return)

        return total

    def gradient(self, x: np.ndarray, markets: List[Market], capital: float = 10000.0) -> np.ndarray:
        """Compute gradient of the Kelly objective with slippage awareness."""
        grad = np.zeros_like(x)
        eps = 1e-10

        for i, market in enumerate(markets):
            if x[i] >= 1 - eps:
                grad[i] = -1e10
                continue

            q = np.clip(market.price, eps, 1 - eps)
            p = np.clip(market.probability, eps, 1 - eps)

            # Incorporate slippage into derivative of win return
            liquidity = max(1.0, market.liquidity)
            size_usd = x[i] * capital
            slippage_pct = min(0.99, 0.1 * np.sqrt(size_usd / liquidity))
            q_eff = q * (1 + slippage_pct)

            win_return = x[i] * (1 / q_eff - 1)

            # W'(x) derivative calculation
            deriv_win_return = (1 + 0.5 * slippage_pct) / (q * (1 + slippage_pct) ** 2) - 1

            win_deriv = p * deriv_win_return / (1 + win_return)
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

        Enforces three levels of constraint simultaneously:
          1. Total portfolio exposure  (max_total_exposure)
          2. Per-category exposure     (max_category_exposure)
          3. Per-condition exposure    (max_condition_exposure) — joint constraint
             across all outcome tokens of the same condition_id so the optimizer
             cannot simultaneously bet YES and NO on the same market.
        """
        n = len(markets)
        s = np.zeros(n)
        remaining = constraints.max_total_exposure

        # Track running allocations per category and per condition_id
        category_alloc: Dict[str, float] = {}
        condition_alloc: Dict[str, float] = {}

        sorted_indices = np.argsort(-grad)

        for i in sorted_indices:
            if remaining <= 0:
                break
            if grad[i] <= 0:
                continue

            market = markets[i]
            category = market.category
            condition = market.condition_id  # joint constraint key

            # ── Category cap ────────────────────────────────────────────────
            category_limit = constraints.max_category_exposure.get(
                category,
                constraints.max_category_exposure.get("default", 0.25),
            )
            category_used = category_alloc.get(category, 0.0)
            category_remaining = category_limit - category_used

            # ── Condition (joint) cap ────────────────────────────────────────
            condition_used = condition_alloc.get(condition, 0.0)
            condition_remaining = constraints.max_condition_exposure - condition_used

            max_alloc = min(
                constraints.max_single_position,
                remaining,
                category_remaining,
                condition_remaining,
            )

            if max_alloc > 0:
                s[i] = max_alloc
                remaining -= max_alloc
                category_alloc[category] = category_used + max_alloc
                condition_alloc[condition] = condition_used + max_alloc

        return s


    def line_search(self, x: np.ndarray, direction: np.ndarray, markets: List[Market], capital: float = 10000.0) -> float:
        """Backtracking line search for step size."""
        gamma = 1.0
        current_obj = self.kelly_objective(x, markets, capital=capital)
        grad = self.gradient(x, markets, capital=capital)

        for _ in range(20):
            x_new = (1 - gamma) * x + gamma * direction
            x_new = np.clip(x_new, 0, 1)
            new_obj = self.kelly_objective(x_new, markets, capital=capital)

            if new_obj > current_obj + 1e-4 * gamma * np.dot(grad, direction - x):
                return gamma

            gamma *= 0.5

        return gamma

    def optimize(
        self,
        markets: List[Market],
        constraints: PortfolioConstraints,
        initial_guess: Optional[np.ndarray] = None,
        capital: float = 10000.0,
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

        previous_objective = self.kelly_objective(x, markets, capital=capital)

        for iteration in range(self.max_iterations):
            grad = self.gradient(x, markets, capital=capital)
            s = self.solve_linear_subproblem(grad, markets, constraints)

            raw_gap = float(np.dot(grad, s - x))
            gap = max(0.0, raw_gap)

            self.history.append({
                "iteration": iteration,
                "objective": self.kelly_objective(x, markets, capital=capital),
                "gap": gap,
                "raw_gap": raw_gap,
            })

            if gap < self.tolerance or np.allclose(s, x, atol=self.tolerance, rtol=0):
                logger.info("Converged at iteration %s", iteration)
                return x, OptimizationStatus.CONVERGED, {
                    "iterations": iteration,
                    "final_objective": self.kelly_objective(x, markets, capital=capital),
                    "fw_gap": gap,
                    "raw_fw_gap": raw_gap,
                }

            gamma = self.line_search(x, s, markets, capital=capital)
            if gamma < self.min_step_size:
                logger.info(
                    "Converged at iteration %s: line search step below %s",
                    iteration,
                    self.min_step_size,
                )
                return x, OptimizationStatus.CONVERGED, {
                    "iterations": iteration,
                    "final_objective": self.kelly_objective(x, markets, capital=capital),
                    "fw_gap": gap,
                    "raw_fw_gap": raw_gap,
                    "reason": "line_search_stalled",
                }

            previous_x = x.copy()
            x = (1 - gamma) * x + gamma * s
            current_objective = self.kelly_objective(x, markets, capital=capital)

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
            "final_objective": self.kelly_objective(x, markets, capital=capital),
            "fw_gap": gap,
            "raw_fw_gap": raw_gap,
        }
