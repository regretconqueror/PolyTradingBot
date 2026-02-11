"""
ProjectFW Optimizer
Frank-Wolfe algorithm for Kelly Criterion portfolio optimization
"""
import numpy as np
from typing import List, Tuple, Dict, Optional
import logging

from .models import Market, PortfolioConstraints, OptimizationStatus

logger = logging.getLogger(__name__)

class ProjectFWOptimizer:
    """
    Frank-Wolfe (Conditional Gradient) algorithm for convex optimization.
    
    Solves: max Σᵢ [pᵢ·log(1+xᵢ·(1/qᵢ-1)) + (1-pᵢ)·log(1-xᵢ)]
    subject to linear constraints (budget, position limits, category limits)
    """
    
    def __init__(self, max_iterations: int = 1000, tolerance: float = 1e-6):
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.history = []
    
    def kelly_objective(self, x: np.ndarray, markets: List[Market]) -> float:
        """Calculate expected log utility (Kelly Criterion)"""
        total = 0.0
        eps = 1e-10
        
        for i, m in enumerate(markets):
            if x[i] <= 0:
                continue
            
            q = np.clip(m.current_price, eps, 1 - eps)
            p = np.clip(m.your_probability, eps, 1 - eps)
            
            # Win scenario
            win_return = x[i] * (1/q - 1)
            # Lose scenario
            lose_return = -x[i]
            
            # Expected log utility
            expected = p * np.log1p(win_return) + (1 - p) * np.log1p(lose_return)
            total += expected
            
        return total
    
    def gradient(self, x: np.ndarray, markets: List[Market]) -> np.ndarray:
        """Compute gradient of Kelly objective"""
        grad = np.zeros_like(x)
        eps = 1e-10
        
        for i, m in enumerate(markets):
            if x[i] >= 1 - eps:
                grad[i] = -1e10
                continue
            
            q = np.clip(m.current_price, eps, 1 - eps)
            p = np.clip(m.your_probability, eps, 1 - eps)
            
            win_factor = (1/q - 1)
            win_deriv = p * win_factor / (1 + x[i] * win_factor)
            lose_deriv = -(1 - p) / (1 - x[i])
            
            grad[i] = win_deriv + lose_deriv
            
        return grad
    
    def solve_linear_subproblem(self, grad: np.ndarray, markets: List[Market],
                               constraints: PortfolioConstraints) -> np.ndarray:
        """
        Solve LP: max <grad, s> s.t. s in feasible set
        Uses greedy algorithm - efficient for this constraint structure
        """
        n = len(markets)
        s = np.zeros(n)
        remaining = constraints.max_total_exposure
        
        # Sort by gradient descending (highest marginal utility first)
        sorted_indices = np.argsort(-grad)
        
        for i in sorted_indices:
            if remaining <= 0:
                break
            
            # Skip negative gradients (not profitable at margin)
            if grad[i] <= 0:
                continue
            
            m = markets[i]
            cat = m.category
            
            # Check category limit
            cat_limit = constraints.max_category_exposure.get(cat, 0.25)
            cat_current = sum(s[j] for j in range(n) if markets[j].category == cat)
            cat_remaining = cat_limit - cat_current
            
            # Calculate maximum allowable allocation
            max_alloc = min(
                constraints.max_single_position,
                remaining,
                cat_remaining
            )
            
            if max_alloc >= constraints.min_bet_size:
                s[i] = max_alloc
                remaining -= max_alloc
        
        return s
    
    def line_search(self, x: np.ndarray, direction: np.ndarray, 
                   markets: List[Market]) -> float:
        """Backtracking line search for step size"""
        gamma = 1.0
        current_obj = self.kelly_objective(x, markets)
        
        for _ in range(20):
            x_new = (1 - gamma) * x + gamma * direction
            x_new = np.clip(x_new, 0, 1)
            new_obj = self.kelly_objective(x_new, markets)
            
            # Armijo condition for sufficient decrease
            if new_obj > current_obj + 1e-4 * gamma * np.dot(
                self.gradient(x, markets), direction - x):
                return gamma
            
            gamma *= 0.5
            
        return gamma
    
    def optimize(self, markets: List[Market], constraints: PortfolioConstraints,
                initial_guess: Optional[np.ndarray] = None) -> Tuple[np.ndarray, OptimizationStatus, Dict]:
        """
        Main Frank-Wolfe optimization loop
        
        Returns:
            x: Optimal allocations
            status: Convergence status
            info: Dictionary with iteration count, objective value, etc.
        """
        n = len(markets)
        if n == 0:
            return np.array([]), OptimizationStatus.INFEASIBLE, {"error": "No markets provided"}
        
        # Initialize
        x = initial_guess.copy() if initial_guess is not None else np.zeros(n)
        x = np.clip(x, 0, constraints.max_single_position)
        if np.sum(x) > constraints.max_total_exposure:
            x *= constraints.max_total_exposure / np.sum(x)
        
        self.history = []
        
        for iteration in range(self.max_iterations):
            grad = self.gradient(x, markets)
            s = self.solve_linear_subproblem(grad, markets, constraints)
            
            # Frank-Wolfe gap (duality gap)
            gap = np.dot(grad, s - x)
            
            self.history.append({
                'iteration': iteration,
                'objective': self.kelly_objective(x, markets),
                'gap': gap
            })
            
            # Check convergence
            if abs(gap) < self.tolerance:
                logger.info(f"✅ Converged at iteration {iteration}")
                return x, OptimizationStatus.CONVERGED, {
                    'iterations': iteration,
                    'final_objective': self.kelly_objective(x, markets),
                    'fw_gap': gap
                }
            
            # Line search and update
            gamma = self.line_search(x, s, markets)
            x = (1 - gamma) * x + gamma * s
            
            if iteration % 50 == 0:
                logger.info(f"Iter {iteration:3d}: obj={self.kelly_objective(x, markets):.6f}, gap={gap:.6e}")
        
        logger.warning(f"⚠️  Max iterations ({self.max_iterations}) reached")
        return x, OptimizationStatus.MAX_ITER, {
            'iterations': self.max_iterations,
            'final_objective': self.kelly_objective(x, markets),
            'fw_gap': gap
        }