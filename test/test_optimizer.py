"""Unit tests for optimizer"""
import unittest
import numpy as np
from core import ProjectFWOptimizer, Market, PortfolioConstraints

class TestOptimizer(unittest.TestCase):
    def setUp(self):
        self.optimizer = ProjectFWOptimizer(max_iterations=100)
        self.constraints = PortfolioConstraints(
            max_total_exposure=0.75,
            max_single_position=0.20
        )
        
        # Create test markets
        self.markets = [
            Market("1", "Test 1", "A", 0.6, 0.7, 100000, 50000, "Crypto", ""),
            Market("2", "Test 2", "C", 0.4, 0.3, 100000, 50000, "Politics", ""),
            Market("3", "Test 3", "E", 0.3, 0.4, 100000, 50000, "Sports", ""),
        ]
    
    def test_convergence(self):
        """Test that optimizer converges"""
        allocations, status, info = self.optimizer.optimize(self.markets, self.constraints)
        
        self.assertEqual(status.value, "converged")
        self.assertTrue(
            info['fw_gap'] < self.optimizer.tolerance or 'reason' in info
        )
        self.assertEqual(len(allocations), len(self.markets))
    
    def test_constraints(self):
        """Test that constraints are satisfied"""
        allocations, _, _ = self.optimizer.optimize(self.markets, self.constraints)
        
        # Check total exposure
        self.assertLessEqual(np.sum(allocations), self.constraints.max_total_exposure + 1e-6)
        
        # Check individual limits
        for a in allocations:
            self.assertLessEqual(a, self.constraints.max_single_position + 1e-6)
            self.assertGreaterEqual(a, 0)

if __name__ == '__main__':
    unittest.main()
