"""
Unit tests for risk manager
"""
import unittest
import numpy as np
import sys
import os
from datetime import datetime

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.risk_manager import RiskManager
from core.models import PortfolioConstraints


class TestRiskManager(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures"""
        self.constraints = PortfolioConstraints(
            max_total_exposure=0.75,
            max_single_position=0.20,
            max_drawdown=0.15,
            min_bet_size=0.02,
            max_category_exposure={
                "Crypto": 0.30,
                "Politics": 0.25,
                "Sports": 0.20,
                "Science": 0.15,
                "default": 0.25
            }
        )
        self.risk_manager = RiskManager(self.constraints)

    def test_initialization(self):
        """Test that risk manager initializes correctly"""
        self.assertEqual(self.risk_manager.constraints.max_total_exposure, 0.75)
        self.assertEqual(len(self.risk_manager.positions), 0)

    def test_update_position(self):
        """Test updating a position"""
        self.risk_manager.update_position(
            token_id="TEST_TOKEN",
            size=100.0,
            price=0.5,
            side="BUY",
            category="Crypto"
        )

        self.assertIn("TEST_TOKEN", self.risk_manager.positions)
        position = self.risk_manager.positions["TEST_TOKEN"]
        self.assertEqual(position['size'], 100.0)
        self.assertEqual(position['entry_price'], 0.5)
        self.assertEqual(position['side'], "BUY")
        self.assertEqual(position['category'], "Crypto")
        # Check that the position has the expected fields
        self.assertIn('timestamp', position)
        self.assertIn('unrealized_pnl', position)
        self.assertIn('realized_pnl', position)

    def test_update_position_sell(self):
        """Test updating a SELL position"""
        self.risk_manager.update_position(
            token_id="TEST_TOKEN",
            size=50.0,
            price=0.6,
            side="SELL",
            category="Politics"
        )

        position = self.risk_manager.positions["TEST_TOKEN"]
        self.assertEqual(position['side'], "SELL")
        # Size should be stored as positive, side indicates direction
        self.assertEqual(position['size'], 50.0)
        self.assertEqual(position['entry_price'], 0.6)

    def test_calculate_portfolio_value(self):
        """Test portfolio value calculation"""
        # Add some positions
        self.risk_manager.update_position("TOKEN1", 100.0, 0.5, "BUY", "Crypto")   # long
        self.risk_manager.update_position("TOKEN2", 50.0, 0.8, "BUY", "Politics")    # long
        self.risk_manager.update_position("TOKEN3", 75.0, 0.4, "SELL", "Sports")     # short

        # Test with current prices
        current_prices = {
            "TOKEN1": 0.55,  # Up 10%
            "TOKEN2": 0.75,  # Down 6.25% from 0.8
            "TOKEN3": 0.35   # Down 12.5% from 0.4 (good for our short)
        }

        value = self.risk_manager.calculate_portfolio_value(current_prices)

        # Calculations:
        # TOKEN1 long: 100 * 0.55 = $55
        # TOKEN2 long: 50 * 0.75 = $37.50
        # TOKEN3 short: 75 * max(0.0, 2*0.4 - 0.35) = 75 * max(0.0, 0.8 - 0.35) = 75 * 0.45 = $33.75
        # (The formula in the code is: size * max(0.0, 2 * entry_price - current_price))
        expected = 55.0 + 37.5 + 33.75
        self.assertAlmostEqual(value, expected, places=2)

    def test_calculate_portfolio_value_missing_price(self):
        """Test portfolio value with missing price data"""
        self.risk_manager.update_position("TOKEN1", 100.0, 0.5, "BUY", "Crypto")

        # Missing price for TOKEN1
        current_prices = {}
        value = self.risk_manager.calculate_portfolio_value(current_prices)
        self.assertEqual(value, 0.0)  # Should return 0 for missing prices

    def test_check_risk_limits_within_bounds(self):
        """Test risk limit checking when within bounds"""
        # Add a small position
        self.risk_manager.update_position("TOKEN1", 50.0, 0.5, "BUY", "Crypto")  # $25 at price 0.5

        current_prices = {"TOKEN1": 0.5}
        portfolio_value = 1000.0  # $1000 portfolio

        violations = self.risk_manager.check_risk_limits(current_prices, portfolio_value)
        self.assertEqual(len(violations), 0)  # Should be no violations

    def test_check_risk_limits_exceeds_exposure(self):
        """Test risk limit checking when exposure too high"""
        # Add a large position that exceeds max exposure
        self.risk_manager.update_position("TOKEN1", 400.0, 0.5, "BUY", "Crypto")  # $200

        current_prices = {"TOKEN1": 0.5}
        portfolio_value = 200.0  # $200 portfolio (100% exposed)

        violations = self.risk_manager.check_risk_limits(current_prices, portfolio_value)
        self.assertGreater(len(violations), 0)
        self.assertTrue(any("exposure" in v['message'].lower() for v in violations))

    def test_check_risk_limits_exceeds_position_limit(self):
        """Test risk limit checking when position too large"""
        # Add position exceeding max single position (20% of portfolio)
        self.risk_manager.update_position("TOKEN1", 250.0, 0.5, "BUY", "Crypto")  # $125

        current_prices = {"TOKEN1": 0.5}
        portfolio_value = 500.0  # $500 portfolio
        # Position value = $125/$500 = 25% > 20% limit

        violations = self.risk_manager.check_risk_limits(current_prices, portfolio_value)
        self.assertGreater(len(violations), 0)
        self.assertTrue(any("position" in v['message'].lower() for v in violations))

    def test_calculate_portfolio_var(self):
        """Test VaR calculation"""
        # Add some positions
        self.risk_manager.update_position("TOKEN1", 100.0, 0.5, "BUY", "Crypto")
        self.risk_manager.update_position("TOKEN2", 100.0, 0.5, "BUY", "Politics")

        current_prices = {"TOKEN1": 0.5, "TOKEN2": 0.5}

        var_95 = self.risk_manager.calculate_portfolio_var(current_prices, confidence=0.95)
        # With these simple positions, VaR should be calculable
        self.assertIsInstance(var_95, (int, float))
        self.assertGreaterEqual(var_95, 0)  # VaR should be non-negative

    def test_correlation_matrix_single_position(self):
        """Test correlation matrix with single position"""
        self.risk_manager.update_position("TOKEN1", 100.0, 0.5, "BUY", "Crypto")

        # Add some price history for correlation calculation
        self.risk_manager.update_price_history("TOKEN1", 0.5)
        self.risk_manager.update_price_history("TOKEN1", 0.52)

        corr_matrix, tokens = self.risk_manager.calculate_correlation_matrix()

        # With only one position with price history, we might get empty arrays
        # depending on how much price history we have
        # The key thing is that it doesn't crash and returns appropriate types
        self.assertIsInstance(corr_matrix, np.ndarray)
        self.assertIsInstance(tokens, list)

    def test_get_high_correlation_pairs(self):
        """Test detection of high correlation pairs"""
        # Add two positions
        self.risk_manager.update_position("TOKEN1", 100.0, 0.5, "BUY", "Crypto")
        self.risk_manager.update_position("TOKEN2", 100.0, 0.5, "BUY", "Crypto")  # Same category

        # Add some price history to enable correlation calculation
        self.risk_manager.update_price_history("TOKEN1", 0.5)
        self.risk_manager.update_price_history("TOKEN1", 0.52)
        self.risk_manager.update_price_history("TOKEN2", 0.5)
        self.risk_manager.update_price_history("TOKEN2", 0.52)

        high_pairs = self.risk_manager.get_high_correlation_pairs(threshold=0.7)
        self.assertIsInstance(high_pairs, list)
        # Each pair should be (token1, token2, correlation)
        for pair in high_pairs:
            self.assertEqual(len(pair), 3)
            self.assertIsInstance(pair[0], str)
            self.assertIsInstance(pair[1], str)
            self.assertIsInstance(pair[2], float)


if __name__ == '__main__':
    unittest.main()