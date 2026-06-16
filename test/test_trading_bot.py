"""
Unit tests for trading bot
"""
import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
import os
from datetime import datetime

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from bot.trading_bot import PolymarketTradingBot
from core.models import Market, PortfolioConstraints, TradeStatus
from core import ProjectFWOptimizer


class TestTradingBot(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures"""
        self.constraints = PortfolioConstraints(
            max_total_exposure=0.75,
            max_single_position=0.20
        )
        self.bot = PolymarketTradingBot(
            capital=10000.0,
            constraints=self.constraints,
            paper_mode=True
        )

    def test_initialization(self):
        """Test that bot initializes correctly"""
        self.assertEqual(self.bot.capital, 10000.0)
        self.assertEqual(self.bot.constraints.max_total_exposure, 0.75)
        self.assertTrue(self.bot.paper_mode)
        self.assertIsInstance(self.bot.optimizer, ProjectFWOptimizer)
        self.assertEqual(len(self.bot.trade_history), 0)
        self.assertEqual(self.bot.performance_metrics['total_trades'], 0)

    def test_generate_orders(self):
        """Test order generation from allocations"""
        # Create some test markets
        markets = [
            Market("1", "Test Market 1", "TOKEN1", 0.6, 0.7, 100000, 50000, "Crypto", "", outcome="YES"),
            Market("2", "Test Market 2", "TOKEN2", 0.4, 0.3, 100000, 50000, "Politics", "", outcome="NO"),
        ]

        # Test allocations (50% in each)
        allocations = [0.5, 0.5]

        orders = self.bot.generate_orders(markets, allocations)

        self.assertEqual(len(orders), 2)
        # First market: probability 0.7 > price 0.6 -> positive edge -> BUY YES
        self.assertEqual(orders[0]['direction'], 'BUY YES')
        # Due to slippage guard, order is reduced from $5000 to $2250
        self.assertEqual(orders[0]['size'], 2250.0)
        self.assertAlmostEqual(orders[0]['allocation'], 0.225)
        # Second market: probability 0.3 < price 0.4 -> negative edge -> BUY NO (since we always buy)
        self.assertEqual(orders[1]['direction'], 'BUY NO')
        # Due to slippage guard, order is reduced from $5000 to $2250
        self.assertEqual(orders[1]['size'], 2250.0)
        self.assertAlmostEqual(orders[1]['allocation'], 0.225)

    def test_generate_orders_zero_allocation(self):
        """Test order generation with zero allocations"""
        markets = [
            Market("1", "Test Market 1", "TOKEN1", 0.6, 0.7, 100000, 50000, "Crypto", ""),
        ]

        # Zero allocation
        allocations = [0.0]

        orders = self.bot.generate_orders(markets, allocations)

        self.assertEqual(len(orders), 0)  # No orders should be generated

    def test_generate_orders_small_allocation(self):
        """Test order generation with very small allocations"""
        markets = [
            Market("1", "Test Market 1", "TOKEN1", 0.6, 0.7, 100000, 50000, "Crypto", ""),
        ]

        # Very small allocation (below 0.001 threshold)
        allocations = [0.0005]

        orders = self.bot.generate_orders(markets, allocations)

        self.assertEqual(len(orders), 0)  # No orders should be generated

    def test_update_performance_metrics_no_trades(self):
        """Test performance metrics update with no trades"""
        # Should not crash and should not change metrics
        initial_metrics = self.bot.performance_metrics.copy()
        self.bot.update_performance_metrics()
        # Metrics should remain essentially unchanged (last_updated might change)
        self.assertEqual(self.bot.performance_metrics['total_trades'], 0)
        self.assertEqual(self.bot.performance_metrics['winning_trades'], 0)
        self.assertEqual(self.bot.performance_metrics['losing_trades'], 0)

    def test_update_performance_metrics_with_trades(self):
        """Test performance metrics update with trades"""
        # Add some mock trades to history
        self.bot.trade_history = [
            {
                'status': TradeStatus.EXITED.value,
                'realized_pnl': 10.0,  # $10 profit
                'type': 'regular'
            },
            {
                'status': TradeStatus.EXITED.value,
                'realized_pnl': -10.0,  # $10 loss
                'type': 'regular'
            },
            {
                'status': TradeStatus.FILLED.value,  # Open trade - expected P&L only
                'filled_value': 50.0,
                'edge': 0.2,  # 10% edge -> $10 expected profit
                'type': 'regular'
            }
        ]
        # Add an unfilled trade
        self.bot.trade_history.append({
            'status': 'open',  # Not filled
            'filled_value': 50.0,
            'edge': 0.2,
            'type': 'regular'
        })

        self.bot.update_performance_metrics()

        # Should have 2 completed trades (the exited ones)
        self.assertEqual(self.bot.performance_metrics['total_trades'], 2)
        self.assertEqual(self.bot.performance_metrics['winning_trades'], 1)  # First trade
        self.assertEqual(self.bot.performance_metrics['losing_trades'], 1)  # Second trade
        # P&L: $10 profit + $10 loss = $0
        self.assertEqual(self.bot.performance_metrics['total_pnl'], 0.0)
        self.assertEqual(self.bot.performance_metrics['win_rate'], 0.5)  # 1/2

    def test_update_positions_from_trades_empty(self):
        """Test position updates with empty trade history"""
        # Should not crash
        self.bot.update_positions_from_trades()
        self.assertEqual(len(self.bot.positions), 0)

    def test_update_positions_from_trades(self):
        """Test position updates from trade history"""
        # Add some mock trades
        self.bot.trade_history = [
            {
                'status': 'filled',
                'token_id': 'TOKEN1',
                'filled_value': 1000.0,
                'execution_price': 0.5,
                'side': 'BUY',
                'size': 2000.0,  # $1000 at $0.5 = 2000 tokens
                'type': 'regular'
            },
            {
                'status': 'filled',
                'token_id': 'TOKEN2',
                'filled_value': 500.0,
                'execution_price': 0.25,
                'side': 'BUY',
                'size': 2000.0,  # $500 at $0.25 = 2000 tokens
                'type': 'regular'
            },
            {
                'status': 'filled',
                'token_id': 'TOKEN1',  # Same token, SELL
                'filled_value': 300.0,
                'execution_price': 0.6,
                'side': 'SELL',
                'size': 500.0,  # $300 at $0.6 = 500 tokens
                'type': 'regular'
            }
        ]

        self.bot.update_positions_from_trades()

        # TOKEN1: +2000 (BUY) - 500 (SELL) = 1500 tokens
        self.assertAlmostEqual(self.bot.positions['TOKEN1'], 1500.0)
        # TOKEN2: +2000 tokens
        self.assertAlmostEqual(self.bot.positions['TOKEN2'], 2000.0)

        # Risk manager should also have the positions
        self.assertIn('TOKEN1', self.bot.risk_manager.positions)
        self.assertIn('TOKEN2', self.bot.risk_manager.positions)

    def test_scan_yes_no_arbitrage_disabled(self):
        """Test arbitrage scanning when disabled"""
        bot_no_arb = PolymarketTradingBot(
            capital=10000.0,
            constraints=self.constraints,
            paper_mode=True,
            enable_yes_no_arb=False  # Disable arbitrage
        )

        orders = bot_no_arb.scan_yes_no_arbitrage()
        self.assertEqual(len(orders), 0)

    def test_get_performance_summary(self):
        """Test getting performance summary"""
        summary = self.bot.get_performance_summary()
        self.assertIsInstance(summary, dict)
        self.assertIn('Performance Summary', summary)
        self.assertIsInstance(summary['Performance Summary'], str)
        # Should contain key metrics
        self.assertIn('Total Trades', summary['Performance Summary'])
        self.assertIn('Win Rate', summary['Performance Summary'])

    def test_print_performance_report(self):
        """Test printing performance report (should not crash)"""
        # This just prints to stdout, so we mainly test it doesn't crash
        try:
            self.bot.print_performance_report()
            success = True
        except Exception as e:
            success = False
            print(f"Error: {e}")
        self.assertTrue(success)


if __name__ == '__main__':
    unittest.main()