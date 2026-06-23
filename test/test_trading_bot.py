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

    def test_update_performance_metrics_biggest_wins(self):
        """Test calculation of history and today biggest wins"""
        import datetime
        # Setup mock trade history
        today_str = datetime.datetime.now().date().isoformat()
        yesterday_str = (datetime.datetime.now() - datetime.timedelta(days=1)).date().isoformat()
        
        self.bot.trade_history = [
            {
                'status': TradeStatus.EXITED.value,
                'realized_pnl': 50.0,  # All-time biggest win
                'closed_at': f"{yesterday_str}T12:00:00",
                'type': 'regular'
            },
            {
                'status': TradeStatus.EXITED.value,
                'realized_pnl': 20.0,
                'closed_at': f"{today_str}T08:00:00",
                'type': 'regular'
            },
            {
                'status': TradeStatus.EXITED.value,
                'realized_pnl': 35.0,  # Today's biggest win
                'closed_at': f"{today_str}T14:00:00",
                'type': 'regular'
            },
            {
                'status': TradeStatus.EXITED.value,
                'realized_pnl': -15.0,  # Loss - shouldn't count towards wins
                'closed_at': f"{today_str}T16:00:00",
                'type': 'regular'
            }
        ]

        self.bot.update_performance_metrics()

        # Check all-time and today biggest wins
        self.assertEqual(self.bot.performance_metrics['biggest_win_history'], 50.0)
        self.assertEqual(self.bot.performance_metrics['biggest_win_today'], 35.0)

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

    def test_whale_tracker_model(self):
        """Test WhaleTrackerModel probability adjustments"""
        from strategies.example_strategy import WhaleTrackerModel
        
        # Test case 1: NO whale positions -> returns current price (no change)
        model = WhaleTrackerModel(whale_wallets=["0xWhale1", "0xWhale2"])
        market = {"token_id": "TOKEN1", "current_price": 0.5, "conditionId": "COND1", "clobTokenIds": ["TOKEN1", "TOKEN2"]}
        prob = model.estimate_probability(market)
        self.assertEqual(prob, 0.5)

        # Test case 2: Whales holding YES positions -> increases YES probability
        fallback = {
            "TOKEN1": {
                "0xWhale1": {"size": 1000.0, "outcome": "YES"}
            }
        }
        model_with_whale = WhaleTrackerModel(whale_wallets=["0xWhale1"], fallback_positions=fallback, impact_factor=0.05)
        prob = model_with_whale.estimate_probability(market)
        # net_skew = 1.0, impact = 0.05 -> adjusted prob = 0.5 + 0.05 = 0.55
        self.assertAlmostEqual(prob, 0.55)

        # Test case 3: Whales holding NO positions -> decreases YES probability
        fallback_no = {
            "TOKEN1": {
                "0xWhale1": {"size": 1000.0, "outcome": "NO"}
            }
        }
        model_with_whale_no = WhaleTrackerModel(whale_wallets=["0xWhale1"], fallback_positions=fallback_no, impact_factor=0.05)
        prob = model_with_whale_no.estimate_probability(market)
        # net_skew = -1.0, impact = 0.05 -> adjusted prob = 0.5 - 0.05 = 0.45
        self.assertAlmostEqual(prob, 0.45)

    def test_manage_open_limit_orders_filled(self):
        """Test active limit order lifecycle manager marking orders as filled"""
        self.bot.use_limit_orders = True
        token_id = "TOKEN_TEST"
        order_id = "dry_test_123"
        
        # Mock API and trades
        self.bot.trade_history = [
            {
                "order_id": order_id,
                "token_id": token_id,
                "status": "dry_run",
                "execution_price": 0.5,
                "size": 1000.0,
                "direction": "BUY YES",
                "category": "Crypto"
            }
        ]
        
        self.bot.limit_quoter._open_quotes[token_id] = {
            "order_id": order_id,
            "side": "BUY",
            "size": 1000.0,
            "price": 0.5,
            "timestamp": 123456
        }
        
        # Mock api.get_price to return 0.45 (which is below limit_price 0.5 for BUY)
        with patch.object(self.bot.api, 'get_price', return_value=0.45):
            self.bot.manage_open_limit_orders()
            
        # Order should be filled
        self.assertEqual(self.bot.trade_history[0]["status"], TradeStatus.FILLED.value)
        self.assertNotIn(token_id, self.bot.limit_quoter.open_quotes)
        # Position should be added to risk manager
        self.assertIn(token_id, self.bot.risk_manager.positions)
        self.assertEqual(self.bot.risk_manager.positions[token_id]["size"], 1000.0)

    def test_manage_open_limit_orders_requote(self):
        """Test active limit order lifecycle manager requoting when price moves"""
        self.bot.use_limit_orders = True
        token_id = "TOKEN_TEST"
        order_id = "dry_test_123"
        
        # Mock API and trades
        self.bot.trade_history = [
            {
                "order_id": order_id,
                "token_id": token_id,
                "status": "dry_run",
                "execution_price": 0.5,
                "size": 1000.0,
                "direction": "BUY YES",
                "category": "Crypto"
            }
        ]
        
        self.bot.limit_quoter._open_quotes[token_id] = {
            "order_id": order_id,
            "side": "BUY",
            "size": 1000.0,
            "price": 0.5,
            "timestamp": 123456
        }
        
        # Mock api.get_price to return 0.55 (not filled for BUY at 0.5)
        # Mock limit_quoter.requote to return a new requoted order
        mock_requote_res = {
            "status": "dry_run",
            "order_id": "dry_test_456",
            "price": 0.53
        }
        
        with patch.object(self.bot.api, 'get_price', return_value=0.55):
            with patch.object(self.bot.limit_quoter, 'requote', return_value=mock_requote_res):
                self.bot.manage_open_limit_orders()
                
        # Trade history entry should be updated with new order ID and new execution price
        self.assertEqual(self.bot.trade_history[0]["order_id"], "dry_test_456")
        self.assertEqual(self.bot.trade_history[0]["execution_price"], 0.53)

    def test_yes_no_arb_depth_scanning(self):
        """Test yes/no arbitrage scanner's multi-level book depth walk"""
        from strategies.yes_no_arb import YesNoArbScanner
        
        scanner = YesNoArbScanner(fee_buffer=0.01, max_per_market=0.1, min_edge=0.01)
        
        # Mock order books with multiple levels
        # Buying 1000 shares
        orderbook_yes = {
            "asks": [
                {"price": 0.40, "size": 200.0},  # First level
                {"price": 0.42, "size": 800.0}   # Second level
            ]
        }
        orderbook_no = {
            "asks": [
                {"price": 0.50, "size": 500.0},  # First level
                {"price": 0.52, "size": 500.0}   # Second level
            ]
        }
        
        # Walk YES book for 1000 shares:
        # 200 * 0.40 = 80
        # 800 * 0.42 = 336
        # Total cost = 416 -> average price = 0.416
        yes_avg, yes_filled = scanner._walk_book(orderbook_yes["asks"], 1000.0)
        self.assertAlmostEqual(yes_avg, 0.416)
        self.assertEqual(yes_filled, 1000.0)
        
        # Walk NO book for 1000 shares:
        # 500 * 0.50 = 250
        # 500 * 0.52 = 260
        # Total cost = 510 -> average price = 0.51
        no_avg, no_filled = scanner._walk_book(orderbook_no["asks"], 1000.0)
        self.assertAlmostEqual(no_avg, 0.51)
        self.assertEqual(no_filled, 1000.0)
        
        # Test full scan with depth
        market = {"conditionId": "COND1", "question": "Test Arb question", "clobTokenIds": ["YES_T", "NO_T"]}
        signal = scanner.scan(market, orderbook_yes, orderbook_no, capital=10000.0)
        
        self.assertIsNotNone(signal)
        # Average sum price = 0.416 + 0.51 = 0.926
        # Edge = 1.0 - 0.926 - 0.01 (fee) = 0.064 (6.4%)
        self.assertAlmostEqual(signal.sum_price, 0.926)
        self.assertAlmostEqual(signal.edge, 0.064)

    def test_risk_limit_exit_signals(self):
        """Test that RISK_LIMIT exit signals are generated when a category or position limit is violated"""
        from bot.trading_bot import ExitReason
        token_id = "TOKEN_BREACH"
        
        # Setup trade history to provide entry info
        self.bot.trade_history = [{
            'status': 'filled',
            'token_id': token_id,
            'execution_price': 0.50,
            'filled_at': datetime.now().isoformat(),
            'side': 'BUY'
        }]
        
        # Setup position in risk manager
        self.bot.risk_manager.positions[token_id] = {
            'size': 1000.0,
            'entry_price': 0.50,
            'side': 'BUY',
            'category': 'Crypto',
            'timestamp': datetime.now(),
            'unrealized_pnl': 0.0,
            'realized_pnl': 0.0
        }
        
        # Setup category breach violation
        risk_violations = [{
            'type': 'category_exposure',
            'category': 'Crypto',
            'current': 0.45,
            'limit': 0.30,
            'message': 'Category Crypto exposure 45.0% exceeds limit 30.0%'
        }]
        
        signals = self.bot.evaluate_exit_signals(
            token_id=token_id,
            current_price=0.52,
            estimated_prob=0.51,
            market={"category": "Crypto", "price": 0.52},
            risk_violations=risk_violations
        )
        
        self.assertTrue(len(signals) > 0)
        self.assertEqual(signals[0].reason, ExitReason.RISK_LIMIT)
        self.assertEqual(signals[0].token_id, token_id)


if __name__ == '__main__':
    unittest.main()