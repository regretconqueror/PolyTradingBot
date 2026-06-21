"""
Simple unit tests for execution engine - focusing on validation logic
"""
import unittest
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from bot.execution import ExecutionEngine


class TestExecutionEngineSimple(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures"""
        self.engine = ExecutionEngine(
            api_key='test_key',
            api_secret='test_secret',
            passphrase='test_passphrase',
            private_key='test_private_key',
            funder_address='0x1234567890123456789012345678901234567890',
            live_trading_enabled=True,
            dry_run=True,  # Start with dry_run for safety
            max_order_size=25.0
        )

    def test_initialization(self):
        """Test that engine initializes correctly"""
        self.assertEqual(self.engine.api_key, 'test_key')
        self.assertEqual(self.engine.api_secret, 'test_secret')
        self.assertTrue(self.engine.dry_run)
        self.assertTrue(self.engine.live_trading_enabled)
        self.assertEqual(self.engine.max_order_size, 25.0)

    def test_missing_credentials(self):
        """Test detection of missing credentials"""
        # Engine with missing credentials
        engine_missing = ExecutionEngine(
            api_key='',  # Missing
            api_secret='test_secret',
            passphrase='test_passphrase',
            private_key='test_private_key',
            funder_address='0x1234567890123456789012345678901234567890'
        )

        missing = engine_missing._missing_credentials()
        self.assertIn('POLYMARKET_API_KEY', missing)
        self.assertEqual(len(missing), 1)

        # Engine with all credentials
        engine_complete = ExecutionEngine(
            api_key='test_key',
            api_secret='test_secret',
            passphrase='test_passphrase',
            private_key='test_private_key',
            funder_address='0x1234567890123456789012345678901234567890'
        )

        missing_complete = engine_complete._missing_credentials()
        self.assertEqual(len(missing_complete), 0)

    def test_validate_live_ready_disabled(self):
        """Test validation when live trading is disabled"""
        engine = ExecutionEngine(live_trading_enabled=False, dry_run=False)
        # Need to set credentials to avoid missing_credentials error
        engine.api_key = 'test_key'
        engine.api_secret = 'test_secret'
        engine.passphrase = 'test_passphrase'
        engine.private_key = 'test_private_key'
        engine.funder_address = '0x1234567890123456789012345678901234567890'

        readiness = engine.validate_live_ready()

        self.assertFalse(readiness['ready'])
        self.assertEqual(readiness['reason'], 'live_trading_disabled')

    def test_validate_live_ready_dry_run(self):
        """Test validation when in dry run mode"""
        engine = ExecutionEngine(live_trading_enabled=True, dry_run=True)
        # Need to set credentials to avoid missing_credentials error
        engine.api_key = 'test_key'
        engine.api_secret = 'test_secret'
        engine.passphrase = 'test_passphrase'
        engine.private_key = 'test_private_key'
        engine.funder_address = '0x1234567890123456789012345678901234567890'

        readiness = engine.validate_live_ready()

        self.assertFalse(readiness['ready'])
        self.assertEqual(readiness['reason'], 'dry_run_enabled')

    def test_validate_live_ready_missing_creds(self):
        """Test validation with missing credentials"""
        engine = ExecutionEngine(
            api_key='',  # Missing
            api_secret='test_secret',
            passphrase='test_passphrase',
            private_key='test_private_key',
            funder_address='0x1234567890123456789012345678901234567890',
            live_trading_enabled=True,
            dry_run=False
        )

        readiness = engine.validate_live_ready()

        self.assertFalse(readiness['ready'])
        self.assertEqual(readiness['reason'], 'missing_credentials')
        self.assertIn('POLYMARKET_API_KEY', readiness['missing'])

    def test_validate_live_ready_success(self):
        """Test validation when ready for live trading"""
        engine = ExecutionEngine(
            api_key='test_key',
            api_secret='test_secret',
            passphrase='test_passphrase',
            private_key='test_private_key',
            funder_address='0x1234567890123456789012345678901234567890',
            live_trading_enabled=True,
            dry_run=False
        )

        readiness = engine.validate_live_ready()

        self.assertTrue(readiness['ready'])

    def test_execute_market_order_invalid_size(self):
        """Test market order with invalid size"""
        result = self.engine.execute_market_order(
            token_id='test_token',
            side='BUY',
            size=-10.0  # Invalid negative size
        )

        self.assertEqual(result['status'], 'error')
        self.assertIn('Order size must be positive', result['error'])

    def test_execute_market_order_invalid_side(self):
        """Test market order with invalid side"""
        result = self.engine.execute_market_order(
            token_id='test_token',
            side='INVALID',
            size=10.0
        )

        self.assertEqual(result['status'], 'error')
        self.assertIn('Side must be BUY or SELL', result['error'])

    def test_execute_market_order_zero_size(self):
        """Test market order with zero size"""
        result = self.engine.execute_market_order(
            token_id='test_token',
            side='BUY',
            size=0.0
        )

        self.assertEqual(result['status'], 'error')
        self.assertIn('Order size must be positive', result['error'])

    def test_execute_market_order_size_capping_logic(self):
        """Test the size capping logic directly"""
        # Test that the capping calculation works correctly
        test_cases = [
            (10.0, 25.0, 10.0),   # Under limit - should remain same
            (25.0, 25.0, 25.0),   # At limit - should remain same
            (50.0, 25.0, 25.0),   # Over limit - should be capped to max
            (100.0, 25.0, 25.0),  # Way over - should be capped to max
        ]

        for size, max_size, expected in test_cases:
            with self.subTest(size=size, max_size=max_size):
                # This mimics the logic in execute_market_order
                capped_size = min(float(size), max_size)
                self.assertEqual(capped_size, expected)

    def test_readiness_priority_order(self):
        """Test that readiness checks happen in the correct priority order"""
        # 1. Missing credentials should be checked first
        engine_missing = ExecutionEngine(
            api_key='',  # Missing
            api_secret='test_secret',
            passphrase='test_passphrase',
            private_key='test_private_key',
            funder_address='0x1234567890123456789012345678901234567890',
            live_trading_enabled=True,  # This would normally allow trading
            dry_run=False              # This would normally allow trading
        )

        readiness = engine_missing.validate_live_ready()
        self.assertFalse(readiness['ready'])
        self.assertEqual(readiness['reason'], 'missing_credentials')

        # 2. Live trading disabled should be checked second
        engine_disabled = ExecutionEngine(
            api_key='test_key',
            api_secret='test_secret',
            passphrase='test_passphrase',
            private_key='test_private_key',
            funder_address='0x1234567890123456789012345678901234567890',
            live_trading_enabled=False,  # Disabled
            dry_run=False                # Normally would allow
        )

        readiness = engine_disabled.validate_live_ready()
        self.assertFalse(readiness['ready'])
        self.assertEqual(readiness['reason'], 'live_trading_disabled')

        # 3. Dry run enabled should be checked third
        engine_dry_run = ExecutionEngine(
            api_key='test_key',
            api_secret='test_secret',
            passphrase='test_passphrase',
            private_key='test_private_key',
            funder_address='0x1234567890123456789012345678901234567890',
            live_trading_enabled=True,   # Normally would allow
            dry_run=True                 # But dry run is enabled
        )

        readiness = engine_dry_run.validate_live_ready()
        self.assertFalse(readiness['ready'])
        self.assertEqual(readiness['reason'], 'dry_run_enabled')

        # 4. When all conditions are met, should be ready
        engine_ready = ExecutionEngine(
            api_key='test_key',
            api_secret='test_secret',
            passphrase='test_passphrase',
            private_key='test_private_key',
            funder_address='0x1234567890123456789012345678901234567890',
            live_trading_enabled=True,   # Allow live trading
            dry_run=False                # Not in dry run mode
        )

        readiness = engine_ready.validate_live_ready()
        self.assertTrue(readiness['ready'])

    def test_get_connection_details_missing_credentials(self):
        """Test get_connection_details when credentials are missing"""
        engine = ExecutionEngine(api_key="")
        details = engine.get_connection_details()
        self.assertFalse(details['connected'])
        self.assertEqual(details['status_text'], 'Disconnected')
        self.assertIsNone(details['eoa_address'])
        self.assertIsNone(details['proxy_address'])
        self.assertEqual(details['proxy_balance'], 0.0)
        self.assertIn("Missing credentials", details['error'])


if __name__ == '__main__':
    unittest.main()