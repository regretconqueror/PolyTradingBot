"""
Unit tests for execution engine
"""
import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from bot.execution import ExecutionEngine


class TestExecutionEngine(unittest.TestCase):

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

    @patch('bot.execution.ExecutionEngine.validate_live_ready')
    def test_execute_market_order_not_ready_dry_run_true(self, mock_validate):
        """Test market order when not ready for live trading with dry_run=True"""
        mock_validate.return_value = {
            'ready': False,
            'reason': 'live_trading_disabled',
            'message': 'Set LIVE_TRADING_ENABLED=true to allow real submissions.'
        }

        # Need to set credentials for the engine to get past missing credentials check
        self.engine.api_key = 'test_key'
        self.engine.api_secret = 'test_secret'
        self.engine.passphrase = 'test_passphrase'
        self.engine.private_key = 'test_private_key'
        self.engine.funder_address = '0x1234567890123456789012345678901234567890'
        # dry_run remains True from setUp

        result = self.engine.execute_market_order(
            token_id='test_token',
            side='BUY',
            size=10.0
        )

        # When dry_run=True, not ready returns 'dry_run' status
        self.assertEqual(result['status'], 'dry_run')
        self.assertEqual(result['readiness']['reason'], 'live_trading_disabled')

    @patch('bot.execution.ExecutionEngine.validate_live_ready')
    def test_execute_market_order_not_ready_dry_run_false(self, mock_validate):
        """Test market order when not ready for live trading with dry_run=False"""
        mock_validate.return_value = {
            'ready': False,
            'reason': 'live_trading_disabled',
            'message': 'Set LIVE_TRADING_ENABLED=true to allow real submissions.'
        }

        # Need to set credentials for the engine to get past missing credentials check
        self.engine.api_key = 'test_key'
        self.engine.api_secret = 'test_secret'
        self.engine.passphrase = 'test_passphrase'
        self.engine.private_key = 'test_private_key'
        self.engine.funder_address = '0x1234567890123456789012345678901234567890'
        self.engine.dry_run = False  # Change to False for this test
        self.engine.live_trading_enabled = True  # Enable live trading

        result = self.engine.execute_market_order(
            token_id='test_token',
            side='BUY',
            size=10.0
        )

        # When dry_run=False, not ready returns 'blocked' status
        self.assertEqual(result['status'], 'blocked')
        self.assertEqual(result['readiness']['reason'], 'live_trading_disabled')

    @patch('bot.execution.ExecutionEngine.validate_live_ready')
    @patch('bot.execution.ExecutionEngine._get_client')
    def test_execute_market_order_success(self, mock_get_client, mock_validate):
        """Test successful market order execution"""
        # Mock validation to return ready
        mock_validate.return_value = {'ready': True}

        # Mock client and response
        mock_client = Mock()
        mock_client.create_and_post_market_order.return_value = {
            'success': True,
            'orderID': 'test_order_123',
            'status': 'filled'
        }
        mock_get_client.return_value = mock_client

        # Need to set credentials for the engine to get past missing credentials check
        self.engine.api_key = 'test_key'
        self.engine.api_secret = 'test_secret'
        self.engine.passphrase = 'test_passphrase'
        self.engine.private_key = 'test_private_key'
        self.engine.funder_address = '0x1234567890123456789012345678901234567890'
        # Set dry_run=False for live execution test
        self.engine.dry_run = False

        # Mock the imports inside the method by patching where they are used
        with patch('py_clob_client_v2.clob_types.MarketOrderArgsV2') as mock_market_order_args_v2, \
             patch('py_clob_client_v2.clob_types.OrderType') as mock_order_type:

            mock_market_order_args_v2.return_value = Mock()
            mock_order_type.FOK = 'FOK'

            result = self.engine.execute_market_order(
                token_id='test_token',
                side='BUY',
                size=10.0,
                price=0.6
            )

            self.assertEqual(result['status'], 'success')
            self.assertEqual(result['order_id'], 'test_order_123')
            self.assertEqual(result['side'], 'BUY')
            self.assertEqual(result['size'], 10.0)
            mock_client.create_and_post_market_order.assert_called_once()

    @patch('bot.execution.ExecutionEngine.validate_live_ready')
    @patch('bot.execution.ExecutionEngine._get_client')
    def test_execute_market_order_size_capping(self, mock_get_client, mock_validate):
        """Test that order sizes are capped at max_order_size"""
        mock_validate.return_value = {'ready': True}

        # Mock client
        mock_client = Mock()
        mock_client.create_and_post_market_order.return_value = {
            'success': True,
            'orderID': 'test_order_123',
            'status': 'filled'
        }
        mock_get_client.return_value = mock_client

        # Need to set credentials for the engine to get past missing credentials check
        self.engine.api_key = 'test_key'
        self.engine.api_secret = 'test_secret'
        self.engine.passphrase = 'test_passphrase'
        self.engine.private_key = 'test_private_key'
        self.engine.funder_address = '0x1234567890123456789012345678901234567890'
        # Set dry_run=False for live execution test
        self.engine.dry_run = False

        # Mock the imports inside the method by patching where they are used
        with patch('py_clob_client_v2.clob_types.MarketOrderArgsV2') as mock_market_order_args_v2, \
             patch('py_clob_client_v2.clob_types.OrderType') as mock_order_type:

            mock_market_order_args_v2.return_value = Mock()
            mock_order_type.FOK = 'FOK'

            # Try to order more than max_order_size (25.0)
            result = self.engine.execute_market_order(
                token_id='test_token',
                side='BUY',
                size=50.0  # Should be capped to 25.0
            )

            self.assertEqual(result['status'], 'success')
            self.assertEqual(result['size'], 25.0)  # Should be capped

            # Verify MarketOrderArgsV2 was called with the capped size
            mock_market_order_args_v2.assert_called_once()
            call_kwargs = mock_market_order_args_v2.call_args
            self.assertEqual(call_kwargs[1].get('amount', call_kwargs[0][0] if call_kwargs[0] else None), 25.0)

    @patch('bot.execution.ExecutionEngine.validate_live_ready')
    @patch('bot.execution.ExecutionEngine._get_client')
    def test_execute_market_order_dry_run(self, mock_get_client, mock_validate):
        """Test market order in dry run mode"""
        mock_validate.return_value = {
            'ready': False,
            'reason': 'dry_run_enabled',
            'message': 'Set LIVE_DRY_RUN=false to submit real orders.'
        }

        # Need to set credentials to get past missing credentials check
        self.engine.api_key = 'test_key'
        self.engine.api_secret = 'test_secret'
        self.engine.passphrase = 'test_passphrase'
        self.engine.private_key = 'test_private_key'
        self.engine.funder_address = '0x1234567890123456789012345678901234567890'
        # Keep dry_run=True (default from setUp)

        result = self.engine.execute_market_order(
            token_id='test_token',
            side='BUY',
            size=10.0
        )

        self.assertEqual(result['status'], 'dry_run')
        self.assertEqual(result['readiness']['reason'], 'dry_run_enabled')


if __name__ == '__main__':
    unittest.main()