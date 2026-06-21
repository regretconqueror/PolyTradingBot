"""
Main trading bot implementation
"""
import json
import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta, timezone
import logging
from enum import Enum

from core import ProjectFWOptimizer, Market, PortfolioConstraints, OptimizationStatus, TradeStatus
from data import PolymarketAPI
from strategies import ProbabilityModel, SimpleEdgeModel, YesNoArbScanner, WeightedMovingAverageModel, VolatilityAdjustedModel, EnsembleModel
from core.risk_manager import RiskManager
from bot.execution import ExecutionEngine
from bot.alert_manager import AlertManager
from bot.limit_quoter import LimitQuoter

logger = logging.getLogger(__name__)


class ExitReason(Enum):
    """Reasons why an exit may be triggered"""
    TAKE_PROFIT = "take_profit"        # Price moved in our favor
    STOP_LOSS = "stop_loss"            # Price moved against us
    TIME_EXIT = "time_exit"            # Market resolving soon, exit before close
    MOMENTUM_REVERSAL = "momentum_reversal"  # Momentum signal flipped
    STALE_POSITION = "stale_position"  # Position held too long without edge
    RISK_LIMIT = "risk_limit"          # Portfolio risk limit breached


class ExitSignal:
    """Represents an exit signal for a position"""
    def __init__(self, token_id: str, reason: ExitReason,
                 current_price: float, entry_price: float,
                 estimated_prob: float, market_price: float,
                 unrealized_pnl: float, hold_duration_hours: float,
                 confidence: float = 1.0):
        self.token_id = token_id
        self.reason = reason
        self.current_price = current_price
        self.entry_price = entry_price
        self.estimated_prob = estimated_prob
        self.market_price = market_price
        self.unrealized_pnl = unrealized_pnl
        self.hold_duration_hours = hold_duration_hours
        self.confidence = confidence  # How strongly the signal fires (0-1)

    def to_dict(self) -> Dict:
        return {
            "token_id": self.token_id,
            "reason": self.reason.value,
            "current_price": self.current_price,
            "entry_price": self.entry_price,
            "estimated_prob": self.estimated_prob,
            "market_price": self.market_price,
            "unrealized_pnl": self.unrealized_pnl,
            "hold_hours": round(self.hold_duration_hours, 1),
            "confidence": round(self.confidence, 2)
        }

class PolymarketTradingBot:
    """
    Complete trading bot using ProjectFW + Kelly Criterion
    """
    def __init__(self,
                 capital: float = 10000.0,
                 constraints: Optional[PortfolioConstraints] = None,
                 model: Optional[ProbabilityModel] = None,
                 api_key: Optional[str] = None,
                 api_secret: Optional[str] = None,
                 passphrase: Optional[str] = None,
                 private_key: Optional[str] = None,
                 funder_address: Optional[str] = None,
                 signature_type: int = 3,
                 live_trading_enabled: bool = False,
                 live_dry_run: bool = True,
                 max_live_order_size: float = 25.0,
                 max_live_orders_per_cycle: int = 3,
                 paper_mode: bool = True,
                 enable_yes_no_arb: bool = True,
                 arb_fee_buffer: float = 0.02,
                 arb_max_per_market: float = 0.05,
                 slippage_tolerance: float = 0.015,
                 use_limit_orders: bool = False,
                 quote_aggressiveness: float = 0.3):

        self.capital = capital
        self.constraints = constraints or PortfolioConstraints()
        self.model = model or EnsembleModel()
        self.optimizer = ProjectFWOptimizer()
        self.api = PolymarketAPI()
        self.execution_engine = ExecutionEngine(
            api_key=api_key,
            api_secret=api_secret,
            passphrase=passphrase,
            private_key=private_key,
            funder_address=funder_address,
            signature_type=signature_type,
            live_trading_enabled=live_trading_enabled,
            dry_run=live_dry_run,
            max_order_size=max_live_order_size,
        )

        self.paper_mode = paper_mode
        self.api_key = api_key
        self.api_secret = api_secret
        self.passphrase = passphrase
        self.private_key = private_key
        self.funder_address = funder_address
        self.signature_type = signature_type
        self.live_trading_enabled = live_trading_enabled
        self.live_dry_run = live_dry_run
        self.max_live_orders_per_cycle = max_live_orders_per_cycle
        self.slippage_tolerance = slippage_tolerance
        self.use_limit_orders = use_limit_orders
        self.quote_aggressiveness = quote_aggressiveness

        # Limit order quoter (active when use_limit_orders=True)
        self.limit_quoter = LimitQuoter(
            execution_engine=self.execution_engine,
            aggressiveness=quote_aggressiveness,
            dry_run=live_dry_run or not live_trading_enabled,
        )

        # Risk monitoring thresholds
        self.drawdown_alert_threshold = 0.10  # Alert at 10% drawdown
        self.var_alert_threshold = 0.05       # Alert at 5% VaR
        self.correlation_alert_threshold = 0.8 # Alert at 80% correlation

        # Yes/No arbitrage
        self.enable_yes_no_arb = enable_yes_no_arb
        self.arb_scanner = YesNoArbScanner(
            fee_buffer=arb_fee_buffer,
            max_per_market=arb_max_per_market,
            min_edge=0.005  # 0.5% minimum edge for arbitrage
        )

        self.trade_history = []
        self.performance_log = []
        self.open_orders = {}  # Track open orders for management
        self.performance_metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0,
            'total_pnl_percent': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'win_rate': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'profit_factor': 0.0,
            'biggest_win_history': 0.0,
            'biggest_win_today': 0.0,
            'last_updated': None
        }

        # Initialize risk manager
        self.risk_manager = RiskManager(self.constraints)

        # Initialize position tracking
        self.positions = {}  # token_id -> dollar amount currently invested

        # Rebalancing threshold (as fraction of capital)
        self.rebalance_threshold = 0.05  # 5% of capital

        # Performance tracking
        self.start_time = datetime.now()
        self.cycles_completed = 0
        self.successful_cycles = 0
        self.failed_cycles = 0

        # Peak portfolio value for drawdown calculation
        self.peak_portfolio_value = 0.0

        # API failure tracking
        self.api_failures = 0
        self.max_api_failures = 5

        # Initialize alert manager
        self.alert_manager = AlertManager()

        # --- Exit Signal Configuration (Freqtrade-style exit framework) ---
        # Take-profit: exit when estimated probability moves this much toward 1.0
        self.take_profit_threshold = 0.08    # 8% probability drift in our favor
        self.stop_loss_threshold = 0.05     # 5% probability drift against us
        # Time-exit: exit when market resolves within this many hours (None = hold to resolution)
        self.time_exit_hours = None          # None = no time exit; set e.g. 2.0 for 2hrs before close
        # Stale position: exit if held longer than this without favorable drift
        self.stale_exit_hours = 168          # 7 days — force exit if no momentum
        # Momentum exit: minimum price drift per hour to stay in (chop filter)
        self.min_momentum_hold = 0.003       # 0.3% price drift per hour minimum
        # Only exit if confidence >= threshold
        self.min_exit_confidence = 0.6       # 60% minimum confidence to trigger exit

        # Track price history per token for momentum calculation
        self._position_price_history: Dict[str, List[Tuple[float, datetime]]] = {}

        # Periodic position poll (WebSocket drop safeguard)
        self._last_position_poll: Optional[datetime] = None
        self._position_poll_interval_secs = 30

        # Cache of raw markets keyed by token_id (populated in fetch_markets)
        self._token_market_cache: Dict[str, Dict] = {}

    def _update_performance_metrics(self):
        """Update performance metrics from trade history.

        Uses **realized** P&L for exited/settled trades and *expected* P&L
        (size * edge) only for still-open fills.  The `realized_pnl` and
        `expected_pnl` fields are reported separately so the user can see the
        distinction.
        """
        if not self.trade_history:
            return

        try:
            # -- Closed trades: realized P&L --------------------------------
            closed_trades = [t for t in self.trade_history
                              if t.get('status') in (TradeStatus.EXITED.value, TradeStatus.SETTLED.value)]
            realized_pnl = 0.0
            realized_wins = 0
            realized_losses = 0
            realized_pnls = []

            for trade in closed_trades:
                pnl = float(trade.get('realized_pnl', 0) or 0)
                realized_pnl += pnl
                realized_pnls.append(pnl)
                if pnl >= 0:
                    realized_wins += 1
                else:
                    realized_losses += 1

            # -- Open fills: expected P&L -----------------------------------
            open_fills = [t for t in self.trade_history
                          if t.get('status') == TradeStatus.FILLED.value]
            expected_pnl = 0.0
            for trade in open_fills:
                edge = float(trade.get('edge', 0) or 0)
                size = float(trade.get('filled_value', trade.get('size', 0)) or 0)
                if trade.get('type') == 'YES_NO_ARB':
                    expected_pnl += size * edge
                else:
                    expected_pnl += size * edge

            # -- Combined totals (realized P&L only for trade counts) -----------
            total_pnl = realized_pnl  # Only realized P&L counts for total_pnl metric
            total_trades = len(closed_trades)  # Only closed trades count as "trades"
            winning_trades = realized_wins  # open trades are not counted as wins/losses yet
            losing_trades = realized_losses
            win_rate = realized_wins / max(len(closed_trades), 1)

            all_pnls = realized_pnls[:]
            wins = [p for p in all_pnls if p >= 0]
            losses = [p for p in all_pnls if p < 0]
            avg_win = float(np.mean(wins)) if wins else 0.0
            avg_loss = float(np.mean(losses)) if losses else 0.0
            gross_profit = sum(wins)
            gross_loss = abs(sum(losses))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0.0

            # Calculate biggest wins
            wins = [p for p in realized_pnls if p > 0]
            biggest_win_history = max(wins) if wins else 0.0

            today_str = datetime.now().date().isoformat()
            today_wins = [float(t.get('realized_pnl', 0) or 0) for t in closed_trades
                          if t.get('closed_at', '').startswith(today_str) and float(t.get('realized_pnl', 0) or 0) > 0]
            biggest_win_today = max(today_wins) if today_wins else 0.0

            self.performance_metrics.update({
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'realized_pnl': realized_pnl,
                'expected_pnl': expected_pnl,
                'total_pnl': total_pnl,
                'total_pnl_percent': (total_pnl / self.capital) * 100 if self.capital > 0 else 0,
                'win_rate': win_rate,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_factor': profit_factor,
                'biggest_win_history': biggest_win_history,
                'biggest_win_today': biggest_win_today,
                'last_updated': datetime.now().isoformat()
            })

            # Sharpe ratio from realized P&L series
            returns = [p / self.capital for p in realized_pnls[-30:] if self.capital > 0]
            if len(returns) > 1:
                avg_return = np.mean(returns)
                std_return = np.std(returns)
                if std_return > 0:
                    self.performance_metrics['sharpe_ratio'] = (avg_return / std_return) * np.sqrt(252)

            self.performance_log.append({
                'timestamp': datetime.now().isoformat(),
                'realized_pnl': realized_pnl,
                'expected_pnl': expected_pnl,
                'total_pnl': total_pnl,
                'total_trades': total_trades,
                'win_rate': win_rate,
                'note': 'realized + expected_pnl'
            })

        except Exception as e:
            logger.error(f"Error updating performance metrics: {e}")

    def get_performance_summary(self) -> Dict:
        """Get current performance summary"""
        self._update_performance_metrics()
        metrics = self.performance_metrics.copy()
        metrics['Performance Summary'] = f"""
            Total Trades: {metrics['total_trades']}
            Win Rate: {metrics['win_rate']:.1%}
            Total P&L: ${metrics['total_pnl']:,.2f} ({metrics['total_pnl_percent']:.2f}%)
            Profit Factor: {metrics['profit_factor']:.2f}
            Sharpe Ratio: {metrics['sharpe_ratio']:.2f}
            Avg Win: ${metrics['avg_win']:.2f}
            Avg Loss: ${metrics['avg_loss']:.2f}
            Biggest Win (All-Time): ${metrics['biggest_win_history']:.2f}
            Biggest Win (Today): ${metrics['biggest_win_today']:.2f}
            Last Updated: {metrics['last_updated'] or 'Never'}
            """
        return metrics

    def get_api_connection_details(self) -> Dict:
        """Get Polymarket API connection status, wallet addresses, and balance."""
        return self.execution_engine.get_connection_details()



    def _simulate_order_execution(self, token_id: str, side: str, size_usd: float, market_price: float) -> Dict:
        """
        Simulate order execution with realistic slippage based on order book depth.

        Key: Polymarket order book sizes are in SHARES, not dollars.
        We convert our dollar budget → shares to walk the book correctly.
        Slippage is measured as absolute price difference (not % of a tiny price).

        Returns:
            Dict with execution details including slippage
        """
        try:
            # Get order book for realistic simulation
            orderbook = self.api.get_orderbook(token_id)

            # Shares we want to buy with our dollar budget
            target_shares = size_usd / market_price if market_price > 0 else 0

            if side.upper() == 'BUY':
                asks = orderbook.get('asks', [])
                if not asks:
                    slippage_percent = self.slippage_tolerance
                    execution_price = market_price * (1 + slippage_percent)
                else:
                    # Walk the ask book in SHARES
                    remaining_shares = target_shares
                    weighted_price_sum = 0

                    normalized_asks = []
                    for a in asks:
                        if isinstance(a, dict):
                            p = float(a.get('price', 0) or a.get('p', 0))
                            s = float(a.get('size', 0) or a.get('q', 0) or a.get('quantity', 0))
                        elif isinstance(a, (list, tuple)) and len(a) >= 2:
                            p, s = float(a[0]), float(a[1])
                        else:
                            continue
                        if p > 0 and s > 0:
                            normalized_asks.append((p, s))

                    normalized_asks.sort(key=lambda x: x[0])  # Cheapest asks first

                    for ask_price, ask_size in normalized_asks:
                        fill_shares = min(remaining_shares, ask_size)
                        weighted_price_sum += fill_shares * ask_price
                        remaining_shares -= fill_shares

                        if remaining_shares <= 0:
                            break

                    if remaining_shares > 0 and normalized_asks:
                        # Fill remaining at worst available price
                        worst_price = normalized_asks[-1][0]
                        weighted_price_sum += remaining_shares * worst_price
                        remaining_shares = 0

                    shares_filled = target_shares - remaining_shares
                    execution_price = weighted_price_sum / shares_filled if shares_filled > 0 else market_price

                    # Slippage: absolute difference in cents, not % of tiny price
                    # A 0.001 market with 0.0012 exec = 0.2¢ slippage, not 20%!
                    slippage_pct = (execution_price - market_price)  # absolute price diff
                    # Normalise to a percentage that makes sense (relative to 0.50 mid)
                    slippage_percent = slippage_pct / 0.50  # % of a "typical" 50¢ price

            else:  # SELL
                bids = orderbook.get('bids', [])
                if not bids:
                    slippage_percent = self.slippage_tolerance
                    execution_price = market_price * (1 - slippage_percent)
                else:
                    remaining_shares = target_shares
                    weighted_price_sum = 0

                    normalized_bids = []
                    for b in bids:
                        if isinstance(b, dict):
                            p = float(b.get('price', 0) or b.get('p', 0))
                            s = float(b.get('size', 0) or b.get('q', 0) or b.get('quantity', 0))
                        elif isinstance(b, (list, tuple)) and len(b) >= 2:
                            p, s = float(b[0]), float(b[1])
                        else:
                            continue
                        if p > 0 and s > 0:
                            normalized_bids.append((p, s))

                    normalized_bids.sort(key=lambda x: -x[0])  # Best bids first

                    for bid_price, bid_size in normalized_bids:
                        fill_shares = min(remaining_shares, bid_size)
                        weighted_price_sum += fill_shares * bid_price
                        remaining_shares -= fill_shares

                        if remaining_shares <= 0:
                            break

                    if remaining_shares > 0 and normalized_bids:
                        worst_price = normalized_bids[-1][0]
                        weighted_price_sum += remaining_shares * worst_price
                        remaining_shares = 0

                    shares_filled = target_shares - remaining_shares
                    execution_price = weighted_price_sum / shares_filled if shares_filled > 0 else market_price

                    slippage_pct = (market_price - execution_price)
                    slippage_percent = slippage_pct / 0.50

            # Cap slippage at ±50% for safety
            slippage_percent = max(-0.50, min(0.50, slippage_percent))

            filled_shares = target_shares
            filled_value = filled_shares * execution_price
            slippage_amount = filled_value - size_usd  # Positive = we paid more than ideal

            return {
                'status': 'success',
                'execution_price': execution_price,
                'filled_size': filled_shares,
                'filled_value': filled_value,
                'slippage': slippage_amount,
                'slippage_percent': slippage_percent,
                'reason': None
            }

        except Exception as e:
            logger.warning(f"Order simulation failed for {token_id}: {e}")
            # Fallback to simple slippage model
            slippage_percent = np.random.uniform(-self.slippage_tolerance, self.slippage_tolerance)
            execution_price = market_price * (1 + slippage_percent * 0.50)  # scale to mid-price
            filled_shares = size_usd / execution_price if execution_price > 0 else 0
            filled_value = filled_shares * execution_price
            slippage_amount = filled_value - size_usd

            return {
                'status': 'success',
                'execution_price': execution_price,
                'filled_size': filled_shares,
                'filled_value': filled_value,
                'slippage': slippage_amount,
                'slippage_percent': slippage_percent,
                'reason': f"Fallback simulation: {str(e)}"
            }

    def _update_portfolio_after_trades(self, orders: List[Dict]):
        """Update portfolio tracking after paper trades"""
        for order in orders:
            if order.get('status') == 'filled':
                token_id = order['token_id']
                size = order.get('filled_size', 0)
                price = order.get('execution_price', order.get('market_price', 0))

                # Update position in risk manager
                self.risk_manager.update_position(
                    token_id=token_id,
                    size=size,
                    price=price,
                    side='BUY',
                    category=order.get('category', 'General')
                )

    def _get_portfolio_value(self) -> float:
        """Get current estimated portfolio value (positions + uninvested cash)."""
        # Get current prices for all positions
        current_prices = {}
        for token_id in self.risk_manager.positions.keys():
            price = self.api.get_price(token_id)
            if price > 0:
                current_prices[token_id] = price

        if not current_prices:
            return self.capital  # No positions yet → portfolio = cash

        position_value = self.risk_manager.calculate_portfolio_value(current_prices, apply_slippage=True)
        invested_value = sum(
            abs(pos['size']) * current_prices.get(tid, pos['entry_price'])
            for tid, pos in self.risk_manager.positions.items()
        )
        cash_value = max(0.0, self.capital - invested_value)
        return position_value + cash_value

    def _attempt_recovery(self):
        """Attempt to recover from API failures"""
        self.api_failures += 1
        logger.warning(f"API failure #{self.api_failures}. Attempting recovery...")

        # Add alert for API failures
        if self.api_failures >= 3:  # Alert after 3 failures
            self.alert_manager.warning(
                f"API failure #{self.api_failures}. Attempting recovery...",
                source="trading_bot",
                metadata={"api_failures": self.api_failures}
            )

        if self.api_failures >= self.max_api_failures:
            logger.error("Max API failures reached. Consider checking network connectivity or API status.")
            # Add critical alert for max failures reached
            self.alert_manager.critical(
                f"Max API failures ({self.max_api_failures}) reached. Bot may not function properly.",
                source="trading_bot",
                metadata={"max_api_failures": self.max_api_failures}
            )
            # Reset failure count to allow continued operation but with warnings
            self.api_failures = 0

    # -------------------------------------------------------------------------
    # EXIT SIGNAL FRAMEWORK  (inspired by Freqtrade's populate_exit_trend)
    # -------------------------------------------------------------------------

    def _track_position_price(self, token_id: str, current_price: float):
        """Track price history for a position (for momentum calculation)"""
        now = datetime.now()
        if token_id not in self._position_price_history:
            self._position_price_history[token_id] = []
        self._position_price_history[token_id].append((current_price, now))
        # Keep last 100 entries max
        if len(self._position_price_history[token_id]) > 100:
            self._position_price_history[token_id] = \
                self._position_price_history[token_id][-100:]

    def _get_position_entry_info(self, token_id: str) -> Tuple[float, datetime, str]:
        """Get entry price, entry time, and side for a position"""
        # Scan trade_history for the oldest fill
        fills = [t for t in self.trade_history
                 if t.get('token_id') == token_id and t.get('status') == 'filled']
        if not fills:
            return 0.0, datetime.now(), 'BUY'
        first_fill = min(fills, key=lambda t: t.get('filled_at', ''))
        try:
            entry_time = datetime.fromisoformat(first_fill.get('filled_at', datetime.now().isoformat()))
        except Exception:
            entry_time = datetime.now()
        return (
            float(first_fill.get('execution_price', 0.5)),
            entry_time,
            first_fill.get('side', 'BUY')
        )

    def _calculate_momentum(self, token_id: str, current_price: float) -> Tuple[float, float]:
        """
        Calculate price momentum for a position.
        Returns (momentum, confidence) where:
          - momentum: price change fraction (positive = trending our way)
          - confidence: 0-1, how reliable the momentum signal is
        """
        if token_id not in self._position_price_history:
            return 0.0, 0.0

        history = self._position_price_history[token_id]
        if len(history) < 3:
            return 0.0, 0.0

        # Look at last N data points (use up to 20)
        lookback = min(len(history), 20)
        recent = history[-lookback:]

        # Linear regression slope as momentum
        prices = np.array([h[0] for h in recent])
        times = np.arange(len(recent))
        if len(prices) < 2:
            return 0.0, 0.0

        # Simple linear slope: positive = price rising
        slope = np.polyfit(times, prices, 1)[0]
        # Normalize to fraction
        avg_price = np.mean(prices)
        momentum = slope / avg_price if avg_price > 0 else 0.0

        # Confidence based on R² of the fit
        if len(prices) >= 3:
            coeffs = np.polyfit(times, prices, 1)
            predicted = np.polyval(coeffs, times)
            ss_res = np.sum((prices - predicted) ** 2)
            ss_tot = np.sum((prices - avg_price) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        else:
            r_squared = 0.0

        return float(momentum), max(0.0, min(1.0, r_squared))

    def _eval_take_profit(self, token_id: str, entry_price: float,
                          estimated_prob: float, market_price: float) -> Tuple[bool, float]:
        """
        TAKE PROFIT signal: our probability estimate has moved FAVORABLY.
        For a BUY position: estimated_prob > entry_price + threshold.
        Returns (triggered, confidence).
        """
        entry_prob = self.model.estimate_probability({
            'token_id': token_id, 'current_price': entry_price, 'price': entry_price
        })
        prob_drift = estimated_prob - entry_prob

        if prob_drift >= self.take_profit_threshold:
            # Map drift to confidence: bigger drift = higher confidence
            confidence = min(1.0, prob_drift / (self.take_profit_threshold * 2))
            return True, max(0.5, confidence)
        return False, 0.0

    def _eval_stop_loss(self, token_id: str, entry_price: float,
                         estimated_prob: float, market_price: float) -> Tuple[bool, float]:
        """
        STOP LOSS signal: probability has moved AGAINST our position.
        For a BUY: estimated_prob has dropped below entry_price - threshold.
        Called BEFORE take-profit so stop-loss always takes priority.
        """
        entry_prob = self.model.estimate_probability({
            'token_id': token_id, 'current_price': entry_price, 'price': entry_price
        })
        prob_drift = entry_prob - estimated_prob  # positive = price moved against us

        if prob_drift >= self.stop_loss_threshold:
            confidence = min(1.0, prob_drift / (self.stop_loss_threshold * 2))
            return True, max(0.5, confidence)
        return False, 0.0

    def _eval_time_exit(self, market: Dict, entry_time: datetime) -> Tuple[bool, float, str]:
        """
        TIME EXIT signal: market resolving soon and probability hasn't converged.
        Returns (triggered, confidence, reason_str).
        """
        if self.time_exit_hours is None:
            return False, 0.0, ""

        end_date_str = market.get('endDateIso') or market.get('endDate', '')
        if not end_date_str:
            return False, 0.0, ""

        try:
            end_date = datetime.fromisoformat(end_date_str.replace('Z', '+00:00'))
        except Exception:
            return False, 0.0, ""

        hours_remaining = (end_date - datetime.now(timezone.utc)).total_seconds() / 3600

        if hours_remaining <= self.time_exit_hours:
            reason = f"Resolves in {hours_remaining:.1f}h (threshold: {self.time_exit_hours}h)"
            confidence = 1.0 - (hours_remaining / self.time_exit_hours) if self.time_exit_hours > 0 else 1.0
            return True, max(0.5, min(1.0, confidence)), reason

        return False, 0.0, ""

    def _eval_momentum_reversal(self, token_id: str,
                                current_price: float, entry_price: float,
                                hold_hours: float) -> Tuple[bool, float]:
        """
        MOMENTUM REVERSAL signal: price was trending our way but has stalled/reversed.
        Stale positions (no momentum) also trigger here.
        """
        momentum, confidence = self._calculate_momentum(token_id, current_price)

        if len(self._position_price_history.get(token_id, [])) < 3:
            return False, 0.0

        # Positive momentum is good (price trending in our direction for a BUY)
        # Negative momentum means reversal — exit
        if momentum < -self.min_momentum_hold:
            return True, max(0.5, min(1.0, confidence))

        # Stale position: price barely moving for extended period
        if hold_hours > self.stale_exit_hours and abs(momentum) < self.min_momentum_hold:
            return True, 0.7

        return False, 0.0

    def evaluate_exit_signals(self, token_id: str, current_price: float,
                               estimated_prob: float, market: Dict) -> List[ExitSignal]:
        """
        Check ALL exit signals for a position and return list of triggered exits.
        Inspired by Freqtrade's populate_exit_trend — returns ExitSignal objects
        so the strategy decides which exits to act on.
        """
        entry_price, entry_time, _ = self._get_position_entry_info(token_id)
        if entry_price <= 0:
            return []

        hold_duration = (datetime.now() - entry_time).total_seconds() / 3600
        pnl = (current_price - entry_price) / entry_price if entry_price > 0 else 0

        signals = []

        # 1. Stop loss (check FIRST — highest priority)
        triggered, confidence = self._eval_stop_loss(
            token_id, entry_price, estimated_prob, current_price)
        if triggered and confidence >= self.min_exit_confidence:
            signals.append(ExitSignal(
                token_id=token_id,
                reason=ExitReason.STOP_LOSS,
                current_price=current_price, entry_price=entry_price,
                estimated_prob=estimated_prob, market_price=market.get('price', 0),
                unrealized_pnl=pnl, hold_duration_hours=hold_duration,
                confidence=confidence
            ))

        # 2. Take profit
        triggered, confidence = self._eval_take_profit(
            token_id, entry_price, estimated_prob, current_price)
        if triggered and confidence >= self.min_exit_confidence:
            signals.append(ExitSignal(
                token_id=token_id,
                reason=ExitReason.TAKE_PROFIT,
                current_price=current_price, entry_price=entry_price,
                estimated_prob=estimated_prob, market_price=market.get('price', 0),
                unrealized_pnl=pnl, hold_duration_hours=hold_duration,
                confidence=confidence
            ))

        # 3. Momentum reversal
        triggered, confidence = self._eval_momentum_reversal(
            token_id, current_price, entry_price, hold_duration)
        if triggered and confidence >= self.min_exit_confidence:
            signals.append(ExitSignal(
                token_id=token_id,
                reason=ExitReason.MOMENTUM_REVERSAL,
                current_price=current_price, entry_price=entry_price,
                estimated_prob=estimated_prob, market_price=market.get('price', 0),
                unrealized_pnl=pnl, hold_duration_hours=hold_duration,
                confidence=confidence
            ))

        # 4. Time exit
        triggered, confidence, reason = self._eval_time_exit(market, entry_time)
        if triggered and confidence >= self.min_exit_confidence:
            signals.append(ExitSignal(
                token_id=token_id,
                reason=ExitReason.TIME_EXIT,
                current_price=current_price, entry_price=entry_price,
                estimated_prob=estimated_prob, market_price=market.get('price', 0),
                unrealized_pnl=pnl, hold_duration_hours=hold_duration,
                confidence=confidence
            ))

        # 5. Stale position
        if hold_duration > self.stale_exit_hours:
            signals.append(ExitSignal(
                token_id=token_id,
                reason=ExitReason.STALE_POSITION,
                current_price=current_price, entry_price=entry_price,
                estimated_prob=estimated_prob, market_price=market.get('price', 0),
                unrealized_pnl=pnl, hold_duration_hours=hold_duration,
                confidence=0.8
            ))

        return signals

    def manage_exits(self) -> Tuple[List[Dict], List[Dict]]:
        """
        Check all open positions for exit signals and generate closing orders.

        Returns (exit_orders, exit_signals):
          - exit_orders: orders to close positions
          - exit_signals: all signals found (for logging/alerting)

        Called at the START of each run() cycle — before new signal generation.
        This is the key insight from the Polymarket blog: exit management before entry.
        """
        exit_orders = []
        exit_signals = []

        if not self.risk_manager.positions:
            return exit_orders, exit_signals

        # Periodic position poll (WebSocket drop safeguard from Polymarket blog)
        now = datetime.now()
        poll_due = (
            self._last_position_poll is None
            or (now - self._last_position_poll).total_seconds() >= self._position_poll_interval_secs
        )
        if poll_due:
            logger.debug("Polling open positions (WebSocket safeguard)")
            self._last_position_poll = now
            # Rebuild positions from trade history to catch any WebSocket drops
            self.update_positions_from_trades()
            # Force price history refresh for all tokens
            for token_id in self.risk_manager.positions:
                price = self.api.get_price(token_id)
                if price > 0:
                    self._track_position_price(token_id, price)

        for token_id in list(self.risk_manager.positions.keys()):
            try:
                current_price = self.api.get_price(token_id)
                if current_price <= 0:
                    continue

                # Track price for momentum
                self._track_position_price(token_id, current_price)

                # Get current market's raw data for time-exit checks
                market = self._token_market_cache.get(token_id, {})

                # Get fresh probability estimate
                market_data = {
                    'token_id': token_id,
                    'current_price': current_price,
                    'price': current_price,
                    **market
                }
                estimated_prob = float(self.model.estimate_probability(market_data))

                # Evaluate all exit signals
                signals = self.evaluate_exit_signals(token_id, current_price, estimated_prob, market)
                exit_signals.extend(signals)

                if not signals:
                    continue

                # Use highest-confidence exit signal for this position
                best_signal = max(signals, key=lambda s: s.confidence * (s.reason == ExitReason.STOP_LOSS and 2.0 or 1.0))

                # Build exit order
                position = self.risk_manager.positions.get(token_id, {})
                size = position.get('size', 0)
                if size <= 0:
                    continue

                exit_order = {
                    'token_id': token_id,
                    'size': size,
                    'exit_price': current_price,
                    'entry_price': best_signal.entry_price,
                    'side_to_close': 'SELL',  # Close BUY position with SELL
                    'reason': best_signal.reason.value,
                    'confidence': best_signal.confidence,
                    'unrealized_pnl': best_signal.unrealized_pnl,
                    'hold_hours': best_signal.hold_duration_hours,
                    'exit_value': size * current_price,
                    'market_question': market.get('question', token_id),
                    'timestamp': datetime.now().isoformat()
                }
                exit_orders.append(exit_order)

                logger.info(f"EXIT SIGNAL [{best_signal.reason.value}] {token_id}: "
                            f"price={current_price:.3f}, prob={estimated_prob:.3f}, "
                            f"confidence={best_signal.confidence:.0%}, "
                            f"pnl={best_signal.unrealized_pnl:+.1%}")

            except Exception as e:
                logger.warning(f"Error evaluating exits for {token_id}: {e}")
                continue

        return exit_orders, exit_signals

    def manage_open_limit_orders(self):
        """
        Manage lifecycle of open limit orders.
        Polls the status of pending/open limit orders on the CLOB,
        updates trade history and positions when filled, and
        re-quotes if the market price has moved.
        """
        if not self.use_limit_orders:
            return

        open_quotes = self.limit_quoter.open_quotes
        if not open_quotes:
            return

        logger.info(f"Managing {len(open_quotes)} open limit order quotes...")

        for token_id, quote in list(open_quotes.items()):
            order_id = quote.get("order_id")
            side = quote.get("side", "BUY")
            size = quote.get("size", 0)
            limit_price = quote.get("price", 0.5)

            if not order_id:
                continue

            is_dry = order_id.startswith("dry_") or self.paper_mode or self.live_dry_run
            is_filled = False
            is_cancelled = False
            actual_fill_price = limit_price
            actual_filled_size = size

            current_price = self.api.get_price(token_id)

            if is_dry:
                # Simulate fill if price crossed limit price
                if current_price > 0:
                    if side == "BUY" and current_price <= limit_price:
                        is_filled = True
                    elif side == "SELL" and current_price >= limit_price:
                        is_filled = True
            else:
                # Poll live CLOB order status
                try:
                    status_res = self.execution_engine.get_order_status(order_id)
                    if status_res.get("status") == "success":
                        order_data = status_res.get("order_data", {})
                        clob_status = str(order_data.get("status", "")).upper()
                        
                        if clob_status == "FILLED":
                            is_filled = True
                            actual_fill_price = float(order_data.get("price", limit_price))
                            actual_filled_size = float(order_data.get("size", size))
                        elif clob_status in ("CANCELED", "CANCELLED", "EXPIRED"):
                            is_cancelled = True
                    else:
                        logger.warning(f"Failed to fetch order status for {order_id}: {status_res.get('error')}")
                except Exception as e:
                    logger.error(f"Error polling live order status for {order_id}: {e}")

            if is_filled:
                logger.info(f"Limit order {order_id} FILLED at {actual_fill_price:.3f} (size={actual_filled_size})")
                # Update trade_history status to FILLED or EXITED
                trade_found = False
                for trade in self.trade_history:
                    if trade.get("order_id") == order_id:
                        # Determine if this was an exit order or entry order
                        is_exit = trade.get("side_to_close") == "SELL" or trade.get("reason") in [r.value for r in ExitReason]
                        
                        if is_exit:
                            # It's an exit order
                            realized_pnl = (actual_fill_price - trade.get("entry_price", 0)) * actual_filled_size
                            trade.update({
                                "status": TradeStatus.EXITED.value,
                                "exit_price": actual_fill_price,
                                "realized_pnl": realized_pnl,
                                "closed_at": datetime.now().isoformat()
                            })
                            # Remove from risk manager
                            self.risk_manager.update_position(
                                token_id=token_id,
                                size=actual_filled_size,
                                price=actual_fill_price,
                                side="SELL",
                                category="exit"
                            )
                        else:
                            # It's an entry order
                            trade.update({
                                "status": TradeStatus.FILLED.value,
                                "execution_price": actual_fill_price,
                                "filled_size": actual_filled_size,
                                "filled_value": actual_filled_size * actual_fill_price,
                                "filled_at": datetime.now().isoformat()
                            })
                            # Add to risk manager
                            self.risk_manager.update_position(
                                token_id=token_id,
                                size=actual_filled_size,
                                price=actual_fill_price,
                                side="BUY",
                                category=trade.get("category", "default")
                            )
                        trade_found = True
                        break
                
                # Clean up tracking
                self.limit_quoter._open_quotes.pop(token_id, None)

            elif is_cancelled:
                logger.info(f"Limit order {order_id} was cancelled or expired.")
                for trade in self.trade_history:
                    if trade.get("order_id") == order_id:
                        trade["status"] = TradeStatus.FAILED.value
                        break
                self.limit_quoter._open_quotes.pop(token_id, None)

            else:
                # Order still open/pending. Re-evaluate quoting if price moved.
                try:
                    requote_res = self.limit_quoter.requote(token_id)
                    req_status = requote_res.get("status")
                    if req_status in ("success", "submitted", "dry_run"):
                        new_order_id = requote_res.get("order_id")
                        new_price = requote_res.get("price")
                        logger.info(f"Re-quoted order {order_id} -> new order {new_order_id} at {new_price:.3f}")
                        
                        # Update the corresponding trade in history with the new order details
                        for trade in self.trade_history:
                            if trade.get("order_id") == order_id:
                                trade["order_id"] = new_order_id
                                if "execution_price" in trade:
                                    trade["execution_price"] = new_price
                                break
                except Exception as e:
                    logger.error(f"Error requoting limit order for {token_id}: {e}")

    def execute_exit_order(self, exit_order: Dict):
        """Execute a single exit order (paper or live).

        Records realized P&L (sell proceeds - buy cost) on the closed trade so
        ``_update_performance_metrics`` can report actual gains/losses.
        """
        try:
            if self.paper_mode:
                entry_price = exit_order.get('entry_price', 0)
                exit_price = exit_order.get('exit_price', 0)
                size_shares = exit_order.get('size', 0)

                # realized P&L = (exit_price - entry_price) * shares
                realized_pnl = (exit_price - entry_price) * size_shares

                logger.info(f"[PAPER EXIT] {exit_order.get('market_question', '')[:40]}: "
                            f"closed at {exit_price:.3f} "
                            f"({exit_order['reason']}, conf={exit_order['confidence']:.0%}, "
                            f"pnl=${realized_pnl:+.2f})")
                # Record as closed trade with realized P&L
                closed_trade = exit_order.copy()
                closed_trade['status'] = TradeStatus.EXITED.value
                closed_trade['closed_at'] = datetime.now().isoformat()
                closed_trade['realized_pnl'] = realized_pnl
                self.trade_history.append(closed_trade)
                # Remove from risk manager
                self.risk_manager.update_position(
                    token_id=exit_order['token_id'],
                    size=exit_order['size'],
                    price=exit_price,
                    side=exit_order['side_to_close'],
                    category='exit'
                )
                self.alert_manager.info(
                    f"Paper exit: {exit_order['reason']} for {exit_order['token_id']} "
                    f"at {exit_price:.3f} (pnl=${realized_pnl:+.2f}, "
                    f"conf={exit_order['confidence']:.0%})",
                    source="exit_manager",
                    metadata=exit_order
                )
            else:
                # Live: use execution engine
                entry_price = exit_order.get('entry_price', 0)
                exit_price = exit_order.get('exit_price', 0)
                size_shares = exit_order.get('size', 0)

                result = self.execution_engine.execute_market_order(
                    token_id=exit_order['token_id'],
                    side=exit_order['side_to_close'],
                    size=exit_order['exit_value'],
                    price=exit_price
                )
                if result.get('status') == 'success':
                    actual_fill_price = float(result.get('price') or exit_price)
                    realized_pnl = (actual_fill_price - entry_price) * size_shares
                    exit_order['status'] = TradeStatus.EXITED.value
                    exit_order['execution_result'] = result
                    exit_order['realized_pnl'] = realized_pnl
                    self.trade_history.append(exit_order)
                    logger.info(f"Live exit executed: {exit_order['token_id']}")
        except Exception as e:
            logger.error(f"Exit order failed for {exit_order['token_id']}: {e}")

    def fetch_markets(self, min_edge: float = 0.03) -> List[Market]:
        logger.info("Fetching markets from Polymarket...")
        raw_markets = self.api.get_active_markets(limit=50)

        # Rebuild token_id → raw_market cache for exit signal time-exit checks
        self._token_market_cache.clear()
        import json as _json
        for raw in raw_markets:
            token_ids = raw.get("clobTokenIds") or []
            if isinstance(token_ids, str):
                try:
                    token_ids = _json.loads(token_ids)
                except Exception:
                    continue
            for tid in token_ids:
                self._token_market_cache[str(tid)] = raw
        markets = []

        for raw in raw_markets:
            try:
                # Gamma API currently provides token ids via `clobTokenIds` and prices via `outcomePrices`.
                token_ids = raw.get("clobTokenIds") or []
                prices = raw.get("outcomePrices") or []

                # Gamma sometimes returns these as JSON-encoded strings.
                import json
                if isinstance(token_ids, str):
                    token_ids = json.loads(token_ids)
                if isinstance(prices, str):
                    prices = json.loads(prices)

                if len(token_ids) < 2:
                    continue

                # Convert to proper types
                token_ids = [str(tid) for tid in token_ids]
                prices = [float(p) for p in prices]

                # Skip if we don't have matching arrays
                if len(token_ids) != len(prices):
                    continue

                # Validate prices are in reasonable range
                if any(p <= 0 or p >= 1 for p in prices):
                    continue

                # Estimate probabilities outcome-by-outcome. Most bundled models are
                # single-token models, so pass a normalized outcome view.
                your_probs = []
                for token_id, price in zip(token_ids, prices):
                    outcome_raw = dict(raw)
                    outcome_raw.update({
                        "token_id": token_id,
                        "token_id_yes": token_id,
                        "current_price": price,
                        "price": price,
                    })
                    your_probs.append(float(self.model.estimate_probability(outcome_raw)))

                # Create a Market object for each outcome
                for i, (token_id, price, prob) in enumerate(zip(token_ids, prices, your_probs)):
                    # Calculate edge for this outcome
                    edge = prob - price

                    # Skip if edge doesn't meet minimum threshold
                    if abs(edge) < min_edge:
                        continue

                    market = Market(
                        condition_id=raw.get("conditionId", ""),
                        question=raw.get("question", "")[:60],
                        token_id=token_id,  # Single token ID for this outcome
                        price=price,        # Single price for this outcome
                        probability=prob,   # Single probability for this outcome
                        liquidity=float(raw.get("liquidity", 0)) / len(token_ids),  # Distribute liquidity
                        volume_24h=float(raw.get("volume", 0)) / len(token_ids),    # Distribute volume
                        category=raw.get("category", "General"),
                        resolution_date=raw.get("endDateIso", raw.get("endDate", "")),
                        outcome="YES" if i == 0 else "NO",  # Index 0 = YES, 1+ = NO
                    )

                    markets.append(market)

            except Exception as e:
                logger.error(f"Error processing market: {e}")
                self.api_failures += 1
                continue

        logger.info(f"Selected {len(markets)} outcomes with edge > {min_edge:.1%}")
        # Reset API failure counter on successful fetch
        self.api_failures = 0
        self.last_successful_fetch = datetime.now()
        return markets
    
    def optimize_portfolio(self, markets: List[Market]) -> np.ndarray:
        if not markets:
            return np.array([])
        
        allocations, status, info = self.optimizer.optimize(markets, self.constraints, capital=self.capital)
        logger.info(f"Optimization {status.value} in {info['iterations']} iterations")
        logger.info(f"Expected log utility: {info['final_objective']:.6f}")
        return allocations
    
    def generate_orders(self, markets: List[Market], allocations: np.ndarray) -> List[Dict]:

        """Convert allocations to orders.

        When the slippage guard flags a proposed size as too large, the order
        is *downsized* to the maximum that fits within the tolerance rather than
        dropped entirely (the old "skip" behaviour meant the bot never traded).
        """

        orders = []

        for i, m in enumerate(markets):
            alloc = allocations[i]
            if alloc < 0.001:
                continue

            amount = alloc * self.capital

            # Check for slippage risk — downsize if necessary instead of skipping
            slippage_risk = self.risk_manager.check_trade_slippage_risk(
                token_id=m.token_id,
                size_usd=amount,
                liquidity_usd=m.liquidity,
                max_tolerance=self.slippage_tolerance
            )

            if not slippage_risk.get('safe', True):
                suggested = slippage_risk.get('suggested_size', 0.0)
                if suggested > 0:
                    logger.info(f"Downsizing order for {m.token_id}: "
                                f"${amount:.2f} -> ${suggested:.2f} (slippage guard)")
                    amount = suggested
                    # Recompute allocation fraction after downsizing
                    alloc = amount / self.capital if self.capital > 0 else 0
                else:
                    logger.warning(f"Skipping order for {m.token_id}: "
                                   f"{slippage_risk.get('message', 'No room within tolerance')}")
                    continue

            if amount < (self.constraints.min_bet_size * self.capital):
                logger.debug(f"Skipping {m.token_id}: downsized amount "
                             f"${amount:.2f} below min-bet threshold")
                continue

            order = {
                "market": m.question,
                "condition_id": m.condition_id,
                "token_id": m.token_id,
                "direction": f"BUY {m.outcome}",  # e.g. "BUY YES" or "BUY NO"
                "size": amount,
                "allocation": alloc,
                "market_price": m.price,
                "your_prob": m.probability,
                "edge": abs(m.edge),
                "category": m.category,
                "estimated_slippage": slippage_risk.get('estimated_slippage', 0.0),
                "timestamp": datetime.now().isoformat()
            }
            orders.append(order)

        return orders

    
    def execute_paper_trades(self, orders: List[Dict]):
        """Execute trades in paper trading mode with realistic slippage simulation"""
        print("\n" + "=" * 80)
        print("PAPER TRADE EXECUTION")
        print("=" * 80)

        total_cost = 0
        total_value = 0

        for order in orders:
            # Simulate order book and slippage for paper trading
            execution_result = self._simulate_order_execution(
                token_id=order['token_id'],
                side='BUY',  # We always buy in this bot
                size_usd=order['size'],
                market_price=order.get('market_price', 0.5)
            )

            if execution_result['status'] == 'failed':
                logger.warning(f"Paper trade failed for {order['market']}: {execution_result.get('reason')}")
                continue

            filled_order = order.copy()
            filled_order.update({
                'side': 'BUY',
                'status': TradeStatus.FILLED.value,
                'execution_price': execution_result['execution_price'],
                'filled_size': execution_result['filled_size'],
                'filled_value': execution_result['filled_value'],
                'slippage': execution_result['slippage'],
                'slippage_percent': execution_result['slippage_percent'],
                'filled_at': datetime.now().isoformat(),
            })

            print(f"\n[TARGET] {order['market']}")
            print(f"   Direction: BUY {order['direction']}")
            print(f"   Size: ${order['size']:,.2f} ({order['allocation']:.1%})")
            print(f"   Market Price: {order.get('market_price', 0):.1%}")
            print(f"   Execution Price: {execution_result['execution_price']:.1%}")
            print(f"   Your Estimate: {order['your_prob']:.1%}")
            print(f"   Edge: {order['edge']:.1%}")
            print(f"   Slippage: {execution_result['slippage_percent']:.2%}")
            total_cost += execution_result['filled_value']
            total_value += execution_result['filled_value']  # In paper trading, we assume immediate fill at execution price
            self.trade_history.append(filled_order)

        # Update portfolio tracking
        self._update_portfolio_after_trades(orders)

        print(f"\n{'=' * 80}")
        print(f"TOTAL COST: ${total_cost:,.2f} ({total_cost/self.capital:.1%})")
        print(f"ESTIMATED VALUE: ${total_value:,.2f}")
        print(f"CASH: ${self.capital - total_cost:,.2f}")
        print(f"PORTFOLIO VALUE: ${self._get_portfolio_value():,.2f}")
        print("=" * 80)

    def execute_live_trades(self, orders: List[Dict]):
        """Execute orders in live trading mode with enhanced error handling"""
        readiness = self.execution_engine.validate_live_ready()
        if not readiness["ready"]:
            logger.warning("Live execution is not enabled: %s", readiness)
        orders_to_submit = orders[:self.max_live_orders_per_cycle]
        logger.info(f"Executing {len(orders_to_submit)} live trades")
        for order in orders_to_submit:
            try:
                market_price = float(order.get('market_price') or 0)

                if self.use_limit_orders and market_price > 0:
                    # ── Limit Order (post-only GTC) ─────────────────────────
                    # Re-quote any existing position; place new if none.
                    token_id = order['token_id']
                    size_usd = order['size']
                    # Convert dollar size → shares for limit order
                    size_shares = size_usd / market_price if market_price > 0 else size_usd
                    quote_result = self.limit_quoter.quote(
                        token_id=token_id,
                        side='BUY',
                        size=size_shares,
                    )
                    result = quote_result
                    order_type_label = "LIMIT"
                    logger.info(
                        "Limit order quoted: %s @ %.4f (agg=%.1f, status=%s)",
                        token_id[:12], quote_result.get('price', 0),
                        self.quote_aggressiveness,
                        quote_result.get('status'),
                    )
                else:
                    # ── Market Order (FOK) ───────────────────────────────────
                    result = self.execution_engine.execute_market_order(
                        token_id=order['token_id'],
                        side='BUY',
                        size=order['size'],
                        price=market_price or None,
                    )
                    order_type_label = "MARKET"

                # Add execution result to order for tracking
                order['execution_result'] = result
                # Only add to history if successful
                if result.get('status') == 'success':
                    order.update({
                        'side': 'BUY',
                        'status': result.get('fill_status', TradeStatus.FILLED.value),
                        'execution_price': result.get('price') or order.get('market_price'),
                        'filled_size': result.get('filled_size') or (
                            order['size'] / max(float(order.get('market_price', 0)), 1e-9)
                        ),
                        'filled_value': result.get('filled_value', order['size']),
                        'order_id': result.get('order_id'),
                        'filled_at': datetime.now().isoformat(),
                    })
                    self.trade_history.append(order)
                    logger.info(f"Live trade executed: {order['market']} {order['direction']} ${order['size']:.2f}")
                elif result.get('status') == 'dry_run':
                    dry_run_order = order.copy()
                    dry_run_order['status'] = TradeStatus.DRY_RUN.value
                    dry_run_order['execution_result'] = result
                    dry_run_order['order_id'] = result.get('order_id')
                    dry_run_order['execution_price'] = result.get('price') or order.get('market_price')
                    self.trade_history.append(dry_run_order)
                    logger.info(f"Live dry run: {order['market']} {order['direction']} ${order['size']:.2f}")
                else:
                    logger.warning(f"Live trade failed for {order['market']}: {result.get('error')}")
                    # Add failed trade to history for tracking
                    failed_order = order.copy()
                    failed_order['execution_result'] = result
                    self.trade_history.append(failed_order)

            except Exception as e:
                logger.error(f"Failed to execute live trade for {order['market']}: {e}")
                # Add failed trade to history for tracking
                failed_order = order.copy()
                failed_order['execution_error'] = str(e)
                self.trade_history.append(failed_order)

    def execute_live_arbitrage(self, arb_orders: List[Dict]):
        """Execute arbitrage orders in live trading mode with enhanced error handling"""
        readiness = self.execution_engine.validate_live_ready()
        if not readiness["ready"]:
            logger.warning("Live arbitrage execution is not enabled: %s", readiness)
        arb_orders_to_submit = arb_orders[:self.max_live_orders_per_cycle]
        logger.info(f"Executing {len(arb_orders_to_submit)} live arbitrage opportunities")
        for order in arb_orders_to_submit:
            try:
                # Buy YES token
                result_yes = self.execution_engine.execute_market_order(
                    token_id=order['token_id_yes'],
                    side='BUY',
                    size=order['size']
                )
                # Buy NO token
                result_no = self.execution_engine.execute_market_order(
                    token_id=order['token_id_no'],
                    side='BUY',
                    size=order['size']
                )
                # Record the arbitrage trade
                arb_record = order.copy()
                arb_record['execution_result_yes'] = result_yes
                arb_record['execution_result_no'] = result_no
                # Only add to history if both executions were successful
                if result_yes.get('status') == 'success' and result_no.get('status') == 'success':
                    self.trade_history.append(arb_record)
                    logger.info(f"Live arbitrage executed: {order['market']} YES+NO ${order['size']:.2f} each")
                elif result_yes.get('status') == 'dry_run' or result_no.get('status') == 'dry_run':
                    arb_record['status'] = TradeStatus.DRY_RUN.value
                    self.trade_history.append(arb_record)
                    logger.info(f"Live arbitrage dry run: {order['market']} YES+NO ${order['size']:.2f} each")
                else:
                    logger.warning(f"Live arbitrage partially failed for {order['market']}: "
                                 f"YES={result_yes.get('status')}, NO={result_no.get('status')}")
                    # Add failed arbitrage to history for tracking
                    failed_order = order.copy()
                    failed_order['execution_result_yes'] = result_yes
                    failed_order['execution_result_no'] = result_no
                    self.trade_history.append(failed_order)

            except Exception as e:
                logger.error(f"Failed to execute live arbitrage for {order['market']}: {e}")
                # Add failed arbitrage to history for tracking
                failed_order = order.copy()
                failed_order['execution_error'] = str(e)
                self.trade_history.append(failed_order)

    def scan_yes_no_arbitrage(self) -> List[Dict]:
        if not self.enable_yes_no_arb:
            return []

        logger.info("Scanning YES/NO sum arbitrage...")
        raw_markets = self.api.get_active_markets(limit=50)
        orders: List[Dict] = []

        for raw in raw_markets:
            try:
                token_ids = raw.get("clobTokenIds") or []
                if isinstance(token_ids, str):
                    import json
                    token_ids = json.loads(token_ids)
                if len(token_ids) < 2:
                    continue

                token_yes = str(token_ids[0])
                token_no = str(token_ids[1])

                book_yes = self.api.get_orderbook(token_yes)
                book_no = self.api.get_orderbook(token_no)

                signal = self.arb_scanner.scan(raw, book_yes, book_no, self.capital)
                if not signal:
                    continue

                order = {
                    "type": "YES_NO_ARB",
                    "market": signal.question,
                    "condition_id": signal.condition_id,
                    "token_id_yes": signal.token_yes,
                    "token_id_no": signal.token_no,
                    "ask_yes": signal.ask_yes,
                    "ask_no": signal.ask_no,
                    "sum_price": signal.sum_price,
                    "edge": signal.edge,
                    "size": signal.size_dollars,
                    "timestamp": datetime.now().isoformat()
                }
                orders.append(order)

            except Exception as e:
                logger.error(f"Arb scan error: {e}")
                continue

        logger.info(f"Found {len(orders)} YES/NO arb opportunities")
        return orders
    
    def run(self):

        """Execute one trading cycle"""
        if not self.paper_mode:
            try:
                live_balance = self.execution_engine.get_collateral_balance()
                if live_balance > 0:
                    logger.info(f"Dynamic capital allocation: updating capital from ${self.capital:,.2f} to wallet balance ${live_balance:,.2f}")
                    self.capital = live_balance
            except Exception as e:
                logger.error(f"Failed to fetch live balance for dynamic capital allocation: {e}")

        print(f"\n{'=' * 80}")
        print(f"POLYMARKET BOT - {datetime.now()}")
        print(f"Mode: {'PAPER' if self.paper_mode else 'LIVE'}")
        print(f"Capital: ${self.capital:,.2f}")
        print(f"{'=' * 80}\n")

        # Increment cycle counter
        self.cycles_completed += 1

        try:
            # -----------------------------------------------------------------
            # STEP 0: LIMIT ORDER LIFECYCLE MANAGEMENT
            # -----------------------------------------------------------------
            self.manage_open_limit_orders()

            # -----------------------------------------------------------------
            # STEP 0.1: EXIT MANAGEMENT (before new entries — this is non-negotiable)
            # Inspired by Polymarket blog lesson: "build exit management before entry logic"
            # -----------------------------------------------------------------
            exit_orders, exit_signals = self.manage_exits()
            for exit_order in exit_orders:
                self.execute_exit_order(exit_order)
            if exit_orders:
                print(f"\n[WARNING] Closed {len(exit_orders)} position(s) this cycle")
                for sig in exit_signals:
                    print(f"   {sig.reason.value}: {sig.token_id} (conf={sig.confidence:.0%})")

            # -----------------------------------------------------------------
            # STEP 0.5: MARKET SETTLEMENT CHECK
            # Check if any held positions have markets that have settled
            # -----------------------------------------------------------------
            self.check_market_settlements()

            # -----------------------------------------------------------------
            # STEP 1: Find new markets
            # -----------------------------------------------------------------
            markets = self.fetch_markets()
            if not markets:
                print("No tradeable markets found")
                self.failed_cycles += 1
                self.alert_manager.warning(
                    "No tradeable markets found in current cycle",
                    source="trading_bot",
                    metadata={"cycle": self.cycles_completed}
                )
                return

            allocations = self.optimize_portfolio(markets)
            orders = self.generate_orders(markets, allocations)

            arb_orders = self.scan_yes_no_arbitrage()

            if not orders and not arb_orders:
                print("No orders generated")
                self.failed_cycles += 1
                self.alert_manager.info(
                    "No orders generated in current cycle",
                    source="trading_bot",
                    metadata={"cycle": self.cycles_completed, "markets_found": len(markets)}
                )
                return

            if self.paper_mode:
                if orders:
                    self.execute_paper_trades(orders)

                if arb_orders:
                    print("\n" + "=" * 80)
                    print("YES/NO ARBITRAGE (PAPER)")
                    print("=" * 80)
                    for o in arb_orders:
                        print(f"\n[TARGET] {o['market']}")
                        print(f"   Buy YES @ {o['ask_yes']:.3f} + NO @ {o['ask_no']:.3f} = {o['sum_price']:.3f}")
                        print(f"   Edge: {o['edge']:.2%} | Size: ${o['size']:.2f}")
                        filled_arb = o.copy()
                        filled_arb.update({
                            'status': TradeStatus.FILLED.value,
                            'filled_value': o['size'],
                            'filled_at': datetime.now().isoformat(),
                        })
                        self.trade_history.append(filled_arb)
                    print("=" * 80)
            else:
                # Live trading execution
                if orders:
                    self.execute_live_trades(orders)

                if arb_orders:
                    self.execute_live_arbitrage(arb_orders)

            # Update and display performance metrics
            self._update_performance_metrics()
            if len(self.trade_history) > 0:
                self.print_performance_report()

            # Check if portfolio rebalancing is needed
            self.check_and_execute_rebalance()

            # Check risk conditions and generate alerts
            self._check_risk_conditions()

            # Mark cycle as successful
            self.successful_cycles += 1

            # Save state after successful cycle
            self.save_state()

        except Exception as e:
            logger.error(f"Trading cycle failed: {e}", exc_info=True)
            # Attempt recovery by resetting connection state
            self._attempt_recovery()
            self.failed_cycles += 1
            self.alert_manager.error(
                f"Trading cycle failed: {str(e)}",
                source="trading_bot",
                metadata={"cycle": self.cycles_completed}
            )

    def print_performance_report(self):
        """Print a formatted performance report"""
        metrics = self.get_performance_summary()

        print("\n" + "=" * 80)
        print("PERFORMANCE REPORT")
        print("=" * 80)
        print(f"Total Trades: {metrics['total_trades']}")
        print(f"Winning Trades: {metrics['winning_trades']}")
        print(f"Losing Trades: {metrics['losing_trades']}")
        print(f"Win Rate: {metrics['win_rate']:.1%}")
        print(f"Total P&L: ${metrics['total_pnl']:,.2f} ({metrics['total_pnl_percent']:.1%})")
        print(f"Average Win: ${metrics['avg_win']:,.2f}")
        print(f"Average Loss: ${metrics['avg_loss']:,.2f}")
        print(f"Profit Factor: {metrics['profit_factor']:.2f}")
        print(f"Biggest Win (All-Time): ${metrics['biggest_win_history']:,.2f}")
        print(f"Biggest Win (Today): ${metrics['biggest_win_today']:,.2f}")
        print(f"Max Drawdown: {metrics['max_drawdown']:.1%}")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"Last Updated: {metrics['last_updated']}")
        print("=" * 80)

    def _check_risk_conditions(self):
        """Check risk conditions and generate alerts if thresholds are breached"""
        try:
            # Use risk_manager.positions (canonical, detailed) for all risk checks
            if not self.risk_manager.positions:
                return

            # Get current prices for all positions
            current_prices = {}
            for token_id in self.risk_manager.positions.keys():
                # For long positions, SELL is the conservative liquidation side.
                price = self.api.get_price(token_id, side="SELL")
                if price > 0:
                    current_prices[token_id] = price

            if not current_prices:
                return

            # Calculate portfolio value including uninvested cash.
            position_value = self.risk_manager.calculate_portfolio_value(current_prices, apply_slippage=True)
            invested_value = sum(
                abs(pos['size']) * current_prices.get(tid, pos['entry_price'])
                for tid, pos in self.risk_manager.positions.items()
            )
            cash_value = max(0.0, self.capital - invested_value)
            portfolio_value = position_value + cash_value

            # Update peak portfolio value for drawdown calculation
            if portfolio_value > self.peak_portfolio_value:
                self.peak_portfolio_value = portfolio_value

            # Check drawdown
            if self.peak_portfolio_value > 0:
                drawdown = (self.peak_portfolio_value - portfolio_value) / self.peak_portfolio_value
                if drawdown > self.drawdown_alert_threshold:
                    self.alert_manager.warning(
                        f"Drawdown alert: {drawdown:.1%} exceeds threshold {self.drawdown_alert_threshold:.1%}",
                        source="risk_manager",
                        metadata={
                            "drawdown": drawdown,
                            "threshold": self.drawdown_alert_threshold,
                            "portfolio_value": portfolio_value,
                            "peak_value": self.peak_portfolio_value
                        }
                    )

            # Check VaR
            var_95 = self.risk_manager.calculate_portfolio_var(current_prices, confidence=0.95)
            if portfolio_value > 0:
                var_percent = var_95 / portfolio_value
                if var_percent > self.var_alert_threshold:
                    self.alert_manager.warning(
                        f"VaR alert: {var_percent:.1%} exceeds threshold {self.var_alert_threshold:.1%}",
                        source="risk_manager",
                        metadata={
                            "var_95": var_95,
                            "var_percent": var_percent,
                            "threshold": self.var_alert_threshold,
                            "portfolio_value": portfolio_value
                        }
                    )

            # Check correlation
            corr_matrix, tokens = self.risk_manager.calculate_correlation_matrix()
            if len(corr_matrix) > 1:  # Need at least 2 positions for correlation
                # Get maximum absolute correlation excluding diagonal
                mask = ~np.eye(len(corr_matrix), dtype=bool)
                if np.any(mask):
                    max_correlation = np.max(np.abs(corr_matrix[mask]))
                    if max_correlation > self.correlation_alert_threshold:
                        high_pairs = self.risk_manager.get_high_correlation_pairs(
                            threshold=self.correlation_alert_threshold
                        )
                        if high_pairs:
                            pair_info = ", ".join([f"{pair[0]}/{pair[1]} ({pair[2]:.2f})"
                                                 for pair in high_pairs[:3]])  # Show top 3
                            self.alert_manager.warning(
                                f"High correlation detected: {pair_info}",
                                source="risk_manager",
                                metadata={
                                    "max_correlation": max_correlation,
                                    "threshold": self.correlation_alert_threshold,
                                    "high_pairs": high_pairs
                                }
                            )

            # Check risk limits
            risk_violations = self.risk_manager.check_risk_limits(current_prices, portfolio_value)
            for violation in risk_violations:
                self.alert_manager.error(
                    f"Risk limit violation: {violation['message']}",
                    source="risk_manager",
                    metadata=violation
                )

        except Exception as e:
            logger.error(f"Error checking risk conditions: {e}")
            # Don't let risk checking break the main trading cycle

    def check_market_settlements(self):
        """
        Check if any held positions have markets that have settled.
        Updates trade status and records realized P&L for settled markets.
        """
        try:
            # Get currently active markets to check settlement status
            active_markets = self.api.get_active_markets(limit=100)

            # Build a set of condition IDs that are still active
            active_condition_ids = {market.get("conditionId") for market in active_markets if market.get("conditionId")}

            # Check each position we hold
            for token_id in list(self.risk_manager.positions.keys()):
                # Get market data for this token from our cache
                market_raw = self._token_market_cache.get(token_id)
                if not market_raw:
                    continue

                condition_id = market_raw.get("conditionId")
                if not condition_id:
                    continue

                # If market is no longer active, it has settled/closed/resolved
                if condition_id not in active_condition_ids:
                    # Get current price for final valuation
                    current_price = self.api.get_price(token_id)
                    if current_price <= 0:
                        continue

                    # Find all filled trades for this token that aren't already settled/closed
                    settled_count = 0
                    for trade in self.trade_history:
                        if (trade.get('token_id') == token_id and
                            trade.get('status') == TradeStatus.FILLED.value and
                            trade.get('type') != 'YES_NO_ARB'):  # Skip arb for now (different logic)

                            # Calculate settlement P&L: for binary outcomes, resolution is 0 or 1
                            # Since we don't have direct resolution data, we'll use current price as proxy
                            # In a real implementation, you'd check the actual market resolution
                            entry_price = trade.get('entry_price', 0)
                            size = trade.get('size', 0)

                            if entry_price > 0 and size > 0:
                                # For simplicity, we'll use the current price as the resolution price
                                # A more sophisticated approach would check the actual market outcome
                                realized_pnl = (current_price - entry_price) * size

                                # Update the trade
                                trade['status'] = TradeStatus.SETTLED.value
                                trade['exit_price'] = current_price
                                trade['realized_pnl'] = realized_pnl
                                trade['exit_reason'] = 'market_settled'
                                trade['closed_at'] = datetime.now().isoformat()

                                # Update performance metrics will pick this up on next cycle
                                settled_count += 1

                                logger.info(f"Market settled for {token_id}: "
                                          f"entry={entry_price:.3f}, exit={current_price:.3f}, "
                                          f"size={size}, pnl=${realized_pnl:+.2f}")

                    if settled_count > 0:
                        # Also update risk manager position to zero since it's settled
                        self.risk_manager.update_position(
                            token_id=token_id,
                            size=0,  # Zero out position
                            price=current_price,
                            side='BUY',  # Direction doesn't matter for zero size
                            category='settled'
                        )

        except Exception as e:
            logger.error(f"Error checking market settlements: {e}")

    def update_performance_metrics(self):
        """Public wrapper — updates performance metrics based on trade history."""
        self._update_performance_metrics()

    def update_positions_from_trades(self):
        """Update current positions based on trade history.

        Rebuilds `self.positions` (token_id -> shares) and
        `self.risk_manager.positions` from the trade history.

        Only *open* fills are counted.  Trades with status ``'exited'``,
        ``'settled'``, or ``'closed'`` are excluded — they have already been
        accounted for.
        """
        # Reset positions
        self.positions = {}
        self.risk_manager.positions = {}

        for trade in self.trade_history:
            status = trade.get('status')
            # Only count trades that are live fills (not exits / settlements)
            if status != TradeStatus.FILLED.value:
                continue

            if trade.get('type') == 'YES_NO_ARB':
                for token_key, price_key in (('token_id_yes', 'ask_yes'),
                                              ('token_id_no', 'ask_no')):
                    token_id = trade.get(token_key)
                    price = float(trade.get(price_key, 0) or 0)
                    value = float(trade.get('size', 0) or 0)
                    if token_id and price > 0 and value > 0:
                        shares = value / price
                        self.positions[token_id] = self.positions.get(token_id, 0.0) + shares
                        self.risk_manager.update_position(
                            token_id=token_id,
                            size=shares,
                            price=price,
                            side='BUY',
                            category=trade.get('category', 'Arbitrage')
                        )
                continue

            token_id = trade.get('token_id')
            if not token_id:
                continue

            price = float(trade.get('execution_price', trade.get('market_price', 0)) or 0)
            side = trade.get('side', 'BUY')

            # Use filled_size (shares) if available, else derive from value
            filled_size = float(trade.get('filled_size', 0) or 0)
            if filled_size > 0:
                shares = filled_size
                value = filled_size * price
            else:
                value = float(trade.get('filled_value', trade.get('size', 0)) or 0)
                shares = value / price if price > 0 and value > 0 else 0

            if shares <= 0:
                continue

            sign = 1 if side == 'BUY' else -1
            self.positions[token_id] = self.positions.get(token_id, 0.0) + sign * shares

            if price > 0:
                self.risk_manager.update_position(
                    token_id=token_id,
                    size=shares,
                    price=price,
                    side=side,
                    category=trade.get('category', 'default')
                )

    def _make_serializable(self, obj):
        """Convert objects to JSON-serializable formats."""
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, Enum):
            return obj.value
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        else:
            return obj

    def save_state(self, path: str = "bot_state.json"):
        """Save bot state to JSON file for persistence."""
        try:
            state = {
                'trade_history': self.trade_history,
                'positions': self.positions,
                'performance_metrics': self.performance_metrics,
                'risk_manager': {
                    'positions': self.risk_manager.positions,
                    'price_history': self.risk_manager.price_history,
                },
                'cycles_completed': self.cycles_completed,
                'successful_cycles': self.successful_cycles,
                'failed_cycles': self.failed_cycles,
                'peak_portfolio_value': self.peak_portfolio_value,
                'api_failures': self.api_failures,
                'alerts': self.alert_manager.get_recent_alerts(limit=1000),  # Save all alerts
                'timestamp': datetime.now().isoformat()
            }

            # Convert non-serializable objects
            serializable_state = json.loads(json.dumps(state, default=self._make_serializable))

            with open(path, 'w') as f:
                json.dump(serializable_state, f, indent=2)

            logger.info(f"Bot state saved to {path}")
            return True
        except Exception as e:
            logger.error(f"Error saving bot state: {e}")
            return False

    def load_state(self, path: str = "bot_state.json"):
        """Load bot state from JSON file."""
        try:
            import os
            if not os.path.exists(path):
                logger.info(f"No state file found at {path}, starting fresh")
                return False

            with open(path, 'r') as f:
                state = json.load(f)

            # Restore state
            self.trade_history = state.get('trade_history', [])
            self.positions = state.get('positions', {})
            self.performance_metrics = state.get('performance_metrics', {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'total_pnl': 0.0,
                'total_pnl_percent': 0.0,
                'max_drawdown': 0.0,
                'sharpe_ratio': 0.0,
                'win_rate': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'profit_factor': 0.0,
                'biggest_win_history': 0.0,
                'biggest_win_today': 0.0,
                'last_updated': None
            })

            # Restore risk manager state
            risk_state = state.get('risk_manager', {})
            self.risk_manager.positions = risk_state.get('positions', {})
            self.risk_manager.price_history = risk_state.get('price_history', {})

            self.cycles_completed = state.get('cycles_completed', 0)
            self.successful_cycles = state.get('successful_cycles', 0)
            self.failed_cycles = state.get('failed_cycles', 0)
            self.peak_portfolio_value = state.get('peak_portfolio_value', 0.0)
            self.api_failures = state.get('api_failures', 0)

            # Restore alerts
            alerts_data = state.get('alerts', [])
            if hasattr(self.alert_manager, 'alert_file'):
                # Backup current alerts file
                import shutil
                if os.path.exists(self.alert_manager.alert_file):
                    shutil.copy2(self.alert_manager.alert_file, self.alert_manager.alert_file + '.backup')

                # Write restored alerts
                with open(self.alert_manager.alert_file, 'w') as f:
                    json.dump(alerts_data, f, indent=2)

            logger.info(f"Bot state loaded from {path}: {len(self.trade_history)} trades, "
                       f"{self.cycles_completed} cycles completed")
            return True
        except Exception as e:
            logger.error(f"Error loading bot state: {e}")
            return False

    def check_and_execute_rebalance(self):
        """Check if portfolio needs rebalancing and execute if needed"""
        try:
            # Update current positions
            self.update_positions_from_trades()

            if not self.positions:
                return

            # Get current market prices for position valuation
            current_prices = {}
            # In a real implementation, we'd fetch current prices for all tokens
            # For now, we'll use a simplified check based on trade history

            # Calculate current portfolio value
            total_invested = sum(abs(pos) for pos in self.positions.values())
            if total_invested == 0:
                return

            # Check if any position exceeds rebalance threshold
            needs_rebalance = False
            max_deviation = 0.0

            for token_id, position_amount in self.positions.items():
                position_weight = abs(position_amount) / self.capital
                # Check if position deviates significantly from target (simplified)
                # In reality, we'd compare to optimal weights from optimizer
                if position_weight > self.rebalance_threshold:
                    needs_rebalance = True
                    max_deviation = max(max_deviation, position_weight)

            if needs_rebalance and max_deviation > self.rebalance_threshold:
                logger.info(f"Portfolio rebalancing triggered: max deviation {max_deviation:.2%}")
                self.execute_rebalance()

        except Exception as e:
            logger.error(f"Error checking rebalance: {e}")

    def execute_rebalance(self):
        """Execute portfolio rebalancing"""
        try:
            logger.info("Executing portfolio rebalancing...")

            # Fetch current markets
            markets = self.fetch_markets(min_edge=0.01)
            if not markets:
                logger.warning("No markets available for rebalancing")
                return

            # Run optimization to get target allocations
            allocations, status, info = self.optimizer.optimize(markets, self.constraints, capital=self.capital)

            if status.value != "converged":
                logger.warning(f"Optimization did not converge: {status.value}")

            logger.info(f"Rebalancing optimization: {status.value} in {info.get('iterations', 0)} iterations")

            # Generate rebalancing orders (simplified)
            # In a full implementation, we'd calculate the difference between current and target
            # and generate orders to rebalance

            # For now, we'll log that rebalancing would occur
            target_positions = []
            for i, (market, alloc) in enumerate(zip(markets, allocations)):
                if alloc > 0.001:  # Minimum allocation threshold
                    target_size = alloc * self.capital
                    target_positions.append({
                        'market': market.question,
                        'token_id': market.token_id,
                        'target_size': target_size,
                        'allocation': alloc,
                        'edge': market.edge
                    })

            logger.info(f"Rebalance target: {len(target_positions)} positions")
            for pos in target_positions[:3]:  # Show first 3
                logger.info(f"  {pos['market'][:30]}...: ${pos['target_size']:.2f} ({pos['allocation']:.1%})")

            # In a full implementation, we would:
            # 1. Calculate current vs target positions
            # 2. Generate orders to close/open positions as needed
            # 3. Execute those orders

            logger.info("Rebalancing analysis complete (orders not executed in this simplified version)")

        except Exception as e:
            logger.error(f"Error executing rebalance: {e}")



