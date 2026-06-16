"""Risk management module"""
import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import logging

from .models import Market, PortfolioConstraints

logger = logging.getLogger(__name__)

class RiskManager:
    """Advanced risk management with stop-loss, correlation analysis, and stress testing"""

    def __init__(self, constraints: PortfolioConstraints):
        self.constraints = constraints
        self.positions = {}  # Current positions: {token_id: {size, entry_price, timestamp, ...}}
        self.stop_losses = {}  # Active stop-loss orders: {token_id: stop_price}
        self.price_history = {}  # Price history for correlation analysis: {token_id: [prices]}
        self.var_confidence = 0.95  # Value at Risk confidence level

    def update_position(self, token_id: str, size: float, price: float,
                       side: str = 'BUY', timestamp: Optional[datetime] = None,
                       category: str = 'default'):
        """Update or add a position"""
        if timestamp is None:
            timestamp = datetime.now()

        if token_id in self.positions:
            # Update existing position
            pos = self.positions[token_id]
            if side == 'BUY':
                # Average in
                total_size = pos['size'] + size
                if total_size > 0:
                    pos['entry_price'] = (pos['entry_price'] * pos['size'] + price * size) / total_size
                pos['size'] = total_size
                pos['category'] = category or pos.get('category', 'default')
            else:  # SELL
                # Reduce or close position
                pos['size'] -= size
                if pos['size'] <= 0:
                    # Position closed
                    del self.positions[token_id]
                    if token_id in self.stop_losses:
                        del self.stop_losses[token_id]
        else:
            # New position
            if size > 0:
                self.positions[token_id] = {
                    'size': size,
                    'entry_price': price,
                    'side': side,
                    'category': category or 'default',
                    'timestamp': timestamp,
                    'unrealized_pnl': 0.0,
                    'realized_pnl': 0.0
                }

    def update_price_history(self, token_id: str, price: float):
        """Update price history for correlation analysis"""
        if token_id not in self.price_history:
            self.price_history[token_id] = []

        self.price_history[token_id].append(price)

        # Keep only recent history (last 100 prices)
        if len(self.price_history[token_id]) > 100:
            self.price_history[token_id] = self.price_history[token_id][-100:]

    def estimate_slippage(self, size_usd: float, liquidity_usd: float,
                          base_spread: float = 0.0, impact_coef: float = 0.1) -> float:
        """
        Estimate marginal price slippage for a trade using a square-root market
        impact model.

            slippage = base_spread + impact_coef * sqrt(size_usd / liquidity_usd)

        `base_spread` defaults to 0 because the bot already buys at the ask (the
        spread is paid regardless of order size), so it is not part of the
        size-dependent slippage we can control by sizing down.
        """
        if liquidity_usd <= 0:
            return 1.0  # 100% impact if no liquidity
        if size_usd <= 0:
            return 0.0
        impact = base_spread + impact_coef * np.sqrt(size_usd / liquidity_usd)
        return min(impact, 0.99)  # Cap at 99%

    def max_size_within_slippage(self, liquidity_usd: float, tolerance: float,
                                 base_spread: float = 0.0,
                                 impact_coef: float = 0.1) -> float:
        """
        Largest order size (USD) whose marginal slippage stays within `tolerance`.

        Inverts the square-root model:
            tolerance = base_spread + impact_coef * sqrt(size / liquidity)
            => size = liquidity * ((tolerance - base_spread) / impact_coef)^2

        Returns 0.0 when the tolerance is below the base spread (i.e. even a
        dust-size order cannot avoid the spread cost) or there is no liquidity.
        """
        headroom = tolerance - base_spread
        if headroom <= 0 or liquidity_usd <= 0:
            return 0.0
        return liquidity_usd * (headroom / impact_coef) ** 2

    def check_trade_slippage_risk(self, token_id: str, size_usd: float, liquidity_usd: float,
                                  max_tolerance: float = 0.05) -> Dict:
        """
        Check whether a proposed trade stays within the marginal-slippage tolerance.

        The result carries a `suggested_size` -- the largest order that fits the
        tolerance -- so callers can shrink the order instead of dropping it.
        Dropping every order (the old behaviour) meant the bot never traded.
        """
        slippage_pct = self.estimate_slippage(size_usd, liquidity_usd)
        suggested_size = self.max_size_within_slippage(liquidity_usd, max_tolerance)

        if slippage_pct > max_tolerance:
            return {
                'safe': False,
                'type': 'slippage_limit',
                'token_id': token_id,
                'estimated_slippage': slippage_pct,
                'limit': max_tolerance,
                'suggested_size': suggested_size,
                'message': (f"Estimated slippage {slippage_pct:.1%} for {token_id} exceeds "
                            f"tolerance {max_tolerance:.1%}; downsize to <= ${suggested_size:.2f}")
            }

        return {
            'safe': True,
            'estimated_slippage': slippage_pct,
            'suggested_size': suggested_size,
        }

    def calculate_portfolio_value(self, current_prices: Dict[str, float],
                                  market_liquidities: Optional[Dict[str, float]] = None,
                                  apply_slippage: bool = False) -> float:
        """Calculate current portfolio value, optionally factoring in liquidation slippage"""
        total_value = 0.0
        for token_id, position in self.positions.items():
            if token_id in current_prices:
                current_price = current_prices[token_id]
                
                if position['side'] == 'BUY':
                    position_value = position['size'] * current_price
                    base_pnl = (current_price - position['entry_price']) * position['size']
                else:
                    position_value = position['size'] * max(0.0, 2 * position['entry_price'] - current_price)
                    base_pnl = (position['entry_price'] - current_price) * position['size']
                
                # Apply slippage discount if requested
                slippage_cost = 0.0
                if apply_slippage:
                    liquidity = market_liquidities.get(token_id, 0.0) if market_liquidities else 0.0
                    if liquidity > 0:
                        slippage_pct = self.estimate_slippage(position_value, liquidity)
                    else:
                        slippage_pct = 0.05  # Default 5% liquidation penalty if liquidity unknown
                    slippage_cost = position_value * slippage_pct
                
                position['unrealized_pnl'] = base_pnl - slippage_cost
                position['slippage_cost'] = slippage_cost
                total_value += (position_value - slippage_cost)
                
        return total_value

    def calculate_drawdown(self, peak_value: float, current_value: float) -> float:
        """Calculate current drawdown from peak"""
        if peak_value <= 0:
            return 0.0
        return (peak_value - current_value) / peak_value

    def check_stop_losses(self, current_prices: Dict[str, float]) -> List[Dict]:
        """Check and trigger stop-loss orders"""
        triggered_stops = []

        for token_id, stop_price in self.stop_losses.items():
            if token_id in current_prices:
                current_price = current_prices[token_id]

                # Check if stop-loss is triggered
                if current_price <= stop_price:
                    position = self.positions.get(token_id)
                    if position and position['size'] > 0:
                        triggered_stops.append({
                            'token_id': token_id,
                            'size': position['size'],
                            'stop_price': stop_price,
                            'current_price': current_price,
                            'reason': 'stop_loss_triggered',
                            'timestamp': datetime.now()
                        })

                        logger.warning(f"Stop-loss triggered for {token_id}: "
                                     f"size={position['size']}, stop={stop_price:.4f}, "
                                     f"current={current_price:.4f}")

        return triggered_stops

    def set_stop_loss(self, token_id: str, stop_price: float,
                     stop_type: str = 'percent', reference_price: Optional[float] = None):
        """Set a stop-loss order for a position"""
        if token_id not in self.positions:
            logger.warning(f"Cannot set stop-loss for {token_id}: no position found")
            return False

        position = self.positions[token_id]

        if stop_type == 'percent':
            if reference_price is None:
                reference_price = position['entry_price']
            stop_price = reference_price * (1 - stop_price)  # Assume stop_price is percentage like 0.05 for 5%
        elif stop_type == 'trailing':
            # Trailing stop based on highest price since entry
            if token_id in self.price_history and len(self.price_history[token_id]) > 0:
                highest_price = max(self.price_history[token_id])
                stop_price = highest_price * (1 - stop_price)

        self.stop_losses[token_id] = stop_price
        logger.info(f"Set stop-loss for {token_id}: {stop_price:.4f} (type: {stop_type})")
        return True

    def calculate_correlation_matrix(self, lookback: int = 30) -> Tuple[np.ndarray, List[str]]:
        """Calculate correlation matrix between positions"""
        tokens = list(self.price_history.keys())
        if len(tokens) < 2:
            return np.array([]), tokens

        # Prepare returns data
        returns_data = []
        min_length = float('inf')

        for token in tokens:
            prices = self.price_history[token]
            if len(prices) < 2:
                continue

            # Calculate returns
            returns = np.diff(prices) / prices[:-1]
            returns_data.append(returns)
            min_length = min(min_length, len(returns))

        if len(returns_data) < 2:
            return np.array([]), tokens

        # Trim to same length
        trimmed_returns = [r[-min_length:] for r in returns_data if len(r) >= min_length]

        if len(trimmed_returns) < 2:
            return np.array([]), [tokens[i] for i, r in enumerate(returns_data) if len(r) >= min_length]

        # Calculate correlation matrix
        returns_matrix = np.array(trimmed_returns)
        correlation_matrix = np.corrcoef(returns_matrix)

        # Handle NaN values (replace with 0)
        correlation_matrix = np.nan_to_num(correlation_matrix, nan=0.0)

        return correlation_matrix, [tokens[i] for i, r in enumerate(returns_data) if len(r) >= min_length]

    def get_high_correlation_pairs(self, threshold: float = 0.7, lookback: int = 30) -> List[Tuple[str, str, float]]:
        """Find pairs of positions with high correlation"""
        corr_matrix, tokens = self.calculate_correlation_matrix(lookback)

        if len(corr_matrix) == 0 or len(tokens) == 0:
            return []

        high_corr_pairs = []
        n = len(tokens)

        for i in range(n):
            for j in range(i+1, n):
                corr = abs(corr_matrix[i, j])  # Use absolute correlation
                if corr >= threshold:
                    high_corr_pairs.append((tokens[i], tokens[j], corr))

        return sorted(high_corr_pairs, key=lambda x: x[2], reverse=True)

    def calculate_portfolio_var(self, current_prices: Dict[str, float],
                               confidence: float = None, lookback: int = 25) -> float:
        """Calculate Value at Risk (VaR) for the portfolio"""
        if confidence is None:
            confidence = self.var_confidence

        if len(self.positions) == 0:
            return 0.0

        # Get returns history
        tokens = list(self.positions.keys())
        if len(tokens) == 0:
            return 0.0

        returns_data = []
        for token in tokens:
            if token in self.price_history and len(self.price_history[token]) > 1:
                prices = self.price_history[token]
                returns = np.diff(prices) / prices[:-1]
                # Use most recent data
                if len(returns) > lookback:
                    returns = returns[-lookback:]
                returns_data.append(returns)

        if len(returns_data) == 0:
            return 0.0

        # Ensure same length
        min_length = min(len(r) for r in returns_data)
        if min_length < 10:  # Need minimum data
            return 0.0

        trimmed_returns = [r[-min_length:] for r in returns_data]
        returns_matrix = np.array(trimmed_returns)

        # Calculate portfolio weights based on current positions
        weights = []
        total_value = 0.0
        position_values = {}

        for token in tokens:
            if token in current_prices and token in self.positions:
                position_value = abs(self.positions[token]['size']) * current_prices[token]
                position_values[token] = position_value
                total_value += position_value

        if total_value == 0:
            return 0.0

        for token in tokens:
            if token in position_values:
                weights.append(position_values[token] / total_value)
            else:
                weights.append(0.0)

        weights = np.array(weights)
        weights = weights / np.sum(weights)  # Normalize

        # Calculate portfolio returns
        if returns_matrix.shape[0] == len(weights):
            portfolio_returns = np.dot(weights, returns_matrix)

            # Calculate VaR using historical simulation
            sorted_returns = np.sort(portfolio_returns)
            var_index = int((1 - confidence) * len(sorted_returns))
            if var_index < len(sorted_returns):
                var = -sorted_returns[var_index]  # VaR is positive number representing loss
                return var * total_value  # Return in dollar terms

        return 0.0

    def stress_test_scenario(self, current_prices: Dict[str, float],
                           scenario_shocks: Dict[str, float]) -> Dict:
        """
        Stress test portfolio under various scenarios

        Args:
            current_prices: Current prices for each token
            scenario_shocks: Expected price changes as percentages (e.g., {'TOKEN1': -0.1 for 10% drop)

        Returns:
            Dictionary with stress test results
        """
        base_value = self.calculate_portfolio_value(current_prices)

        stressed_value = 0.0
        position_impacts = {}

        for token_id, position in self.positions.items():
            base_price = current_prices.get(token_id, 0)
            if base_price <= 0:
                continue

            shock = scenario_shocks.get(token_id, 0.0)
            stressed_price = base_price * (1 + shock)

            if position['side'] == 'BUY':
                position_value = position['size'] * stressed_price
                base_position_value = position['size'] * base_price
                pnl_impact = position_value - base_position_value
            else:  # SELL
                # Simplified for short positions
                position_value = position['size'] * (2 * base_price - stressed_price)
                base_position_value = position['size'] * base_price
                pnl_impact = base_position_value - position_value

            position_impacts[token_id] = {
                'base_value': base_position_value,
                'stressed_value': position_value,
                'pnl_impact': pnl_impact,
                'shock_applied': shock
            }

            stressed_value += position_value

        total_pnl_impact = stressed_value - base_value
        total_pnl_percent = (total_pnl_impact / base_value) if base_value > 0 else 0

        return {
            'base_portfolio_value': base_value,
            'stressed_portfolio_value': stressed_value,
            'total_pnl_impact': total_pnl_impact,
            'total_pnl_percent': total_pnl_percent,
            'position_impacts': position_impacts,
            'scenario_applied': scenario_shocks
        }

    def check_risk_limits(self, current_prices: Dict[str, float],
                         portfolio_value: float) -> List[Dict]:
        """Check if current portfolio violates any risk limits"""
        violations = []

        if portfolio_value <= 0:
            return violations

        # Check total exposure
        total_exposure = 0.0
        for token_id, position in self.positions.items():
            if token_id in current_prices:
                position_value = abs(position['size']) * current_prices[token_id]
                total_exposure += position_value

        exposure_ratio = total_exposure / portfolio_value
        if exposure_ratio > self.constraints.max_total_exposure:
            violations.append({
                'type': 'total_exposure',
                'current': exposure_ratio,
                'limit': self.constraints.max_total_exposure,
                'message': f"Total exposure {exposure_ratio:.1%} exceeds limit {self.constraints.max_total_exposure:.1%}"
            })

        # Check individual position limits
        for token_id, position in self.positions.items():
            if token_id in current_prices:
                position_value = abs(position['size']) * current_prices[token_id]
                position_ratio = position_value / portfolio_value
                if position_ratio > self.constraints.max_single_position:
                    violations.append({
                        'type': 'single_position',
                        'token_id': token_id,
                        'current': position_ratio,
                        'limit': self.constraints.max_single_position,
                        'message': f"Position {token_id} {position_ratio:.1%} exceeds limit {self.constraints.max_single_position:.1%}"
                    })

        # Check category limits
        category_exposure = {}
        for token_id, position in self.positions.items():
            if token_id in current_prices:
                category = position.get("category", "default")
                position_value = abs(position['size']) * current_prices[token_id]
                if category not in category_exposure:
                    category_exposure[category] = 0.0
                category_exposure[category] += position_value

        for category, exposure in category_exposure.items():
            category_ratio = exposure / portfolio_value
            limit = self.constraints.max_category_exposure.get(category,
                                                             self.constraints.max_category_exposure.get('default', 0.25))
            if category_ratio > limit:
                violations.append({
                    'type': 'category_exposure',
                    'category': category,
                    'current': category_ratio,
                    'limit': limit,
                    'message': f"Category {category} exposure {category_ratio:.1%} exceeds limit {limit:.1%}"
                })

        # Check drawdown limit (would need peak value tracking)
        # This is simplified - in practice you'd track portfolio peak value

        return violations

    def get_risk_summary(self, current_prices: Dict[str, float]) -> Dict:
        """Get a comprehensive risk summary"""
        if len(self.positions) == 0:
            return {
                'total_positions': 0,
                'portfolio_value': 0.0,
                'total_exposure': 0.0,
                'exposure_ratio': 0.0,
                'var_95': 0.0,
                'max_correlation': 0.0,
                'stop_losses_set': 0,
                'risk_violations': []
            }

        portfolio_value = self.calculate_portfolio_value(current_prices)
        total_exposure = sum(abs(pos['size']) * current_prices.get(tid, 0)
                           for tid, pos in self.positions.items() if tid in current_prices)
        exposure_ratio = total_exposure / portfolio_value if portfolio_value > 0 else 0

        var_95 = self.calculate_portfolio_var(current_prices, confidence=0.95)
        corr_matrix, tokens = self.calculate_correlation_matrix()
        max_correlation = 0.0
        if len(corr_matrix) > 0:
            # Get maximum absolute correlation excluding diagonal (self-correlation)
            mask = ~np.eye(len(corr_matrix), dtype=bool)
            if np.any(mask):
                max_correlation = np.max(np.abs(corr_matrix[mask]))

        risk_violations = self.check_risk_limits(current_prices, portfolio_value)

        return {
            'total_positions': len(self.positions),
            'portfolio_value': portfolio_value,
            'total_exposure': total_exposure,
            'exposure_ratio': exposure_ratio,
            'var_95': var_95,
            'max_correlation': max_correlation,
            'stop_losses_set': len(self.stop_losses),
            'risk_violations': risk_violations,
            'high_correlation_pairs': self.get_high_correlation_pairs(threshold=0.7)
        }

class StopLossManager:
    """Manages stop-loss orders and trailing stops"""

    def __init__(self, risk_manager: RiskManager):
        self.risk_manager = risk_manager
        self.trailing_stops = {}  # {token_id: {trailing_percent, high_price}}

    def set_trailing_stop(self, token_id: str, trailing_percent: float):
        """Set a trailing stop-loss"""
        if token_id not in self.risk_manager.positions:
            logger.warning(f"Cannot set trailing stop for {token_id}: no position")
            return False

        position = self.risk_manager.positions[token_id]
        entry_price = position['entry_price']

        self.trailing_stops[token_id] = {
            'trailing_percent': trailing_percent,
            'high_price': entry_price,
            'activation_price': entry_price * (1 + trailing_percent)  # For long positions
        }

        # Set initial stop-loss
        initial_stop = entry_price * (1 - trailing_percent)
        self.risk_manager.set_stop_loss(token_id, initial_stop, 'percent')

        logger.info(f"Set trailing stop for {token_id}: {trailing_percent:.1%}")
        return True

    def update_trailing_stops(self, current_prices: Dict[str, float]):
        """Update trailing stops based on current prices"""
        for token_id, trail_data in self.trailing_stops.items():
            if token_id not in current_prices or token_id not in self.risk_manager.positions:
                continue

            current_price = current_prices[token_id]
            position = self.risk_manager.positions[token_id]

            # Update high price for long positions
            if position['side'] == 'BUY' and current_price > trail_data['high_price']:
                trail_data['high_price'] = current_price
                # Update stop-loss to trail behind the high price
                new_stop = trail_data['high_price'] * (1 - trail_data['trailing_percent'])
                current_stop = self.risk_manager.stop_losses.get(token_id, 0)

                # Only move stop-loss up (never down for longs)
                if new_stop > current_stop:
                    self.risk_manager.set_stop_loss(token_id, new_stop, 'percent')
                    logger.debug(f"Updated trailing stop for {token_id}: {new_stop:.4f}")

            # For short positions, you'd update low price and trail stop up
            # Simplified for now - assuming mostly long positions
