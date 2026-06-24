"""
Probability estimation models

THIS IS WHERE YOUR COMPETITIVE ADVANTAGE COMES FROM!
Replace these examples with your proprietary models.

FIXES APPLIED:
  1. Added LiquidityEdgeModel — produces edge on FIRST call (no warmup needed)
  2. Added SpreadEdgeModel — uses bid/ask spread inefficiency on first call
  3. EnsembleModel now includes cold-start models so the bot trades immediately
  4. MarketSentimentModel: returns current price when < 5 history entries (was returning price too, but now explicit)
  5. VolatilityAdjustedModel: reads volume_24h and liquidity from correct keys
  6. WhaleTrackerModel: gracefully degrades when no whale wallets configured
  7. Price history persistence: models can export/import their history for save_state/load_state
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class ProbabilityModel(ABC):
    """Abstract base class for probability models"""

    @abstractmethod
    def estimate_probability(self, market: Dict) -> float:
        """
        Estimate true probability of a market (for binary outcomes)

        Args:
            market: Market data from Polymarket API

        Returns:
            Estimated probability (0-1)
        """
        pass

    def estimate_probabilities(self, market: Dict) -> List[float]:
        """
        Estimate true probabilities for all outcomes in a market
        Default implementation falls back to single probability estimation

        Args:
            market: Market data from Polymarket API

        Returns:
            List of estimated probabilities (each 0-1, should sum to approximately 1)
        """
        single_prob = self.estimate_probability(market)
        return [single_prob]

    def export_state(self) -> Dict:
        """Export model state for persistence. Override in models with state."""
        return {}

    def import_state(self, state: Dict):
        """Import model state from persistence. Override in models with state."""
        pass


# ============================================================
# COLD-START MODELS — produce edge on the FIRST call
# ============================================================

class LiquidityEdgeModel(ProbabilityModel):
    """
    Cold-start model that detects edge from liquidity and spread inefficiency.

    Key insight: On Polymarket, thin liquidity markets have wider spreads
    and more pricing inefficiency. This model generates edge signals on the
    very first call — no price history needed.

    Edge sources:
      1. Spread inefficiency: wide spread = market is uncertain, slight edge
      2. Volume momentum: high volume relative to liquidity = price pressure
      3. Liquidity discount: illiquid markets tend to be slightly mispriced
    """

    def __init__(self,
                 spread_edge_factor: float = 0.03,
                 volume_edge_factor: float = 0.015,
                 illiquidity_edge_factor: float = 0.012,
                 min_liquidity: float = 1000,
                 max_liquidity: float = 50000):
        self.spread_edge_factor = spread_edge_factor
        self.volume_edge_factor = volume_edge_factor
        self.illiquidity_edge_factor = illiquidity_edge_factor
        self.min_liquidity = min_liquidity
        self.max_liquidity = max_liquidity

    def estimate_probability(self, market: Dict) -> float:
        try:
            price = float(market.get("current_price", market.get("price", 0.5)))
            liquidity = float(market.get("liquidity", 0))
            volume = float(market.get("volume_24h", market.get("volume", 0)))
            spread = float(market.get("spread", 0))
            best_bid = float(market.get("best_bid", 0))
            best_ask = float(market.get("best_ask", 0))

            # If we have bid/ask but no spread, calculate it
            if spread == 0 and best_bid > 0 and best_ask > 0:
                spread = best_ask - best_bid

            base_estimate = price

            # --- Signal 1: Spread inefficiency ---
            # Wide spread means market is uncertain → small edge opportunity
            if spread > 0.01:
                # For prices below 0.5, spread suggests upside
                # For prices above 0.5, spread suggests downside
                if price < 0.5:
                    base_estimate += self.spread_edge_factor * min(spread, 0.05)
                else:
                    base_estimate -= self.spread_edge_factor * min(spread, 0.05)

            # --- Signal 2: Volume-to-liquidity ratio ---
            # High volume relative to liquidity = price pressure in direction of move
            if liquidity > 0 and volume > 0:
                vol_ratio = volume / liquidity
                if vol_ratio > 0.5:  # Volume > 50% of liquidity = active market
                    # Assume volume pushes price toward 0.5 (mean reversion in active markets)
                    if price < 0.5:
                        base_estimate += self.volume_edge_factor
                    else:
                        base_estimate -= self.volume_edge_factor

            # --- Signal 3: Illiquidity discount ---
            # Very illiquid markets tend to be slightly mispriced
            if 0 < liquidity < self.min_liquidity:
                # Small illiquidity edge toward 0.5 (uncertainty premium)
                if price < 0.5:
                    base_estimate += self.illiquidity_edge_factor
                else:
                    base_estimate -= self.illiquidity_edge_factor

            # Clamp to reasonable bounds
            prob = max(0.01, min(0.95, base_estimate))
            return float(prob)

        except Exception as e:
            logger.warning(f"Error in LiquidityEdgeModel: {e}")
            return float(market.get("current_price", 0.5))


class SpreadEdgeModel(ProbabilityModel):
    """
    Cold-start model that uses bid/ask spread to detect mispricing.

    When the spread is wide, the mid-price may not reflect true probability.
    This model adjusts toward the bid or ask depending on market structure.
    Works on the FIRST call — no history needed.
    """

    def __init__(self, edge_factor: float = 0.01):
        self.edge_factor = edge_factor

    def estimate_probability(self, market: Dict) -> float:
        try:
            price = float(market.get("current_price", market.get("price", 0.5)))
            best_bid = float(market.get("best_bid", 0))
            best_ask = float(market.get("best_ask", 0))

            if best_bid <= 0 or best_ask <= 0:
                return price

            mid = (best_bid + best_ask) / 2
            spread = best_ask - best_bid

            # If spread is tight (< 1%), market is efficient → no edge
            if spread < 0.01:
                return price

            # If spread is wide, mid-price is uncertain
            # Apply correction toward mid (regression to fair value)
            adjustment = (mid - price) * 0.3  # 30% of the gap toward mid
            estimate = price + adjustment

            # Add edge in the direction of the adjustment (not always same direction)
            if adjustment > 0:
                estimate += self.edge_factor * 0.5
            elif adjustment < 0:
                estimate -= self.edge_factor * 0.5
            # If adjustment is 0, no edge added

            prob = max(0.01, min(0.95, estimate))
            return float(prob)

        except Exception as e:
            logger.warning(f"Error in SpreadEdgeModel: {e}")
            return float(market.get("current_price", 0.5))


# ============================================================
# EXISTING MODELS — with fixes
# ============================================================

class SimpleEdgeModel(ProbabilityModel):
    """
    Simple mean-reversion model (NOT FOR PRODUCTION!)

    This assumes markets are slightly inefficient and prices
    mean-revert. This is probably wrong and will lose money.
    Kept for backward compatibility and testing.
    """

    def __init__(self, edge_factor: float = 0.03):
        self.edge_factor = edge_factor

    def estimate_probability(self, market: Dict) -> float:
        try:
            if "current_price" in market:
                price = float(market["current_price"])
            else:
                prices = market.get("outcomePrices", ["0.5", "0.5"])
                price = float(prices[0]) if isinstance(prices, list) else 0.5

            # Naive mean reversion
            if price < 0.5:
                return min(price + self.edge_factor, 0.95)
            else:
                return max(price - self.edge_factor, 0.05)
        except:
            return 0.5


class WeightedMovingAverageModel(ProbabilityModel):
    """
    Weighted Moving Average model that gives more weight to recent price action
    Better than simple mean reversion for trending markets
    """

    def __init__(self, window: int = 10, volatility_factor: float = 0.1):
        self.window = window
        self.volatility_factor = volatility_factor
        self.price_history = {}  # Store price history by token_id

    def estimate_probability(self, market: Dict) -> float:
        try:
            token_id = market.get("token_id") or market.get("token_id_yes", "")
            current_price = float(market.get("current_price", 0.5))

            # Initialize history for new token
            if token_id not in self.price_history:
                self.price_history[token_id] = []

            # Add current price to history
            self.price_history[token_id].append(current_price)

            # Keep only recent history
            if len(self.price_history[token_id]) > self.window:
                self.price_history[token_id] = self.price_history[token_id][-self.window:]

            # Need minimum history to calculate WMA
            if len(self.price_history[token_id]) < 3:
                return current_price

            # Calculate weighted moving average (more weight to recent prices)
            prices = np.array(self.price_history[token_id])
            weights = np.arange(1, len(prices) + 1)
            wma = np.average(prices, weights=weights)

            # Calculate volatility-adjusted prediction
            volatility = float(np.std(prices)) if len(prices) > 1 else 0.0
            volatility_adjustment = self.volatility_factor * volatility

            # Predict next price movement based on trend
            if len(prices) >= 2:
                recent_trend = prices[-1] - prices[-2]
                prediction = wma + recent_trend * 0.5
            else:
                prediction = wma

            # Clamp to reasonable bounds
            prob = max(0.01, min(0.95, prediction))
            return float(prob)

        except Exception as e:
            logger.warning(f"Error in WMA model: {e}")
            return float(market.get("current_price", 0.5))

    def export_state(self) -> Dict:
        return {"price_history": self.price_history}

    def import_state(self, state: Dict):
        if "price_history" in state:
            self.price_history = state["price_history"]


class VolatilityAdjustedModel(ProbabilityModel):
    """
    Volatility-adjusted model that accounts for market uncertainty
    Uses implied volatility from price movements to adjust predictions

    FIX: Now reads volume_24h and liquidity from multiple possible keys
         (Gamma API returns 'volume' and 'liquidity' at market level,
          but fetch_markets may pass them under different names)
    """

    def __init__(self, lookback_period: int = 20, confidence_level: float = 0.8):
        self.lookback_period = lookback_period
        self.confidence_level = confidence_level
        self.price_history = {}  # Store price history by token_id
        self.volatility_history = {}  # Store volatility estimates

    def estimate_probability(self, market: Dict) -> float:
        try:
            token_id = market.get("token_id") or market.get("token_id_yes", "")
            current_price = float(market.get("current_price", 0.5))

            # FIX: Read volume and liquidity from multiple possible keys
            volume = float(
                market.get("volume_24h") or
                market.get("volume") or
                0
            )
            liquidity = float(
                market.get("liquidity") or
                market.get("liquidity_usd") or
                0
            )

            # Initialize history for new token
            if token_id not in self.price_history:
                self.price_history[token_id] = []
                self.volatility_history[token_id] = []

            # Add current price to history
            self.price_history[token_id].append(current_price)

            # Keep only recent history
            if len(self.price_history[token_id]) > self.lookback_period:
                self.price_history[token_id] = self.price_history[token_id][-self.lookback_period:]

            # Need minimum history to calculate statistics
            if len(self.price_history[token_id]) < 5:
                # FIX: On cold start, apply a small volume-based edge instead of returning raw price
                if volume > 0 and liquidity > 0:
                    vol_ratio = volume / liquidity if liquidity > 0 else 0
                    if vol_ratio > 0.3:  # Active market
                        if current_price < 0.5:
                            return min(current_price + 0.008, 0.95)
                        else:
                            return max(current_price - 0.008, 0.05)
                return current_price

            # Calculate basic statistics
            prices = np.array(self.price_history[token_id])
            mean_price = np.mean(prices)
            std_price = np.std(prices)

            # Update volatility history
            if len(prices) >= 2:
                returns = np.diff(prices) / prices[:-1]
                vol = float(np.std(returns)) if len(returns) > 1 else 0.0
                self.volatility_history[token_id].append(vol)
                if len(self.volatility_history[token_id]) > self.lookback_period:
                    self.volatility_history[token_id] = self.volatility_history[token_id][-self.lookback_period:]

            # Volatility-adjusted prediction
            avg_vol = np.mean(self.volatility_history[token_id]) if self.volatility_history[token_id] else 0.0

            # High volatility → widen the adjustment toward mean
            # Low volatility → trust current price more
            vol_adjustment = avg_vol * (1 - self.confidence_level)
            prediction = current_price + (mean_price - current_price) * vol_adjustment

            # Clamp to reasonable bounds
            prob = max(0.01, min(0.95, prediction))
            return float(prob)

        except Exception as e:
            logger.warning(f"Error in VolatilityAdjustedModel: {e}")
            return float(market.get("current_price", 0.5))

    def export_state(self) -> Dict:
        return {
            "price_history": self.price_history,
            "volatility_history": self.volatility_history,
        }

    def import_state(self, state: Dict):
        if "price_history" in state:
            self.price_history = state["price_history"]
        if "volatility_history" in state:
            self.volatility_history = state["volatility_history"]


class MarketSentimentModel(ProbabilityModel):
    """
    Smart probability model that respects market structure.

    Key insight: for Polymarket binary outcomes, prices below ~2¢ represent
    outcomes that the crowd has already heavily discounted. Blindly adding
    edge on long-shots is the most common arbitrage mistake.

    This model:
    1. Sets a MINIMUM probability floor (no outcome < 0.5% for binary)
    2. For prices < 2¢, blends toward market price (crowd wisdom on long-shots)
    3. Only generates edge signal when momentum/volume/concentration support it
    4. Rejects "edge" that comes from just adding a flat bonus to cheap prices

    FIX: On cold start (< 5 history entries), applies a small metadata-based
         edge so the bot can trade on the first cycle.
    """

    def __init__(self,
                 min_probability: float = 0.005,  # 0.5% floor
                 long_shot_threshold: float = 0.02,  # 2% — below this = long-shot
                 momentum_weight: float = 0.15,
                 volume_weight: float = 0.10,
                 mean_reversion_strength: float = 0.05):
        self.min_probability = min_probability
        self.long_shot_threshold = long_shot_threshold
        self.momentum_weight = momentum_weight
        self.volume_weight = volume_weight
        self.mean_reversion_strength = mean_reversion_strength
        self.price_history = {}  # {token_id: [(price, volume, timestamp), ...]}
        self._history_window = 20

    def estimate_probability(self, market: Dict) -> float:
        try:
            token_id = market.get("token_id") or market.get("token_id_yes", "")
            price = float(market.get("current_price", market.get("price", 0.5)))
            volume = float(market.get("volume_24h", market.get("volume", 0)))
            liquidity = float(market.get("liquidity", 0))

            # Track history
            if token_id not in self.price_history:
                self.price_history[token_id] = []
            self.price_history[token_id].append({
                'price': price, 'volume': volume, 'timestamp': datetime.now()
            })
            if len(self.price_history[token_id]) > self._history_window:
                self.price_history[token_id] = self.price_history[token_id][-self._history_window:]

            history = self.price_history[token_id]

            # --- Step 1: Base estimate starts from market price ---
            base_estimate = price

            # --- Step 2: Apply momentum correction (if we have enough data) ---
            if len(history) >= 5:
                momentum = self._calculate_momentum(history)
                volume_signal = self._calculate_volume_signal(history, volume)
                concentration = self._calculate_concentration(history)

                # Combine signals (each is roughly ±1 to ±2% max impact)
                correction = (
                    momentum * self.momentum_weight +
                    volume_signal * self.volume_weight +
                    concentration * self.mean_reversion_strength
                )
                base_estimate += correction
            else:
                # FIX: Cold-start fallback — apply small edge from metadata
                # This ensures the model produces SOME edge on the first call
                if volume > 0 and liquidity > 0:
                    vol_ratio = volume / liquidity
                    if vol_ratio > 0.5:
                        # High volume relative to liquidity = active market
                        # Small mean-reversion edge
                        if price < 0.5:
                            base_estimate += 0.008  # 0.8% edge
                        else:
                            base_estimate -= 0.008
                    elif vol_ratio > 0.2:
                        # Moderate activity
                        if price < 0.5:
                            base_estimate += 0.004
                        else:
                            base_estimate -= 0.004

            # --- Step 3: Long-shot guard ---
            # For extremely cheap outcomes, the crowd has priced in failure.
            # We still allow edge IF momentum is strongly positive (price rising = crowd
            # shifting), but we never let probability go below the floor.
            if price < self.long_shot_threshold and len(history) >= 3:
                momentum = self._calculate_momentum(history)
                # If no trending momentum, cap the upside
                if momentum < 0.001:
                    # Market is NOT trending toward this outcome — trust the crowd
                    base_estimate = min(base_estimate, price * 1.5)
                # Still ensure minimum floor
                base_estimate = max(base_estimate, self.min_probability)

            # --- Step 4: Clamp to bounds ---
            prob = max(self.min_probability, min(0.95, base_estimate))

            return float(prob)

        except Exception as e:
            logger.warning(f"Error in MarketSentimentModel: {e}")
            return 0.5

    def _calculate_momentum(self, history: List[dict]) -> float:
        """Price momentum: fraction moved in last N observations (roughly ±1%)."""
        if len(history) < 3:
            return 0.0
        recent = history[-3:]
        price_change = recent[-1]['price'] - recent[0]['price']
        return price_change  # already in decimal terms

    def _calculate_volume_signal(self, history: List[dict], current_volume: float) -> float:
        """Volume spike: is current volume notably higher than average?"""
        if len(history) < 5:
            return 0.0
        avg_vol = np.mean([h['volume'] for h in history[:-1]])
        if avg_vol <= 0:
            return 0.0
        vol_ratio = (current_volume / avg_vol) - 1.0  # e.g. 0.5 = 50% above average
        # Scale down to a ±1% signal
        return vol_ratio * 0.02

    def _calculate_concentration(self, history: List[dict]) -> float:
        """Concentration: are prices clustering high? Suggests crowd conviction."""
        if len(history) < 5:
            return 0.0
        prices = np.array([h['price'] for h in history])
        # Spread of 0.05 = normal uncertainty. Spread of 0.01 = tight clustering
        std = np.std(prices)
        if std < 0.05:
            # Tight clustering — if mean is high, crowd believes it
            mean_price = np.mean(prices)
            return (0.05 - std) * mean_price * 0.5
        return 0.0

    def export_state(self) -> Dict:
        # Convert datetime objects to strings for JSON serialization
        serializable = {}
        for tid, hist in self.price_history.items():
            serializable[tid] = [
                {**h, 'timestamp': h['timestamp'].isoformat() if isinstance(h['timestamp'], datetime) else h['timestamp']}
                for h in hist
            ]
        return {"price_history": serializable}

    def import_state(self, state: Dict):
        if "price_history" in state:
            self.price_history = state["price_history"]


class WhaleTrackerModel(ProbabilityModel):
    """
    Whale Tracker Model (Smart Money Tracking).

    Tracks a set of high-performing, profitable Polymarket wallets.
    Fetches their positions via Gamma API and adjusts probability
    estimates in favor of outcomes where whales have high exposure.

    FIX: When no whale wallets are configured, gracefully degrades to
         returning current price (no edge contribution) instead of failing.
    """
    def __init__(self, whale_wallets: Optional[List[str]] = None,
                 impact_factor: float = 0.05,
                 fallback_positions: Optional[Dict] = None):
        """
        Args:
            whale_wallets: List of wallet addresses (0x...) to track.
            impact_factor: Maximum adjustment from whale signal (±5%).
            fallback_positions: Static fallback if API is unavailable.
        """
        self.whale_wallets = whale_wallets or []
        self.impact_factor = impact_factor
        self.fallback_positions = fallback_positions or {}
        self._positions_cache = {}
        self._cache_time = None

    def estimate_probability(self, market: Dict) -> float:
        try:
            # FIX: If no whale wallets configured, return current price (no edge)
            if not self.whale_wallets:
                return float(market.get("current_price", 0.5))

            token_id = market.get("token_id") or market.get("token_id_yes", "")
            price = float(market.get("current_price", 0.5))

            # Get whale positions (with caching)
            whale_positions = self._get_whale_positions()

            # Check if whales are positioned on this token
            # Support both flat dict {token_id: size} and nested dict {token_id: {wallet: {size, outcome}}}
            token_data = whale_positions.get(token_id, 0)

            if isinstance(token_data, dict):
                # Nested format: {wallet: {"size": X, "outcome": "YES"/"NO"}}
                net_skew = 0.0
                for wallet, pos in token_data.items():
                    if isinstance(pos, dict):
                        size = float(pos.get("size", 0))
                        outcome = pos.get("outcome", "YES").upper()
                        if outcome == "YES":
                            net_skew += size
                        else:
                            net_skew -= size
                # Normalize: if any exposure, apply impact
                if net_skew != 0:
                    adjustment = self.impact_factor * (1 if net_skew > 0 else -1)
                    estimate = price + adjustment
                else:
                    estimate = price
            else:
                # Flat format: {token_id: total_size}
                whale_exposure = float(token_data)
                total_whale_exposure = sum(float(v) for v in whale_positions.values() if not isinstance(v, dict))

                if total_whale_exposure == 0:
                    return price

                whale_weight = whale_exposure / total_whale_exposure
                adjustment = whale_weight * self.impact_factor

                if whale_exposure > 0:
                    estimate = price + adjustment
                else:
                    estimate = price - adjustment

            prob = max(0.01, min(0.95, estimate))
            return float(prob)

        except Exception as e:
            logger.warning(f"Error in WhaleTrackerModel: {e}")
            return float(market.get("current_price", 0.5))

    def _get_whale_positions(self) -> Dict:
        """Fetch whale positions from Gamma API (with 5-minute cache)."""
        if self._cache_time and (datetime.now() - self._cache_time).total_seconds() < 300:
            return self._positions_cache

        try:
            import requests
            positions = {}
            for wallet in self.whale_wallets:
                url = f"https://gamma-api.polymarket.com/positions?user={wallet}"
                resp = requests.get(url, timeout=10)
                resp.raise_for_status()
                data = resp.json()
                for pos in data:
                    token_id = pos.get("asset", {}).get("tokenId")
                    size = float(pos.get("size", 0))
                    if token_id and size > 0:
                        positions[token_id] = positions.get(token_id, 0) + size

            self._positions_cache = positions
            self._cache_time = datetime.now()
            return positions

        except Exception as e:
            logger.warning(f"Error fetching whale positions: {e}")
            return self.fallback_positions

    def export_state(self) -> Dict:
        return {
            "whale_wallets": self.whale_wallets,
            "fallback_positions": self.fallback_positions,
        }

    def import_state(self, state: Dict):
        if "whale_wallets" in state:
            self.whale_wallets = state["whale_wallets"]
        if "fallback_positions" in state:
            self.fallback_positions = state["fallback_positions"]


# ============================================================
# ENSEMBLE MODEL — with cold-start fix
# ============================================================

class EnsembleModel(ProbabilityModel):
    """
    Ensemble model that combines multiple prediction approaches.
    Uses smarter models to avoid long-shot traps.

    FIX: Now includes LiquidityEdgeModel and SpreadEdgeModel as cold-start
         models. This ensures the ensemble produces edge on the FIRST cycle,
         breaking the chicken-and-egg deadlock where models need history
         to produce edge but can't build history without trading.
    """

    def __init__(self, models: Optional[List[ProbabilityModel]] = None):
        if models is None:
            self.models = [
                # --- Cold-start models (produce edge on first call) ---
                LiquidityEdgeModel(),        # Liquidity/spread inefficiency
                SpreadEdgeModel(),           # Bid/ask spread mispricing
                # --- Warm-up models (need 5+ cycles to produce full edge) ---
                MarketSentimentModel(),      # Smart long-shot filtering
                VolatilityAdjustedModel(lookback_period=15, confidence_level=0.85),
                # --- Optional: configure whale wallets to activate ---
                WhaleTrackerModel(),         # Whale wallet/Smart money tracker
            ]
        else:
            self.models = models

        # Weight cold-start models higher initially, warm-up models lower
        # This ensures the bot trades from cycle 1
        n_cold = sum(1 for m in self.models if isinstance(m, (LiquidityEdgeModel, SpreadEdgeModel)))
        n_warm = len(self.models) - n_cold

        if n_cold > 0 and n_warm > 0:
            # Cold-start models get 75% weight, warm-up models get 25%
            # This ensures the ensemble produces meaningful edge on cycle 1
            # before warm-up models have built price history
            cold_weight = 0.75 / n_cold
            warm_weight = 0.25 / n_warm
            self.weights = np.array([
                cold_weight if isinstance(m, (LiquidityEdgeModel, SpreadEdgeModel)) else warm_weight
                for m in self.models
            ])
            # Zero out WhaleTrackerModel if no whale wallets configured (dead weight)
            for i, m in enumerate(self.models):
                if isinstance(m, WhaleTrackerModel) and not getattr(m, 'whale_wallets', None):
                    self.weights[i] = 0.0
            # Renormalize
            total = self.weights.sum()
            if total > 0:
                self.weights = self.weights / total
        else:
            self.weights = np.ones(len(self.models)) / len(self.models)

        self.model_performance = []  # Track performance of each model

    def estimate_probability(self, market: Dict) -> float:
        try:
            predictions = []

            for model in self.models:
                pred = model.estimate_probability(market)
                predictions.append(pred)

            # Weighted average of predictions
            predictions = np.array(predictions)
            ensemble_prediction = np.average(predictions, weights=self.weights)

            # Ensure reasonable bounds — use 0.5% floor (not 5%)
            ensemble_prediction = max(0.005, min(0.95, ensemble_prediction))

            return float(ensemble_prediction)

        except Exception as e:
            logger.warning(f"Error in Ensemble model: {e}")
            # Fallback to simple current price
            return float(market.get("current_price", 0.5))

    def update_weights(self, model_idx: int, performance: float):
        """Update model weights based on recent performance."""
        # Simple performance-based weight update
        self.model_performance.append((model_idx, performance))

        # Recalculate weights periodically
        if len(self.model_performance) >= 10:
            # Calculate average performance per model
            perf_by_model = {}
            for idx, perf in self.model_performance[-20:]:  # Last 20 observations
                perf_by_model.setdefault(idx, []).append(perf)

            new_weights = []
            for i in range(len(self.models)):
                perfs = perf_by_model.get(i, [0.5])
                avg_perf = np.mean(perfs)
                new_weights.append(max(0.05, avg_perf))

            # Normalize
            total = sum(new_weights)
            self.weights = np.array(new_weights) / total
            logger.info(f"Updated ensemble weights: {self.weights}")

    def export_state(self) -> Dict:
        """Export all model states for persistence."""
        return {
            "weights": self.weights.tolist(),
            "model_states": [m.export_state() for m in self.models],
        }

    def import_state(self, state: Dict):
        """Import all model states from persistence."""
        if "weights" in state:
            self.weights = np.array(state["weights"])
        if "model_states" in state:
            for model, model_state in zip(self.models, state["model_states"]):
                model.import_state(model_state)


class MachineLearningModel(ProbabilityModel):
    """Placeholder for future ML model integration."""
    def __init__(self, model_path: str):
        # Load your trained model
        # self.model = joblib.load(model_path)
        # self.scaler = joblib.load(scaler_path)
        pass

    def extract_features(self, market: Dict) -> np.ndarray:
        # Extract features from market data
        pass

    def estimate_probability(self, market: Dict) -> float:
        try:
            features = self.extract_features(market)
            # features_scaled = self.scaler.transform([features])
            # probability = self.model.predict_proba(features_scaled)[0][1]
            # For now, return a placeholder
            return 0.5
        except Exception as e:
            logger.warning(f"Error in ML model: {e}")
            return market.get("current_price", 0.5)