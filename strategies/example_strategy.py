"""
Probability estimation models

THIS IS WHERE YOUR COMPETITIVE ADVANTAGE COMES FROM!
Replace these examples with your proprietary models.
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
        # Fallback: return single probability for first outcome
        # Override this method in models that support multi-outcome estimation
        single_prob = self.estimate_probability(market)
        return [single_prob]

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
            current_price = market.get("current_price", 0.5)

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
                # Mean reversion with trend consideration
                prediction = wma - (recent_trend * 0.3)  # Counter-trend factor
            else:
                prediction = wma

            # Apply volatility bounds
            lower_bound = max(0.05, wma - volatility_adjustment)
            upper_bound = min(0.95, wma + volatility_adjustment)

            # Clamp prediction to reasonable bounds
            prediction = max(lower_bound, min(upper_bound, prediction))

            return float(prediction)

        except Exception as e:
            logger.warning(f"Error in WMA model: {e}")
            return float(market.get("current_price", 0.5))

class VolatilityAdjustedModel(ProbabilityModel):
    """
    Volatility-adjusted model that accounts for market uncertainty
    Uses implied volatility from price movements to adjust predictions
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
            volume = float(market.get("volume_24h", 0))
            liquidity = float(market.get("liquidity", 0))

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
                return current_price

            # Calculate basic statistics
            prices = np.array(self.price_history[token_id])
            mean_price = np.mean(prices)
            std_price = np.std(prices)

            # Update volatility history
            if len(prices) >= 2:
                returns = np.diff(prices) / prices[:-1]
                volatility = np.std(returns) * np.sqrt(252)  # Annualized
                self.volatility_history[token_id].append(volatility)

                if len(self.volatility_history[token_id]) > self.lookback_period:
                    self.volatility_history[token_id] = self.volatility_history[token_id][-self.lookback_period:]

            # Calculate market efficiency score based on volume and liquidity
            efficiency_score = min(1.0, (volume / 10000) * (liquidity / 50000))  # Normalize
            efficiency_score = max(0.1, min(1.0, efficiency_score))

            # In inefficient markets, trust the current price more
            # In efficient markets, look for deviations from fair value
            trust_current_price = 1.0 - (efficiency_score * 0.5)  # 0.5 to 1.0 range

            # Calculate z-score of current price relative to recent history
            if std_price > 0:
                z_score = (current_price - mean_price) / std_price
            else:
                z_score = 0

            # Mean reversion expectation (stronger in inefficient markets)
            mean_reversion_component = -z_score * std_price * (1 - trust_current_price) * 0.1

            # Momentum component (weaker but still present)
            if len(prices) >= 3:
                short_ma = np.mean(prices[-3:])
                long_ma = np.mean(prices[-min(10, len(prices)):])
                momentum = (short_ma - long_ma) * trust_current_price * 0.05
            else:
                momentum = 0

            # Final prediction
            prediction = mean_price + mean_reversion_component + momentum

            # Apply confidence bounds
            confidence_interval = std_price * (1 - self.confidence_level)
            lower_bound = max(0.05, mean_price - confidence_interval)
            upper_bound = min(0.95, mean_price + confidence_interval)

            prediction = max(lower_bound, min(upper_bound, prediction))

            return float(prediction)

        except Exception as e:
            logger.warning(f"Error in Volatility Adjusted model: {e}")
            return float(market.get("current_price", 0.5))

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
    """

    def __init__(self,
                 min_probability: float = 0.005,      # 0.5% floor
                 long_shot_threshold: float = 0.02,  # 2% — below this = long-shot
                 momentum_weight: float = 0.15,
                 volume_weight: float = 0.10,
                 mean_reversion_strength: float = 0.05):
        self.min_probability = min_probability
        self.long_shot_threshold = long_shot_threshold
        self.momentum_weight = momentum_weight
        self.volume_weight = volume_weight
        self.mean_reversion_strength = mean_reversion_strength
        self.price_history = {}   # {token_id: [(price, volume, timestamp), ...]}
        self._history_window = 20

    def estimate_probability(self, market: Dict) -> float:
        try:
            token_id = market.get("token_id") or market.get("token_id_yes", "")
            price = float(market.get("current_price", market.get("price", 0.5)))
            volume = float(market.get("volume_24h", 0))
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


class WhaleTrackerModel(ProbabilityModel):
    """
    Whale Tracker Model (Smart Money Tracking).
    
    Tracks a set of high-performing, profitable Polymarket wallets.
    Fetches their positions via Gamma API and adjusts probability
    estimates in favor of outcomes where whales have high exposure.
    """
    def __init__(self, whale_wallets: Optional[List[str]] = None, 
                 impact_factor: float = 0.05,
                 fallback_positions: Optional[Dict] = None):
        """
        Args:
            whale_wallets: List of wallet addresses (0x...) to track.
            impact_factor: Maximum adjustment to probability (e.g. 0.05 = ±5%).
            fallback_positions: Mock dictionary for testing or paper mode.
        """
        # Default whale wallets from Polymarket leaderboard if none provided
        self.whale_wallets = whale_wallets or [
            "0x4b7a2a192c73295c2560ec0a887b474328574169", # Mock/leaderboard whale 1
            "0x5da8f8cb9cbef0c85c276313ef31102dbd668270"  # Mock/leaderboard whale 2
        ]
        self.impact_factor = impact_factor
        # format: {token_id: {wallet: {"size": float, "side": "BUY"|"SELL", "outcome": "YES"|"NO"}}}
        self.fallback_positions = fallback_positions or {}
        
        # Cache of whale positions to prevent hitting API for every single token/outcome
        self._positions_cache = {}
        self._cache_timestamp = None
        self._cache_duration = timedelta(minutes=5)

    def estimate_probability(self, market: Dict) -> float:
        try:
            token_id = market.get("token_id") or market.get("token_id_yes", "")
            current_price = float(market.get("current_price", 0.5))
            condition_id = market.get("conditionId", "")
            token_ids = market.get("clobTokenIds", [])

            if not token_id:
                return current_price

            # Fetch whale positions
            whale_yes_shares = 0.0
            whale_no_shares = 0.0
            
            # Check fallback/mock positions first (useful for testing or paper trading)
            if token_id in self.fallback_positions:
                mock_data = self.fallback_positions[token_id]
                for wallet, pos in mock_data.items():
                    size = float(pos.get("size", 0))
                    outcome = pos.get("outcome", "YES")
                    if outcome == "YES":
                        whale_yes_shares += size
                    else:
                        whale_no_shares += size

            # Attempt live lookup via Gamma API if not using fallback values
            if not whale_yes_shares and not whale_no_shares:
                import requests
                now = datetime.now()
                # Use cache if it's less than 5 minutes old
                use_cache = (
                    self._cache_timestamp is not None
                    and now - self._cache_timestamp < self._cache_duration
                )
                
                if not use_cache:
                    new_cache = {}
                    for wallet in self.whale_wallets:
                        try:
                            url = f"https://gamma-api.polymarket.com/positions?userAddress={wallet}"
                            res = requests.get(url, timeout=3)
                            if res.status_code == 200:
                                new_cache[wallet] = res.json()
                        except Exception as e:
                            logger.warning(f"Error fetching whale positions for {wallet}: {e}")
                    self._positions_cache = new_cache
                    self._cache_timestamp = now

                # Extract positions from cached data
                for wallet in self.whale_wallets:
                    positions = self._positions_cache.get(wallet, [])
                    if isinstance(positions, list):
                        for pos in positions:
                            pos_condition = pos.get("conditionId")
                            asset_id = pos.get("asset")
                            size = float(pos.get("size", 0))
                            if asset_id == token_id:
                                whale_yes_shares += size
                            elif len(token_ids) >= 2 and asset_id == token_ids[1]:
                                whale_no_shares += size
                            elif pos_condition == condition_id:
                                # Fallback outcome checks
                                outcome = str(pos.get("outcome", "")).upper()
                                if outcome == "YES":
                                    whale_yes_shares += size
                                elif outcome == "NO":
                                    whale_no_shares += size

            # Calculate adjustment based on relative holdings
            total_shares = whale_yes_shares + whale_no_shares
            if total_shares == 0:
                return current_price

            net_skew = (whale_yes_shares - whale_no_shares) / total_shares
            adjustment = net_skew * self.impact_factor

            adjusted_prob = current_price + adjustment
            return float(max(0.005, min(0.995, adjusted_prob)))

        except Exception as e:
            logger.warning(f"Error in WhaleTrackerModel: {e}")
            return float(market.get("current_price", 0.5))



# Ensemble model that combines multiple approaches
class EnsembleModel(ProbabilityModel):
    """
    Ensemble model that combines multiple prediction approaches.
    Uses smarter models to avoid long-shot traps.
    """

    def __init__(self, models: Optional[List[ProbabilityModel]] = None):
        if models is None:
            self.models = [
                MarketSentimentModel(),                # Smart long-shot filtering
                VolatilityAdjustedModel(lookback_period=15, confidence_level=0.85),
                WhaleTrackerModel()                    # Whale wallet/Smart money tracker
            ]
        else:
            self.models = models

        # Equal weights initially, can be optimized based on performance
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
            # Previously 0.05 was reasonable for SimpleEdgeModel. Now our
            # MarketSentimentModel can return 0.5%–1% for long-shots, and
            # the 5% floor was destroying that signal.
            ensemble_prediction = max(0.005, min(0.95, ensemble_prediction))

            return float(ensemble_prediction)

        except Exception as e:
            logger.warning(f"Error in Ensemble model: {e}")
            # Fallback to simple current price
            return float(market.get("current_price", 0.5))

    def update_weights(self, performance_scores: List[float]):
        """Update model weights based on recent performance"""
        try:
            if len(performance_scores) != len(self.models):
                logger.warning("Performance scores length doesn't match number of models")
                return

            # Convert performance to weights (better performance = higher weight)
            # Use softmax to ensure weights sum to 1 and are positive
            exp_scores = np.exp(np.array(performance_scores))
            self.weights = exp_scores / np.sum(exp_scores)

            logger.info(f"Updated model weights: {self.weights}")

        except Exception as e:
            logger.warning(f"Error updating model weights: {e}")

# Example of how to implement a machine learning model
# (Uncomment and implement based on your available data)
"""
class MachineLearningModel(ProbabilityModel):
    def __init__(self, model_path: str):
        # Load your trained model
        # self.model = joblib.load(model_path)
        # self.scaler = joblib.load(scaler_path)
        pass

    def extract_features(self, market: Dict) -> np.ndarray:
        # Extract features from market data
        # Features might include:
        # - Current price, volume, liquidity
        # - Historical price statistics (mean, std, momentum)
        # - Time to expiration
        # - Category-specific factors
        # - Market sentiment indicators
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
"""
