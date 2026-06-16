"""Polymarket API client"""
import requests
from typing import List, Dict, Optional
import logging
import time
import numpy as np
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class PolymarketAPI:
    """Client for Polymarket Gamma and CLOB APIs with circuit breaker and rate limiting"""

    def __init__(self):
        self.gamma_url = "https://gamma-api.polymarket.com"
        self.clob_url = "https://clob.polymarket.com"
        self.session = requests.Session()

        # Circuit breaker state
        self.failure_count = 0
        self.last_failure_time = None
        self.circuit_open = False
        self.circuit_timeout = timedelta(minutes=5)  # Open circuit for 5 minutes
        self.failure_threshold = 5  # Open circuit after 5 failures

        # Rate limiting
        self.last_request_time = None
        self.min_request_interval = timedelta(milliseconds=200)  # 5 req/sec max

        # Request statistics
        self.request_count = 0
        self.error_count = 0
        self.success_count = 0

    def _check_circuit_breaker(self) -> bool:
        """Check if circuit breaker is open"""
        if not self.circuit_open:
            return False

        # Check if timeout has passed
        if self.last_failure_time and datetime.now() - self.last_failure_time > self.circuit_timeout:
            logger.info("Circuit breaker half-open: attempting request")
            self.circuit_open = False
            self.failure_count = 0
            return False

        return True

    def _record_success(self):
        """Record successful request"""
        self.failure_count = 0
        self.circuit_open = False
        self.success_count += 1
        self.request_count += 1

    def _record_failure(self):
        """Record failed request"""
        self.failure_count += 1
        self.error_count += 1
        self.request_count += 1
        self.last_failure_time = datetime.now()

        if self.failure_count >= self.failure_threshold:
            self.circuit_open = True
            logger.warning(f"Circuit breaker opened after {self.failure_count} failures")

    def _enforce_rate_limit(self):
        """Enforce rate limiting between requests"""
        if self.last_request_time:
            elapsed = datetime.now() - self.last_request_time
            if elapsed < self.min_request_interval:
                sleep_time = (self.min_request_interval - elapsed).total_seconds()
                time.sleep(sleep_time)
        self.last_request_time = datetime.now()

    def _get_json(self, endpoint: str, params: Dict = None, timeout: int = 10,
                  retries: int = 3, backoff: float = 0.5):
        """GET JSON with bounded retry/backoff for transient API failures and circuit breaker."""
        # Check circuit breaker
        if self._check_circuit_breaker():
            raise requests.exceptions.RequestException("Circuit breaker is open")

        # Enforce rate limiting
        self._enforce_rate_limit()

        last_error = None
        for attempt in range(retries):
            try:
                response = self.session.get(endpoint, params=params, timeout=timeout)
                response.raise_for_status()
                self._record_success()
                return response.json()
            except requests.exceptions.RequestException as e:
                last_error = e
                logger.warning(f"API request failed (attempt {attempt + 1}/{retries}): {e}")
                if attempt == retries - 1:
                    break
                # Exponential backoff with jitter
                sleep_time = backoff * (2 ** attempt) + np.random.uniform(0, 0.1)
                time.sleep(sleep_time)

        self._record_failure()
        raise last_error
    
    def get_active_markets(self, limit: int = 100, min_liquidity: float = 5000) -> List[Dict]:
        """
        Fetch active markets from Gamma API
        
        Args:
            limit: Maximum number of markets to fetch
            min_liquidity: Minimum liquidity threshold
            
        Returns:
            List of market dictionaries
        """
        endpoint = f"{self.gamma_url}/markets"
        # NOTE: Gamma API validation may reject sort/order fields.
        # Keep params minimal to avoid 422 validation errors.
        params = {
            "active": "true",
            "closed": "false",
            "limit": limit
        }
        
        try:
            markets = self._get_json(endpoint, params=params, timeout=30)
            
            # Filter by liquidity
            filtered = [m for m in markets if float(m.get("liquidity", 0)) >= min_liquidity]
            logger.info(f"Fetched {len(filtered)} markets with liquidity >= ${min_liquidity:,.0f}")
            return filtered
            
        except Exception as e:
            logger.error(f"Error fetching markets: {e}")
            return []
    
    def get_price(self, token_id: str, side: str = "BUY") -> float:
        """Get current best CLOB price for buying or selling a token."""
        if not token_id:
            return 0.0

        side = side.upper()
        if side not in {"BUY", "SELL"}:
            raise ValueError(f"side must be BUY or SELL, got {side}")

        endpoint = f"{self.clob_url}/price"
        params = {"token_id": token_id, "side": side}

        try:
            return float(self._get_json(endpoint, params=params, timeout=10).get("price", 0))
        except Exception as e:
            logger.error(f"Error fetching {side} price for {token_id}: {e}")
            return 0.0
    
    def get_orderbook(self, token_id: str) -> Dict:
        """Get orderbook for a token"""
        endpoint = f"{self.clob_url}/book"
        params = {"token_id": token_id}
        
        try:
            return self._get_json(endpoint, params=params, timeout=10)
        except Exception as e:
            logger.error(f"Error fetching orderbook: {e}")
            return {"bids": [], "asks": []}
