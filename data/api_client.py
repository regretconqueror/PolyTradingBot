"""Polymarket API client"""
import requests
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

class PolymarketAPI:
    """Client for Polymarket Gamma and CLOB APIs"""
    
    def __init__(self):
        self.gamma_url = "https://gamma-api.polymarket.com"
        self.clob_url = "https://clob.polymarket.com"
        self.session = requests.Session()
    
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
        params = {
            "active": "true",
            "closed": "false",
            "limit": limit,
            "sort": "volume",
            "order": "desc"
        }
        
        try:
            response = self.session.get(endpoint, params=params, timeout=30)
            response.raise_for_status()
            markets = response.json()
            
            # Filter by liquidity
            filtered = [m for m in markets if float(m.get("liquidity", 0)) >= min_liquidity]
            logger.info(f"Fetched {len(filtered)} markets with liquidity >= ${min_liquidity:,.0f}")
            return filtered
            
        except Exception as e:
            logger.error(f"Error fetching markets: {e}")
            return []
    
    def get_price(self, token_id: str) -> float:
        """Get current price for a token"""
        if not token_id:
            return 0.0
        
        endpoint = f"{self.clob_url}/price"
        params = {"token_id": token_id}
        
        try:
            response = self.session.get(endpoint, params=params, timeout=10)
            response.raise_for_status()
            return float(response.json().get("price", 0))
        except Exception as e:
            logger.error(f"Error fetching price for {token_id}: {e}")
            return 0.0
    
    def get_orderbook(self, token_id: str) -> Dict:
        """Get orderbook for a token"""
        endpoint = f"{self.clob_url}/book"
        params = {"token_id": token_id}
        
        try:
            response = self.session.get(endpoint, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error fetching orderbook: {e}")
            return {"bids": [], "asks": []}