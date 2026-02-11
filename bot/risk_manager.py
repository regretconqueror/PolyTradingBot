"""Risk management utilities"""
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

class RiskManager:
    """Manages trading risks"""
    
    def __init__(self, max_drawdown: float = 0.15):
        self.max_drawdown = max_drawdown
        self.peak_value = 0
        self.current_value = 0
    
    def check_drawdown(self, portfolio_value: float) -> bool:
        """
        Check if drawdown exceeds limit
        
        Returns:
            True if trading should continue, False if circuit breaker triggered
        """
        if portfolio_value > self.peak_value:
            self.peak_value = portfolio_value
        
        drawdown = (self.peak_value - portfolio_value) / self.peak_value
        
        if drawdown > self.max_drawdown:
            logger.error(f"🚨 CIRCUIT BREAKER: Drawdown {drawdown:.1%} exceeds limit {self.max_drawdown:.1%}")
            return False
        
        return True
    
    def calculate_position_size(self, edge: float, confidence: float, 
                               bankroll: float, kelly_fraction: float = 0.25) -> float:
        """
        Calculate position size using fractional Kelly
        
        Args:
            edge: Expected edge (decimal)
            confidence: Model confidence (0-1)
            bankroll: Available capital
            kelly_fraction: Fraction of full Kelly to use (safety factor)
            
        Returns:
            Recommended position size
        """
        # Simplified Kelly: f* = edge / odds
        # Using fractional Kelly for safety
        kelly_bet = edge * confidence * kelly_fraction
        return min(kelly_bet * bankroll, bankroll * 0.20)  # Cap at 20%