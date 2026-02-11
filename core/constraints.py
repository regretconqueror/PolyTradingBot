"""Constraint checking and validation"""
from typing import List, Dict
import logging

from .models import PortfolioConstraints

logger = logging.getLogger(__name__)

class ConstraintChecker:
    """Validates that allocations satisfy all constraints"""
    
    def __init__(self, constraints: PortfolioConstraints):
        self.constraints = constraints
    
    def check(self, allocations: Dict, capital: float) -> bool:
        """
        Check if orders satisfy all constraints
        
        Args:
            allocations: Dictionary of market_id -> allocation_amount
            capital: Total capital
            
        Returns:
            True if all constraints satisfied, False otherwise
        """
        total_exposure = sum(allocations.values())
        
        # Check total exposure
        if total_exposure > capital * self.constraints.max_total_exposure:
            logger.error(f"Total exposure ${total_exposure:.0f} exceeds limit "
                        f"${capital * self.constraints.max_total_exposure:.0f}")
            return False
        
        # Check individual positions
        for market_id, amount in allocations.items():
            if amount > capital * self.constraints.max_single_position:
                logger.error(f"Position {market_id} ${amount:.0f} exceeds limit "
                           f"${capital * self.constraints.max_single_position:.0f}")
                return False
        
        return True