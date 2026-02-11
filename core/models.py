"""Data models"""
from dataclasses import dataclass, field
from typing import Dict
from enum import Enum

class OptimizationStatus(Enum):
    CONVERGED = "converged"
    MAX_ITER = "max_iterations"
    INFEASIBLE = "infeasible"

@dataclass
class Market:
    condition_id: str
    question: str
    token_id_yes: str
    token_id_no: str
    current_price: float
    your_probability: float
    liquidity: float
    volume_24h: float
    category: str
    resolution_date: str
    
    @property
    def edge(self) -> float:
        return self.your_probability - self.current_price
    
    def __repr__(self):
        return f"Market({self.question[:30]}..., edge={self.edge:.2%})"

@dataclass
class PortfolioConstraints:
    max_total_exposure: float = 0.75
    max_single_position: float = 0.20
    max_category_exposure: Dict[str, float] = field(default_factory=lambda: {
        "Crypto": 0.30,
        "Politics": 0.25,
        "Sports": 0.20,
        "Science": 0.15,
        "default": 0.25
    })
    min_bet_size: float = 0.02
    max_drawdown: float = 0.15