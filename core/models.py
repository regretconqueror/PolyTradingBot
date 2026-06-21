"""Data models"""
from dataclasses import dataclass, field
from typing import Dict, Optional
from enum import Enum
import uuid
from datetime import datetime


class OptimizationStatus(Enum):
    CONVERGED = "converged"
    MAX_ITER = "max_iterations"
    INFEASIBLE = "infeasible"


class TradeStatus(Enum):
    """Lifecycle states for a trade."""
    PENDING  = "pending"
    FILLED   = "filled"
    EXITED   = "exited"
    SETTLED  = "settled"
    EXPIRED  = "expired"
    FAILED   = "failed"
    DRY_RUN  = "dry_run"


@dataclass
class Trade:
    """Canonical trade record with full lifecycle tracking."""
    token_id: str
    side: str = "BUY"
    status: str = TradeStatus.PENDING.value
    entry_price: float = 0.0
    exit_price: float = 0.0
    realized_pnl: float = 0.0
    size: float = 0.0
    opened_at: str = ""
    closed_at: str = ""
    exit_reason: str = ""
    category: str = "default"
    type: str = "regular"
    market: str = ""
    condition_id: str = ""
    trade_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    allocation: float = 0.0
    edge: float = 0.0
    filled_value: float = 0.0
    slippage: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat)

    def close(self, exit_price: float, reason: str,
              status: str = TradeStatus.EXITED.value):
        """Transition to a closed state with realized P&L."""
        self.exit_price = exit_price
        self.exit_reason = reason
        self.realized_pnl = (exit_price - self.entry_price) * self.size
        self.status = status
        self.closed_at = datetime.now().isoformat()

    def settle(self, resolution_price: float):
        """Mark as settled (market resolved). resolution_price is 0 or 1."""
        self.close(
            exit_price=resolution_price,
            reason="market_settled",
            status=TradeStatus.SETTLED.value,
        )

    def to_dict(self) -> Dict:
        return {
            "trade_id": self.trade_id,
            "token_id": self.token_id,
            "side": self.side,
            "status": self.status,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "realized_pnl": self.realized_pnl,
            "size": self.size,
            "opened_at": self.opened_at,
            "closed_at": self.closed_at,
            "exit_reason": self.exit_reason,
            "category": self.category,
            "type": self.type,
            "market": self.market,
            "condition_id": self.condition_id,
            "allocation": self.allocation,
            "edge": self.edge,
            "filled_value": self.filled_value,
            "slippage": self.slippage,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "Trade":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class Market:
    condition_id: str
    question: str
    token_id: str
    price: float
    probability: float
    liquidity: float
    volume_24h: float
    category: str
    resolution_date: str
    outcome: str = "YES"  # "YES" or "NO" — which outcome token this represents

    @property
    def edge(self) -> float:
        return self.probability - self.price

    def __repr__(self):
        return f"Market({self.question[:30]}..., {self.outcome}, edge={self.edge:.2%})"


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
    # Joint constraint: max total allocation across ALL outcome tokens of the same
    # market condition (e.g. YES + NO of the same market must not exceed this).
    # Prevents the optimizer from simultaneously betting every side of one event.
    max_condition_exposure: float = 0.25