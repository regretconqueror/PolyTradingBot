"""
Yes/No sum arbitrage scanner

If best_ask_yes + best_ask_no < 1 - fee_buffer - min_edge, buy both sides.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

@dataclass
class YesNoArbSignal:
    condition_id: str
    question: str
    token_yes: str
    token_no: str
    ask_yes: float
    ask_no: float
    sum_price: float
    edge: float
    size_dollars: float

class YesNoArbScanner:
    def __init__(self, fee_buffer: float = 0.02, max_per_market: float = 0.05, min_edge: float = 0.005):
        self.fee_buffer = fee_buffer
        self.max_per_market = max_per_market
        self.min_edge = min_edge

    def _best_ask(self, book: Dict) -> Optional[Tuple[float, float]]:
        """Return (price, size) for best ask. Handles dict or list entries."""
        asks = book.get("asks", []) or []
        if not asks:
            return None

        # Normalize asks to list of (price, size)
        norm = []
        for a in asks:
            if isinstance(a, dict):
                p = a.get("price") or a.get("p")
                s = a.get("size") or a.get("q") or a.get("quantity")
            elif isinstance(a, (list, tuple)) and len(a) >= 2:
                p, s = a[0], a[1]
            else:
                continue
            try:
                p = float(p)
                s = float(s)
            except Exception:
                continue
            norm.append((p, s))

        if not norm:
            return None

        # Best ask = lowest price
        norm.sort(key=lambda x: x[0])
        return norm[0]

    def _walk_book(self, asks: List, target_shares: float) -> Tuple[float, float]:
        """
        Walk order book asks to determine average execution price for target_shares.
        Returns (average_price, shares_filled).
        """
        if not asks or target_shares <= 0:
            return 0.0, 0.0

        # Normalize asks to list of (price, size)
        norm = []
        for a in asks:
            if isinstance(a, dict):
                p = a.get("price") or a.get("p")
                s = a.get("size") or a.get("q") or a.get("quantity")
            elif isinstance(a, (list, tuple)) and len(a) >= 2:
                p, s = a[0], a[1]
            else:
                continue
            try:
                p = float(p)
                s = float(s)
            except Exception:
                continue
            norm.append((p, s))

        if not norm:
            return 0.0, 0.0

        # Sort by price ascending (cheapest first)
        norm.sort(key=lambda x: x[0])

        shares_filled = 0.0
        total_cost = 0.0

        for price, size in norm:
            fill = min(target_shares - shares_filled, size)
            if fill <= 0:
                break
            total_cost += fill * price
            shares_filled += fill
            if shares_filled >= target_shares:
                break

        avg_price = total_cost / shares_filled if shares_filled > 0 else 0.0
        return avg_price, shares_filled

    def scan(self, market: Dict, orderbook_yes: Dict, orderbook_no: Dict, capital: float) -> Optional[YesNoArbSignal]:
        best_yes = self._best_ask(orderbook_yes)
        best_no = self._best_ask(orderbook_no)
        if not best_yes or not best_no:
            return None

        best_ask_yes, _ = best_yes
        best_ask_no, _ = best_no

        # Validate prices are in valid range
        if not (0 < best_ask_yes < 1 and 0 < best_ask_no < 1):
            return None

        # Budget cap per market
        budget = capital * self.max_per_market

        # Estimate target shares based on best asks
        target_shares = budget / (best_ask_yes + best_ask_no)
        if target_shares <= 0:
            return None

        # Walk the books to calculate average execution price
        yes_avg, yes_filled = self._walk_book(orderbook_yes.get("asks", []), target_shares)
        no_avg, no_filled = self._walk_book(orderbook_no.get("asks", []), target_shares)

        # Actual shares we can buy is limited by the side with less liquidity
        shares = min(yes_filled, no_filled)
        if shares <= 0:
            return None

        # Re-walk the books with the actual matched shares to get precise weighted average prices
        yes_avg, _ = self._walk_book(orderbook_yes.get("asks", []), shares)
        no_avg, _ = self._walk_book(orderbook_no.get("asks", []), shares)

        sum_price = yes_avg + no_avg

        # Edge is how far below 1 the combined price is (after fee buffer)
        edge = 1.0 - sum_price - self.fee_buffer
        if edge <= self.min_edge:  # Require edge to exceed minimum threshold
            return None

        size_dollars = shares * sum_price

        # Validate token IDs exist
        token_ids = market.get("clobTokenIds", ["", ""])
        if len(token_ids) < 2:
            return None

        return YesNoArbSignal(
            condition_id=market.get("conditionId", ""),
            question=market.get("question", "")[:80],
            token_yes=str(token_ids[0]),
            token_no=str(token_ids[1]),
            ask_yes=yes_avg,
            ask_no=no_avg,
            sum_price=sum_price,
            edge=edge,
            size_dollars=size_dollars,
        )