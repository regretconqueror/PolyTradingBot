"""
Yes/No sum arbitrage scanner

If best_ask_yes + best_ask_no < 1 - fee_buffer, buy both sides.
"""
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

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
    def __init__(self, fee_buffer: float = 0.02, max_per_market: float = 0.05):
        self.fee_buffer = fee_buffer
        self.max_per_market = max_per_market

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

    def scan(self, market: Dict, orderbook_yes: Dict, orderbook_no: Dict, capital: float) -> Optional[YesNoArbSignal]:
        best_yes = self._best_ask(orderbook_yes)
        best_no = self._best_ask(orderbook_no)
        if not best_yes or not best_no:
            return None

        ask_yes, size_yes = best_yes
        ask_no, size_no = best_no
        sum_price = ask_yes + ask_no

        # Edge is how far below 1 the combined price is (after fee buffer)
        edge = 1.0 - sum_price - self.fee_buffer
        if edge <= 0:
            return None

        # Budget cap per market
        budget = capital * self.max_per_market
        # Max shares limited by book sizes and budget
        max_shares_by_budget = budget / sum_price
        max_shares_by_book = min(size_yes, size_no)
        shares = max(0.0, min(max_shares_by_budget, max_shares_by_book))
        if shares <= 0:
            return None

        size_dollars = shares * sum_price

        return YesNoArbSignal(
            condition_id=market.get("conditionId", ""),
            question=market.get("question", "")[:80],
            token_yes=str(market.get("clobTokenIds", ["", ""])[0]),
            token_no=str(market.get("clobTokenIds", ["", ""])[1]),
            ask_yes=ask_yes,
            ask_no=ask_no,
            sum_price=sum_price,
            edge=edge,
            size_dollars=size_dollars,
        )