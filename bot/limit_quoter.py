"""
Limit Order Quoting Engine for Polymarket CLOB.

Places post-only GTC limit orders at configurable aggressiveness relative
to the current bid/ask spread.  Tracks open quote order IDs per token so
they can be cancelled and re-quoted when the market moves.
"""
import logging
import time
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class LimitQuoter:
    """
    Manages resting limit orders on the Polymarket CLOB.

    Quoting logic
    ─────────────
    Given best_bid and best_ask and a tick size T:

        BUY  quote price = best_bid  + round(aggressiveness * (spread / T)) * T
        SELL quote price = best_ask  - round(aggressiveness * (spread / T)) * T

    aggressiveness=0.0  → quote AT best_bid / best_ask  (cheapest, slowest fill)
    aggressiveness=0.5  → quote at midpoint              (balanced default)
    aggressiveness=1.0  → quote one tick inside spread   (fastest fill)

    All orders are submitted with ``post_only=True`` so they are NEVER taker
    orders — they either rest on the book or are rejected.  This keeps fees at
    the maker level (often zero or rebated on Polymarket).
    """

    # Minimum spread (in price units) below which we skip quoting to avoid
    # crossing the spread accidentally on a very tight book.
    MIN_SPREAD = 0.01

    def __init__(
        self,
        execution_engine,
        aggressiveness: float = 0.3,
        requote_on_move: bool = True,
        min_spread_to_quote: float = 0.01,
        dry_run: bool = True,
    ):
        """
        Parameters
        ----------
        execution_engine : ExecutionEngine
            Live execution engine used to submit / cancel orders.
        aggressiveness : float
            0.0 = passive (at best_bid/ask), 1.0 = at midpoint.
            Default 0.3 — slightly passive for better fill price.
        requote_on_move : bool
            If True, cancel and re-place when price has drifted by >= 1 tick.
        min_spread_to_quote : float
            Minimum bid-ask spread required before we attempt to quote.
            Guards against crossing on extremely tight markets.
        dry_run : bool
            If True, log intended actions but do not submit real orders.
        """
        self.engine = execution_engine
        self.aggressiveness = max(0.0, min(1.0, aggressiveness))
        self.requote_on_move = requote_on_move
        self.min_spread_to_quote = min_spread_to_quote
        self.dry_run = dry_run

        # token_id → {order_id, price, side, size, timestamp}
        self._open_quotes: Dict[str, Dict] = {}

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def quote(
        self,
        token_id: str,
        side: str,
        size: float,
        override_price: Optional[float] = None,
    ) -> Dict:
        """
        Place (or re-place) a limit order for *token_id*.

        Parameters
        ----------
        token_id   : Polymarket conditional token ID.
        side       : "BUY" or "SELL".
        size       : Order size in USDC (BUY) or shares (SELL).
        override_price : If supplied, use this exact price instead of computing
                         from the spread.  Useful for manual adjustments.

        Returns
        -------
        dict with keys: status, token_id, side, size, price, order_id
        """
        side = side.upper()

        # Cancel any existing quote for this token first
        if token_id in self._open_quotes:
            self._cancel_quote(token_id)

        # Fetch spread and compute limit price
        try:
            best_bid, best_ask, tick = self._fetch_spread(token_id)
        except Exception as exc:
            logger.warning("LimitQuoter: could not fetch spread for %s: %s", token_id, exc)
            return {"status": "error", "error": str(exc), "token_id": token_id}

        spread = best_ask - best_bid
        if spread < self.min_spread_to_quote:
            logger.info(
                "LimitQuoter: spread %.4f < min %.4f for %s — skipping quote",
                spread, self.min_spread_to_quote, token_id,
            )
            return {
                "status": "skipped",
                "reason": "spread_too_tight",
                "spread": spread,
                "token_id": token_id,
            }

        if override_price is not None:
            limit_price = override_price
        else:
            limit_price = self._compute_limit_price(
                side, best_bid, best_ask, tick, spread
            )

        # Dry-run: log and return without submitting
        if self.dry_run:
            logger.info(
                "LimitQuoter [DRY_RUN] %s %s @ %.4f (size=%.4f) spread=[%.4f, %.4f]",
                side, token_id[:12], limit_price, size, best_bid, best_ask,
            )
            fake_id = f"dry_{token_id[:8]}_{int(time.time())}"
            self._open_quotes[token_id] = {
                "order_id": fake_id,
                "price": limit_price,
                "side": side,
                "size": size,
                "timestamp": int(time.time()),
            }
            return {
                "status": "dry_run",
                "token_id": token_id,
                "side": side,
                "size": size,
                "price": limit_price,
                "order_id": fake_id,
                "best_bid": best_bid,
                "best_ask": best_ask,
            }

        # Live: submit limit order
        result = self.engine.execute_limit_order(
            token_id=token_id,
            side=side,
            size=size,
            price=limit_price,
            post_only=True,
        )

        if result.get("status") in ("success", "submitted"):
            order_id = result.get("order_id", "")
            self._open_quotes[token_id] = {
                "order_id": order_id,
                "price": limit_price,
                "side": side,
                "size": size,
                "timestamp": int(time.time()),
            }
            logger.info(
                "LimitQuoter: placed %s limit order %s for %s @ %.4f (size=%.4f)",
                side, order_id, token_id[:12], limit_price, size,
            )

        return {**result, "price": limit_price, "best_bid": best_bid, "best_ask": best_ask}

    def requote(self, token_id: str) -> Dict:
        """
        Re-evaluate the current quote for *token_id* and re-place if the
        price has moved by at least one tick.

        Returns
        -------
        dict with status "requoted", "unchanged", or "error".
        """
        if token_id not in self._open_quotes:
            return {"status": "no_open_quote", "token_id": token_id}

        existing = self._open_quotes[token_id]

        try:
            best_bid, best_ask, tick = self._fetch_spread(token_id)
        except Exception as exc:
            return {"status": "error", "error": str(exc), "token_id": token_id}

        spread = best_ask - best_bid
        new_price = self._compute_limit_price(
            existing["side"], best_bid, best_ask, tick, spread
        )

        # Only re-quote if price drifted by >= 1 tick
        drift = abs(new_price - existing["price"])
        if drift < tick * 0.9:
            logger.debug(
                "LimitQuoter: %s quote unchanged (drift=%.4f < tick=%.4f)",
                token_id[:12], drift, tick,
            )
            return {"status": "unchanged", "token_id": token_id, "price": existing["price"]}

        logger.info(
            "LimitQuoter: requoting %s — old %.4f → new %.4f (drift=%.4f)",
            token_id[:12], existing["price"], new_price, drift,
        )
        return self.quote(
            token_id=token_id,
            side=existing["side"],
            size=existing["size"],
        )

    def requote_all(self) -> Dict[str, Dict]:
        """Re-evaluate and refresh all open quotes. Returns per-token results."""
        results = {}
        for token_id in list(self._open_quotes.keys()):
            results[token_id] = self.requote(token_id)
        return results

    def cancel_all_quotes(self) -> Dict:
        """Cancel every open quote. Emergency wipe."""
        cancelled = []
        errors = []
        for token_id in list(self._open_quotes.keys()):
            res = self._cancel_quote(token_id)
            if res.get("status") in ("success", "dry_run", "cancelled"):
                cancelled.append(token_id)
            else:
                errors.append(token_id)

        logger.info("LimitQuoter: cancelled %d quotes, %d errors", len(cancelled), len(errors))
        return {"cancelled": cancelled, "errors": errors}

    @property
    def open_quotes(self) -> Dict[str, Dict]:
        """Read-only view of currently tracked open quotes."""
        return dict(self._open_quotes)

    # ─────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_limit_price(
        self,
        side: str,
        best_bid: float,
        best_ask: float,
        tick: float,
        spread: float,
    ) -> float:
        """
        Compute the limit price using aggressiveness parameter.

        aggressiveness=0  → at the current best_bid (BUY) or best_ask (SELL)
        aggressiveness=1  → one tick inside the spread (closest to midpoint)
        """
        ticks_in_spread = max(1, round(spread / tick))
        tick_offset = round(self.aggressiveness * ticks_in_spread) * tick

        if side == "BUY":
            price = best_bid + tick_offset
        else:  # SELL
            price = best_ask - tick_offset

        # Clamp to valid Polymarket price range (0.01 – 0.99)
        price = max(0.01, min(0.99, round(price, 4)))
        return price

    def _fetch_spread(self, token_id: str) -> Tuple[float, float, float]:
        """
        Fetch best_bid, best_ask, and tick_size from the CLOB.

        Returns (best_bid, best_ask, tick_size)
        """
        client = self.engine._get_client()

        spread_data = client.get_spread(token_id)
        tick_data = client.get_tick_size(token_id)

        # Spread response can be a dict or object
        if isinstance(spread_data, dict):
            best_bid = float(spread_data.get("bid", 0.01))
            best_ask = float(spread_data.get("ask", 0.99))
        else:
            best_bid = float(getattr(spread_data, "bid", 0.01))
            best_ask = float(getattr(spread_data, "ask", 0.99))

        # Tick size response
        if isinstance(tick_data, dict):
            tick = float(tick_data.get("minimum_tick_size", 0.01))
        elif isinstance(tick_data, str):
            tick = float(tick_data)
        else:
            tick = float(tick_data) if tick_data else 0.01

        return best_bid, best_ask, tick

    def _cancel_quote(self, token_id: str) -> Dict:
        """Cancel the tracked open quote for *token_id* and remove from tracking."""
        quote = self._open_quotes.get(token_id)
        if not quote:
            return {"status": "no_quote"}

        order_id = quote["order_id"]

        if self.dry_run or order_id.startswith("dry_"):
            logger.info("LimitQuoter [DRY_RUN] cancel %s for %s", order_id, token_id[:12])
            self._open_quotes.pop(token_id, None)
            return {"status": "dry_run", "order_id": order_id}

        result = self.engine.cancel_order(order_id)
        self._open_quotes.pop(token_id, None)
        return result
