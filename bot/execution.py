"""Order execution engine for Polymarket CLOB."""
import logging
import time
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class ExecutionEngine:
    """Handles guarded live order execution via the official CLOB client."""

    def __init__(
        self,
        api_key: str = None,
        api_secret: str = None,
        passphrase: str = None,
        private_key: str = None,
        funder_address: str = None,
        signature_type: int = 3,
        live_trading_enabled: bool = False,
        dry_run: bool = True,
        max_order_size: float = 25.0,
    ):
        self.api_key = api_key
        self.api_secret = api_secret
        self.passphrase = passphrase
        self.private_key = private_key
        self.funder_address = funder_address
        self.signature_type = signature_type
        self.live_trading_enabled = live_trading_enabled
        self.dry_run = dry_run
        self.max_order_size = max_order_size
        self._client = None

    def _missing_credentials(self):
        required = {
            "POLYMARKET_PRIVATE_KEY": self.private_key,
            "POLYMARKET_API_KEY": self.api_key,
            "POLYMARKET_API_SECRET": self.api_secret,
            "POLYMARKET_API_PASSPHRASE": self.passphrase,
            "POLYMARKET_FUNDER_ADDRESS": self.funder_address,
        }
        return [name for name, value in required.items() if not value]

    def validate_live_ready(self) -> Dict:
        """Return a status dict describing whether real order submission is allowed."""
        missing = self._missing_credentials()
        if missing:
            return {
                "ready": False,
                "reason": "missing_credentials",
                "missing": missing,
            }
        if not self.live_trading_enabled:
            return {
                "ready": False,
                "reason": "live_trading_disabled",
                "message": "Set LIVE_TRADING_ENABLED=true to allow real submissions.",
            }
        if self.dry_run:
            return {
                "ready": False,
                "reason": "dry_run_enabled",
                "message": "Set LIVE_DRY_RUN=false to submit real orders.",
            }
        return {"ready": True}

    def get_connection_details(self) -> Dict:
        """
        Check connection status and retrieve wallet details.
        
        Returns a dict:
            {
                "connected": bool,
                "status_text": str,
                "eoa_address": str or None,
                "proxy_address": str or None,
                "proxy_balance": float,
                "error": str or None
            }
        """
        missing = self._missing_credentials()
        if missing:
            return {
                "connected": False,
                "status_text": "Disconnected",
                "eoa_address": None,
                "proxy_address": None,
                "proxy_balance": 0.0,
                "error": f"Missing credentials: {', '.join(missing)}"
            }
            
        try:
            client = self._get_client()
            
            # Derive addresses
            eoa_address = client.get_address()
            proxy_address = client.builder.funder
            
            # Try fetching balance to verify API key validity
            from py_clob_client_v2.clob_types import BalanceAllowanceParams, AssetType
            params = BalanceAllowanceParams(asset_type=AssetType.COLLATERAL)
            try:
                data = client.get_balance_allowance(params)
                balance = float(data.get("balance", 0.0))
                error_msg = None
                connected = True
                status_text = "Connected"
            except Exception as api_err:
                logger.warning("Polymarket API authentication failed, returning derived addresses: %s", api_err)
                balance = 0.0
                error_msg = str(api_err)
                connected = True
                status_text = "Connected (Simulated)"
                
            return {
                "connected": connected,
                "status_text": status_text,
                "eoa_address": eoa_address,
                "proxy_address": proxy_address,
                "proxy_balance": balance,
                "error": error_msg
            }
        except Exception as e:
            logger.error("Error establishing API connection: %s", e)
            return {
                "connected": False,
                "status_text": "Connection Error",
                "eoa_address": None,
                "proxy_address": None,
                "proxy_balance": 0.0,
                "error": str(e)
            }


    def _get_client(self):
        if self._client is not None:
            return self._client

        try:
            from py_clob_client_v2 import ClobClient
            from py_clob_client_v2.clob_types import ApiCreds
        except ImportError as exc:
            raise RuntimeError(
                "py-clob-client-v2 is required for live execution. "
                "Install dependencies from requirements.txt."
            ) from exc

        creds = ApiCreds(
            api_key=self.api_key,
            api_secret=self.api_secret,
            api_passphrase=self.passphrase,
        )
        self._client = ClobClient(
            host="https://clob.polymarket.com",
            chain_id=137,
            key=self.private_key,
            creds=creds,
            signature_type=self.signature_type,
            funder=self.funder_address,
        )
        return self._client

    def execute_market_order(
        self,
        token_id: str,
        side: str,
        size: float,
        price: float = None,
        timeout: int = 10,
    ) -> Dict:
        """
        Execute a market order.

        BUY size is dollar amount. SELL size is shares. This bot currently only
        creates BUY orders, so SELL support is intentionally guarded by callers.
        """
        try:
            if size <= 0:
                raise ValueError(f"Order size must be positive, got {size}")

            side = side.upper()
            if side not in {"BUY", "SELL"}:
                raise ValueError(f"Side must be BUY or SELL, got {side}")

            capped_size = min(float(size), self.max_order_size)
            if capped_size < float(size):
                logger.warning(
                    "Capped live order size from %.2f to %.2f",
                    size,
                    capped_size,
                )

            readiness = self.validate_live_ready()
            if not readiness["ready"]:
                return {
                    "status": "dry_run" if self.dry_run else "blocked",
                    "token_id": token_id,
                    "side": side,
                    "size": capped_size,
                    "price": price,
                    "readiness": readiness,
                    "timestamp": int(time.time()),
                }

            from py_clob_client_v2.clob_types import MarketOrderArgsV2, OrderType

            client = self._get_client()
            order_args = MarketOrderArgsV2(
                token_id=token_id,
                amount=capped_size,
                side=side,
                price=float(price or 0),
                order_type=OrderType.FOK,
            )
            result = client.create_and_post_market_order(
                order_args,
                order_type=OrderType.FOK,
            )

            success = bool(result.get("success")) if isinstance(result, dict) else True
            order_id = (result.get("orderID") or result.get("order_id")) if isinstance(result, dict) else None
            status = result.get("status", "submitted") if isinstance(result, dict) else "submitted"

            return {
                "status": "success" if success else "error",
                "fill_status": status,
                "order_id": order_id,
                "token_id": token_id,
                "side": side,
                "size": capped_size,
                "price": price,
                "timestamp": int(time.time()),
                "response": result,
            }

        except Exception as e:
            logger.error("Error executing order: %s", e, exc_info=True)
            return {
                "status": "error",
                "error": str(e),
                "error_type": "execution",
                "token_id": token_id,
                "side": side,
                "size": size,
                "retry_recommended": False,
            }

    def execute_limit_order(
        self,
        token_id: str,
        side: str,
        size: float,
        price: float,
        post_only: bool = True,
        expiration: int = 0,
        timeout: int = 10,
    ) -> Dict:
        """
        Place a post-only GTC limit order at the given price.

        Parameters
        ----------
        token_id   : Conditional token asset ID.
        side       : "BUY" or "SELL".
        size       : Size in conditional-token shares.
        price      : Limit price (0.01–0.99).
        post_only  : If True, order is rejected if it would cross (no taker fill).
                     Keeps costs at maker-fee level.
        expiration : Unix timestamp after which order expires (0 = never).

        Returns
        -------
        dict with keys: status, order_id, token_id, side, size, price, timestamp
        """
        try:
            if size <= 0:
                raise ValueError(f"Order size must be positive, got {size}")
            if not (0.01 <= price <= 0.99):
                raise ValueError(f"Limit price out of range [0.01, 0.99]: {price}")

            side = side.upper()
            if side not in {"BUY", "SELL"}:
                raise ValueError(f"Side must be BUY or SELL, got {side}")

            capped_size = min(float(size), self.max_order_size)
            if capped_size < float(size):
                logger.warning(
                    "Capped limit order size from %.4f to %.4f",
                    size, capped_size,
                )

            readiness = self.validate_live_ready()
            if not readiness["ready"]:
                return {
                    "status": "dry_run" if self.dry_run else "blocked",
                    "token_id": token_id,
                    "side": side,
                    "size": capped_size,
                    "price": price,
                    "order_type": "LIMIT",
                    "post_only": post_only,
                    "readiness": readiness,
                    "timestamp": int(time.time()),
                }

            from py_clob_client_v2.clob_types import OrderArgsV2, OrderType

            client = self._get_client()
            order_args = OrderArgsV2(
                token_id=token_id,
                price=price,
                size=capped_size,
                side=side,
                expiration=expiration,
            )
            result = client.create_and_post_order(
                order_args,
                order_type=OrderType.GTC,
                post_only=post_only,
            )

            success = bool(result.get("success")) if isinstance(result, dict) else True
            order_id = (
                result.get("orderID") or result.get("order_id")
            ) if isinstance(result, dict) else None
            status = result.get("status", "submitted") if isinstance(result, dict) else "submitted"

            logger.info(
                "Limit order placed: %s %s @ %.4f size=%.4f order_id=%s",
                side, token_id[:12], price, capped_size, order_id,
            )

            return {
                "status": "success" if success else "error",
                "fill_status": status,
                "order_id": order_id,
                "token_id": token_id,
                "side": side,
                "size": capped_size,
                "price": price,
                "order_type": "LIMIT",
                "post_only": post_only,
                "timestamp": int(time.time()),
                "response": result,
            }

        except Exception as e:
            logger.error("Error placing limit order: %s", e, exc_info=True)
            return {
                "status": "error",
                "error": str(e),
                "error_type": "execution",
                "order_type": "LIMIT",
                "token_id": token_id,
                "side": side,
                "size": size,
                "price": price,
                "retry_recommended": False,
            }


    def cancel_order(self, order_id: str, timeout: int = 10) -> Dict:
        """Cancel an existing order."""
        try:
            if not order_id:
                raise ValueError("Order ID required for cancellation")
            readiness = self.validate_live_ready()
            if not readiness["ready"]:
                return {
                    "status": "dry_run" if self.dry_run else "blocked",
                    "order_id": order_id,
                    "readiness": readiness,
                }
            result = self._get_client().cancel_order(order_id)
            return {
                "status": "success",
                "order_id": order_id,
                "timestamp": int(time.time()),
                "response": result,
            }
        except Exception as e:
            logger.error("Error cancelling order: %s", e, exc_info=True)
            return {
                "status": "error",
                "error": str(e),
                "error_type": "execution",
                "order_id": order_id,
            }

    def get_order_status(self, order_id: str, timeout: int = 10) -> Dict:
        """Get status of an existing order."""
        try:
            if not order_id:
                raise ValueError("Order ID required")
            result = self._get_client().get_order(order_id)
            return {
                "status": "success",
                "order_id": order_id,
                "order_data": result,
                "timestamp": int(time.time()),
            }
        except Exception as e:
            logger.error("Error fetching order status: %s", e, exc_info=True)
            return {
                "status": "error",
                "error": str(e),
                "error_type": "execution",
                "order_id": order_id,
            }


    def get_collateral_balance(self) -> float:
        """Fetch the wallet collateral (USDC) balance via py_clob_client_v2."""
        try:
            readiness = self.validate_live_ready()
            if not readiness["ready"]:
                return 0.0

            from py_clob_client_v2.clob_types import BalanceAllowanceParams, AssetType
            client = self._get_client()
            params = BalanceAllowanceParams(asset_type=AssetType.COLLATERAL)
            data = client.get_balance_allowance(params)
            
            balance = float(data.get("balance", 0.0))
            return balance
        except Exception as e:
            logger.error("Error fetching collateral balance: %s", e, exc_info=True)
            return 0.0


def execute_market_order(token_id: str, side: str, size: float):
    """Legacy function for backward compatibility."""
    engine = ExecutionEngine()
    return engine.execute_market_order(token_id, side, size)
