"""Angel One (SmartAPI) broker integration.

Implements BaseBroker for live trading via Angel One's SmartAPI.
Includes retry logic, graceful failure handling, and order verification.

Requirements:
  pip install smartapi-python pyotp

Environment variables / config needed:
  - ANGEL_API_KEY
  - ANGEL_CLIENT_ID
  - ANGEL_PASSWORD
  - ANGEL_TOTP_SECRET (for TOTP-based 2FA)
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from typing import Any

from ai_trader.broker.base import BaseBroker, Order, OrderSide, OrderType, OrderStatus
from ai_trader.logs import get_logger

logger = get_logger(__name__)

# SmartAPI order variety/product mappings
_ORDER_TYPE_MAP = {
    OrderType.MARKET: "MARKET",
    OrderType.LIMIT: "LIMIT",
    OrderType.STOP_LOSS: "STOPLOSS_MARKET",
    OrderType.STOP_LIMIT: "STOPLOSS_LIMIT",
}

_TRANSACTION_MAP = {
    OrderSide.BUY: "BUY",
    OrderSide.SELL: "SELL",
}


class AngelOneBroker(BaseBroker):
    """Live broker using Angel One SmartAPI.

    Features:
    - Automatic session management with TOTP 2FA
    - Configurable retry logic for all API calls
    - Order verification after placement (confirms fill)
    - Graceful degradation on API failures
    """

    def __init__(
        self,
        api_key: str,
        client_id: str,
        password: str,
        totp_secret: str,
        max_retries: int = 3,
        retry_delay_s: float = 1.0,
        order_verify_delay_s: float = 0.5,
        product_type: str = "INTRADAY",
        exchange: str = "NSE",
    ):
        self._api_key = api_key
        self._client_id = client_id
        self._password = password
        self._totp_secret = totp_secret
        self._max_retries = max_retries
        self._retry_delay_s = retry_delay_s
        self._order_verify_delay_s = order_verify_delay_s
        self._product_type = product_type
        self._exchange = exchange

        self._session: Any = None
        self._auth_token: str | None = None
        self._last_login: float = 0.0
        self._session_ttl_s = 3600  # re-login after 1 hour

    async def _ensure_session(self) -> None:
        """Lazily initialize or refresh the SmartAPI session."""
        now = time.time()
        if self._session is not None and (now - self._last_login) < self._session_ttl_s:
            return

        try:
            from SmartApi import SmartConnect
            import pyotp
        except ImportError as e:
            raise RuntimeError(
                "smartapi-python and pyotp are required for Angel One integration. "
                "Install with: pip install smartapi-python pyotp"
            ) from e

        totp = pyotp.TOTP(self._totp_secret).now()
        obj = SmartConnect(api_key=self._api_key)

        data = await self._retry(
            lambda: obj.generateSession(self._client_id, self._password, totp),
            context="login",
        )

        if data is None or data.get("status") is False:
            raise ConnectionError(f"Angel One login failed: {data}")

        self._session = obj
        self._auth_token = data["data"].get("jwtToken")
        self._last_login = now
        logger.info("angelone_session_established", client_id=self._client_id)

    async def place_order(self, order: Order) -> Order:
        """Place an order via SmartAPI with retry and verification."""
        await self._ensure_session()

        order_params = self._build_order_params(order)
        logger.info("placing_order", symbol=order.symbol, side=order.side.value, qty=order.quantity)

        response = await self._retry(
            lambda: self._session.placeOrder(order_params),
            context=f"place_order({order.symbol})",
        )

        if response is None:
            order.status = OrderStatus.REJECTED
            order.metadata["reject_reason"] = "API call failed after retries"
            return order

        broker_order_id = response

        if not broker_order_id:
            order.status = OrderStatus.REJECTED
            order.metadata["reject_reason"] = "No order ID returned"
            return order

        order.metadata["broker_order_id"] = broker_order_id

        # Verify the order status after a short delay
        await asyncio.sleep(self._order_verify_delay_s)
        verified_status = await self._verify_order(broker_order_id)

        order.status = verified_status
        if verified_status == OrderStatus.FILLED:
            fill_info = await self._get_fill_price(broker_order_id)
            order.filled_price = fill_info.get("price", order.price)
            order.filled_at = datetime.now(timezone.utc)

        logger.info(
            "order_placed",
            broker_order_id=broker_order_id,
            status=order.status.value,
        )
        return order

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order by its broker order ID."""
        await self._ensure_session()

        response = await self._retry(
            lambda: self._session.cancelOrder(order_id, "NORMAL"),
            context=f"cancel_order({order_id})",
        )
        success = response is not None
        logger.info("order_cancel_attempt", order_id=order_id, success=success)
        return success

    async def get_order_status(self, order_id: str) -> OrderStatus:
        """Query the current status of a specific order."""
        await self._ensure_session()

        order_book = await self._retry(
            lambda: self._session.orderBook(),
            context="get_order_status",
        )

        if not order_book or not order_book.get("data"):
            return OrderStatus.PENDING

        for entry in order_book["data"]:
            if str(entry.get("orderid")) == str(order_id):
                return self._map_status(entry.get("orderstatus", ""))

        return OrderStatus.PENDING

    async def get_positions(self) -> list[dict[str, Any]]:
        """Fetch all open positions from Angel One."""
        await self._ensure_session()

        response = await self._retry(
            lambda: self._session.position(),
            context="get_positions",
        )

        if not response or not response.get("data"):
            return []

        positions = []
        for pos in response["data"]:
            if int(pos.get("netqty", 0)) != 0:
                positions.append({
                    "symbol": pos.get("tradingsymbol", ""),
                    "exchange": pos.get("exchange", ""),
                    "quantity": int(pos.get("netqty", 0)),
                    "avg_price": float(pos.get("averageprice", 0)),
                    "pnl": float(pos.get("pnl", 0)),
                    "product": pos.get("producttype", ""),
                })

        return positions

    async def get_account_balance(self) -> dict[str, float]:
        """Fetch account balance and margin info."""
        await self._ensure_session()

        response = await self._retry(
            lambda: self._session.rmsLimit(),
            context="get_account_balance",
        )

        if not response or not response.get("data"):
            return {"cash": 0.0, "margin_used": 0.0, "margin_available": 0.0}

        data = response["data"]
        return {
            "cash": float(data.get("net", 0)),
            "margin_used": float(data.get("utiliseddebits", 0)),
            "margin_available": float(data.get("availablecash", 0)),
            "collateral": float(data.get("collateral", 0)),
        }

    async def health_check(self) -> bool:
        """Check connectivity to Angel One API."""
        try:
            await self._ensure_session()
            profile = await self._retry(
                lambda: self._session.getProfile(self._auth_token),
                context="health_check",
            )
            return profile is not None and profile.get("status") is not False
        except Exception:
            return False

    def _build_order_params(self, order: Order) -> dict[str, str]:
        """Map internal Order to SmartAPI order parameters."""
        params = {
            "variety": "NORMAL",
            "tradingsymbol": order.symbol,
            "symboltoken": order.metadata.get("token", ""),
            "transactiontype": _TRANSACTION_MAP[order.side],
            "exchange": self._exchange,
            "ordertype": _ORDER_TYPE_MAP.get(order.order_type, "MARKET"),
            "producttype": self._product_type,
            "duration": "DAY",
            "quantity": str(order.quantity),
        }

        if order.order_type in (OrderType.LIMIT, OrderType.STOP_LIMIT) and order.price:
            params["price"] = str(order.price)
        else:
            params["price"] = "0"

        if order.order_type in (OrderType.STOP_LOSS, OrderType.STOP_LIMIT) and order.stop_loss:
            params["triggerprice"] = str(order.stop_loss)
        else:
            params["triggerprice"] = "0"

        return params

    async def _verify_order(self, broker_order_id: str) -> OrderStatus:
        """Verify order status after placement (handles partial fills)."""
        for _ in range(3):
            status = await self.get_order_status(broker_order_id)
            if status in (OrderStatus.FILLED, OrderStatus.REJECTED, OrderStatus.CANCELLED):
                return status
            await asyncio.sleep(self._order_verify_delay_s)

        logger.warning("order_verification_timeout", broker_order_id=broker_order_id)
        return OrderStatus.SUBMITTED

    async def _get_fill_price(self, broker_order_id: str) -> dict[str, float]:
        """Get actual fill price from order book."""
        order_book = await self._retry(
            lambda: self._session.orderBook(),
            context="get_fill_price",
        )

        if order_book and order_book.get("data"):
            for entry in order_book["data"]:
                if str(entry.get("orderid")) == str(broker_order_id):
                    return {"price": float(entry.get("averageprice", 0))}

        return {"price": 0.0}

    async def _retry(self, fn, context: str = "") -> Any:
        """Execute a function with exponential backoff retry.

        Handles API downtime gracefully — logs and returns None on exhaustion.
        """
        last_error = None
        for attempt in range(self._max_retries):
            try:
                result = fn()
                return result
            except Exception as e:
                last_error = e
                delay = self._retry_delay_s * (2 ** attempt)
                logger.warning(
                    "api_retry",
                    context=context,
                    attempt=attempt + 1,
                    max_retries=self._max_retries,
                    error=str(e),
                    next_delay=delay,
                )
                await asyncio.sleep(delay)

        logger.error("api_exhausted_retries", context=context, error=str(last_error))
        return None

    @staticmethod
    def _map_status(angel_status: str) -> OrderStatus:
        """Map Angel One order status strings to our OrderStatus enum."""
        mapping = {
            "complete": OrderStatus.FILLED,
            "rejected": OrderStatus.REJECTED,
            "cancelled": OrderStatus.CANCELLED,
            "open": OrderStatus.SUBMITTED,
            "pending": OrderStatus.PENDING,
            "trigger pending": OrderStatus.PENDING,
        }
        return mapping.get(angel_status.lower(), OrderStatus.PENDING)
