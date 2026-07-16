"""Transport layer between the TUI and the FastAPI backend.

Two responsibilities:
- Read-model queries + intent submission over plain HTTP (async).
- A live Server-Sent Events subscription bridged from the backend EventBus.

The client is deliberately swappable and injectable: production uses a real
``httpx.AsyncClient`` pointed at the server URL; tests inject an ASGI-backed
client so the whole stack runs in-process without a socket.
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator

import httpx

DEFAULT_BASE_URL = "http://localhost:8000"


class TransportError(Exception):
    """Raised when a backend call fails (network, 5xx, or invalid response)."""


class BackendTransport:
    """Async HTTP + SSE client for the ``/tui`` and ``/trading`` endpoints."""

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        client: httpx.AsyncClient | None = None,
        timeout: float = 10.0,
    ):
        self._base_url = base_url.rstrip("/")
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(base_url=self._base_url, timeout=timeout)
        self._timeout = timeout

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    # --- Read model ----------------------------------------------------

    async def snapshot(self) -> dict[str, Any]:
        return await self._get("/tui/snapshot")

    async def diagnostics(self) -> dict[str, Any]:
        return await self._get("/tui/diagnostics")

    async def strategies(self) -> dict[str, Any]:
        return await self._get("/tui/strategies")

    async def agents(self) -> dict[str, Any]:
        return await self._get("/tui/agents")

    async def rl(self) -> dict[str, Any]:
        return await self._get("/tui/rl")

    async def integrations(self) -> dict[str, Any]:
        return await self._get("/tui/config/integrations")

    async def positions(self) -> list[dict[str, Any]]:
        return await self._get("/trading/positions")

    async def balance(self) -> dict[str, Any]:
        return await self._get("/trading/balance")

    async def pending_approvals(self) -> list[dict[str, Any]]:
        return await self._get("/trading/approvals/pending")

    async def approval_history(self) -> list[dict[str, Any]]:
        return await self._get("/trading/approvals/history")

    async def kill_switch_status(self) -> dict[str, Any]:
        return await self._get("/trading/kill-switch/status")

    async def logs(
        self,
        limit: int = 200,
        cursor: int | None = None,
        level: str | None = None,
        component: str | None = None,
        search: str | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"limit": limit}
        if cursor is not None:
            params["cursor"] = cursor
        if level:
            params["level"] = level
        if component:
            params["component"] = component
        if search:
            params["search"] = search
        return await self._get("/tui/logs", params=params)

    # --- Intents (write) ----------------------------------------------

    async def submit_trade(self, intent: dict[str, Any]) -> dict[str, Any]:
        return await self._post("/tui/trade/submit", intent)

    async def respond_approval(self, request_id: str, action: str, reason: str = "") -> dict[str, Any]:
        return await self._post(
            "/trading/approvals/respond",
            {"request_id": request_id, "action": action, "reason": reason},
        )

    async def set_kill_switch(self, action: str, reason: str = "") -> dict[str, Any]:
        return await self._post("/trading/kill-switch", {"action": action, "reason": reason})

    # --- Event stream --------------------------------------------------

    async def stream_events(self) -> AsyncIterator[dict[str, Any]]:
        """Yield parsed SSE events from the backend until the stream closes."""
        try:
            async with self._client.stream("GET", "/tui/events", timeout=None) as response:
                if response.status_code != 200:
                    raise TransportError(f"event stream returned {response.status_code}")
                async for line in response.aiter_lines():
                    if not line or line.startswith(":"):
                        continue  # comment / heartbeat
                    if line.startswith("data:"):
                        raw = line[len("data:"):].strip()
                        try:
                            yield json.loads(raw)
                        except json.JSONDecodeError:
                            continue
        except httpx.HTTPError as e:
            raise TransportError(str(e)) from e

    # --- Internals -----------------------------------------------------

    async def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        return await self._request_with_retry("GET", path, params=params)

    async def _post(self, path: str, payload: dict[str, Any]) -> Any:
        return await self._request_with_retry("POST", path, json=payload)

    async def _request_with_retry(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
        max_attempts: int = 3,
    ) -> Any:
        """Retry transient network/5xx failures with short backoff."""
        import asyncio

        delay = 0.25
        last_err: Exception | None = None
        for attempt in range(max_attempts):
            try:
                if method == "GET":
                    resp = await self._client.get(path, params=params)
                else:
                    resp = await self._client.post(path, json=json)
                return self._handle(resp)
            except TransportError as e:
                last_err = e
                # Do not retry client errors (4xx) — only server/network.
                if "server error" not in str(e) and attempt == 0:
                    raise
            except httpx.HTTPError as e:
                last_err = TransportError(str(e))
            if attempt < max_attempts - 1:
                await asyncio.sleep(delay)
                delay = min(delay * 2, 2.0)
        raise last_err or TransportError("request failed")

    @staticmethod
    def _handle(resp: httpx.Response) -> Any:
        if resp.status_code >= 500:
            raise TransportError(f"server error {resp.status_code}")
        try:
            body = resp.json()
        except ValueError as e:
            raise TransportError(f"invalid JSON response: {e}") from e
        if resp.status_code >= 400:
            detail = body.get("detail") if isinstance(body, dict) else body
            raise TransportError(f"{resp.status_code}: {detail}")
        return body
