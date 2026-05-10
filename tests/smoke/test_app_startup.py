import pytest
from httpx import ASGITransport, AsyncClient

from ai_trader.app import create_app


@pytest.mark.asyncio
async def test_app_boot_health_ready():
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        h = await client.get("/health")
        assert h.status_code == 200
        r = await client.get("/ready")
        assert r.status_code == 200
