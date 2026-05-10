from pathlib import Path

import pytest


@pytest.mark.asyncio
async def test_rl_rollback_endpoint_removes_file(tmp_path, monkeypatch):
    from httpx import ASGITransport, AsyncClient

    from ai_trader.app import create_app

    ckpt_dir = tmp_path / "rl"
    ckpt_dir.mkdir()
    p = ckpt_dir / "ppo_polyvitrade.zip"
    p.write_bytes(b"x")

    import ai_trader.routes.rl as rl_mod

    monkeypatch.setattr(
        rl_mod,
        "_load_rl_section",
        lambda: {"checkpoint_dir": str(ckpt_dir)},
    )

    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/rl/rollback")
        assert resp.status_code == 200
        assert not p.exists()
