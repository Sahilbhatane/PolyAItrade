"""HTTP endpoints for offline RL training and checkpoint management."""

from __future__ import annotations

from pathlib import Path

import yaml
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix="/rl", tags=["rl"])


def _load_rl_section() -> dict:
    path = Path("ml_config.yaml")
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data.get("rl", {}) if isinstance(data, dict) else {}


class TrainRequest(BaseModel):
    timesteps: int = Field(default=256, ge=64, le=500_000)


@router.post("/train")
def rl_train(body: TrainRequest | None = None):
    """Trigger a short PPO training run (offline)."""
    body = body or TrainRequest()
    cfg = _load_rl_section()
    cfg.setdefault("checkpoint_dir", "models/rl")
    cfg.setdefault("seed", 42)
    try:
        from ai_trader.rl.trainer import train_ppo_checkpoint

        path = train_ppo_checkpoint(cfg, total_timesteps=body.timesteps)
        return {"status": "ok", "checkpoint": str(path)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/status")
def rl_status():
    cfg = _load_rl_section()
    ckpt = Path(cfg.get("checkpoint_dir", "models/rl")) / "ppo_polyvitrade.zip"
    return {"checkpoint_exists": ckpt.exists(), "path": str(ckpt.resolve())}


@router.post("/rollback")
def rl_rollback():
    """Delete latest checkpoint as a simple rollback hook."""
    cfg = _load_rl_section()
    ckpt = Path(cfg.get("checkpoint_dir", "models/rl")) / "ppo_polyvitrade.zip"
    if ckpt.exists():
        ckpt.unlink()
        return {"status": "rolled_back", "removed": str(ckpt)}
    return {"status": "noop", "detail": "checkpoint missing"}
