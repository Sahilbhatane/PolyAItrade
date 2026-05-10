"""PPO training wrapper with lightweight defaults for offline runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def train_ppo_checkpoint(cfg: dict[str, Any], total_timesteps: int = 512) -> Path:
    """Train PPO on TradingEnv and persist checkpoint under checkpoint_dir."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    from ai_trader.rl.env import TradingEnv

    ckpt_dir = Path(cfg.get("checkpoint_dir", "models/rl"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    def _make():
        return TradingEnv(cfg)

    env = DummyVecEnv([_make])
    model = PPO(
        "MlpPolicy",
        env,
        verbose=0,
        seed=int(cfg.get("seed", 42)),
        learning_rate=float(cfg.get("learning_rate", 3e-4)),
    )
    model.learn(total_timesteps=int(total_timesteps))
    path = ckpt_dir / "ppo_polyvitrade"
    model.save(str(path))
    return path
