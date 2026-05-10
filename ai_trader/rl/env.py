"""Gymnasium environment for bounded strategy-parameter adjustments (offline use)."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class TradingEnv(gym.Env):
    """Synthetic MDP for smoke-testing PPO; real deployments feed replay-derived observations."""

    metadata = {"render_modes": []}

    def __init__(self, cfg: dict[str, Any] | None = None):
        super().__init__()
        self._cfg = cfg or {}
        self._seed = int(self._cfg.get("seed", 42))
        self._rng = np.random.default_rng(self._seed)
        self._obs_dim = int(self._cfg.get("obs_dim", 16))
        self._n_strategies = int(self._cfg.get("n_strategies", 6))
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(self._obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self._n_strategies + 1,), dtype=np.float32
        )
        self._state_vec = np.zeros(self._obs_dim, dtype=np.float32)
        self._weights = np.ones(self._n_strategies, dtype=np.float32) / self._n_strategies

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._state_vec = self._rng.normal(0, 0.15, size=self._obs_dim).astype(np.float32)
        self._weights = np.ones(self._n_strategies, dtype=np.float32) / self._n_strategies
        return self._state_vec.copy(), {}

    def step(self, action: np.ndarray):
        action = np.clip(action.astype(np.float32), -1.0, 1.0)
        deltas = action[: self._n_strategies] * float(self._cfg.get("max_weight_delta", 0.15))
        self._weights = np.clip(self._weights * (1.0 + deltas), 0.05, 0.6)
        self._weights /= max(float(self._weights.sum()), 1e-9)

        conf_delta = float(action[-1]) * float(self._cfg.get("max_confidence_delta", 0.05))

        reward = float(self._rng.normal(0.008, 0.015))
        reward -= 0.02 * float(np.mean(np.abs(deltas)))
        reward -= 0.01 * abs(conf_delta)

        self._state_vec = self._rng.normal(0, 0.15, size=self._obs_dim).astype(np.float32)
        terminated = False
        truncated = False
        info = {"weights": self._weights.copy(), "confidence_delta": conf_delta}
        return self._state_vec.copy(), reward, terminated, truncated, info
