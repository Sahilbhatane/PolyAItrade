"""Apply RL policy outputs as bounded proposals (never executes trades)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ai_trader.agents.state import StateKeys, StateManager
from ai_trader.logs import get_logger

logger = get_logger(__name__)

STRATEGY_NAMES_DEFAULT = [
    "rule_based_v1",
    "momentum_breakout",
    "mean_reversion",
    "vwap_reversion",
    "ma_crossover",
    "fibonacci_confluence",
]


class RLOptimizer:
    """Loads SB3 checkpoint (optional) and writes RL_WEIGHT_PROPOSAL only."""

    def __init__(self, cfg: dict[str, Any] | None = None):
        self._cfg = cfg or {}
        self._max_delta = float(self._cfg.get("max_weight_delta", 0.2))
        self._checkpoint_dir = Path(self._cfg.get("checkpoint_dir", "models/rl"))

    def bounded_blend(self, deltas: dict[str, float]) -> dict[str, float]:
        out = {}
        for k, v in deltas.items():
            out[k] = float(np.clip(v, -self._max_delta, self._max_delta))
        return out

    def propose_from_checkpoint(self, state: StateManager, checkpoint_stem: str | None = None) -> dict[str, Any]:
        """Run one policy forward pass if checkpoint exists; otherwise heuristic proposal."""
        stem = checkpoint_stem or "ppo_polyvitrade"
        path = self._checkpoint_dir / f"{stem}.zip"
        names = list(self._cfg.get("strategy_names", STRATEGY_NAMES_DEFAULT))

        if path.exists():
            try:
                from stable_baselines3 import PPO

                model = PPO.load(str(path))
                obs = np.zeros(model.observation_space.shape, dtype=np.float32)
                action, _ = model.predict(obs, deterministic=True)
                action = np.asarray(action, dtype=np.float32).reshape(-1)
            except Exception as e:
                logger.warning("rl_checkpoint_load_failed", error=str(e))
                action = None
        else:
            action = None

        if action is None or len(action) < len(names) + 1:
            # Heuristic zero-centered proposal for smoke tests
            proposal_vec = np.zeros(len(names) + 1, dtype=np.float32)
        else:
            proposal_vec = action

        deltas = {names[i]: float(proposal_vec[i]) * 0.05 for i in range(len(names))}
        deltas = self.bounded_blend(deltas)
        conf_delta = float(np.clip(float(proposal_vec[len(names)]), -1.0, 1.0)) * float(
            self._cfg.get("max_confidence_delta", 0.05)
        )

        payload = {
            "strategy_weights": deltas,
            "confidence_threshold_delta": conf_delta,
            "max_trades_per_day": self._cfg.get("max_trades_per_day_hint"),
            "source": "rl_optimizer",
            "checkpoint": str(path) if path.exists() else None,
        }
        self.validate_payload(payload)
        state.write_sync(StateKeys.RL_WEIGHT_PROPOSAL, payload, writer="rl_optimizer")
        logger.info("rl_proposal_written", keys=list(deltas.keys()))
        return payload

    def validate_payload(self, payload: dict[str, Any]) -> None:
        sw = payload.get("strategy_weights", {})
        if not isinstance(sw, dict):
            raise ValueError("invalid_strategy_weights")
        for k, v in sw.items():
            if abs(float(v)) > self._max_delta + 1e-9:
                raise ValueError(f"rl_delta_out_of_bounds:{k}:{v}")

    def rollback_prior_checkpoint(self, prior_path: Path | None = None) -> bool:
        """Remove latest checkpoint file if rollback requested (simple filesystem rollback)."""
        if prior_path and prior_path.exists():
            try:
                prior_path.unlink()
                logger.warning("rl_checkpoint_removed", path=str(prior_path))
                return True
            except OSError:
                return False
        return False
