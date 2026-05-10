"""Reward helpers for RL feedback (risk-adjusted, anti-overtrading)."""

from __future__ import annotations


def compute_reward(
    pnl: float,
    max_drawdown: float,
    trades_per_day: float,
    volatility_spike: float,
    excess_loss: float,
    *,
    lambda_dd: float = 1.5,
    lambda_trade: float = 0.08,
    lambda_vol: float = 0.5,
    lambda_loss: float = 1.0,
    eps: float = 1e-6,
) -> float:
    """Risk-adjusted reward with penalties for overtrading and tail risks."""
    dd = max(max_drawdown, eps)
    core = pnl / dd
    return (
        core
        - lambda_trade * trades_per_day
        - lambda_vol * volatility_spike
        - lambda_loss * excess_loss
    )
