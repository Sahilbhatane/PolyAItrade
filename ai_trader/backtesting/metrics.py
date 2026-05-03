"""Performance metrics for backtesting results."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class PerformanceMetrics:
    """Complete set of backtest performance metrics."""

    total_return: float
    annualized_return: float
    max_drawdown: float
    sharpe_ratio: float
    profit_factor: float
    win_rate: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    avg_trade_duration_bars: float


def calculate_total_return(equity_curve: list[float]) -> float:
    """(final - initial) / initial"""
    if len(equity_curve) < 2 or equity_curve[0] == 0:
        return 0.0
    return (equity_curve[-1] - equity_curve[0]) / equity_curve[0]


def calculate_max_drawdown(equity_curve: list[float]) -> float:
    """Maximum peak-to-trough decline as a fraction."""
    if len(equity_curve) < 2:
        return 0.0
    peak = equity_curve[0]
    max_dd = 0.0
    for value in equity_curve:
        if value > peak:
            peak = value
        drawdown = (peak - value) / peak if peak > 0 else 0.0
        max_dd = max(max_dd, drawdown)
    return max_dd


def calculate_sharpe_ratio(
    equity_curve: list[float],
    risk_free_rate: float = 0.05,
    trading_days: int = 252,
) -> float:
    """Annualized Sharpe ratio from daily equity curve."""
    if len(equity_curve) < 3:
        return 0.0
    arr = np.array(equity_curve, dtype=np.float64)
    returns = np.diff(arr) / arr[:-1]

    if len(returns) == 0:
        return 0.0

    excess_return = np.mean(returns) - (risk_free_rate / trading_days)
    std = np.std(returns, ddof=1)

    if std == 0 or math.isnan(std):
        return 0.0
    return float((excess_return / std) * math.sqrt(trading_days))


def calculate_profit_factor(trades: list[dict]) -> float:
    """Gross profit / gross loss. Returns inf if no losing trades."""
    gross_profit = sum(t["pnl"] for t in trades if t.get("pnl", 0) > 0)
    gross_loss = abs(sum(t["pnl"] for t in trades if t.get("pnl", 0) < 0))

    if gross_loss == 0:
        return float("inf") if gross_profit > 0 else 0.0
    return gross_profit / gross_loss


def compute_all_metrics(
    equity_curve: list[float],
    trades: list[dict],
    risk_free_rate: float = 0.05,
    trading_days: int = 252,
) -> PerformanceMetrics:
    """Compute full performance metrics from backtest output."""
    winning = [t for t in trades if t.get("pnl", 0) > 0]
    losing = [t for t in trades if t.get("pnl", 0) < 0]

    avg_win = np.mean([t["pnl"] for t in winning]) if winning else 0.0
    avg_loss = np.mean([t["pnl"] for t in losing]) if losing else 0.0
    largest_win = max((t["pnl"] for t in winning), default=0.0)
    largest_loss = min((t["pnl"] for t in losing), default=0.0)

    durations = [t.get("duration_bars", 0) for t in trades]
    avg_duration = float(np.mean(durations)) if durations else 0.0

    total = len(trades)
    total_return = calculate_total_return(equity_curve)
    n_days = max(len(equity_curve) - 1, 1)
    annualized = ((1 + total_return) ** (trading_days / n_days)) - 1 if total_return > -1 else -1.0

    return PerformanceMetrics(
        total_return=total_return,
        annualized_return=annualized,
        max_drawdown=calculate_max_drawdown(equity_curve),
        sharpe_ratio=calculate_sharpe_ratio(equity_curve, risk_free_rate, trading_days),
        profit_factor=calculate_profit_factor(trades),
        win_rate=len(winning) / total if total > 0 else 0.0,
        total_trades=total,
        winning_trades=len(winning),
        losing_trades=len(losing),
        avg_win=float(avg_win),
        avg_loss=float(avg_loss),
        largest_win=float(largest_win),
        largest_loss=float(largest_loss),
        avg_trade_duration_bars=avg_duration,
    )
