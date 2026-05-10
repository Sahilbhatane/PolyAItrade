"""Concrete strategy voters (one module per voter for maintainability)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ai_trader.strategies.composite_signal import weighted_signal_decision
from ai_trader.strategies.indicators import Indicators
from ai_trader.strategies.vote_types import VoteContext, VoteResult


def _hold(name: str, reason: str) -> VoteResult:
    return VoteResult(name=name, action="HOLD", confidence=0.0, reasoning=reason)


def vote_rule_based_v1(ctx: VoteContext, params: dict[str, Any]) -> VoteResult:
    """Weighted composite of rsi/macd/ma/vwap — same semantics as StrategyAgent."""
    cfg = ctx.strategy_config
    if cfg is None:
        return _hold("rule_based_v1", "no_strategy_config")

    weights = {
        "rsi": cfg.weights.rsi,
        "macd": cfg.weights.macd,
        "ma_crossover": cfg.weights.ma_crossover,
        "vwap": cfg.weights.vwap,
    }
    thr = float(params.get("confidence_threshold", cfg.overtrading.min_confidence))
    d = weighted_signal_decision(ctx.signals, ctx.bar_index, weights, thr)
    return VoteResult(
        name="rule_based_v1",
        action=d["action"],
        confidence=float(d["confidence"]),
        reasoning=d["reasoning"],
        signals_used={k: float(v) for k, v in d["signals"].items()},
    )


def vote_momentum_breakout(ctx: VoteContext, params: dict[str, Any]) -> VoteResult:
    lookback = int(params.get("lookback", 20))
    breakout_pct = float(params.get("breakout_pct", 0.005))
    vol_z_min = float(params.get("volume_z_min", 0.25))
    i = ctx.bar_index
    if i < lookback + 2:
        return _hold("momentum_breakout", "warmup")

    window = ctx.df.iloc[i - lookback : i]
    hh = float(window["high"].max())
    ll = float(window["low"].min())
    close = float(ctx.df["close"].iloc[i])
    vol_z = float(ctx.signals["volume_z"].iloc[i])

    if hh <= 0:
        return _hold("momentum_breakout", "invalid_window")

    up_break = close > hh * (1.0 + breakout_pct) and vol_z >= vol_z_min
    down_break = close < ll * (1.0 - breakout_pct) and vol_z >= vol_z_min

    if up_break:
        conf = float(min(1.0, (close / hh - 1.0) / max(breakout_pct, 1e-9) * 0.35 + 0.35))
        return VoteResult(
            name="momentum_breakout",
            action="BUY",
            confidence=conf,
            reasoning=f"up_break hh={hh:.4f} close={close:.4f} vol_z={vol_z:.2f}",
            signals_used={"volume_z": vol_z},
        )
    if down_break:
        conf = float(min(1.0, (1.0 - close / ll) / max(breakout_pct, 1e-9) * 0.35 + 0.35))
        return VoteResult(
            name="momentum_breakout",
            action="SELL",
            confidence=conf,
            reasoning=f"down_break ll={ll:.4f} close={close:.4f} vol_z={vol_z:.2f}",
            signals_used={"volume_z": vol_z},
        )
    return _hold("momentum_breakout", "no_breakout")


def vote_mean_reversion(ctx: VoteContext, params: dict[str, Any]) -> VoteResult:
    rsi_low = float(params.get("rsi_low", 35.0))
    rsi_high = float(params.get("rsi_high", 65.0))
    i = ctx.bar_index
    rsi_series = ctx.signals.get("raw_rsi")
    if rsi_series is None or i >= len(rsi_series):
        return _hold("mean_reversion", "no_rsi")
    rsi_val = float(rsi_series.iloc[i])
    if np.isnan(rsi_val):
        return _hold("mean_reversion", "nan_rsi")

    if rsi_val <= rsi_low:
        conf = float(min(1.0, (rsi_low - rsi_val) / max(rsi_low, 1e-9)))
        return VoteResult(
            name="mean_reversion",
            action="BUY",
            confidence=conf,
            reasoning=f"rsi_oversold rsi={rsi_val:.2f}",
            signals_used={"raw_rsi": rsi_val},
        )
    if rsi_val >= rsi_high:
        conf = float(min(1.0, (rsi_val - rsi_high) / max(100.0 - rsi_high, 1e-9)))
        return VoteResult(
            name="mean_reversion",
            action="SELL",
            confidence=conf,
            reasoning=f"rsi_overbought rsi={rsi_val:.2f}",
            signals_used={"raw_rsi": rsi_val},
        )
    return _hold("mean_reversion", "rsi_neutral")


def vote_vwap_reversion(ctx: VoteContext, params: dict[str, Any]) -> VoteResult:
    deviation_pct = float(params.get("deviation_pct", 0.008))
    atr_cap_mult = float(params.get("atr_volatility_cap_mult", 3.5))
    i = ctx.bar_index
    sub = ctx.df.iloc[: i + 1]
    if len(sub) < 5:
        return _hold("vwap_reversion", "warmup")

    vwap_series = Indicators.vwap(sub)
    close = float(sub["close"].iloc[-1])
    vwap_val = float(vwap_series.iloc[-1])
    atr = ctx.signals.get("atr")
    atr_val = float(atr.iloc[i]) if atr is not None and i < len(atr) else 0.0

    if vwap_val <= 0 or np.isnan(vwap_val):
        return _hold("vwap_reversion", "bad_vwap")

    dev = (close - vwap_val) / vwap_val
    # ATR volatility filter: skip extreme spikes
    if atr_val > 0 and abs(close - vwap_val) > atr_cap_mult * atr_val:
        return _hold("vwap_reversion", "atr_volatility_filter")

    if dev <= -deviation_pct:
        conf = float(min(1.0, abs(dev) / max(deviation_pct, 1e-9) * 0.5))
        return VoteResult(
            name="vwap_reversion",
            action="BUY",
            confidence=conf,
            reasoning=f"below_vwap dev={dev:.4f}",
            signals_used={"vwap_dev": dev},
        )
    if dev >= deviation_pct:
        conf = float(min(1.0, abs(dev) / max(deviation_pct, 1e-9) * 0.5))
        return VoteResult(
            name="vwap_reversion",
            action="SELL",
            confidence=conf,
            reasoning=f"above_vwap dev={dev:.4f}",
            signals_used={"vwap_dev": dev},
        )
    return _hold("vwap_reversion", "near_vwap")


def vote_ma_crossover(ctx: VoteContext, params: dict[str, Any]) -> VoteResult:
    """Dedicated MA trend voter using precomputed ma_crossover signal."""
    i = ctx.bar_index
    series = ctx.signals.get("ma_crossover")
    if series is None or i >= len(series):
        return _hold("ma_crossover", "no_signal")
    val = float(series.iloc[i])
    if pd.isna(val):
        return _hold("ma_crossover", "nan")

    deadband = float(params.get("deadband", 0.05))
    if abs(val) < deadband:
        return _hold("ma_crossover", "weak_trend")

    action = "BUY" if val > 0 else "SELL"
    conf = float(min(1.0, abs(val)))
    return VoteResult(
        name="ma_crossover",
        action=action,
        confidence=conf,
        reasoning=f"ma_signal={val:.3f}",
        signals_used={"ma_crossover": val},
    )


def vote_fibonacci_confluence(ctx: VoteContext, params: dict[str, Any]) -> VoteResult:
    """Fibonacci zones require multi-signal confirmation (never standalone)."""
    swing_window = int(params.get("swing_window", 20))
    tolerance_pct = float(params.get("tolerance_pct", 0.004))
    min_confirmations = int(params.get("min_confirmations", 2))
    i = ctx.bar_index
    if i < swing_window + 2:
        return _hold("fibonacci_confluence", "warmup")

    sub = ctx.df.iloc[i - swing_window : i + 1]
    swing_high = float(sub["high"].max())
    swing_low = float(sub["low"].min())
    diff = swing_high - swing_low
    if diff <= 0:
        return _hold("fibonacci_confluence", "flat_range")

    price = float(ctx.df["close"].iloc[i])
    levels = {
        "fib_382": swing_high - 0.382 * diff,
        "fib_500": swing_high - 0.500 * diff,
        "fib_618": swing_high - 0.618 * diff,
    }
    near = any(abs(price - lv) / max(price, 1e-9) <= tolerance_pct for lv in levels.values())
    if not near:
        return _hold("fibonacci_confluence", "not_near_fib")

    confirmations = 0
    signals_used: dict[str, float] = {}

    sub_v = ctx.df.iloc[: i + 1]
    vwap_series = Indicators.vwap(sub_v)
    vwap_val = float(vwap_series.iloc[-1])
    if vwap_val > 0:
        dev = (price - vwap_val) / vwap_val
        signals_used["vwap_dev"] = dev
        if price <= vwap_val * (1 - tolerance_pct):
            confirmations += 1  # VWAP confirms bullish bounce context

    vol_z = float(ctx.signals["volume_z"].iloc[i])
    signals_used["volume_z"] = vol_z
    if vol_z >= float(params.get("volume_z_confirm", 0.35)):
        confirmations += 1

    atr_series = ctx.signals.get("atr")
    atr_pct_series = atr_series / ctx.df["close"].replace(0, np.nan)
    atr_pct = float(atr_pct_series.iloc[i]) if atr_series is not None else 0.0
    signals_used["atr_pct"] = atr_pct
    if atr_pct <= float(params.get("atr_pct_max", 0.045)):
        confirmations += 1

    ma_sig = float(ctx.signals["ma_crossover"].iloc[i])
    signals_used["ma_crossover"] = ma_sig
    if ma_sig >= float(params.get("ma_bull_confirm", 0.15)):
        confirmations += 1
    if ma_sig <= -float(params.get("ma_bear_confirm", 0.15)):
        confirmations += 1

    if confirmations < min_confirmations:
        return _hold(
            "fibonacci_confluence",
            f"insufficient_confirmations ({confirmations}<{min_confirmations})",
        )

    # Direction from retracement context: above midpoint favors shorts from high, below favors longs
    mid = swing_low + 0.5 * diff
    action = "BUY" if price < mid else "SELL"
    conf = float(min(1.0, 0.35 + 0.15 * confirmations))
    return VoteResult(
        name="fibonacci_confluence",
        action=action,
        confidence=conf,
        reasoning=f"fib_confluence confirmations={confirmations}",
        signals_used=signals_used,
    )
