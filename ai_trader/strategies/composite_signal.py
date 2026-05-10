"""Shared weighted composite from indicator signals (used by StrategyAgent and voters)."""

from __future__ import annotations

from typing import Any

import pandas as pd


def weighted_signal_decision(
    signals: dict[str, Any],
    bar_index: int,
    weights: dict[str, float],
    confidence_threshold: float,
) -> dict[str, Any]:
    """Weighted composite of indicator signals → action + confidence (same semantics as StrategyAgent)."""
    component_values: dict[str, float] = {}
    for indicator, weight in weights.items():
        series = signals.get(indicator)
        if series is None or bar_index >= len(series):
            component_values[indicator] = 0.0
            continue
        val = series.iloc[bar_index]
        component_values[indicator] = float(val) if not pd.isna(val) else 0.0

    total_weight = sum(weights.values())
    if total_weight == 0:
        composite = 0.0
    else:
        composite = sum(component_values[k] * weights[k] for k in weights) / total_weight

    confidence = abs(composite)

    if confidence < confidence_threshold:
        action = "HOLD"
        reasoning = f"Low confidence ({confidence:.3f} < {confidence_threshold})"
    elif composite > 0:
        action = "BUY"
        reasoning = f"Bullish composite={composite:.3f}"
    else:
        action = "SELL"
        reasoning = f"Bearish composite={composite:.3f}"

    close_series = signals.get("close")
    current_price = float(close_series.iloc[bar_index]) if close_series is not None else 0.0

    atr_series = signals.get("atr")
    atr_val = (
        float(atr_series.iloc[bar_index])
        if atr_series is not None and not pd.isna(atr_series.iloc[bar_index])
        else None
    )

    return {
        "action": action,
        "confidence": confidence,
        "composite_score": composite,
        "reasoning": reasoning,
        "bar_index": bar_index,
        "current_price": current_price,
        "atr": atr_val,
        "signals": component_values,
    }
