"""RegimeDetectionAgent — rolling-window market state classification (no future leakage)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ai_trader.agents.base import BaseAgent
from ai_trader.agents.event_bus import Event, EventBus, EventType
from ai_trader.agents.state import StateKeys, StateManager
from ai_trader.strategies.indicators import Indicators


class RegimeDetectionAgent(BaseAgent):
    """Classifies regime using only data up to bar_index."""

    def __init__(
        self,
        event_bus: EventBus,
        state: StateManager,
        regime_config: dict[str, Any] | None = None,
        agent_id: str | None = None,
    ):
        super().__init__(agent_id=agent_id or "regime_detection_agent")
        self._bus = event_bus
        self._state = state
        self._cfg = regime_config or {}

    async def run(self, context: dict[str, Any] | None = None) -> dict[str, Any]:
        df = self._state.read(StateKeys.MARKET_DATA)
        signals = self._state.read(StateKeys.SIGNALS)
        if df is None or signals is None:
            raise ValueError("RegimeDetectionAgent requires MARKET_DATA and SIGNALS")

        ctx = context or {}
        bar_index = ctx.get("bar_index")
        if bar_index is None:
            bar_index = len(signals["close"]) - 1

        window = int(self._cfg.get("window_bars", 20))
        slope_period = int(self._cfg.get("sma_slope_period", 10))
        trend_thr = float(self._cfg.get("trend_slope_threshold", 0.0005))
        atr_pct_vol = float(self._cfg.get("atr_percentile_volatile", 0.75))
        vol_z_low = float(self._cfg.get("volume_z_low_liquidity", -1.25))
        sideways_atr_max = float(self._cfg.get("sideways_atr_pct_max", 0.55))

        close = df["close"].iloc[: bar_index + 1]
        if len(close) < max(window, slope_period) + 2:
            payload = {
                "label": "sideways",
                "confidence": 0.3,
                "features": {"reason": "warmup"},
                "bar_index": bar_index,
            }
            await self._state.write(StateKeys.REGIME, payload, writer=self.agent_id)
            return payload

        sma_win = min(20, len(close) - 1)
        sma_series = Indicators.sma(close, sma_win)
        sma_now = float(sma_series.iloc[-1])
        sma_prev = float(sma_series.iloc[-slope_period]) if len(sma_series) >= slope_period else sma_now
        denom = max(abs(sma_prev), 1e-9)
        slope = (sma_now - sma_prev) / denom

        atr_series = signals["atr"].iloc[: bar_index + 1]
        atr_pct = (atr_series / close.replace(0, np.nan)).iloc[-window:].dropna()
        cur_atr_pct = float((atr_series.iloc[bar_index] / max(float(close.iloc[-1]), 1e-9)))

        atr_percentile = 0.5
        if len(atr_pct) > 3:
            atr_percentile = float(np.mean(atr_pct.values <= cur_atr_pct))

        vol_z = float(signals["volume_z"].iloc[bar_index])
        ma_sig = float(signals["ma_crossover"].iloc[bar_index])

        features: dict[str, Any] = {
            "slope": slope,
            "atr_percentile": atr_percentile,
            "atr_pct_now": cur_atr_pct,
            "volume_z": vol_z,
            "ma_crossover": ma_sig,
            "window": window,
        }

        label = "sideways"
        confidence = 0.55

        # Priority: volatile → low liquidity → trends → sideways
        if atr_percentile >= atr_pct_vol:
            label = "volatile"
            confidence = float(min(1.0, 0.55 + (atr_percentile - atr_pct_vol)))
        elif vol_z <= vol_z_low:
            label = "low_liquidity"
            confidence = float(min(1.0, 0.5 + abs(vol_z) / 5.0))
        elif slope > trend_thr and ma_sig > 0:
            label = "bullish_trend"
            confidence = float(min(1.0, 0.5 + min(slope / max(trend_thr, 1e-9), 1.0) * 0.25))
        elif slope < -trend_thr and ma_sig < 0:
            label = "bearish_trend"
            confidence = float(min(1.0, 0.5 + min(abs(slope) / max(trend_thr, 1e-9), 1.0) * 0.25))
        elif atr_percentile <= sideways_atr_max:
            label = "sideways"
            confidence = 0.55
        else:
            label = "sideways"
            confidence = 0.5

        payload = {"label": label, "confidence": confidence, "features": features, "bar_index": bar_index}
        await self._state.write(StateKeys.REGIME, payload, writer=self.agent_id)

        await self._bus.publish(Event(
            event_type=EventType.SIGNAL_GENERATED,
            payload={"regime": label},
            source_agent=self.agent_id,
        ))
        self.log("regime_detected", label=label, confidence=f"{confidence:.3f}")
        return payload
