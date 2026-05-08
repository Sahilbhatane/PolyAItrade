"""ReflectionAgent — post-trade analysis and bounded strategy adaptation.

Responsibilities:
- Log every trade with full context (entry reason, exit reason, outcome)
- After each closed trade, evaluate:
  - Signal correctness (did the signals predict the actual move?)
  - Timing error (was entry/exit too early or late?)
  - Risk issues (was stop loss adequate? position sizing correct?)
- Propose bounded adjustments to strategy weights and confidence thresholds
- Generate trade feedback reports and performance trend analysis

Constraints (from agent rules):
- CANNOT directly modify strategy — only proposes bounded updates
- All adjustments are clamped within configurable bounds
- Every proposed change is logged with reasoning
- Does NOT override logging or bypass validation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np

from ai_trader.agents.base import BaseAgent
from ai_trader.agents.event_bus import Event, EventBus, EventType
from ai_trader.agents.state import StateKeys, StateManager
from ai_trader.logs import get_logger

logger = get_logger(__name__)


@dataclass
class TradeRecord:
    """Full audit record of a single trade."""

    trade_id: str
    symbol: str
    side: str
    entry_price: float
    exit_price: float | None = None
    quantity: int = 0
    stop_loss: float | None = None
    entry_bar: int = 0
    exit_bar: int | None = None
    entry_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    exit_time: datetime | None = None
    entry_reason: str = ""
    exit_reason: str = ""
    entry_signals: dict[str, float] = field(default_factory=dict)
    entry_confidence: float = 0.0
    pnl: float = 0.0
    pnl_pct: float = 0.0
    duration_bars: int = 0
    slippage_cost: float = 0.0
    is_closed: bool = False


@dataclass
class TradeEvaluation:
    """Post-trade evaluation result."""

    trade_id: str
    signal_correctness: float  # -1 (completely wrong) to +1 (perfectly right)
    timing_score: float        # -1 (very bad timing) to +1 (perfect timing)
    risk_assessment: str       # "adequate", "too_tight", "too_loose"
    stop_loss_hit: bool
    slippage_impact: float     # percentage of PnL lost to slippage
    recommendations: list[str] = field(default_factory=list)


@dataclass
class WeightAdjustment:
    """A single proposed weight change — bounded and logged."""

    indicator: str
    current_weight: float
    proposed_weight: float
    reason: str
    magnitude: float  # how much it changed (absolute)


@dataclass
class ReflectionReport:
    """Complete reflection output for a batch of trades."""

    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_pnl: float
    avg_signal_correctness: float
    avg_timing_score: float
    weight_adjustments: list[WeightAdjustment]
    confidence_adjustment: float
    performance_trend: str  # "improving", "stable", "degrading"
    recommendations: list[str]


class ReflectionAgent(BaseAgent):
    """Analyzes completed trades and proposes bounded strategy adjustments.

    Subscribes to TRADE_CLOSED events. After each trade:
    1. Logs the full trade context
    2. Evaluates signal correctness, timing, and risk
    3. Accumulates statistics over a window
    4. Proposes bounded weight/threshold adjustments

    This agent CANNOT directly modify the strategy. It only writes
    proposed adjustments to state — the orchestrator decides whether to apply.
    """

    def __init__(
        self,
        event_bus: EventBus,
        state: StateManager,
        config: dict[str, Any] | None = None,
        agent_id: str | None = None,
    ):
        super().__init__(agent_id=agent_id or "reflection_agent")
        self._bus = event_bus
        self._state = state
        self._config = config or {}

        # Adjustment bounds (prevents uncontrolled changes)
        self._max_weight_delta = self._config.get("max_weight_delta", 0.05)
        self._max_confidence_delta = self._config.get("max_confidence_delta", 0.05)
        self._min_weight = self._config.get("min_weight", 0.05)
        self._max_weight = self._config.get("max_weight", 0.50)
        self._min_confidence = self._config.get("min_confidence", 0.3)
        self._max_confidence = self._config.get("max_confidence", 0.8)

        # Lookback window for trend analysis
        self._lookback_window = self._config.get("lookback_window", 20)
        self._min_trades_for_adjustment = self._config.get("min_trades_for_adjustment", 5)

        # Internal trade log
        self._trade_log: list[TradeRecord] = []
        self._evaluations: list[TradeEvaluation] = []

        # Current strategy weights (read from state when available)
        self._current_weights: dict[str, float] = {
            "rsi": 0.25,
            "macd": 0.25,
            "ma_crossover": 0.25,
            "vwap": 0.25,
        }
        self._current_confidence_threshold: float = 0.5

    @property
    def trade_log(self) -> list[TradeRecord]:
        return self._trade_log

    @property
    def evaluations(self) -> list[TradeEvaluation]:
        return self._evaluations

    async def run(self, context: dict[str, Any] | None = None) -> dict[str, Any]:
        """Analyze the most recent closed trade and update internal state.

        Expected context:
            trade: dict with trade details (entry/exit/pnl)
            market_data: DataFrame at time of trade (for evaluation)
            signals_at_entry: dict of signal values when trade was opened

        Writes: StateKeys.TRADE_LOG, StateKeys.REFLECTION_REPORT, StateKeys.WEIGHT_ADJUSTMENTS
        """
        if not context:
            raise ValueError("ReflectionAgent requires trade context")

        trade_data = context.get("trade", {})
        signals_at_entry = context.get("signals_at_entry", {})
        actual_move = context.get("actual_price_move", 0.0)

        # Step 1: Log the trade
        record = self._log_trade(trade_data, signals_at_entry)

        # Step 2: Evaluate the trade
        evaluation = self._evaluate_trade(record, actual_move)
        self._evaluations.append(evaluation)

        # Step 3: Generate report and adjustments (if enough data)
        report = None
        if len(self._trade_log) >= self._min_trades_for_adjustment:
            report = self._generate_report()
            await self._state.write(StateKeys.REFLECTION_REPORT, report, writer=self.agent_id)

            if report.weight_adjustments:
                adjustments = {
                    "weights": {a.indicator: a.proposed_weight for a in report.weight_adjustments},
                    "confidence_threshold": self._current_confidence_threshold + report.confidence_adjustment,
                    "reasoning": report.recommendations,
                }
                await self._state.write(StateKeys.WEIGHT_ADJUSTMENTS, adjustments, writer=self.agent_id)

                await self._bus.publish(Event(
                    event_type=EventType.ADJUSTMENT_PROPOSED,
                    payload=adjustments,
                    source_agent=self.agent_id,
                ))

        # Write updated trade log to state
        log_summary = [{"trade_id": t.trade_id, "pnl": t.pnl, "side": t.side} for t in self._trade_log[-50:]]
        await self._state.write(StateKeys.TRADE_LOG, log_summary, writer=self.agent_id)

        await self._bus.publish(Event(
            event_type=EventType.REFLECTION_COMPLETE,
            payload={
                "trade_id": record.trade_id,
                "signal_correctness": evaluation.signal_correctness,
                "timing_score": evaluation.timing_score,
            },
            source_agent=self.agent_id,
        ))

        self.log(
            "reflection_complete",
            trade_id=record.trade_id,
            pnl=f"{record.pnl:.2f}",
            signal_correctness=f"{evaluation.signal_correctness:.2f}",
        )

        return {
            "trade_id": record.trade_id,
            "evaluation": {
                "signal_correctness": evaluation.signal_correctness,
                "timing_score": evaluation.timing_score,
                "risk_assessment": evaluation.risk_assessment,
            },
            "has_adjustments": report is not None and len(report.weight_adjustments) > 0,
        }

    def _log_trade(self, trade_data: dict[str, Any], signals: dict[str, float]) -> TradeRecord:
        """Create and store a full trade record."""
        record = TradeRecord(
            trade_id=trade_data.get("trade_id", f"t_{len(self._trade_log)}"),
            symbol=trade_data.get("symbol", "UNKNOWN"),
            side=trade_data.get("side", "BUY"),
            entry_price=trade_data.get("entry_price", 0.0),
            exit_price=trade_data.get("exit_price"),
            quantity=trade_data.get("quantity", 0),
            stop_loss=trade_data.get("stop_loss"),
            entry_bar=trade_data.get("entry_bar", 0),
            exit_bar=trade_data.get("exit_bar"),
            entry_reason=trade_data.get("entry_reason", ""),
            exit_reason=trade_data.get("exit_reason", ""),
            entry_signals=signals,
            entry_confidence=trade_data.get("confidence", 0.0),
            pnl=trade_data.get("pnl", 0.0),
            pnl_pct=trade_data.get("pnl_pct", 0.0),
            duration_bars=trade_data.get("duration_bars", 0),
            slippage_cost=trade_data.get("slippage_cost", 0.0),
            is_closed=True,
        )
        self._trade_log.append(record)
        return record

    def _evaluate_trade(self, record: TradeRecord, actual_move: float) -> TradeEvaluation:
        """Evaluate a closed trade for signal correctness, timing, and risk."""
        # Signal correctness: did the composite signal direction match the actual move?
        composite_signal = sum(record.entry_signals.values()) / max(len(record.entry_signals), 1)
        if actual_move != 0:
            # +1 if signal and move same direction, -1 if opposite
            signal_correctness = np.clip(
                composite_signal * np.sign(actual_move) * 2.0, -1.0, 1.0
            )
        else:
            signal_correctness = 0.0

        # Timing score: based on PnL relative to potential
        if record.entry_price > 0 and record.exit_price is not None:
            achieved_move = (record.exit_price - record.entry_price) / record.entry_price
            if abs(actual_move) > 0:
                timing_score = np.clip(achieved_move / actual_move, -1.0, 1.0) if actual_move != 0 else 0.0
            else:
                timing_score = 0.0
        else:
            timing_score = 0.0

        # Risk assessment
        if record.stop_loss is not None and record.exit_reason == "stop_loss":
            if record.pnl < 0 and abs(record.pnl_pct) < 0.02:
                risk_assessment = "adequate"
            elif abs(record.pnl_pct) > 0.05:
                risk_assessment = "too_loose"
            else:
                risk_assessment = "adequate"
        elif record.pnl < 0 and record.stop_loss is None:
            risk_assessment = "missing_stop_loss"
        else:
            risk_assessment = "adequate"

        # Slippage impact
        slippage_impact = 0.0
        if record.pnl != 0 and record.slippage_cost > 0:
            slippage_impact = record.slippage_cost / abs(record.pnl)

        recommendations = []
        if signal_correctness < -0.5:
            recommendations.append(f"Signal was wrong — consider reducing weight of dominant indicator")
        if timing_score < -0.3:
            recommendations.append("Entry timing was poor — consider tighter entry conditions")
        if risk_assessment == "too_loose":
            recommendations.append("Stop loss was too far — consider tighter ATR multiplier")
        if slippage_impact > 0.2:
            recommendations.append("High slippage impact — consider limit orders or lower frequency")

        return TradeEvaluation(
            trade_id=record.trade_id,
            signal_correctness=float(signal_correctness),
            timing_score=float(timing_score),
            risk_assessment=risk_assessment,
            stop_loss_hit=record.exit_reason == "stop_loss",
            slippage_impact=slippage_impact,
            recommendations=recommendations,
        )

    def _generate_report(self) -> ReflectionReport:
        """Generate a full reflection report from recent trades."""
        recent = self._trade_log[-self._lookback_window:]
        recent_evals = self._evaluations[-self._lookback_window:]

        winning = [t for t in recent if t.pnl > 0]
        losing = [t for t in recent if t.pnl <= 0]

        win_rate = len(winning) / max(len(recent), 1)
        avg_pnl = np.mean([t.pnl for t in recent]) if recent else 0.0

        avg_signal_corr = np.mean([e.signal_correctness for e in recent_evals]) if recent_evals else 0.0
        avg_timing = np.mean([e.timing_score for e in recent_evals]) if recent_evals else 0.0

        # Determine trend (compare first half vs second half of window)
        trend = self._compute_trend(recent)

        # Compute bounded weight adjustments
        adjustments = self._compute_weight_adjustments(recent, recent_evals)

        # Compute confidence threshold adjustment
        confidence_adj = self._compute_confidence_adjustment(recent_evals)

        recommendations = self._compile_recommendations(recent_evals, trend)

        return ReflectionReport(
            total_trades=len(recent),
            winning_trades=len(winning),
            losing_trades=len(losing),
            win_rate=win_rate,
            avg_pnl=float(avg_pnl),
            avg_signal_correctness=float(avg_signal_corr),
            avg_timing_score=float(avg_timing),
            weight_adjustments=adjustments,
            confidence_adjustment=confidence_adj,
            performance_trend=trend,
            recommendations=recommendations,
        )

    def _compute_weight_adjustments(
        self,
        trades: list[TradeRecord],
        evals: list[TradeEvaluation],
    ) -> list[WeightAdjustment]:
        """Compute bounded weight changes based on per-indicator performance.

        For each indicator, measure correlation between its signal at entry
        and actual trade outcome. Increase weight if positive, decrease if negative.
        Changes are ALWAYS bounded by max_weight_delta.
        """
        if len(trades) < self._min_trades_for_adjustment:
            return []

        indicator_scores: dict[str, list[float]] = {k: [] for k in self._current_weights}

        for trade in trades:
            outcome = 1.0 if trade.pnl > 0 else -1.0
            for indicator, signal_val in trade.entry_signals.items():
                if indicator in indicator_scores:
                    # Score: did the indicator's direction match the trade outcome?
                    score = signal_val * outcome
                    indicator_scores[indicator].append(score)

        adjustments = []
        for indicator, scores in indicator_scores.items():
            if not scores:
                continue

            avg_score = np.mean(scores)
            current_w = self._current_weights.get(indicator, 0.25)

            # Scale adjustment proportionally, but clamp to bound
            raw_delta = avg_score * 0.02  # conservative scaling
            clamped_delta = np.clip(raw_delta, -self._max_weight_delta, self._max_weight_delta)

            new_weight = np.clip(current_w + clamped_delta, self._min_weight, self._max_weight)

            if abs(new_weight - current_w) > 0.001:
                reason = f"avg_score={avg_score:.3f} over {len(scores)} trades"
                adjustments.append(WeightAdjustment(
                    indicator=indicator,
                    current_weight=current_w,
                    proposed_weight=float(new_weight),
                    reason=reason,
                    magnitude=abs(float(new_weight) - current_w),
                ))
                self._current_weights[indicator] = float(new_weight)

        # Re-normalize weights to sum to 1.0, then re-clamp
        total = sum(self._current_weights.values())
        if total > 0:
            for k in self._current_weights:
                self._current_weights[k] /= total
            # Clamp again after normalization to respect bounds
            for k in self._current_weights:
                self._current_weights[k] = float(np.clip(
                    self._current_weights[k], self._min_weight, self._max_weight
                ))

        return adjustments

    def _compute_confidence_adjustment(self, evals: list[TradeEvaluation]) -> float:
        """Adjust confidence threshold based on recent signal correctness.

        If signals are mostly correct → lower threshold (trade more).
        If signals are mostly wrong → raise threshold (trade less).
        """
        if len(evals) < self._min_trades_for_adjustment:
            return 0.0

        avg_correctness = np.mean([e.signal_correctness for e in evals])

        # If signals are bad, increase threshold (be more selective)
        # If signals are good, decrease threshold (be less selective)
        raw_adj = -avg_correctness * 0.02  # inverted: good signals → lower threshold
        clamped = np.clip(raw_adj, -self._max_confidence_delta, self._max_confidence_delta)

        new_threshold = np.clip(
            self._current_confidence_threshold + clamped,
            self._min_confidence,
            self._max_confidence,
        )
        adj = float(new_threshold - self._current_confidence_threshold)
        self._current_confidence_threshold = float(new_threshold)

        return adj

    @staticmethod
    def _compute_trend(trades: list[TradeRecord]) -> str:
        """Compare first half vs second half to detect performance trend."""
        if len(trades) < 4:
            return "stable"

        mid = len(trades) // 2
        first_half_wr = sum(1 for t in trades[:mid] if t.pnl > 0) / max(mid, 1)
        second_half_wr = sum(1 for t in trades[mid:] if t.pnl > 0) / max(len(trades) - mid, 1)

        delta = second_half_wr - first_half_wr
        if delta > 0.1:
            return "improving"
        elif delta < -0.1:
            return "degrading"
        return "stable"

    @staticmethod
    def _compile_recommendations(evals: list[TradeEvaluation], trend: str) -> list[str]:
        """Aggregate recommendations from individual evaluations."""
        recs = []

        if trend == "degrading":
            recs.append("Performance is degrading — consider reducing trade frequency")

        # Count common issues
        timing_issues = sum(1 for e in evals if e.timing_score < -0.3)
        signal_issues = sum(1 for e in evals if e.signal_correctness < -0.3)
        risk_issues = sum(1 for e in evals if e.risk_assessment != "adequate")

        if timing_issues > len(evals) * 0.4:
            recs.append(f"Timing issues in {timing_issues}/{len(evals)} trades — review entry/exit logic")
        if signal_issues > len(evals) * 0.4:
            recs.append(f"Signal accuracy poor in {signal_issues}/{len(evals)} trades — review indicator weights")
        if risk_issues > len(evals) * 0.3:
            recs.append(f"Risk management issues in {risk_issues}/{len(evals)} trades — review stop loss params")

        return recs

    def reset(self) -> None:
        """Reset all accumulated state."""
        self._trade_log.clear()
        self._evaluations.clear()
        self._current_weights = {"rsi": 0.25, "macd": 0.25, "ma_crossover": 0.25, "vwap": 0.25}
        self._current_confidence_threshold = 0.5

    def get_performance_summary(self) -> dict[str, Any]:
        """Get current performance stats for external monitoring."""
        if not self._trade_log:
            return {"total_trades": 0}

        recent = self._trade_log[-self._lookback_window:]
        return {
            "total_trades": len(self._trade_log),
            "recent_win_rate": sum(1 for t in recent if t.pnl > 0) / max(len(recent), 1),
            "recent_avg_pnl": float(np.mean([t.pnl for t in recent])),
            "current_weights": self._current_weights.copy(),
            "current_confidence_threshold": self._current_confidence_threshold,
            "trend": self._compute_trend(recent),
        }
