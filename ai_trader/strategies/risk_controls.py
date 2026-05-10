"""Risk management controls enforced during backtesting and live trading."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from ai_trader.strategies.config_loader import RiskParams, OvertradingControls
from ai_trader.logs import get_logger

logger = get_logger(__name__)


@dataclass
class DailyState:
    """Tracks per-day trading state for risk enforcement."""

    current_date: date | None = None
    trades_today: int = 0
    daily_pnl: float = 0.0
    consecutive_losses: int = 0
    bars_since_last_trade: int = 0


class RiskController:
    """Enforces risk limits and overtrading controls.

    Acts as a gate between strategy signals and execution.
    Can veto any trade that violates configured limits.
    """

    def __init__(self, risk_params: RiskParams, overtrading: OvertradingControls):
        self.risk = risk_params
        self.overtrading = overtrading
        self._state = DailyState()

    @property
    def state(self) -> DailyState:
        return self._state

    def reset(self) -> None:
        """Reset all state (for new backtest runs)."""
        self._state = DailyState()

    def on_new_bar(self, bar_date: date) -> None:
        """Called at the start of each bar to update state."""
        if self._state.current_date != bar_date:
            self._state.current_date = bar_date
            self._state.trades_today = 0
            self._state.daily_pnl = 0.0
        self._state.bars_since_last_trade += 1

    def can_trade(self, capital: float) -> tuple[bool, str]:
        """Check if a new trade is allowed under current risk constraints.

        Returns:
            (allowed, reason) — reason is empty if allowed.
        """
        if self._state.consecutive_losses >= self.risk.max_consecutive_losses:
            return False, f"consecutive_losses_limit ({self._state.consecutive_losses})"

        if self._state.trades_today >= self.overtrading.max_trades_per_day:
            return False, f"daily_trade_limit ({self._state.trades_today})"

        if self._state.bars_since_last_trade < self.overtrading.cooldown_bars:
            return False, f"cooldown_active ({self._state.bars_since_last_trade}/{self.overtrading.cooldown_bars})"

        if capital > 0 and abs(self._state.daily_pnl) / capital >= self.risk.daily_loss_limit:
            if self._state.daily_pnl < 0:
                return False, f"daily_loss_limit_hit ({self._state.daily_pnl:.2f})"

        return True, ""

    def calculate_position_size(self, capital: float, price: float) -> int:
        """Determine position size respecting max capital per trade."""
        risk_capital = capital * self.risk.max_capital_per_trade
        quantity = int(risk_capital / price)
        return max(quantity, 0)

    def calculate_stop_loss(
        self,
        entry_price: float,
        atr: float | None = None,
        regime_label: str | None = None,
    ) -> float:
        """Calculate stop loss price using ATR (regime-scaled) or fixed percentage."""
        mult = self.risk.atr_stop_multiplier
        if regime_label == "volatile":
            mult *= 0.88
        elif regime_label in ("bullish_trend", "bearish_trend"):
            mult *= 1.06

        if atr is not None and atr > 0:
            return entry_price - (atr * mult)
        return entry_price * (1.0 - self.risk.stop_loss_pct)

    def on_trade_completed(self, pnl: float) -> None:
        """Update state after a trade completes."""
        self._state.trades_today += 1
        self._state.daily_pnl += pnl
        self._state.bars_since_last_trade = 0

        if pnl < 0:
            self._state.consecutive_losses += 1
        else:
            self._state.consecutive_losses = 0

    def check_confidence(self, confidence: float) -> bool:
        """Check if signal confidence meets minimum threshold."""
        return confidence >= self.overtrading.min_confidence
