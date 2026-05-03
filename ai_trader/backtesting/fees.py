"""Configurable fee and tax model for trade simulation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FeeModel:
    """Models all costs associated with trade execution.

    Defaults approximate Indian equity intraday costs.
    All rates are expressed as fractions (e.g., 0.0003 = 0.03%).
    """

    brokerage_rate: float = 0.0003        # per-side brokerage
    stt_rate: float = 0.00025             # Securities Transaction Tax (sell side)
    gst_rate: float = 0.18                # GST on brokerage
    exchange_txn_rate: float = 0.0000345  # Exchange transaction charges
    sebi_rate: float = 0.000001           # SEBI turnover fee
    stamp_duty_rate: float = 0.00003      # Stamp duty (buy side)
    slippage_rate: float = 0.0001         # Estimated slippage per trade

    def calculate_buy_cost(self, price: float, quantity: int) -> float:
        """Total cost to buy (price * qty + all buy-side fees)."""
        turnover = price * quantity
        brokerage = turnover * self.brokerage_rate
        gst = brokerage * self.gst_rate
        exchange_fee = turnover * self.exchange_txn_rate
        sebi_fee = turnover * self.sebi_rate
        stamp_duty = turnover * self.stamp_duty_rate
        slippage = turnover * self.slippage_rate

        return turnover + brokerage + gst + exchange_fee + sebi_fee + stamp_duty + slippage

    def calculate_sell_proceeds(self, price: float, quantity: int) -> float:
        """Net proceeds from sell (price * qty - all sell-side fees)."""
        turnover = price * quantity
        brokerage = turnover * self.brokerage_rate
        gst = brokerage * self.gst_rate
        stt = turnover * self.stt_rate
        exchange_fee = turnover * self.exchange_txn_rate
        sebi_fee = turnover * self.sebi_rate
        slippage = turnover * self.slippage_rate

        return turnover - brokerage - gst - stt - exchange_fee - sebi_fee - slippage

    def total_round_trip_cost(self, price: float, quantity: int) -> float:
        """Total fees for a complete buy+sell round trip."""
        buy_cost = self.calculate_buy_cost(price, quantity)
        sell_proceeds = self.calculate_sell_proceeds(price, quantity)
        return buy_cost - sell_proceeds
