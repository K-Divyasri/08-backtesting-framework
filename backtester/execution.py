"""Simulated execution: market orders fill at the latest close, adjusted for
slippage, with a commission charged in basis points of notional.

Both costs are in bps (1 bp = 0.01%). Defaults are deliberately non-zero — a
backtest with zero costs is the single most common way to fool yourself.
"""

from __future__ import annotations

from .event import FillEvent


class SimulatedExecutionHandler:
    def __init__(self, events, data, commission_bps: float = 1.0,
                 slippage_bps: float = 2.0, exchange: str = "SIM"):
        self.events = events
        self.data = data
        self.commission_bps = float(commission_bps)
        self.slippage_bps = float(slippage_bps)
        self.exchange = exchange

    def execute_order(self, event) -> None:
        if event.type != "ORDER":
            return

        ref_price = self.data.get_latest_bar_value(event.symbol, "close")
        slip = ref_price * self.slippage_bps / 10_000.0
        # Buyers pay up, sellers receive less — slippage always hurts.
        fill_price = ref_price + slip if event.direction == "BUY" else ref_price - slip

        notional = fill_price * event.quantity
        commission = notional * self.commission_bps / 10_000.0

        fill = FillEvent(
            timeindex=self.data.get_latest_bar_datetime(event.symbol),
            symbol=event.symbol,
            exchange=self.exchange,
            quantity=event.quantity,
            direction=event.direction,
            fill_price=fill_price,
            commission=commission,
        )
        self.events.put(fill)
