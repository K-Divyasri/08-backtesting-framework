"""Portfolio: turns SignalEvents into sized OrderEvents, applies FillEvents to
cash/positions, and records the equity curve one bar at a time.

Sizing is naive-but-honest: go long with ~95% of available cash when flat, and
fully exit on EXIT. No leverage, no pyramiding — easy to reason about.
"""

from __future__ import annotations

import pandas as pd

from .event import OrderEvent


class Portfolio:
    def __init__(self, data, events, initial_capital: float = 100_000.0,
                 alloc: float = 0.95):
        self.data = data
        self.events = events
        self.symbols = data.symbols
        self.initial_capital = float(initial_capital)
        self.alloc = float(alloc)

        self.current_positions = {s: 0 for s in self.symbols}
        self.current_holdings = {s: 0.0 for s in self.symbols}
        self.current_holdings["cash"] = self.initial_capital
        self.current_holdings["commission"] = 0.0

        self.all_holdings: list[dict] = []

    # ----- record equity each bar ----------------------------------------- #
    def update_timeindex(self, event) -> None:
        dt = self.data.get_latest_bar_datetime(self.symbols[0])
        snapshot = {
            "datetime": dt,
            "cash": self.current_holdings["cash"],
            "commission": self.current_holdings["commission"],
        }
        total = self.current_holdings["cash"]
        for s in self.symbols:
            market_value = self.current_positions[s] * self.data.get_latest_bar_value(s, "close")
            snapshot[s] = market_value
            total += market_value
        snapshot["total"] = total
        self.all_holdings.append(snapshot)

    # ----- signal -> order ------------------------------------------------ #
    def update_signal(self, event) -> None:
        if event.type == "SIGNAL":
            order = self._generate_order(event)
            if order is not None:
                self.events.put(order)

    def _generate_order(self, signal):
        symbol = signal.symbol
        direction = signal.direction
        cur_qty = self.current_positions[symbol]
        price = self.data.get_latest_bar_value(symbol, "close")
        if price <= 0:
            return None

        cash = self.current_holdings["cash"]
        target_qty = int((cash * self.alloc * signal.strength) / price)

        if direction == "LONG" and cur_qty == 0 and target_qty > 0:
            return OrderEvent(symbol, "MKT", target_qty, "BUY")
        if direction == "SHORT" and cur_qty == 0 and target_qty > 0:
            return OrderEvent(symbol, "MKT", target_qty, "SELL")
        if direction == "EXIT" and cur_qty > 0:
            return OrderEvent(symbol, "MKT", cur_qty, "SELL")
        if direction == "EXIT" and cur_qty < 0:
            return OrderEvent(symbol, "MKT", abs(cur_qty), "BUY")
        return None

    # ----- fill -> cash/positions ----------------------------------------- #
    def update_fill(self, event) -> None:
        if event.type != "FILL":
            return
        sign = 1 if event.direction == "BUY" else -1
        self.current_positions[event.symbol] += sign * event.quantity

        cost = sign * event.fill_price * event.quantity
        self.current_holdings["cash"] -= (cost + event.commission)
        self.current_holdings["commission"] += event.commission

    # ----- results -------------------------------------------------------- #
    def equity_curve(self) -> pd.DataFrame:
        curve = pd.DataFrame(self.all_holdings).set_index("datetime")
        curve["returns"] = curve["total"].pct_change().fillna(0.0)
        curve["equity_curve"] = (1.0 + curve["returns"]).cumprod()
        return curve
