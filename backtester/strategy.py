"""Strategy interface + a worked example (SMA crossover).

A strategy only ever *reads* bars through the DataFeed and *emits* SignalEvents.
It never touches cash, sizing, or fills — that separation is what lets the same
strategy run in both replay and live mode.
"""

from __future__ import annotations

import numpy as np

from .event import SignalEvent


class Strategy:
    """Base class. Subclass and implement ``calculate_signals``."""

    def __init__(self, data, events):
        self.data = data
        self.events = events
        self.symbols = data.symbols

    def calculate_signals(self, event) -> None:  # pragma: no cover - interface
        raise NotImplementedError


class MovingAverageCrossStrategy(Strategy):
    """Long when the short SMA is above the long SMA, flat otherwise.

    A deliberately simple, well-understood baseline so the framework's mechanics
    (sizing, costs, equity curve) can be verified against intuition.
    """

    def __init__(self, data, events, short_window: int = 20, long_window: int = 50):
        super().__init__(data, events)
        if short_window >= long_window:
            raise ValueError("short_window must be < long_window")
        self.short_window = short_window
        self.long_window = long_window
        self._invested = {s: False for s in self.symbols}

    def calculate_signals(self, event) -> None:
        if event.type != "MARKET":
            return
        for s in self.symbols:
            bars = self.data.get_latest_bars(s, n=self.long_window)
            if len(bars) < self.long_window:
                continue

            closes = np.array([b[4] for b in bars], dtype=float)
            short_sma = closes[-self.short_window:].mean()
            long_sma = closes.mean()
            dt = self.data.get_latest_bar_datetime(s)

            if short_sma > long_sma and not self._invested[s]:
                self.events.put(SignalEvent(s, dt, "LONG"))
                self._invested[s] = True
            elif short_sma < long_sma and self._invested[s]:
                self.events.put(SignalEvent(s, dt, "EXIT"))
                self._invested[s] = False
