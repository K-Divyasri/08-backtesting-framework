"""The event loop (``Backtest``) plus a vectorized fast-path for quick sweeps.

Event-driven loop, per bar:
    1. DataFeed.update_bars()        -> MarketEvent
    2. drain the queue:
         MARKET -> Strategy.calculate_signals + Portfolio.update_timeindex
         SIGNAL -> Portfolio.update_signal      (-> OrderEvent)
         ORDER  -> ExecutionHandler.execute_order (-> FillEvent)
         FILL   -> Portfolio.update_fill

Because signals are generated at a bar's close and only affect the *position*
recorded on the next bar, there's no lookahead: you can't trade on information
you wouldn't have had.
"""

from __future__ import annotations

import queue

import numpy as np
import pandas as pd

from .data import DataFeed
from .execution import SimulatedExecutionHandler
from .portfolio import Portfolio
from .strategy import MovingAverageCrossStrategy


class Backtest:
    def __init__(self, data: dict, strategy_cls=MovingAverageCrossStrategy,
                 strategy_params: dict | None = None,
                 initial_capital: float = 100_000.0,
                 commission_bps: float = 1.0, slippage_bps: float = 2.0):
        self.events: queue.Queue = queue.Queue()
        self.feed = DataFeed(self.events, data)
        self.strategy = strategy_cls(self.feed, self.events, **(strategy_params or {}))
        self.portfolio = Portfolio(self.feed, self.events, initial_capital)
        self.execution = SimulatedExecutionHandler(
            self.events, self.feed, commission_bps, slippage_bps
        )
        self.signals = 0
        self.fills = 0

    def _dispatch(self, event) -> None:
        if event.type == "MARKET":
            self.strategy.calculate_signals(event)
            self.portfolio.update_timeindex(event)
        elif event.type == "SIGNAL":
            self.signals += 1
            self.portfolio.update_signal(event)
        elif event.type == "ORDER":
            self.execution.execute_order(event)
        elif event.type == "FILL":
            self.fills += 1
            self.portfolio.update_fill(event)

    def run(self) -> pd.DataFrame:
        """Run to completion and return the equity-curve DataFrame."""
        while self.feed.continue_backtest:
            self.feed.update_bars()
            while True:
                try:
                    event = self.events.get(False)
                except queue.Empty:
                    break
                if event is not None:
                    self._dispatch(event)
        return self.portfolio.equity_curve()

    def step(self, symbol, bar) -> None:
        """Live mode: push one externally-sourced bar and process it.

        ``bar`` is a (timestamp, open, high, low, close, volume) tuple. Call
        ``portfolio.equity_curve()`` whenever you want the latest state.
        """
        self.feed.push_bar(symbol, bar)
        while True:
            try:
                event = self.events.get(False)
            except queue.Empty:
                break
            if event is not None:
                self._dispatch(event)


# --------------------------------------------------------------------------- #
# Vectorized fast-path — same SMA logic, no event loop, for parameter sweeps.
# --------------------------------------------------------------------------- #
def vectorized_sma_backtest(prices: pd.DataFrame, short_window: int = 20,
                            long_window: int = 50,
                            cost_bps: float = 3.0) -> pd.DataFrame:
    """Approximate the SMA-crossover result without the event loop.

    Used to sanity-check the event-driven engine and to sweep parameters quickly.
    ``cost_bps`` is charged on each change in position (round-trip ~= 2x).
    """
    df = prices[["close"]].copy()
    df["short"] = df["close"].rolling(short_window).mean()
    df["long"] = df["close"].rolling(long_window).mean()
    df["signal"] = (df["short"] > df["long"]).astype(int)
    df["position"] = df["signal"].shift(1).fillna(0)  # trade next bar -> no lookahead
    df["ret"] = df["close"].pct_change().fillna(0.0)
    df["trades"] = df["position"].diff().abs().fillna(0.0)
    df["strat_ret"] = df["position"] * df["ret"] - df["trades"] * cost_bps / 10_000.0
    df["equity_curve"] = (1.0 + df["strat_ret"]).cumprod()
    df["returns"] = df["strat_ret"]
    return df
