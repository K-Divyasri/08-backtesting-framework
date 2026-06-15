"""Data layer: a DataFeed that streams bars into the event queue, plus loaders
for synthetic (offline) data and Yahoo Finance.

Bars are stored as plain tuples for speed:
    (timestamp, open, high, low, close, volume)
Access them through ``get_latest_bars`` / ``get_latest_bar_value`` so strategies
never see future data — they only get what has been "streamed" so far.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .event import MarketEvent

# Column index inside a bar tuple (timestamp is index 0).
_FIELD_IDX = {"open": 1, "high": 2, "low": 3, "close": 4, "volume": 5}


class DataFeed:
    """Replays a dict of ``{symbol: DataFrame}`` one bar at a time.

    Each DataFrame must be indexed by datetime and have lowercase columns
    open/high/low/close/volume. All symbols are aligned onto a common index so
    every ``update_bars`` advances them together.
    """

    def __init__(self, events, data: dict[str, pd.DataFrame]):
        self.events = events
        self.symbols = list(data.keys())
        self.continue_backtest = True

        comb_index = None
        for df in data.values():
            comb_index = df.index if comb_index is None else comb_index.union(df.index)
        comb_index = comb_index.sort_values()

        self._generators = {}
        self.latest_symbol_data: dict[str, list] = {s: [] for s in self.symbols}
        for s in self.symbols:
            df = data[s].reindex(index=comb_index).ffill()
            self._generators[s] = self._bar_generator(df)

    @staticmethod
    def _bar_generator(df: pd.DataFrame):
        for ts, row in df.iterrows():
            yield (ts, row["open"], row["high"], row["low"], row["close"], row["volume"])

    def update_bars(self) -> None:
        """Advance every symbol by one bar and emit a MarketEvent.

        When every generator is exhausted, stop the backtest *without* emitting a
        final MarketEvent — otherwise the last bar would be recorded twice.
        """
        advanced = False
        for s in self.symbols:
            try:
                bar = next(self._generators[s])
            except StopIteration:
                self.continue_backtest = False
            else:
                if bar is not None and not np.isnan(bar[4]):
                    self.latest_symbol_data[s].append(bar)
                    advanced = True
        if advanced:
            self.events.put(MarketEvent())

    def push_bar(self, symbol, bar) -> None:
        """Live mode: inject a single externally-sourced bar, then emit a MarketEvent.

        ``bar`` is a (timestamp, open, high, low, close, volume) tuple.
        """
        self.latest_symbol_data[symbol].append(bar)
        self.events.put(MarketEvent())

    def get_latest_bars(self, symbol, n: int = 1):
        return self.latest_symbol_data[symbol][-n:]

    def get_latest_bar_value(self, symbol, field: str):
        return self.latest_symbol_data[symbol][-1][_FIELD_IDX[field]]

    def get_latest_bar_datetime(self, symbol):
        return self.latest_symbol_data[symbol][-1][0]


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #
def make_synthetic(n: int = 750, seed: int = 42, s0: float = 100.0,
                   mu: float = 0.08, sigma: float = 0.20,
                   start: str = "2021-01-01") -> pd.DataFrame:
    """Geometric-Brownian-motion OHLCV series for fully offline testing."""
    rng = np.random.default_rng(seed)
    dt = 1.0 / 252.0
    shocks = rng.normal((mu - 0.5 * sigma**2) * dt, sigma * np.sqrt(dt), n)
    close = s0 * np.exp(np.cumsum(shocks))
    dates = pd.bdate_range(start=start, periods=n)

    df = pd.DataFrame(index=dates)
    df["close"] = close
    df["open"] = pd.Series(close, index=dates).shift(1).fillna(s0).values
    df["high"] = df[["open", "close"]].max(axis=1) * 1.005
    df["low"] = df[["open", "close"]].min(axis=1) * 0.995
    df["volume"] = rng.integers(1_000_000, 5_000_000, n)
    df.index.name = "datetime"
    return df


def load_yfinance(ticker: str, period: str = "5y",
                  interval: str = "1d") -> pd.DataFrame:
    """Download OHLCV from Yahoo Finance and normalize to lowercase columns."""
    import yfinance as yf

    raw = yf.download(ticker, period=period, interval=interval,
                      auto_adjust=True, progress=False)
    if raw.empty:
        raise ValueError(f"No data returned for {ticker!r}.")

    # yfinance may return a MultiIndex (field, ticker) for single tickers too.
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    raw = raw.rename(columns=str.lower)
    df = raw[["open", "high", "low", "close", "volume"]].dropna()
    df.index.name = "datetime"
    return df
