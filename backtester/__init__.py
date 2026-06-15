"""
Event-driven backtesting framework.

A small, honest backtester: Market -> Signal -> Order -> Fill events flow through
a queue, fills carry commission + slippage, and the same engine runs in replay
(historical) or live (streaming) mode behind one Strategy interface.

Public API
----------
    from backtester import Backtest, MovingAverageCrossStrategy
    from backtester.data import make_synthetic, load_yfinance
    from backtester.metrics import tearsheet
    from backtester.engine import vectorized_sma_backtest
"""

from .engine import Backtest, vectorized_sma_backtest
from .strategy import Strategy, MovingAverageCrossStrategy
from .metrics import tearsheet

__all__ = [
    "Backtest",
    "vectorized_sma_backtest",
    "Strategy",
    "MovingAverageCrossStrategy",
    "tearsheet",
]

__version__ = "0.1.0"
