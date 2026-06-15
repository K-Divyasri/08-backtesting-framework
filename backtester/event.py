"""The four event types that flow through the engine's queue.

The whole framework is just these events being produced and consumed:

    MarketEvent   -> a new bar is available (produced by the DataFeed)
    SignalEvent   -> the strategy wants exposure (produced by Strategy)
    OrderEvent    -> the portfolio sized that into a concrete order
    FillEvent     -> the execution handler filled it (with costs)
"""

from __future__ import annotations


class Event:
    """Base class; every event carries a ``type`` string the engine switches on."""

    type: str = "EVENT"


class MarketEvent(Event):
    """Emitted once per bar, after the DataFeed advances all symbols."""

    type = "MARKET"


class SignalEvent(Event):
    """A strategy's desired exposure for a symbol.

    direction is one of ``LONG``, ``SHORT`` or ``EXIT``. ``strength`` is a free
    [0, 1] knob a strategy can use for conviction-based sizing (default 1.0).
    """

    type = "SIGNAL"

    def __init__(self, symbol, datetime, direction, strength: float = 1.0):
        self.symbol = symbol
        self.datetime = datetime
        self.direction = direction
        self.strength = strength

    def __repr__(self) -> str:
        return f"SIGNAL({self.symbol}, {self.direction}, {self.datetime})"


class OrderEvent(Event):
    """A concrete order the execution handler will try to fill."""

    type = "ORDER"

    def __init__(self, symbol, order_type, quantity: int, direction):
        self.symbol = symbol
        self.order_type = order_type  # "MKT" only, for now
        self.quantity = int(abs(quantity))
        self.direction = direction  # "BUY" or "SELL"

    def __repr__(self) -> str:
        return (
            f"ORDER({self.symbol}, {self.order_type}, "
            f"qty={self.quantity}, {self.direction})"
        )


class FillEvent(Event):
    """A filled order, including the modeled commission and the actual fill price
    (which already includes slippage)."""

    type = "FILL"

    def __init__(self, timeindex, symbol, exchange, quantity, direction,
                 fill_price, commission):
        self.timeindex = timeindex
        self.symbol = symbol
        self.exchange = exchange
        self.quantity = int(quantity)
        self.direction = direction
        self.fill_price = float(fill_price)
        self.commission = float(commission)

    def __repr__(self) -> str:
        return (
            f"FILL({self.symbol}, {self.direction}, qty={self.quantity}, "
            f"px={self.fill_price:.2f}, comm={self.commission:.2f})"
        )
