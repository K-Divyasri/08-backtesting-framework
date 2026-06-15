# 📊 Event-Driven Backtesting Framework

A small, **honest** backtesting engine that the other quant projects in this repo
plug into. The same engine runs in **replay** (historical) and **live** (streaming)
mode behind one `Strategy` interface — write a strategy once, run it both ways.

> ⚠️ **Educational project, not financial advice.**

---

## Why this exists

Per-project backtests tend to quietly cheat: they peek at the future, ignore
trading costs, and report a great Sharpe that came from trying 200 parameter
combos. This framework is built to *not* do that:

- **No lookahead** — signals are generated at a bar's close and only affect the
  *next* bar's position. You can't trade on data you wouldn't have had.
- **Costs are modeled** — commission *and* slippage, both in basis points, on by
  default. A zero-cost backtest is the #1 way to fool yourself.
- **Overfitting is visible** — every run reports the **Probabilistic Sharpe Ratio
  (PSR)**, which discounts a high Sharpe that came from few, skewed, fat-tailed
  returns.

---

## How it works (the 30-second version)

Everything is just four events flowing through a queue, once per bar:

```
MarketEvent  ──>  Strategy   ──> SignalEvent
SignalEvent  ──>  Portfolio  ──> OrderEvent      (position sizing)
OrderEvent   ──>  Execution  ──> FillEvent       (commission + slippage)
FillEvent    ──>  Portfolio                      (update cash + positions)
```

The `Strategy` only *reads* bars and *emits* signals — it never touches cash or
fills. That separation is what lets the identical strategy run in replay and live
mode.

---

## Project structure

```
08-backtesting-framework/
├── app.py                # Streamlit UI: pick strategy/params, run, tearsheet
├── requirements.txt
├── smoke_test.py         # full pipeline on synthetic data — no internet needed
└── backtester/           # the engine, no UI
    ├── event.py          # Market / Signal / Order / Fill events
    ├── data.py           # DataFeed (replay + live push) + synthetic/yfinance loaders
    ├── strategy.py       # Strategy base class + SMA-crossover example
    ├── portfolio.py      # sizing, cash/positions, equity curve
    ├── execution.py      # simulated fills with commission + slippage
    ├── engine.py         # the event loop (Backtest) + vectorized fast-path
    └── metrics.py        # Sharpe, Sortino, drawdown, CAGR, PSR, tearsheet
```

This mirrors the repo's house style (logic in a package, UI in `app.py`).

---

## Run it locally

```bash
cd 08-backtesting-framework

# (recommended) isolated environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux

pip install -r requirements.txt

# 1) sanity-check the engine offline (no internet)
python smoke_test.py

# 2) launch the app
streamlit run app.py
```

Leave the data source on **Demo (synthetic)** for a guaranteed first run, then
switch to **Yahoo Finance** and try `SPY`, `AAPL`, or `BTC-USD`.

---

## Use it from code

```python
from backtester import Backtest, tearsheet
from backtester.data import make_synthetic, load_yfinance

prices = make_synthetic(n=750)          # or load_yfinance("SPY", period="5y")

bt = Backtest(
    {"SPY": prices},
    strategy_params={"short_window": 20, "long_window": 50},
    initial_capital=100_000,
    commission_bps=1.0,
    slippage_bps=2.0,
)
curve = bt.run()
print(tearsheet(curve))
```

### Write your own strategy

```python
from backtester.strategy import Strategy
from backtester.event import SignalEvent

class MyStrategy(Strategy):
    def calculate_signals(self, event):
        if event.type != "MARKET":
            return
        for s in self.symbols:
            bars = self.data.get_latest_bars(s, n=20)
            if len(bars) < 20:
                continue
            # ... your logic ...
            dt = self.data.get_latest_bar_datetime(s)
            self.events.put(SignalEvent(s, dt, "LONG"))   # or "EXIT" / "SHORT"
```

Pass it via `Backtest({...}, strategy_cls=MyStrategy, strategy_params={...})`.

### Live / paper-trade mode

```python
bt = Backtest({"SPY": prices}, strategy_cls=MyStrategy)
for ts, row in stream_of_bars():            # your live feed
    bt.step("SPY", (ts, o, h, l, c, v))
    latest = bt.portfolio.equity_curve()    # inspect any time
```

---

## What the metrics mean

| Metric | Reading |
|---|---|
| **Sharpe / Sortino** | Return per unit of total / downside risk (annualized). |
| **Max Drawdown** | Worst peak-to-trough loss. Watch this vs buy-and-hold. |
| **Hit Rate** | Fraction of active periods that were positive. |
| **PSR (vs 0)** | Probability the *true* Sharpe > 0 after adjusting for sample size, skew, and fat tails. **High Sharpe + low PSR ⇒ probably overfit.** |

---

## Honest limitations

- Single position per symbol, naive % -of-cash sizing — no leverage or pyramiding.
- Market orders fill at the bar close ± slippage; no partial fills or queue position
  (that realism lives in project **#03**, the LOB simulator).
- PSR guards a single track record; for full multiple-testing correction add the
  **Deflated Sharpe / PBO** across your parameter sweep (a natural v2).
- Past performance doesn't predict the future.

---

## How other projects reuse this

Projects **#01 (pairs trading)**, **#06 (regime allocator)**, **#11 (algo zoo)**,
**#12 (options-flow)**, and **#13 (macro-nowcast)** all import this engine for their
backtests instead of re-implementing one — so the *same* honest cost/lookahead
discipline applies everywhere. See `../project-specs/`.

## License
MIT — do whatever you like, no warranty.
