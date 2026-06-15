"""Streamlit UI for the event-driven backtesting framework.

    streamlit run app.py

Pick a data source and SMA parameters, run the backtest, and read the tearsheet,
equity curve vs buy-and-hold, drawdown, and the event-driven vs vectorized check.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from backtester import Backtest, vectorized_sma_backtest
from backtester.data import load_yfinance, make_synthetic
from backtester.metrics import max_drawdown, summary_stats

st.set_page_config(page_title="Backtesting Framework", page_icon="📊", layout="wide")
st.title("📊 Event-Driven Backtesting Framework")
st.caption("Educational, not financial advice. One engine, replay + live, honest costs.")

# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("Data")
    source = st.radio("Source", ["Demo (synthetic)", "Yahoo Finance"], index=0)
    if source == "Yahoo Finance":
        ticker = st.text_input("Ticker", "SPY").strip().upper()
        period = st.selectbox("Period", ["1y", "2y", "5y", "10y", "max"], index=2)
    else:
        seed = st.number_input("Random seed", 0, 9999, 7)
        n_days = st.slider("Days", 250, 2000, 750, step=50)

    st.header("Strategy — SMA crossover")
    short_w = st.slider("Short window", 5, 100, 20)
    long_w = st.slider("Long window", 20, 300, 50)

    st.header("Account & costs")
    capital = st.number_input("Initial capital ($)", 1_000, 10_000_000, 100_000, step=1_000)
    commission_bps = st.slider("Commission (bps)", 0.0, 20.0, 1.0, step=0.5)
    slippage_bps = st.slider("Slippage (bps)", 0.0, 50.0, 2.0, step=0.5)

    run = st.button("▶ Run backtest", type="primary", use_container_width=True)


@st.cache_data(show_spinner=False)
def _load(source, **kw) -> pd.DataFrame:
    if source == "Yahoo Finance":
        return load_yfinance(kw["ticker"], period=kw["period"])
    return make_synthetic(n=kw["n_days"], seed=kw["seed"])


if short_w >= long_w:
    st.error("Short window must be smaller than the long window.")
    st.stop()

if not run:
    st.info("Set parameters in the sidebar and click **Run backtest**.")
    st.stop()

# --------------------------------------------------------------------------- #
# Load data
# --------------------------------------------------------------------------- #
try:
    if source == "Yahoo Finance":
        prices = _load(source, ticker=ticker, period=period)
        label = ticker
    else:
        prices = _load(source, n_days=n_days, seed=seed)
        label = "SYNTH"
except Exception as exc:  # noqa: BLE001 - surface any data error to the user
    st.error(f"Could not load data: {exc}")
    st.stop()

# --------------------------------------------------------------------------- #
# Run
# --------------------------------------------------------------------------- #
params = {"short_window": short_w, "long_window": long_w}
bt = Backtest(
    {label: prices},
    strategy_params=params,
    initial_capital=capital,
    commission_bps=commission_bps,
    slippage_bps=slippage_bps,
)
curve = bt.run()

# Buy & hold benchmark on the same series.
bh = prices["close"] / prices["close"].iloc[0]
strat_eq = curve["equity_curve"]

stats = summary_stats(curve)
bh_returns = prices["close"].pct_change().fillna(0.0)
bh_sharpe = summary_stats(
    pd.DataFrame({"returns": bh_returns, "equity_curve": bh})
)["Sharpe"]

# --------------------------------------------------------------------------- #
# Headline metrics
# --------------------------------------------------------------------------- #
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Return", f"{stats['Total Return']:.1%}")
c2.metric("Sharpe", f"{stats['Sharpe']:.2f}", delta=f"{stats['Sharpe'] - bh_sharpe:+.2f} vs B&H")
c3.metric("Max Drawdown", f"{stats['Max Drawdown']:.1%}")
c4.metric("Hit Rate", f"{stats['Hit Rate']:.0%}")
c5.metric("PSR (vs 0)", f"{stats['PSR (vs 0)']:.2f}")

st.caption(
    "PSR = probability the true Sharpe is > 0 after adjusting for sample size, "
    "skew and fat tails. Low PSR with a high Sharpe ⇒ likely overfit / lucky."
)

# --------------------------------------------------------------------------- #
# Equity curve
# --------------------------------------------------------------------------- #
fig = go.Figure()
fig.add_trace(go.Scatter(x=strat_eq.index, y=strat_eq.values,
                         name="Strategy", line=dict(color="#2E86DE", width=2)))
fig.add_trace(go.Scatter(x=bh.index, y=bh.values,
                         name="Buy & Hold", line=dict(color="#999", width=1.5, dash="dash")))
fig.update_layout(title=f"Growth of $1 — {label}", height=420,
                  yaxis_title="Equity (×)", legend=dict(orientation="h"))
st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------------------------------- #
# Drawdown
# --------------------------------------------------------------------------- #
dd = strat_eq / strat_eq.cummax() - 1.0
dd_fig = go.Figure()
dd_fig.add_trace(go.Scatter(x=dd.index, y=dd.values, fill="tozeroy",
                            line=dict(color="#E74C3C"), name="Drawdown"))
dd_fig.update_layout(title="Strategy drawdown", height=260,
                     yaxis_tickformat=".0%")
st.plotly_chart(dd_fig, use_container_width=True)

# --------------------------------------------------------------------------- #
# Stats table + event-driven vs vectorized cross-check
# --------------------------------------------------------------------------- #
left, right = st.columns(2)

with left:
    st.subheader("Full tearsheet")
    table = {}
    for k, v in stats.items():
        table[k] = round(v, 3) if k in ("Sharpe", "Sortino", "PSR (vs 0)") else f"{v:.2%}"
    st.table(pd.DataFrame.from_dict(table, orient="index", columns=["Strategy"]))
    st.caption(f"signals fired: {bt.signals} · fills: {bt.fills} · bars: {len(curve)}")

with right:
    st.subheader("Engine cross-check")
    vec = vectorized_sma_backtest(prices, short_window=short_w, long_window=long_w,
                                  cost_bps=commission_bps + slippage_bps)
    vec_final = vec["equity_curve"].dropna().iloc[-1]
    ed_final = strat_eq.iloc[-1]
    st.write(
        "The vectorized fast-path should land close to the event-driven engine. "
        "Small differences come from cash-based sizing and per-fill costs."
    )
    st.dataframe(pd.DataFrame({
        "Engine": ["Event-driven", "Vectorized"],
        "Final equity (×)": [round(ed_final, 3), round(vec_final, 3)],
        "Max DD": [f"{max_drawdown(strat_eq):.1%}",
                   f"{max_drawdown(vec['equity_curve'].dropna()):.1%}"],
    }), hide_index=True, use_container_width=True)

st.divider()
st.caption(
    "Costs are modeled (commission + slippage in bps). Signals are generated at "
    "the close and affect the next bar's position — no lookahead. Swap in your own "
    "strategy by subclassing `backtester.strategy.Strategy`."
)
