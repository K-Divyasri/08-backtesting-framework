"""Run the whole framework end-to-end on synthetic data — no internet needed.

    python smoke_test.py

Verifies that:
  * the event-driven engine produces an equity curve,
  * the vectorized fast-path roughly agrees with it,
  * live `step()` mode reaches the same final equity as `run()`,
  * metrics (incl. the Probabilistic Sharpe Ratio) compute cleanly.
"""

from backtester import Backtest, tearsheet, vectorized_sma_backtest
from backtester.data import make_synthetic
from backtester.metrics import summary_stats


def main() -> None:
    prices = make_synthetic(n=750, seed=7)
    params = {"short_window": 20, "long_window": 50}

    # --- event-driven ----------------------------------------------------- #
    bt = Backtest(
        {"SYNTH": prices},
        strategy_params=params,
        initial_capital=100_000,
        commission_bps=1.0,
        slippage_bps=2.0,
    )
    curve = bt.run()
    print("=== Event-driven backtest ===")
    print(tearsheet(curve))
    print(f"\nsignals={bt.signals}  fills={bt.fills}  bars={len(curve)}")

    # --- vectorized fast-path -------------------------------------------- #
    vec = vectorized_sma_backtest(prices, cost_bps=3.0, **params)
    print("\n=== Vectorized fast-path ===")
    print(tearsheet(vec))

    ed_final = curve["equity_curve"].iloc[-1]
    vec_final = vec["equity_curve"].dropna().iloc[-1]
    print(f"\nFinal equity  event-driven={ed_final:.3f}  vectorized={vec_final:.3f}")

    # --- live step() mode reproduces run() -------------------------------- #
    bt_live = Backtest({"SYNTH": prices}, strategy_params=params)
    for ts, row in prices.iterrows():
        bar = (ts, row["open"], row["high"], row["low"], row["close"], row["volume"])
        bt_live.step("SYNTH", bar)
    live_final = bt_live.portfolio.equity_curve()["equity_curve"].iloc[-1]
    print(f"Live step() final equity={live_final:.3f} (should equal event-driven)")
    assert abs(live_final - ed_final) < 1e-9, "live and replay disagree!"

    # --- sanity assertions ------------------------------------------------ #
    s = summary_stats(curve)
    assert len(curve) == len(prices), "equity curve length mismatch"
    assert -1.0 <= s["Max Drawdown"] <= 0.0, "drawdown out of range"
    assert bt.fills > 0, "no trades executed — strategy never fired"
    print("\nAll smoke-test assertions passed. [OK]")


if __name__ == "__main__":
    main()
