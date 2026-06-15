"""Performance + risk statistics, and the anti-overfitting metric (PSR).

Everything takes a per-period returns Series and assumes daily data (252
periods/year) unless told otherwise.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def annualized_return(returns: pd.Series, periods: int = 252) -> float:
    n = len(returns)
    if n == 0:
        return 0.0
    total_growth = (1.0 + returns).prod()
    if total_growth <= 0:
        return -1.0
    return total_growth ** (periods / n) - 1.0


def annualized_vol(returns: pd.Series, periods: int = 252) -> float:
    return float(returns.std(ddof=1) * np.sqrt(periods))


def sharpe_ratio(returns: pd.Series, rf: float = 0.0, periods: int = 252) -> float:
    excess = returns - rf / periods
    sd = excess.std(ddof=1)
    if sd == 0 or np.isnan(sd):
        return 0.0
    return float(np.sqrt(periods) * excess.mean() / sd)


def sortino_ratio(returns: pd.Series, rf: float = 0.0, periods: int = 252) -> float:
    excess = returns - rf / periods
    downside = excess[excess < 0]
    dd = downside.std(ddof=1)
    if dd == 0 or np.isnan(dd):
        return 0.0
    return float(np.sqrt(periods) * excess.mean() / dd)


def max_drawdown(equity_curve: pd.Series) -> float:
    """Worst peak-to-trough decline of a cumulative equity curve (negative)."""
    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1.0
    return float(drawdown.min())


def cagr(equity_curve: pd.Series, periods: int = 252) -> float:
    n = len(equity_curve)
    if n < 2 or equity_curve.iloc[0] <= 0:
        return 0.0
    total = equity_curve.iloc[-1] / equity_curve.iloc[0]
    return float(total ** (periods / n) - 1.0)


def probabilistic_sharpe_ratio(returns: pd.Series, benchmark_sr: float = 0.0,
                               periods: int = 252) -> float:
    """Probability that the *true* annualized Sharpe exceeds ``benchmark_sr``.

    Adjusts the observed Sharpe for sample length, skew, and kurtosis
    (Bailey & López de Prado). A high in-sample Sharpe from few, fat-tailed,
    skewed returns gets appropriately discounted — the first defense against
    being fooled by a lucky backtest.
    """
    n = len(returns)
    if n < 3:
        return float("nan")

    sr = sharpe_ratio(returns, periods=periods)
    sr_per_period = sr / np.sqrt(periods)
    bench_per_period = benchmark_sr / np.sqrt(periods)

    skew = float(stats.skew(returns, bias=False))
    kurt = float(stats.kurtosis(returns, fisher=False, bias=False))  # normal -> 3

    denom = 1.0 - skew * sr_per_period + (kurt - 1.0) / 4.0 * sr_per_period**2
    if denom <= 0:
        return float("nan")

    z = (sr_per_period - bench_per_period) * np.sqrt(n - 1) / np.sqrt(denom)
    return float(stats.norm.cdf(z))


def summary_stats(curve: pd.DataFrame, periods: int = 252) -> dict:
    """Compute the headline numbers from an equity-curve DataFrame.

    ``curve`` needs a ``returns`` column and an ``equity_curve`` column (both are
    produced by ``Portfolio.equity_curve`` and the vectorized backtest).
    """
    returns = curve["returns"].dropna()
    equity = curve["equity_curve"].dropna()

    wins = (returns > 0).sum()
    active = (returns != 0).sum()
    hit_rate = float(wins / active) if active else 0.0

    return {
        "Total Return": float(equity.iloc[-1] - 1.0) if len(equity) else 0.0,
        "CAGR": cagr(equity, periods),
        "Ann. Return": annualized_return(returns, periods),
        "Ann. Volatility": annualized_vol(returns, periods),
        "Sharpe": sharpe_ratio(returns, periods=periods),
        "Sortino": sortino_ratio(returns, periods=periods),
        "Max Drawdown": max_drawdown(equity),
        "Hit Rate": hit_rate,
        "PSR (vs 0)": probabilistic_sharpe_ratio(returns, 0.0, periods),
    }


def tearsheet(curve: pd.DataFrame, periods: int = 252) -> pd.DataFrame:
    """A printable one-column table of the summary stats."""
    stats_dict = summary_stats(curve, periods)
    formatted = {}
    for k, v in stats_dict.items():
        if k in ("Sharpe", "Sortino", "PSR (vs 0)"):
            formatted[k] = round(v, 3)
        else:
            formatted[k] = f"{v:.2%}"
    return pd.DataFrame.from_dict(formatted, orient="index", columns=["Strategy"])
