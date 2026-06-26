# Concept notebooks — Backtesting framework (+ shared foundations)

Tiny, interactive notebooks that teach the ideas behind this project from zero. No
prior finance knowledge assumed — open one, run the cells top to bottom (Shift+Enter),
and change the numbers. Each ends by pointing at the exact `backtester/` file that
does the same thing for real.

**This folder also holds the shared foundations** for *all* the projects, because the
backtester is the base everything plugs into. If you're new, start here.

| # | Notebook | Teaches |
|---|----------|---------|
| 01 | `01_prices_returns_volatility.ipynb` | prices → returns (simple vs log) → volatility & √time annualization |
| 02 | `02_distributions_and_fat_tails.ipynb` | normal vs fat tails, skew/kurtosis, the Jarque–Bera test |
| 03 | `03_random_walk_gbm.ipynb` | random walks, Geometric Brownian Motion, Monte Carlo |
| 04 | `04_performance_metrics.ipynb` | equity curve, Sharpe, Sortino, max drawdown, CAGR |
| 05 | `05_backtesting_and_lookahead.ipynb` | SMA signals, the lookahead trap, trading costs |

**Prerequisites:** none — 01→05 is itself the recommended starting order for the
whole repo.

**Run them:** any Jupyter (`jupyter lab` / VS Code / etc.). They need only
`numpy`, `pandas`, `scipy`, `matplotlib`.

See the big-picture writeup in [`../../QUANT_KNOWLEDGE_BASE.md`](../../QUANT_KNOWLEDGE_BASE.md)
(Part 1 = foundations, §08 = this project).
