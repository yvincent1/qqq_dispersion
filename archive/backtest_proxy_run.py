"""
Runs backtest.py against ~5 years of REAL returns + a PROXY implied-vol
panel (see data/proxy_iv.py) -- the interim option while WRDS/OptionMetrics
access is pending.

This is real market data, not synthetic -- but the IV side is still an
approximation (VXN-derived vol risk premium applied uniformly across
names, not actual single-name options prices). Treat results here as
"is the pipeline plausible on real data" and a rough first look, not a
final answer on whether the strategy is profitable -- re-run this same
script against data/wrds_fetch.py (or a paid vendor) once real
single-name historical IV is available, swapping only the data-loading
lines below.
"""

import numpy as np

from data.fetch import get_daily_returns, QQQ_TOP_CONSTITUENTS
from data.proxy_iv import build_proxy_iv_panel
from backtest import run_backtest, evaluate_trades

INDEX = "QQQ"
WEIGHTS = QQQ_TOP_CONSTITUENTS

print("=" * 70)
print("Pulling ~5y real daily returns for the basket + QQQ")
print("=" * 70)
tickers = list(WEIGHTS.keys())
returns_panel = get_daily_returns(tickers + [INDEX], period="5y")
print(f"Returns: {returns_panel.shape}, "
      f"{returns_panel.index.min().date()} to {returns_panel.index.max().date()}")
missing = returns_panel.columns[returns_panel.isna().all()].tolist()
if missing:
    print(f"WARNING: no return data at all for {missing} -- dropping from the basket")
    tickers = [t for t in tickers if t not in missing]
    returns_panel = returns_panel.drop(columns=missing)

print("\n" + "=" * 70)
print("Building proxy IV panel (VXN vol-risk-premium x trailing realized vol)")
print("=" * 70)
iv_panel = build_proxy_iv_panel(returns_panel, index_ticker=INDEX)
usable_frac = iv_panel.dropna().shape[0] / len(iv_panel)
print(f"{iv_panel.dropna().shape[0]} of {len(iv_panel)} dates have a full proxy IV row "
      f"({usable_frac:.0%})")

print("\n" + "=" * 70)
print("Running backtest (weekly checks, 5-day holding period, 63-day lookback)")
print("=" * 70)
trades = run_backtest(returns_panel, iv_panel, WEIGHTS, index_ticker=INDEX)
stats = evaluate_trades(trades)

print(f"\nn_trades={stats['n_trades']}")
if stats["n_trades"] > 0:
    print(f"win_rate={stats['win_rate']:.1%}  avg_pnl={stats['avg_pnl']:.3f}  "
          f"std_pnl={stats['std_pnl']:.3f}  sharpe={stats['sharpe']:.2f}")
    print(f"signal mix: {trades['signal'].value_counts().to_dict()}")
    if stats["by_spread_bucket"] is not None:
        print("\nP&L by |spread| bucket:")
        print(stats["by_spread_bucket"])
    print("\nMost recent 5 trades:")
    print(trades.tail()[["entry_date", "exit_date", "signal", "spread", "pnl"]]
          .to_string(index=False))
else:
    print("No trades fired -- check spread_threshold / data coverage.")

print(
    "\nReminder: IV side is a VXN-derived proxy, not real single-name options data, and "
    "trades aren't delta-hedged (see backtest.py's module docstring) -- treat this as a "
    "plausibility check on the pipeline, not a profitability verdict."
)
