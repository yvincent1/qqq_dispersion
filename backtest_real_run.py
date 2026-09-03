"""
Runs backtest.py against real daily returns + the REAL historical ATM IV
panel pulled from ThetaData (data/thetadata_fetch.py) -- both legs are
genuine market data, no synthetic/proxy approximation on either side.

This version implements the trading-desk audit's fixes for QQQ, chosen
as PRAGMATIC scope rather than DIA's full rigor (see backtest_real_run_dia.py):

A1 (survivorship bias) -- PARTIALLY fixed, disclosed limit: membership is
held fixed at today's top-15 QQQ_TOP_CONSTITUENTS -- this project does
NOT reconstruct the true ~100-name Nasdaq-100's historical membership or
ranking (no free historical weight file exists, and expanding to ~100
names was explicitly scoped out as too large an undertaking for this
pass). What IS fixed: given these 15 names, each one's real point-in-time
weight (data/fetch.py's build_qqq_weights_panel -- real historical
market cap: split-adjusted price x real historical shares outstanding,
not today's single static weight snapshot applied across 5.5 years).
This is a real improvement (NVDA's real 2021 weight in this basket was
~3%, not the ~8.5% the old static table implied), but does NOT answer
"were these actually the right 15 names for a given historical date" --
that gap stays open and should be read as a standing caveat on every
result below, not as resolved.

A2 (strike-selection look-ahead) -- FIXED at the data layer, same as DIA:
thetadata_fetch.py's pull_ticker_atm_iv_history now does banded per-day
strike selection instead of a single fixed strike anchored to
expiration-date spot. Nothing in THIS script needed to change for that.

A3 (correlation formula mismatch) -- Same partial-fix ceiling as A1: the
DOUBLE mismatch (trimming further within the 15 for liquidity/ratio
reasons, then comparing against the real QQQ index IV) is fixed via
run_backtest's min_weight_coverage floor, same mechanism as DIA. The
DEEPER mismatch (15-of-~100 real index weight) is NOT fixed and can't be
without the A1 scope expansion -- captured_weight in the trade log
reflects coverage within the 15-name universe only, not true Nasdaq-100
weight capture.
"""

import numpy as np
import pandas as pd

from data.fetch import get_daily_returns, QQQ_TOP_CONSTITUENTS, build_qqq_weights_panel
from backtest import run_backtest, evaluate_trades, compute_max_drawdown, STRESS_SHOCK_PCT

MAX_CONCENTRATION_WARN = 0.35  # same threshold optimizer.py's optimize_for_dispersion_edge
                                 # already uses for its own max_concentration bound -- reused
                                 # here as a reporting flag (audit finding B5), not enforced.

INDEX = "QQQ"
MIN_LIQUIDITY_SCORE = 0.5  # see module docstring -- stable across 0.4-0.6
MAX_IV_REALIZED_RATIO = 2.0  # see module docstring -- isolates data-quality outliers cleanly
MIN_WEIGHT_COVERAGE = 0.90  # audit A3 fix -- see module docstring

print("=" * 70)
print("Loading real historical ATM IV panel (ThetaData)")
print("=" * 70)
iv_panel = pd.read_csv("data/thetadata_iv_panel.csv", index_col=0, parse_dates=True)
iv_panel.index = iv_panel.index.tz_localize(None)  # match get_daily_returns' tz-naive index below --
                                                     # a tz mismatch here silently zeroes every trade
                                                     # (entry_date lookups just never match) rather than
                                                     # raising, so this is easy to get wrong unnoticed.
print(f"IV panel: {iv_panel.shape}, {iv_panel.index.min().date()} to {iv_panel.index.max().date()}")

print("\n" + "!" * 70)
print("SCOPE LIMIT (audit A1/A3, disclosed not fixed): this basket is the top-15 QQQ names")
print("BY WEIGHT TODAY, not a reconstruction of the true ~100-name Nasdaq-100's historical")
print("membership. Point-in-time WEIGHT (within these 15) is real; point-in-time MEMBERSHIP")
print("(which 15 names) is not. Treat results as basket-quality evidence, not full-index proof.")
print("!" * 70)

print("\n" + "=" * 70)
print("Data-quality gate: liquidity (coverage) + implied/realized ratio filters")
print("=" * 70)
all_tickers = list(QQQ_TOP_CONSTITUENTS.keys())
counts = iv_panel[all_tickers].count()
liquidity_score = counts / counts.max()
liquidity_failed = liquidity_score[liquidity_score < MIN_LIQUIDITY_SCORE].index.tolist()
print(f"Liquidity-failed ({len(liquidity_failed)}): {liquidity_failed}")

candidates = [t for t in all_tickers if t not in liquidity_failed]
returns_for_ratio = get_daily_returns(candidates, period="10y")
returns_for_ratio.index = returns_for_ratio.index.tz_localize(None)
realized_vol_21d = returns_for_ratio.rolling(21).std() * np.sqrt(252)

ratio_failed = []
for t in candidates:
    combined = pd.DataFrame({"implied": iv_panel[t]}).join(
        realized_vol_21d[t].rename("realized"), how="left").dropna()
    if combined.empty:
        continue
    median_ratio = (combined["implied"] / combined["realized"]).median()
    flag = " *** FAILED ***" if median_ratio > MAX_IV_REALIZED_RATIO else ""
    print(f"  {t}: median implied/realized ratio = {median_ratio:.2f}x{flag}")
    if median_ratio > MAX_IV_REALIZED_RATIO:
        ratio_failed.append(t)

usable_tickers = [t for t in candidates if t not in ratio_failed]
print(f"\n{len(usable_tickers)} of {len(all_tickers)} names pass data-quality gates. "
      f"Excluded: {liquidity_failed + ratio_failed}")

print("\n" + "=" * 70)
print("Building point-in-time market-cap-proxy weights within the fixed 15-name universe")
print("=" * 70)
weights_panel_full = build_qqq_weights_panel(iv_panel.index, tickers=usable_tickers)
weights_panel = weights_panel_full  # membership is fixed (no era exclusion) in this pragmatic version
print(f"Weights panel: {weights_panel.shape}")

print("\n" + "=" * 70)
print("Pulling real daily returns for the usable basket + QQQ")
print("=" * 70)
tickers = list(weights_panel.columns)
returns_panel = get_daily_returns(tickers + [INDEX], period="10y")
returns_panel.index = returns_panel.index.tz_localize(None)
print(f"Returns: {returns_panel.shape}, "
      f"{returns_panel.index.min().date()} to {returns_panel.index.max().date()}")
missing = returns_panel.columns[returns_panel.isna().all()].tolist()
if missing:
    print(f"WARNING: no return data at all for {missing} -- these will simply never contribute weight")

print("\n" + "=" * 70)
print(f"Running backtest (daily scan, 5-day holding period, 63-day lookback, "
      f"min_weight_coverage={MIN_WEIGHT_COVERAGE:.0%})")
print("=" * 70)
trades = run_backtest(returns_panel, iv_panel, weights_panel, index_ticker=INDEX,
                       check_every=1, min_weight_coverage=MIN_WEIGHT_COVERAGE)
stats = evaluate_trades(trades, pnl_col="pnl")
hedged_stats = evaluate_trades(trades, pnl_col="hedged_pnl")

print(f"\nn_trades={stats['n_trades']}")
if stats["n_trades"] > 0:
    print(f"{'UN-hedged':<12} win_rate={stats['win_rate']:.1%}  avg_pnl={stats['avg_pnl']:.3f}  "
          f"std_pnl={stats['std_pnl']:.3f}  sharpe={stats['sharpe']:.2f}")
    print(f"{'Hedged':<12} win_rate={hedged_stats['win_rate']:.1%}  avg_pnl={hedged_stats['avg_pnl']:.3f}  "
          f"std_pnl={hedged_stats['std_pnl']:.3f}  sharpe={hedged_stats['sharpe']:.2f}")
    print(f"signal mix: {trades['signal'].value_counts().to_dict()}")
    print(f"avg names traded per trade: {trades['n_names'].mean():.1f}  "
          f"avg captured weight (of the 15-name universe): {trades['captured_weight'].mean():.1%}")
    if stats["by_spread_bucket"] is not None:
        print("\nUN-hedged P&L by |spread| bucket:")
        print(stats["by_spread_bucket"])
    if hedged_stats["by_spread_bucket"] is not None:
        print("\nHedged P&L by |spread| bucket:")
        print(hedged_stats["by_spread_bucket"])

    print("\n" + "=" * 70)
    print("Risk reporting (audit finding B5)")
    print("=" * 70)
    dd = compute_max_drawdown(trades, pnl_col="pnl")
    dd_hedged = compute_max_drawdown(trades, pnl_col="hedged_pnl")
    print(f"Max drawdown (un-hedged): {dd['max_drawdown']:.2f}  "
          f"(peak {dd['peak_date']} -> trough {dd['trough_date']})")
    print(f"Max drawdown (hedged):    {dd_hedged['max_drawdown']:.2f}  "
          f"(peak {dd_hedged['peak_date']} -> trough {dd_hedged['trough_date']})")

    conc_flagged = trades[trades["max_name_weight"] > MAX_CONCENTRATION_WARN]
    print(f"\nConcentration: {len(conc_flagged)} of {len(trades)} trades exceed "
          f"{MAX_CONCENTRATION_WARN:.0%} single-name weight (max observed: "
          f"{trades['max_name_weight'].max():.1%})")

    stress_stats = evaluate_trades(trades, pnl_col="stress_pnl")
    print(f"\nStress scenario (uniform {STRESS_SHOCK_PCT:.0%} simultaneous shock to every leg, "
          f"applied at entry): avg_pnl={stress_stats['avg_pnl']:.2f}  "
          f"worst_trade={trades['stress_pnl'].min():.2f}  "
          f"{(trades['stress_pnl'] < 0).sum()} of {len(trades)} trades would have lost money")

    print("\nAll trades:")
    print(trades[["entry_date", "exit_date", "signal", "spread", "n_names",
                   "captured_weight", "max_name_weight", "pnl", "hedged_pnl",
                   "stress_pnl"]].to_string(index=False))
else:
    print("No trades fired -- check spread_threshold / data coverage.")

print(
    "\nBoth legs are real market data. UN-hedged pnl marks entry/exit only (directional/gamma "
    "noise included); hedged_pnl daily-delta-hedges each leg against the real observed spot path "
    "(audit finding B1 fix -- see backtest.py's _daily_delta_hedge_pnl). Compare the two: if hedged "
    "looks meaningfully different from un-hedged, direction was doing real work in the un-hedged "
    "number. Treat both as a rough first read given the still-small sample size, not a statistically "
    "powered verdict. Remember the SCOPE LIMIT printed above: this is a 15-name proxy basket, not "
    "the true Nasdaq-100."
)
