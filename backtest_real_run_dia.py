"""
Runs backtest.py against real daily returns + the REAL historical ATM IV
panel pulled from ThetaData for DIA (Dow) + its constituents
(data/thetadata_fetch_dia.py). This version implements the trading-desk
audit's critical fixes for DIA (chosen for full rigor, unlike QQQ's
pragmatic-scope treatment -- see backtest_real_run.py):

A1 (survivorship bias) -- FIXED: data/fetch.py's build_dia_weights_panel()
reconstructs BOTH real Dow membership (DIA_ERAS) and real price-weight
per historical date, instead of applying today's 30-name roster and a
live weight snapshot across the whole 2021-2026 backtest. INTC and DOW
Inc. (Dow Inc., the chemical company) are real former/era-specific
members pulled alongside the 30 current names for this reason -- INTC is
reused from the QQQ panel (already pulled there) rather than re-pulled.
WBA (Walgreens Boots Alliance) is the one gap: unobtainable for free
(yfinance purged all history after its 2025 delisting; Nasdaq's API has
no record either) -- the pre-2024-02-26 era runs 29 of 30 real members,
disclosed, not silently substituted.

A2 (strike-selection look-ahead) -- FIXED at the data layer:
thetadata_fetch.py's pull_ticker_atm_iv_history now does banded per-day
strike selection (v3) instead of a single fixed strike anchored to
expiration-date spot. Nothing in THIS script needed to change for that
fix -- it's upstream, in how thetadata_iv_panel_dia.csv was pulled.

A3 (correlation formula mismatch) -- FIXED via run_backtest's new
min_weight_coverage parameter: instead of silently renormalizing over
whatever subset of a FIXED basket happened to have data (or requiring
100% and getting almost no trades), each date's ACTUAL captured weight
fraction (out of that date's real point-in-time basket) is computed and
checked against a floor. Below the floor, the date doesn't trade at all;
above it, the trade fires on the actually-available subset, renormalized
-- an explicit, checked trade-off instead of a silent one. DIA's now-
near-full roster coverage (29-32 of 30 real members, depending on era)
means this floor rarely binds hard, unlike QQQ's inherent 15-of-~100
scope limit.
"""

import numpy as np
import pandas as pd

from data.fetch import get_daily_returns, DIA_CONSTITUENTS, build_dia_weights_panel
from backtest import run_backtest, evaluate_trades, compute_max_drawdown, STRESS_SHOCK_PCT

MAX_CONCENTRATION_WARN = 0.35  # same threshold optimizer.py's optimize_for_dispersion_edge
                                 # already uses for its own max_concentration bound -- reused
                                 # here as a reporting flag (audit finding B5), not enforced.

INDEX = "DIA"
MIN_LIQUIDITY_SCORE = 0.5  # same threshold/reasoning as backtest_real_run.py -- data-quality
                            # gate on whether a ticker's PULL is trustworthy at all, separate
                            # from whether/when it was a real Dow member (that's DIA_ERAS' job).
MAX_IV_REALIZED_RATIO = 2.0  # same threshold/reasoning as backtest_real_run.py
MIN_WEIGHT_COVERAGE = 0.90  # see module docstring's A3 section -- up to 10% of a date's real
                             # basket weight can be missing without blocking that date's trade.

print("=" * 70)
print("Loading real historical ATM IV panels (ThetaData: DIA basket + QQQ, for INTC reuse)")
print("=" * 70)
iv_panel = pd.read_csv("data/thetadata_iv_panel_dia.csv", index_col=0, parse_dates=True)
iv_panel.index = iv_panel.index.tz_localize(None)

qqq_panel = pd.read_csv("data/thetadata_iv_panel.csv", index_col=0, parse_dates=True)
qqq_panel.index = qqq_panel.index.tz_localize(None)
if "INTC" not in iv_panel.columns and "INTC" in qqq_panel.columns:
    iv_panel = iv_panel.join(qqq_panel[["INTC"]], how="left")
    print("Reused INTC's IV column from the QQQ panel (real former Dow member, "
          "already pulled there -- see module docstring).")

print(f"IV panel: {iv_panel.shape}, {iv_panel.index.min().date()} to {iv_panel.index.max().date()}")

print("\n" + "=" * 70)
print("Data-quality gate: liquidity (coverage) + implied/realized ratio filters")
print("=" * 70)
all_pulled_tickers = [c for c in iv_panel.columns if c != INDEX]
counts = iv_panel[all_pulled_tickers].count()
liquidity_score = counts / counts.max()
liquidity_failed = liquidity_score[liquidity_score < MIN_LIQUIDITY_SCORE].index.tolist()
print(f"Liquidity-failed ({len(liquidity_failed)}): {liquidity_failed}")

candidates = [t for t in all_pulled_tickers if t not in liquidity_failed]
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
    if median_ratio > MAX_IV_REALIZED_RATIO:
        ratio_failed.append(t)
        print(f"  {t}: median implied/realized ratio = {median_ratio:.2f}x *** FAILED ***")

usable_tickers = [t for t in candidates if t not in ratio_failed]
excluded_tickers = liquidity_failed + ratio_failed
print(f"\n{len(usable_tickers)} of {len(all_pulled_tickers)} pulled tickers pass data-quality "
      f"gates. Excluded: {excluded_tickers}")

print("\n" + "=" * 70)
print("Building point-in-time Dow membership + price-weights (data/fetch.py's DIA_ERAS)")
print("=" * 70)
weights_panel = build_dia_weights_panel(iv_panel.index)
# Zero out anything that failed the data-quality gate above, regardless of
# which era(s) it was a real member in -- a name with bad/insufficient IV
# data shouldn't trade just because it was really in the index.
weights_panel = weights_panel[[c for c in weights_panel.columns if c in usable_tickers]]
weights_panel = weights_panel.reindex(columns=sorted(set(usable_tickers)))
# Sums to <1.0 on dates where a real Dow member exists but failed the
# data-quality gate above -- run_backtest's min_weight_coverage check
# (against IV *availability* on the specific day, a stricter, later check)
# is the one that actually gates trading; this print is just a sanity read.
per_date_captured = weights_panel.sum(axis=1, skipna=True)
print(f"Weights panel: {weights_panel.shape}. Median captured weight per date (before IV-availability "
      f"trimming, i.e. era-membership x data-quality only): {per_date_captured.median():.1%}")

print("\n" + "=" * 70)
print("Pulling real daily returns for every candidate ticker + DIA")
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
          f"avg captured weight: {trades['captured_weight'].mean():.1%}")
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
    "(audit finding B1 fix). Compare the two: if hedged looks meaningfully different from "
    "un-hedged, direction was doing real work in the un-hedged number. Treat both as a rough "
    "first read given the still-small sample size, not a statistically powered verdict."
)
