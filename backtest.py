"""
Backtest engine for the dispersion signal.

Deliberately data-source agnostic: run_backtest() takes pre-built panels
of historical returns and implied vols and doesn't care whether they came
from WRDS/OptionMetrics, a paid vendor (Polygon/ORATS), or a synthetic
generator for validating the mechanics (see backtest_demo.py). Swapping
the data source later means writing a new loader, not touching this file.

Trade construction mirrors demo_pipeline.py / run_live.py: vega-sized,
NOT delta-hedged (matches optimizer.py's existing scope). That means any
single trade's P&L carries real directional/gamma noise on top of the
correlation-gap edge this strategy is actually targeting -- only
aggregate stats over many trades are informative. Adding delta-hedging
to isolate the pure vol/correlation view is a natural next extension,
same spirit as the other simplifications flagged in README.md.
"""

import numpy as np
import pandas as pd

from iv_solver import bs_price, bs_vega, bs_delta
from correlation import implied_correlation, realized_correlation, dispersion_signal
from optimizer import size_basket_by_vega

STRESS_SHOCK_PCT = -0.20  # uniform, simultaneous spot shock applied to every name (index
                           # AND every basket leg) for the per-trade stress test (audit
                           # finding B5) -- a round, deliberately severe number (a 20% single-
                           # day move is a real, if rare, market event -- e.g. 1987's crash,
                           # single-day COVID-crash sessions), not tuned to produce any
                           # particular result.


def run_backtest(returns_panel, iv_panel, weights, index_ticker="QQQ",
                  lookback_days=63, holding_days=5, check_every=5,
                  spread_threshold=0.10, option_tenor_days=30,
                  r=0.045, n_index_straddles=10.0, cost_bps=5.0,
                  min_weight_coverage=1.0):
    """
    returns_panel: DataFrame, index=date, columns include every ticker
                   that could ever appear in `weights` plus index_ticker.
                   Daily returns.
    iv_panel: DataFrame, same column universe as returns_panel. Annualized
              ATM IV at ~option_tenor_days DTE, per name per date. May
              have gaps (real vendor data will) -- names missing a value
              on a given date are excluded from THAT date's basket (see
              min_weight_coverage), not treated as an all-or-nothing block.
    weights: EITHER
             (a) dict or pd.Series of {ticker: index weight}, basket
                 tickers only (not index_ticker) -- held STATIC across the
                 whole backtest, the original behavior. Use this when the
                 basket's true membership genuinely doesn't change (or
                 when that's an accepted, disclosed scope limit -- see
                 data/fetch.py's build_qqq_weights_panel docstring).
             (b) pd.DataFrame, index=date, columns=every ticker that was
                 EVER a member, values=weight-or-NaN (NaN = "not a member
                 on this date", not a data gap). Use this for a real
                 point-in-time basket, e.g. data/fetch.py's
                 build_dia_weights_panel() -- fixes the survivorship-bias
                 finding from the trading-desk audit (basket composition
                 AND weight both vary by date, not just weight).
    lookback_days: trailing window for the realized-correlation estimate.
    holding_days: fixed holding period per trade, in trading days.
    check_every: how often (in trading days) to check for a new signal.
                 Set equal to holding_days so trades don't overlap.
    spread_threshold: |implied - realized| corr gap needed to trade.
    option_tenor_days: tenor used to price the straddles (both legs
                        share strikes fixed at entry spot, ATM at entry).
    n_index_straddles: size of the index leg; the basket leg is sized to
                        match its vega via size_basket_by_vega.
    cost_bps: flat transaction-cost assumption, in bps of straddle
              premium notional, charged on BOTH entry and exit (so a
              full round trip pays ~2x cost_bps drag). Crude placeholder
              for the "no costs modeled" gap noted in README.md.
    min_weight_coverage: minimum fraction of that DATE's basket weight
              (from `weights`, before IV-coverage trimming) that must
              actually have a usable IV reading for the date to trade at
              all -- e.g. 0.90 means up to 10% of basket weight can be
              missing that day, silently excluded and the rest
              renormalized, without blocking the trade. Defaults to 1.0
              (require every basket name present, the original
              all-or-nothing behavior) so existing static-weights callers
              are unaffected unless they opt into a lower floor.
              Direct fix for the audit's correlation-formula finding: the
              captured weight fraction is now an explicit, checked number
              instead of a silent renormalization over whatever happened
              to be non-NaN.

    Returns: pd.DataFrame trade log, one row per executed (non-neutral)
             signal, columns: entry_date, exit_date, signal, spread,
             implied_corr, realized_corr, captured_weight, pnl, index_pnl,
             basket_pnl, costs.
    """
    time_varying = isinstance(weights, pd.DataFrame)
    if time_varying:
        universe_tickers = list(weights.columns)
    else:
        universe_tickers = list(weights.keys())

    all_possible_cols = [c for c in universe_tickers + [index_ticker] if c in returns_panel.columns]
    prices = 100 * (1 + returns_panel[all_possible_cols]).cumprod()
    dates = returns_panel.index

    # check_every controls the scan step when NOTHING fires (can be 1, to scan
    # every day and catch sparse-coverage good dates the old fixed-5-day stride
    # would mostly miss) -- but once a trade DOES fire, always skip a full
    # holding_days ahead regardless of check_every, so trades never overlap.
    # Previously check_every was the ONLY step size, which only worked because
    # it defaulted to exactly holding_days; decoupling them here is what makes
    # check_every=1 safe to use.
    trades = []
    t = lookback_days
    while t + holding_days < len(dates):
        entry_date = dates[t]
        exit_date = dates[t + holding_days]

        if entry_date not in iv_panel.index or exit_date not in iv_panel.index:
            t += check_every
            continue

        # Resolve this date's basket weight (point-in-time roster+weight if
        # `weights` is a DataFrame, otherwise the single static Series).
        if time_varying:
            if entry_date not in weights.index:
                t += check_every
                continue
            w_full = weights.loc[entry_date].dropna()
            if w_full.empty:
                t += check_every
                continue
        else:
            w_full = pd.Series(weights, dtype=float)

        # Which of THIS date's basket names actually have a usable entry IV
        # reading -- captured_weight makes the coverage trade-off explicit
        # (audit A3 fix) instead of silently requiring 100% or silently
        # renormalizing over an arbitrary non-NaN subset.
        available_cols = [c for c in w_full.index if c in iv_panel.columns]
        entry_iv = iv_panel.loc[entry_date, available_cols] if available_cols else pd.Series(dtype=float)
        present_tickers = entry_iv.dropna().index.tolist()

        captured_weight = w_full[present_tickers].sum() / w_full.sum() if present_tickers else 0.0
        if captured_weight < min_weight_coverage:
            t += check_every
            continue

        tickers = present_tickers
        w = w_full[tickers]
        w = w / w.sum()
        needed_cols = tickers + [index_ticker]

        if iv_panel.loc[entry_date, [index_ticker]].isna().any():
            t += check_every
            continue

        window = returns_panel[tickers].iloc[t - lookback_days: t]
        realized_rho = realized_correlation(window, weights=w)

        iv_entry = iv_panel.loc[entry_date]
        implied_rho = implied_correlation(iv_entry[index_ticker], w.values,
                                           iv_entry[tickers].values)

        if np.isnan(realized_rho) or np.isnan(implied_rho):
            t += check_every
            continue

        spread = implied_rho - realized_rho
        signal = dispersion_signal(implied_rho, realized_rho, spread_threshold)

        if signal != "neutral" and not iv_panel.loc[exit_date, needed_cols].isna().any():
            pnl, legs = _simulate_trade(
                signal, entry_date, exit_date, prices, iv_panel, w, tickers,
                index_ticker, option_tenor_days, r, n_index_straddles, cost_bps,
            )
            trades.append({
                "entry_date": entry_date, "exit_date": exit_date,
                "signal": signal, "spread": spread,
                "implied_corr": implied_rho, "realized_corr": realized_rho,
                "captured_weight": captured_weight, "n_names": len(tickers),
                "max_name_weight": float(w.max()),  # concentration check (audit B5) --
                                                      # largest single basket name's share
                                                      # of THIS trade's weight, post-
                                                      # renormalization.
                "pnl": pnl, **legs,
            })
            t += max(check_every, holding_days)
            continue

        t += check_every

    return pd.DataFrame(trades)


def _straddle_delta(S, K, T, r, sigma):
    return bs_delta(S, K, T, r, sigma, "call") + bs_delta(S, K, T, r, sigma, "put")


def _daily_delta_hedge_pnl(spot_path, K, T_entry, r, sigma, entry_date, exit_date):
    """
    Daily-rebalanced delta-hedge P&L for ONE LONG straddle (1 unit,
    call+put), held from entry_date to exit_date along the REAL observed
    daily spot path -- audit finding B1 (backtest P&L conflates direction
    with the correlation view since the original version only marks
    entry/exit, with the straddle sitting fully un-hedged in between).

    IV is held fixed at entry (sigma) for the whole hold, used both to
    price the straddle and to compute each day's hedge delta -- a
    standard backtest simplification, not a live-desk assumption: real
    intraday-updated IV isn't reliably available for every name on every
    day in this dataset (the same sparsity min_weight_coverage exists to
    handle), so re-solving vol daily isn't something this pipeline can
    actually support yet.

    Returns (raw_mtm_pnl, hedge_pnl) for a LONG position:
      raw_mtm_pnl = straddle value at exit minus at entry (exactly what
                    the ORIGINAL entry/exit-only P&L already computed).
      hedge_pnl   = P&L from shorting delta_t shares of the underlying
                    each day and closing it out the next day, repeated
                    daily -- this is what's NEW, and is what a real
                    hedged position would additionally earn/lose.
    Total UN-hedged P&L (long) = raw_mtm_pnl.
    Total    hedged P&L (long) = raw_mtm_pnl + hedge_pnl.
    Caller applies the actual position sign (long/short) and size.
    """
    idx = spot_path.index
    if len(idx) < 2:
        # Degenerate holding window (shouldn't happen with holding_days>=1
        # trading days between real dates, but guard rather than crash).
        S0 = float(spot_path.iloc[0])
        price0 = bs_price(S0, K, T_entry, r, sigma, "call") + bs_price(S0, K, T_entry, r, sigma, "put")
        return 0.0, 0.0

    hedge_pnl = 0.0
    for i in range(len(idx) - 1):
        d_t, d_next = idx[i], idx[i + 1]
        days_since_entry = (d_t - entry_date).days
        T_t = max(T_entry - days_since_entry / 365.0, 1e-6)
        S_t = float(spot_path.loc[d_t])
        S_next = float(spot_path.loc[d_next])
        delta_t = _straddle_delta(S_t, K, T_t, r, sigma)
        # Long straddle hedges by shorting delta_t shares -- the hedge P&L
        # from holding that short position for one day is -delta_t times
        # the day's price change.
        hedge_pnl += -delta_t * (S_next - S_t)

    days_held = (exit_date - entry_date).days
    T_exit = max(T_entry - days_held / 365.0, 1e-6)
    S_entry_val, S_exit_val = float(spot_path.loc[idx[0]]), float(spot_path.loc[idx[-1]])
    price_entry = bs_price(S_entry_val, K, T_entry, r, sigma, "call") + bs_price(S_entry_val, K, T_entry, r, sigma, "put")
    price_exit = bs_price(S_exit_val, K, T_exit, r, sigma, "call") + bs_price(S_exit_val, K, T_exit, r, sigma, "put")
    raw_mtm_pnl = price_exit - price_entry
    return raw_mtm_pnl, hedge_pnl


def _simulate_trade(signal, entry_date, exit_date, prices, iv_panel, w, tickers,
                     index_ticker, option_tenor_days, r, n_index_straddles, cost_bps):
    """
    Prices an index straddle vs. a vega-matched basket of straddles at
    entry (strikes fixed ATM-at-entry), marks both to market at exit
    using that date's IV and the realized spot move, and returns net P&L
    -- both UN-hedged (original behavior, positions held flat for the
    whole holding period) and daily-DELTA-HEDGED (audit finding B1: a
    real desk rebalances a stock hedge daily against each leg, stripping
    out most of the directional/gamma noise that dominates the un-hedged
    number). Both are returned so the difference is visible, not just
    the hedged number substituted in silently.

    sell_dispersion = short index straddle, long basket straddles.
    buy_dispersion  = long index straddle, short basket straddles.
    """
    T_entry = option_tenor_days / 365.0
    days_held = (exit_date - entry_date).days
    T_exit = max(T_entry - days_held / 365.0, 1e-6)

    S_entry, S_exit = prices.loc[entry_date], prices.loc[exit_date]
    iv_entry, iv_exit = iv_panel.loc[entry_date], iv_panel.loc[exit_date]

    def straddle_price(S, K, T, sigma):
        return bs_price(S, K, T, r, sigma, "call") + bs_price(S, K, T, r, sigma, "put")

    def straddle_vega(S, K, T, sigma):
        return 2 * bs_vega(S, K, T, r, sigma)

    K_index = S_entry[index_ticker]
    index_price_entry = straddle_price(S_entry[index_ticker], K_index, T_entry, iv_entry[index_ticker])
    index_price_exit = straddle_price(S_exit[index_ticker], K_index, T_exit, iv_exit[index_ticker])
    index_vega_entry = straddle_vega(S_entry[index_ticker], K_index, T_entry, iv_entry[index_ticker])

    target_vega = n_index_straddles * index_vega_entry
    basket_vegas_entry = np.array([
        straddle_vega(S_entry[t], S_entry[t], T_entry, iv_entry[t]) for t in tickers
    ])
    counts = size_basket_by_vega(target_vega, w.values, basket_vegas_entry)

    basket_price_entry = np.array([
        straddle_price(S_entry[t], S_entry[t], T_entry, iv_entry[t]) for t in tickers
    ])
    basket_price_exit = np.array([
        straddle_price(S_exit[t], S_entry[t], T_exit, iv_exit[t]) for t in tickers
    ])

    sign = 1.0 if signal == "sell_dispersion" else -1.0
    index_pnl = sign * n_index_straddles * (index_price_entry - index_price_exit)
    basket_pnl = -sign * float(np.sum(counts * (basket_price_entry - basket_price_exit)))

    notional_traded = (
        n_index_straddles * (index_price_entry + index_price_exit)
        + float(np.sum(np.abs(counts) * (basket_price_entry + basket_price_exit)))
    )
    costs = (cost_bps / 10000.0) * notional_traded

    pnl = index_pnl + basket_pnl - costs

    # Daily delta-hedge P&L (audit finding B1) -- added ON TOP of the
    # un-hedged marks above, not replacing them. Position sign convention
    # verified algebraically against the existing index_pnl/basket_pnl
    # formulas: index_position_sign = -sign (matches "sell_dispersion =
    # short index"), basket_position_sign = +sign uniformly across all
    # basket names (counts from size_basket_by_vega are always positive
    # sizes, never signed -- the actual long/short direction is `sign`
    # itself, same for every leg, matching "sell_dispersion = long basket").
    index_spot_path = prices.loc[entry_date:exit_date, index_ticker]
    _, index_hedge = _daily_delta_hedge_pnl(
        index_spot_path, K_index, T_entry, r, iv_entry[index_ticker], entry_date, exit_date)
    index_hedge_pnl = (-sign) * n_index_straddles * index_hedge

    basket_hedge_pnl = 0.0
    for i, t in enumerate(tickers):
        leg_spot_path = prices.loc[entry_date:exit_date, t]
        _, leg_hedge = _daily_delta_hedge_pnl(
            leg_spot_path, S_entry[t], T_entry, r, iv_entry[t], entry_date, exit_date)
        basket_hedge_pnl += sign * counts[i] * leg_hedge

    hedged_pnl = pnl + index_hedge_pnl + basket_hedge_pnl

    # Stress scenario (audit finding B5): what would this trade's raw,
    # un-hedged P&L have been if every name -- index AND every basket leg
    # -- moved STRESS_SHOCK_PCT simultaneously, right at entry (instant
    # shock: no time decay, IV held at entry values)? A correlated,
    # market-wide move is exactly the scenario an un-hedged dispersion
    # trade is NOT protected against -- both legs can lose at once, which
    # normal win-rate/Sharpe stats over calm historical windows won't
    # surface on their own. Simpler than a full path simulation (matches
    # the delta-hedge helper's fixed-IV simplification), sufficient to
    # size the tail exposure per trade.
    S_index_shocked = S_entry[index_ticker] * (1 + STRESS_SHOCK_PCT)
    index_price_shocked = straddle_price(S_index_shocked, K_index, T_entry, iv_entry[index_ticker])
    index_stress_pnl = sign * n_index_straddles * (index_price_entry - index_price_shocked)

    basket_price_shocked = np.array([
        straddle_price(S_entry[t] * (1 + STRESS_SHOCK_PCT), S_entry[t], T_entry, iv_entry[t]) for t in tickers
    ])
    basket_stress_pnl = -sign * float(np.sum(counts * (basket_price_entry - basket_price_shocked)))
    stress_pnl = index_stress_pnl + basket_stress_pnl - costs

    return pnl, {
        "index_pnl": index_pnl, "basket_pnl": basket_pnl, "costs": costs,
        "hedged_pnl": hedged_pnl, "hedge_pnl": index_hedge_pnl + basket_hedge_pnl,
        "stress_pnl": stress_pnl,
    }


def evaluate_trades(trades_df, pnl_col="pnl"):
    """
    Summary stats for a trade log from run_backtest(), plus a
    spread-magnitude-bucketed breakdown to test the core question:
    does a bigger implied-vs-realized gap predict bigger/more reliable
    P&L? Empty-safe (returns NaNs, not an error, if no trades fired).

    pnl_col: which P&L column to summarize -- "pnl" (default, UN-hedged,
             entry/exit marks only) or "hedged_pnl" (daily delta-hedged,
             audit finding B1). Compute both on the same trades_df to see
             how much of the un-hedged edge survives once directional/
             gamma noise is stripped out -- they're computed from the
             exact same signals/trades, just different P&L accounting.
    """
    if trades_df.empty:
        return {"n_trades": 0, "win_rate": np.nan, "avg_pnl": np.nan,
                "std_pnl": np.nan, "sharpe": np.nan, "by_spread_bucket": None}

    n = len(trades_df)
    pnl = trades_df[pnl_col]
    win_rate = (pnl > 0).mean()
    avg_pnl = pnl.mean()
    std_pnl = pnl.std()

    span_days = (trades_df["exit_date"].max() - trades_df["entry_date"].min()).days
    trades_per_year = n / (span_days / 365.25) if span_days > 0 else np.nan
    sharpe = (avg_pnl / std_pnl) * np.sqrt(trades_per_year) if std_pnl > 0 else np.nan

    by_spread_bucket = None
    if n >= 6 and trades_df["spread"].nunique() >= 3:
        buckets = pd.qcut(trades_df["spread"].abs(), q=3, duplicates="drop")
        by_spread_bucket = trades_df.groupby(buckets, observed=True)[pnl_col].agg(["count", "mean"])

    return {
        "n_trades": n, "win_rate": win_rate, "avg_pnl": avg_pnl,
        "std_pnl": std_pnl, "sharpe": sharpe, "by_spread_bucket": by_spread_bucket,
    }


def compute_max_drawdown(trades_df, pnl_col="pnl"):
    """
    Peak-to-trough max drawdown of the CUMULATIVE trade P&L, ordered by
    exit_date (audit finding B5 -- no drawdown/risk reporting existed
    before this). Trades are non-overlapping by construction
    (run_backtest never opens a new trade before the prior one's holding
    period ends), so summing realized P&L in exit order is a legitimate
    equity curve, not an approximation.

    Returns a dict: max_drawdown (most negative point, i.e. worst
    peak-to-trough loss), the date range it occurred over, and the full
    running drawdown Series for plotting/inspection. NaN-safe (returns
    0.0 for the drawdown fields, not an error, if empty).
    """
    if trades_df.empty:
        return {"max_drawdown": 0.0, "peak_date": None, "trough_date": None, "drawdown_series": pd.Series(dtype=float)}

    ordered = trades_df.sort_values("exit_date")
    cumulative = ordered[pnl_col].cumsum()
    cumulative.index = ordered["exit_date"].values
    running_max = cumulative.cummax()
    drawdown = cumulative - running_max

    trough_date = drawdown.idxmin()
    max_dd = float(drawdown.loc[trough_date])
    # peak = the running-max value's date at or before the trough (where the
    # equity curve was at its local high before this drawdown started).
    peak_date = cumulative.loc[:trough_date].idxmax()

    return {
        "max_drawdown": max_dd, "peak_date": peak_date, "trough_date": trough_date,
        "drawdown_series": drawdown,
    }


def _zero_net_block(day_returns):
    """
    Replaces the last element of `day_returns` so the block's COMPOUNDED
    return is exactly zero (not just the simple sum) -- gives an exact
    "spot pinned at strike" price path with no residual drift from
    compounding, so a P&L-sign check against it is fully deterministic.
    """
    partial = np.prod([1 + r for r in day_returns[:-1]])
    day_returns[-1] = 1.0 / partial - 1.0
    return day_returns


if __name__ == "__main__":
    # Deterministic mechanics check: construct a return path where every
    # 5-day block (aligned exactly to each trade's holding window) has
    # ZERO compounded return -- spot is pinned at the strike for the
    # whole holding period, every single trade, with no direction/gamma
    # noise to confound sign conventions. The lookback window still sees
    # real (nonzero, non-degenerate) daily variation, so realized
    # correlation is well-defined rather than NaN. A short straddle
    # under this construction must show positive P&L (pure decay
    # capture); a long straddle the exact opposite -- every time.
    dates = pd.bdate_range("2024-01-02", periods=80)
    tickers = ["A", "B"]
    weights = pd.Series({"A": 0.6, "B": 0.4})

    block_A = _zero_net_block([0.02, -0.03, 0.01, 0.015, 0.0])
    block_B = _zero_net_block([-0.01, 0.02, -0.015, 0.005, 0.0])
    n_blocks = len(dates) // 5
    flat_price_returns = pd.DataFrame({
        "A": np.tile(block_A, n_blocks),
        "B": np.tile(block_B, n_blocks),
        "IDX": 0.0,  # index leg pinned flat outright -- simplest case
    }, index=dates)

    # IDX iv set absurdly high/low (not a realistic market input -- this
    # is a mechanics test) so implied corr is unambiguously above/below
    # ANY possible realized corr (which is bounded in [-1, 1] for a
    # 2-name basket), making the resulting signal direction robust
    # without needing to hand-compute the exact realized corr value.
    flat_iv_high_idx = pd.DataFrame({"A": 0.30, "B": 0.25, "IDX": 3.00}, index=dates)
    flat_iv_low_idx = pd.DataFrame({"A": 0.30, "B": 0.25, "IDX": 0.01}, index=dates)

    sell_trades = run_backtest(flat_price_returns, flat_iv_high_idx, weights,
                                index_ticker="IDX", lookback_days=10,
                                holding_days=5, check_every=5)
    print(f"Pinned-spot check (high IDX iv): {len(sell_trades)} trades, "
          f"signals={sell_trades['signal'].unique().tolist()}")
    assert len(sell_trades) > 0, "Expected trades to fire -- check didn't run"
    assert (sell_trades["signal"] == "sell_dispersion").all()
    assert (sell_trades["pnl"] > 0).all(), \
        "Sell dispersion with spot pinned at strike should always collect time decay"
    print("OK: sell-dispersion P&L sign is correct with spot pinned at strike "
          "(pure theta capture, no direction/gamma confound).\n")

    buy_trades = run_backtest(flat_price_returns, flat_iv_low_idx, weights,
                               index_ticker="IDX", lookback_days=10,
                               holding_days=5, check_every=5)
    print(f"Pinned-spot check (low IDX iv): {len(buy_trades)} trades, "
          f"signals={buy_trades['signal'].unique().tolist()}")
    assert len(buy_trades) > 0
    assert (buy_trades["signal"] == "buy_dispersion").all()
    print("OK: buy_dispersion fires correctly when implied corr sits well below "
          "realized (signal-generation wiring is correct).\n")

    # Sign-convention check: on IDENTICAL market data, flipping
    # sell_dispersion <-> buy_dispersion must exactly negate the raw
    # (pre-transaction-cost) P&L. This holds regardless of vol levels --
    # unlike "decay must be positive for sell / negative for buy", which
    # turns out NOT to be universally true (see note below).
    prices = 100 * (1 + flat_price_returns[["A", "B", "IDX"]]).cumprod()
    sample_entry, sample_exit = sell_trades.iloc[0][["entry_date", "exit_date"]]
    w_norm = weights / weights.sum()

    sell_pnl, sell_legs = _simulate_trade(
        "sell_dispersion", sample_entry, sample_exit, prices, flat_iv_high_idx,
        w_norm, tickers, "IDX", option_tenor_days=30, r=0.045,
        n_index_straddles=10.0, cost_bps=5.0,
    )
    buy_pnl, buy_legs = _simulate_trade(
        "buy_dispersion", sample_entry, sample_exit, prices, flat_iv_high_idx,
        w_norm, tickers, "IDX", option_tenor_days=30, r=0.045,
        n_index_straddles=10.0, cost_bps=5.0,
    )
    sell_raw, buy_raw = sell_pnl + sell_legs["costs"], buy_pnl + buy_legs["costs"]
    assert abs(sell_raw + buy_raw) < 1e-8, \
        "Flipping trade direction on identical market data should exactly negate raw P&L"
    assert abs(sell_legs["costs"] - buy_legs["costs"]) < 1e-8, \
        "Transaction costs should be direction-independent (same notional either way)"
    print("OK: sell_dispersion <-> buy_dispersion exactly negates raw P&L on identical "
          "market data (long/short leg sign conventions are internally consistent).\n")

    print(
        "Note: the low-IDX-iv scenario above produced POSITIVE buy_dispersion p&l\n"
        "even with spot pinned flat -- that's real, not a bug. With un-hedged straddles,\n"
        "whichever leg sits at the higher implied vol also carries the bigger dollar theta,\n"
        "and that can dominate pinned-spot P&L regardless of which side is long or short.\n"
        "That scenario used an extreme 0.01 vs 0.30 vol gap to force the signal direction\n"
        "deterministically; real index-vs-component vol levels sit much closer together.\n"
        "Still, it's a genuine reminder that this backtest's P&L (matching optimizer.py's\n"
        "existing non-delta-hedged sizing) reflects theta/vega-level exposure on top of the\n"
        "correlation bet, not a pure correlation view -- worth keeping in mind once real\n"
        "trade stats start coming out of this."
    )
