"""
Data fetching via yfinance.

NOTE: This module requires live internet access to Yahoo Finance and
will NOT run in a sandboxed environment without network access to
finance.yahoo.com. Run this locally.

yfinance doesn't provide official index weights, so this module ships
a hardcoded approximate QQQ top-15 weight table (update periodically --
weights drift as constituent market caps change and QQQ rebalances
quarterly). For production use, pull real weights from Invesco's
official QQQ holdings page/CSV.
"""

from datetime import date

import yfinance as yf
import pandas as pd
import numpy as np

# Approximate QQQ top-15 constituents and weights (~57% of index as of
# 2026-08-13; cross-checked against stockanalysis.com/etf/qqq/holdings
# and etfchannel.com -- Invesco's own page renders holdings via JS and
# can't be scraped directly). UPDATE THIS periodically -- these drift
# with price moves and quarterly rebalances.
#
# Note: SpaceX (SPCX) IPO'd 2026-06-12 and joined the Nasdaq-100 on
# 2026-07-07 at ~1.2% weight (~rank 22) -- not in the top 15 here, but
# if you expand this basket, expect its options chain to be thin/absent
# given how recently it started trading.
QQQ_TOP_CONSTITUENTS = {
    "NVDA": 0.0850,
    "AAPL": 0.0699,
    "MSFT": 0.0575,
    "MU": 0.0461,
    "AMZN": 0.0444,
    "AMD": 0.0339,
    "GOOGL": 0.0314,
    "AVGO": 0.0309,
    "GOOG": 0.0292,
    "META": 0.0275,
    "TSLA": 0.0265,
    "WMT": 0.0243,
    "INTC": 0.0226,
    "CSCO": 0.0192,
    "COST": 0.0184,
}

# Dow Jones Industrial Average -- all 30 members, verified 2026-08-21 against
# multiple current sources (not relied on from training-data memory alone,
# since two real membership changes happened in the interim: NVDA replaced
# INTC and SHW replaced DOW Inc. in Nov 2024; AMZN replaced WBA in Feb 2024).
# Unlike QQQ_TOP_CONSTITUENTS above, this is just the ticker list, no
# hardcoded weights -- the Dow is PRICE-weighted (each member's index weight
# is proportional to its raw share price, not market cap), so weights are
# computed fresh from live prices in get_dia_weights() below rather than
# hardcoded and left to go stale like the QQQ table already is.
DIA_CONSTITUENTS = [
    "AAPL", "MSFT", "NVDA", "CRM", "CSCO", "IBM", "AMZN", "VZ",
    "JPM", "GS", "AXP", "V", "TRV", "UNH", "JNJ", "MRK", "AMGN",
    "BA", "CAT", "HON", "MMM", "WMT", "PG", "KO", "MCD", "NKE",
    "DIS", "HD", "CVX", "SHW",
]


def get_dia_weights(tickers=None):
    """
    Live price-weights for the Dow -- each member's weight is its own
    share price divided by the sum of all members' share prices (the
    divisor that converts summed price into the actual index LEVEL is
    a constant scaling factor and cancels out of relative weights, so
    it's irrelevant here). Computed fresh every call, since price-
    weights genuinely change every time any member's price moves --
    there's no "stale weight table" problem for this index the way
    there is for QQQ_TOP_CONSTITUENTS' market-cap weights.

    Returns a pd.Series indexed by ticker, summing to 1.0.

    NOTE: this is a single LIVE snapshot -- for a historical backtest,
    use build_dia_weights_panel() below instead, which reconstructs
    both membership and weight per historical date rather than applying
    today's roster/weights across the whole backtest (the audit's
    survivorship-bias finding).
    """
    if tickers is None:
        tickers = DIA_CONSTITUENTS
    prices = get_price_history(tickers, period="1d").iloc[-1]
    return prices / prices.sum()


# Point-in-time Dow membership, verified 2026-08-22 against Wikipedia's
# "Historical components of the Dow Jones Industrial Average" -- only two
# real changes fall inside this project's DATA_FLOOR (2021-01-01) window:
#   2024-02-26: AMZN replaced WBA (Walgreens Boots Alliance)
#   2024-11-08: NVDA replaced INTC, SHW replaced DOW Inc. (the chemical co.)
# WBA is UNOBTAINABLE for free: yfinance purged ALL history (not just
# recent) after WBA's 2025-08-28 going-private delisting (Sycamore
# Partners), and Nasdaq's own historical-quote API has no record of the
# symbol either -- confirmed by direct live query, not assumed. Era 1
# below therefore runs 29 of 30 real historical members, WBA excluded
# rather than silently substituted with something else or left
# unlabeled -- its weight is redistributed across the other 29 via the
# price-weight formula, the same honest-gap handling used everywhere
# else in this project (skip and disclose, don't fabricate).
DIA_ERAS = [
    (date(2021, 1, 1), date(2024, 2, 26), [  # Era 1: pre-AMZN (WBA gap)
        "AAPL", "MSFT", "INTC", "CRM", "CSCO", "IBM", "DOW", "VZ",
        "JPM", "GS", "AXP", "V", "TRV", "UNH", "JNJ", "MRK", "AMGN",
        "BA", "CAT", "HON", "MMM", "WMT", "PG", "KO", "MCD", "NKE",
        "DIS", "HD", "CVX",
    ]),
    (date(2024, 2, 26), date(2024, 11, 8), [  # Era 2: AMZN in
        "AAPL", "MSFT", "INTC", "CRM", "CSCO", "IBM", "DOW", "AMZN", "VZ",
        "JPM", "GS", "AXP", "V", "TRV", "UNH", "JNJ", "MRK", "AMGN",
        "BA", "CAT", "HON", "MMM", "WMT", "PG", "KO", "MCD", "NKE",
        "DIS", "HD", "CVX",
    ]),
    (date(2024, 11, 8), date(2100, 1, 1), DIA_CONSTITUENTS),  # Era 3: current
]


def dia_roster_asof(d):
    """Real Dow 30 membership for calendar date `d` (a datetime.date)."""
    for start, end, roster in DIA_ERAS:
        if start <= d < end:
            return roster
    raise ValueError(f"No DIA_ERAS entry covers {d} -- extend the table.")


def build_dia_weights_panel(dates, price_panel=None):
    """
    Real point-in-time Dow price-weights, one row per date in `dates`.
    Fixes BOTH halves of the survivorship-bias audit finding at once:
    membership (which tickers are even eligible, via dia_roster_asof)
    and weight (each eligible ticker's real price-weight AS OF that
    date, not today's price applied retroactively).

    Returns a DataFrame (index=dates, columns=union of every DIA_ERAS
    ticker), values are weight-or-NaN. NaN means "not a Dow member on
    this date" -- an intentional zero, not a data gap -- so callers
    (run_backtest's weights_panel path) should treat NaN here as
    "exclude from this date's basket", not "skip this date entirely".

    price_panel: optional pre-fetched get_real_spot_history() output
    (real, non-dividend-adjusted close -- see thetadata_fetch.py's
    "SPOT SOURCE FIX") covering the union of all era tickers. Pass this
    in when calling repeatedly to avoid re-pulling from yfinance each time.
    """
    if price_panel is None:
        all_era_tickers = sorted(set(t for _, _, roster in DIA_ERAS for t in roster))
        price_panel = get_real_spot_history(all_era_tickers, period="10y")
        price_panel.index = pd.DatetimeIndex(price_panel.index).tz_localize(None)

    weights = pd.DataFrame(index=dates, columns=price_panel.columns, dtype=float)
    price_dates = price_panel.index
    for d in dates:
        d_date = d.date() if hasattr(d, "date") else d
        roster = dia_roster_asof(d_date)
        available = price_dates[price_dates.date <= d_date]
        if len(available) == 0:
            continue
        px = price_panel.loc[available[-1], roster].dropna()
        if px.empty:
            continue
        weights.loc[d, px.index] = (px / px.sum()).values
    return weights


def build_qqq_weights_panel(dates, tickers=None):
    """
    Market-cap-PROXY point-in-time weights for QQQ_TOP_CONSTITUENTS.

    Deliberate, disclosed SCOPE LIMIT (see the audit's A1/A3 findings and
    the decision to keep QQQ pragmatic rather than expand to the full
    ~100-name Nasdaq-100): membership is held fixed at today's top-15 --
    this does NOT answer "were these actually the top-15 names on a given
    historical date" (they weren't; e.g. NVDA's real Nasdaq-100 weight in
    2021 was far below its ~8.5% weight today). It DOES fix the narrower
    "given these 15 names, what was each one's real relative weight on
    this date" question, replacing the single static 2026-08-13 snapshot
    that get_price_history()-based backtests used before.

    No official historical Nasdaq-100 weight file is available for free,
    so real market cap (real historical split-adjusted price x real
    historical shares outstanding, both from yfinance -- confirmed via
    Ticker.get_shares_full(), which correctly reflects e.g. NVDA's 2024
    10:1 split via the raw share count, not just the price) stands in as
    the proxy. Nasdaq-100 is actually MODIFIED-cap-weighted (per-name
    caps, periodic special rebalances) -- this proxy doesn't replicate
    those mechanics, just plain market-cap share.

    Returns a DataFrame (index=dates, columns=tickers), weights summing
    to 1.0 across the FIXED 15-name universe on every date (no NaN --
    membership doesn't vary in this pragmatic version, only weight does).
    """
    if tickers is None:
        tickers = list(QQQ_TOP_CONSTITUENTS.keys())

    price_panel = get_real_spot_history(tickers, period="10y")
    price_panel.index = pd.DatetimeIndex(price_panel.index).tz_localize(None)

    # META's get_shares_full() only goes back to 2022-06-09 under the
    # renamed ticker -- confirmed live: nothing earlier is returned, not a
    # date-range argument issue. The pre-rename "FB" ticker DOES have
    # shares history back further (confirmed live, 58 rows from 2020-07),
    # so pull under the old symbol and treat it as META's own history for
    # this purpose -- same underlying company/share count, just filed
    # under the old ticker before the Oct 2021 rename took effect in
    # yfinance's own records.
    SHARES_TICKER_OVERRIDE = {"META": "FB"}

    shares_panel = {}
    for t in tickers:
        shares_ticker = SHARES_TICKER_OVERRIDE.get(t, t)
        s = yf.Ticker(shares_ticker).get_shares_full(start="2020-06-01")
        if s is None or len(s) == 0:
            shares_panel[t] = None
            continue
        s.index = pd.DatetimeIndex(s.index).tz_localize(None).normalize()
        s = s[~s.index.duplicated(keep="last")].sort_index()
        # get_shares_full() gives the RAW share count as actually reported
        # on that date -- NOT adjusted for splits that happen later. This
        # USED to need scaling up by later split ratios, because
        # get_real_spot_history()'s price used to always come back
        # split-adjusted (matching post-split terms even for old dates),
        # so raw pre-split shares had to be scaled up to match. That's no
        # longer true: get_real_spot_history() was fixed on 2026-08-25 to
        # recover REAL, un-adjusted historical prices (see its own
        # docstring's "SPLIT FIX"), which are already on the SAME
        # contemporaneous basis as get_shares_full()'s raw share counts --
        # real price x real shares outstanding, no adjustment needed on
        # either side. Leaving the old scaling in after that fix was a
        # real, separate bug: it double-corrected, overstating market cap
        # (and weight) by the split ratio for every date before a split --
        # caught directly via NVDA showing an implausible ~58% weight of
        # this 15-name mega-cap basket right before its 2024-06-10 split
        # (larger than AAPL+MSFT individually), collapsing to a much more
        # plausible ~11-15% immediately after, for a ticker whose real
        # market cap didn't actually change across its own split.
        shares_panel[t] = s

    weights = pd.DataFrame(index=dates, columns=tickers, dtype=float)
    price_dates = price_panel.index
    for d in dates:
        d_date = d.date() if hasattr(d, "date") else d
        available_px = price_dates[price_dates.date <= d_date]
        if len(available_px) == 0:
            continue
        px_date = available_px[-1]
        mcaps = {}
        for t in tickers:
            px = price_panel.loc[px_date, t]
            s = shares_panel[t]
            if pd.isna(px) or s is None or s.empty:
                continue
            prior_shares = s[s.index.date <= d_date]
            if prior_shares.empty:
                continue
            mcaps[t] = float(px) * float(prior_shares.iloc[-1])
        if not mcaps:
            continue
        mcap_series = pd.Series(mcaps)
        weights.loc[d, mcap_series.index] = (mcap_series / mcap_series.sum()).values
    return weights


def get_price_history(tickers, period="6mo", interval="1d"):
    """
    Fetch adjusted close prices for a list of tickers.
    Returns a DataFrame: index=date, columns=tickers.
    """
    data = yf.download(tickers, period=period, interval=interval,
                        auto_adjust=True, progress=False)["Close"]
    if isinstance(data, pd.Series):
        data = data.to_frame(name=tickers[0])
    return data


def get_real_spot_history(tickers, period="10y", interval="1d"):
    """
    Real (dividend-UNADJUSTED, split-UNADJUSTED) close prices -- for
    matching against option strikes / spot-in-IV-solving, NOT for returns.
    yfinance's default auto_adjust=True backs dividends out of historical
    closes, understating real historical spot (worse for high-dividend
    names, worse further back) -- confirmed via VZ: real 2021-01-04 close
    was $58.85, adjusted was $41.53. That mismatch was silently corrupting
    strike selection and IV solving throughout the ThetaData basket-IV
    pipeline (thetadata_fetch.py).

    Originally this was going to move to ThetaData's own stock_history_eod,
    reasoning that same-vendor spot/options data must be self-consistent --
    but that endpoint requires a separate Stocks-feed Value subscription
    ($40/mo on top of the Options Value tier already in use), which turned
    out unnecessary: yfinance's real (auto_adjust=False) close is free and
    matches actual market prints, so it works fine as the spot source too.
    Keep get_price_history() (auto_adjust=True) for get_daily_returns() --
    dividend reinvestment IS the correct convention for realized return/
    correlation calculations, just wrong for spot-matching.

    SPLIT FIX (found 2026-08-25, auditing why WMT/NVDA/AVGO/AMZN/GOOGL/
    GOOG/TSLA had a hard IV-coverage cliff landing exactly on each name's
    real historical stock-split date): auto_adjust=False only suppresses
    DIVIDEND back-adjustment -- yfinance's "Close" column is ALWAYS
    split-back-adjusted regardless of that flag. Confirmed directly: WMT's
    2022-01-04 "real" close came back $47.33 (its post-2024-split-adjusted
    value); the real unadjusted price WMT traded at that day was ~$142. For
    any date before a ticker's split, this silently understated spot by
    the split ratio, so pull_ticker_atm_iv_history's +/-25% strike band
    (centered on this wrong, too-low spot) missed every real strike that
    ticker actually traded at pre-split -- total data loss for the whole
    pre-split period, not scattered gaps, since MAX_MONEYNESS_PCT rejected
    the mismatch too. Fixed by pulling each ticker's real split events
    (yf.Ticker(t).splits) and multiplying pre-split-date closes by the
    split ratio to recover the true historical unadjusted price -- e.g.
    WMT's real 3:1 ratio turns that $47.33 back into ~$142, matching what
    it actually traded at.
    """
    data = yf.download(tickers, period=period, interval=interval,
                        auto_adjust=False, progress=False)["Close"]
    if isinstance(data, pd.Series):
        data = data.to_frame(name=tickers[0])
    if data.index.tz is not None:
        data.index = data.index.tz_localize(None)

    for t in data.columns:
        try:
            splits = yf.Ticker(t).splits
        except Exception:
            continue
        if splits.empty:
            continue
        splits.index = pd.DatetimeIndex(splits.index).tz_localize(None)
        for split_date, ratio in splits.items():
            pre_split = data.index < split_date
            data.loc[pre_split, t] = data.loc[pre_split, t] * ratio

    return data


def get_daily_returns(tickers, period="6mo"):
    """Log returns from price history -- feed directly into realized_correlation."""
    prices = get_price_history(tickers, period=period)
    return np.log(prices / prices.shift(1)).dropna()


def get_atm_iv(ticker, target_dte_days=30, r=0.045):
    """
    Pull the options chain for `ticker`, find the expiration closest to
    target_dte_days, and back out ATM implied vol from the call closest
    to the current spot price using iv_solver.implied_vol.

    Returns: (iv, actual_dte_days, expiration_date_str)
    """
    import sys, os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from iv_solver import implied_vol

    tk = yf.Ticker(ticker)
    spot = tk.history(period="1d")["Close"].iloc[-1]

    expirations = tk.options
    if not expirations:
        return np.nan, None, None

    today = pd.Timestamp.today()
    dtes = [(pd.Timestamp(e) - today).days for e in expirations]
    best_idx = int(np.argmin([abs(d - target_dte_days) for d in dtes]))
    expiry = expirations[best_idx]
    actual_dte = dtes[best_idx]
    T = actual_dte / 365

    chain = tk.option_chain(expiry)
    calls = chain.calls
    calls["dist"] = (calls["strike"] - spot).abs()
    atm_row = calls.sort_values("dist").iloc[0]

    mid_price = (atm_row["bid"] + atm_row["ask"]) / 2
    if mid_price <= 0:
        mid_price = atm_row["lastPrice"]

    iv = implied_vol(mid_price, spot, atm_row["strike"], T, r, "call")
    return iv, actual_dte, expiry


if __name__ == "__main__":
    print("This module requires live network access -- run locally, not in sandbox.")
    print(f"Configured basket: {list(QQQ_TOP_CONSTITUENTS.keys())}")
    print(f"Total basket weight (renormalize before use): {sum(QQQ_TOP_CONSTITUENTS.values()):.2%}")
