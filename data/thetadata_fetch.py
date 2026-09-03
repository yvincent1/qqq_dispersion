"""
Real historical ATM implied vol, per name, from ThetaData's free tier --
the replacement path for proxy_iv.py's abandoned VXN-multiplier proxy
(see that module's docstring and README.md's "Backtest engine status"
section for why it was abandoned: v1 canceled out inside
implied_correlation() entirely, v2 fixed that but introduced look-ahead
bias by calibrating from today's IV and projecting it backward across
years of history).

This module pulls REAL historical option prices and solves IV from them
via iv_solver.implied_vol -- no calibration, no projection backward, no
look-ahead risk of that specific kind.

CORRECTIONS found against the REAL installed `thetadata` library (v1 of
this file was written from ThetaData's docs, which turned out to
disagree with the actual package in several ways -- each found by
actually running this against a live account, not by reading docs
further):
  1. The parameter is `expiration`, not `expiration_date`, on both
     option_list_strikes and option_history_eod -- the docs' example
     was simply wrong. This was the TypeError the first real run hit.
  2. There is NO `date` column in the response at all -- the trading
     date is derived from `created`, the EOD record's generation
     timestamp (NY-time, seen at ~1h20m after market close in the
     sample pulled): `pd.to_datetime(df["created"]).dt.date`.
  3. `right` values are the full words 'CALL'/'PUT', not 'C'/'P'. This
     module requests right='both' and filters locally rather than
     passing right='C' as the request param, since only 'both' was
     actually confirmed to work as an input value.
  4. ThetaData caps each request to a 365-day span -- confirmed via a
     real "Too many days between start and end date" error once a
     fixed start_date (from the full spot-history window) grew past
     that cap for later expirations. Each request is now bounded to
     the 45 days before that expiration (tightened from an initial 90
     after a real accuracy check found single-name gaps vs. a live
     pull -- see pull_ticker_atm_iv_history's docstring for the full
     finding, including a correction to how ref_spot actually works).
  5. strike='*' (wildcard, whole chain in one call) SEEMED like a clean
     efficiency win -- fewer calls, and per-day-accurate ATM selection
     instead of one fixed strike for the whole window -- but it
     consistently TIMED OUT for NVDA, every single expiration, not
     intermittently. NVDA's chain is far larger than QQQ's (the ticker
     the wildcard approach was validated against) -- many more strikes,
     heavy retail volume -- and streaming the whole chain across a
     90-day window was apparently too large a response for a name like
     that. Reverted to requesting ONE specific strike (nearest to spot
     at the start of each expiration's window, right='both' still, but
     now only ~2 rows/day instead of the whole chain) -- small,
     reliable requests, at the cost of that strike not adapting
     per-day the way the wildcard version did. Reliability mattered
     more than that precision once the wildcard approach proved it
     couldn't complete for several of the basket's busier names.
Real columns seen: symbol, expiration, strike, right, created, last_trade,
open, high, low, close, volume, count, bid_size, bid_exchange, bid,
bid_condition, ask_size, ask_exchange, ask, ask_condition.

Also has NO built-in call timeout anywhere in the library (checked its
constructor and option_history_eod's source directly) -- a real
indefinite hang was observed in production. _call_with_timeout below
enforces one so a stalled request costs CALL_TIMEOUT_SEC, not forever.

DESIGN DECISION, same reasoning as the skew project's puller: uses only
monthly (3rd-Friday) expirations as the ~30-DTE rolling target rather
than the true nearest-to-30-DTE expiration get_atm_iv() picks live --
keeps API call volume tractable against the free tier's 20 req/min
limit and 1-year history window: ~16 names x ~12-13 monthly expirations
x 2 calls each (list_strikes + history_eod) = ~400 calls total.

SPOT SOURCE FIX (found while investigating why DIA's 5.6-year pull showed
21 of 31 tickers with severe implied/realized IV ratio outliers, some
extreme -- VZ at 11.9x, MMM at 10.2x): spot was being pulled from
data/fetch.py's get_price_history(), which uses yfinance with
auto_adjust=True. That returns DIVIDEND-ADJUSTED historical prices --
correct for computing true total returns, but NOT the real price a
stock actually traded at on a historical date. ThetaData's option
quotes are priced against the REAL, unadjusted spot. Confirmed directly:
VZ's real close on 2021-01-04 was $58.85; yfinance's adjusted series
said $41.53 -- a 30% understatement, growing further back in time as
more dividends accumulate. Using the understated adjusted price as
"spot" made high-dividend names' strikes look artificially deep ITM,
triggering the same low-vega/unstable-IV mechanism diagnosed earlier
for CSCO -- except systematically, across every dividend payer, worse
for higher yields and older dates. First fix attempt pulled spot from
ThetaData's own stock_history_eod() instead (same vendor/feed as the
options data, so consistency-by-construction) -- but that endpoint
requires a separate Stocks-feed Value subscription, confirmed live via
a clean PERMISSION_DENIED ("requiring a VALUE subscription, but you
only have a FREE subscription") on every ticker, distinct from the
Options-feed Value tier already in use. Landed instead on
data/fetch.py's get_real_spot_history(), which is just yfinance with
auto_adjust=False -- free, and confirmed to return the correct real
close (VZ 2021-01-04 -> $58.85, matching ThetaData's own quotes) rather
than the dividend-adjusted one. get_daily_returns() (used separately
for realized correlation in the backtest, not here) correctly keeps
using get_price_history()'s adjusted prices -- dividend-adjusted total
returns are the right input for that use case; only the
spot-for-strike-matching role was wrong.

Reuses implied_vol (iv_solver.py) for the actual math, and the shared
common/thetadata_client.py + common/option_calendar.py rather than
duplicating either across projects.
"""

import os
import sys
import time
import threading
import concurrent.futures  # only for the TimeoutError type -- see _call_with_timeout
from datetime import date

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "common"))
from iv_solver import implied_vol
from thetadata_client import get_client
from option_calendar import third_fridays

from data.fetch import QQQ_TOP_CONSTITUENTS, get_real_spot_history

RISK_FREE_RATE = 0.045
DATA_FLOOR = date(2021, 1, 1)  # Value tier's confirmed history floor (upgraded from free
                                # tier's 2023-06-01 -- confirmed directly from ThetaData's own
                                # subscription table, not the free-tier docs page this constant
                                # used to be sourced from). Was named FREE_TIER_FLOOR; renamed
                                # since it's now tier-dependent, not free-tier-specific.
RATE_LIMIT_SLEEP_SEC = 0.1   # Value tier is concurrency-based (2 concurrent, confirmed from
                              # ThetaData's own table), not rate-limited per minute the way free
                              # tier was -- the old 3.1s sleep was sized for free tier's ~20-30
                              # req/min cap, which no longer applies. Keeping a small nonzero
                              # sleep as a politeness margin rather than hammering the API with
                              # zero gap between sequential calls, not because a rate limit
                              # requires it.
CALL_TIMEOUT_SEC = 30                # see _call_with_timeout -- a real indefinite hang was
                                      # observed in production; ThetaClient exposes no deadline
                                      # anywhere (checked its __init__ and option_history_eod's
                                      # source directly), so one is enforced here instead.
MIN_DTE = 5          # reject any daily reading this close to expiration -- found via a
                      # real rollover-day bug: when the ~30-DTE expiration's data was
                      # missing for a specific day (e.g. a newly-relevant strike hadn't
                      # traded yet), the dedup-by-closest-to-30-DTE logic fell back to
                      # whatever WAS available, including a 1-DTE contract with a 51%
                      # bid-ask spread -- producing a nonsensical 372% IV that got
                      # accepted as valid data. There was no floor rejecting that
                      # fallback; this is it.
MAX_SPREAD_PCT = 0.30  # reject quotes with bid-ask spread this wide relative to mid --
                        # a general illiquidity guard, independent of MIN_DTE (the same
                        # rollover-day quote that triggered this fix also had a 51%
                        # spread, so this catches the same failure mode a second way,
                        # and would also catch illiquid-but-not-near-expiry quotes).
MAX_MONEYNESS_PCT = 0.20  # reject a daily reading if the fixed strike has drifted more
                           # than this far from THAT DAY's actual spot. Found via a real
                           # 5.6-year pull: CSCO alone had 170 nonsensical IV readings
                           # (up to 369%), heavily concentrated in 2021-2022 (the big
                           # tech/semis drawdown), tapering to ~0 by 2025-2026 -- a
                           # DIFFERENT failure mode from the rollover bug above. During a
                           # genuinely volatile stretch, the underlying can drift far
                           # enough from where the strike was fixed that the option ends
                           # up deep ITM/OTM. Deep ITM/OTM options have very low vega, so
                           # a perfectly normal, tight-spread quote (passes MAX_SPREAD_PCT
                           # fine) still inverts to a wildly unstable IV via Black-Scholes,
                           # because price barely moves with vol out there. Neither MIN_DTE
                           # nor MAX_SPREAD_PCT catches this -- it's a moneyness problem,
                           # not a time-to-expiry or liquidity problem.
STRIKE_BAND_PCT = 0.25  # width of the pre-fetched strike band around each expiration
                         # window's START-of-window spot (see pull_ticker_atm_iv_history's
                         # "BANDED PER-DAY" section -- the audit's look-ahead-in-strike-
                         # selection fix). Deliberately wider than MAX_MONEYNESS_PCT (0.20)
                         # so the band itself is never the binding constraint on a real
                         # move -- MAX_MONEYNESS_PCT should always be what rejects a day
                         # that's drifted too far, not "ran out of pre-fetched strikes".


def _eod_to_pandas(eod):
    return eod.to_pandas() if hasattr(eod, "to_pandas") else pd.DataFrame(eod)


def _call_with_timeout(fn, *args, timeout_sec=CALL_TIMEOUT_SEC, **kwargs):
    """
    Runs fn in a helper thread and enforces timeout_sec regardless of what
    the call is blocked on (needed because ThetaClient's gRPC calls have no
    built-in deadline -- confirmed by reading the library's source, and by
    a real hang that sat well past 2-3 minutes with zero response during
    testing).

    v2: a first version used `with ThreadPoolExecutor() as executor:` --
    that turned out to NOT actually fix the hang. The `with` block's exit
    calls executor.shutdown(wait=True), which blocks until the submitted
    call finishes even after future.result(timeout=...) has already raised
    TimeoutError -- confirmed live: the real script hung a SECOND time,
    inside that shutdown call, after already having correctly raised and
    printed a TIMEOUT message. This version uses a raw daemon thread
    instead of ThreadPoolExecutor entirely: daemon threads are never
    joined at interpreter exit (confirmed by testing -- even
    shutdown(wait=False) doesn't avoid this, because concurrent.futures
    registers an atexit hook that joins EVERY thread any ThreadPoolExecutor
    ever created, regardless of that flag), so a call that never returns
    can't block either this function OR the script's eventual exit.
    """
    result = {}

    def _target():
        try:
            result["value"] = fn(*args, **kwargs)
        except Exception as e:
            result["error"] = e

    t = threading.Thread(target=_target, daemon=True)
    t.start()
    t.join(timeout=timeout_sec)
    if t.is_alive():
        raise concurrent.futures.TimeoutError(f"Call did not complete within {timeout_sec}s")
    if "error" in result:
        raise result["error"]
    return result.get("value")


def pull_ticker_atm_iv_history(client, ticker, spot_series, expirations):
    """
    For ONE ticker, across the given monthly expirations, pull a BAND of
    strikes' EOD history and pick, per trading day, whichever strike in
    that band is nearest to THAT DAY's real spot -- deduped across
    expirations by whichever reading is closest to 30 DTE on any given
    date.

    v3, BANDED PER-DAY SELECTION -- fixes a real look-ahead bug in v2
    (single fixed strike per expiration, picked from spot AS OF THE
    EXPIRATION DATE and reused for the whole ~45-day window leading up
    to it): a trading-desk audit of this project flagged that a reading
    taken e.g. 40 days before expiration was being priced off a strike
    chosen using where the stock ended up 40 days LATER -- the IV series
    wasn't a clean point-in-time observable. v2's own docstring already
    knew a single fixed strike was imprecise (misread skew once the name
    moved), but framed it as a tolerable precision limit rather than the
    look-ahead problem it actually was.

    Real per-day strike selection (one dedicated API call per ticker per
    trading day) was ruled out as too expensive -- 10-20x the call volume
    of v2, likely many hours per basket. This version is the middle
    ground: pre-fetch EOD history for a BAND of strikes (STRIKE_BAND_PCT
    wide) around the window's START-of-window spot -- not the expiration-
    date spot, which is what actually removes the look-ahead, since only
    information available at the earliest point in the window is used to
    decide what to fetch. Then, independently for each trading day,
    whichever band strike is nearest to THAT DAY's own real spot is
    selected -- a real per-day ATM choice, just constrained to strikes
    that were already pulled. A day whose real spot drifts outside the
    band still gets caught by the existing MAX_MONEYNESS_PCT guard below,
    same as any other real coverage gap in this project (skip, don't
    guess) -- STRIKE_BAND_PCT is deliberately wider than
    MAX_MONEYNESS_PCT so that guard, not an accidentally-narrow band, is
    what actually decides whether a drifted day gets rejected.

    v2's other reason for reverting away from a strike='*' full-chain
    wildcard (timed out on NVDA -- see prior docstring, preserved in git
    history) still applies: this version pulls a bounded BAND, not the
    full chain, so it doesn't reintroduce that specific timeout.
    """
    all_daily = []
    for exp in expirations:
        end_date = min(exp, spot_series.index.max().date())
        # ThetaData caps each request to a 365-day span (hit as a real error:
        # "Too many days between start and end date; max 365 days allowed").
        # A fixed start_date from the full spot-history window grows past that
        # cap for later expirations while end_date keeps advancing. Bounding
        # to 45 days comfortably covers the ~30-day gap between consecutive
        # monthly expirations rolling over.
        start_date = max(DATA_FLOOR, spot_series.index.min().date(), end_date - pd.Timedelta(days=45))
        if start_date >= end_date:
            continue

        # Band center = spot at the START of the window -- the only spot
        # value legitimately "known" before any of this window's trading
        # days have happened. This (not the old expiration-date ref_spot)
        # is what removes the look-ahead.
        window_spot_start = spot_series[spot_series.index.date <= start_date]
        if window_spot_start.empty:
            continue
        band_center = float(window_spot_start.iloc[-1])

        try:
            strikes_raw = _call_with_timeout(
                client.option_list_strikes, symbol=ticker, expiration=exp,
            )
        except Exception as e:
            print(f"    ERROR listing strikes for {ticker} {exp}: {e} -- skipping this expiration")
            continue

        # option_list_strikes uses the same _convert_response_stream path as
        # option_history_eod (confirmed by reading both methods' source) --
        # it returns a DataFrame (polars by default), not a plain list. The
        # first real run treated it as a list directly (`if not strikes`,
        # `min(strikes, key=...)`), which crashed EVERY ticker immediately
        # with a polars "truth value of a DataFrame is ambiguous" error.
        strikes_df = _eod_to_pandas(strikes_raw)
        if strikes_df.empty:
            continue
        strike_col = "strike" if "strike" in strikes_df.columns else strikes_df.columns[0]
        all_strikes = strikes_df[strike_col].tolist()
        if not all_strikes:
            continue

        lo, hi = band_center * (1 - STRIKE_BAND_PCT), band_center * (1 + STRIKE_BAND_PCT)
        # ThetaData decodes prices server-side as integer * scale-factor (confirmed by
        # reading the library's response-parsing source), which produces float noise
        # beyond 3 decimal places (e.g. 415.78 -> 415.78000000000003). A real run hit
        # this exact case: the API rejected the noisy value with "at most 3 decimal
        # places" once it round-tripped back through str(). Real strikes are always
        # cent-level at most, so rounding to 2 decimals is always safe, not lossy.
        band_strikes = sorted({round(s, 2) for s in all_strikes if lo <= s <= hi})
        if not band_strikes:
            # Band came up empty (very wide real strike spacing, or a band_center
            # far from any listed strike) -- fall back to just the single
            # nearest strike rather than skipping the expiration entirely.
            band_strikes = [round(min(all_strikes, key=lambda s: abs(s - band_center)), 2)]

        band_rows = []
        for strike in band_strikes:
            try:
                # strike must be a string -- confirmed via the protobuf ContractSpec
                # descriptor directly (TYPE_STRING, code 9). The library's own
                # option_history_eod does str(strike) only for its internal logging
                # metadata, then passes the RAW strike argument into the protobuf
                # message -- it does not convert on the caller's behalf.
                eod = _call_with_timeout(
                    client.option_history_eod,
                    symbol=ticker, expiration=exp, strike=str(strike), right="both",
                    start_date=start_date, end_date=end_date,
                )
            except concurrent.futures.TimeoutError:
                print(f"    TIMEOUT pulling {ticker} {exp} strike {strike} after "
                      f"{CALL_TIMEOUT_SEC}s -- skipping this strike")
                time.sleep(RATE_LIMIT_SLEEP_SEC)
                continue
            except Exception as e:
                print(f"    ERROR pulling {ticker} {exp} strike {strike}: {e} -- skipping this strike")
                time.sleep(RATE_LIMIT_SLEEP_SEC)
                continue
            time.sleep(RATE_LIMIT_SLEEP_SEC)
            df = _eod_to_pandas(eod)
            if df.empty:
                continue
            df = df[df["right"] == "CALL"]
            if df.empty:
                continue
            # No `date` column in the real response (confirmed via a live validation
            # run) -- derive the trading date from `created`, the EOD record's
            # generation timestamp (NY-time, ~1h20m after close in the sample seen).
            df["date"] = pd.to_datetime(df["created"]).dt.date
            df["strike"] = strike
            band_rows.append(df)

        if not band_rows:
            continue
        band_df = pd.concat(band_rows, ignore_index=True)

        # Per-day selection: among whichever band strikes actually quoted on
        # this specific date, pick whichever is nearest to THAT DAY's real
        # spot. Both the band (fixed using only start-of-window info) and
        # the day's own spot are legitimately known as of that day -- no
        # look-ahead.
        for d, day_rows in band_df.groupby("date"):
            spot_matches = spot_series[spot_series.index.date == d]
            if spot_matches.empty:
                continue
            spot = float(spot_matches.iloc[0])
            day_rows = day_rows.copy()
            day_rows["dist"] = (day_rows["strike"] - spot).abs()
            best = day_rows.sort_values("dist").iloc[0]
            chosen_strike = best["strike"]

            if abs(chosen_strike - spot) / spot > MAX_MONEYNESS_PCT:
                continue
            dte = (exp - d).days
            if dte < MIN_DTE:
                continue
            T = dte / 365

            bid, ask = best["bid"], best["ask"]
            if pd.isna(bid) or pd.isna(ask) or bid <= 0 or ask <= 0:
                continue
            mid = (bid + ask) / 2
            if (ask - bid) / mid > MAX_SPREAD_PCT:
                continue
            iv = implied_vol(mid, spot, chosen_strike, T, RISK_FREE_RATE, "call")
            if np.isnan(iv):
                continue
            all_daily.append({"date": d, "expiration": exp, "dte": dte, "iv": iv})

    if not all_daily:
        return pd.Series(dtype=float, name=ticker)

    combined = pd.DataFrame(all_daily)
    combined["dte_dist"] = (combined["dte"] - 30).abs()
    combined = combined.sort_values("dte_dist").drop_duplicates("date").sort_values("date")
    return combined.set_index("date")["iv"].rename(ticker)


def build_basket_iv_panel(start, end, tickers=None, index_ticker="QQQ", checkpoint_path=None):
    """
    Real historical ATM IV panel for an index + its basket -- drop-in
    replacement for proxy_iv.py's build_proxy_iv_panel() output shape
    (index=date, columns=tickers incl. index_ticker), so backtest.py's
    run_backtest() doesn't need to change to use this instead of the
    abandoned proxy. Originally QQQ-only (index_ticker hardcoded); now
    parameterized so the same function serves any basket -- e.g. DIA's
    30 Dow constituents, added after the QQQ pull was already working.

    Each ticker is wrapped in its own try/except and, if checkpoint_path
    is given, the panel-so-far is re-saved after every ticker -- a real
    run previously lost ALL progress (multiple tickers already pulled)
    to a single mid-run error on one ticker. One bad ticker now costs
    just that ticker, not the whole run, and a crash anywhere still
    leaves the completed tickers on disk.

    RESUMES from checkpoint_path if it already exists: any ticker with a
    non-empty column there is reused as-is rather than re-pulled -- a
    real run was intentionally Ctrl+C'd partway through once already
    (NVDA/AAPL had completed), and re-pulling already-good data on every
    retry would waste real time for no reason.
    """
    if tickers is None:
        tickers = list(QQQ_TOP_CONSTITUENTS.keys())
    all_tickers = tickers + [index_ticker]

    existing = pd.DataFrame()
    if checkpoint_path and os.path.exists(checkpoint_path):
        existing = pd.read_csv(checkpoint_path, index_col=0, parse_dates=True)
        # pd.read_csv(parse_dates=True) gives a Timestamp index (e.g. midnight-
        # normalized datetime64), but pull_ticker_atm_iv_history's fresh output
        # is indexed by plain datetime.date objects. Mixing the two in the same
        # pd.concat(axis=1) call does NOT align them as the same calendar day --
        # confirmed via a real corrupted run: it silently created a separate
        # near-duplicate row per date (one Timestamp-formatted, one plain-
        # formatted), which doubled the resumed tickers' apparent observation
        # counts and collapsed "rows where every ticker has data" from ~580 to
        # 110. Normalizing to plain date here, matching fresh pulls exactly,
        # is what actually fixes the alignment.
        existing.index = existing.index.date
        done = [c for c in existing.columns if c in all_tickers and existing[c].notna().any()]
        if done:
            print(f"Resuming from checkpoint -- already have data for: {done}")

    client = get_client()
    expirations = third_fridays(start, end)
    print(f"Pulling {len(expirations)} monthly expirations for {len(all_tickers)} tickers: {all_tickers}")

    # Seed `results` with every already-good column from the checkpoint UP
    # FRONT, keyed by ticker rather than a plain list built up as the loop
    # walks each ticker in order -- a real run was killed (by the sandbox's
    # background-task lifetime cap) right after completing only the FIRST
    # ticker (NVDA), and the old version only ever wrote whatever had been
    # appended to that list SO FAR, which silently overwrote every
    # not-yet-reached ticker's good data still sitting in `existing` on disk
    # (the checkpoint save fires unconditionally after each ticker, and
    # to_csv replaces the whole file). Confirmed live: the checkpoint
    # collapsed from 10 good tickers to just NVDA's column, discarding 9
    # tickers' real pulled data with no way to recover it except re-pulling
    # from scratch. Keying by ticker and seeding from `existing` before the
    # loop starts means even a checkpoint save after the very first ticker
    # still includes every other already-good column, regardless of
    # iteration order or how early a kill lands.
    results = {c: existing[c] for c in existing.columns if c in all_tickers and existing[c].notna().any()}

    for ticker in all_tickers:
        if ticker in results:
            print(f"  {ticker} ... skipped, already in checkpoint ({results[ticker].notna().sum()} observations)")
            continue

        print(f"  {ticker} ...")
        try:
            # get_real_spot_history = yfinance with auto_adjust=False (real,
            # non-dividend-adjusted close) -- see the module docstring's "SPOT
            # SOURCE FIX" section. get_price_history's default auto_adjust=True
            # dividend-adjusts historical closes, understating real historical
            # spot (worse for high-dividend names, worse further back), which
            # corrupted strike selection and IV solving for every ticker here.
            spot_series = get_real_spot_history([ticker], period="10y")[ticker]
            spot_series.index = pd.DatetimeIndex(spot_series.index).tz_localize(None)
            s = pull_ticker_atm_iv_history(client, ticker, spot_series, expirations)
            print(f"    {len(s)} daily IV observations")
        except Exception as e:
            print(f"    ERROR on {ticker}: {e} -- skipping this ticker entirely, continuing with the rest")
            s = pd.Series(dtype=float, name=ticker)
        results[ticker] = s

        if checkpoint_path:
            # sort_index() -- pd.concat(axis=1) on per-ticker Series with
            # different date ranges does NOT sort the resulting index (default
            # sort=False), so without this the saved CSV's row order is
            # scrambled (confirmed on a real run: last rows were 2022 dates,
            # not the true most-recent date). Doesn't affect run_backtest()
            # correctness (it only ever does label-based iv_panel.loc[date]
            # lookups, never positional), but scrambled order is a landmine
            # for any future .iloc/.tail()-based code and just confusing to
            # read directly.
            partial = pd.concat(list(results.values()), axis=1).sort_index()
            partial.to_csv(checkpoint_path)

    panel = pd.concat([results[t] for t in all_tickers], axis=1).sort_index()
    return panel[all_tickers]


if __name__ == "__main__":
    print("=" * 70)
    print("VALIDATION FIRST -- run this small check before the full historical pull.")
    print("Prints the REAL column names returned -- confirm these match what")
    print("pull_ticker_atm_iv_history() assumes (strike, right, bid, ask, date) before")
    print("running the full build; if they don't, this is a quick rename fix, not a bug hunt.")
    print("=" * 70)

    client = get_client()

    test_exp = third_fridays(date(2025, 8, 1), date(2025, 9, 1))[0]
    print(f"\nTest expiration: {test_exp}")

    test_strikes_raw = client.option_list_strikes(symbol="QQQ", expiration=test_exp)
    test_strikes_df = _eod_to_pandas(test_strikes_raw)
    print(f"\noption_list_strikes returned a {type(test_strikes_raw).__name__}, "
          f"columns: {list(test_strikes_df.columns)}, {len(test_strikes_df)} rows")
    print(f"Sample:\n{test_strikes_df.head()}")

    test_eod = client.option_history_eod(
        symbol="QQQ", expiration=test_exp, strike="*", right="both",
        start_date=date(2025, 8, 1), end_date=date(2025, 8, 8),
    )
    test_df = _eod_to_pandas(test_eod)
    print(f"\nColumns returned: {list(test_df.columns)}")
    print(f"Row count: {len(test_df)}")
    print(f"\nSample rows:\n{test_df.head(10)}")

    # Also test a SPECIFIC numeric strike (stringified), not just strike='*' --
    # this is the actual call shape pull_ticker_atm_iv_history uses, and the
    # first real run crashed exactly here (passed a raw float, not a string;
    # strike is a protobuf TYPE_STRING field, confirmed via the descriptor).
    validation_ok = True
    if test_strikes_df.empty or "strike" not in test_strikes_df.columns:
        print("\n*** option_list_strikes didn't return a usable 'strike' column -- ABORTING. ***")
        validation_ok = False
    else:
        single_strike = str(round(float(test_strikes_df["strike"].iloc[len(test_strikes_df) // 2]), 2))
        try:
            single_eod = client.option_history_eod(
                symbol="QQQ", expiration=test_exp, strike=single_strike, right="both",
                start_date=date(2025, 8, 1), end_date=date(2025, 8, 8),
            )
            single_df = _eod_to_pandas(single_eod)
            print(f"\nSingle-strike ({single_strike!r}) pull: {len(single_df)} rows")
            if single_df.empty:
                print("*** WARNING: single-strike pull returned 0 rows -- the string format may be "
                      "wrong even though it didn't error. Check this before trusting the full pull. ***")
        except Exception as e:
            print(f"\n*** Single-strike pull FAILED: {e} -- this is exactly what broke the first real "
                  "run. ABORTING. ***")
            validation_ok = False
    if "right" not in test_df.columns:
        print("\n*** No 'right' column found -- ABORTING before the full pull. ***")
        validation_ok = False
    elif "created" not in test_df.columns:
        print("\n*** No 'created' column found -- date derivation would fail. ABORTING. ***")
        validation_ok = False
    else:
        sides = test_df["right"].unique()
        print(f"\nDistinct 'right' values present: {sides}")
        if len(sides) < 2:
            print("*** WARNING: only one side present in a right='both' pull -- ABORTING. ***")
            validation_ok = False

    # Sanity-check get_real_spot_history (yfinance, auto_adjust=False) against
    # a known real-vs-adjusted gap -- VZ's real 2021-01-04 close is $58.85 vs
    # get_price_history's (auto_adjust=True) $41.53. If these ever come back
    # equal, auto_adjust isn't doing what's assumed and this fix is silently
    # inert.
    print("\nValidating get_real_spot_history against the known VZ dividend-adjustment gap...")
    from data.fetch import get_price_history
    real_vz = get_real_spot_history(["VZ"], period="10y")["VZ"]
    real_vz.index = pd.DatetimeIndex(real_vz.index).tz_localize(None)
    adj_vz = get_price_history(["VZ"], period="10y")["VZ"]
    adj_vz.index = pd.DatetimeIndex(adj_vz.index).tz_localize(None)
    target = pd.Timestamp("2021-01-04")
    if target not in real_vz.index or target not in adj_vz.index:
        print(f"\n*** 2021-01-04 not in VZ history (period='10y' may not reach back far enough "
              "from today) -- can't validate. ABORTING. ***")
        validation_ok = False
    else:
        real_val, adj_val = float(real_vz.loc[target]), float(adj_vz.loc[target])
        print(f"VZ 2021-01-04: real={real_val:.2f}  adjusted={adj_val:.2f}")
        if abs(real_val - 58.85) > 1.0 or abs(real_val - adj_val) < 5.0:
            print("\n*** real/adjusted values don't match the expected known gap -- "
                  "ABORTING before trusting this as the spot source. ***")
            validation_ok = False

    # Live smoke-test of the v3 BANDED PER-DAY logic itself (STRIKE_BAND_PCT,
    # multi-strike pull, per-day nearest-strike selection) -- every other new
    # mechanism in this file has needed a real live check before being
    # trusted in a long run, and this is the biggest rewrite yet. Uses a
    # short, recent, real window rather than the full historical range.
    print("\nValidating v3 banded per-day strike selection (real call, short recent window)...")
    smoke_spot = get_real_spot_history(["AAPL"], period="1mo")["AAPL"]
    smoke_spot.index = pd.DatetimeIndex(smoke_spot.index).tz_localize(None)
    if smoke_spot.empty:
        print("\n*** get_real_spot_history returned no AAPL data for the smoke test -- ABORTING. ***")
        validation_ok = False
    else:
        smoke_exp = third_fridays(date.today() - pd.Timedelta(days=45), date.today() + pd.Timedelta(days=10))
        smoke_exp = [e for e in smoke_exp if e > date.today()][:1] or smoke_exp[-1:]
        smoke_result = pull_ticker_atm_iv_history(client, "AAPL", smoke_spot, smoke_exp)
        print(f"AAPL banded pull over the last month: {len(smoke_result)} daily IV readings")
        print(smoke_result.tail(10))
        if smoke_result.empty:
            print("\n*** Banded pull returned ZERO readings for a real, recent, liquid name -- "
                  "ABORTING before trusting this on the full historical range. ***")
            validation_ok = False
        elif smoke_result.isna().any() or (smoke_result <= 0).any() or (smoke_result > 5).any():
            print("\n*** Banded pull returned implausible IV values (NaN, <=0, or >500%) -- "
                  "ABORTING. ***")
            validation_ok = False

    if not validation_ok:
        print("\nSchema doesn't match what pull_ticker_atm_iv_history() expects -- fix that function "
              "before retrying, not just re-running this validation block.")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("Validation passed -- schema matches what pull_ticker_atm_iv_history() expects.")
    print("Running the full historical build now (Value tier, DATA_FLOOR=2021-01-01, ~5.6y).")
    print("v3's banded per-day strike selection pulls roughly STRIKE_BAND_PCT-many strikes per")
    print("expiration instead of v2's single strike -- meaningfully more API calls than the old")
    print("~2,100-call/15-25-minute estimate. Expect a couple of hours, not minutes. Progress checkpoints")
    print("to disk after every ticker.")
    print("=" * 70)

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "thetadata_iv_panel.csv")
    panel = build_basket_iv_panel(DATA_FLOOR, date.today(), checkpoint_path=out_path)
    panel.to_csv(out_path)
    print(f"\nSaved panel with shape {panel.shape} to {out_path}")
    print(panel.tail(10))
