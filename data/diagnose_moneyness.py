"""
One-off diagnostic: replicate pull_ticker_atm_iv_history()'s exact logic
for a specific ticker/date that's still showing an extreme IV after the
MAX_MONEYNESS_PCT guard was added, to see exactly what's passing through
and why -- CSCO's 2022-05-13 reading (IV=3.69) was byte-for-byte
identical before and after the guard was added, which shouldn't happen
if the guard were actually rejecting something for this case.

Usage: python diagnose_moneyness.py TICKER YYYY-MM-DD
"""

import os
import sys
from datetime import date, datetime

import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "common"))
from thetadata_fetch import (
    get_client, third_fridays, _eod_to_pandas, RISK_FREE_RATE,
    MIN_DTE, MAX_SPREAD_PCT, MAX_MONEYNESS_PCT, DATA_FLOOR,
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from iv_solver import implied_vol

from fetch import get_price_history

TICKER = sys.argv[1] if len(sys.argv) > 1 else "CSCO"
TARGET_DATE = datetime.strptime(sys.argv[2], "%Y-%m-%d").date() if len(sys.argv) > 2 else date(2022, 5, 13)

client = get_client()

spot_series = get_price_history([TICKER], period="10y")[TICKER]
spot_series.index = pd.DatetimeIndex(spot_series.index).tz_localize(None)
spot_match = spot_series[spot_series.index.date == TARGET_DATE]
spot = float(spot_match.iloc[0]) if not spot_match.empty else None
print(f"{TICKER} actual spot on {TARGET_DATE}: {spot}")

# Find which expiration(s) could plausibly have contributed a reading for TARGET_DATE
expirations = third_fridays(DATA_FLOOR, date.today())
covering = [e for e in expirations if e > TARGET_DATE and (e - TARGET_DATE).days <= 46]
print(f"Expirations whose ~45-day window could cover {TARGET_DATE}: {covering}")

for exp in covering:
    dte = (exp - TARGET_DATE).days
    # Replicate ref_spot exactly as the real puller computes it
    window_spot = spot_series[spot_series.index.date <= exp]
    if window_spot.empty:
        print(f"\n{exp}: no spot data at/before expiration")
        continue
    ref_spot = float(window_spot.iloc[-1])
    ref_spot_date = window_spot.index[-1].date()

    strikes_raw = client.option_list_strikes(symbol=TICKER, expiration=exp)
    strikes_df = _eod_to_pandas(strikes_raw)
    if strikes_df.empty:
        print(f"\n{exp}: no strikes listed")
        continue
    strike_list = strikes_df["strike"].tolist()
    nearest_strike = round(min(strike_list, key=lambda s: abs(s - ref_spot)), 2)

    moneyness_pct = abs(nearest_strike - spot) / spot if spot else None

    eod = client.option_history_eod(
        symbol=TICKER, expiration=exp, strike=str(nearest_strike), right="CALL",
        start_date=TARGET_DATE, end_date=TARGET_DATE,
    )
    eod_df = _eod_to_pandas(eod)
    if eod_df.empty:
        print(f"\n{exp} (DTE={dte}, ref_spot={ref_spot:.2f} as of {ref_spot_date}, "
              f"strike={nearest_strike}, moneyness={moneyness_pct:.1%}): NO DATA for {TARGET_DATE}")
        continue

    row = eod_df.iloc[0]
    bid, ask = row["bid"], row["ask"]
    mid = (bid + ask) / 2 if bid > 0 and ask > 0 else None
    spread_pct = (ask - bid) / mid if mid else None
    T = dte / 365
    iv = implied_vol(mid, spot, nearest_strike, T, RISK_FREE_RATE, "call") if mid and spot else np.nan

    print(f"\n{exp} (DTE={dte}, ref_spot={ref_spot:.2f} as of {ref_spot_date}):")
    print(f"  strike={nearest_strike}  today's spot={spot}  moneyness={moneyness_pct:.1%} "
          f"(guard limit: {MAX_MONEYNESS_PCT:.0%}, {'WOULD REJECT' if moneyness_pct > MAX_MONEYNESS_PCT else 'passes'})")
    print(f"  dte={dte} (MIN_DTE={MIN_DTE}, {'WOULD REJECT' if dte < MIN_DTE else 'passes'})")
    print(f"  bid={bid} ask={ask} mid={mid} spread_pct={spread_pct:.1%} "
          f"(MAX_SPREAD_PCT={MAX_SPREAD_PCT:.0%}, {'WOULD REJECT' if spread_pct and spread_pct > MAX_SPREAD_PCT else 'passes'})")
    print(f"  -> IV = {iv}")
