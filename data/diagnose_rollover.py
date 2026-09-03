"""
One-off diagnostic: for a ticker+date that showed an extreme IV outlier
in the DIA panel, check EVERY monthly expiration that had data covering
that date, showing DTE and raw bid/ask for each -- to see whether the
final picked reading (whichever was "closest to 30 DTE" among what was
AVAILABLE, not necessarily what was best) ended up being a noisy,
near-expiration contract because the actually-close-to-30-DTE
expiration's data was missing/filtered for that specific day.

Usage: python diagnose_rollover.py TICKER YYYY-MM-DD
"""

import os
import sys
from datetime import date, datetime

import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "common"))
from thetadata_fetch import get_client, third_fridays, RISK_FREE_RATE, _eod_to_pandas

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from iv_solver import implied_vol

from fetch import get_price_history

TICKER = sys.argv[1] if len(sys.argv) > 1 else "VZ"
TARGET_DATE = datetime.strptime(sys.argv[2], "%Y-%m-%d").date() if len(sys.argv) > 2 else date(2025, 9, 18)

client = get_client()

spot_series = get_price_history([TICKER], period="2y")[TICKER]
spot_series.index = pd.DatetimeIndex(spot_series.index).tz_localize(None)
spot_match = spot_series[spot_series.index.date == TARGET_DATE]
spot = float(spot_match.iloc[0]) if not spot_match.empty else None
print(f"{TICKER} spot on {TARGET_DATE}: {spot}")

expirations = third_fridays(date(2025, 6, 1), date(2026, 8, 21))
covering = [e for e in expirations if e > TARGET_DATE and (e - TARGET_DATE).days <= 60]
print(f"\nExpirations that could plausibly cover {TARGET_DATE}: {covering}")

for exp in covering:
    dte = (exp - TARGET_DATE).days
    window_spot = spot_series[spot_series.index.date <= exp]
    ref_spot = float(window_spot.iloc[-1])

    strikes_raw = client.option_list_strikes(symbol=TICKER, expiration=exp)
    strikes_df = _eod_to_pandas(strikes_raw)
    if strikes_df.empty:
        print(f"\n{exp} (DTE={dte}): no strikes listed")
        continue
    nearest_strike = min(strikes_df["strike"].tolist(), key=lambda s: abs(s - ref_spot))

    eod = client.option_history_eod(
        symbol=TICKER, expiration=exp, strike=str(nearest_strike), right="CALL",
        start_date=TARGET_DATE, end_date=TARGET_DATE,
    )
    eod_df = _eod_to_pandas(eod)
    if eod_df.empty:
        print(f"\n{exp} (DTE={dte}, ref_spot={ref_spot:.2f}, strike={nearest_strike}): NO DATA for {TARGET_DATE}")
        continue

    row = eod_df.iloc[0]
    bid, ask = row["bid"], row["ask"]
    mid = (bid + ask) / 2 if bid > 0 and ask > 0 else None
    T = dte / 365
    iv = implied_vol(mid, spot, nearest_strike, T, RISK_FREE_RATE, "call") if mid and spot else np.nan
    dte_dist = abs(dte - 30)
    print(f"\n{exp} (DTE={dte}, dte_dist={dte_dist}, ref_spot={ref_spot:.2f}, strike={nearest_strike}):")
    print(f"  bid={bid} ask={ask} mid={mid} -> IV={iv}")
