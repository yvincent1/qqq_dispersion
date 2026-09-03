"""
One-off diagnostic: compare exactly what pull_ticker_atm_iv_history()
chose for a ticker's 2026-09-18 expiration against the live yfinance
pull, for whatever ticker is named on the command line.

Already resolved for AAPL this way: same strike (310), same tight
spread both days, gap explained by a real ~4% spot swing that week
(305.59 -> 316.83 -> 309.35) plus genuine short-term IV settling, not
a bug. Re-running for COST -- live side already checked separately:
spot=947.74, strike=950.0, bid/ask=19.65/22.30 (mid=20.98), volume=112,
OI=952 (much thinner than AAPL's 4466/19873). COST also had a bigger
single-day move: 956.99 -> 933.51 (-2.5%) on 2026-08-20, the historical
panel's date, then bounced to 947.74 by 2026-08-21.

Usage: python diagnose_aapl_gap.py [TICKER]  (defaults to AAPL)
"""

import os
import sys
from datetime import date

import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "common"))
from thetadata_client import get_client

from fetch import get_price_history

TICKER = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
EXP = date(2026, 9, 18)
TARGET_DATE = date(2026, 8, 20)
print(f"Diagnosing: {TICKER}\n")

client = get_client()

spot_series = get_price_history([TICKER], period="2y")[TICKER]
spot_series.index = pd.DatetimeIndex(spot_series.index).tz_localize(None)
window_spot = spot_series[spot_series.index.date <= EXP]
ref_spot = float(window_spot.iloc[-1])
print(f"ref_spot (what pull_ticker_atm_iv_history used): {ref_spot:.2f}")
print(f"  as of: {window_spot.index[-1].date()}")

strikes_raw = client.option_list_strikes(symbol=TICKER, expiration=EXP)
strikes_df = strikes_raw.to_pandas() if hasattr(strikes_raw, "to_pandas") else pd.DataFrame(strikes_raw)
strikes = strikes_df["strike"].tolist()
nearest_strike = min(strikes, key=lambda s: abs(s - ref_spot))
print(f"\nChosen strike (nearest to ref_spot among {len(strikes)} listed): {nearest_strike}")

nearby = sorted(strikes, key=lambda s: abs(s - ref_spot))[:8]
print(f"Nearest 8 listed strikes to ref_spot: {sorted(nearby)}")

eod = client.option_history_eod(
    symbol=TICKER, expiration=EXP, strike=str(nearest_strike), right="both",
    start_date=date(2026, 8, 1), end_date=date(2026, 8, 21),
)
eod_df = eod.to_pandas() if hasattr(eod, "to_pandas") else pd.DataFrame(eod)
eod_df["trade_date"] = pd.to_datetime(eod_df["created"]).dt.date
calls = eod_df[eod_df["right"] == "CALL"].sort_values("trade_date")

print(f"\nAll CALL EOD rows for strike {nearest_strike}, Aug 2026:")
print(calls[["trade_date", "bid", "ask", "close", "volume"]].to_string(index=False))

target_row = calls[calls["trade_date"] == TARGET_DATE]
if not target_row.empty:
    row = target_row.iloc[0]
    mid = (row["bid"] + row["ask"]) / 2
    print(f"\n--- {TARGET_DATE} specifically ---")
    print(f"strike={nearest_strike}  bid={row['bid']}  ask={row['ask']}  mid={mid:.2f}  "
          f"volume={row['volume']}  spread_pct={(row['ask']-row['bid'])/mid:.1%}")
else:
    print(f"\nNo row found for exactly {TARGET_DATE} -- check the full table above for nearby dates.")
