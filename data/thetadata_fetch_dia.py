"""
Real historical ATM IV panel for DIA (Dow) + its 30 CURRENT constituents
PLUS 2 former members needed for point-in-time reconstruction (INTC, DOW
Inc. -- see data/fetch.py's DIA_ERAS), reusing thetadata_fetch.py's pull
logic entirely (build_basket_iv_panel, now parameterized by index_ticker
rather than hardcoded to QQQ, and now doing banded per-day strike
selection -- see pull_ticker_atm_iv_history's v3 docstring). No new
pulling logic here -- everything that was found and fixed against the
QQQ pull (expiration= not expiration_date=, no `date` column, right=
'CALL'/'PUT' not 'C'/'P', 365-day span cap, strike must be a string,
no built-in call timeout, resume/checkpoint index-alignment) already
applies here unchanged, since it's the same client and the same
option_history_eod/option_list_strikes calls underneath.

POINT-IN-TIME FIX (audit finding A1, survivorship bias): DIA_CONSTITUENTS
is today's 30-member roster, but two real membership changes happened
inside this project's backtest window (AMZN replaced WBA 2024-02-26;
NVDA+SHW replaced INTC+DOW Inc. 2024-11-08 -- verified against Wikipedia's
"Historical components of the DJIA", not assumed from memory). SHW/NVDA/
AMZN are already in DIA_CONSTITUENTS and get pulled below regardless; DOW
Inc. (ticker "DOW", the chemical company) is NOT in the current roster,
so it's added explicitly here. INTC is NOT added here -- it's already
pulled as part of the QQQ basket (data/thetadata_fetch.py's own
QQQ_TOP_CONSTITUENTS includes it), and backtest_real_run_dia.py reuses
that panel's INTC column rather than paying for a duplicate pull of the
same underlying options market. WBA (Walgreens Boots Alliance) is
UNOBTAINABLE for free -- yfinance purged its entire price history after
the 2025-08-28 going-private delisting, and Nasdaq's own historical API
has no record either (confirmed via direct live query) -- so the pre-
2024-02-26 era runs 29 of 30 real members, WBA excluded and disclosed
rather than silently substituted (see DIA_ERAS's docstring in fetch.py).

Scale note: banded per-day strike selection (v3) pulls roughly
STRIKE_BAND_PCT-many strikes per expiration instead of v2's single
strike -- meaningfully more API calls than the original ~4,200 estimate
below (that estimate is now stale; expect several hours for this larger,
32-ticker pull, not 30-50 minutes). Checkpointing after every ticker
still applies, so a long run surviving a sleep/network interruption
resumes cleanly.
"""

import os
import sys
from datetime import date

import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from thetadata_fetch import build_basket_iv_panel, _eod_to_pandas, get_client, third_fridays, DATA_FLOOR
from fetch import DIA_CONSTITUENTS

# DOW Inc. (chemicals) is a real former Dow member (removed 2024-11-08)
# needed for point-in-time reconstruction but not part of today's roster.
DIA_PULL_TICKERS = DIA_CONSTITUENTS + ["DOW"]

if __name__ == "__main__":
    print("=" * 70)
    print("VALIDATION FIRST -- quick check that DIA's own chain behaves like QQQ's did")
    print("(the underlying API/schema was already thoroughly validated against QQQ; this")
    print("just confirms DIA specifically returns usable data before the full 31-ticker pull).")
    print("=" * 70)

    client = get_client()
    test_exp = third_fridays(date(2025, 8, 1), date(2025, 9, 1))[0]

    test_strikes_raw = client.option_list_strikes(symbol="DIA", expiration=test_exp)
    test_strikes_df = _eod_to_pandas(test_strikes_raw)
    print(f"\nDIA option_list_strikes: {len(test_strikes_df)} strikes returned")

    validation_ok = not test_strikes_df.empty and "strike" in test_strikes_df.columns
    if not validation_ok:
        print("\n*** DIA option_list_strikes didn't return a usable 'strike' column -- ABORTING. ***")
        import sys
        sys.exit(1)

    single_strike = str(round(float(test_strikes_df["strike"].iloc[len(test_strikes_df) // 2]), 2))
    try:
        single_eod = client.option_history_eod(
            symbol="DIA", expiration=test_exp, strike=single_strike, right="both",
            start_date=date(2025, 8, 1), end_date=date(2025, 8, 8),
        )
        single_df = _eod_to_pandas(single_eod)
        print(f"Single-strike ({single_strike!r}) pull: {len(single_df)} rows")
        if single_df.empty:
            print("*** WARNING: 0 rows -- check before trusting the full pull. ***")
    except Exception as e:
        print(f"\n*** DIA single-strike pull FAILED: {e} -- ABORTING. ***")
        import sys
        sys.exit(1)

    print("\n" + "=" * 70)
    print("Validation passed. Running the full historical build now (Value tier,")
    print(f"DATA_FLOOR=2021-01-01, ~5.6y): {len(DIA_PULL_TICKERS)} tickers (30 current Dow")
    print("constituents + DOW Inc. for point-in-time reconstruction) + DIA index, banded")
    print("per-day strike selection (multiple strikes/expiration, not v2's single strike) --")
    print("expect several hours, not the old ~30-50 minute estimate. Progress checkpoints")
    print("to disk after every ticker, so a long run resumes cleanly if interrupted.")
    print("=" * 70)

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "thetadata_iv_panel_dia.csv")
    panel = build_basket_iv_panel(
        DATA_FLOOR, date.today(),
        tickers=DIA_PULL_TICKERS, index_ticker="DIA", checkpoint_path=out_path,
    )
    panel.to_csv(out_path)
    print(f"\nSaved panel with shape {panel.shape} to {out_path}")
    print(panel.tail(10))
