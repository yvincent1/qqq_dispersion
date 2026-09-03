"""
Live-data run of the dispersion pipeline for the DOW (DIA) instead of
QQQ -- pulls real returns and options IVs for all 30 Dow constituents,
computes implied vs. realized correlation, and prints the current
signal. Mirrors run_live.py exactly, swapping the index/basket.

Weights come from get_dia_weights() (live, price-based -- the Dow is
price-weighted, not market-cap-weighted like QQQ) rather than a
hardcoded table, so there's no staleness to manage the way there is
for QQQ_TOP_CONSTITUENTS.
"""

import numpy as np
import pandas as pd

from data.fetch import get_daily_returns, get_atm_iv, DIA_CONSTITUENTS, get_dia_weights
from correlation import implied_correlation, realized_correlation, realized_vol

tickers = DIA_CONSTITUENTS
weights = get_dia_weights(tickers)

print("=" * 60)
print("STEP 1: Realized correlation from historical returns (3mo)")
print("=" * 60)

returns_df = get_daily_returns(tickers, period="3mo")
print(f"Returns shape: {returns_df.shape}, date range: "
      f"{returns_df.index.min().date()} to {returns_df.index.max().date()}")
missing = [t for t in tickers if t not in returns_df.columns or returns_df[t].isna().all()]
if missing:
    print(f"WARNING: no return data for {missing}")

realized_vols = {t: realized_vol(returns_df[t]) for t in tickers if t in returns_df.columns}
print("\nRealized annualized vol per name:")
for t, v in realized_vols.items():
    print(f"  {t}: {v:.1%}")

realized_rho = realized_correlation(returns_df, weights=weights)
print(f"\nWeighted realized correlation (3mo): {realized_rho:.3f}")

print("\n" + "=" * 60)
print("STEP 2: Implied vols from live options chains (~30 DTE)")
print("=" * 60)

implied_vols = {}
failures = []
for t in tickers:
    iv, dte, expiry = get_atm_iv(t, target_dte_days=30)
    implied_vols[t] = iv
    status = f"{iv:.1%}" if not np.isnan(iv) else "FAILED (NaN)"
    print(f"  {t}: IV={status}, DTE={dte}, expiry={expiry}")
    if np.isnan(iv):
        failures.append(t)

dia_iv, dia_dte, dia_expiry = get_atm_iv("DIA", target_dte_days=30)
print(f"\nDIA index IV: {dia_iv:.1%}, DTE={dia_dte}, expiry={dia_expiry}")

if failures:
    print(f"\nWARNING: IV extraction failed for {failures} -- excluding from implied corr calc.")

print("\n" + "=" * 60)
print("STEP 3: Implied correlation")
print("=" * 60)

good_tickers = [t for t in tickers if not np.isnan(implied_vols[t])]
w = weights[good_tickers].values
sig = np.array([implied_vols[t] for t in good_tickers])

recovered_implied_corr = implied_correlation(dia_iv, w, sig)
print(f"Implied correlation ({len(good_tickers)} names, "
      f"{weights[good_tickers].sum():.1%} of basket weight used): {recovered_implied_corr:.3f}")

print("\n" + "=" * 60)
print("STEP 4: The signal")
print("=" * 60)

spread = recovered_implied_corr - realized_rho
print(f"Implied correlation:  {recovered_implied_corr:.3f}")
print(f"Realized correlation: {realized_rho:.3f}")
print(f"Spread (implied - realized): {spread:+.3f}")
if spread > 0.10:
    print("-> Signal: SELL DISPERSION (index straddle rich vs. components)")
elif spread < -0.10:
    print("-> Signal: BUY DISPERSION (index straddle cheap vs. components)")
else:
    print("-> Signal: NEUTRAL (spread too small to trade)")
