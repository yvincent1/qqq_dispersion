"""
End-to-end pipeline demo, using synthetic data so it runs without
network access. Validates that all the pieces connect correctly.

Once this makes sense, swap the synthetic data generation in Step 1-2
for real calls into data/fetch.py (run locally, where yfinance can
actually reach the internet).
"""

import numpy as np
import pandas as pd

from iv_solver import bs_price, bs_vega, implied_vol
from correlation import implied_correlation, realized_correlation, realized_vol
from optimizer import select_basket, size_basket_by_vega, optimize_for_dispersion_edge

np.random.seed(7)

print("=" * 60)
print("STEP 1: Universe setup (synthetic QQQ-like basket)")
print("=" * 60)

universe = pd.DataFrame({
    "weight": [0.088, 0.085, 0.085, 0.055, 0.045, 0.045, 0.035, 0.030],
    "liquidity_score": [0.95, 0.95, 0.90, 0.85, 0.75, 0.80, 0.70, 0.65],
}, index=["AAPL", "MSFT", "NVDA", "AMZN", "AVGO", "META", "GOOGL", "TSLA"])
print(universe)

basket = select_basket(universe, target_cumulative_weight=0.75,
                        max_names=6, min_liquidity_score=0.5)
print(f"\nSelected basket ({len(basket)} names, "
      f"{universe.loc[basket, 'weight'].sum():.1%} of universe weight): {basket}")

print("\n" + "=" * 60)
print("STEP 2: Simulate historical returns with a known correlation structure")
print("=" * 60)

n_days = 90
true_realized_corr = 0.48  # what we'll plant in the synthetic data
common_factor = np.random.normal(0, 1, n_days)
idio = np.random.normal(0, 1, (n_days, len(basket)))
loading = np.sqrt(true_realized_corr)
idio_scale = np.sqrt(1 - true_realized_corr)
daily_vols = np.random.uniform(0.015, 0.03, len(basket))  # daily return stdev per name
rets = (loading * common_factor[:, None] + idio_scale * idio) * daily_vols
returns_df = pd.DataFrame(rets, columns=basket)

realized_vols_annualized = {t: realized_vol(returns_df[t]) for t in basket}
print("Realized annualized vol per name:")
for t, v in realized_vols_annualized.items():
    print(f"  {t}: {v:.1%}")

realized_rho = realized_correlation(returns_df, weights=universe.loc[basket, "weight"])
print(f"\nWeighted realized correlation (last {n_days}d): {realized_rho:.3f}")

print("\n" + "=" * 60)
print("STEP 3: Synthetic 'current' implied vols (options market pricing)")
print("=" * 60)

# Simulate implied vols running a bit above realized (normal vol risk premium),
# and simulate an index IV consistent with HIGHER-than-realized implied correlation
# -- i.e., manufacture the "sell dispersion" setup from the course example.
implied_vols = {t: realized_vols_annualized[t] * np.random.uniform(1.05, 1.20)
                 for t in basket}
print("Implied vol per name (options-market, synthetic):")
for t, v in implied_vols.items():
    print(f"  {t}: {v:.1%}")

target_implied_corr = 0.75  # deliberately set above realized (0.48) to create a signal
w = universe.loc[basket, "weight"].values
w = w / w.sum()
sig = np.array([implied_vols[t] for t in basket])
weighted_sum_sq = np.sum((w * sig) ** 2)
weighted_sum_total = (np.sum(w * sig)) ** 2
cross_capacity = weighted_sum_total - weighted_sum_sq
synthetic_index_var = weighted_sum_sq + target_implied_corr * cross_capacity
synthetic_index_iv = np.sqrt(synthetic_index_var)
print(f"\nSynthetic index IV (back-solved to hit target implied corr): {synthetic_index_iv:.1%}")

recovered_implied_corr = implied_correlation(synthetic_index_iv, w, sig)
print(f"Recovered implied correlation (round-trip check): {recovered_implied_corr:.3f} "
      f"(should equal {target_implied_corr})")

print("\n" + "=" * 60)
print("STEP 4: The signal")
print("=" * 60)

spread = recovered_implied_corr - realized_rho
print(f"Implied correlation: {recovered_implied_corr:.3f}")
print(f"Realized correlation: {realized_rho:.3f}")
print(f"Spread (implied - realized): {spread:+.3f}")
if spread > 0.10:
    print("-> Signal: SELL DISPERSION (index straddle rich vs. components)")
elif spread < -0.10:
    print("-> Signal: BUY DISPERSION (index straddle cheap vs. components)")
else:
    print("-> Signal: NEUTRAL (spread too small to trade)")

print("\n" + "=" * 60)
print("STEP 5: Size the basket by vega")
print("=" * 60)

S = 100  # normalize each name to spot=100 for illustration
T = 30 / 365
r = 0.045

straddle_vegas = []
for t in basket:
    vega_call = bs_vega(S, S, T, r, implied_vols[t])
    straddle_vegas.append(2 * vega_call)  # straddle = call + put, same vega each leg

index_vega = 2 * bs_vega(S, S, T, r, synthetic_index_iv)
print(f"Index straddle vega (per contract, spot-normalized): {index_vega:.2f}")

target_total_vega_to_offset = index_vega * 10  # e.g. short 10 index straddles
counts = size_basket_by_vega(target_total_vega_to_offset, w, straddle_vegas)
sizing_table = pd.DataFrame({
    "weight": w,
    "implied_vol": sig,
    "straddle_vega": straddle_vegas,
    "straddle_count": np.round(counts, 1),
}, index=basket)
print(f"\nTo offset {target_total_vega_to_offset:.1f} vega (short 10 index straddles):")
print(sizing_table)

print("\n" + "=" * 60)
print("STEP 6: Basket weight optimization")
print("=" * 60)

liq = universe.loc[basket, "liquidity_score"].values
optimized = optimize_for_dispersion_edge(
    weights=w, component_vols=sig, component_liquidity=liq,
    implied_corr=recovered_implied_corr, realized_corr=realized_rho,
    liquidity_floor=0.5, max_concentration=0.35,
)
opt_table = pd.DataFrame({
    "original_weight": w,
    "optimized_weight": optimized.values,
    "liquidity": liq,
}, index=basket)
print(opt_table)

print("\nPipeline ran end-to-end successfully with synthetic data.")
print("Next: replace Steps 2-3's synthetic generation with real calls into")
print("data/fetch.py (get_daily_returns, get_atm_iv), run locally.")
