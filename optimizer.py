"""
Basket optimization for a dispersion trade.

Two related but distinct problems, both handled here:

1. SELECTION: given a universe of candidate names with weights, vols,
   and liquidity scores, choose a subset that captures enough index
   weight while respecting a max-names constraint and a liquidity floor.

2. SIZING: given a chosen basket, compute per-name straddle counts so
   the basket's total vega matches (offsets) a target index vega,
   weighted proportionally to each name's index weight.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def select_basket(constituents_df, target_cumulative_weight=0.70,
                   max_names=15, min_liquidity_score=0.0):
    """
    constituents_df: DataFrame with columns ['weight', 'liquidity_score']
                      indexed by ticker. liquidity_score is any consistent
                      proxy (e.g. avg daily options volume, normalized 0-1).
    target_cumulative_weight: stop adding names once this fraction of
                      total universe weight is captured.
    max_names: hard cap on basket size.
    min_liquidity_score: filter out anything below this threshold first.

    Returns: list of selected tickers, in the order they were added.
    """
    df = constituents_df[constituents_df["liquidity_score"] >= min_liquidity_score].copy()
    df = df.sort_values("weight", ascending=False)

    total_universe_weight = df["weight"].sum()
    target_weight = target_cumulative_weight * total_universe_weight

    selected = []
    cumulative = 0.0
    for ticker, row in df.iterrows():
        if len(selected) >= max_names or cumulative >= target_weight:
            break
        selected.append(ticker)
        cumulative += row["weight"]

    return selected


def size_basket_by_vega(index_vega_target, weights, single_straddle_vegas):
    """
    Given a target total vega to offset (from the short index straddle
    position) and each name's per-straddle vega, compute how many
    straddles of each name to buy so total component vega matches the
    target, distributed proportionally to index weight.

    weights: array-like, index weight per name (will be renormalized)
    single_straddle_vegas: array-like, vega of ONE straddle (call+put)
                            per name, same order as weights

    Returns: np.array of straddle counts per name (can be fractional --
             round for actual execution, understanding that rounding
             introduces small residual vega mismatch).
    """
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    vegas = np.asarray(single_straddle_vegas, dtype=float)

    # Target vega contribution per name = index_vega_target * w_i
    target_vega_per_name = index_vega_target * w

    # Number of straddles needed = target vega / vega per straddle
    counts = target_vega_per_name / vegas
    return counts


def optimize_for_dispersion_edge(weights, component_vols, component_liquidity,
                                  implied_corr, realized_corr,
                                  liquidity_floor=0.3, max_concentration=0.35):
    """
    A simple constrained optimizer: reweight the basket (within bounds)
    to maximize expected dispersion edge, where edge is proxied by
    each name's contribution to the implied-vs-realized correlation gap,
    weighted by liquidity, subject to:
      - weights sum to 1
      - no single name exceeds max_concentration
      - names below liquidity_floor get zero weight

    This is intentionally a simple starting optimizer -- a real version
    would model expected P&L directly (e.g. via vega * (implied_rho -
    realized_rho) per name using single-name implied vs realized vol
    spreads too, not just a single scalar correlation gap). Treat this
    as a scaffold to extend once you have real per-name data.

    Returns: pd.Series of optimized weights indexed same as input order.
    """
    n = len(weights)
    w0 = np.asarray(weights, dtype=float)
    w0 = w0 / w0.sum()
    liq = np.asarray(component_liquidity, dtype=float)

    edge_scalar = implied_corr - realized_corr  # positive -> sell dispersion edge exists

    def neg_objective(w):
        # Reward weight placed on liquid names when there's a real edge;
        # otherwise (edge near zero) just track original index weights
        # to avoid taking unnecessary tracking-error risk for no reason.
        liquidity_bonus = np.sum(w * liq) * abs(edge_scalar)
        tracking_penalty = np.sum((w - w0) ** 2)  # stay close to true index weights
        return -(liquidity_bonus) + 5.0 * tracking_penalty

    bounds = [(0, max_concentration) for _ in range(n)]
    # zero out illiquid names via bounds
    bounds = [(0, 0) if liq[i] < liquidity_floor else b for i, b in enumerate(bounds)]

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    result = minimize(neg_objective, w0, bounds=bounds, constraints=constraints,
                       method="SLSQP")

    return pd.Series(result.x, index=range(n)) if result.success else pd.Series(w0, index=range(n))


if __name__ == "__main__":
    # Sanity check: selection
    universe = pd.DataFrame({
        "weight": [0.09, 0.08, 0.07, 0.05, 0.03, 0.02, 0.01],
        "liquidity_score": [0.9, 0.9, 0.8, 0.7, 0.2, 0.6, 0.1],
    }, index=["AAPL", "MSFT", "NVDA", "AMZN", "SMALLCAP1", "META", "SMALLCAP2"])

    basket = select_basket(universe, target_cumulative_weight=0.70,
                            max_names=5, min_liquidity_score=0.3)
    print(f"Selected basket: {basket}")
    assert "SMALLCAP1" not in basket and "SMALLCAP2" not in basket, \
        "Illiquid names should be filtered out"
    print("OK: illiquid names correctly excluded.")

    # Sanity check: sizing
    weights = [0.4, 0.35, 0.25]
    vegas = [12.0, 15.0, 10.0]  # dollar vega per straddle, hypothetical
    counts = size_basket_by_vega(index_vega_target=1000, weights=weights,
                                  single_straddle_vegas=vegas)
    print(f"Straddle counts per name: {np.round(counts, 1)}")
    total_vega_check = np.sum(counts * np.array(vegas))
    print(f"Total component vega achieved: {total_vega_check:.1f} (target: 1000)")
    assert abs(total_vega_check - 1000) < 1e-6, "Vega sizing should match target exactly"
    print("OK: vega sizing matches target.")
