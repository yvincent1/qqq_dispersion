"""
Implied correlation (from options-implied vols) and realized correlation
(from historical returns), both properly weighted by index weights --
not the simplified equal-weight approximation used for intuition-building.

Formula (weighted version):
    IndexVar = sum_i(w_i^2 * sigma_i^2) + sum_{i != j}(w_i * w_j * sigma_i * sigma_j * rho_ij)

If we assume a single "average" pairwise correlation rho for all pairs
(a common simplifying assumption -- the "implied average correlation"
that index options markets actually quote), this collapses to:

    IndexVar = sum_i(w_i^2 * sigma_i^2) + rho * [ (sum_i w_i*sigma_i)^2 - sum_i(w_i^2 * sigma_i^2) ]

Solving for rho given a known/observed IndexVol (from index options)
and each component's sigma_i (from single-stock options) gives the
weighted implied correlation.
"""

import numpy as np
import pandas as pd


def implied_correlation(index_vol, weights, component_vols):
    """
    Back out the single 'average implied correlation' consistent with
    observed index vol and component vols, given index weights.

    index_vol: float, annualized index IV
    weights: array-like, index weights (should sum to ~1; will be
             renormalized if the basket is a subset of the full index)
    component_vols: array-like, annualized IV per component, same order as weights

    Returns: float, implied average correlation (can exceed [0,1] with
             noisy/inconsistent inputs -- caller should sanity check).
    """
    w = np.asarray(weights, dtype=float)
    sig = np.asarray(component_vols, dtype=float)
    w = w / w.sum()  # renormalize in case this is a subset basket

    weighted_sum_sq = np.sum((w * sig) ** 2)          # sum_i(w_i^2 sigma_i^2)
    weighted_sum_total = (np.sum(w * sig)) ** 2         # (sum_i w_i sigma_i)^2
    cross_term_capacity = weighted_sum_total - weighted_sum_sq

    index_var = index_vol ** 2

    if cross_term_capacity <= 0:
        return np.nan  # degenerate basket (e.g. single name)

    rho = (index_var - weighted_sum_sq) / cross_term_capacity
    return rho


def realized_correlation(returns_df, weights=None):
    """
    Compute the weighted-average pairwise realized correlation from a
    DataFrame of daily returns (columns = tickers, rows = dates).

    weights: optional dict or Series of {ticker: weight}. If provided,
             returns the weight-weighted average of the pairwise
             correlation matrix (off-diagonal only). If None, uses a
             simple unweighted average of all pairs.

    Returns: float
    """
    corr_matrix = returns_df.corr()
    tickers = corr_matrix.columns

    if weights is None:
        w = pd.Series(1.0, index=tickers)
    else:
        w = pd.Series(weights).reindex(tickers).fillna(0.0)
    w = w / w.sum()

    numerator = 0.0
    denominator = 0.0
    for i in tickers:
        for j in tickers:
            if i == j:
                continue
            pair_weight = w[i] * w[j]
            numerator += pair_weight * corr_matrix.loc[i, j]
            denominator += pair_weight

    return numerator / denominator if denominator > 0 else np.nan


def realized_vol(returns_series, trading_days=252):
    """Annualized realized volatility from a daily return series."""
    return returns_series.std() * np.sqrt(trading_days)


def dispersion_signal(implied_corr, realized_corr, threshold=0.10):
    """
    Classify the implied-vs-realized correlation spread into a trade
    signal. Shared by the live pipeline and the backtester so the two
    can't silently drift apart on what counts as a signal.

    Returns: "sell_dispersion", "buy_dispersion", or "neutral".
    """
    spread = implied_corr - realized_corr
    if spread > threshold:
        return "sell_dispersion"
    elif spread < -threshold:
        return "buy_dispersion"
    return "neutral"


if __name__ == "__main__":
    # Sanity check against the course example:
    # 3 equal-weighted stocks, vols 35/30/25%, index vol 27% -> rho should be ~0.81
    weights = [1 / 3, 1 / 3, 1 / 3]
    vols = [0.35, 0.30, 0.25]
    index_vol = 0.27

    rho = implied_correlation(index_vol, weights, vols)
    print(f"Implied correlation: {rho:.3f} (expected ~0.81 from the course example)")

    # Sanity check realized correlation with synthetic data
    np.random.seed(0)
    n_days = 252
    common_factor = np.random.normal(0, 1, n_days)
    idio = np.random.normal(0, 1, (n_days, 3))
    target_rho = 0.55
    # construct returns with a known average correlation via a single-factor model
    loadings = np.sqrt(target_rho)
    idio_scale = np.sqrt(1 - target_rho)
    rets = loadings * common_factor[:, None] + idio_scale * idio
    df = pd.DataFrame(rets, columns=["A", "B", "C"])

    realized_rho = realized_correlation(df)
    print(f"Realized correlation (synthetic, target={target_rho}): {realized_rho:.3f}")
