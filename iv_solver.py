"""
Black-Scholes pricing and implied volatility extraction.

This is the foundation everything else builds on: given an option's
market price, back out the volatility the market is implying.
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq


def bs_price(S, K, T, r, sigma, option_type="call", q=0.0):
    """
    Black-Scholes price for a European option.

    S: spot price
    K: strike price
    T: time to expiration in years
    r: risk-free rate (annualized, decimal)
    sigma: volatility (annualized, decimal)
    option_type: "call" or "put"
    q: dividend yield (annualized, decimal)
    """
    if T <= 0 or sigma <= 0:
        # At/after expiration, or degenerate vol -> intrinsic value only
        if option_type == "call":
            return max(S - K, 0.0)
        else:
            return max(K - S, 0.0)

    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)

    return price


def bs_vega(S, K, T, r, sigma, q=0.0):
    """Vega: dPrice/dSigma. Same formula for calls and puts."""
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)


def bs_delta(S, K, T, r, sigma, option_type="call", q=0.0):
    """
    Black-Scholes delta: dPrice/dS.

    Returns a value in (0, 1) for calls, (-1, 0) for puts -- the
    "probability-like" delta convention used to quote option moneyness
    (e.g. "25-delta put") independent of raw strike distance.
    """
    if T <= 0 or sigma <= 0:
        if option_type == "call":
            return 1.0 if S > K else 0.0
        else:
            return -1.0 if S < K else 0.0

    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

    if option_type == "call":
        return np.exp(-q * T) * norm.cdf(d1)
    else:
        return -np.exp(-q * T) * norm.cdf(-d1)


def implied_vol(market_price, S, K, T, r, option_type="call", q=0.0,
                 lo=0.001, hi=5.0):
    """
    Solve for the implied volatility that reproduces market_price,
    using Brent's method (robust, doesn't need a derivative or a
    good starting guess like Newton-Raphson does).

    Returns np.nan if no solution exists in [lo, hi] (e.g. the quoted
    price is below intrinsic value / arbitrage-violating, which does
    happen with stale or crossed quotes in real data).
    """
    intrinsic = max(S - K, 0.0) if option_type == "call" else max(K - S, 0.0)
    if market_price < intrinsic - 1e-6:
        return np.nan

    def objective(sigma):
        return bs_price(S, K, T, r, sigma, option_type, q) - market_price

    try:
        return brentq(objective, lo, hi, xtol=1e-6)
    except ValueError:
        # objective doesn't change sign in [lo, hi] -> price outside
        # what any vol in that range could produce
        return np.nan


if __name__ == "__main__":
    # Sanity check: price an option, then recover the vol we priced it at
    S, K, T, r, true_sigma = 100, 100, 30 / 365, 0.05, 0.30
    price = bs_price(S, K, T, r, true_sigma, "call")
    recovered = implied_vol(price, S, K, T, r, "call")
    print(f"Priced with sigma={true_sigma:.4f}, recovered sigma={recovered:.4f}")
    assert abs(recovered - true_sigma) < 1e-4, "IV solver round-trip failed"
    print("OK: IV solver round-trips correctly.")
