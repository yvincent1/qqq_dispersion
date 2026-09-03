"""
Interim historical implied-vol proxy while WRDS/OptionMetrics access is
pending.

v2: per-name proxy calibrated from TODAY's real single-name IV (via
get_atm_iv -- live options chains, same as run_live.py), rather than a
single shared, time-varying multiplier.

Why not a shared multiplier: the first version of this module applied
ONE multiplier (VXN / realized QQQ vol, varying by date but identical
across names) to every basket ticker. That's mathematically invariant
under implied_correlation() -- multiplying every component vol by the
same constant leaves the computed correlation completely unchanged
(verified numerically: feeding VXN=0.20 vs VXN=0.45 through the same
inputs produced an identical implied_rho). So that version wasn't
actually testing implied-vs-realized dispersion at all -- it silently
reduced to comparing two different realized-vol-based correlation
estimates with mismatched lookback windows.

This version instead calibrates a FIXED, per-name multiplier from
today's real options-market data:

    k_i = today's real ATM IV_i / today's trailing realized vol_i

and applies k_i to that name's full historical trailing realized vol
series. Because k_i differs across names (real, current cross-sectional
differences in how richly each name's options trade relative to its own
vol), the correlation math is no longer invariant to it. The index leg
keeps using real historical VXN directly -- no calibration needed, no
cancellation risk, and it's the one piece of this whole proxy that's
fully real history rather than a backward projection.

Remaining approximation: k_i is calibrated ONCE (today) and held fixed
across the whole historical window, so it can't capture genuine
time-variation in a name's relative richness (e.g. an earnings-week IV
spike five years ago). Better than the v1 shared-scalar construction,
still not real single-name historical IV -- swap this module out for a
real WRDS/vendor loader once available; backtest.py's interface doesn't
change either way.
"""

import numpy as np
import pandas as pd

from data.fetch import get_price_history, get_atm_iv


def get_vxn_history(period="max"):
    """Daily VXN close as a decimal (e.g. 0.225, not 22.5)."""
    vxn = get_price_history(["^VXN"], period=period)["^VXN"]
    return vxn / 100.0


def calibrate_name_multipliers(tickers, returns_panel, window=21):
    """
    For each ticker, pulls today's real ATM IV (live options chain) and
    divides by that ticker's own trailing realized vol as of the most
    recent date in returns_panel, giving a fixed per-name vol-risk-
    premium multiplier. Falls back to 1.0 (IV == realized vol, no
    premium) with a printed warning if get_atm_iv can't return a usable
    value (illiquid name / no chain) -- keeps that name in the basket
    rather than silently dropping it from every historical date.
    """
    current_realized = returns_panel[tickers].iloc[-window:].std() * np.sqrt(252)
    multipliers = {}
    for t in tickers:
        iv, dte, expiry = get_atm_iv(t, target_dte_days=30)
        if np.isnan(iv) or current_realized[t] <= 0:
            print(f"WARNING: no usable live IV for {t} (iv={iv}) -- "
                  f"falling back to multiplier=1.0 (IV==realized vol)")
            multipliers[t] = 1.0
        else:
            multipliers[t] = iv / current_realized[t]
    return pd.Series(multipliers)


def build_proxy_iv_panel(returns_panel, index_ticker="QQQ", window=21):
    """
    returns_panel: DataFrame, index=date, columns include index_ticker
                   and every basket ticker -- same shape run_backtest()
                   expects for its returns_panel argument.
    window: trailing window (trading days) for both the realized-vol
            estimate underlying the proxy and the calibration snapshot.
            21 trading days ~= the 30-calendar-day tenor used elsewhere
            in this project (get_atm_iv's default).

    Returns: DataFrame, same index/columns as returns_panel -- proxy
             annualized implied vol per name per date (index_ticker's
             column is real historical VXN, not a proxy).
    """
    tickers = [c for c in returns_panel.columns if c != index_ticker]
    trailing_vol = returns_panel.rolling(window).std() * np.sqrt(252)

    multipliers = calibrate_name_multipliers(tickers, returns_panel, window)
    proxy_iv = trailing_vol[tickers].mul(multipliers, axis=1)

    vxn = get_vxn_history().reindex(returns_panel.index).ffill()
    proxy_iv[index_ticker] = vxn

    return proxy_iv[list(returns_panel.columns)]


if __name__ == "__main__":
    from data.fetch import get_daily_returns, QQQ_TOP_CONSTITUENTS
    from correlation import implied_correlation

    tickers = list(QQQ_TOP_CONSTITUENTS.keys())
    returns_panel = get_daily_returns(tickers + ["QQQ"], period="2y")
    iv_panel = build_proxy_iv_panel(returns_panel, index_ticker="QQQ")

    usable = iv_panel.dropna()
    print(f"\n{len(usable)} of {len(returns_panel)} dates have a usable proxy IV")
    print("\nMost recent proxy IVs:")
    print(usable.tail())

    # The whole point of the rewrite: VXN should now actually matter.
    w = pd.Series(QQQ_TOP_CONSTITUENTS)[tickers]
    w = w / w.sum()
    last = usable.iloc[-1]
    rho_real = implied_correlation(last["QQQ"], w.values, last[tickers].values)
    rho_doubled = implied_correlation(last["QQQ"] * 2, w.values, last[tickers].values)
    print(f"\nimplied_rho at real VXN: {rho_real:.3f}, at 2x VXN (hypothetical): "
          f"{rho_doubled:.3f} -- should differ now (confirms VXN isn't canceling out).")
