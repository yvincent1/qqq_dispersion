"""
5-year synthetic demonstration of backtest.py running end-to-end.

Real historical implied vols require WRDS/OptionMetrics (pending as of
writing) or a paid vendor (Polygon/ORATS). This validates the PIPELINE
end-to-end -- data flow, signal timing over a real multi-year date range,
trade sizing, evaluation stats -- against synthetic data with a known,
planted correlation regime, the same way demo_pipeline.py validates the
live-signal math before real data.

This is a smoke test, not a profitability proof: trades here are NOT
delta-hedged (matches optimizer.py's current scope -- see backtest.py's
module docstring), so any individual trade's P&L carries real un-hedged
directional/gamma noise on top of the correlation-gap edge. Only
aggregate stats over many trades are informative, and even those are a
demonstration of the mechanics working, not evidence the real strategy
is profitable -- that answer has to come from real data.

Swap make_synthetic_panels() for a real WRDS/vendor loader once historical
IV data is available; nothing in backtest.py itself needs to change.
"""

import numpy as np
import pandas as pd

from backtest import run_backtest, evaluate_trades

np.random.seed(42)

TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "AVGO"]
WEIGHTS = pd.Series({"AAPL": 0.28, "MSFT": 0.27, "NVDA": 0.27, "AMZN": 0.10, "AVGO": 0.08})
INDEX = "QQQ"
N_YEARS = 5
N_DAYS = N_YEARS * 252


def make_synthetic_panels(true_rho, implied_rho):
    """
    N_YEARS of daily returns for TICKERS with a KNOWN, constant true
    pairwise correlation (true_rho), single-factor model. Per-name
    implied vol is set equal to that name's own generating vol (i.e.
    single-name vol is "fairly priced" -- isolates the correlation
    effect being tested), and index IV is back-solved to hit a planted
    implied_rho, held constant across the whole window.
    """
    dates = pd.bdate_range("2021-01-04", periods=N_DAYS)
    daily_vols = np.random.uniform(0.015, 0.03, len(TICKERS))

    common = np.random.normal(0, 1, N_DAYS)
    idio = np.random.normal(0, 1, (N_DAYS, len(TICKERS)))
    loading, idio_scale = np.sqrt(true_rho), np.sqrt(1 - true_rho)
    rets = (loading * common[:, None] + idio_scale * idio) * daily_vols
    returns_df = pd.DataFrame(rets, columns=TICKERS, index=dates)
    returns_df[INDEX] = returns_df[TICKERS].mul(WEIGHTS[TICKERS], axis=1).sum(axis=1)

    annualized_vols = daily_vols * np.sqrt(252)
    w = (WEIGHTS[TICKERS] / WEIGHTS[TICKERS].sum()).values
    weighted_sum_sq = np.sum((w * annualized_vols) ** 2)
    weighted_sum_total = np.sum(w * annualized_vols) ** 2
    cross_capacity = weighted_sum_total - weighted_sum_sq
    index_iv = np.sqrt(weighted_sum_sq + implied_rho * cross_capacity)

    iv_row = {t: v for t, v in zip(TICKERS, annualized_vols)}
    iv_row[INDEX] = index_iv
    iv_panel = pd.DataFrame([iv_row] * N_DAYS, index=dates)

    return returns_df, iv_panel


def run_scenario(name, true_rho, implied_rho):
    print("=" * 70)
    print(f"SCENARIO: {name}")
    print(f"  true_rho (realized regime) = {true_rho}, implied_rho (priced in) = {implied_rho}")
    print("=" * 70)
    returns_df, iv_panel = make_synthetic_panels(true_rho, implied_rho)
    trades = run_backtest(returns_df, iv_panel, WEIGHTS, index_ticker=INDEX)
    stats = evaluate_trades(trades)
    print(f"  n_trades={stats['n_trades']}, win_rate={stats['win_rate']:.1%}, "
          f"avg_pnl={stats['avg_pnl']:.3f}, sharpe={stats['sharpe']:.2f}")
    if stats["by_spread_bucket"] is not None:
        print("  P&L by |spread| bucket (bigger spread -> should trend toward bigger |pnl| "
              "if the edge is real):")
        print(stats["by_spread_bucket"].to_string().replace("\n", "\n  "))
    print()
    return trades, stats


if __name__ == "__main__":
    sell_trades, sell_stats = run_scenario(
        "implied >> realized correlation -> should mostly fire SELL DISPERSION",
        true_rho=0.35, implied_rho=0.70,
    )
    assert sell_stats["n_trades"] > 0, "Expected trades to fire in the sell scenario"
    assert (sell_trades["signal"] == "sell_dispersion").all(), \
        "Expected only sell signals given the size of this planted gap"

    buy_trades, buy_stats = run_scenario(
        "implied << realized correlation -> should mostly fire BUY DISPERSION",
        true_rho=0.70, implied_rho=0.35,
    )
    assert buy_stats["n_trades"] > 0
    assert (buy_trades["signal"] == "buy_dispersion").all()

    neutral_trades, neutral_stats = run_scenario(
        "implied ~= realized correlation -> should mostly NOT trade",
        true_rho=0.50, implied_rho=0.50,
    )
    print(f"Neutral scenario fired {neutral_stats['n_trades']} trades out of up to "
          f"~{(N_DAYS - 63) // 5} weekly checks (expect a small number from sampling noise "
          f"around the trailing realized-corr estimate crossing the +-0.10 band by chance, "
          f"not a systematic edge).")
    assert neutral_stats["n_trades"] < 30, \
        "Too many trades fired with no planted edge -- signal threshold may be too tight " \
        "relative to the realized-corr estimator's sampling noise"

    print("\nPipeline ran end-to-end over 5 years of synthetic data without error, and the "
          "signal direction tracked the planted correlation gap correctly in all three "
          "regimes. This validates the MECHANICS (data flow, signal timing, trade sizing, "
          "evaluation stats) -- it is NOT evidence the real strategy is profitable, since "
          "these trades aren't delta-hedged and the data is synthetic. That answer needs "
          "real historical IV data (WRDS pending, or a paid vendor in the meantime).")
