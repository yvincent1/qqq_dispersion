You're teaching me `qqq_dispersion`, a dispersion-trading research project, as if I'm a student who understands basic options concepts (calls, puts, implied vol) but has never seen a dispersion trade before. I'm giving you the full project narrative and the core code directly in this message — you don't need file access.

## How to teach it

1. **The trading idea first, no code.** Explain what a dispersion trade is and why it might make money: the market prices an index option's volatility partly based on how correlated its components are expected to be. If the options market is pricing in *more* correlation than components actually deliver, index options look relatively rich compared to single-name options — sell the index vol, buy the components' vol. Use a simple 2-3 stock example with made-up numbers before touching any real code.
2. **Then walk the code in the order data actually flows**: `iv_solver.py` (market price → implied vol) → `correlation.py` (implied and realized correlation, and how the spread becomes a signal) → `optimizer.py` (signal → sized trade) → `data/fetch.py` (where real numbers come from) → `backtest.py` (testing historically). For each, explain *why* before *how*.
3. **Point out the real design decisions and trade-offs**, not just what the code does — I've included the reasoning behind several below, use it.
4. **Check my understanding** with a short question after each major concept before moving on, like a tutor would, not a straight lecture.
5. Be upfront about what's a validated real-data result versus a known simplification — the project narrative below tells you which is which.

---

## Project narrative — what's real, what's been found

This project computes implied-vs-realized correlation for a stock basket against an index, generates a sell/buy-dispersion signal, and backtests it. It exists in two basket variants: **QQQ** (15-name tech-heavy basket, market-cap weighted, hardcoded table) and **DIA** (all 30 Dow members, price-weighted — the Dow's actual weighting scheme — computed live rather than hardcoded, since price-weights don't go stale the way cap-weights do).

**Data sources, in order of how the project evolved:**
1. **Live/yfinance** (`get_atm_iv`) — real-time only, no history. Used for the live signal (`run_live.py`).
2. **`proxy_iv.py`, abandoned** — tried approximating historical IV from realized vol scaled by a VXN-derived multiplier. v1 (one shared multiplier) turned out to be *mathematically invariant* under `implied_correlation()` — it canceled out completely, so it wasn't testing anything. v2 (per-name multiplier) fixed that but introduced look-ahead bias by calibrating from today's IV and projecting backward across years of history. Both are real, instructive dead ends — worth explaining to a student as an example of a plausible-looking approach that turned out to be silently wrong.
3. **ThetaData (real historical options data), current approach** — pulls real EOD bid/ask, solves IV via `iv_solver.py`, same math as the live path. No calibration, no look-ahead. This is what actually answers "is dispersion profitable" empirically, once enough history accumulates.

**Real bugs found and fixed while building the ThetaData pipeline** (good material for teaching that data pipelines fail in specific, findable ways, not just abstractly):
- Wrong parameter names vs. the library's actual (undocumented-correctly) signature.
- No `date` column in the response at all — had to derive trading date from a `created` timestamp field.
- `right` (call/put) values are the words `'CALL'`/`'PUT'`, not `'C'`/`'P'`.
- Strike must be passed as a *string*, not a float — confirmed by inspecting the protobuf schema directly (`TYPE_STRING`), not by guessing.
- No built-in timeout on the client's network calls — a real request hung indefinitely; fixed with a daemon-thread timeout wrapper (first attempt using `ThreadPoolExecutor` didn't actually work — its cleanup re-blocked on the same hung call — worth explaining as a genuine "the obvious fix doesn't fix it" lesson).
- **The most subtle one**: a resume/checkpoint bug where reloading a CSV gave a different index type (`Timestamp`) than a fresh pull (`datetime.date`) — `pandas.concat` silently treated the same calendar day as two different rows instead of erroring, corrupting the panel without any visible error. Caught by noticing a ticker's observation count had exactly doubled.
- **Also subtle**: a rollover-day bug — when the "correct" (~30-days-to-expiry) contract had no data for a specific day, the code fell back to whatever *was* available, including a 1-day-to-expiry contract with a 51% bid-ask spread, producing a nonsensical 372% annualized IV that got silently accepted. Fixed by adding a minimum-days-to-expiry floor and a max-spread filter — there was no data-quality floor at all before this was found.

**Accuracy validated, not just assumed:** cross-checked the historical panel against a live, independent pull. The index leg (QQQ) matched almost exactly. Two single-name gaps (AAPL, COST) were *investigated*, not dismissed — traced to real, verifiable spot-price movement during a volatile week, not data corruption. This is a good example for a student of the difference between "the numbers don't match, therefore it's broken" and actually running down *why* before concluding anything.

**Current honest status:** both QQQ and DIA panels are real, cleaned, historical ATM IV data (free-tier history, ~1 year, monthly-expiration sampling). Running the actual backtest on this data produces very few or zero trades — verified as a genuine consequence of small sample size (needing every basket ticker to have data on both a trade's entry AND exit date simultaneously is a hard constraint that compounds with basket size — DIA's 30 names produced *zero* usable candidate dates, QQQ's 15 produced 4, none clearing the signal threshold). This is a data-volume problem, not a broken pipeline — the math at every step has been individually verified against real, sane numbers.

---

## Core code

### `iv_solver.py` — Black-Scholes pricing + implied vol

```python
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
    Returns a value in (0, 1) for calls, (-1, 0) for puts.
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
        return np.nan
```

### `correlation.py` — implied and realized correlation, and the signal

```python
"""
Implied correlation (from options-implied vols) and realized correlation
(from historical returns), both properly weighted by index weights.

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
    """
    w = np.asarray(weights, dtype=float)
    sig = np.asarray(component_vols, dtype=float)
    w = w / w.sum()

    weighted_sum_sq = np.sum((w * sig) ** 2)
    weighted_sum_total = (np.sum(w * sig)) ** 2
    cross_term_capacity = weighted_sum_total - weighted_sum_sq

    index_var = index_vol ** 2

    if cross_term_capacity <= 0:
        return np.nan

    rho = (index_var - weighted_sum_sq) / cross_term_capacity
    return rho


def realized_correlation(returns_df, weights=None):
    """
    Weighted-average pairwise realized correlation from a DataFrame of
    daily returns (columns = tickers, rows = dates).
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
    signal. Returns: "sell_dispersion", "buy_dispersion", or "neutral".
    """
    spread = implied_corr - realized_corr
    if spread > threshold:
        return "sell_dispersion"
    elif spread < -threshold:
        return "buy_dispersion"
    return "neutral"
```

### `optimizer.py` — basket selection and vega sizing

```python
"""
Basket optimization for a dispersion trade.

1. SELECTION: given a universe of candidate names with weights, vols,
   and liquidity scores, choose a subset that captures enough index
   weight while respecting a max-names constraint and a liquidity floor.
2. SIZING: given a chosen basket, compute per-name straddle counts so
   the basket's total vega matches (offsets) a target index vega,
   weighted proportionally to each name's index weight.
"""

import numpy as np
import pandas as pd


def select_basket(constituents_df, target_cumulative_weight=0.70,
                   max_names=15, min_liquidity_score=0.0):
    """
    constituents_df: DataFrame with columns ['weight', 'liquidity_score']
                      indexed by ticker.
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
    Given a target total vega to offset and each name's per-straddle
    vega, compute how many straddles of each name to buy so total
    component vega matches the target, distributed proportionally to
    index weight.
    """
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    vegas = np.asarray(single_straddle_vegas, dtype=float)

    target_vega_per_name = index_vega_target * w
    counts = target_vega_per_name / vegas
    return counts
```

*(There's also `optimize_for_dispersion_edge()`, a scaffolded constrained reweighting optimizer the README itself calls a placeholder — mention it exists but don't over-index on it; it's explicitly flagged as not yet modeling real per-name P&L.)*

### `backtest.py` — the backtest engine (condensed)

```python
"""
Backtest engine for the dispersion signal.

Deliberately data-source agnostic: run_backtest() takes pre-built panels
of historical returns and implied vols and doesn't care whether they came
from a real vendor or a synthetic generator for validating the mechanics.

Trade construction: vega-sized, NOT delta-hedged. That means any single
trade's P&L carries real directional/gamma noise on top of the
correlation-gap edge this strategy is actually targeting -- only
aggregate stats over many trades are informative.
"""

def run_backtest(returns_panel, iv_panel, weights, index_ticker="QQQ",
                  lookback_days=63, holding_days=5, check_every=5,
                  spread_threshold=0.10, option_tenor_days=30,
                  r=0.045, n_index_straddles=10.0, cost_bps=5.0):
    """
    Walks forward through the date range. At each check (every
    check_every trading days), computes realized correlation over the
    trailing lookback_days and implied correlation from that day's IV
    panel. If the spread clears spread_threshold, simulates a
    vega-matched straddle pair (index vs. basket) held for holding_days,
    marks to market at exit, logs P&L net of a flat cost_bps assumption.
    Skips any date where any needed ticker is missing IV data --
    real vendor data will have gaps.
    """
    # ... walks dates, computes implied_correlation() and
    # realized_correlation() at each check point, calls
    # dispersion_signal() to classify, and _simulate_trade() to price
    # the actual position and mark it to market at exit.


def _simulate_trade(signal, entry_date, exit_date, prices, iv_panel, w,
                     tickers, index_ticker, option_tenor_days, r,
                     n_index_straddles, cost_bps):
    """
    Prices an index straddle vs. a vega-matched basket of straddles at
    entry (strikes fixed ATM-at-entry), marks both to market at exit
    using that date's IV and the realized spot move, returns net P&L.

    sell_dispersion = short index straddle, long basket straddles.
    buy_dispersion  = long index straddle, short basket straddles.
    """
    # Uses bs_price/bs_vega from iv_solver.py and size_basket_by_vega
    # from optimizer.py directly -- no new option math here.
```

*(Full file also includes `evaluate_trades()` — win rate, avg P&L, Sharpe, and a P&L-by-spread-magnitude-bucket breakdown to test whether bigger spreads predict bigger/more reliable P&L — and a set of deterministic self-tests using a "spot pinned at strike" synthetic price path to verify sign conventions without any direction/gamma noise confounding the check. Worth mentioning that these self-tests exist and what they verify, without necessarily reproducing all of them.)*

### `data/fetch.py` — real data, both baskets (excerpt)

```python
"""
Data fetching via yfinance. Requires live internet access.
"""

import yfinance as yf
import pandas as pd
import numpy as np

# QQQ top-15, market-cap weighted, HARDCODED (goes stale -- needs
# periodic manual updates as weights drift and QQQ rebalances quarterly)
QQQ_TOP_CONSTITUENTS = {
    "NVDA": 0.0850, "AAPL": 0.0699, "MSFT": 0.0575, "MU": 0.0461,
    "AMZN": 0.0444, "AMD": 0.0339, "GOOGL": 0.0314, "AVGO": 0.0309,
    "GOOG": 0.0292, "META": 0.0275, "TSLA": 0.0265, "WMT": 0.0243,
    "INTC": 0.0226, "CSCO": 0.0192, "COST": 0.0184,
}

# Dow -- all 30 members, PRICE-weighted (not market-cap). No hardcoded
# weights here -- computed fresh every call in get_dia_weights() below,
# since price-weights change with every price move and there's no
# "stale weight table" problem the way there is for QQQ's cap weights.
DIA_CONSTITUENTS = [
    "AAPL", "MSFT", "NVDA", "CRM", "CSCO", "IBM", "AMZN", "VZ",
    "JPM", "GS", "AXP", "V", "TRV", "UNH", "JNJ", "MRK", "AMGN",
    "BA", "CAT", "HON", "MMM", "WMT", "PG", "KO", "MCD", "NKE",
    "DIS", "HD", "CVX", "SHW",
]


def get_dia_weights(tickers=None):
    """
    Live price-weights: each member's weight is its own share price
    divided by the sum of all members' share prices.
    """
    if tickers is None:
        tickers = DIA_CONSTITUENTS
    prices = get_price_history(tickers, period="1d").iloc[-1]
    return prices / prices.sum()


def get_atm_iv(ticker, target_dte_days=30, r=0.045):
    """
    Pull the options chain for `ticker`, find the expiration closest to
    target_dte_days, and back out ATM implied vol from the call closest
    to the current spot price using iv_solver.implied_vol.
    """
    tk = yf.Ticker(ticker)
    spot = tk.history(period="1d")["Close"].iloc[-1]

    expirations = tk.options
    today = pd.Timestamp.today()
    dtes = [(pd.Timestamp(e) - today).days for e in expirations]
    best_idx = int(np.argmin([abs(d - target_dte_days) for d in dtes]))
    expiry = expirations[best_idx]
    actual_dte = dtes[best_idx]
    T = actual_dte / 365

    chain = tk.option_chain(expiry)
    calls = chain.calls
    calls["dist"] = (calls["strike"] - spot).abs()
    atm_row = calls.sort_values("dist").iloc[0]

    mid_price = (atm_row["bid"] + atm_row["ask"]) / 2
    if mid_price <= 0:
        mid_price = atm_row["lastPrice"]

    from iv_solver import implied_vol
    iv = implied_vol(mid_price, spot, atm_row["strike"], T, r, "call")
    return iv, actual_dte, expiry
```

---

That's everything you need. Start with the trading idea, no code, and check my understanding before moving on.
