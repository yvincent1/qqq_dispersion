# QQQ / DIA Dispersion Trading Backtest

Tests whether implied-vs-realized correlation dispersion (short index
vol, long basket vol -- or the reverse) was historically profitable on
two baskets: QQQ's top-15 constituents and the real Dow 30 (DIA). Both
legs are genuine market data pulled from ThetaData (options) and
yfinance (equity prices/returns) -- no synthetic or proxy data feeds
into the real backtest runs.

## Structure

- `iv_solver.py` -- Black-Scholes pricing, vega, delta, and implied
  volatility solver (Brent's method).
- `correlation.py` -- implied correlation (from options IVs) and realized
  correlation (from historical returns), both weighted by index weight.
- `optimizer.py` -- basket selection (cumulative-weight cutoff + a
  liquidity floor) and vega-based sizing.
- `data/fetch.py` -- yfinance data pulls: real (non-dividend-adjusted)
  and dividend-adjusted price history, point-in-time basket weights for
  both QQQ (market-cap proxy) and DIA (real price-weighted, with real
  historical Dow membership reconstruction). **Requires live internet
  access.**
- `data/thetadata_fetch.py` / `data/thetadata_fetch_dia.py` -- the real
  historical ATM IV pipeline. Pulls banded per-day option quotes from
  ThetaData and solves IV via `iv_solver.implied_vol`. **Requires a
  `THETADATA_API_KEY` env var and a ThetaData Value-tier (options)
  subscription.** A full historical pull takes hours, not minutes (see
  that module's own docstring for why) -- run it locally, expect it to
  run for a while, and expect to resume it via its checkpoint if
  interrupted.
- `backtest.py` -- the actual backtest engine: scans historical dates,
  fires trades on a real spread threshold, and prices both an UN-hedged
  (entry/exit marks only) and a daily-DELTA-HEDGED P&L per trade.
- `backtest_real_run.py` / `backtest_real_run_dia.py` -- the real,
  end-to-end runs: load the pulled IV panel, apply data-quality/basket
  selection, build point-in-time weights, and run the backtest.
- `demo_pipeline.py` / `backtest_demo.py` -- synthetic-data sanity
  checks, useful for confirming the mechanics work without needing
  network access or a ThetaData subscription.

## Running it for real

1. Set `THETADATA_API_KEY` in your environment (ThetaData Value tier,
   options feed).
2. Pull the IV panels (each takes hours -- see the module docstrings for
   current runtime estimates and why):
   ```
   cd data
   python thetadata_fetch.py
   python thetadata_fetch_dia.py
   ```
   Both validate live against a real, short window before committing to
   the full historical pull, and checkpoint to disk after every ticker
   so an interruption (sleep, network drop) just resumes.
3. Run the backtests:
   ```
   cd ..
   python backtest_real_run.py
   python backtest_real_run_dia.py
   ```
   Each prints its own basket-selection reasoning (which names were
   dropped and why), then both UN-hedged and hedged P&L stats side by
   side.

## Methodology notes worth knowing before trusting the output

- **Spot source**: strike selection and IV solving use REAL (non-
  dividend-adjusted) historical closes, not yfinance's default adjusted
  price -- the adjusted price understates real historical spot for
  dividend payers (confirmed directly: VZ's real 2021-01-04 close was
  $58.85, the adjusted figure was $41.53), which was corrupting strike
  selection for every dividend-paying name in both baskets before this
  was found and fixed.
- **Strike selection is banded and per-day**, not a single fixed strike
  reused across a whole expiration window -- the earlier fixed-strike
  version had a real look-ahead problem (the strike was chosen using
  spot AS OF THE EXPIRATION DATE, i.e. future information relative to
  earlier readings in that window). See `thetadata_fetch.py`'s
  `pull_ticker_atm_iv_history` docstring for the full story.
- **Basket composition is point-in-time for DIA** (real historical Dow
  membership changes reconstructed from Wikipedia's change history, real
  historical price-weights) but only PARTIALLY point-in-time for QQQ:
  membership is held fixed at today's top-15 names (no free historical
  Nasdaq-100 weight/membership file exists, and reconstructing the true
  ~100-name index was scoped out as too large an undertaking for now) --
  only the WEIGHT of those 15 names varies by date, not which 15 they
  are. Every QQQ result should be read with that scope limit in mind.
- **`min_weight_coverage`** (in `run_backtest`) lets a trade fire on
  whatever subset of that date's basket actually has usable data,
  provided it's above a floor (default runs use 90%) -- an explicit,
  logged trade-off (`captured_weight` is in every trade record) instead
  of either requiring 100% coverage (which starved trade counts) or
  silently renormalizing over an arbitrary subset.
- **Delta hedging**: every trade reports both `pnl` (un-hedged, entry/
  exit marks only) and `hedged_pnl` (daily-rebalanced delta hedge
  against the real observed spot path, IV held fixed at entry for the
  hedge-delta calc). Compare the two before trusting either -- if they
  diverge a lot, direction was doing real work in the un-hedged number.

## Known limitations

- **Dividend yield is still `q=0` everywhere in the option pricer**
  (`iv_solver.py`'s `bs_price`/`bs_vega`/`implied_vol` all default to
  it and nothing overrides it). This is a smaller, related bias to the
  spot-source fix above -- both the IV solved from real market quotes
  and the straddle marks in the backtest assume no dividends. Fixing it
  properly requires touching the IV-solve step too (not just backtest
  pricing), which means another full re-pull -- not done yet.
- **No transaction-cost/execution realism beyond a flat bps assumption.**
  Fills are priced at bid/ask mid; a real book would cross the spread.
  The spread itself IS pulled (used as a data-quality filter) but not
  currently saved per-quote, so modeling real execution cost would need
  another data-layer change.
- **No margin, capital, or risk-limit modeling.** No max-drawdown check,
  no position-concentration cap, no stress scenario.
- **IV extraction uses only ATM calls**, not an average of call and put
  IV at the same strike (they should match via put-call parity;
  averaging would reduce bid-ask noise).
- **Sample sizes are still small** even after the fixes above increased
  trade counts meaningfully -- treat any single backtest run's stats as
  a directional read, not a statistically powered verdict, until many
  more genuinely independent trades have accumulated.

## Status

A trading-desk-style audit of this project (methodology, data integrity,
statistical rigor) found and this codebase has since fixed: basket
survivorship bias (DIA at full rigor, QQQ pragmatically scoped), strike-
selection look-ahead bias, a correlation-formula mismatch between
trimmed baskets and the real index IV, and the lack of delta-hedging.
Remaining known gaps are listed above. Full historical re-pulls
reflecting the strike-selection fix are the current long pole before the
next set of real backtest numbers is trustworthy.
