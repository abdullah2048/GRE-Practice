# PSX Price Predictor

Predicts the next-day closing price of PSX stocks (starting with **ENGROH**,
Engro Holdings), scores itself against the real close the next day, and
learns from its own errors.

## How it runs by itself
`.github/workflows/stock-predict.yml` runs every weekday at 5:30pm PKT
(after the PSX close). It:
1. Downloads full price history from the PSX data portal (Yahoo Finance fallback)
   plus the KSE-100 index.
2. Fills in the actual close for yesterday's prediction (error, direction hit, in-range).
3. Replays all history walk-forward, updating every model with each day's error,
   and writes tomorrow's forecast.
4. Commits `stocks/data/*.json`; the static page `stocks/index.html` reads it.

## The model (`predictor/model.py`, numpy only)
Several candidate models run side by side every day (champion/challenger):

| Candidate | Inputs | Sub-models |
|---|---|---|
| Original ensemble | returns, moving-average gaps, RSI, volatility, volume, KSE-100 | adaptive regressions, k-NN pattern match, trend, no-change |
| Shrunk regression + new variables | all of the above + 3-month return, 52-week range position, price-limit days, Monday flag | two ridge regressions, no-change |
| Strongly shrunk regression | same | one ridge regression |

The published forecast uses the champion. A challenger takes over only after
its average error over the trailing ~250 trading days is at least 1% lower.
Every candidate's live predictions are logged, so the leaderboard on the page
shows both backtest and live accuracy.

**To try an idea:** add an entry to `CANDIDATES`. It is scored on the same
days as the others and only takes over if it actually wins.

Every backtest prediction uses only data available on that day. Live
predictions are append-only.

## Add another stock
Add an entry to `predictor/config.py` (`TICKERS`). The dropdown picks it up.

## Run locally
    pip install -r stocks/requirements.txt
    python stocks/predictor/run.py          # needs internet access to dps.psx.com.pk
    python -m http.server -d stocks 8000    # open http://localhost:8000
