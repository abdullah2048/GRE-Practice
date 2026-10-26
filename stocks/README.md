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
Input variables: today's / lagged returns, 5-day return, price vs 5- and 20-day
averages, RSI(14), 10-day volatility, volume vs average, KSE-100 1- and 5-day return.

Five sub-models are blended, re-weighted daily by recent accuracy:
- Adaptive regression (fast and slow): recursive least squares, updated with every error
- Pattern match: k-nearest past days, what happened next
- Trend: 10-day exponentially weighted average return
- No change: tomorrow = today (the baseline to beat)

Every backtest prediction uses only data available on that day, so the
backtest is an honest out-of-sample record. Live predictions are append-only.

## Add another stock
Add an entry to `predictor/config.py` (`TICKERS`). The dropdown picks it up.

## Run locally
    pip install -r stocks/requirements.txt
    python stocks/predictor/run.py          # needs internet access to dps.psx.com.pk
    python -m http.server -d stocks 8000    # open http://localhost:8000
