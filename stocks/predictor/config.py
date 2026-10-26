"""Which stocks to track. Add a ticker here and the daily job picks it up."""

TICKERS = [
    {
        "symbol": "ENGROH",
        "name": "Engro Holdings Limited",
        "exchange": "PSX",
        # Market index used as an extra input variable.
        "index": "KSE100",
    },
]

# Days of history needed before the model starts making predictions.
WARMUP_DAYS = 30
