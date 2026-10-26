"""Daily job: fetch prices, score yesterday's prediction, predict tomorrow.

    python stocks/predictor/run.py                 # live: download from PSX
    python stocks/predictor/run.py --csv FILE.csv  # offline: date,close,volume

Writes stocks/data/<SYMBOL>.json and stocks/data/tickers.json for the website.
Live predictions are append-only: once made, a prediction is never edited,
only filled in with the actual price when it arrives.
"""

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config  # noqa: E402
import model  # noqa: E402
from fetch import fetch_history  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")


def load_csv(path):
    with open(path) as f:
        return [
            {"date": r["date"], "close": float(r["close"]), "volume": float(r.get("volume") or 0)}
            for r in csv.DictReader(f)
        ]


def align_index(dates, index_rows):
    """Index close on each stock date, carrying the last known value forward."""
    if not index_rows:
        return None
    lookup = {r["date"]: r["close"] for r in index_rows}
    out, last = [], index_rows[0]["close"]
    for d in dates:
        last = lookup.get(d, last)
        out.append(last)
    return out


def resolve_live(live, rows):
    """Fill in actual prices for live predictions whose next session has now traded."""
    dates = [r["date"] for r in rows]
    for p in live:
        if p.get("actual") is not None:
            continue
        nxt = next((r for r in rows if r["date"] > p["based_on"]), None)
        if nxt is None:
            continue
        p["date"] = nxt["date"]
        p["actual"] = nxt["close"]
        p["error"] = round(p["pred"] - p["actual"], 4)
        p["error_pct"] = round(100 * (p["pred"] - p["actual"]) / p["actual"], 3)
        if p["actual"] != p["last_close"]:
            p["direction_hit"] = (p["pred"] > p["last_close"]) == (p["actual"] > p["last_close"])
        p["in_band"] = p["low"] <= p["actual"] <= p["high"]
    return dates


def process(ticker, rows, index_rows, source):
    symbol = ticker["symbol"]
    path = os.path.join(DATA_DIR, f"{symbol}.json")
    previous = {}
    if os.path.exists(path):
        with open(path) as f:
            previous = json.load(f)

    dates = [r["date"] for r in rows]
    close = [r["close"] for r in rows]
    volume = [r["volume"] for r in rows]
    backtest, forecast, state = model.run(
        dates, close, volume, align_index(dates, index_rows), config.WARMUP_DAYS
    )

    live = previous.get("live", [])
    resolve_live(live, rows)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    if forecast and not any(p["based_on"] == forecast["based_on"] for p in live):
        live.append({
            "made_at": now,
            "based_on": forecast["based_on"],
            "last_close": forecast["last_close"],
            "pred": forecast["pred"],
            "low": forecast["low"],
            "high": forecast["high"],
            "actual": None,
        })

    out = {
        "symbol": symbol,
        "name": ticker["name"],
        "exchange": ticker["exchange"],
        "currency": "PKR",
        "source": source,
        "updated_at": now,
        "last_date": dates[-1],
        "last_close": close[-1],
        "forecast": forecast,
        "state": state,
        "metrics": {
            "live": model.score(live),
            "backtest": model.score(backtest),
            "backtest_last30": model.score(backtest[-30:]),
        },
        "live": live,
        "backtest": backtest,
        "prices": [[r["date"], r["close"], r["volume"]] for r in rows],
    }
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, separators=(",", ":"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", help="offline price file for the first ticker (date,close,volume)")
    ap.add_argument("--index-csv", help="offline index file (date,close)")
    args = ap.parse_args()

    summary, failed = [], []
    index_cache = {}
    for ticker in config.TICKERS:
        try:
            if args.csv:
                rows, source = load_csv(args.csv), "CSV"
                index_rows = load_csv(args.index_csv) if args.index_csv else None
            else:
                rows, source = fetch_history(ticker["symbol"])
                idx = ticker.get("index")
                if idx and idx not in index_cache:
                    try:
                        index_cache[idx] = fetch_history(idx)[0]
                    except Exception as e:
                        print(f"warning: index {idx} unavailable ({e}); continuing without it")
                        index_cache[idx] = None
                index_rows = index_cache.get(idx)
            out = process(ticker, rows, index_rows, source)
        except Exception as e:
            print(f"{ticker['symbol']}: FAILED - {e}")
            failed.append(ticker["symbol"])
            continue
        f = out["forecast"] or {}
        m = out["metrics"]["backtest"] or {}
        print(
            f"{out['symbol']}: last {out['last_date']} close {out['last_close']} -> "
            f"next {f.get('pred')} ({f.get('pred_return_pct')}%); "
            f"backtest MAE {m.get('mae')} vs naive {m.get('naive_mae')}, "
            f"direction {m.get('direction_acc_pct')}%"
        )
        summary.append({"symbol": out["symbol"], "name": out["name"], "last_date": out["last_date"]})

    if summary:
        with open(os.path.join(DATA_DIR, "tickers.json"), "w") as f:
            json.dump(summary, f, indent=1)
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
