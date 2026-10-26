"""Download daily end-of-day prices.

Primary source is the Pakistan Stock Exchange data portal (dps.psx.com.pk).
Yahoo Finance (SYMBOL.KA) is used as a fallback if PSX is unreachable.
Both return a list of {"date", "close", "volume"} sorted oldest first.
"""

import json
import time
import urllib.request
from datetime import datetime, timedelta, timezone

PKT = timezone(timedelta(hours=5))  # Pakistan time, no DST
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"


def _get_json(url, retries=3):
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:  # network errors, bad JSON
            last = e
            time.sleep(2 ** attempt)
    raise RuntimeError(f"failed to fetch {url}: {last}")


def _to_date(ts):
    return datetime.fromtimestamp(int(ts), tz=PKT).strftime("%Y-%m-%d")


def _dedupe(rows):
    by_date = {}
    for r in rows:
        if r["close"] and r["close"] > 0:
            by_date[r["date"]] = r  # last one wins
    return [by_date[d] for d in sorted(by_date)]


def fetch_psx(symbol):
    # Response: {"status": 1, "data": [[unix_ts, close, volume, open], ...]} newest first.
    payload = _get_json(f"https://dps.psx.com.pk/timeseries/eod/{symbol}")
    data = payload.get("data") if isinstance(payload, dict) else payload
    if not data:
        raise RuntimeError(f"PSX returned no data for {symbol}")
    rows = []
    for row in data:
        rows.append({
            "date": _to_date(row[0]),
            "close": float(row[1]),
            "volume": float(row[2]) if len(row) > 2 and row[2] is not None else 0.0,
        })
    return _dedupe(rows)


def fetch_yahoo(symbol):
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}.KA?range=5y&interval=1d"
    payload = _get_json(url)
    res = payload["chart"]["result"][0]
    quote = res["indicators"]["quote"][0]
    rows = []
    for ts, c, v in zip(res["timestamp"], quote["close"], quote["volume"]):
        if c is None:
            continue
        rows.append({"date": _to_date(ts), "close": float(c), "volume": float(v or 0)})
    return _dedupe(rows)


def fetch_history(symbol):
    """Return (rows, source_name). Raises if every source fails."""
    errors = []
    for name, fn in (("PSX", fetch_psx), ("Yahoo Finance", fetch_yahoo)):
        try:
            rows = fn(symbol)
            if len(rows) >= 2:
                return rows, name
            errors.append(f"{name}: too few rows")
        except Exception as e:
            errors.append(f"{name}: {e}")
    raise RuntimeError("; ".join(errors))
