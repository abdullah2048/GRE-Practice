"""Research: do overnight global moves predict ENGROH's next close?

US markets and oil trade after PSX closes (3:30pm PKT) and before it opens
(9:30am PKT). If their moves on day t help predict ENGROH from close t to
close t+1, a morning forecast could use them. Walk-forward test, no lookahead.
Data: PSX (psxdata) and FRED public CSVs (SP500, Brent, VIX).
"""

import csv
import io
import math
import os
import sys
import urllib.request

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "predictor"))
import model as M  # noqa: E402
from fetch import fetch_history  # noqa: E402


def fred(series):
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
    with urllib.request.urlopen(url, timeout=60) as resp:
        rows = list(csv.reader(io.StringIO(resp.read().decode())))
    out = {}
    for date, value in rows[1:]:
        try:
            out[date] = float(value)
        except ValueError:
            pass
    return out


def daily_returns(series):
    dates = sorted(series)
    return {d: math.log(series[d] / series[p]) for p, d in zip(dates, dates[1:])}


def latest_on_or_before(rets, sorted_dates, day):
    i = np.searchsorted(sorted_dates, day, side="right") - 1
    return rets[sorted_dates[i]] if i >= 0 else 0.0


def walk_forward(X, y, forget=0.999, rho=300):
    sc = M.Scaler(X.shape[1])
    m = M.Ridge(X.shape[1], forget, rho, "r")
    preds = np.zeros(len(y))
    for t in range(len(y)):
        sc.update(X[t])
        z = sc.transform(X[t])
        preds[t] = m.predict(z)
        m.update(z, y[t])
    return preds


def report(name, preds, y, start):
    p, a = preds[start:], y[start:]
    err = np.abs(np.exp(p) - np.exp(a)).mean()
    naive = np.abs(1 - np.exp(a)).mean()
    moved = a != 0
    hit = (np.sign(p[moved]) == np.sign(a[moved])).mean()
    print(f"{name:42s} error {err*100:.3f}%  no-change {naive*100:.3f}%  "
          f"skill {100*(1-err/naive):+6.2f}%  direction {100*hit:5.1f}%")


def main():
    rows, _ = fetch_history("ENGROH")
    dates = [r["date"] for r in rows]
    close = np.array([r["close"] for r in rows])
    volume = np.array([r["volume"] for r in rows])
    X0, r = M.build_features(close, volume, None, dates)

    ext = {}
    for key in ("SP500", "DCOILBRENTEU", "VIXCLS"):
        try:
            rets = daily_returns(fred(key))
            ext[key] = (rets, sorted(rets))
            print(f"{key}: {len(rets)} days")
        except Exception as e:
            print(f"{key}: unavailable ({e})")

    # Feature for PSX day t: the move on the latest foreign trading day <= t.
    # That foreign close happens after PSX closes on day t, so it is known
    # before PSX opens on day t+1 (a morning forecast), not at 3:45pm on day t.
    n = len(close)
    extra = np.zeros((n, len(ext)))
    for j, (rets, sd) in enumerate(ext.values()):
        for t in range(n):
            extra[t, j] = latest_on_or_before(rets, sd, dates[t])

    idx = np.arange(M.LOOKBACK, n - 1)
    y = r[idx + 1]
    base = X0[idx]
    start = 300
    print(f"\nTest days: {len(idx) - start} (ENGROH {dates[idx[start]]} to {dates[-1]})\n")
    report("Current variables (3:45pm forecast)", walk_forward(base, y), y, start)
    report("+ overnight US/oil/VIX (morning forecast)", walk_forward(np.c_[base, extra[idx]], y), y, start)
    report("Overnight variables only", walk_forward(extra[idx], y), y, start)
    for j, key in enumerate(ext):
        c = np.corrcoef(extra[idx, j], y)[0, 1]
        h = len(y) // 2
        c1 = np.corrcoef(extra[idx[:h], j], y[:h])[0, 1]
        c2 = np.corrcoef(extra[idx[h:], j], y[h:])[0, 1]
        print(f"corr({key} overnight, ENGROH next day) = {c:+.3f}  (1st half {c1:+.3f}, 2nd half {c2:+.3f})")


if __name__ == "__main__":
    main()
