"""Self-correcting next-day price model.

Every trading day is processed in order (walk-forward), exactly as it would
have happened live:

  1. Build input variables from data known at today's close.
  2. Each sub-model predicts tomorrow's return; the ensemble blends them.
  3. When tomorrow's actual close arrives, every sub-model is updated with
     its own error, and the ensemble re-weights toward whichever models have
     been most accurate recently.

Nothing ever sees the future, so the history of predictions doubles as an
honest out-of-sample backtest. Pure numpy; a few thousand days take well
under a second.
"""

import math

import numpy as np

FEATURES = [
    ("ret_1d", "Today's return"),
    ("ret_lag1", "Yesterday's return"),
    ("ret_lag2", "Return 2 days ago"),
    ("ret_5d", "5-day return"),
    ("gap_sma5", "Price vs 5-day average"),
    ("gap_sma20", "Price vs 20-day average"),
    ("rsi14", "RSI (14)"),
    ("vol10", "10-day volatility"),
    ("volume_ratio", "Volume vs 20-day average"),
    ("index_ret_1d", "KSE-100 today's return"),
    ("index_ret_5d", "KSE-100 5-day return"),
]
LOOKBACK = 20


def build_features(close, volume, index_close):
    """Return (X, r): X[t] uses only data up to and including day t; r[t] is day t's log return."""
    n = len(close)
    logc = np.log(close)
    r = np.zeros(n)
    r[1:] = np.diff(logc)
    ilog = np.log(index_close) if index_close is not None else None
    ir = np.zeros(n)
    if ilog is not None:
        ir[1:] = np.nan_to_num(np.diff(ilog))

    X = np.zeros((n, len(FEATURES)))
    for t in range(LOOKBACK, n):
        sma5 = close[t - 4:t + 1].mean()
        sma20 = close[t - 19:t + 1].mean()
        win = r[t - 13:t + 1]
        gain = win[win > 0].sum()
        loss = -win[win < 0].sum()
        rsi = 100.0 if loss == 0 else 100 - 100 / (1 + gain / loss)
        vavg = volume[t - 19:t + 1].mean()
        vratio = math.log((volume[t] + 1) / (vavg + 1))
        X[t] = [
            r[t],
            r[t - 1],
            r[t - 2],
            logc[t] - logc[t - 5],
            close[t] / sma5 - 1,
            close[t] / sma20 - 1,
            (rsi - 50) / 50,
            r[t - 9:t + 1].std(),
            vratio,
            ir[t],
            ir[t - 4:t + 1].sum(),
        ]
    return X, r


class Scaler:
    """Running mean/std (Welford) so features are standardised without lookahead."""

    def __init__(self, d):
        self.n = 0
        self.mean = np.zeros(d)
        self.m2 = np.zeros(d)

    def update(self, x):
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        self.m2 += delta * (x - self.mean)

    def transform(self, x):
        if self.n < 2:
            return np.zeros_like(x)
        std = np.sqrt(self.m2 / (self.n - 1))
        std[std < 1e-12] = 1.0
        return np.clip((x - self.mean) / std, -5, 5)


class RLS:
    """Linear regression updated online by recursive least squares.

    `forget` < 1 discounts old days so the model adapts to regime changes.
    """

    def __init__(self, d, forget, name, delta=10.0):
        self.name = name
        self.d = d + 1  # + intercept
        self.forget = forget
        self.delta = delta
        self.w = np.zeros(self.d)
        self.P = np.eye(self.d) * delta

    def _x(self, z):
        return np.append(z, 1.0)

    def predict(self, z):
        return float(self.w @ self._x(z))

    def update(self, z, y):
        x = self._x(z)
        Px = self.P @ x
        k = Px / (self.forget + x @ Px)
        self.w += k * (y - self.w @ x)
        self.P = (self.P - np.outer(k, Px)) / self.forget
        if not np.isfinite(self.P).all() or np.trace(self.P) > 1e6:
            self.P = np.eye(self.d) * self.delta  # keep numerically stable


class Analog:
    """k-nearest-neighbour: find past days that looked like today, average what came next."""

    def __init__(self, k, name):
        self.k = k
        self.name = name
        self.xs = []
        self.ys = []

    def predict(self, z):
        if len(self.ys) < self.k * 2:
            return 0.0
        xs = np.asarray(self.xs)
        d = np.sqrt(((xs - z) ** 2).sum(axis=1))
        idx = np.argpartition(d, self.k)[: self.k]
        w = 1.0 / (d[idx] + 1e-6)
        return float((np.asarray(self.ys)[idx] * w).sum() / w.sum())

    def update(self, z, y):
        self.xs.append(z.copy())
        self.ys.append(y)


class Drift:
    """Exponentially weighted average of recent daily returns (trend follower)."""

    def __init__(self, halflife, name):
        self.name = name
        self.alpha = 1 - 0.5 ** (1 / halflife)
        self.m = 0.0

    def predict(self, z):
        return self.m

    def update(self, z, y):
        self.m += self.alpha * (y - self.m)


class Naive:
    """Tomorrow = today. The baseline every real model must beat."""

    name = "No change (baseline)"

    def predict(self, z):
        return 0.0

    def update(self, z, y):
        pass


class Ensemble:
    """Blend sub-models with weights proportional to 1 / recent squared error."""

    def __init__(self, models, decay=0.9):
        self.models = models
        self.decay = decay
        self.mse = np.full(len(models), np.nan)
        self.err_var = None

    def weights(self):
        if np.isnan(self.mse).any():
            return np.full(len(self.models), 1 / len(self.models))
        inv = 1 / (self.mse + 1e-10)
        return inv / inv.sum()

    def predict(self, z):
        preds = np.array([m.predict(z) for m in self.models])
        return float(preds @ self.weights()), preds

    def update(self, z, y, preds, ens_pred):
        se = (preds - y) ** 2
        self.mse = np.where(np.isnan(self.mse), se, self.decay * self.mse + (1 - self.decay) * se)
        e2 = (ens_pred - y) ** 2
        self.err_var = e2 if self.err_var is None else 0.94 * self.err_var + 0.06 * e2
        for m in self.models:
            m.update(z, y)


def run(dates, close, volume, index_close, warmup):
    """Walk forward through history. Returns backtest rows, tomorrow's forecast and model state."""
    close = np.asarray(close, dtype=float)
    volume = np.asarray(volume, dtype=float)
    n = len(close)
    X, r = build_features(close, volume, index_close)
    d = X.shape[1]

    scaler = Scaler(d)
    main = RLS(d, 0.99, "Adaptive regression (fast)")
    models = [
        main,
        RLS(d, 0.998, "Adaptive regression (slow)"),
        Analog(10, "Pattern match (k-NN)"),
        Drift(10, "Trend (10-day)"),
        Naive(),
    ]
    ens = Ensemble(models)
    backtest = []
    forecast = None
    max_move = 0.075  # PSX circuit breaker is +-7.5% (or 10 rupees) per day

    for t in range(LOOKBACK, n):
        scaler.update(X[t])
        z = scaler.transform(X[t])
        pred_ret, preds = ens.predict(z)
        pred_ret = max(-max_move, min(max_move, pred_ret))
        sigma = math.sqrt(ens.err_var) if ens.err_var else float(np.std(r[1:t + 1]))

        if t + 1 < n:
            actual_ret = r[t + 1]
            if t >= warmup:
                backtest.append({
                    "based_on": dates[t],
                    "date": dates[t + 1],
                    "last_close": round(close[t], 4),
                    "pred": round(close[t] * math.exp(pred_ret), 4),
                    "low": round(close[t] * math.exp(pred_ret - 1.2816 * sigma), 4),
                    "high": round(close[t] * math.exp(pred_ret + 1.2816 * sigma), 4),
                    "actual": round(close[t + 1], 4),
                })
            ens.update(z, actual_ret, preds, pred_ret)
        elif t >= warmup:
            forecast = {
                "based_on": dates[t],
                "last_close": round(close[t], 4),
                "pred_return_pct": round(pred_ret * 100, 3),
                "pred": round(close[t] * math.exp(pred_ret), 4),
                "low": round(close[t] * math.exp(pred_ret - 1.2816 * sigma), 4),
                "high": round(close[t] * math.exp(pred_ret + 1.2816 * sigma), 4),
                "direction": "up" if pred_ret > 0 else "down" if pred_ret < 0 else "flat",
                "components": [
                    {"model": m.name, "pred_return_pct": round(p * 100, 3)}
                    for m, p in zip(models, preds)
                ],
            }

    weights = ens.weights()
    state = {
        "model_weights": [
            {"model": m.name, "weight": round(float(w), 4)} for m, w in zip(models, weights)
        ],
        # Standardised coefficients of the fast regression: sign and size of each
        # variable's current influence on tomorrow's return.
        "feature_influence": sorted(
            [
                {"feature": label, "key": key, "coef": round(float(c) * 100, 4)}
                for (key, label), c in zip(FEATURES, main.w[:-1])
            ],
            key=lambda f: -abs(f["coef"]),
        ),
    }
    return backtest, forecast, state


def score(rows):
    """Accuracy metrics for rows that have pred, actual and last_close."""
    rows = [r for r in rows if r.get("actual") is not None]
    if not rows:
        return None
    pred = np.array([r["pred"] for r in rows])
    act = np.array([r["actual"] for r in rows])
    last = np.array([r["last_close"] for r in rows])
    err = pred - act
    naive_mae = float(np.abs(last - act).mean())
    mae = float(np.abs(err).mean())
    moved = act != last
    hits = np.sign(pred[moved] - last[moved]) == np.sign(act[moved] - last[moved])
    in_band = [r["low"] <= r["actual"] <= r["high"] for r in rows if "low" in r]
    return {
        "n": len(rows),
        "mae": round(mae, 3),
        "mape_pct": round(float((np.abs(err) / act).mean() * 100), 3),
        "rmse": round(float(np.sqrt((err ** 2).mean())), 3),
        "direction_acc_pct": round(float(hits.mean() * 100), 1) if hits.size else None,
        "naive_mae": round(naive_mae, 3),
        # >0 means the model beats "tomorrow = today"
        "skill_vs_naive_pct": round((1 - mae / naive_mae) * 100, 2) if naive_mae else None,
        "band_coverage_pct": round(100 * sum(in_band) / len(in_band), 1) if in_band else None,
    }
