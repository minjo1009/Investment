#!/usr/bin/env python3
import argparse, json, os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


def _ece(p, y, bins=10):
    p = np.clip(np.asarray(p, dtype=float), 0, 1)
    y = np.asarray(y, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.digitize(p, edges, right=True) - 1
    ece = 0.0
    for b in range(bins):
        m = idx == b
        if not np.any(m):
            continue
        p_bin = p[m].mean()
        y_bin = y[m].mean()
        ece += abs(p_bin - y_bin) * (m.mean())
    return float(ece)


def _brier(p, y):
    p = np.clip(np.asarray(p, dtype=float), 0, 1)
    y = np.asarray(y, dtype=float)
    return float(np.mean((p - y) ** 2))


def _fit_platt(x, y):
    if len(np.unique(y)) < 2:
        class _Dummy:
            def __init__(self, v):
                self.v = float(v)
            def predict_proba(self, x):
                return np.column_stack([1 - self.v, np.full(len(x), self.v)])
        return _Dummy(np.mean(y))
    lr = LogisticRegression(solver="lbfgs")
    lr.fit(x.reshape(-1, 1), y)
    return lr


def _predict_platt(lr, x):
    if hasattr(lr, 'v'):
        return np.full(len(x), lr.v)
    return lr.predict_proba(x.reshape(-1, 1))[:, 1]


ap = argparse.ArgumentParser()
ap.add_argument("--preds", required=True)
ap.add_argument("--outdir", required=True)
ap.add_argument("--n-splits", type=int, default=5)
args = ap.parse_args()

outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(args.preds)
col_p = "p_trend" if "p_trend" in df.columns else df.columns[1]
col_y = "label" if "label" in df.columns else df.columns[-1]
p = df[col_p].astype(float).to_numpy()
y = df[col_y].astype(float).to_numpy()

# time-series CV
spl = TimeSeriesSplit(n_splits=max(2, args.n_splits))
metrics = {"platt": {"ece": [], "brier": []}, "isotonic": {"ece": [], "brier": []}}
for tr_idx, te_idx in spl.split(p):
    p_tr, p_te = p[tr_idx], p[te_idx]
    y_tr, y_te = y[tr_idx], y[te_idx]
    # Platt
    lr = _fit_platt(p_tr, y_tr)
    pred_platt = _predict_platt(lr, p_te)
    metrics["platt"]["ece"].append(_ece(pred_platt, y_te))
    metrics["platt"]["brier"].append(_brier(pred_platt, y_te))
    # Isotonic
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p_tr, y_tr)
    pred_iso = iso.predict(p_te)
    metrics["isotonic"]["ece"].append(_ece(pred_iso, y_te))
    metrics["isotonic"]["brier"].append(_brier(pred_iso, y_te))

avg_ece = {k: float(np.mean(v["ece"])) for k, v in metrics.items()}
avg_brier = {k: float(np.mean(v["brier"])) for k, v in metrics.items()}

chosen = min(avg_ece.items(), key=lambda kv: kv[1])[0]

# fit chosen on full data
xs = np.linspace(0, 1, 101)
if chosen == "platt":
    model = _fit_platt(p, y)
    ys = _predict_platt(model, xs)
else:
    model = IsotonicRegression(out_of_bounds="clip").fit(p, y)
    ys = model.predict(xs)

calib = {"maps": {"_default": {"x": xs.tolist(), "y": np.clip(ys, 0, 1).tolist()}}}

# atomic save
for name, obj in [("calibrator.json", calib), ("ece.json", {"platt": avg_ece["platt"], "isotonic": avg_ece["isotonic"], "chosen": chosen, "value": avg_ece[chosen]}), ("brier.json", {"platt": avg_brier["platt"], "isotonic": avg_brier["isotonic"], "chosen": chosen, "value": avg_brier[chosen]})]:
    tmp = outdir / (name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, outdir / name)

# reliability plot
bins = np.linspace(0,1,11)
idx = np.digitize(p, bins, right=True)-1
bin_centers = 0.5*(bins[:-1]+bins[1:])
obs = []
for b in range(10):
    m = idx==b
    obs.append(y[m].mean() if np.any(m) else 0)
if plt is not None:
    plt.figure(figsize=(4,4))
    plt.plot([0,1],[0,1],"k--")
    plt.plot(bin_centers, obs, marker="o")
    plt.xlabel("Predicted")
    plt.ylabel("Observed")
    plt.tight_layout()
    plt.savefig(outdir / "reliability.png")
    plt.close()
else:  # pragma: no cover
    open(outdir / "reliability.png", "wb").close()
