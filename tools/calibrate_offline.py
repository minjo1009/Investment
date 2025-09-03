#!/usr/bin/env python3
import argparse, json, os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import TimeSeriesSplit
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except Exception:
    matplotlib = None
    plt = None


def _metrics(p: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    brier = float(np.mean((p - y) ** 2))
    bins = np.linspace(0, 1, 11)
    idx = np.digitize(p, bins) - 1
    ece = 0.0
    n = len(p)
    for i in range(10):
        m = idx == i
        if m.any():
            acc = y[m].mean()
            conf = p[m].mean()
            ece += abs(acc - conf) * m.sum() / n
    return ece, brier


def _fit_platt(x: np.ndarray, y: np.ndarray):
    lr = LogisticRegression(solver='lbfgs')
    lr.fit(x.reshape(-1, 1), y)
    xs = np.linspace(0, 1, 101)
    ys = lr.predict_proba(xs.reshape(-1, 1))[:, 1]
    return xs, ys, lr


def _fit_isotonic(x: np.ndarray, y: np.ndarray):
    ir = IsotonicRegression(out_of_bounds='clip')
    ir.fit(x, y)
    xs = np.linspace(0, 1, 101)
    ys = ir.predict(xs)
    return xs, ys, ir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--preds', required=True, help='preds_test.csv from runner')
    ap.add_argument('--outdir', required=True, help='directory to save calibrator & metrics')
    args = ap.parse_args()

    df = pd.read_csv(args.preds)
    if 'p_raw' in df.columns:
        x = df['p_raw'].astype(float).to_numpy()
    elif 'p_trend' in df.columns:
        x = df['p_trend'].astype(float).to_numpy()
    else:
        raise SystemExit('preds csv must contain p_raw or p_trend')
    if 'label' in df.columns:
        y = df['label'].astype(float).to_numpy()
    else:
        raise SystemExit('preds csv must contain label column')

    if len(np.unique(y)) < 2:
        xs = np.linspace(0, 1, 101)
        ys = xs.copy()
        best = {'type': 'identity', 'X_': xs.tolist(), 'y_': ys.tolist()}
        best_ece, best_brier = _metrics(x, y)
    else:
        tscv = TimeSeriesSplit(n_splits=5)
        best = None
        best_ece, best_brier = 1e9, 1e9
        for name, fitter in {'platt': _fit_platt, 'isotonic': _fit_isotonic}.items():
            eces, briers = [], []
            for tr_idx, te_idx in tscv.split(x):
                y_tr, y_te = y[tr_idx], y[te_idx]
                if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
                    continue
                xs, ys, model = fitter(x[tr_idx], y_tr)
                if name == 'platt':
                    pred = model.predict_proba(x[te_idx].reshape(-1,1))[:,1]
                else:
                    pred = model.predict(x[te_idx])
                e, b = _metrics(pred, y_te)
                eces.append(e); briers.append(b)
            if not eces:
                continue
            ece = float(np.mean(eces)); brier = float(np.mean(briers))
            if (ece < best_ece) or (abs(ece - best_ece) < 1e-12 and brier < best_brier):
                xs, ys, _ = fitter(x, y)
                best = {'type': name, 'X_': xs.tolist(), 'y_': ys.tolist()}
                best_ece, best_brier = ece, brier
        if best is None:
            xs = np.linspace(0, 1, 101)
            ys = xs.copy()
            best = {'type': 'identity', 'X_': xs.tolist(), 'y_': ys.tolist()}
            best_ece, best_brier = _metrics(x, y)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    tmp = outdir / 'calibrator.json.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(best, f)
    os.replace(tmp, outdir / 'calibrator.json')

    with open(outdir / 'ece.json', 'w', encoding='utf-8') as f:
        json.dump({'ece': best_ece}, f)
    with open(outdir / 'brier.json', 'w', encoding='utf-8') as f:
        json.dump({'brier': best_brier}, f)

    # reliability plot
    bins = np.linspace(0, 1, 11)
    idx = np.digitize(x, bins) - 1
    xs_plot, ys_plot = [], []
    for i in range(10):
        m = idx == i
        if m.any():
            xs_plot.append(x[m].mean())
            ys_plot.append(y[m].mean())
    if plt is not None:
        plt.figure()
        if xs_plot:
            plt.plot(xs_plot, ys_plot, 'o-')
        plt.plot([0,1],[0,1],'k--')
        plt.xlabel('p')
        plt.ylabel('empirical')
        plt.tight_layout()
        plt.savefig(outdir / 'reliability.png')
    else:
        # placeholder if matplotlib missing
        (outdir / 'reliability.png').write_bytes(b'')

if __name__ == '__main__':
    main()
