#!/usr/bin/env python3
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from trend4u.calib import drift


def _ece(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.digitize(p, bins) - 1
    ece = 0.0
    n = len(p)
    for i in range(n_bins):
        m = idx == i
        if m.any():
            acc = y[m].mean()
            conf = p[m].mean()
            ece += abs(acc - conf) * m.sum() / n
    return float(ece)


def _mcc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    denom = np.sqrt(max((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), 1))
    return float(((tp * tn) - (fp * fn)) / denom) if denom > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True, help='runner output directory')
    args = ap.parse_args()
    out = Path(args.out)

    preds = pd.read_csv(out / 'preds_test.csv')
    y = preds['label'].astype(float).to_numpy()
    p = preds['p_trend'].astype(float).to_numpy()

    try:
        auc = roc_auc_score(y, p) if len(np.unique(y)) > 1 else float('nan')
    except Exception:
        auc = float('nan')
    mcc = _mcc(y, (p >= 0.5).astype(int))
    ece = _ece(p, y)
    psi = float('nan')
    if 'p_raw' in preds.columns:
        try:
            psi = drift.psi(preds['p_raw'].astype(float).to_numpy(), p)
        except Exception:
            psi = float('nan')
    key_hit_rate = 0.0
    wf_path = out / 'gate_waterfall.json'
    if wf_path.exists():
        try:
            wf = json.load(open(wf_path))
            key_hit_rate = float(wf.get('entries', 0)) / float(max(wf.get('total', 1), 1))
        except Exception:
            key_hit_rate = 0.0

    metrics = {
        'auc': auc,
        'mcc': mcc,
        'ece': ece,
        'psi': float(psi),
        'key_hit_rate': key_hit_rate,
    }
    with open(out / 'diag_metrics.json', 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
