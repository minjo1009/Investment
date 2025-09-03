#!/usr/bin/env python3
"""Lightweight post-run diagnostics utilities.

This module expands the previous minimal metrics dump with additional
reporting required by research workflows:

* Calibration bin statistics (predicted vs. empirical frequency)
* Per-regime MCC scan over a range of probability thresholds
* Simple policy scale summary derived from ``gating_debug.json``

The intent is to keep the implementation dependency free (aside from
``pandas``/``numpy``/``sklearn`` already used elsewhere) and resilient to
missing columns.  Earlier revisions raised ``KeyError`` when the caller
attempted to operate on a non-existent ``regime`` column.  The new version
guards against that by synthesising a catch-all "all" regime when necessary.
"""

import argparse
import json
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
    if 'regime' not in preds.columns:
        preds['regime'] = 'all'
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

    # --- Calibration bins -------------------------------------------------
    diag_dir = out / 'diagnostics'
    diag_dir.mkdir(parents=True, exist_ok=True)
    bins = np.linspace(0, 1, 21)
    centers = (bins[:-1] + bins[1:]) / 2.0
    idx = np.digitize(p, bins) - 1
    rows = []
    for i, c in enumerate(centers):
        m = idx == i
        if m.any():
            rows.append({'bin': float(c), 'empirical': float(y[m].mean()), 'count': int(m.sum())})
    pd.DataFrame(rows).to_csv(diag_dir / 'calibration_bins.csv', index=False)

    # --- Regime threshold scan -------------------------------------------
    thrs = np.arange(0.50, 0.801, 0.01)
    scan_rows = []
    for reg, gdf in preds.groupby('regime'):
        yy = gdf['label'].to_numpy()
        pp = gdf['p_trend'].to_numpy()
        for thr in thrs:
            if len(yy):
                mcc_r = _mcc(yy, (pp >= thr).astype(int))
            else:
                mcc_r = float('nan')
            scan_rows.append({'regime': reg, 'p_thr': round(float(thr), 2), 'mcc': mcc_r})
    scan_df = pd.DataFrame(scan_rows)
    scan_df.to_csv(diag_dir / 'regime_threshold_scan.csv', index=False)
    best_thr = (scan_df.sort_values('mcc', ascending=False)
                        .groupby('regime', as_index=False)
                        .first())

    # --- Policy summary ---------------------------------------------------
    gd_path = out / 'gating_debug.json'
    pol_rows = []
    if gd_path.exists():
        try:
            gd = pd.DataFrame(json.load(open(gd_path)))
        except Exception:
            gd = pd.DataFrame()
    else:
        gd = pd.DataFrame()
    if gd.empty:
        gd = pd.DataFrame({'regime': ['all']})
        gd['decision'] = []
    if 'regime' not in gd.columns:
        gd['regime'] = 'all'
    for reg, gdf in gd.groupby('regime'):
        rec = {'regime': reg}
        if 'decision' in gdf.columns:
            for dec in ['enter', 'exit', 'hold']:
                rec[f'cnt_{dec}'] = int((gdf['decision'] == dec).sum())
        if 'tp_bps_i' in gdf.columns:
            rec['tp_mean'] = float(gdf['tp_bps_i'].dropna().mean()) if gdf['tp_bps_i'].notna().any() else float('nan')
        if 'sl_bps_i' in gdf.columns:
            rec['sl_mean'] = float(gdf['sl_bps_i'].dropna().mean()) if gdf['sl_bps_i'].notna().any() else float('nan')
        pol_rows.append(rec)
    pd.DataFrame(pol_rows).to_csv(diag_dir / 'policy_summary.csv', index=False)

    metrics = {
        'auc': auc,
        'mcc': mcc,
        'ece': ece,
        'psi': float(psi),
        'key_hit_rate': key_hit_rate,
        'best_p_thr': {row['regime']: row['p_thr'] for _, row in best_thr.iterrows()},
    }
    with open(diag_dir / 'diagnostics.json', 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
