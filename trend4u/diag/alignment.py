#!/usr/bin/env python3
"""Alignment utility for ResearchV2 runs.

This script performs a simple isotonic calibration using ``gating_debug.json``
produced by a runner execution.  The first 70% of records (by index ``i``) are
used to fit the calibrator, which is then applied to the remaining 30% to
produce calibrated probabilities (``pop_cal``).  A per-regime MCC scan over
p-thresholds in [0.50, 0.80] is performed on the tail portion to determine
optimal entry thresholds.

Outputs:
  * calibrator_isotonic.pkl        - pickled scikit-learn IsotonicRegression
  * calibrated_tail.json           - tail records with added ``pop_cal``
  * alignment_threshold_scan.csv   - MCC scan table

Additionally ``conf/feature_flags.yml`` is updated so that ``entry.p_thr`` and
``ev.p_ev_req`` reflect the best thresholds for each regime.
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
import yaml

from trend4u.diag.v2_diagnostics import _mcc


def _ensure_regime(df: pd.DataFrame) -> pd.DataFrame:
    if 'regime' not in df.columns:
        df['regime'] = 'all'
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True, help='runner output directory')
    args = ap.parse_args()
    outdir = Path(args.out)

    gd_path = outdir / 'gating_debug.json'
    if not gd_path.exists():
        raise SystemExit(f"missing gating_debug.json at {gd_path}")

    data = pd.DataFrame(json.load(open(gd_path)))
    if data.empty:
        raise SystemExit('gating_debug.json has no records')
    data = _ensure_regime(data)
    if 'i' in data.columns:
        data.sort_values('i', inplace=True)
    else:
        data.reset_index(inplace=True)
        data.rename(columns={'index': 'i'}, inplace=True)

    mask = data['decision'] == 'enter'
    y = mask.astype(int).to_numpy()
    x = data['pop'].astype(float).to_numpy()

    n = len(data)
    split = int(n * 0.7)
    head_x, head_y = x[:split], y[:split]
    tail_x, tail_y = x[split:], y[split:]
    tail = data.iloc[split:].copy().reset_index(drop=True)

    iso = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0)
    iso.fit(head_x, head_y)

    # save calibrator
    with open(outdir / 'calibrator_isotonic.pkl', 'wb') as f:
        pickle.dump(iso, f)

    tail['pop_cal'] = iso.predict(tail_x)
    tail.to_json(outdir / 'calibrated_tail.json', orient='records', lines=False)

    thrs = np.arange(0.50, 0.801, 0.01)
    rows = []
    tail = _ensure_regime(tail)
    for reg, gdf in tail.groupby('regime'):
        yy = gdf['decision'].eq('enter').astype(int).to_numpy()
        pp = gdf['pop_cal'].to_numpy()
        for thr in thrs:
            mcc = _mcc(yy, (pp >= thr).astype(int)) if len(yy) else float('nan')
            rows.append({'regime': reg, 'p_thr': round(float(thr), 2), 'mcc': mcc})
    scan = pd.DataFrame(rows)
    scan.to_csv(outdir / 'alignment_threshold_scan.csv', index=False)

    best = (scan.sort_values('mcc', ascending=False)
                .groupby('regime', as_index=False)
                .first())

    # update feature_flags.yml
    ff_path = Path('conf/feature_flags.yml')
    flags = yaml.safe_load(open(ff_path))
    flags.setdefault('entry', {}).setdefault('p_thr', {})
    flags.setdefault('ev', {}).setdefault('p_ev_req', {})
    for _, row in best.iterrows():
        reg = row['regime']
        thr = float(row['p_thr'])
        flags['entry']['p_thr'][reg] = thr
        flags['ev']['p_ev_req'][reg] = thr
    with open(ff_path, 'w') as f:
        yaml.safe_dump(flags, f)

    print(json.dumps({r['regime']: r['p_thr'] for r in best.to_dict('records')}, indent=2))


if __name__ == '__main__':
    main()

