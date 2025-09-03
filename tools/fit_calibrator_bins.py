#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json, argparse, os
import numpy as np
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--preds', required=True, help='preds_test.csv path')
    ap.add_argument('--out', default='conf/calibrator_bins.json')
    args = ap.parse_args()

    df = pd.read_csv(args.preds)
    if 'p_trend' not in df.columns:
        raise SystemExit("need column p_trend in preds csv")

    p = df['p_trend'].astype(float).clip(0,1).to_numpy()
    bins = np.linspace(0,1,21)
    centers = (bins[:-1] + bins[1:]) / 2.0

    # If label available (optional), compute empirical mean per bin
    y = None
    if 'label' in df.columns:
        y = df['label'].astype(float).to_numpy()
    if y is not None and len(y) == len(p):
        idx = np.digitize(p, bins) - 1
        means = []
        for b in range(20):
            mask = (idx == b)
            means.append(float(np.mean(y[mask])) if mask.any() else float(centers[b]))
    else:
        # conservative identity-like initial mapping
        means = centers.tolist()

    out = [{"x": float(x), "y": float(y)} for x, y in zip(centers, means)]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

if __name__ == "__main__":
    main()

