#!/usr/bin/env python3
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True, help='CSV with macd_z, ofi_z, label')
    ap.add_argument('--out', required=True, help='JSON file to save coefficients')
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    cols = [c for c in df.columns if c.lower() in ('macd_z','ofi_z','label')]
    lc = {c.lower(): c for c in df.columns}
    X = df[[lc.get('macd_z'), lc.get('ofi_z')]].astype(float).to_numpy()
    y = df[lc.get('label')].astype(int).to_numpy()

    tscv = TimeSeriesSplit(n_splits=5)
    briers = []
    for tr, te in tscv.split(X):
        lr = LogisticRegression(solver='lbfgs')
        lr.fit(X[tr], y[tr])
        p = lr.predict_proba(X[te])[:,1]
        briers.append(float(np.mean((p - y[te])**2)))
    lr = LogisticRegression(solver='lbfgs')
    lr.fit(X, y)
    coefs = {
        'coef': lr.coef_[0].tolist(),
        'intercept': float(lr.intercept_[0]),
        'brier_cv': float(np.mean(briers))
    }
    Path(args.out).write_text(json.dumps(coefs, indent=2))

if __name__ == '__main__':
    main()
