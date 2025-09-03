#!/usr/bin/env python3
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit


ap = argparse.ArgumentParser()
ap.add_argument('--csv', required=True, help='CSV containing MACD_z, OFI_z and label')
ap.add_argument('--out', required=True, help='output CSV with p_raw')
ap.add_argument('--coef-json', default=None, help='optional JSON path to save coefficients')
ap.add_argument('--n-splits', type=int, default=5)
args = ap.parse_args()

df = pd.read_csv(args.csv)
X = df[['MACD_z','OFI_z']].astype(float).to_numpy()
y = df['label'].astype(float).to_numpy()

spl = TimeSeriesSplit(n_splits=max(2,args.n_splits))
preds = np.zeros(len(df))
coefs = []
for tr, te in spl.split(X):
    lr = LogisticRegression(solver='lbfgs')
    lr.fit(X[tr], y[tr])
    preds[te] = lr.predict_proba(X[te])[:,1]
    coefs.append({'intercept': float(lr.intercept_[0]), 'coef': lr.coef_[0].tolist()})

df_out = df.copy()
df_out['p_raw'] = np.clip(preds,0,1)
df_out.to_csv(args.out, index=False)

if args.coef_json:
    json.dump(coefs, open(args.coef_json,'w'), ensure_ascii=False, indent=2)
