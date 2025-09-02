import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression


def main():
    ap = argparse.ArgumentParser(description="Fit isotonic calibrator")
    ap.add_argument('--preds', required=True, help='CSV with p_raw and optional y or close columns')
    ap.add_argument('--horizon', type=int, default=5, help='label horizon in bars')
    ap.add_argument('--out', default='conf/calibrator_isotonic.json', help='output JSON path')
    args = ap.parse_args()

    df = pd.read_csv(args.preds)
    if 'p_raw' not in df.columns:
        raise SystemExit('need p_raw column')
    if 'y' not in df.columns:
        if 'close' in df.columns:
            df['y'] = (df['close'].shift(-args.horizon) > df['close']).astype(float)
        else:
            raise SystemExit('need y or close column')
    ir = IsotonicRegression(out_of_bounds='clip')
    X = df['p_raw'].astype(float).to_numpy()
    y = df['y'].astype(float).to_numpy()
    ir.fit(X, y)
    cal = {'X_': ir.X_.tolist(), 'y_': ir.y_.tolist()}
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(cal, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()

