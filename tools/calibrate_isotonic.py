import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression


def fit_isotonic(x, y):
  ir = IsotonicRegression(out_of_bounds='clip')
  ir.fit(x, y)
  xs = getattr(ir, 'X_thresholds_', getattr(ir, 'X_', None))
  ys = getattr(ir, 'y_thresholds_', getattr(ir, 'y_', None))
  return {'x': xs.tolist(), 'y': ys.tolist()}


def main():
  ap = argparse.ArgumentParser(description="Fit isotonic calibrator")
  ap.add_argument("--preds", required=True, help="CSV with p_raw and session,regime and label columns")
  ap.add_argument("--labels", help="optional labels CSV")
  ap.add_argument('--horizon', type=int, default=5, help='label horizon in bars')
  ap.add_argument('--out', default='conf/calibrator_isotonic.json', help='output JSON path')
  ap.add_argument('--group-keys', default='session,regime', help='comma separated column names')
  ap.add_argument('--min-bin', type=int, default=100)
  args = ap.parse_args()

  df = pd.read_csv(args.preds)
  if args.labels:
    _ = args.labels
  if 'p_raw' not in df.columns:
    raise SystemExit('need p_raw column')
  if 'label' in df.columns:
    df['y'] = df['label']
  if 'y' not in df.columns:
    if 'close' in df.columns:
      df['y'] = (df['close'].shift(-args.horizon) > df['close']).astype(float)
    else:
      raise SystemExit('need y or close column')
  all_preds = df['p_raw'].astype(float).to_numpy()
  all_labels = df['y'].astype(float).to_numpy()
  models = {}
  if args.group_keys:
    keys = [k.strip() for k in args.group_keys.split(',') if k.strip()]
    for key, g in df.groupby(keys):
      if len(g) < args.min_bin:
        continue
      k = key if isinstance(key, str) else '_'.join(map(str, key))
      models[k] = fit_isotonic(g['p_raw'].astype(float).to_numpy(), g['y'].astype(float).to_numpy())
  models['_default'] = models.get('_default', fit_isotonic(all_preds, all_labels))
  out_path = Path(args.out)
  out_path.parent.mkdir(parents=True, exist_ok=True)
  with open(out_path, 'w', encoding='utf-8') as f:
    json.dump({'maps': models}, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
  main()
