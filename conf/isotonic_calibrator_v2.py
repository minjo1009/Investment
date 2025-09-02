
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import sys
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

try:
    from sklearn.isotonic import IsotonicRegression
except Exception as e:
    print("ERROR: scikit-learn is required. pip install scikit-learn", file=sys.stderr)
    raise

# ---------- Utils ----------

def ensure_dir(p: str):
    d = os.path.dirname(p)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def read_csv_smart(path: str) -> pd.DataFrame:
    # Allow feather/parquet maybe later; for now CSV only
    return pd.read_csv(path)

def write_csv(df: pd.DataFrame, path: str):
    ensure_dir(path)
    df.to_csv(path, index=False)

def pct_change(a: np.ndarray, n: int = 1) -> np.ndarray:
    x = np.empty_like(a, dtype=float)
    x[:] = np.nan
    if n <= 0:
        return x
    x[n:] = (a[n:] - a[:-n]) / a[:-n]
    return x

def rolling_max(a: np.ndarray, n: int) -> np.ndarray:
    from collections import deque
    N=len(a); out=np.full(N, np.nan)
    dq=deque()
    for i,val in enumerate(a):
        while dq and dq[-1][0] <= val:
            dq.pop()
        dq.append((val,i))
        while dq and dq[0][1] <= i-n:
            dq.popleft()
        if i>=n-1:
            out[i] = dq[0][0]
    return out

def rolling_min(a: np.ndarray, n: int) -> np.ndarray:
    from collections import deque
    N=len(a); out=np.full(N, np.nan)
    dq=deque()
    for i,val in enumerate(a):
        while dq and dq[-1][0] >= val:
            dq.pop()
        dq.append((val,i))
        while dq and dq[0][1] <= i-n:
            dq.popleft()
        if i>=n-1:
            out[i] = dq[0][0]
    return out

def atr_like(high, low, close, n=14):
    # Simple ATR approximation
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close,1)), np.abs(low - np.roll(close,1))))
    tr[0] = high[0] - low[0]
    a = pd.Series(tr).rolling(n, min_periods=n).mean().values
    return a

def classify_regime(high, low, close, n_atr=14, z_win=200, z_thr=0.5):
    atr = atr_like(high, low, close, n_atr)
    atr_z = (pd.Series(atr).rolling(z_win, min_periods=z_win).apply(lambda s: 0 if s.std()==0 else (s.iloc[-1]-s.mean())/s.std(), raw=False)).values
    regime = np.where(atr_z >= z_thr, 'trend', 'range')
    return regime

def forward_label_tp_sl(close: np.ndarray, horizon: int, tp_bps: int, sl_bps: int) -> np.ndarray:
    # Label=1 if TP (up move) hits before SL (down move) within horizon, else 0. Ignore ties -> 0.
    N=len(close); lbl=np.zeros(N, dtype=int)
    for i in range(N-horizon):
        base = close[i]
        tp = base * (1 + tp_bps/10000.0)
        sl = base * (1 - sl_bps/10000.0)
        future = close[i+1:i+1+horizon]
        hit_tp = np.where(future >= tp)[0]
        hit_sl = np.where(future <= sl)[0]
        if hit_tp.size and (not hit_sl.size or hit_tp[0] < hit_sl[0]):
            lbl[i] = 1
        else:
            lbl[i] = 0
    lbl[-horizon:] = 0
    return lbl

def smart_join(left: pd.DataFrame, right: pd.DataFrame, on='ts'):
    if on in left.columns and on in right.columns:
        return left.merge(right[[on,'p_hat']], on=on, how='left')
    # fallback: align by index length
    m=min(len(left),len(right))
    left = left.iloc[:m].copy()
    left['p_hat'] = right['p_hat'].iloc[:m].values
    return left

# ---------- Core ----------

def make_samples_from_csv(csv_path: str, horizon: int, tp_bps: int, sl_bps: int, p_from: Optional[str]=None) -> pd.DataFrame:
    """Create calibration samples from a single OHLCV CSV. p_from can be a preds file to join p_hat; otherwise we use a proxy score."""
    df = read_csv_smart(csv_path)
    # tolerate column names
    cols = {c.lower(): c for c in df.columns}
    for need in ['open','high','low','close']:
        if need not in cols:
            raise ValueError(f"CSV {csv_path} missing column like {need}")
    O = df[cols['open']].values.astype(float)
    H = df[cols['high']].values.astype(float)
    L = df[cols['low']].values.astype(float)
    C = df[cols['close']].values.astype(float)
    # regime
    reg = classify_regime(H,L,C)
    # label
    y = forward_label_tp_sl(C, horizon=horizon, tp_bps=tp_bps, sl_bps=sl_bps)
    out = pd.DataFrame({
        'p_hat': np.nan,  # to be filled
        'label': y.astype(int),
        'regime': reg.astype(str)
    })
    # fill p_hat
    if p_from and os.path.exists(p_from):
        try:
            pr = read_csv_smart(p_from)
            if 'p_hat' not in pr.columns:
                raise ValueError("preds has no p_hat column")
            pr = pr.reset_index(drop=True)
            out = smart_join(out.reset_index(drop=True), pr.reset_index(drop=True))
        except Exception as e:
            print(f"WARNING: failed to join preds at {p_from}: {e}", file=sys.stderr)
    if out['p_hat'].isna().all():
        # simple proxy: scaled momentum + range position -> sigmoid
        r1 = pd.Series(C).pct_change(3).fillna(0).values
        r2 = pd.Series(C).pct_change(12).fillna(0).values
        box = (C - pd.Series(rolling_min(C, 50)).fillna(C).values) / (pd.Series(rolling_max(C, 50)).fillna(C).values - pd.Series(rolling_min(C, 50)).fillna(C).values + 1e-8)
        z = 4.0*r1 + 2.0*r2 + 1.0*(box-0.5)
        # sigmoid to [0,1]
        out['p_hat'] = 1/(1+np.exp(-z))
    # drop NA p_hat
    out = out.dropna(subset=['p_hat']).reset_index(drop=True)
    return out

def fit_isotonic(xs: np.ndarray, ys: np.ndarray):
    # xs in [0,1], ys in {0,1}
    xs = np.clip(xs, 1e-6, 1-1e-6)
    iso = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0)
    iso.fit(xs, ys)
    # Extract a piecewise-linear map sampled on unique points
    grid_x = np.unique(xs)
    grid_y = iso.predict(grid_x)
    return grid_x.tolist(), grid_y.tolist()

def fit_isotonic_by_regime(df: pd.DataFrame, regime_col='regime'):
    maps = {}
    if regime_col in df.columns:
        for g, gdf in df.groupby(regime_col):
            if len(gdf) >= 100:  # need enough points
                gx, gy = fit_isotonic(gdf['p_hat'].values, gdf['label'].values)
                maps[str(g)] = {'x': gx, 'y': gy, 'n': int(len(gdf))}
    # default/global
    gx, gy = fit_isotonic(df['p_hat'].values, df['label'].values)
    maps['_default'] = {'x': gx, 'y': gy, 'n': int(len(df))}
    return maps

def apply_map_one(p: float, mp: Dict[str, List[float]]):
    xs = mp['x']; ys = mp['y']
    # linear interp
    if p <= xs[0]: return ys[0]
    if p >= xs[-1]: return ys[-1]
    import bisect
    i = bisect.bisect_right(xs, p) - 1
    x1,x2 = xs[i], xs[i+1]
    y1,y2 = ys[i], ys[i+1]
    t = (p-x1)/(x2-x1+1e-12)
    return y1 + t*(y2-y1)

def apply_calibrator(df: pd.DataFrame, calib: Dict, regime_col='regime') -> pd.DataFrame:
    maps = calib['maps']
    def row_apply(r):
        p=float(r['p_hat'])
        if regime_col in df.columns and str(r[regime_col]) in maps:
            return apply_map_one(p, maps[str(r[regime_col])])
        else:
            return apply_map_one(p, maps['_default'])
    df = df.copy()
    df['p_hat_calibrated'] = df.apply(row_apply, axis=1)
    return df

# ---------- CLI ----------

def main():
    ap = argparse.ArgumentParser(description="Isotonic calibration utility (ResearchV2).")
    sub = ap.add_subparsers(dest='cmd', required=True)

    mk = sub.add_parser('make-samples', help="Create calibration samples from OHLCV CSVs")
    mk.add_argument('--data-root', default='data')
    mk.add_argument('--csv-glob', required=True)
    mk.add_argument('--horizon', type=int, default=30)
    mk.add_argument('--tp-bps', type=int, default=38)
    mk.add_argument('--sl-bps', type=int, default=22)
    mk.add_argument('--preds', default='', help="Optional path to preds CSV (with p_hat) to join")
    mk.add_argument('--out', default='out/calib_samples.csv')

    ft = sub.add_parser('fit', help="Fit isotonic calibrator from samples or preds")
    ft.add_argument('--samples', default='', help="CSV with columns p_hat,label[,regime]")
    ft.add_argument('--from-outdir', default='', help="Read preds_test.csv under this dir")
    ft.add_argument('--out', default='conf/calibrator_isotonic.json')

    aply = sub.add_parser('apply', help="Apply calibrator to preds CSV")
    aply.add_argument('--calibrator', required=True)
    aply.add_argument('--preds', required=True, help="CSV with column p_hat and optional regime")
    aply.add_argument('--out', default='out/preds_calibrated.csv')

    args = ap.parse_args()

    if args.cmd == 'make-samples':
        import glob
        import os
        pattern = os.path.join(args.data_root, args.csv_glob)
        files = glob.glob(pattern, recursive=True)
        if not files:
            print(f"ERROR: no CSVs match {pattern}", file=sys.stderr); sys.exit(1)
        frames = []
        for f in files:
            try:
                frames.append(make_samples_from_csv(f, args.horizon, args.tp_bps, args.sl_bps, p_from=args.preds or None))
            except Exception as e:
                print(f"WARNING: skip {f} due to {e}", file=sys.stderr)
        if not frames:
            print("ERROR: no samples produced", file=sys.stderr); sys.exit(1)
        out = pd.concat(frames, ignore_index=True)
        write_csv(out, args.out)
        print(f"[OK] wrote samples -> {args.out} (n={len(out)})")
        return

    if args.cmd == 'fit':
        df = None
        if args.samples and os.path.exists(args.samples):
            df = read_csv_smart(args.samples)
        elif args.from_outdir:
            p = os.path.join(args.from_outdir, 'preds_test.csv')
            if not os.path.exists(p):
                print(f"ERROR: {p} not found", file=sys.stderr); sys.exit(1)
            df = read_csv_smart(p)
            if 'label' not in df.columns:
                print("ERROR: preds_test.csv missing label column. Build samples first.", file=sys.stderr); sys.exit(1)
        else:
            print("ERROR: provide --samples or --from-outdir", file=sys.stderr); sys.exit(1)

        need = set(['p_hat','label'])
        if not need.issubset(df.columns):
            print("ERROR: input CSV must have columns: p_hat,label[,regime]", file=sys.stderr); sys.exit(1)

        maps = fit_isotonic_by_regime(df)
        calib = {
            'meta': {
                'created_utc': pd.Timestamp.utcnow().isoformat(),
                'n': int(len(df)),
                'cols': list(df.columns)
            },
            'maps': maps
        }
        ensure_dir(args.out)
        with open(args.out, 'w') as f:
            json.dump(calib, f, indent=2)
        print(f"[OK] wrote calibrator -> {args.out} (n={len(df)})")
        return

    if args.cmd == 'apply':
        with open(args.calibrator, 'r') as f:
            calib = json.load(f)
        df = read_csv_smart(args.preds)
        if 'p_hat' not in df.columns:
            print("ERROR: preds has no p_hat", file=sys.stderr); sys.exit(1)
        out = apply_calibrator(df, calib)
        write_csv(out, args.out)
        print(f"[OK] wrote calibrated preds -> {args.out} (n={len(out)})")
        return

if __name__ == '__main__':
    main()
