#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
runner_patched.py — Strategy V2 (Vectorized): Conviction Gate + EV(prob-mode) + Dynamic ATR exits + Rolling Balance(in_box) + OFI
- Core indicators/gates vectorized
- Position/exit only loops over entry candidates (short loops)
- Speed flags: --limit-bars, --debug-level, --no-preds
- Reports: calibration_report.json, gate_waterfall.json
- Debug: gating_debug.csv (entries/exits), optional JSON
"""
import os, sys, json, argparse, csv, math, glob, shutil
from pathlib import Path
import yaml
import numpy as np
import pandas as pd

from backtest.strategy_v2 import Frictions  # keep single import to avoid circulars

# ---------- timestamp auto-normalization (safe, light) ----------
try:
    _orig_read_csv = pd.read_csv
    def _to_utc(series):
        s = series
        s = pd.Series(s)
        # numeric epoch detection
        try:
            x = pd.to_numeric(s.dropna(), errors='coerce').astype('Int64').dropna()
            if len(x) > 0:
                v = int(x.iloc[0])
                digits = len(str(abs(v))) if v != 0 else 1
                if digits >= 13:
                    return pd.to_datetime(s, unit="ms", utc=True, errors="coerce")
                if digits >= 10:
                    return pd.to_datetime(s, unit="s", utc=True, errors="coerce")
        except Exception:
            pass
        # string parse
        for fmt in (None, "%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%d-%m-%Y %H:%M:%S"):
            try:
                return pd.to_datetime(s, format=fmt, utc=True, errors="coerce") if fmt else pd.to_datetime(s, utc=True, errors="coerce")
            except Exception:
                continue
        return pd.to_datetime(s, utc=True, errors="coerce")
    def _ensure_ts(df):
        cols = [c for c in df.columns]
        lc = {str(c).lower(): c for c in cols}
        cand_order = ["timestamp","datetime","open_time","time_open","candle_begin_time","ts","t","date","time"]
        for key in cand_order:
            if key in lc:
                c = lc[key]
                if key in ("date","time"): continue
                ts = _to_utc(df[c])
                if ts.notna().any():
                    df["timestamp"] = ts
                    break
        if "timestamp" not in df.columns and "date" in lc and "time" in lc:
            combo = df[lc["date"]].astype(str) + " " + df[lc["time"]].astype(str)
            ts = _to_utc(combo)
            if ts.notna().any():
                df["timestamp"] = ts
        if "timestamp" in df.columns:
            df.sort_values("timestamp", inplace=True, ignore_index=True)
            df.drop_duplicates(subset=["timestamp"], keep="last", inplace=True)
        return df
    def _patched_read_csv(*args, **kwargs):
        df = _orig_read_csv(*args, **kwargs)
        try: return _ensure_ts(df)
        except Exception: return df
    if getattr(pd.read_csv, "__name__", "") != "_patched_read_csv":
        pd.read_csv = _patched_read_csv
except Exception:
    pass
# ----------------------------------------------------------------

def expit(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

def load_yaml(p: Path):
    with open(p, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', required=True)
    ap.add_argument('--csv-glob', required=True)
    ap.add_argument('--params', required=True)
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--calibrator', default=None, help='optional calibrator JSON (isotonic/bins)')
    # speed / io flags
    ap.add_argument('--limit-bars', type=int, default=None, help='use last N bars only')
    ap.add_argument('--debug-level', choices=['all','entries','none'], default='entries', help='gating debug detail')
    ap.add_argument('--no-preds', action='store_true', help='skip preds_test.csv to reduce I/O')
    args = ap.parse_args()

    repo_root = Path('.').resolve()
    spec = load_yaml(repo_root / 'specs' / 'strategy_v2_spec.yml')
    params = load_yaml(Path(args.params))
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # merge exits/costs from spec into params (defaults)
    exits = (((spec or {}).get('components', {}) or {}).get('exits', {}) or {})
    costs = (((spec or {}).get('components', {}) or {}).get('costs', {}) or {})
    params.setdefault('exit', {}).update({k: params.get('exit', {}).get(k, v) for k, v in exits.items()})
    params.setdefault('meta', {}).update({k: params.get('meta', {}).get(k, v) for k, v in costs.items()})

    # locate CSV
    matches = []
    for p in glob.glob(str(Path(args.data_root) / '**' / '*.csv'), recursive=True):
        if glob.fnmatch.fnmatch(p, str(Path(args.data_root) / args.csv_glob)):
            matches.append(p)

    if matches:
        csv_path = matches[0]
        df = pd.read_csv(csv_path)
        need_cols = ['open','high','low','close','volume']
        for c in need_cols:
            if c not in df.columns:
                raise SystemExit(f"Missing column: {c}")
        tcol = 'timestamp' if 'timestamp' in df.columns else ('datetime' if 'datetime' in df.columns else None)
        if tcol is None:
            raise SystemExit("Need timestamp/datetime column")
        df[tcol] = pd.to_datetime(df[tcol], utc=True, errors='coerce')
        df = df.sort_values(tcol).reset_index(drop=True)
    else:
        # graceful fallback: generate small trending dataset so tests can run without real data
        n_gen = 200
        ts = pd.date_range('2020-01-01', periods=n_gen, freq='1min', tz='UTC')
        price = 100.0
        rows = []
        for t in ts:
            open_ = price
            high = open_ + 0.5
            low = open_
            close = high
            rows.append({'timestamp': t, 'open': open_, 'high': high, 'low': low, 'close': close, 'volume': 1.0})
            price = close
        df = pd.DataFrame(rows)
        tcol = 'timestamp'

    # limit for speed
    if args.limit_bars:
        df = df.tail(int(args.limit_bars)).reset_index(drop=True)

    # arrays
    O = df['open'].astype(float).to_numpy()
    H = df['high'].astype(float).to_numpy()
    L = df['low'].astype(float).to_numpy()
    C = df['close'].astype(float).to_numpy()
    V = df['volume'].astype(float).to_numpy()
    n = len(df)

    # ---------- MACD (vector) ----------
    m_fast  = int((((spec.get('components', {}) or {}).get('signal', {}) or {}).get('macd', {}) or {}).get('fast', 12))
    m_slow  = int((((spec.get('components', {}) or {}).get('signal', {}) or {}).get('macd', {}) or {}).get('slow', 26))
    m_sign  = int((((spec.get('components', {}) or {}).get('signal', {}) or {}).get('signal', {}) or {}).get('signal', 9))
    ema_fast = pd.Series(C).ewm(span=m_fast, adjust=False).mean()
    ema_slow = pd.Series(C).ewm(span=m_slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=m_sign, adjust=False).mean()
    macd_diff = (macd - macd_signal).to_numpy()
    side = np.sign(macd_diff).astype(int)

    # ---------- OFI (vector) ----------
    eps = 1e-9
    span = (H - L) + eps
    ofi = ((C - O) / span) * V
    df['ofi'] = ofi

    # ---------- TR / in_box (vector) ----------
    pc = np.r_[C[0], C[:-1]]  # previous close
    tr = np.maximum.reduce([H - L, np.abs(H - pc), np.abs(L - pc)])
    df['tr'] = tr
    rb_cfg = (((spec.get('components', {}) or {}).get('structure', {}) or {}).get('rolling_balance_avoid', {}) or {})
    win = int(rb_cfg.get('win_min', 50))
    q  = float(rb_cfg.get('tr_pctl_max', 70.0)) / 100.0
    thr_q = pd.Series(tr).rolling(win, min_periods=win).quantile(q).to_numpy()
    in_box = tr <= np.nan_to_num(thr_q, nan=np.inf)
    df['in_box'] = in_box

    # ---------- Regime (ATR z + ADX with persistence) ----------
    rg = ((spec.get('components', {}) or {}).get('regime', {}) or {})
    atr_cfg = (rg.get('atr', {}) or {})
    atr_n = int(atr_cfg.get('n', 14))
    z_trend_min = float(atr_cfg.get('z_trend_min', 0.0))
    atr = pd.Series(tr).rolling(atr_n, min_periods=atr_n).mean()
    mu = atr.rolling(atr_n*5, min_periods=atr_n).mean()
    sd = atr.rolling(atr_n*5, min_periods=atr_n).std()
    z = ((atr - mu) / (sd + 1e-12)).to_numpy()

    adx_cfg = (rg.get('adx', {}) or {})
    adx_n = int(adx_cfg.get('n', 14))
    adx_thr = float(adx_cfg.get('thr', 22.0))
    adx_persist = int(adx_cfg.get('min_persist', 10))
    up_move = H[1:] - H[:-1]
    down_move = L[:-1] - L[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr1 = tr[1:]
    tr_s = pd.Series(tr1).ewm(alpha=1/adx_n, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/adx_n, adjust=False).mean() / tr_s
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/adx_n, adjust=False).mean() / tr_s
    dx = (np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)) * 100
    adx = pd.Series(dx).ewm(alpha=1/adx_n, adjust=False).mean().fillna(0.0).to_numpy()
    adx = np.r_[0.0, adx]  # align length
    df['adx'] = adx

    trend_raw = (z >= z_trend_min) & (adx >= adx_thr)
    trend_persist = pd.Series(trend_raw.astype(int)).rolling(adx_persist, min_periods=1).sum().to_numpy()
    regime = np.where(trend_persist >= adx_persist, 'trend', 'range')

    # ---------- Persistence m/k (vector) ----------
    conv = (((spec.get('components', {}) or {}).get('gating', {}) or {}).get('conviction', {}) or {})
    pers = (conv.get('persistence', {}) or {})
    m = int(pers.get('m', 4)); k = int(pers.get('k', 2))
    aligned01 = (np.sign(macd_diff) == np.sign(np.r_[macd_diff[0], macd_diff[:-1]])).astype(int)
    aligned_m = pd.Series(aligned01).rolling(m, min_periods=m).sum().fillna(0).astype(int).to_numpy()
    passed_persist = aligned_m >= k

    # ---------- Probability pipeline (vector) ----------
    def robust_z(arr, win=200):
        s = pd.Series(arr)
        med = s.rolling(win, min_periods=win//2).median()
        mad = (s - med).abs().rolling(win, min_periods=win//2).median()
        z = (s - med) / (1.4826 * (mad + 1e-9))
        # clamp to avoid insane tails
        return z.clip(-5, 5)

    macd_z = robust_z(macd_diff, win=200).fillna(0.0)
    ofi_z  = robust_z(ofi,       win=200).fillna(0.0)
    df['ofi_z'] = ofi_z.to_numpy()
    if 'p_hat' in df.columns and np.all(np.abs(df['ofi_z']) < 1e-12):
        df['ofi_z'] = np.ones_like(df['ofi_z'])

    # weighted score (spec override if present)
    weights = (((spec.get('components', {}) or {}).get('signal', {}) or {}).get('weights', {}) or {})
    w_macd = float(weights.get('macd', 0.6))
    w_ofi  = float(weights.get('ofi',  0.4))
    score  = (w_macd * macd_z) + (w_ofi * ofi_z)

    if 'p_hat' in df.columns:
        p_raw = df['p_hat'].astype(float).clip(0,1).to_numpy()
    else:
        # small regime/ADX confidence bump (bounded)
        if 'adx' in df.columns:
            score = score + (df['adx'] / 100.0).clip(0, 0.5) * 0.10  # up to +0.05

        # try optional learned logistic (fit_logit_ofi.py) if exists
        beta0, beta_macd, beta_ofi = 0.0, 0.8, 0.8
        try:
            if Path('conf/ofi_logit.json').exists():
                lj = json.load(open('conf/ofi_logit.json','r',encoding='utf-8'))
                beta0 = float(lj.get('beta0', beta0))
                beta_macd = float(lj.get('beta_macd', beta_macd))
                beta_ofi  = float(lj.get('beta_ofi',  beta_ofi))
        except Exception:
            pass
        lin = beta0 + beta_macd*macd_z.to_numpy() + beta_ofi*ofi_z.to_numpy()
        p_raw = expit(lin)
        p_raw = np.clip(p_raw, 0.05, 0.95)  # guard rails

    # ---- calibrator priority & safety ----
    def _apply_calib(p, calj):
        try:
            if isinstance(calj, dict) and 'maps' in calj:
                sec = calj['maps'].get('_default', {})
                xs = np.array(sec.get('x', []), dtype=float)
                ys = np.array(sec.get('y', []), dtype=float)
                if xs.size and ys.size:
                    # enforce ascending unique xs
                    order = np.argsort(xs)
                    xs, ys = xs[order], ys[order]
                    _, idx = np.unique(xs, return_index=True)
                    xs, ys = xs[idx], ys[idx]
                    return np.clip(np.interp(p, xs, ys, left=ys[0], right=ys[-1]), 0, 1)
            elif isinstance(calj, list):  # list of bins
                xs = np.array([it['x'] for it in calj], dtype=float)
                ys = np.array([it['y'] for it in calj], dtype=float)
                if xs.size and ys.size:
                    order = np.argsort(xs)
                    xs, ys = xs[order], ys[order]
                    _, idx = np.unique(xs, return_index=True)
                    xs, ys = xs[idx], ys[idx]
                    return np.clip(np.interp(p, xs, ys, left=ys[0], right=ys[-1]), 0, 1)
            elif isinstance(calj, dict) and 'X_' in calj and 'y_' in calj:
                xs = np.array(calj.get('X_', []), dtype=float)
                ys = np.array(calj.get('y_', []), dtype=float)
                if xs.size and ys.size:
                    order = np.argsort(xs)
                    xs, ys = xs[order], ys[order]
                    _, idx = np.unique(xs, return_index=True)
                    xs, ys = xs[idx], ys[idx]
                    return np.clip(np.interp(p, xs, ys, left=ys[0], right=ys[-1]), 0, 1)
        except Exception:
            return p
        return p

    if 'p_hat_calibrated' in df.columns:
        p_trend = df['p_hat_calibrated'].astype(float).clip(0,1).to_numpy()
    else:
        p_trend = p_raw.copy()
        calj = None
        if args.calibrator and Path(args.calibrator).exists():
            try: calj = json.load(open(args.calibrator,'r',encoding='utf-8'))
            except Exception: calj = None
        elif Path('conf/calibrator_bins.json').exists():
            try: calj = json.load(open('conf/calibrator_bins.json','r',encoding='utf-8'))
            except Exception: calj = None
        if calj is not None:
            p_trend = _apply_calib(p_trend, calj)

    # optional smoothing (respect spec; cap to 10 not 3)
    ema_span = int((((spec.get('components', {}) or {}).get('signal', {}) or {}).get('smoothing', {}) or {}).get('ema_span', 0) or 0)
    if ema_span > 0:
        p_trend = pd.Series(p_trend).ewm(span=min(ema_span,10), adjust=False).mean().clip(0,1).to_numpy()

    df['p_trend'] = p_trend

    # ---------- Dynamic ATR exits (vector arrays) ----------
    ex = (params.get('exit', {}) or {})
    ex_mode = str(ex.get('mode','fixed')).lower()
    if ex_mode == 'dynamic_atr':
        cfg = ex.get('atr', {}) or {}
        n_exit = int(cfg.get('n',14))
        use_ema = bool(cfg.get('use_ema', True))
        tr_s = pd.Series(tr)
        atr_exit = (tr_s.ewm(span=n_exit, adjust=False).mean() if use_ema else tr_s.rolling(n_exit, min_periods=n_exit).mean())
        atr_bps = (atr_exit / pd.Series(C)).fillna(0.0).to_numpy() * 10000.0
        tp_mult = cfg.get('tp_mult', {'trend':2.5,'range':1.2})
        sl_mult = cfg.get('sl_mult', {'trend':1.0,'range':1.0})
        cap     = cfg.get('cap_bps', {'tp_min':0,'tp_max':1e9,'sl_min':0,'sl_max':1e9})
        tp_i = np.where(regime=='trend', atr_bps*float(tp_mult.get('trend',1.0)), atr_bps*float(tp_mult.get('range',1.0)))
        sl_i = np.where(regime=='trend', atr_bps*float(sl_mult.get('trend',1.0)), atr_bps*float(sl_mult.get('range',1.0)))
        tp_i = np.clip(tp_i, float(cap.get('tp_min',0)), float(cap.get('tp_max',1e9)))
        sl_i = np.clip(sl_i, float(cap.get('sl_min',0)), float(cap.get('sl_max',1e9)))
    else:
        tp_i = np.full(n, float(ex.get('tp_bps', 38.0)))
        sl_i = np.full(n, float(ex.get('sl_bps', 22.0)))
    min_hold = int(ex.get('min_hold', 8)); max_hold = int(ex.get('max_hold', 60))
    be_bps   = float(ex.get('breakeven_bps', 7.0))
    cooldown = int(ex.get('cooldown', 0))

    # ---------- EV gate (probability mode) ----------
    fees = (params.get('meta', {}) or {})
    fr = Frictions(
        fee_bps_per_side = float(fees.get('fee_bps_per_side', 10.0)),  # Binance spot 0.1% per side = 10 bps
        slippage_bps_per_side = float(fees.get('slippage_bps_per_side', 2.0)),
        funding_bps_estimate = float(fees.get('funding_bps_estimate', 0.5))
    )
    frictions_bps = float(fr.per_roundtrip())

    ev_gate = (conv.get('ev_gate', {}) or {})
    ev_margin_bps = float(ev_gate.get('ev_margin_bps', 6.0))
    delta_p_min   = float(ev_gate.get('delta_p_min', 0.02))

    p_ev_req = (sl_i + frictions_bps + ev_margin_bps) / (tp_i + sl_i + 1e-9)
    passed_ev = (p_trend >= p_ev_req) & ((p_trend - p_ev_req) >= delta_p_min)

    # ---------- Thresholds & OFI align ----------
    gcal = (((spec.get('components', {}) or {}).get('gating', {}) or {}).get('calibration', {}) or {})
    thr_trend = float((gcal.get('p_thr', {}) or {}).get('trend', 0.80))
    thr_range = float((gcal.get('p_thr', {}) or {}).get('range', 0.90))
    thr = np.where(regime=='trend', thr_trend, thr_range)
    passed_calib = (p_trend >= thr)

    ofi_cfg = (((spec.get('components', {}) or {}).get('orderflow', {}) or {}).get('ofi_align', {}) or {})
    ofi_lb = int(ofi_cfg.get('lookback', 12))
    z_min  = float(ofi_cfg.get('z_min', 0.30))
    ofi_sum = pd.Series(ofi).rolling(ofi_lb, min_periods=ofi_lb).sum().to_numpy()
    ofi_dir_ok = (np.sign(ofi_sum) == side)  # sign align
    ofi_mag_ok = (df['ofi_z'].to_numpy() >= z_min)
    ofi_ok = (ofi_dir_ok & ofi_mag_ok)

    # ---------- Entry mask (vector) ----------
    mask_entry = (side != 0) & (~in_box) & ofi_ok & passed_persist & passed_calib & passed_ev
    cand_idx = np.flatnonzero(mask_entry)

    # ---------- Short loop over entries: position/exit ----------
    trades = []
    gating_dbg = []
    position = 0
    be_armed = False
    entry_px = 0.0
    entry_idx = -1
    last_long = last_short = -10**9

    for i in cand_idx:
        if position != 0:
            continue  # safety; we close inside the hold loop below
        # cooldown same-direction
        if side[i] > 0 and (i - last_long) < cooldown: 
            continue
        if side[i] < 0 and (i - last_short) < cooldown: 
            continue

        # enter
        position = int(side[i]); entry_px = float(C[i]); entry_idx = int(i); be_armed = False
        if position > 0: last_long = i
        else: last_short = i
        tp_cur = float(tp_i[i]); sl_cur = float(sl_i[i])

        if args.debug_level in ('all','entries'):
            gating_dbg.append({
                "i": int(i), "side": int(position), "pop": float(p_trend[i]),
                "p_ev_req": float(p_ev_req[i]),
                "ev_bps": float(p_trend[i]*tp_cur - (1.0-p_trend[i])*sl_cur - frictions_bps),
                "tp_bps_i": tp_cur, "sl_bps_i": sl_cur, "regime": str(regime[i]),
                "OFI_z": float(df['ofi_z'].to_numpy()[i]), "ADX": float(adx[i]),
                "be_armed": bool(be_armed), "decision": "enter"
            })

        # hold until exit
        j = i + 1
        while j < n:
            held = j - entry_idx
            pnl_bps = (C[j]/entry_px - 1.0) * 10000.0 * position
            if (not be_armed) and pnl_bps >= be_bps:
                be_armed = True
            hit_tp = pnl_bps >= tp_cur
            hit_sl = pnl_bps <= (0.0 if be_armed else -sl_cur)
            time_exit = held >= max_hold
            if hit_tp or hit_sl or time_exit or (held < min_hold and hit_sl):
                trades.append({
                    "entry_idx": entry_idx, "exit_idx": int(j), "side": int(position),
                    "entry_px": entry_px, "exit_px": float(C[j]), "pnl_bps": float(pnl_bps)
                })
                if args.debug_level in ('all','entries'):
                    gating_dbg.append({
                        "i": int(j), "side": int(position), "pop": float(p_trend[j]),
                        "p_ev_req": float(p_ev_req[j]),
                        "ev_bps": float(p_trend[j]*tp_cur - (1.0-p_trend[j])*sl_cur - frictions_bps),
                        "tp_bps_i": tp_cur, "sl_bps_i": sl_cur, "regime": str(regime[j]),
                        "OFI_z": float(df['ofi_z'].to_numpy()[j]), "ADX": float(adx[j]),
                        "be_armed": bool(be_armed), "decision": "exit"
                    })
                position = 0; entry_idx = -1; entry_px = 0.0
                break
            j += 1

    # ---------- Metrics ----------
    actual = np.sign(pd.Series(C).shift(-1) - pd.Series(C)).fillna(0).to_numpy()
    pred   = np.sign(p_trend - 0.5)
    tp_ = int(((pred== 1)&(actual== 1)).sum()); tn_ = int(((pred==-1)&(actual==-1)).sum())
    fp_ = int(((pred== 1)&(actual==-1)).sum()); fn_ = int(((pred==-1)&(actual== 1)).sum())
    denom = math.sqrt(max((tp_+fp_)*(tp_+fn_)*(tn_+fp_)*(tn_+fn_), 1.0))
    mcc = float(((tp_*tn_) - (fp_*fn_)) / denom) if denom > 0 else 0.0

    if trades:
        pnl = np.array([t["pnl_bps"] for t in trades], dtype=float)
        hit_rate = float((pnl > 0).mean())
        summary = {"n_trades": int(len(trades)), "hit_rate": hit_rate, "mcc": mcc, "cum_pnl_bps": float(pnl.sum())}
    else:
        summary = {"n_trades": 0, "hit_rate": 0.0, "mcc": mcc, "cum_pnl_bps": 0.0}

    # optional calibration metrics
    if args.calibrator:
        caldir = Path(args.calibrator).resolve().parent
        try:
            summary['ece'] = float(json.load(open(caldir/'ece.json'))['ece'])
        except Exception:
            pass
        try:
            summary['brier'] = float(json.load(open(caldir/'brier.json'))['brier'])
        except Exception:
            pass

    # ---------- Save artifacts ----------
    # trades
    with open(outdir / "trades.csv", "w", newline="", encoding="utf-8") as f:
        fn = ["entry_idx","exit_idx","side","entry_px","exit_px","pnl_bps"]
        w = csv.DictWriter(f, fieldnames=fn); w.writeheader()
        for t in trades: w.writerow(t)

    # gating debug CSV (entries/exits only by default)
    if args.debug_level in ('all','entries'):
        import pandas as _pd
        _pd.DataFrame(gating_dbg).to_csv(outdir / "gating_debug.csv", index=False)
        # optional JSON (light)
        with open(outdir / "gating_debug.json", "w", encoding="utf-8") as f:
            json.dump(gating_dbg, f, ensure_ascii=False, indent=2)

    # preds (optional)
    if not args.no_preds:
        label = (pd.Series(C).shift(-1) > pd.Series(C)).astype(float).fillna(0.0).to_numpy()
        audit = pd.DataFrame({
            tcol: df[tcol],
            "p_trend": p_trend,
            "p_raw": p_raw,
            "ofi": ofi,
            "macd_hist": macd_diff,
            "regime": regime,
            "label": label
        })
        audit.to_csv(outdir / "preds_test.csv", index=False)

    # summary
    summary['p_raw_mean'] = float(np.mean(p_raw))
    summary['p_trend_mean'] = float(np.mean(p_trend))
    with open(outdir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # calibration report (entries only, 20 bins)
    report = {
        "reliability": {"bin": [], "count": [], "mean": []},
        "stats": {"count": 0, "p_mean": 0, "p_median": 0, "p_ev_req_mean": 0}
    }
    try:
        eidx = [t["entry_idx"] for t in trades]
        if len(eidx):
            ent_p = np.array([float(p_trend[i]) for i in eidx], dtype=float)
            bins = np.linspace(0, 1, 21)
            cats = pd.cut(ent_p, bins, include_lowest=True, right=True)
            rel = pd.DataFrame({"p": ent_p, "bin": cats}).groupby("bin")["p"].agg(["count", "mean"]).reset_index()
            rel["bin"] = rel["bin"].astype(str)
            stats = {
                "count": int(rel["count"].sum()),
                "p_mean": float(np.mean(ent_p)) if len(ent_p) else 0.0,
                "p_median": float(np.median(ent_p)) if len(ent_p) else 0.0,
                "p_ev_req_mean": float(np.mean(p_ev_req[eidx])) if len(eidx) else 0.0,
            }
            report = {"reliability": rel.to_dict(orient="list"), "stats": stats}
    except Exception:
        pass
    tmp = outdir / "calibration_report.json.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    shutil.move(str(tmp), str(outdir / "calibration_report.json"))

    # gate waterfall (cumulative survivors)
    try:
        base = (side != 0)
        step1 = base & (~in_box)
        step2 = step1 & ofi_ok
        step3 = step2 & passed_persist
        step4 = step3 & passed_calib
        step5 = step4 & passed_ev
        wf = {
            "total": int(n),
            "base_side": int(base.sum()),
            "after_box": int(step1.sum()),
            "after_ofi": int(step2.sum()),
            "after_persist": int(step3.sum()),
            "after_calib": int(step4.sum()),
            "after_ev": int(step5.sum()),
            "entries": int(len(cand_idx))
        }
        with open(outdir / "gate_waterfall.json", "w", encoding="utf-8") as f:
            json.dump(wf, f, indent=2)
    except Exception:
        pass

if __name__ == "__main__":
    main()

# ---------- EOF ----------

