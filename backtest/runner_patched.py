# -*- coding: utf-8 -*-
"""
runner_patched.py — Strategy V2 wiring (Conviction Gate + EV + Rolling Balance + OFI)
- Dynamic ATR-based TP/SL (optional)
- EV gate supports 'probability' mode or 'bps multiple' mode
- Optional p_smoothing via EMA
- Debug artifacts: trades.csv, gating_debug.json, preds_test.csv, summary.json
"""
import os, sys, json, argparse, csv, math
from pathlib import Path
import yaml, pandas as pd, numpy as np
from backtest.strategy_v2 import (
    passes_conviction, ConvictionParams, ConvictionState,
    RollingBalance, RollingBalanceParams, approx_ofi, Frictions
)

# --- PATCH: timestamp auto-normalization (injected) ---------------------------
try:
    import pandas as _pd
    from pandas.api.types import is_numeric_dtype as _isnum
    _orig_read_csv = _pd.read_csv

    def _to_utc(series):
        s = series
        if _isnum(s):
            x = s.dropna().astype("int64")
            if len(x) > 0:
                v = int(x.iloc[0]); digits = len(str(abs(v))) if v != 0 else 1
                if digits >= 13:
                    return _pd.to_datetime(s, unit="ms", utc=True, errors="coerce")
                elif digits >= 10:
                    return _pd.to_datetime(s, unit="s", utc=True, errors="coerce")
        out = _pd.to_datetime(s, utc=True, errors="coerce")
        if out.notna().any():
            return out
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%d-%m-%Y %H:%M:%S"):
            try:
                return _pd.to_datetime(s, format=fmt, utc=True, errors="coerce")
            except Exception:
                pass
        return out

    def _ensure_timestamp(df):
        if "timestamp" in df.columns and df["timestamp"].notna().any():
            return df
        cands = [c for c in df.columns if str(c).lower() in (
            "timestamp", "datetime", "open_time", "time_open", "candle_begin_time", "ts", "t", "date", "time")]
        order = ["timestamp", "datetime", "open_time", "time_open", "candle_begin_time", "ts", "t", "date", "time"]
        cands.sort(key=lambda c: order.index(str(c).lower()) if str(c).lower() in order else 999)
        for c in cands:
            if str(c).lower() in ("date", "time"):
                continue
            ts = _to_utc(df[c])
            if ts.notna().any():
                df["timestamp"] = ts
                break
        if "timestamp" not in df.columns or df["timestamp"].isna().all():
            if "date" in df.columns and "time" in df.columns:
                combo = df["date"].astype(str) + " " + df["time"].astype(str)
                ts = _to_utc(combo)
                if ts.notna().any():
                    df["timestamp"] = ts
        if "timestamp" in df.columns:
            df.sort_values("timestamp", inplace=True, ignore_index=True)
            df.drop_duplicates(subset=["timestamp"], keep="last", inplace=True)
        return df

    def _patched_read_csv(*args, **kwargs):
        df = _orig_read_csv(*args, **kwargs)
        try:
            df = _ensure_timestamp(df)
        except Exception:
            pass
        return df

    if getattr(_pd.read_csv, "__name__", "") != "_patched_read_csv":
        _pd.read_csv = _patched_read_csv
except Exception:
    pass
# --- END PATCH ----------------------------------------------------------------


def load_yaml(p: Path):
    with open(p, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def true_range(h, l, pc):
    return max(h - l, abs(h - pc), abs(l - pc))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', required=True)
    ap.add_argument('--csv-glob', required=True)
    ap.add_argument('--params', required=True)
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--calibrator', default=None, help='optional isotonic calibrator JSON')
    args = ap.parse_args()

    repo_root = Path('.').resolve()
    spec = load_yaml(repo_root / 'specs' / 'strategy_v2_spec.yml')
    os.makedirs(args.outdir, exist_ok=True)

    params = load_yaml(Path(args.params))
    exits = spec.get('components', {}).get('exits', {})
    costs = spec.get('components', {}).get('costs', {})
    for k, v in exits.items():
        params.setdefault('exit', {}).setdefault(k, v)
    for k, v in costs.items():
        params.setdefault('meta', {}).setdefault(k, v if isinstance(v, (int, float)) else v)

    # Structure / OFI
    rbp = RollingBalanceParams(
        win=spec['components']['structure']['rolling_balance_avoid']['win_min'],
        tr_pctl_max=spec['components']['structure']['rolling_balance_avoid']['tr_pctl_max']
    )
    rb = RollingBalance(rbp)
    ofi_lb = spec['components']['orderflow']['ofi_align']['lookback']

    # Frictions
    fr = Frictions(
        fee_bps_per_side=params['meta'].get('fee_bps_per_side', 5),
        slippage_bps_per_side=params['meta'].get('slippage_bps_per_side', 2),
        funding_bps_estimate=params['meta'].get('funding_bps_estimate', 0.5)
    )
    frictions_bps = fr.per_roundtrip()

    # Conviction / EV gate config
    conv = spec['components']['gating']['conviction']
    ev_gate_cfg = conv.get('ev_gate', {})
    gate = ConvictionParams(
        m=conv['persistence']['m'],
        k=conv['persistence']['k'],
        thr_entry=conv['hysteresis']['thr_entry'],
        thr_exit=conv['hysteresis']['thr_exit'],
        alpha_cost=ev_gate_cfg.get('alpha_cost', 0.0)  # backward-compatible default
    )
    ev_mode = ev_gate_cfg.get('mode', 'bps')  # 'probability' | 'bps'
    ev_margin = ev_gate_cfg.get('ev_margin_bps', 0.0)

    state = ConvictionState()

    # Optional calibrator
    calibrator = None
    if args.calibrator and Path(args.calibrator).exists():
        try:
            with open(args.calibrator, 'r', encoding='utf-8') as f:
                cal = json.load(f)
            xs = np.array(cal.get('X_', cal.get('x', [])), dtype=float)
            ys = np.array(cal.get('y_', cal.get('y', [])), dtype=float)
            if len(xs) and len(ys):
                def _calib(x):
                    return np.interp(x, xs, ys, left=ys[0], right=ys[-1])
                calibrator = _calib
        except Exception:
            calibrator = None

    # Locate CSV
    import glob
    matches = []
    for p in glob.glob(str(Path(args.data_root) / '**' / '*.csv'), recursive=True):
        if glob.fnmatch.fnmatch(p, str(Path(args.data_root) / args.csv_glob)):
            matches.append(p)
    if not matches:
        raise SystemExit(f"No CSV matched: {args.csv_glob}")
    csv_path = matches[0]

    # Load & validate
    df = pd.read_csv(csv_path)
    req = ['open', 'high', 'low', 'close', 'volume']
    for col in req:
        if col not in df.columns:
            raise SystemExit(f"Missing column: {col}")
    tcol = 'timestamp' if 'timestamp' in df.columns else ('datetime' if 'datetime' in df.columns else None)
    if tcol is None:
        raise SystemExit("Need timestamp/datetime column")
    df[tcol] = pd.to_datetime(df[tcol], utc=True, errors='coerce')
    df = df.sort_values(tcol).reset_index(drop=True)

    # Signal: MACD proxy (until p_hat provided)
    fast = spec['components']['signal']['macd']['fast']
    slow = spec['components']['signal']['macd']['slow']
    signal = spec['components']['signal']['signal']
    ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    macd_diff = macd - macd_signal
    dir_hint = np.sign(macd_diff).astype(int)

    # Features
    ofi_list, tr_list, in_box_list, tr_pctl_list = [], [], [], []
    prev_close = float(df['close'].iloc[0])
    for h, l, c, o, v in zip(df['high'], df['low'], df['close'], df['open'], df['volume']):
        rb.update(float(h), float(l), float(prev_close))
        tr_val = true_range(float(h), float(l), float(prev_close))
        tr_list.append(tr_val)
        ofi_list.append(approx_ofi(float(o), float(h), float(l), float(c), float(v)))
        s = sorted(rb.buffer)
        if s:
            idx = max(0, min(len(s) - 1, (rbp.tr_pctl_max * len(s)) // 100))
            thr = s[idx]
            in_box = rb.buffer[-1] <= thr
            pctl = (np.searchsorted(s, rb.buffer[-1], side='right') / len(s)) * 100.0
        else:
            in_box, pctl = False, 100.0
        in_box_list.append(in_box)
        tr_pctl_list.append(pctl)
        prev_close = float(c)
    df['ofi'] = ofi_list
    df['tr'] = tr_list
    df['in_box'] = in_box_list
    df['tr_pctl'] = tr_pctl_list

    # Persistence
    m = gate.m
    aligned = (np.sign(macd_diff) == np.sign(pd.Series(macd_diff).shift(1))).astype(int)
    aligned = pd.Series(aligned).rolling(m).sum().fillna(0).astype(int)

    # Probability
    if 'p_hat' in df.columns:
        p_raw = df['p_hat'].astype(float).clip(0, 1).values
    else:
        macd_z = (macd_diff - pd.Series(macd_diff).rolling(100, min_periods=10).mean()) / \
                 (pd.Series(macd_diff).rolling(100, min_periods=10).std() + 1e-9)
        ofi_z = (df['ofi'] - pd.Series(df['ofi']).rolling(100, min_periods=10).mean()) / \
                (pd.Series(df['ofi']).rolling(100, min_periods=10).std() + 1e-9)
        score = np.tanh(macd_z.fillna(0) + ofi_z.fillna(0))
        p_raw = ((score - score.min()) / (score.max() - score.min() + 1e-9)).fillna(0.5).values

    if 'p_hat_calibrated' in df.columns:
        p_trend = df['p_hat_calibrated'].astype(float).clip(0, 1).values
    else:
        p_trend = p_raw
        if calibrator is not None:
            try:
                p_trend = np.clip(calibrator(p_raw), 0, 1)
            except Exception:
                p_trend = p_raw

    # Optional smoothing
    smoothing = spec['components']['signal'].get('smoothing', {})
    ema_span = int(smoothing.get('ema_span', 0) or 0)
    if ema_span > 0:
        p_trend = pd.Series(p_trend).ewm(span=ema_span, adjust=False).mean().clip(0, 1).values

    df['p_trend'] = p_trend

    # Regime
    atr_n = spec['components']['regime']['atr']['n']
    tr = pd.Series(df['tr']).rolling(atr_n).mean()
    z = (tr - tr.rolling(atr_n * 5, min_periods=atr_n).mean()) / (tr.rolling(atr_n * 5, min_periods=atr_n).std() + 1e-12)
    regime = np.where(z >= spec['components']['regime']['atr']['z_trend_min'], 'trend', 'range')

    # Gating thresholds
    thr_trend = spec['components']['gating']['calibration']['p_thr']['trend']
    thr_range = spec['components']['gating']['calibration']['p_thr']['range']

    # Exits / holds
    ex_mode = params['exit'].get('mode', 'fixed')  # 'fixed' | 'dynamic_atr'
    min_hold = params['exit'].get('min_hold', 8)
    max_hold = params['exit'].get('max_hold', 60)
    be_bps = params['exit'].get('breakeven_bps', 7)

    if ex_mode == 'dynamic_atr':
        atr_cfg = params['exit'].get('atr', {})
        atr_n_exit = int(atr_cfg.get('n', 14))
        use_ema = bool(atr_cfg.get('use_ema', True))
        tr_series = pd.Series(df['tr'])
        atr_series_exit = (tr_series.ewm(span=atr_n_exit, adjust=False).mean()
                           if use_ema else tr_series.rolling(atr_n_exit).mean())
        atr_bps = (atr_series_exit / df['close']).fillna(0) * 10000
        tp_mult = atr_cfg.get('tp_mult', {})
        sl_mult = atr_cfg.get('sl_mult', {})
        cap = atr_cfg.get('cap_bps', {})  # {tp_min,tp_max,sl_min,sl_max}
        loop_start = max(atr_n, m, atr_n_exit) + 1
    else:
        tp_bps = params['exit'].get('tp_bps', 38)
        sl_bps = params['exit'].get('sl_bps', 22)
        loop_start = max(atr_n, m) + 1

    position, entry_px, entry_idx = 0, 0.0, -1
    be_armed = False
    trades, gating_dbg = [], []
    tp_bps_cur = 0.0
    sl_bps_cur = 0.0

    def calc_ev(pop, tp, sl):
        return pop * tp - (1.0 - pop) * sl - frictions_bps

    for i in range(loop_start, len(df)):
        side = int(np.sign(dir_hint[i]))
        pop = float(p_trend[i])
        thr = thr_trend if regime[i] == 'trend' else thr_range
        mom_k = int(aligned.iloc[i])
        in_box = bool(df['in_box'].iloc[i])
        tr_pctl = float(df['tr_pctl'].iloc[i])
        ofi_ok = (int(np.sign(df['ofi'].iloc[max(0, i - ofi_lb):i].sum())) == side) if side != 0 else False

        # per-bar TP/SL (dynamic or fixed)
        if ex_mode == 'dynamic_atr':
            atr_val = float(atr_bps.iloc[i])
            tp_bps_i = float(np.clip(atr_val * float(tp_mult.get(regime[i], 1.0)),
                                     float(cap.get('tp_min', 0.0)), float(cap.get('tp_max', 1e9))))
            sl_bps_i = float(np.clip(atr_val * float(sl_mult.get(regime[i], 1.0)),
                                     float(cap.get('sl_min', 0.0)), float(cap.get('sl_max', 1e9))))
        else:
            tp_bps_i = float(tp_bps)
            sl_bps_i = float(sl_bps)

        ev_bps = calc_ev(pop, tp_bps_i, sl_bps_i)

        # EV gate
        if ev_mode == 'probability':
            p_ev_req = (sl_bps_i + frictions_bps + float(ev_margin)) / (tp_bps_i + sl_bps_i + 1e-9)
            passed_ev = (pop >= p_ev_req)
        else:  # 'bps' (multiple of frictions)
            p_ev_req = None
            passed_ev = (ev_bps >= gate.alpha_cost * frictions_bps)

        passed_calib = (pop >= thr)
        passed_persist = (mom_k >= gate.k)

        dbg = {
            "i": i, "side": side, "pop": pop, "thr": thr,
            "passed_calib": bool(passed_calib),
            "mom_k_of_m": mom_k,
            "passed_persist": bool(passed_persist),
            "ev_bps": ev_bps,
            "passed_ev": bool(passed_ev),
            "in_box": bool(in_box),
            "tr_pctl": tr_pctl,
            "ofi_ok": bool(ofi_ok),
            "tp_bps_i": tp_bps_i,
            "sl_bps_i": sl_bps_i
        }
        if p_ev_req is not None:
            dbg["p_ev_req"] = p_ev_req

        if position == 0:
            decision = passed_calib and passed_persist and passed_ev and (not in_box) and ofi_ok and side != 0
            if decision:
                position = side
                state.last_side = side
                entry_px = float(df['close'].iloc[i])
                entry_idx = i
                be_armed = False
                tp_bps_cur = tp_bps_i
                sl_bps_cur = sl_bps_i
                dbg["decision"] = "enter"
            else:
                dbg["decision"] = "reject"
        else:
            held = i - entry_idx
            pnl_bps = (float(df['close'].iloc[i]) / entry_px - 1.0) * 10000.0 * position
            if not be_armed and pnl_bps >= be_bps:
                be_armed = True
            if ex_mode == 'dynamic_atr':
                hit_tp = pnl_bps >= tp_bps_cur
                hit_sl = pnl_bps <= (0 if be_armed else -sl_bps_cur)
            else:
                hit_tp = pnl_bps >= tp_bps
                hit_sl = pnl_bps <= (0 if be_armed else -sl_bps)
            time_exit = (held >= max_hold)

            if hit_tp or hit_sl or time_exit or (held < min_hold and hit_sl):
                trades.append({
                    "entry_idx": entry_idx, "exit_idx": i, "side": position,
                    "entry_px": entry_px, "exit_px": float(df['close'].iloc[i]),
                    "pnl_bps": pnl_bps,
                    "tp_bps": tp_bps_cur if ex_mode == 'dynamic_atr' else tp_bps,
                    "sl_bps": sl_bps_cur if ex_mode == 'dynamic_atr' else sl_bps
                })
                position, entry_idx, entry_px = 0, -1, 0.0
                dbg["decision"] = "exit"
            else:
                dbg["decision"] = "hold"

        dbg["be_armed"] = bool(be_armed)
        dbg["regime"] = str(regime[i])
        gating_dbg.append(dbg)

    # Metrics
    actual = np.sign(df['close'].shift(-1) - df['close']).fillna(0).values
    pred = np.sign(p_trend - 0.5)
    tp_ = int(((pred == 1) & (actual == 1)).sum()); tn_ = int(((pred == -1) & (actual == -1)).sum())
    fp_ = int(((pred == 1) & (actual == -1)).sum()); fn_ = int(((pred == -1) & (actual == 1)).sum())
    denom = math.sqrt((tp_ + fp_) * (tp_ + fn_) * (tn_ + fp_) * (tn_ + fn_))
    mcc = float((tp_ * tn_ - fp_ * fn_) / denom) if denom > 0 else 0.0

    if trades:
        pnl = np.array([t["pnl_bps"] for t in trades])
        hit_rate = float((pnl > 0).mean()) if len(pnl) > 0 else 0.0
        summary = {"n_trades": int(len(trades)), "hit_rate": hit_rate, "mcc": mcc, "cum_pnl_bps": float(pnl.sum())}
    else:
        summary = {"n_trades": 0, "hit_rate": 0.0, "mcc": mcc, "cum_pnl_bps": 0.0}

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # trades
    with open(outdir / "trades.csv", "w", newline="", encoding="utf-8") as f:
        fn = list(trades[0].keys()) if trades else ["entry_idx", "exit_idx", "side", "entry_px", "exit_px", "pnl_bps"]
        w = csv.DictWriter(f, fieldnames=fn)
        w.writeheader()
        for t in trades:
            w.writerow(t)

    # gating debug
    with open(outdir / "gating_debug.json", "w", encoding="utf-8") as f:
        json.dump(gating_dbg, f, ensure_ascii=False, indent=2)

    # preds_test (audit)
    audit = pd.DataFrame({tcol: df[tcol], "p_trend": p_trend, "ofi": df["ofi"], "macd_hist": macd_diff})
    with open(outdir / "preds_test.csv", "w", encoding="utf-8", newline="") as f:
        audit.to_csv(f, index=False)

    # summary
    with open(outdir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
