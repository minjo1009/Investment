# -*- coding: utf-8 -*-
"""runner_patched.py — Strategy V2 wiring (OFI soft gate + dyn TP/SL)"""
import os, sys, json, argparse, csv, math
from pathlib import Path
import yaml, pandas as pd, numpy as np

from backtest.strategy_v2.conviction import passes_conviction, ConvictionParams, ConvictionState
from backtest.strategy_v2.filters import ofi_conf_alignment, soft_gate_adjustments, _zscore, ensure_ofi_columns
from backtest.strategy_v2.indicators import atr_1m, dyn_tp_sl_bps
from backtest.strategy_v2.structure import prior_day_levels, value_area_approx
from backtest.strategy_v2.costs import Frictions


def expit(x):
  import numpy as _np
  return 1.0 / (1.0 + _np.exp(-x))

# --- PATCH: timestamp auto-normalization (retained) ---------------------------
try:
  import pandas as _pd
  import numpy as _np
  from pandas.api.types import is_numeric_dtype as _isnum
  _orig_read_csv = _pd.read_csv
  def _to_utc(series):
    s = series
    if _isnum(s):
      x = s.dropna().astype("int64")
      if len(x) > 0:
        v = int(x.iloc[0]); digits = len(str(abs(v))) if v!=0 else 1
        if digits >= 13: return _pd.to_datetime(s, unit="ms", utc=True, errors="coerce")
        elif digits >= 10: return _pd.to_datetime(s, unit="s", utc=True, errors="coerce")
    out = _pd.to_datetime(s, utc=True, errors="coerce")
    if out.notna().any(): return out
    for fmt in ("%Y-%m-%d %H:%M:%S","%Y/%m/%d %H:%M:%S","%Y-%m-%dT%H:%M:%S","%d-%m-%Y %H:%M:%S"):
      try: return _pd.to_datetime(s, format=fmt, utc=True, errors="coerce")
      except Exception: pass
    return out
  def _ensure_timestamp(df):
    if "timestamp" in df.columns and df["timestamp"].notna().any():
      return df
    cands = [c for c in df.columns if str(c).lower() in ("timestamp","datetime","open_time","time_open","candle_begin_time","ts","t","date","time")]
    order = ["timestamp","datetime","open_time","time_open","candle_begin_time","ts","t","date","time"]
    cands.sort(key=lambda c: order.index(str(c).lower()) if str(c).lower() in order else 999)
    for c in cands:
      if str(c).lower() in ("date","time"):
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
    try: df = _ensure_timestamp(df)
    except Exception: pass
    return df
  if getattr(_pd.read_csv, "__name__", "") != "_patched_read_csv":
    _pd.read_csv = _patched_read_csv
except Exception:
  pass
# --- END PATCH ----------------------------------------------------------------


def load_yaml(p):
  with open(p, 'r', encoding='utf-8') as f:
    return yaml.safe_load(f)


def true_range(h, l, pc):
  return max(h - l, abs(h - pc), abs(l - pc))


def tag_session(ts_utc) -> str:
  """Return trading session name for a UTC timestamp.

  The input ``ts_utc`` may be extremely permissive: a scalar timestamp,
  sequence/array/Series of timestamps, string, integer epoch (in seconds or
  milliseconds) or even ``pandas.Timestamp`` without timezone information.
  Whatever the input, this helper makes a best effort to coerce it into a
  timezone-aware ``pandas.Timestamp``.  If coercion fails we fall back to the
  default session ``"US"`` instead of raising an exception.
  """

  try:
    import pandas as _pd
    import numpy as _np

    x = ts_utc
    # If an array/Series/list is passed, grab the first element
    if isinstance(x, (list, tuple, set, _np.ndarray)):
      x = next(iter(x), _pd.NaT)
    elif isinstance(x, _pd.Series):
      x = x.iloc[0] if not x.empty else _pd.NaT

    # Handle numeric epochs (seconds or milliseconds)
    if isinstance(x, (int, float)) and not isinstance(x, bool):
      v = int(x)
      digits = len(str(abs(v))) if v != 0 else 1
      if digits >= 13:
        ts = _pd.to_datetime(v, unit="ms", utc=True, errors="coerce")
      elif digits >= 10:
        ts = _pd.to_datetime(v, unit="s", utc=True, errors="coerce")
      else:
        ts = _pd.to_datetime(v, utc=True, errors="coerce")
    else:
      # to_datetime will localise to UTC when ``utc=True``; for naive
      # timestamps this effectively assumes they are UTC.
      ts = _pd.to_datetime(x, utc=True, errors="coerce")

    if ts is _pd.NaT or _pd.isna(ts):
      return "US"
    h = int(ts.hour)
  except Exception:
    return "US"

  if 0 <= h < 8:
    return "ASIA"
  if 8 <= h < 16:
    return "EU"
  return "US"


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--data-root', required=True)
  ap.add_argument('--csv-glob', required=True)
  ap.add_argument('--params', required=True)
  ap.add_argument('--outdir', required=True)
  ap.add_argument('--calibrator', default=None)
  args = ap.parse_args()

  params = load_yaml(args.params)

  paths = sorted(Path(args.data_root).glob(args.csv_glob))
  if not paths:
    raise SystemExit('no data files')
  df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
  df.rename(columns={'timestamp': 'open_time'}, inplace=True)
  df = ensure_ofi_columns(df)

  # MACD
  fast, slow, sig = 12, 26, 9
  ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
  ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
  macd = ema_fast - ema_slow
  macd_sig = macd.ewm(span=sig, adjust=False).mean()
  macd_hist = macd - macd_sig
  df['macd_hist'] = macd_hist
  dir_hint = np.sign(macd_hist).astype(int)

  # Rolling balance
  rb_win = params['regime']['detector']['donchian_len']
  rb_buf = []
  in_box_list, tr_pctl_list, tr_list = [], [], []
  prev_close = float(df['close'].iloc[0])
  for h, l, c in zip(df['high'], df['low'], df['close']):
    tr = true_range(float(h), float(l), prev_close)
    rb_buf.append(tr)
    if len(rb_buf) > rb_win:
      rb_buf.pop(0)
    s = sorted(rb_buf)
    if s:
      idx = int(0.25 * len(s))
      thr = s[idx]
      in_box = tr <= thr
      pctl = np.searchsorted(s, tr, side='right') / len(s) * 100.0
    else:
      in_box, pctl = False, 100.0
    in_box_list.append(in_box)
    tr_pctl_list.append(pctl)
    tr_list.append(tr)
    prev_close = float(c)
  df['tr'] = tr_list
  df['in_box'] = in_box_list
  df['tr_pctl'] = tr_pctl_list

  # OFI features
  ofi_feats = ofi_conf_alignment(df,
                                params['entry']['ofi']['ema_len'],
                                params['entry']['ofi']['align_window'],
                                params['entry']['ofi']['align_ge'])
  df = pd.concat([df, ofi_feats], axis=1)
  ofi_z = _zscore(ofi_feats['OFI_smooth']).fillna(0.0)

  # Probability model
  weights = {'macd': 0.6, 'ofi': 0.4}
  score = weights['macd'] * _zscore(macd_hist).fillna(0.0) + weights['ofi'] * ofi_z
  p_raw_arr = expit(0.8 * score + 0.0)
  df['p_raw'] = p_raw_arr

  # Regime detection
  atr_n = params['regime']['detector']['atr_len']
  tr = pd.Series(df['tr']).rolling(atr_n).mean()
  z = (tr - tr.rolling(atr_n*5, min_periods=atr_n).mean()) / (tr.rolling(atr_n*5, min_periods=atr_n).std() + 1e-9)
  regime = np.where(z >= 0.5, 'trend', 'range')

  # Structure levels
  struct = prior_day_levels(df)
  va = value_area_approx(df)
  df = pd.concat([df, struct, va], axis=1)

  thr_trend = params['entry']['p_thr']['trend']
  thr_range = params['entry']['p_thr']['range']

  gate = ConvictionParams(m=params['entry']['persistence']['m'],
                          k=params['entry']['persistence']['k'],
                          thr_entry=params['entry']['hysteresis']['thr_entry'],
                          thr_exit=params['entry']['hysteresis']['thr_exit'],
                          alpha_cost=0.0)
  state = ConvictionState()

  fr = Frictions(fee_bps_per_side=params['meta']['fee_bps_side'],
                 slippage_bps_per_side=params['meta']['slip_bps_side'],
                 funding_bps_estimate=params['meta']['funding_bps_rt'])
  frictions_bps = 2*fr.fee_bps_per_side + 2*fr.slippage_bps_per_side + fr.funding_bps_estimate

  calibrator = None
  if args.calibrator and Path(args.calibrator).exists():
    cal = json.load(open(args.calibrator, 'r', encoding='utf-8'))
    calibrator = {'maps': {}}
    for k, v in cal.get('maps', {}).items():
      xs = np.array(v.get('x', v.get('X_', [])), dtype=float)
      ys = np.array(v.get('y', v.get('y_', [])), dtype=float)
      if xs.size and ys.size:
        calibrator['maps'][k] = lambda x, xs=xs, ys=ys: np.interp(x, xs, ys, left=ys[0], right=ys[-1])

  min_hold = params['exit']['min_hold']
  max_hold = params['exit']['max_hold']

  position = 0
  entry_px = 0.0
  entry_idx = -1
  entry_session = ''
  entry_regime = ''
  be_armed = False
  trades = []
  gating_dbg = []
  tp_bps_cur = 0.0
  sl_bps_cur = 0.0

  for i in range(max(atr_n, gate.m)+1, len(df)):
    # ``pd.to_datetime`` may return a ``Series`` if fed a sequence-like input.
    # Ensure ``now_ts`` is always a scalar ``Timestamp`` to keep downstream
    # logic simple.
    now_ts = pd.to_datetime(df['open_time'].iloc[i], utc=True)
    if isinstance(now_ts, pd.Series):
      now_ts = now_ts.iloc[0]
    session = tag_session(now_ts)
    reg = regime[i]
    side = int(np.sign(dir_hint[i]))
    p_raw = float(p_raw_arr[i])
    if calibrator:
      gk = f"{session}_{reg}"
      if gk in calibrator['maps']:
        pop = float(calibrator['maps'][gk](p_raw))
      elif '_default' in calibrator['maps']:
        pop = float(calibrator['maps']['_default'](p_raw))
      else:
        pop = p_raw
    else:
      pop = p_raw

    ofi_row = ofi_feats.iloc[i]
    ofi_adj = soft_gate_adjustments(float(ofi_row['OFI_conf']), params['entry']['ofi']['conf_floor'])
    thr = (thr_trend if reg=='trend' else thr_range) + ofi_adj['thr_add'] + params['session'].get(session.lower(), {}).get('p_thr_adj', 0.0)
    close = df['close'].iloc[i]
    struct_pen = 0.0
    if reg == 'trend':
      if not ((close > df['PDH'].iloc[i]) or (close > df['VAH'].iloc[i])):
        struct_pen = 0.01
    else:
      val = df['VAL'].iloc[i]; vah = df['VAH'].iloc[i]
      if not (abs(close-val)/close < 0.001 or abs(close-vah)/close < 0.001):
        struct_pen = 0.01
    thr += struct_pen

    atr_bps = float(atr_1m(df.iloc[:i+1], atr_n).iloc[-1])
    tp_bps_i, sl_bps_i = dyn_tp_sl_bps(reg, atr_bps,
                                       params['session'].get(session.lower(), {}).get('exit_tp_adj_bps', 0),
                                       ofi_adj['tp_scale'])
    ev = pop * tp_bps_i - (1.0 - pop) * sl_bps_i - frictions_bps - (ofi_adj['pev_add'] * 100.0)
    passed_ev = ev >= 0.0

    mom_k = int((np.sign(macd_hist.values[i-gate.m+1:i+1]) == np.sign(macd_hist.values[i-gate.m:i])).sum())
    passed_persist = mom_k >= gate.k
    passed_calib = pop >= thr
    in_box = bool(df['in_box'].iloc[i])
    ofi_dir_ok = int(ofi_row['OFI_dir_ok'])

    dbg = {
      'i': i,
      'side': side,
      'session': session,
      'regime': reg,
      'pop': float(pop),
      'thr': float(thr),
      'tp_bps_i': float(tp_bps_i),
      'sl_bps_i': float(sl_bps_i),
      'frictions_bps': float(frictions_bps),
      'EV': float(ev),
      'OFI_conf': float(ofi_row['OFI_conf']),
      'OFI_align': float(ofi_row['OFI_align']),
      'OFI_dir_ok': int(ofi_row['OFI_dir_ok']), 'passed_ev': bool(passed_ev)
    }

    if side == -1:
      dbg['decision'] = 'reject'
      gating_dbg.append(dbg)
      continue

    if position == 0:
      decision = passed_calib and passed_persist and passed_ev and (not in_box) and ofi_dir_ok
      if decision:
        position = side
        entry_px = float(df['close'].iloc[i])
        entry_idx = i
        entry_session = session
        entry_regime = reg
        tp_bps_cur = tp_bps_i
        sl_bps_cur = sl_bps_i
        be_armed = False
        dbg['decision'] = 'enter'
      else:
        dbg['decision'] = 'reject'
    else:
      bars_held = i - entry_idx
      pnl_bps = (float(df['close'].iloc[i]) / entry_px - 1.0) * 10000.0 * position
      if not be_armed and bars_held >= params['exit']['be']['arm_after_bars'] and pnl_bps >= params['exit']['be']['arm_trigger_bps']:
        be_armed = True
      hit_tp = pnl_bps >= tp_bps_cur
      hit_sl = pnl_bps <= (0 if be_armed else -sl_bps_cur)
      time_exit = bars_held >= max_hold
      if hit_tp or hit_sl or time_exit:
        if bars_held >= min_hold:
          trades.append({
            'entry_idx': entry_idx,
            'exit_idx': i,
            'side': position,
            'entry_px': entry_px,
            'exit_px': float(df['close'].iloc[i]),
            'pnl_bps': pnl_bps,
            'bars_held': bars_held,
            'session': entry_session,
            'regime': entry_regime
          })
          position = 0
          entry_idx = -1
          entry_px = 0.0
          dbg['decision'] = 'exit'
        else:
          dbg['decision'] = 'hold'
      else:
        dbg['decision'] = 'hold'
      dbg['be_armed'] = bool(be_armed)
    gating_dbg.append(dbg)

  # Metrics
  actual = np.sign(df['close'].shift(-1) - df['close']).fillna(0).values
  pred = np.sign(p_raw_arr - 0.5)
  tp = int(((pred==1)&(actual==1)).sum()); tn = int(((pred==-1)&(actual==-1)).sum())
  fp = int(((pred==1)&(actual==-1)).sum()); fn = int(((pred==-1)&(actual==1)).sum())
  denom = math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
  mcc = float((tp*tn - fp*fn)/denom) if denom>0 else 0.0

  if trades:
    pnl = np.array([t['pnl_bps'] for t in trades])
    hit_rate = float((pnl>0).mean()) if len(pnl)>0 else 0.0
    summary = {'n_trades': int(len(trades)), 'hit_rate': hit_rate, 'mcc': mcc, 'cum_pnl_bps': float(pnl.sum())}
  else:
    summary = {'n_trades': 0, 'hit_rate': 0.0, 'mcc': mcc, 'cum_pnl_bps': 0.0}

  outdir = Path(args.outdir)
  outdir.mkdir(parents=True, exist_ok=True)
  with open(outdir/'trades.csv','w',newline='',encoding='utf-8') as f:
    fn = list(trades[0].keys()) if trades else ['entry_idx','exit_idx','side','entry_px','exit_px','pnl_bps','bars_held','session','regime']
    w = csv.DictWriter(f, fieldnames=fn); w.writeheader()
    for t in trades: w.writerow(t)
  with open(outdir/'gating_debug.json','w',encoding='utf-8') as f:
    json.dump(gating_dbg, f, ensure_ascii=False, indent=2)
  audit = pd.DataFrame({'open_time': df['open_time'], 'p_raw': p_raw_arr, 'p_trend': p_raw_arr, 'macd_hist': macd_hist, 'session': [tag_session(pd.to_datetime(t, utc=True)) for t in df['open_time']], 'regime': regime, 'label': (df['close'].shift(-1) > df['close']).astype(int)})
  audit.to_csv(outdir/'preds_test.csv', index=False)
  with open(outdir/'summary.json','w',encoding='utf-8') as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

if __name__ == '__main__':
  main()
