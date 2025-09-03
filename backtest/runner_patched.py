# -*- coding: utf-8 -*-
"""runner_patched.py — Strategy V2 wiring (vectorized OFI/TR/EV gate)"""
import os, sys, json, argparse, csv, math, zipfile, glob
from pathlib import Path
import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype
from backtest.utils.dedupe import safe_load_no_dupe, dedupe_columns
from backtest.strategy_v2.filters import _zscore, ensure_ofi_columns
from backtest.strategy_v2.indicators import atr_1m
from backtest.strategy_v2.costs import Frictions


def normalize_open_time(df: pd.DataFrame) -> pd.DataFrame:
  """Vectorized conversion of ``open_time`` to UTC timestamps."""
  s = df["open_time"]
  if is_datetime64_any_dtype(s):
    try:
      if getattr(s.dt.tz, "zone", None) == "UTC":
        return df
    except Exception:
      pass
  if is_numeric_dtype(s):
    df["open_time"] = pd.to_datetime(s.astype("int64"), unit="ms", utc=True)
    return df
  ss = s.astype(str)
  sample = ss.dropna().head(200)
  if sample.str.fullmatch(r"\d{12,}").mean() > 0.8:
    df["open_time"] = pd.to_datetime(ss.astype("int64"), unit="ms", utc=True)
    return df
  for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S",
              "%Y-%m-%d %H:%M:%S%z", "%Y-%m-%dT%H:%M:%S%z"):
    try:
      df["open_time"] = pd.to_datetime(ss, format=fmt, utc=True, errors="raise")
      return df
    except Exception:
      continue
  df["open_time"] = pd.to_datetime(ss, utc=True, errors="coerce")
  return df


def expit(x):
  return 1.0 / (1.0 + np.exp(-x))

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
    return safe_load_no_dupe(f)


def tag_session(ts_utc) -> str:
  try:
    h = int(pd.Timestamp(ts_utc, tz='UTC').hour)
  except Exception:
    return "US"
  if 0 <= h < 8:
    return "ASIA"
  if 8 <= h < 16:
    return "EU"
  return "US"


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--params', default='conf/params_champion.yml')
  ap.add_argument('--calibrator', default=None)
  ap.add_argument('--flags', default='conf/feature_flags.yml')
  ap.add_argument('--data', default='ETHUSDT_1min_2020_2025.zip',
                  help='Path to zipped dataset containing a CSV (used if --data-root not provided)')
  ap.add_argument('--data-root', default=None,
                  help='Directory containing CSV files (searched when provided)')
  ap.add_argument('--csv-glob', default=None,
                  help='Glob pattern to match CSV within --data-root')
  ap.add_argument('--outdir', default='out')
  ap.add_argument('--start', default=None)
  ap.add_argument('--end', default=None)
  ap.add_argument('--limit-bars', type=int, default=None)
  ap.add_argument('--debug-level', choices=['all','entries','none'], default='entries')
  ap.add_argument('--no-preds', action='store_true')
  args = ap.parse_args()

  params = load_yaml(args.params)
  flags = load_yaml(args.flags) if args.flags and Path(args.flags).exists() else {}

  # Load data ---------------------------------------------------------------
  if args.data_root:
    pattern = args.csv_glob or '*.csv'
    matches = [p for p in glob.glob(str(Path(args.data_root)/'**'/'*.csv'), recursive=True)
               if glob.fnmatch.fnmatch(Path(p).name, pattern)]
    if not matches:
      raise SystemExit(f'No CSV matched: {pattern}')
    csv_path = matches[0]
    df = pd.read_csv(csv_path)
  else:
    with zipfile.ZipFile(args.data) as z:
      csv_name = next(n for n in z.namelist() if n.endswith('.csv'))
      with z.open(csv_name) as f:
        df = pd.read_csv(f)

  if 'timestamp' in df.columns:
    df['open_time'] = df.pop('timestamp')
  df = ensure_ofi_columns(df)
  df = normalize_open_time(df)
  df = dedupe_columns(df)

  if args.start:
    start_ts = pd.to_datetime(args.start, utc=True)
    df = df[df['open_time'] >= start_ts]
  if args.end:
    end_ts = pd.to_datetime(args.end, utc=True)
    df = df[df['open_time'] <= end_ts]
  df = df.reset_index(drop=True)
  if args.limit_bars:
    df = df.tail(int(args.limit_bars)).reset_index(drop=True)

  # Indicators -------------------------------------------------------------
  fast, slow, sig = 12, 26, 9
  ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
  ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
  macd = ema_fast - ema_slow
  macd_sig = macd.ewm(span=sig, adjust=False).mean()
  macd_hist = macd - macd_sig
  df['macd_hist'] = macd_hist
  macd_z = _zscore(macd_hist).fillna(0.0)

  eps = 1e-9
  ofi = ((df['close'] - df['open']) / ((df['high'] - df['low']).replace(0,np.nan) + eps)) * df['volume']
  ofi = ofi.fillna(0.0)
  df['ofi'] = ofi
  ofi_z = _zscore(ofi).fillna(0.0)
  df['ofi_z'] = ofi_z

  pc = df['close'].shift(1)
  tr = np.maximum(df['high'] - df['low'], np.maximum(np.abs(df['high'] - pc), np.abs(df['low'] - pc)))
  df['tr'] = tr
  rb_win = params['regime']['detector']['donchian_len']
  thr_q = tr.rolling(rb_win).quantile(0.25)
  df['in_box'] = tr <= thr_q

  atr_n = params['regime']['detector']['atr_len']
  atr = tr.rolling(atr_n).mean()
  z = (atr - atr.rolling(atr_n*5, min_periods=atr_n).mean()) / (atr.rolling(atr_n*5, min_periods=atr_n).std() + 1e-9)
  regime = np.where(z >= 0.5, 'trend', 'range')
  df['regime'] = regime

  m = params['entry']['persistence']['m']
  k = params['entry']['persistence']['k']
  aligned01 = (np.sign(macd_hist) == np.sign(macd_hist.shift(1))).astype(int)
  aligned_m = aligned01.rolling(m).sum().fillna(0)
  passed_persist = aligned_m >= k

  weights = {'macd': 0.6, 'ofi': 0.4}
  score = weights['macd'] * macd_z + weights['ofi'] * ofi_z
  p_raw_arr = expit(0.8 * score + 0.0)
  df['p_raw'] = p_raw_arr
  p_trend = p_raw_arr.copy()

  hours = df['open_time'].dt.hour
  session = np.where(hours < 8, 'ASIA', np.where(hours < 16, 'EU', 'US'))
  df['session'] = session

  atr_bps = atr_1m(df, atr_n)
  sess_adj_map = {k.upper(): v.get('exit_tp_adj_bps',0) for k,v in params.get('session',{}).items()}
  sess_adj = np.vectorize(sess_adj_map.get)(session)
  tp_trend = np.clip(1.6 * atr_bps, 26, 60)
  sl_trend = np.clip(0.9 * atr_bps, 14, 30)
  tp_range = np.clip(1.2 * atr_bps, 22, 40)
  sl_range = np.clip(0.8 * atr_bps, 12, 26)
  tp_i = np.where(regime=='trend', tp_trend, tp_range) + sess_adj
  sl_i = np.where(regime=='trend', sl_trend, sl_range)

  fr = Frictions(fee_bps_per_side=params['meta']['fee_bps_side'],
                 slippage_bps_per_side=params['meta']['slip_bps_side'],
                 funding_bps_estimate=params['meta']['funding_bps_rt'])
  frictions_bps = 2*fr.fee_bps_per_side + 2*fr.slippage_bps_per_side + fr.funding_bps_estimate
  ev_params = params.get('gating',{}).get('conviction',{}).get('ev_gate',{})
  delta_p_min = ev_params.get('delta_p_min',0.0)
  ev_margin_bps = ev_params.get('ev_margin_bps',0.0)
  p_ev_req = (sl_i + frictions_bps + ev_margin_bps) / (tp_i + sl_i)
  passed_ev = (p_trend >= p_ev_req) & ((p_trend - p_ev_req) >= delta_p_min)

  side = np.sign(macd_hist).astype(int)
  thr_trend = params['entry']['p_thr']['trend']
  thr_range = params['entry']['p_thr']['range']
  thr = np.where(regime=='trend', thr_trend, thr_range)
  passed_calib = p_trend >= thr
  ofi_lb = params['entry']['ofi']['align_window']
  ofi_dir = ofi.rolling(ofi_lb).sum()
  ofi_ok = (ofi_dir * side > 0) & (ofi_z >= 0)
  mask_entry = (side!=0) & (~df['in_box']) & passed_persist & passed_calib & passed_ev & ofi_ok
  cand_idx = np.flatnonzero(mask_entry.values)

  min_hold = params['exit']['min_hold']
  max_hold = params['exit']['max_hold']
  be_after = params['exit']['be']['arm_after_bars']
  be_trigger = params['exit']['be']['arm_trigger_bps']

  position = 0
  entry_px = 0.0
  entry_idx = -1
  entry_session = ''
  entry_regime = ''
  be_armed = False
  tp_bps_cur = 0.0
  sl_bps_cur = 0.0
  trades = []
  gating_dbg = []

  for i in range(len(df)):
    if position == 0:
      if args.debug_level != 'none' and side[i] != 0:
        gating_dbg.append({'i': i,'side': int(side[i]),'pop': float(p_trend[i]),
                           'p_ev_req': float(p_ev_req[i]),
                           'ev_bps': float(p_trend[i]*tp_i[i] - (1-p_trend[i])*sl_i[i] - frictions_bps),
                           'tp_bps_i': float(tp_i[i]),'sl_bps_i': float(sl_i[i]),
                           'regime': regime[i],'be_armed': False,
                           'EV': float(p_trend[i]*tp_i[i] - (1-p_trend[i])*sl_i[i] - frictions_bps),
                           'decision': 'enter' if mask_entry.iloc[i] else 'reject'})
      if mask_entry.iloc[i]:
        position = int(side[i])
        entry_px = float(df['close'].iloc[i])
        entry_idx = i
        entry_session = session[i]
        entry_regime = regime[i]
        tp_bps_cur = float(tp_i[i])
        sl_bps_cur = float(sl_i[i])
        be_armed = False
    else:
      bars_held = i - entry_idx
      pnl_bps = (df['close'].iloc[i]/entry_px - 1.0) * 10000.0 * position
      if (not be_armed) and bars_held >= be_after and pnl_bps >= be_trigger:
        be_armed = True
      hit_tp = pnl_bps >= tp_bps_cur
      hit_sl = pnl_bps <= (0 if be_armed else -sl_bps_cur)
      time_exit = bars_held >= max_hold
      if hit_tp or hit_sl or time_exit:
        if bars_held >= min_hold:
          trades.append({'entry_idx': entry_idx,'exit_idx': i,'side': position,
                         'entry_px': entry_px,'exit_px': float(df['close'].iloc[i]),
                         'pnl_bps': pnl_bps,'bars_held': bars_held,
                         'session': entry_session,'regime': entry_regime})
          if args.debug_level != 'none':
            gating_dbg.append({'i': i,'side': position,'pop': float(p_trend[i]),
                               'p_ev_req': float(p_ev_req[i]),
                               'ev_bps': float(p_trend[i]*tp_i[i] - (1-p_trend[i])*sl_i[i] - frictions_bps),
                               'tp_bps_i': float(tp_i[i]),'sl_bps_i': float(sl_i[i]),
                               'regime': regime[i],'be_armed': be_armed,
                               'EV': float(p_trend[i]*tp_i[i] - (1-p_trend[i])*sl_i[i] - frictions_bps),
                               'decision': 'exit_timeout' if time_exit and not (hit_tp or hit_sl) else 'exit'})
          position = 0
          entry_idx = -1
          entry_px = 0.0
        else:
          if args.debug_level == 'all':
            gating_dbg.append({'i': i,'side': position,'pop': float(p_trend[i]),
                               'p_ev_req': float(p_ev_req[i]),
                               'ev_bps': float(p_trend[i]*tp_i[i] - (1-p_trend[i])*sl_i[i] - frictions_bps),
                               'tp_bps_i': float(tp_i[i]),'sl_bps_i': float(sl_i[i]),
                               'regime': regime[i],'be_armed': be_armed,
                               'EV': float(p_trend[i]*tp_i[i] - (1-p_trend[i])*sl_i[i] - frictions_bps),
                               'decision': 'hold'})
      elif args.debug_level == 'all':
        gating_dbg.append({'i': i,'side': position,'pop': float(p_trend[i]),
                           'p_ev_req': float(p_ev_req[i]),
                           'ev_bps': float(p_trend[i]*tp_i[i] - (1-p_trend[i])*sl_i[i] - frictions_bps),
                           'tp_bps_i': float(tp_i[i]),'sl_bps_i': float(sl_i[i]),
                           'regime': regime[i],'be_armed': be_armed,
                           'EV': float(p_trend[i]*tp_i[i] - (1-p_trend[i])*sl_i[i] - frictions_bps),
                           'decision': 'hold'})

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

  if args.debug_level != 'none':
    with open(outdir/'gating_debug.json','w',encoding='utf-8') as f:
      json.dump(gating_dbg, f, ensure_ascii=False, indent=2)
    cols = ['i','side','pop','p_ev_req','ev_bps','tp_bps_i','sl_bps_i','regime','be_armed','decision','EV']
    pd.DataFrame(gating_dbg, columns=cols).to_csv(outdir/'gating_debug.csv', index=False)
  else:
    with open(outdir/'gating_debug.json','w',encoding='utf-8') as f:
      json.dump([], f)

  if not args.no_preds:
    audit = pd.DataFrame({'open_time': df['open_time'], 'p_raw': p_raw_arr, 'p_trend': p_trend, 'macd_hist': macd_hist,
                          'session': session, 'regime': regime,
                          'label': (df['close'].shift(-1) > df['close']).astype(int)})
    audit.to_csv(outdir/'preds_test.csv', index=False)

  with open(outdir/'summary.json','w',encoding='utf-8') as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

if __name__ == '__main__':
  main()
