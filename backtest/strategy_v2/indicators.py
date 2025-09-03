import numpy as np
import pandas as pd

def wavelet_denoise_safe(x, level=1, mode="symmetric"):
  import numpy as np
  try:
    import pywt
  except Exception:
    return x
  arr = np.asarray(x, dtype=float)
  coeffs = pywt.wavedec(arr, "db2", mode=mode)
  sigma = np.nanstd(arr) if np.isfinite(arr).any() else 0.0
  coeffs[1:] = [pywt.threshold(c, value=sigma, mode="soft") for c in coeffs[1:]]
  out = pywt.waverec(coeffs, "db2", mode=mode)
  return np.asarray(out, dtype=float)[:len(arr)]

def atr_1m(df: pd.DataFrame, n: int = 14) -> pd.Series:
  h, l, c = df["high"], df["low"], df["close"]
  tr = np.maximum(h - l, np.maximum(np.abs(h - c.shift()), np.abs(l - c.shift())))
  atr = pd.Series(tr).rolling(n).mean()
  return (atr / df["close"].replace(0, np.nan)).fillna(0.0) * 1e4  # bps 환산

def dyn_tp_sl_bps(regime: str, atr_bps: float, session_tp_adj_bps: int,
                  ofi_tp_scale: float) -> tuple[float, float]:
  # ‘상’ 기준 스펙
  if regime == "trend":
    tp = np.clip(1.6 * atr_bps, 26, 60)
    sl = np.clip(0.9 * atr_bps, 14, 30)
  else:  # range
    tp = np.clip(1.2 * atr_bps, 22, 40)
    sl = np.clip(0.8 * atr_bps, 12, 26)
  tp = max(0.0, tp + session_tp_adj_bps) * ofi_tp_scale
  return float(tp), float(sl)
