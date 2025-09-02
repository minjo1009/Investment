import numpy as np
import pandas as pd

def swing_points(df: pd.DataFrame, left: int = 3, right: int = 3):
  hi = df["high"]
  lo = df["low"]
  sh = (hi.shift(left).rolling(left + right + 1).max() == hi).fillna(False)
  sl = (lo.shift(left).rolling(left + right + 1).min() == lo).fillna(False)
  return sh, sl

def macd_divergence(df: pd.DataFrame, macd_hist: pd.Series, min_sep: int = 10, min_mag: float = 0.0, ofi_ok=None, vol=None):
  sh, sl = swing_points(df)
  bull = (df["low"][sl].diff(min_sep) < 0) & (macd_hist[sl].diff(min_sep) > min_mag)
  bear = (df["high"][sh].diff(min_sep) > 0) & (macd_hist[sh].diff(min_sep) < -min_mag)
  sig = pd.Series(False, index=df.index)
  sig.loc[bull[bull==True].index] = True
  sig.loc[bear[bear==True].index] = True
  if ofi_ok is not None:
    sig &= ofi_ok.astype(bool)
  if vol is not None:
    z = (vol - vol.rolling(200).mean()) / (vol.rolling(200).std().replace(0, np.nan))
    sig &= (z > -1.0)
  return sig.fillna(False)

__all__ = ["swing_points", "macd_divergence"]
