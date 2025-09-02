import numpy as np
import pandas as pd

def _swing_points(df: pd.DataFrame, left: int = 3, right: int = 3):
  hi, lo = df["high"], df["low"]
  sh = (hi.shift(left).rolling(left+right+1).max() == hi).fillna(False)
  sl = (lo.shift(left).rolling(left+right+1).min() == lo).fillna(False)
  return sh, sl

def macd_divergence(df: pd.DataFrame, macd_hist: pd.Series,
                    ofi_conf: pd.Series, min_sep: int = 10,
                    min_mag: float = 0.0) -> pd.Series:
  sh, sl = _swing_points(df)
  bull = (df["low"][sl].diff(min_sep) < 0) & (macd_hist[sl].diff(min_sep) > min_mag)
  bear = (df["high"][sh].diff(min_sep) > 0) & (macd_hist[sh].diff(min_sep) < -min_mag)
  sig = pd.Series(False, index=df.index)
  sig.loc[bull[bull==True].index] = True
  sig.loc[bear[bear==True].index] = True
  # 저활성/약한 OFI 구간 제거
  ok = (ofi_conf >= 0.30)
  return (sig & ok).fillna(False)
