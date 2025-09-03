import numpy as np
import pandas as pd

def _swing_points(series: pd.Series, left: int = 3, right: int = 3):
  hi = series
  sh = (hi.shift(left).rolling(left+right+1).max() == hi).fillna(False)
  sl = (hi.shift(left).rolling(left+right+1).min() == hi).fillna(False)
  return sh, sl

def macd_divergence(price: pd.Series, macd_hist: pd.Series,
                    left: int = 3, right: int = 3,
                    min_sep: int = 10, min_mag: float = 0.0) -> pd.Series:
  sh, sl = _swing_points(price, left, right)
  res = pd.Series("none", index=price.index)
  lows = price[sl]
  macd_lows = macd_hist[sl]
  bull = (lows.diff(min_sep) < 0) & (macd_lows.diff(min_sep) > min_mag)
  res.loc[bull[bull==True].index] = "bull"
  highs = price[sh]
  macd_highs = macd_hist[sh]
  bear = (highs.diff(min_sep) > 0) & (macd_highs.diff(min_sep) < -min_mag)
  res.loc[bear[bear==True].index] = "bear"
  return res.fillna("none")
