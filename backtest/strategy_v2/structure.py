import numpy as np
import pandas as pd

def prior_day_levels(df: pd.DataFrame) -> pd.DataFrame:
  # Guard against duplicate column names (e.g. multiple 'open_time' columns)
  # which can cause ``pd.to_datetime`` to raise a ValueError when a column
  # selection returns a DataFrame instead of a Series.
  d = df.loc[:, ~df.columns.duplicated()].copy()
  d["date"] = pd.to_datetime(d["open_time"]).dt.floor("D")
  by = d.groupby("date")
  out = d[["open_time"]].copy()
  out["PDH"] = by["high"].transform("max").shift(1)
  out["PDL"] = by["low"].transform("min").shift(1)
  out["PDc"] = by["close"].transform("last").shift(1)
  return out.set_index(d.index)[["PDH","PDL","PDc"]]

def value_area_approx(df: pd.DataFrame, bins: int = 50) -> pd.DataFrame:
  px = df["close"].to_numpy()
  vw = df["volume"].to_numpy().clip(1e-9)
  hist, edges = np.histogram(px, bins=bins, weights=vw)
  cdf = np.cumsum(hist) / np.sum(hist)
  poc_idx = int(np.argmax(hist))
  low_idx = int(np.searchsorted(cdf, 0.15))
  high_idx = int(np.searchsorted(cdf, 0.85))
  VAL = edges[low_idx]
  VAH = edges[min(high_idx, len(edges)-1)]
  POC = edges[poc_idx]
  out = pd.DataFrame(index=df.index)
  out["VAL"] = VAL
  out["VAH"] = VAH
  out["POC"] = POC
  return out

def bos_choch(df: pd.DataFrame, swing_len: int = 10) -> pd.DataFrame:
  rh = df["high"].rolling(swing_len).max().shift(1)
  rl = df["low"].rolling(swing_len).min().shift(1)
  bos_up = df["close"] > rh
  bos_dn = df["close"] < rl
  choch = ((df["close"] > rh) & (df["close"].shift(1) < rh)) | \
          ((df["close"] < rl) & (df["close"].shift(1) > rl))
  out = pd.DataFrame(index=df.index)
  out["BOS_UP"] = bos_up.fillna(False)
  out["BOS_DN"] = bos_dn.fillna(False)
  out["CHOCH"] = choch.fillna(False)
  return out
