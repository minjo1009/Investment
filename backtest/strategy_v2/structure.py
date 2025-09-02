import numpy as np
import pandas as pd

def prior_day_levels(df: pd.DataFrame) -> pd.DataFrame:
  d = df.copy()
  d["date"] = pd.to_datetime(d["open_time"]).dt.floor("D")
  by = d.groupby("date")
  out = d[["open_time"]].copy()
  out["PDH"] = by["high"].transform("max").shift(1)
  out["PDL"] = by["low"].transform("min").shift(1)
  out["PDc"] = by["close"].transform("last").shift(1)
  return out[["PDH","PDL","PDc"]]

def prior_week_levels(df: pd.DataFrame) -> pd.DataFrame:
  d = df.copy()
  d["week"] = pd.to_datetime(d["open_time"]).dt.to_period("W").dt.start_time
  by = d.groupby("week")
  out = d[["open_time"]].copy()
  out["PWH"] = by["high"].transform("max").shift(1)
  out["PWL"] = by["low"].transform("min").shift(1)
  return out[["PWH","PWL"]]

def value_area_approx(df: pd.DataFrame, window: str = "1D", q: float = 0.70, bins: int = 50) -> pd.DataFrame:
  # 롤링 구간에서 VWAP 분포 기반 VAH/VAL/POC 근사
  d = df.copy()
  d["datewin"] = pd.to_datetime(d["open_time"]).dt.floor(window)
  res = []
  for key, g in d.groupby("datewin"):
    px = g["close"].to_numpy()
    vw = g["volume"].to_numpy().clip(1e-9)
    hist, edges = np.histogram(px, bins=bins, weights=vw)
    cdf = np.cumsum(hist) / np.sum(hist)
    poc_idx = np.argmax(hist)
    low_idx = np.searchsorted(cdf, (1 - q) / 2.0)
    high_idx = np.searchsorted(cdf, 1 - (1 - q) / 2.0)
    VAL = edges[max(low_idx, 0)]
    VAH = edges[min(high_idx, len(edges) - 1)]
    POC = edges[poc_idx]
    res.append(pd.DataFrame({
      "open_time": g["open_time"],
      "VAL": VAL,
      "VAH": VAH,
      "POC": POC
    }))
  return pd.concat(res).set_index("open_time")

def bos_choch(df: pd.DataFrame, swing_len: int = 10):
  # 간단 BOS/ChoCh: 최근 swing 고저 돌파/전환
  roll_high = df["high"].rolling(swing_len).max().shift(1)
  roll_low = df["low"].rolling(swing_len).min().shift(1)
  bos = (df["close"] > roll_high) | (df["close"] < roll_low)
  choch = ((df["close"] > roll_high) & (df["close"].shift(1) < roll_high)) | \
          ((df["close"] < roll_low)  & (df["close"].shift(1) > roll_low))
  return bos.fillna(False), choch.fillna(False)

__all__ = [
  "prior_day_levels",
  "prior_week_levels",
  "value_area_approx",
  "bos_choch"
]
