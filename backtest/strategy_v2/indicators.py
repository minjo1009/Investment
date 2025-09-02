import numpy as np
import pandas as pd

def atr_1m(df: pd.DataFrame, n: int = 14) -> pd.Series:
  tr1 = df["high"] - df["low"]
  tr2 = (df["high"] - df["close"].shift(1)).abs()
  tr3 = (df["low"] - df["close"].shift(1)).abs()
  tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
  atr = tr.rolling(n).mean()
  return (atr / df["close"].replace(0, np.nan)).fillna(0)

__all__ = ["atr_1m"]
