import numpy as np
from dataclasses import dataclass

@dataclass
class RollingBalanceParams:
  win: int = 60  # minutes
  tr_pctl_max: int = 25

class RollingBalance:
  def __init__(self, params: RollingBalanceParams):
    self.params = params
    self.buffer = []  # store true range values

  def update(self, high: float, low: float, prev_close: float):
    tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
    self.buffer.append(tr)
    if len(self.buffer) > self.params.win:
      self.buffer.pop(0)

  def in_balance_box(self) -> bool:
    if not self.buffer:
      return False
    s = sorted(self.buffer)
    idx = max(0, min(len(s) - 1, (self.params.tr_pctl_max * len(s)) // 100))
    thr = s[idx]
    return self.buffer[-1] <= thr

def approx_ofi(df, window=5, align_ge=3):
  """
  경량 OFI: 최근 window 바 중 매수우위/매도우위 부호 정렬이 align_ge 이상일 때 True.
  """
  buy = df["taker_buy_volume"] if "taker_buy_volume" in df.columns else df["volume"] * 0
  sell = (df["volume"] - buy) if "volume" in df.columns else buy * 0
  sign = np.sign(buy.fillna(0) - sell.fillna(0))
  roll = sign.rolling(window).sum()
  return (np.abs(roll) >= align_ge).astype(int)

def agg_buy_sell_ratio(df, window=20):
  buy = df["taker_buy_volume"] if "taker_buy_volume" in df.columns else df["volume"] * 0
  num = buy.rolling(window).sum()
  den = df["volume"].rolling(window).sum().clip(lower=1e-9)
  return (num / den).clip(0, 1)

def absorption_proxy(df, window=20):
  # 가격변동 축소 + 체결량 급증 = 흡수(Absorption) 프록시
  rng = (df["high"] - df["low"]).rolling(window).mean().clip(lower=1e-9)
  vol = df["volume"].rolling(window).mean()
  z = (vol - vol.rolling(200).mean()) / (vol.rolling(200).std().replace(0, np.nan))
  return (z > 1.5) & (rng <= rng.rolling(50).quantile(0.3))
