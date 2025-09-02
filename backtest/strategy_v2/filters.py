import numpy as np
import pandas as pd

def _zscore(s: pd.Series, win: int = 200) -> pd.Series:
  mu = s.rolling(win).mean()
  sd = s.rolling(win).std().replace(0, np.nan)
  return (s - mu) / sd

def ensure_ofi_columns(df: pd.DataFrame) -> pd.DataFrame:
  """
  보장: 'taker_buy_volume' 파생.
  입력 스펙: taker_buy_base_asset_volume.
  """
  if "taker_buy_volume" not in df.columns:
    if "taker_buy_base_asset_volume" in df.columns:
      df = df.copy()
      df["taker_buy_volume"] = df["taker_buy_base_asset_volume"].fillna(0.0)
    else:
      df = df.copy()
      df["taker_buy_volume"] = 0.0
  if "number_of_trades" not in df.columns:
    df = df.copy()
    df["number_of_trades"] = 0.0
  return df

def compute_bvr_ofi(df: pd.DataFrame) -> pd.DataFrame:
  """
  BVR = TBV / volume, OFI = 2*BVR-1 ∈ [-1,1]
  """
  df = ensure_ofi_columns(df)
  vol = df["volume"].replace(0, np.nan)
  tbv = df["taker_buy_volume"].clip(lower=0.0)
  bvr = (tbv / vol).clip(lower=0.0, upper=1.0).fillna(0.0)
  ofi = 2.0 * bvr - 1.0
  out = pd.DataFrame(index=df.index)
  out["BVR"] = bvr
  out["OFI"] = ofi
  return out

def ofi_conf_alignment(df: pd.DataFrame,
                       ema_len: int = 10,
                       align_window: int = 5,
                       align_ge: int = 3) -> pd.DataFrame:
  """
  OFI_smooth = EMA(OFI, L); align = 최근 W바 중 부호 정렬 횟수≥K
  거래활성 보정: number_of_trades, volume zscore
  conf = |OFI_smooth| * max(0, z_trades, z_vol).clip(0, inf)
  """
  feats = compute_bvr_ofi(df)
  ofi = feats["OFI"]
  alpha = 2.0 / (ema_len + 1.0)
  ofi_s = ofi.ewm(alpha=alpha, adjust=False).mean()

  dclose = df["close"].diff().fillna(0.0)
  dir_ok = np.sign(dclose) == np.sign(ofi_s)
  align = dir_ok.rolling(align_window).sum().fillna(0.0)

  z_tr = _zscore(df["number_of_trades"].fillna(0.0), 200).fillna(0.0)
  z_vol = _zscore(df["volume"].fillna(0.0), 200).fillna(0.0)
  act = np.maximum(0.0, np.maximum(z_tr, z_vol))
  conf = np.abs(ofi_s) * act

  out = pd.DataFrame(index=df.index)
  out["BVR"] = feats["BVR"]
  out["OFI_smooth"] = ofi_s
  out["OFI_align"] = align
  out["OFI_conf"] = conf
  out["OFI_dir_ok"] = dir_ok.astype(int)
  return out

def soft_gate_adjustments(ofi_conf: float, conf_floor: float = 0.30) -> dict:
  """
  하드컷 금지: conf 미달이면 문턱/EV에 소프트 페널티로 반영.
  """
  if np.isnan(ofi_conf):
    ofi_conf = 0.0
  if ofi_conf >= conf_floor:
    return {"thr_add": 0.0, "pev_add": 0.0, "tp_scale": 1.0}
  # 약한 OFI: 문턱 +0.02, EV 요구 +0.01, TP 효과 0.9배
  return {"thr_add": 0.02, "pev_add": 0.01, "tp_scale": 0.90}
