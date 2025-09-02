import hashlib
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from backtest.strategy_v2.filters import compute_bvr_ofi
from backtest.strategy_v2.indicators import atr_1m

def test_feature_golden():
  df = pd.DataFrame({
    'open_time': pd.date_range('2020-01-01', periods=5, freq='1min'),
    'high': np.linspace(100, 101, 5),
    'low': np.linspace(99, 100, 5),
    'close': np.linspace(99.5, 100.5, 5),
    'volume': np.arange(1, 6),
    'taker_buy_volume': np.arange(5)
  })
  atr = atr_1m(df)
  ofi = compute_bvr_ofi(df)['OFI']
  payload = pd.DataFrame({'atr': atr, 'ofi': ofi}).fillna(0).to_csv(index=False)
  h = hashlib.sha256(payload.encode()).hexdigest()
  assert h == '359e9344cf27dd0234c722e8124010bff2cc3bc8daac1d69ff9de96154273f7f'
