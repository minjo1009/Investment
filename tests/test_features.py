import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from backtest.strategy_v2.filters import approx_ofi, absorption_proxy

@pytest.fixture
def df_eth_1m():
  n = 300
  return pd.DataFrame({
    "open_time": pd.date_range("2020-01-01", periods=n, freq="1min"),
    "high": np.linspace(100, 101, n),
    "low": np.linspace(99, 100, n),
    "close": np.linspace(99.5, 100.5, n),
    "volume": np.ones(n),
    "taker_buy_volume": np.full(n, 0.5)
  })

def test_ofi_alignment_and_absorption(df_eth_1m):
  ofi = approx_ofi(df_eth_1m, window=5, align_ge=3)
  ab = absorption_proxy(df_eth_1m, window=20)
  assert ofi.isin([0, 1]).all()
  assert ab.dtype == bool
