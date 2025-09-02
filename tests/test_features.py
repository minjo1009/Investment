import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from backtest.strategy_v2.filters import compute_bvr_ofi, ofi_conf_alignment
from backtest.strategy_v2.structure import prior_day_levels


@pytest.fixture
def df_eth_1m():
  n = 300
  return pd.DataFrame({
    'open_time': pd.date_range('2020-01-01', periods=n, freq='1min'),
    'high': np.linspace(100, 101, n),
    'low': np.linspace(99, 100, n),
    'close': np.linspace(99.5, 100.5, n),
    'volume': np.ones(n),
    'taker_buy_base_asset_volume': np.full(n, 0.5),
    'number_of_trades': np.ones(n)
  })


def test_ofi_features(df_eth_1m):
  feats = compute_bvr_ofi(df_eth_1m)
  assert set(['BVR','OFI']).issubset(feats.columns)
  aligned = ofi_conf_alignment(df_eth_1m)
  assert 'OFI_conf' in aligned.columns


def test_prior_day_levels_handles_duplicates(df_eth_1m):
  df = df_eth_1m.copy()
  df.insert(0, 'open_time', df['open_time'], allow_duplicates=True)
  assert df.columns.duplicated().any()
  out = prior_day_levels(df)
  assert set(['PDH','PDL','PDc']).issubset(out.columns)
  assert len(out) == len(df)
