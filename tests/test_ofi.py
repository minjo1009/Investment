import numpy as np
import pandas as pd
import pytest

from backtest.strategy_v2.filters import ofi_conf_alignment


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


def test_ofi_conf_has_range(df_eth_1m):
  f = ofi_conf_alignment(df_eth_1m)
  assert f['OFI_conf'].notna().any()
  assert (f['OFI_conf'] >= 0).all()
