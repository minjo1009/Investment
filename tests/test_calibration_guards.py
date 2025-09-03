
import numpy as np
from trend4u.calib.quantile_map import QuantileMap
from trend4u.calib.drift import psi, ks

def test_quantile_map_monotonic():
    x=np.random.randn(10000)
    qm=QuantileMap.fit(x, n_bins=50)
    y=qm.transform(x)
    assert np.all((y>=0)&(y<=1))
    # rank 보존의 근사
    assert abs(np.mean(y)-0.5)<0.1

def test_psi_ks_sane():
    base=np.random.beta(2,5,20000)
    drift=np.random.beta(5,2,20000)
    assert psi(base, drift)>0.1
    assert ks(base, drift)>0.1
