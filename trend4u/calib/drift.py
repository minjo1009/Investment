
import numpy as np
from scipy import stats


def _bins(x, n=20):
    edges = np.quantile(x, np.linspace(0, 1, n + 1))
    edges[0] -= 1e-9
    edges[-1] += 1e-9
    return edges


def psi(expected, actual, n=20):
    expected = np.asarray(expected, float)
    actual = np.asarray(actual, float)
    e_edges = _bins(expected, n)
    e_hist, _ = np.histogram(expected, bins=e_edges)
    a_hist, _ = np.histogram(actual, bins=e_edges)
    e = np.clip(e_hist / np.maximum(e_hist.sum(), 1), 1e-8, 1)
    a = np.clip(a_hist / np.maximum(a_hist.sum(), 1), 1e-8, 1)
    return float(np.sum((a - e) * np.log(a / e)))


def ks(expected, actual):
    return float(stats.ks_2samp(expected, actual).statistic)
