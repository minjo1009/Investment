"""Label utilities for rebalancing class distributions."""

from __future__ import annotations

import numpy as np
import pandas as pd


def rebalance_labels_by_regime(df: pd.DataFrame, target_ratio: tuple[float, float] = (0.6, 0.4)) -> np.ndarray:
    """Rebalance binary labels within each regime.

    Parameters
    ----------
    df : pd.DataFrame
        Data frame containing at least ``close`` prices and optionally a
        ``regime`` column.  Forward returns are computed using ``close`` to
        derive class labels when rebalancing is required.
    target_ratio : tuple[float, float], default=(0.6, 0.4)
        Desired (majority, minority) ratio for class 1 vs. class 0.  Each
        regime is processed independently to preserve structural differences.

    Returns
    -------
    np.ndarray
        Array of rebalance labels matching the length of ``df``.
    """

    close = pd.to_numeric(df["close"], errors="coerce")
    reg = df.get("regime", pd.Series("all", index=df.index))
    fwd_ret = close.pct_change().shift(-1)

    y = np.zeros(len(df), dtype=int)
    for r, idx in reg.groupby(reg).groups.items():
        rrets = fwd_ret.loc[idx].dropna()
        if rrets.empty:
            continue
        # choose threshold so approximately target_ratio[0] are class 1
        thr = rrets.quantile(1 - target_ratio[0])
        loc_y = (rrets > thr).astype(int)
        y[idx[: len(loc_y)]] = loc_y
    return y


__all__ = ["rebalance_labels_by_regime"]

