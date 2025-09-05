#!/usr/bin/env python3
"""Training script for Strategy V2 logistic model with data guards."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from backtest.utils import rebalance_labels_by_regime


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True, help="CSV produced by runner")
    ap.add_argument("--model-out", default="conf/model.pkl", help="Where to save the model")
    args = ap.parse_args()

    df = pd.read_csv(args.preds)

    use_cols = ["p_trend", "macd_hist", "rsi", "adx", "ofi"]
    df[use_cols] = df[use_cols].apply(pd.to_numeric, errors="coerce")
    assert df[use_cols].isna().mean().max() < 0.01, "NaN too high"
    assert (df[use_cols].std() > 1e-8).all(), "Zero-variance feature"
    X_df = df[use_cols]
    X = X_df.values
    assert np.isfinite(X).all(), "Non-finite in features"

    y = pd.to_numeric(df.get("label", 0), errors="coerce").fillna(0).astype(int).to_numpy()
    vc = pd.Series(y).value_counts()
    if len(vc) < 2 or (vc.min() / vc.max()) < 0.2:
        y = rebalance_labels_by_regime(df, target_ratio=(0.6, 0.4))
        vc = pd.Series(y).value_counts()
        assert len(vc) == 2 and (vc.min() / vc.max()) >= 0.2, f"Label imbalance: {vc.to_dict()}"

    clf = LogisticRegression(max_iter=1000, solver="liblinear")
    clf.fit(X_df, y)
    assert list(clf.feature_names_in_) == use_cols, "feature_names_in_ mismatch"

    # metrics for sanity check
    p = clf.predict_proba(X)[:, 1]
    print("RSI std:", float(df["rsi"].std()))
    print("ADX std:", float(df["adx"].std()))
    print("OFI std:", float(df["ofi"].std()))
    print("label distribution:", vc.to_dict())
    print("predict_proba mean:", float(p.mean()))
    print("predict_proba std:", float(p.std()))

    import joblib

    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, args.model_out)


if __name__ == "__main__":
    main()

