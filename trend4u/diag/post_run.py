#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
import yaml


def _mcc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    denom = np.sqrt(max((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), 1))
    return float(((tp * tn) - (fp * fn)) / denom) if denom > 0 else 0.0


def _ece(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.digitize(p, bins) - 1
    ece = 0.0
    n = len(p)
    for i in range(n_bins):
        m = idx == i
        if m.any():
            acc = y[m].mean()
            conf = p[m].mean()
            ece += abs(acc - conf) * m.sum() / n
    return float(ece)


def _load_join(run_dir: Path) -> pd.DataFrame:
    """Load gating_debug and trades, joining on entry index."""
    gd_path = run_dir / "gating_debug.json"
    tr_path = run_dir / "trades.csv"
    if not gd_path.exists() or not tr_path.exists():
        return pd.DataFrame(columns=["i", "pop", "regime", "pnl_bps"])
    try:
        gdf = pd.read_json(gd_path)
    except ValueError:
        return pd.DataFrame(columns=["i", "pop", "regime", "pnl_bps"])
    if gdf.empty:
        return pd.DataFrame(columns=["i", "pop", "regime", "pnl_bps"])
    gdf = gdf[gdf.get("decision") == "enter"]
    try:
        trades = pd.read_csv(tr_path)
    except Exception:
        trades = pd.DataFrame(columns=["entry_idx", "pnl_bps"])
    df = gdf.merge(trades, left_on="i", right_on="entry_idx", how="left")
    return df


def generate_diagnostics(run_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = _load_join(run_dir)
    if df.empty:
        # create empty outputs
        pd.DataFrame(columns=["bin", "p_mean", "y_rate", "n"]).to_csv(out_dir / "calibration_bins.csv", index=False)
        pd.DataFrame(columns=["regime", "p_thr", "mcc", "n"]).to_csv(out_dir / "regime_threshold_scan.csv", index=False)
        pd.DataFrame(columns=["policy", "count"]).to_csv(out_dir / "policy_summary.csv", index=False)
        with open(out_dir / "diagnostics.json", "w", encoding="utf-8") as f:
            json.dump({}, f, indent=2)
        return

    df["label"] = (df["pnl_bps"].astype(float) > 0).astype(int)

    # Calibration bins
    bins = np.linspace(0, 1, 11)
    df["bin"] = np.digitize(df["pop"], bins) - 1
    calib = df.groupby("bin").agg(p_mean=("pop", "mean"), y_rate=("label", "mean"), n=("label", "count")).reset_index()
    calib.to_csv(out_dir / "calibration_bins.csv", index=False)

    # Regime threshold scan
    rows = []
    for regime, sub in df.groupby(df.get("regime", "all")):
        y = sub["label"].to_numpy()
        if len(y) == 0:
            continue
        for thr in np.linspace(0.5, 0.8, 31):
            pred = (sub["pop"].to_numpy() >= thr).astype(int)
            rows.append({
                "regime": regime,
                "p_thr": round(float(thr), 2),
                "mcc": _mcc(y, pred),
                "n": int(len(y)),
            })
    pd.DataFrame(rows).to_csv(out_dir / "regime_threshold_scan.csv", index=False)

    # Policy summary (TP/SL/Hold)
    tp = int((df["pnl_bps"] > 0).sum())
    sl = int((df["pnl_bps"] <= 0).sum())
    hold = 0
    pol = pd.DataFrame({"policy": ["TP", "SL", "Hold"], "count": [tp, sl, hold]})
    pol.to_csv(out_dir / "policy_summary.csv", index=False)

    # Diagnostics metrics
    p = df["pop"].to_numpy()
    y = df["label"].to_numpy()
    try:
        from sklearn.metrics import roc_auc_score

        auc = float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")
    except Exception:
        auc = float("nan")
    metrics = {
        "auc": auc,
        "mcc": _mcc(y, (p >= 0.5).astype(int)),
        "ece": _ece(p, y),
    }
    with open(out_dir / "diagnostics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def align(run_dir: Path, feature_flags: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = _load_join(run_dir)
    if df.empty:
        return
    df.sort_values("i", inplace=True)
    df["label"] = (df["pnl_bps"].astype(float) > 0).astype(int)

    split = int(len(df) * 0.7)
    head = df.iloc[:split]
    tail = df.iloc[split:]
    if len(head) == 0 or len(tail) == 0:
        return

    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(head["pop"].to_numpy(), head["label"].to_numpy())

    import pickle

    with open(out_dir / "calibrator_isotonic.pkl", "wb") as f:
        pickle.dump(ir, f)

    tail = tail.copy()
    tail["pop_cal"] = ir.transform(tail["pop"].to_numpy())
    tail_out = tail[["i", "regime", "pop_cal", "label"]]
    tail_out.to_json(out_dir / "calibrated_tail.json", orient="records", indent=2)

    # Threshold scan on calibrated tail
    rows = []
    best = {}
    for regime, sub in tail.groupby(tail.get("regime", "all")):
        y = sub["label"].to_numpy()
        if len(y) == 0:
            continue
        best_mcc, best_thr = -1.0, 0.5
        for thr in np.linspace(0.5, 0.8, 31):
            pred = (sub["pop_cal"].to_numpy() >= thr).astype(int)
            mcc = _mcc(y, pred)
            rows.append({"regime": regime, "p_thr": round(float(thr), 2), "mcc": mcc, "n": int(len(y))})
            if mcc > best_mcc:
                best_mcc, best_thr = mcc, float(thr)
        best[regime] = best_thr
    pd.DataFrame(rows).to_csv(out_dir / "alignment_threshold_scan.csv", index=False)

    # Update feature flags
    try:
        cfg = yaml.safe_load(open(feature_flags, "r", encoding="utf-8"))
    except Exception:
        cfg = {}
    cfg.setdefault("entry", {}).setdefault("p_thr", {})
    cfg.setdefault("ev", {}).setdefault("p_ev_req", {})
    for regime, thr in best.items():
        cfg["entry"]["p_thr"][regime] = float(thr)
        cfg["ev"]["p_ev_req"][regime] = float(thr)
    with open(feature_flags, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    diag = sub.add_parser("diag", help="generate diagnostics from run outputs")
    diag.add_argument("--run-dir", required=True)
    diag.add_argument("--out", required=True)

    al = sub.add_parser("align", help="fit calibrator and align feature flags")
    al.add_argument("--run-dir", required=True)
    al.add_argument("--feature-flags", required=True)
    al.add_argument("--out", required=True)

    args = ap.parse_args()
    if args.cmd == "diag":
        generate_diagnostics(Path(args.run_dir), Path(args.out))
    else:
        align(Path(args.run_dir), Path(args.feature_flags), Path(args.out))
