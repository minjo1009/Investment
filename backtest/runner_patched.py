import argparse, os, sys, runpy, glob
import pandas as pd, numpy as np

def _find_codepack_runner():
    cand=os.path.join("_codepack","backtest","runner.py")
    if os.path.exists(cand): return cand
    for p in sys.path:
        rp=os.path.join(p,"backtest","runner.py")
        if os.path.exists(rp): return rp
    raise FileNotFoundError("Cannot locate codepack runner.py")

def _read_one_csv(data_root, pattern):
    matches=glob.glob(os.path.join(data_root,pattern),recursive=True)
    if not matches: raise FileNotFoundError(f"No CSV matched: {pattern}")
    df=pd.read_csv(matches[0])
    low=[c.lower() for c in df.columns]
    def pick(cands):
        for c in cands:
            if c in low: return df.columns[low.index(c)]
        return None
    dcol=pick(["open_time","timestamp","time","datetime","date"])
    ccol=pick(["close","close_price","c"])
    if not dcol or not ccol: raise ValueError("data csv needs datetime+close")
    df[dcol]=pd.to_datetime(df[dcol])
    return df.rename(columns={dcol:"open_time", ccol:"close"})[["open_time","close"]]

def _pair_from_trades(tr):
    evcol=next((c for c in tr.columns if str(c).lower()=="event"), None)
    if evcol is None: return None, None
    t=tr.copy(); t["open_time"]=pd.to_datetime(t["open_time"],errors="coerce")
    e=t[evcol].astype(str).str.upper()
    t["is_entry"]=e.str.contains("ENTRY"); t["is_exit"]=e.str.contains("EXIT")
    t=t.sort_values("open_time",kind="stable").reset_index(drop=True)
    trade_id=[]; q=[]; next_id=0
    for ent,ex in zip(t["is_entry"],t["is_exit"]):
        if ent: next_id+=1; q.append(next_id); trade_id.append(next_id)
        elif ex and q: trade_id.append(q.pop(0))
        else: trade_id.append(np.nan)
    t["trade_id"]=trade_id
    ent=t[t["is_entry"]][["trade_id","open_time"]].rename(columns={"open_time":"entry_time"})
    ex =t[t["is_exit" ]][["trade_id","open_time"]].rename(columns={"open_time":"exit_time"})
    pairs=ent.merge(ex,on="trade_id",how="inner"); return pairs, evcol

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data-root",required=True); ap.add_argument("--csv-glob",required=True)
    ap.add_argument("--params",required=True); ap.add_argument("--outdir",required=True)
    a=ap.parse_args()

    # delegate to original runner
    orig=_find_codepack_runner()
    saved=sys.argv[:]
    sys.argv=[orig,"--data-root",a.data_root,"--csv-glob",a.csv_glob,"--params",a.params,"--outdir",a.outdir]
    print(f"[runner_patched] delegating → {orig}")
    try: runpy.run_path(orig, run_name="__main__")
    finally: sys.argv=saved

    # enrich trades.csv
    tpath=os.path.join(a.outdir,"trades.csv")
    if not os.path.exists(tpath): return
    tr=pd.read_csv(tpath)
    if "open_time" not in [c.lower() for c in tr.columns]: return
    data=_read_one_csv(a.data_root, a.csv_glob)
    pairs,evcol=_pair_from_trades(tr)
    if pairs is None or len(pairs)==0: return

    ent=pairs[["trade_id","entry_time"]].merge(
        data.rename(columns={"open_time":"entry_time","close":"entry_price"}),on="entry_time",how="left")
    ex =pairs[["trade_id","exit_time"]].merge(
        data.rename(columns={"open_time":"exit_time","close":"exit_price"}),on="exit_time",how="left")
    px=ent.merge(ex,on="trade_id",how="inner")
    # side/qty
    side_col=next((c for c in tr.columns if str(c).lower()=="side"), None)
    qty_col =next((c for c in tr.columns if str(c).lower()=="qty"), None)
    eidx=tr[tr[evcol].astype(str).str.upper().str.contains("ENTRY",na=False)].index
    side_seq=tr.loc[eidx, side_col].astype(float).clip(-1,1).fillna(1).astype(int).tolist() if side_col else [1]*len(px)
    qty_seq =tr.loc[eidx, qty_col ].astype(float).fillna(1.0).tolist() if qty_col  else [1.0]*len(px)
    while len(side_seq)<len(px): side_seq.append(1)
    while len(qty_seq)<len(px):  qty_seq.append(1.0)
    px["side"]=side_seq[:len(px)]; px["qty"]=qty_seq[:len(px)]
    px["pnl_close_based"]=(px["exit_price"]-px["entry_price"])*px["side"]*px["qty"]

    exmask=tr[evcol].astype(str).str.upper().str.contains("EXIT",na=False)
    exit_idx=tr[exmask].index.to_list(); k=min(len(exit_idx), len(px))
    if k>0:
        tr.loc[exit_idx[:k], "pnl_close_based"]=px["pnl_close_based"].values[:k]
        tr.loc[exit_idx[:k], "entry_time"]=px["entry_time"].astype(str).values[:k]
        tr.loc[exit_idx[:k], "entry_price"]=px["entry_price"].values[:k]
        tr.loc[exit_idx[:k], "exit_price"]=px["exit_price"].values[:k]
        tr.loc[exit_idx[:k], "side"]=px["side"].values[:k]
        tr.loc[exit_idx[:k], "qty"]=px["qty"].values[:k]
    tr.to_csv(tpath,index=False)
    px.to_csv(os.path.join(a.outdir,"pairs.csv"), index=False)
    print("[runner_patched] enrich done:", {"pairs": len(px)})

if __name__=="__main__": main()
