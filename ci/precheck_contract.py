import argparse, os, glob, pandas as pd
OK,WARN,FAIL="OK","WARN","FAIL"

def _exists(p): return os.path.exists(p)

def check_data(root, pattern):
    matches=glob.glob(os.path.join(root,pattern),recursive=True)
    if not matches: return FAIL,"no CSV matched"
    df=pd.read_csv(matches[0],nrows=5)
    cols={c.lower() for c in df.columns}
    need={"close"}; dt={"open_time","timestamp","time","datetime","date"}
    if "close" not in cols or not (cols & dt):
        return FAIL,f"data columns miss: have={sorted(cols)[:12]}"
    return OK,f"csv_ok: {os.path.basename(matches[0])}"

def check_trades(outdir):
    p=os.path.join(outdir,"trades.csv")
    if not _exists(p): return WARN,"trades.csv missing"
    df=pd.read_csv(p,nrows=100)
    cols={c.lower() for c in df.columns}
    need={"open_time","event"}
    if not need.issubset(cols): return FAIL,f"trades miss: need {need}, have={cols}"
    if "session_tag" not in cols:
        col=next((c for c in df.columns if c.lower()=="open_time"),None)
        ts=pd.to_datetime(df[col],utc=True,errors="coerce") if col else pd.Series(dtype="datetime64[ns, UTC]")
        hrs=ts.dt.hour
        tag=pd.Series("US",index=ts.index)
        tag.loc[hrs<8]="ASIA"
        tag.loc[(hrs>=8)&(hrs<16)]="EU"
        print("[precheck] session_tag generated",sorted(tag.unique()))
    cost_fields={"fee_bps_per_side","slip_bps_per_side","funding_bps_rt"}
    missing=[c for c in cost_fields if c not in cols]
    if missing:
        print(f"[precheck] WARN missing cost fields: {missing}")
    return OK,"trades_ok"

def check_preds(outdir):
    p=os.path.join(outdir,"preds_test.csv")
    if not _exists(p): return WARN,"preds_test.csv missing"
    df=pd.read_csv(p,nrows=5)
    cols={c.lower() for c in df.columns}
    okprob=next((c for c in ["p","p_gate","gatep","prob","score","p_trend","p_range"] if c in cols),None)
    if "open_time" not in cols or not okprob:
        return FAIL,f"preds miss: need open_time + prob, have={cols}"
    return OK,f"preds_ok prob={okprob}"

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data-root",required=True)
    ap.add_argument("--csv-glob",required=True)
    ap.add_argument("--outdir",required=True)
    a=ap.parse_args()
    r={}
    r["data"]=check_data(a.data_root,a.csv_glob)
    r["trades"]=check_trades(a.outdir)
    r["preds"]=check_preds(a.outdir)
    print("[precheck]",r)
    if r["data"][0]==FAIL or r["trades"][0]==FAIL: raise SystemExit(13)
    if r["preds"][0]==FAIL: raise SystemExit(14)

if __name__=="__main__": main()
