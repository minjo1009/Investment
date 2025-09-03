import argparse, os, sys, runpy, zipfile, yaml, shutil
from backtest.utils.dedupe import safe_load_no_dupe
from pathlib import Path
import pandas as pd

def sanitize_glob(g, repo_root):
    g=g.lstrip("./"); rootname=os.path.basename(repo_root.rstrip("/"))
    if g.startswith(rootname+"/"): g=g[len(rootname)+1:]
    while True:
        parts=g.split("/",1)
        if len(parts)==2 and parts[0].lower().replace("_"," ").startswith("multiregime 4t"):
            g=parts[1]
        else: break
    return g

def ensure_codepack(codepack_zip, workdir):
    if not codepack_zip or not os.path.exists(codepack_zip): return None
    out=os.path.join(workdir,"_codepack"); 
    if os.path.exists(out): shutil.rmtree(out)
    with zipfile.ZipFile(codepack_zip) as z: z.extractall(out)
    return out

def patch_params(base_params_path, out_path, thr, hold):
    d=safe_load_no_dupe(open(base_params_path,"r",encoding="utf-8")) or {}
    d.setdefault("entry",{}).setdefault("p_thr",{})
    d["entry"]["p_thr"]["trend"]=float(thr); d["entry"]["p_thr"]["range"]=float(thr)
    d.setdefault("exit",{}); d["exit"]["min_hold"]=int(hold)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    yaml.safe_dump(d, open(out_path,"w",encoding="utf-8"), sort_keys=False, allow_unicode=True)

def post_enrich(outdir):
    tp=os.path.join(outdir,"trades.csv"); sp=os.path.join(outdir,"summary.json")
    if not os.path.exists(tp): return
    df=pd.read_csv(tp); ev=next((c for c in df.columns if str(c).lower()=="event"),None)
    if ev is not None: df=df[df[ev].astype(str).str.upper()=="EXIT"].copy()
    pnl_col=next((c for c in ["pnl_close_based","pnl","pnl_value","pnl_usd","pnl_pct","ret","return"] if c in df.columns),None)
    summ={}
    if os.path.exists(sp): import json; summ=json.load(open(sp,"r",encoding="utf-8"))
    summ["exits"]=int(len(df))
    if pnl_col is not None:
        s=pd.to_numeric(df[pnl_col],errors="coerce").fillna(0.0)
        summ["win_rate"]=float((s>0).sum())/max(1,len(s))
        pos,neg=float(s[s>0].sum()),float(s[s<0].sum())
        summ["profit_factor"]=(pos/abs(neg)) if neg!=0 else None
        summ["cum_pnl_close_based"]=float(s.sum())
    import json; json.dump(summ, open(sp,"w",encoding="utf-8"), ensure_ascii=False, indent=2)

def zip_dir(path,zip_path):
    with zipfile.ZipFile(zip_path,"w",zipfile.ZIP_DEFLATED) as z:
        for p in Path(path).rglob("*"):
            z.write(p, p.relative_to(path))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--params",required=True)
    ap.add_argument("--data-root",required=True)
    ap.add_argument("--csv-glob",required=True)
    ap.add_argument("--thr-list",nargs="+",required=True)
    ap.add_argument("--hold-list",nargs="+",required=True)
    ap.add_argument("--codepack",default="strategy_v2_codepack_v2.1.3.zip")
    ap.add_argument("--runner",default="backtest/runner_patched.py")
    ap.add_argument("--out-bundle",default="")
    a=ap.parse_args()

    repo_root=os.getcwd(); csvg=sanitize_glob(a.csv_glob, repo_root)
    cp_dir=ensure_codepack(a.codepack, repo_root)
    runner_path=a.runner or os.path.join(cp_dir or ".","backtest","runner.py")
    if not os.path.exists(runner_path): raise FileNotFoundError(f"runner not found: {runner_path}")
    sys.path[:0]=[repo_root]; 
    if cp_dir: sys.path[:0]=[cp_dir, os.path.join(cp_dir,"backtest")]

    out_zips=[]
    for thr in a.thr_list:
        for hold in a.hold_list:
            outdir=f"out_thr{thr}_h{hold}"; pfile=f"conf/params_thr{thr}_h{hold}.yml"
            patch_params(a.params, pfile, thr, hold)
            saved=sys.argv[:]
            sys.argv=[runner_path,"--data-root",a.data_root,"--csv-glob",csvg,"--params",pfile,"--outdir",outdir]
            try: 
                import runpy; runpy.run_path(runner_path,run_name="__main__")
            finally: sys.argv=saved
            post_enrich(outdir)
            zpath=f"sweep_{thr}_{hold}.zip"; zip_dir(outdir,zpath); out_zips.append(zpath)

    bundle=a.out_bundle or "sweep_vec_bundle_enforced_local.zip"
    with zipfile.ZipFile(bundle,"w",zipfile.ZIP_DEFLATED) as z:
        for zp in out_zips: z.write(zp, os.path.basename(zp))
    print(f"[bundle] {bundle} ({len(out_zips)} items)")

if __name__=="__main__": main()
