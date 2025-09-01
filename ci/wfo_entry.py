import argparse, os, sys, runpy, zipfile, yaml

def overlay_params(base_path, out_path, thr=None, hold=None):
    d=yaml.safe_load(open(base_path,"r",encoding="utf-8")) or {}
    if thr is not None:
        d.setdefault("entry",{}).setdefault("p_thr",{})
        d["entry"]["p_thr"]["trend"]=float(thr); d["entry"]["p_thr"]["range"]=float(thr)
    if hold is not None:
        d.setdefault("exit",{}); d["exit"]["min_hold"]=int(hold)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    yaml.safe_dump(d, open(out_path,"w",encoding="utf-8"), sort_keys=False, allow_unicode=True)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--params",required=True)
    ap.add_argument("--data-root",required=True)
    ap.add_argument("--csv-glob",required=True)
    ap.add_argument("--outdir",required=True)
    ap.add_argument("--thr",type=float,default=None)
    ap.add_argument("--hold",type=int,default=None)
    ap.add_argument("--codepack",default="strategy_v2_codepack_v2.1.3.zip")
    ap.add_argument("--runner",default="backtest/runner_patched.py")
    a=ap.parse_args()

    # glob 정규화
    csvg=a.csv_glob.lstrip("./"); rootname=os.path.basename(os.getcwd().rstrip("/"))
    if csvg.startswith(rootname+"/"): csvg=csvg[len(rootname)+1:]
    while True:
        parts=csvg.split("/",1)
        if len(parts)==2 and parts[0].lower().replace("_"," ").startswith("multiregime 4t"):
            csvg=parts[1]
        else: break

    used=os.path.join(a.outdir,"params_used.yml"); overlay_params(a.params, used, a.thr, a.hold)

    # 코드팩 unzip
    if a.codepack and os.path.exists(a.codepack):
        import shutil, zipfile
        if os.path.exists("_codepack"): shutil.rmtree("_codepack")
        with zipfile.ZipFile(a.codepack) as z: z.extractall("_codepack")

    runner_path=a.runner
    if not os.path.exists(runner_path):
        raise FileNotFoundError(f"runner not found: {runner_path}")
    sys.path[:0]=[os.getcwd(), "_codepack", os.path.join("_codepack","backtest")]

    os.makedirs(a.outdir, exist_ok=True)
    saved=sys.argv[:]
    sys.argv=[runner_path,"--data-root",a.data_root,"--csv-glob",csvg,"--params",used,"--outdir",a.outdir]
    print("[wfo_entry] exec", runner_path, "argv:", sys.argv[1:])
    try: runpy.run_path(runner_path, run_name="__main__")
    finally: sys.argv=saved

if __name__=="__main__": main()
