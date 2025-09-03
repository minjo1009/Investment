import argparse, json, os, subprocess, sys, time
from pathlib import Path

try:
    import psutil
except Exception:  # pragma: no cover - psutil optional
    psutil = None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', required=True)
    ap.add_argument('--csv-glob', required=True)
    ap.add_argument('--params', required=True)
    ap.add_argument('--limit-bars', type=int, default=100000)
    args = ap.parse_args()

    outdir = Path('bench_out')
    if outdir.exists():
        for p in outdir.iterdir():
            if p.is_file():
                p.unlink()
    else:
        outdir.mkdir()

    cmd = [
        sys.executable, 'backtest/runner_patched.py',
        '--data-root', args.data_root,
        '--csv-glob', args.csv_glob,
        '--params', args.params,
        '--outdir', str(outdir),
        '--limit-bars', str(args.limit_bars),
        '--debug-level', 'none',
        '--no-preds'
    ]
    env = os.environ.copy()
    env.setdefault('PYTHONPATH', '.')

    start = time.time()
    subprocess.check_call(cmd, env=env)
    wall = time.time() - start

    rss = None
    if psutil is not None:
        try:
            rss = max(p.memory_info().rss for p in psutil.Process().children())
        except Exception:
            rss = None

    sizes = {p.name: p.stat().st_size for p in outdir.iterdir() if p.is_file()}
    print(json.dumps({'wall_time': wall, 'rss': rss, 'files': sizes}, indent=2))


if __name__ == '__main__':
    main()
