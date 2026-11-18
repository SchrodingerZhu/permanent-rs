#!/usr/bin/env python3
"""Weighted-permanent benchmark: estimates against exact values over many seeds.

    python3 misc/bench_weighted.py gen       # write the test matrices
    python3 misc/bench_weighted.py run       # every matrix, seed and engine
    python3 misc/bench_weighted.py summary   # the table of misc/theory.md

Paths come from the environment: BENCH_DIR for the matrices and results
(default ./bench), PERMANENT_DIR for the checkout holding the built
binaries (default .). An engine whose binary is missing is skipped.
"""
import json, os, re, subprocess, sys, time
import numpy as np

ROOT = os.environ.get("BENCH_DIR", "bench")
PERM = os.environ.get("PERMANENT_DIR", ".")
ENGINES = {
    name: path
    for name, path in {
        "cpu": f"{PERM}/target/release/permanent",
        "gpu": f"{PERM}/cuda/target/release/permanent-cuda",
    }.items()
    if os.path.exists(path)
}
SEEDS = 40

def support(rng, n, density):
    """0/1 mask with a planted perfect matching."""
    mask = rng.random((n, n)) < density
    mask[np.arange(n), rng.permutation(n)] = True
    return mask

def matrices():
    out = {}
    rng = np.random.default_rng(20261118)
    out["dense-uniform-20"] = rng.random((20, 20))
    out["dense-lognormal2-20"] = np.exp(2.0 * rng.standard_normal((20, 20)))
    out["dense-lognormal4-18"] = np.exp(4.0 * rng.standard_normal((18, 18)))
    out["sparse-lognormal-22"] = np.exp(1.5 * rng.standard_normal((22, 22))) * support(rng, 22, 0.25)
    out["wide-range-18"] = 10.0 ** rng.uniform(-6, 6, (18, 18)) * support(rng, 18, 0.6)
    n = 20
    a = 10.0 ** rng.uniform(-4, -1, (n, n))
    a[np.arange(n), rng.permutation(n)] = rng.uniform(0.5, 1.5, n)
    out["near-permutation-20"] = a
    n = 24
    a = 1e-3 * rng.random((n, n))
    a[:12, :12] = rng.random((12, 12))
    a[12:, 12:] = rng.random((12, 12))
    out["two-blocks-24"] = a
    out["poisson-counts-20"] = rng.poisson(1.2, (20, 20)).astype(float) + np.eye(20)
    out["rank-one-noise-20"] = np.outer(rng.uniform(0.1, 3, 20), rng.uniform(0.1, 3, 20)) + 0.1 * rng.random((20, 20))
    out["zero-one-20"] = support(rng, 20, 0.3).astype(float)
    # no exact value: engines and seeds are compared with each other
    out["big-lognormal2-64"] = np.exp(2.0 * rng.standard_normal((64, 64)))
    out["big-sparse-lognormal-96"] = np.exp(1.5 * rng.standard_normal((96, 96))) * support(rng, 96, 0.15)
    return out

def gen():
    os.makedirs(ROOT, exist_ok=True)
    for name, a in matrices().items():
        json.dump({"matrix": a.tolist()}, open(f"{ROOT}/{name}.json", "w"))
    for name in ["weighted-12", "weighted-16", "weighted-20"]:
        subprocess.run(["cp", f"{PERM}/data/{name}.json", f"{ROOT}/data-{name}.json"], check=True)
    print(sorted(os.listdir(ROOT)))

def run_one(engine, path, seed, exact):
    cmd = [ENGINES[engine], path, "--seed", str(seed)] + (["--exact"] if exact else [])
    start = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, "PERMANENT_LOG_LEVEL": "error"})
    text = proc.stdout
    rec = {"engine": engine, "matrix": os.path.basename(path)[:-5], "seed": seed, "seconds": time.time() - start,
           "ok": proc.returncode == 0}
    m = re.search(r"ln\(permanent\) = ([-\d.]+) ± ([\d.]+)", text)
    if m:
        rec["ln"], rec["se"] = float(m.group(1)), float(m.group(2))
    m = re.search(r"relative error ([-+\d.]+)% \(([-+\d.]+) std errors\)", text)
    if m:
        rec["relerr"], rec["z"] = float(m.group(1)), float(m.group(2))
    m = re.search(r"largest difference ([-+\d.]+) standard errors", text)
    if m:
        rec["boundary"] = float(m.group(1))
    rec["warnings"] = [line.strip() for line in text.splitlines() if "WARNING" in line]
    m = re.search(r"E\[g\]:\s+([\d.]+)", text)
    if m:
        rec["gap"] = float(m.group(1))
    if not rec["ok"]:
        rec["stderr"] = proc.stderr[-300:]
    return rec

def run():
    names = sorted(f[:-5] for f in os.listdir(ROOT) if f.endswith(".json"))
    with open(f"{ROOT}/results.jsonl", "w") as out:
        for name in names:
            big = name.startswith("big-")
            for engine in ENGINES:
                for seed in range(1, (6 if big else SEEDS) + 1):
                    rec = run_one(engine, f"{ROOT}/{name}.json", seed, not big)
                    out.write(json.dumps(rec) + "\n")
                    out.flush()
    open(f"{ROOT}/DONE", "w").write("done")

def summary():
    recs = [json.loads(line) for line in open(f"{ROOT}/results.jsonl")]
    keys = sorted({(r["matrix"], r["engine"]) for r in recs})
    print(f"{'matrix':28} {'eng':3} {'runs':>4} {'fail':>4} {'mean z':>7} {'sd z':>6} {'max|z|':>6} {'|err|<10%':>9} "
          f"{'in CI':>6} {'mean se':>8} {'ln mean':>10} {'ln sd':>7} {'E[g]':>6} {'bnd>3.5':>7} {'warn':>4} {'sec':>6}")
    for matrix, engine in keys:
        rs = [r for r in recs if r["matrix"] == matrix and r["engine"] == engine]
        ok = [r for r in rs if r.get("ok") and "ln" in r]
        z = np.array([r["z"] for r in ok if "z" in r])
        ln = np.array([r["ln"] for r in ok])
        se = np.array([r["se"] for r in ok])
        err = np.array([abs(r["relerr"]) for r in ok if "relerr" in r])
        def f(x, fmt):
            return format(x, fmt) if np.isfinite(x) else "-"
        print(f"{matrix:28} {engine:3} {len(rs):4d} {len(rs) - len(ok):4d} "
              f"{f(z.mean() if len(z) else np.nan, '7.2f')} {f(z.std(ddof=1) if len(z) > 1 else np.nan, '6.2f')} "
              f"{f(np.abs(z).max() if len(z) else np.nan, '6.2f')} {f((err <= 10).mean() if len(err) else np.nan, '9.2f')} "
              f"{f((np.abs(z) <= 2.241).mean() if len(z) else np.nan, '6.2f')} {f(se.mean() if len(se) else np.nan, '8.4f')} "
              f"{f(ln.mean() if len(ln) else np.nan, '10.4f')} {f(ln.std(ddof=1) if len(ln) > 1 else np.nan, '7.4f')} "
              f"{f(np.mean([r.get('gap', np.nan) for r in ok]) if ok else np.nan, '6.3f')} "
              f"{sum(abs(r.get('boundary', 0)) > 3.5 for r in ok):7d} {sum(bool(r['warnings']) for r in ok):4d} "
              f"{f(np.mean([r['seconds'] for r in rs]), '6.2f')}")
    bad = [r for r in recs if not r.get("ok")]
    for r in bad[:5]:
        print("FAILED", r["matrix"], r["engine"], r["seed"], r.get("stderr", "")[-200:])

if __name__ == "__main__":
    {"gen": gen, "run": run, "summary": summary}[sys.argv[1]]()
