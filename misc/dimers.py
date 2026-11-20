#!/usr/bin/env python3
"""Dimer models as permanents: inputs, exact references, and runs.

The dimer coverings of a bipartite lattice are the perfect matchings between
its two sublattices, so the partition function with bond weights is the
permanent of the weighted biadjacency matrix. In two dimensions Kasteleyn's
formula gives it exactly at any size, which makes weighted square lattices
a reference far past where brute force can check; in three there is no
closed form.

    python3 misc/dimers.py check      # the references against known counts
    python3 misc/dimers.py 2d         # square lattices against Kasteleyn
    python3 misc/dimers.py 3d         # cubic lattices
    python3 misc/dimers.py 3d-big     # the 8 x 8 x 8 torus
    python3 misc/dimers.py 3d-aniso   # layers coupled by weak bonds

DIMER_DIR is where the inputs go (default ./dimers) and PERMANENT the
program to run (default permanent, on PATH), with any arguments, e.g.
PERMANENT="target/release/permanent --backend cuda".
"""
import itertools, json, math, os, re, shlex, subprocess, sys, time
import scipy.io, scipy.sparse

OUT = os.environ.get("DIMER_DIR", "dimers")
PERMANENT = shlex.split(os.environ.get("PERMANENT", "permanent"))

def lattice(dims, periodic, weights):
    """Weighted biadjacency matrix of a hypercubic lattice; weights[d] is the bond weight along axis d."""
    sites = list(itertools.product(*[range(n) for n in dims]))
    even = [s for s in sites if sum(s) % 2 == 0]
    odd = {s: i for i, s in enumerate(s for s in sites if sum(s) % 2 == 1)}
    assert len(even) == len(odd)
    rows, cols, values = [], [], []
    for row, s in enumerate(even):
        for d, n in enumerate(dims):
            for step in (-1, 1):
                t = list(s); t[d] += step
                if periodic:
                    assert n >= 4 and n % 2 == 0
                    t[d] %= n
                elif not 0 <= t[d] < n:
                    continue
                rows.append(row); cols.append(odd[tuple(t)]); values.append(float(weights[d]))
    # two bonds between the same sites (a period of 2 would give them) add up
    return scipy.sparse.coo_matrix((values, (rows, cols)), shape=(len(even), len(even)))

def write(path, matrix, comment=""):
    scipy.io.mmwrite(path, matrix, comment=comment, field="real", precision=17, symmetry="general")

def kasteleyn_open(m, n, a, b):
    """ln Z of the m x n open square lattice (m, n even): bonds along the side of length m weigh a, the others b."""
    total = 0.0
    for j in range(1, m // 2 + 1):
        for k in range(1, n // 2 + 1):
            total += math.log(4 * a * a * math.cos(math.pi * j / (m + 1)) ** 2 + 4 * b * b * math.cos(math.pi * k / (n + 1)) ** 2)
    return total

def count_open_cube(l):
    """Dimer coverings of the open l x l x l cube, by a cell-by-cell profile DP."""
    span = l * l
    states = {0: 1}
    for cell in range(l ** 3):
        x, y, z = cell % l, (cell // l) % l, cell // span
        nxt = {}
        for mask, ways in states.items():
            if mask & 1:
                nxt[mask >> 1] = nxt.get(mask >> 1, 0) + ways
                continue
            for ok, bit in ((x + 1 < l, 1), (y + 1 < l, l), (z + 1 < l, span)):
                if ok and not mask & (1 << bit):
                    m = (mask | (1 << bit)) >> 1
                    nxt[m] = nxt.get(m, 0) + ways
        states = nxt
    return states.get(0, 0)

def run(name, problem, extra=()):
    os.makedirs(OUT, exist_ok=True)
    path = f"{OUT}/{name}.mtx"
    write(path, problem)
    start = time.time()
    proc = subprocess.run([*PERMANENT, path, *extra], capture_output=True, text=True, env={**os.environ, "PERMANENT_LOG_LEVEL": "error"})
    text = proc.stdout
    m = re.search(r"ln\(permanent\) = ([-\d.]+) ± ([\d.]+)", text)
    info = {"n": problem.shape[0], "seconds": time.time() - start}
    if m:
        info["ln"], info["se"] = float(m.group(1)), float(m.group(2))
    info["temperatures"] = int(re.search(r"temperatures:\s+(\d+)", text).group(1)) if "temperatures:" in text else None
    info["islands"] = re.findall(r"(\d+) islands", text)[-1:] if "islands" in text else None
    info["warnings"] = [l.strip() for l in text.splitlines() if "WARNING" in l]
    if proc.returncode != 0:
        info["error"] = proc.stderr[-300:]
    return info

if __name__ == "__main__":
    which = sys.argv[1]
    eps = ["-e", "0.05"]
    os.makedirs(OUT, exist_ok=True)
    if which == "check":
        assert round(math.exp(kasteleyn_open(8, 8, 1, 1))) == 12988816, "Kasteleyn formula"
        print("8x8 open, a=b=1: exp(formula) =", round(math.exp(kasteleyn_open(8, 8, 1, 1))))
        # orientation of a and b, against the program's own exact value on a small anisotropic lattice
        print("4x6 open, a=1 b=0.5: formula", math.exp(kasteleyn_open(4, 6, 1.0, 0.5)), "| the two orientations:", math.exp(kasteleyn_open(6, 4, 1.0, 0.5)))
        write(f"{OUT}/small.mtx", lattice((4, 6), False, (1.0, 0.5)))
        out = subprocess.run([*PERMANENT, f"{OUT}/small.mtx", "--exact"], capture_output=True, text=True, env={**os.environ, "PERMANENT_LOG_LEVEL": "error"}).stdout
        print([l for l in out.splitlines() if "exact" in l])
        print("open 4x4x4 cube, exact count by DP:", count_open_cube(4), "| 2x2x2:", count_open_cube(2))
    elif which == "2d":
        for (m, n, a, b) in [(16, 16, 1.0, 1.0), (16, 16, 1.0, 0.5), (16, 16, 1.0, 0.1), (12, 20, 2.0, 0.3)]:
            r = run(f"square-{m}x{n}-{a}-{b}", lattice((m, n), False, (a, b)), eps)
            exact = kasteleyn_open(m, n, a, b)
            r.update(exact=exact, z=(r["ln"] - exact) / r["se"] if "ln" in r else None)
            print(f"{m}x{n} a={a} b={b}", json.dumps(r), flush=True)
    elif which == "3d":
        exact4 = math.log(count_open_cube(4))
        r = run("cube-open-4", lattice((4, 4, 4), False, (1.0, 1.0, 1.0)), eps)
        r.update(exact=exact4, z=(r["ln"] - exact4) / r["se"])
        print("open 4x4x4", json.dumps(r), flush=True)
        for dims in [(4, 4, 4), (4, 4, 6), (4, 6, 6), (6, 6, 6), (6, 6, 8)]:
            r = run("torus-" + "x".join(map(str, dims)), lattice(dims, True, (1.0, 1.0, 1.0)), eps)
            sites = dims[0] * dims[1] * dims[2]
            if "ln" in r:
                r.update(sites=sites, entropy_per_site=r["ln"] / sites, entropy_error=r["se"] / sites)
            print("torus", dims, json.dumps(r), flush=True)
    elif which == "3d-big":
        dims = (8, 8, 8)
        r = run("torus-8x8x8", lattice(dims, True, (1.0, 1.0, 1.0)))
        if "ln" in r:
            r.update(sites=512, entropy_per_site=r["ln"] / 512, entropy_error=r["se"] / 512)
        print("torus", dims, json.dumps(r), flush=True)
    elif which == "3d-aniso":
        # layered limit: weak bonds along z interpolate between decoupled square lattices and the cubic lattice
        for wz in [0.0, 0.25, 0.5, 1.0, 2.0]:
            r = run(f"torus-6x6x6-z{wz}", lattice((6, 6, 6), True, (1.0, 1.0, wz)) if wz > 0 else lattice((6, 6, 6), True, (1.0, 1.0, 1e-300)), eps)
            if "ln" in r:
                r.update(free_energy_per_site=r["ln"] / 216, error=r["se"] / 216)
            print("6x6x6 torus, z-bond weight", wz, json.dumps(r), flush=True)
