# Random bipartite scaling measurements

Question: the parallel-ensemble annealer reproduces grid permanents, but the
Štefankovič–Vempala–Vigoda scheme was previously measured infeasible in
practice (Newman–Vardi) with a single chain. Does the ensemble version scale
to random bipartite graphs, and *why* does the ensemble change the picture?

Instances are planted-perfect-matching `G(n, n, p)` graphs from
`gen_random_bipartite.py` (seeded; committed under `data/random-*.json`).
All runs: CPU backend, 4 cores, release build.

## Does it work on random bipartite graphs?

| n | avg deg | config | schedule | wall | health | result |
|---|---|---|---|---|---|---|
| 16 | 3.7 | 512 ch × 512 spl, W=16 | 1,280 steps (full) | 29 s | clean | 2.26e3 vs exact 2220 (**1.9% err**) |
| 32 | 4.8 | 1024 ch × 512 spl, W=32 | 3,840 steps (full) | 299 s | clean | 3.00e7 / 3.55e7 (two runs) |
| 64 | 5.6 | 1024 ch × 512 spl, W=64 | 10,752 steps (full) | 1,564 s | 4 borderline re-equilibrations | 7.65e19 |
| 128 | 7.9 | 2048 ch × 512 spl, W=128 | 3,584 of 28,672 steps (capped) | 2,699 s | 34 borderline re-equilibrations | trend only |

Health indicators across all sizes: per-step ratio estimates stay near 1
(n=64: mean 1.002, min 0.458 over 10,752 steps), the perfect-class occupancy
stays inside its `[expected/2, expected*2]` band (n=64 grazed the lower edge
4 times in 10,752 steps, each recovering after one retry), and the
hole-weight update never clamps. That is the same clean profile the grids
show: nothing about sparse random bipartite structure (degree ~5-8,
irregular, planted matching) breaks the anneal.

The n=16 instance doubles as an accuracy check (exact Ryser value 2220);
n=32+ are past exact verification (Ryser is 2^n · n), so n=32 was
validated by run-to-run repeatability instead: two independent anneals
gave 3.00e7 and 3.55e7, an 0.17 ln-space spread at this sample budget.

The capped n=128 run completed the entire additive phase (3,584 steps to
beta = 7.0). By the end the running `Z * Pr[perfect and fully active]`
estimator had already left zero (≈5.4e57), i.e. fully-active perfect
matchings of the real graph were being sampled — the anneal was tracking,
not stalling. Its 34 re-equilibration events (of 3,584 steps) were all
single-retry recoveries of a perfect-class occupancy grazing the band
edge; the weight update never clamped and no ratio collapsed below 0.3.

## Why the ensemble makes the difference

Per cooling step the weight bootstrap needs an occupancy histogram over the
n^2 + 1 hole classes, and the ratio estimator needs a batch of stationary
samples. The scheme's practicality hinges on the cost of one *effective*
(independent-ish, stationary) sample:

- **Single chain (Newman–Vardi regime):** independence must be bought with
  time — successive samples are spaced by a decorrelation interval taken
  from the mixing-time analysis (the theoretical bound is Õ(n^7 log^4 n)),
  and the cost is `interval × samples`, serial. The interval multiplies
  into every one of the ~n log^2 n × n samples of the whole run.
- **Ensemble:** samples taken across chains are independent by
  construction. The intra-chain interval only needs to keep each chain
  *tracking* a target distribution that the schedule moves by design in
  O(1/n) increments — every chain is warm-started at ≈stationarity from the
  previous step, so the full mixing price is paid once at warmup, not per
  sample.

Ladder experiment on `random-16-d4` (exact permanent 2220; total per-step
sample budget fixed at 262,144 for the first four arms; the NV-style arm
instead fixes the *proposal* budget to match arm 1, i.e. 4.2M
proposals/step):

| chains × samples/step | interval W | rel. err | warnings | wall |
|---|---|---|---|---|
| 512 × 512 | 16 | 1.9% | 0 | 29 s |
| 64 × 4096 | 16 | 0.5% | 0 | 30 s |
| 8 × 32768 | 16 | 9.0% | 0 | 55 s |
| 1 × 262144 | 16 | 7.5% | 0 | 108 s |
| 1 × 2048 | 2048 (NV-style) | **14.8%** | **629** | 119 s |

Two separate effects are visible:

1. **Correlated samples inflate estimator variance.** Arms 3-4 keep the
   sample count but draw it from few/one chain: the hole pair performs a
   local walk, so successive samples' class labels are strongly
   autocorrelated, the effective sample size of the occupancy histogram
   drops, and the telescoping product accumulates the extra noise
   (error grows from ~1% to ~8-9% at identical budget). The chain never
   *stalls* at n=16 — 4.2M proposals/step sweep the 257 classes many times
   — it just estimates worse.
2. **Decorrelating by time starves the histogram.** The NV-style arm pays
   the same proposals but keeps only every 2048th state, leaving ~4 counts
   per class for the weight update. The update then clamps ~65 of 256
   classes per step, the perfect-class occupancy leaves its band 511 times
   (each triggering re-equilibration retries), per-step ratios dip to 0.35,
   and the final estimate lands 15% off — on an n=16 toy. The per-sample
   histogram requirement is Θ(n^2), so this failure mode compounds
   quadratically, matching the reported infeasibility of the faithful
   single-chain implementation at any interesting n.

The ensemble sidesteps both at once: cross-chain independence gives the
per-step histogram its Θ(n^2) *nearly independent* counts at interval-16
cost, and the warm-started population turns "mix from scratch per sample"
into "track a slowly-moving distribution with 16×-per-sample stirring".
Parallel hardware then makes the chains-dimension nearly free (rayon here,
one GPU invocation per chain in the GPU backend).

A third, quieter benefit: the health checks themselves (occupancy band,
clamp census) are only *observable* with ensemble-sized per-step counts. A
single chain cannot even tell you cheaply that it has lost stationarity.

## Cost trend

Measured s/step (4 cores) and full-schedule projections, `q=512` samples
per chain per step, `W=n` stirring:

| n | steps total | s/step (avg) | full run |
|---|---|---|---|
| 16 | 1,280 | 0.022 | 29 s |
| 32 | 3,840 | 0.078 | 5.0 min |
| 64 | 10,752 | 0.146 | 26 min |
| 128 | 28,672 | 0.75 (0.52 by last decile) | ~5-7 h (projected) |

Per-step work scales as `chains × q × W` with `W ≈ n`, and the step count
as Θ(n log^2 n) — every observed increment matches that arithmetic, with
no sign of the health-driven retries that would signal super-scheduled
mixing demands. Extrapolation past n=128 is gated by compute, not by any
observed breakdown.

## Reproducing

```bash
python3 misc/gen_random_bipartite.py 64 6 3 data/random-64-d6.json
cargo run --release -- -g data/random-64-d6.json -n 1024 -q 512 -W 64 -e 256
```
