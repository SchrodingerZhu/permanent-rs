# permanent

Estimates the permanent of a nonnegative matrix to a requested relative
accuracy. You give the matrix, `--epsilon` and `--delta`; the number of
temperatures, the samples at each, and the sampler's run length are derived.

    cargo run --release -- data/weighted-16.json --exact
    cargo run --release -- data/grid-8x8.json -e 0.05
    cargo run --release -- data/cyc-64-m4.json --plan-only

Input is JSON in one of three forms: `{"matrix": [[...], ...]}` (dense),
`{"size": n, "entries": [[row, col, weight], ...]}` (sparse), or the 0/1
form `{"size": n, "edges": [[columns of row 0], ...]}`.

## Method

The algorithm follows Chen, Guo, Vigoda and Yang, "An Õ(n³)-time FPRAS for
the permanent" (arXiv:2609.20717).

1. **Scaling** (`scaling.rs`). A maximum-product matching and its dual
   potentials turn A into a row-stochastic A' whose matched entries are row
   maxima. Entries below tau = P*/(2 n!) are raised to tau, which makes
   every entry positive. The identity per(A') = per(A~) E[g] is exact, and
   E[g] >= 2/3.
2. **Annealing.** Activities are a~(u,v)^beta for beta from 0 (uniform over
   all n! permutations) to 1.
3. **Sampler** (`sampler.rs`). Independent perfect matchings are drawn by
   annealing relaxed assignments. Only n anchor weights are needed, and the
   output distribution is correct for any weights; poor weights only lower
   the acceptance rate.
4. **Learning** (`far.rs`). The anchor weights at each temperature are
   learned with the FAR chain, counted over cycles between perfect states
   (Kac's formula), which needs no mixing-time assumption.
5. **Estimate** (`estimator.rs`). per(A) = n! · ∏ E[exp(-d_i H)] · E[g] / c.
   Samples are independent, so the standard error is measured from them and
   sampling continues until it meets the target.

## What is derived, and what backs it

| Quantity | How it is chosen |
|---|---|
| Temperatures | Adaptive: the largest step keeping the ratio's relative variance near 0.25 and the weights within a factor 2, never below the paper's step. `--rigorous-schedule` uses the paper's step only. |
| Samples per temperature | Proportional to the square root of each term's variance, sized so the standard error of ln(per) is ln(1+epsilon)/z; topped up from measured variances. |
| Sampler time per stage | Doubled until two independent estimates agree within 2 standard errors; the longer run is reported. `--rigorous-sampler` uses the paper's explicit time instead (hundreds of times longer). |

The reported interval is statistical: it assumes the sampler's draws follow
the target distribution. In the default mode that assumption rests on the
doubling check, not on a theorem. The paper leaves one constant of the
sampler's activity schedule unspecified (A_alpha); `--density` sets it and
defaults to the paper's J_alpha = 32.

## Layout

    src/problem.rs    input formats
    src/scaling.rs    Hungarian method, scaling, regularization
    src/tables.rs     per-temperature B matrix and alias tables
    src/sampler.rs    relaxed-assignment annealing sampler
    src/far.rs        FAR chain and cycle counting
    src/estimator.rs  schedule, planning, estimation, validation
    src/exact.rs      exact subset DP for n <= 26

This rewrite is CPU-only (rayon). The earlier JSV/BSVV implementation with
its CubeCL and CUDA backends and the TUI is in the history before this
branch.
