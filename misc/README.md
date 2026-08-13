# Herbie precision audit

Floating-point audit of the sampler's numeric kernels with
[Herbie](https://herbie.uwplse.org/) 2.3 (`racket -l herbie -- improve/report
--seed 1 permanent.fpcore ...`).

- `permanent.fpcore` — the five kernels in FPCore form, with preconditions
  encoding the algorithm's invariants (weight range `[1/n, 1e12]`,
  `a + b <= W`, `W >= 1`, ...).
- `permanent-improved.fpcore` — Herbie's suggested rewrites.
- `herbie-report/` — the HTML report with per-kernel error measurements.

Findings (average bits of error over the sampled domain):

| kernel | before | Herbie best | disposition |
|---|---|---|---|
| acceptance `((W-a-b+c+d)/W)*E` | — | unchanged | accurate on the real domain |
| update `W' = W-a-b+c+d` | 0.335 | 0.047 (min/max ordering) | superseded: `src/chain.rs` tracks W with Neumaier compensation (exact) |
| normalize `1/(x*(n/S))` | 0.353 | 0.260 (`(S/x)/n`) | applied as `S/(x*n)` in `src/markov_chain.rs` |
| `exp(diff*k)` | 0.000 | 0.023 (worse) | keep `exp`; the suggested constant is a search artifact |
| `1/W + 2*eps` | 0.0005 | unchanged | fine |

The accelerator crossover investigation and its FP32 precision measurements
are recorded in [gpu-parallelism.md](gpu-parallelism.md).

The implementation now uses `L/W + 2*eps`, where `L` is a shared
matching-weight lower bound. Multiplication by that step-constant does not
change the audit's conclusion about the division.
