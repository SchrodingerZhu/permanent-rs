# GPU parallelism audit

Measured on an AMD Ryzen AI MAX+ 395 and its integrated Radeon 8060S, using
CubeCL 0.10 over Vulkan. Timings are release builds and should be treated as
machine-specific crossover measurements, not portable constants.

## Decision and integrated result

The GPU implementation is integrated into the normal executable and selected
with `--backend gpu`; `--backend cpu` remains the default. It uses compensated
`f32`, assigns one invocation to each chain, and keeps permutations, RNG state,
active counts, and matching weights resident across cooling steps. Transitions
within one chain are data-dependent and remain serial; independent chains are
the useful source of GPU parallelism.

Do not create standalone offloads for small reductions, normalization, graph
loading, Dinic, the TUI, or a single exact Ryser call. Their work is too small
to repay dispatch and initialization. The integrated path therefore reads the
compact histogram back and normalizes it on the CPU. The estimator's rejection
loop is included in the backend. It is dispatched in small chunks because a
data-dependent shader `break` crashes CubeCL 0.10's SPIR-V lowering on this
RADV stack. Completed chains skip the remainder of a chunk; the host only
launches another chunk when at least one chain remains.

An integrated `n = 32` comparison with 2,048 chains, 2,048 weight samples at
interval 16, and 64 estimator samples at interval 128 measured:

| run | CPU | GPU evolve | GPU initialization | outcome |
|---|---:|---:|---:|---|
| 1 cooling step | 58.0 ms | 35.3 ms | 43.5 ms | CPU wins cold total |
| 2 cooling steps | 140.1 ms | 60.6 ms | 35.5 ms | GPU wins total, about 1.5x |
| 10 cooling steps | 822.7 ms | 271.8 ms | 49.7 ms | GPU wins total, about 2.6x; evolve is 3.0x |

Thus this workload repays cold startup after roughly two cooling steps. A real
run has hundreds or thousands. The sparse `n = 8` accuracy configuration is
too small: GPU took about 0.47 s end to end versus about 0.38 s on CPU, so the
CLI exposes the choice instead of pretending GPU is universally faster.

## Warm crossover

For `n = 32`, 2,048 chains, interval 16, and a warm GPU pipeline:

| samples per chain | CPU (32 threads) | GPU dispatch + completion | result |
|---:|---:|---:|---|
| 1 | 0.43–0.76 ms | 0.70–0.75 ms | no reliable GPU win |
| 8 | 0.54–0.90 ms | 0.79–0.88 ms | no reliable GPU win |
| 64 | 1.48–2.32 ms | 1.37–1.43 ms | small GPU win |
| 512 | 10.4–10.9 ms | 5.44–5.60 ms | about 1.9x GPU |
| 2,048 | 40.9–45.2 ms | 19.9–20.8 ms | about 2.1x GPU |

GPU runtime initialization cost 29–55 ms, first pipeline compilation another
5–12 ms, and an already-compiled empty dispatch plus synchronization about 49
microseconds. A one-off call does not win; thousands of cooling steps amortize
the cold cost easily.

The table is a phase-level persistent-kernel measurement: buffer
creation/upload happens before the warm timing, and chain state remains on the
GPU. The integrated backend reads back the O(n^2) histogram and O(chains)
estimator reductions once per cooling step; despite that synchronization, its
measured end-to-end kernel speedup is 3.0x on the default `n = 32` work shape.

Retiling the same 4,194,304 samples from `2,048 x 2,048` to `8,192 x 512`
reduced the GPU time to 10.7–11.0 ms. The best CPU layout measured about 39 ms,
so the likely sustained ceiling for this phase is roughly 3.5–3.6x. Changing
the chain/sample decomposition can change statistical behavior and must be
validated before changing defaults; the same-layout 2.1x result is the safer
claim.

At the default estimator shape (2,048 chains, 64 requests, interval 128), the
CPU took 10.0 ms on the initial step. The early GPU probe took 4.24–4.33 ms for
the same 16,777,216 transitions while also doing an O(n) weighted scan and an
atomic increment per request. This is evidence that the estimator phase is
also worth porting. It is not an end-to-end estimator measurement: later steps
have variable rejection counts and warp divergence. For the initial step, the
measured CPU weight-plus-estimator time was 56.9 ms; summing the same-layout
GPU measurements gives about 24.7–25.1 ms, or 2.3x, before small GPU-side
reductions.

The default 2,048-chain warmup (16,384 transitions per chain) took 22.0 ms on
the CPU and 7.15 ms in the warm GPU transition kernel. It loses if used as a
standalone cold GPU call, but wins once the same resident backend is reused by
the cooling schedule.

The weighted-edge selection is an O(n) scan. At 8,192 chains and 512 samples,
the GPU times were 4.32, 6.40, 10.74, 18.86, 38.44, and 84.33 ms for `n` equal
to 8, 16, 32, 64, 128, and 256. A Fenwick or sum tree may become worthwhile at
larger `n`, but it also makes every accepted transition update the tree, so its
crossover needs a separate measurement. Histogram atomic contention added
only about 12% in a worst-case single-bin stress test.

## Precision

The GPU backend uses `f32` for weights, acceptance arithmetic, and cached
exponentials with Neumaier-style compensated updates to each matching's total
weight. In an adversarial CPU shadow test spanning weights from about `1/n` to
`1e12`, one million transitions produced relative total-weight drift on the
order of `1e-7` for compensated `f32`, compared with roughly `1e-14` for
compensated `f64`. That is appropriate for a stochastic estimator but is not a
formal proof of unchanged estimator bias. End-to-end checks include a complete
graph, which remains exactly 120, and the sparse `data/4-cycles.json`, which
returned 3.9601 against exact 4.0 (1.0% relative error) under the documented
accuracy configuration. An ignored Vulkan regression test preserves the latter
check.

With moderate, non-dyadic probe weights, the final GPU state itself showed a
maximum relative tracked-weight drift of `1.19e-7` after 67,108,864 aggregate
transitions.

FP64 is not attractive on this GPU. Exact Ryser is also a poor FP32 target due
to alternating-sum cancellation. The CPU implementation now uses block-parallel
Gray traversal and exact `i128` accumulation: complete `n = 24` fell from about
115 ms to 28 ms on 32 threads while fixing the former visible cancellation
error. That is already below the GPU cold-start cost.

## Larger wins found before offload

The original rejection sampler accepted with `1/W(M)`, although any shared
lower bound `L <= min_M W(M)` permits the much larger probability `L/W(M)`.
Computing the tight assignment lower bound with the Hungarian algorithm changed
an initial `n = 32`, one-estimator-sample step from about 1.46 s to 1.80 ms on
one CPU thread. This algorithmic win is far larger than accelerator speedup and
preserves the target distribution. The reported acceptance metric now counts
the actual inner rejection trials rather than only outer sample requests.

Replacing a shared atomic CPU histogram with Rayon worker-local histograms and
a reduction improved the default weight phase by roughly 15%.

## Running the backend

The same executable selects CPU or GPU. The default Nix shell supplies the
Vulkan loader, and the Nix package wraps the installed binary with the same
runtime library path:

```sh
nix develop
cargo run --release -- --backend gpu --graph-path data/grid-8x8.json
# CPU remains the default; this is equivalent to omitting --backend.
cargo run --release -- --backend cpu --graph-path data/grid-8x8.json
```

For a manual same-work crossover check, run the ignored `profile_phases` and
`profile_gpu_phases` tests with matching `PROFILE_*` variables. Normal tests do
not initialize Vulkan.
