# permanent

Estimates the permanent of a nonnegative matrix to a requested relative
accuracy. You give the matrix, `--epsilon` and `--delta`; the temperatures,
the population and the number of independent runs are derived.

    cargo run --release -- data/weighted-16.json --exact
    cargo run --release -- data/grid-8x8.json -e 0.05
    cargo run --release -- data/cyc-64-m4.json --plan-only
    cargo run --release -- data/cyc-64-m4.json --tui --log-file run.log

Input is JSON in one of three forms: `{"matrix": [[...], ...]}` (dense),
`{"size": n, "entries": [[row, col, weight], ...]}` (sparse), or the 0/1
form `{"size": n, "edges": [[columns of row 0], ...]}`.

The report goes to stdout. Progress is logged to stderr (`-v` for more,
`PERMANENT_LOG_LEVEL` to filter, `--log-file` to redirect); `--tui` shows
the run in a dashboard instead.

## Method

The ingredients are those of Chen, Guo, Vigoda and Yang, "An Õ(n³)-time
FPRAS for the permanent" (arXiv:2609.20717). What differs is how samples
are produced: the paper draws every sample from nothing; here a population
of perfect matchings is carried from one temperature to the next, and the
paper's sampler is what cuts that population into independent pieces.

1. **Scaling** (`scaling.rs`). A maximum-product matching and its dual
   potentials turn A into a row-stochastic A' whose matched entries are row
   maxima. Entries below tau = P*/(2 n!) are raised to tau, which makes
   every entry positive. The identity per(A') = per(A~) E[g] is exact.
2. **Annealing.** Activities are a~(u,v)^beta for beta from 0 (uniform over
   all n! permutations) to 1, so
   per(A) = n! · ∏ E_i[exp(-d_i H)] · E[g] / c.
3. **Learning** (`far.rs`). The n anchor weights of each temperature are
   learned with the FAR chain, counted over cycles between perfect states
   (Kac's formula). The identity holds per cycle, so every particle of a
   population runs a few cycles instead of a few chains running many, and
   the runs are also what moves that population between temperatures.
4. **Population** (`estimator.rs`, `sampler.rs`). P particles at one
   temperature are weighted by exp(-d H), resampled, and moved by the
   relaxed-assignment dynamics at the top activity, run until time t has
   been spent in perfect matchings. That move leaves the next temperature's
   distribution invariant for any weights and contracts it by exp(-t)
   (the paper's spectral gap of 1 carries over to the chain watched on
   perfect matchings).
5. **Segments.** The temperature ladder is cut into S segments. Each one
   after the first starts from fresh draws of the paper's annealing sampler,
   so segments are independent, multiply without bias, and run at the same
   time on different ranks.
6. **Islands.** Every segment is run R times independently; the error bar
   is the measured spread of those runs.

## What is guaranteed, and what backs it

| Statement | Backing |
|---|---|
| The estimate is unbiased for every P, t and any anchor weights, once the schedule and weights are fixed. | Unnormalized particle estimates are unbiased (Del Moral 2004, Section 7.4); segment starts are exact at beta = 0 and from the sampler elsewhere. Checked by a test with 4 particles per island. |
| Its relative variance is about v / P with v ≤ (1 − √g e^(−t))^(−2) · Σ relvar_i. | Asymptotic in P, from the exp(-t) contraction; g bounds a step's largest weight over its mean. Measured, v has stayed within 1.2 of Σ relvar_i, far below the bound. |
| The error bar is honest. | Islands are i.i.d. and unbiased; the bar is their spread, and islands are added until it meets the target. |
| Segment starts follow the target. | `--rigorous-sampler` runs the sampler for the paper's explicit time. Otherwise the boundary check compares, at every segment boundary, the particles carried there with the fresh draws; both target the same distribution. |

Islands are sized for twice the perfect-mixing variance Σ relvar_i / P and
rerun larger if the first ones measure noisier than planned, so the bound
is a ceiling on P, not its size. A wider machine holds more particles per
sweep, and the error falls as 1/√(particles) at the same number of sweeps.

[misc/theory.md](misc/theory.md) has the proofs, the derivation of every
sizing rule, and an enumeration check of them (`misc/verify_variance.py`).

Against exact permanents of twelve weighted matrices (n = 12 to 24: dense
and sparse, lognormal with entries spread over e^±12, entries spanning
twelve decades, near-permutations, weakly coupled blocks) and one 0/1
matrix, 40 seeds each on the CPU and the GPU engine, the reported interval
contained the exact value in 1015 of 1040 runs (97.6%), and (estimate −
exact) / (reported error) had mean between −0.26 and 0.30 and standard
deviation 0.7 to 1.3 on every matrix.

Past the reach of exact methods, weighted square-lattice dimer models have
Kasteleyn's closed form (`misc/dimers.py`): a 512 × 512 matrix (32 × 32
lattice, bond weights 1 and 0.5) comes out at ln Z = 149.280 ± 0.038
against the exact 149.234, in 11 minutes on one GPU, and a 256 × 256 one is
within 1.2 standard errors on each of six seeds, two minutes a run. The
tables are in the theory note.

## GPU

`cuda/` is the same program with the population resident on an NVIDIA GPU.
Its kernels are Rust, compiled with cuda-oxide
(https://github.com/NVIDIA/cuda-rust), which needs its pinned toolchain,
LLVM 21+ and CUDA 13+:

    cd cuda
    cargo oxide build
    target/release/permanent-cuda ../data/cyc-64-m4.json
    cargo oxide test -- --release      # needs a GPU

Every population operation runs on the device. A particle's working state
is its matching, the inverse and a short list of collisions, staged in
shared memory while a warp's worth fits (n up to about 370) and kept in
global memory beyond. FAR chains have very uneven lengths, so a FAR launch
gives each chain a step budget and the few that overrun it are finished on
host threads.

## MPI

Built with `--features mpi`, every rank runs the same driver on its own
share of the work (and, in the GPU build, its own device):

    cargo build --release --features mpi
    mpirun -n 4 target/release/permanent data/cyc-64-m4.json

Ranks split the segments and islands of Phase 2, which share nothing, and
pool their learning statistics in Phase 1. Results do not depend on the
number of ranks beyond the split of the Phase 1 population.

## Nix

    nix build              # the CPU program, tests included
    nix build .#mpi        # with MPI
    nix develop            # cargo and rustc
    nix develop .#mpi      # plus an MPI and libclang for its bindings
    nix develop .#cuda     # the GPU crate's pinned toolchain and CUDA 13

The GPU crate is built inside the `cuda` shell with `cargo oxide`, which
the shell tells you how to install; the NVIDIA driver comes from the host.

## Layout

    src/problem.rs    input formats
    src/scaling.rs    Hungarian method, scaling, regularization
    src/tables.rs     per-temperature B matrix and alias tables
    src/sampler.rs    relaxed-assignment dynamics: annealing sampler, mutation
    src/far.rs        FAR chain and cycle counting
    src/engine.rs     population operations; the CPU engine
    src/comm.rs       reductions across MPI ranks
    src/estimator.rs  schedule, planning, segments, islands
    src/philox.rs     Philox4x32-10, shared with the GPU kernels
    src/cli.rs        command line, report, logging
    src/tui.rs        dashboard
    src/exact.rs      exact subset DP for n <= 26
    cuda/             GPU engine and kernels
