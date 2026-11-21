# permanent

Estimates the permanent of a nonnegative matrix to a requested relative
accuracy. You give the matrix, `--epsilon` and `--delta`; the temperatures,
the population and the number of independent runs are derived.

    cargo run --release -- data/weighted-16.mtx --exact
    cargo run --release -- data/dimer-8x8.mtx -e 0.05
    cargo run --release -- data/dimer-16x16.mtx --plan-only
    cargo run --release -- data/dimer-16x16.mtx --tui --log-file run.log

Input is a square matrix in Matrix Market coordinate format (`.mtx`), read
with `sprs`: `real` or `integer` entries, or `pattern` for a 0/1 matrix;
entries given twice add up. `data/` holds weighted examples: random
matrices small enough for `--exact`, and dimer models on square lattices
(`misc/dimers.py`), whose exact value by Kasteleyn's formula is in the
file's comment.

The report goes to stdout. Progress is logged to stderr (`-v` for more,
`PERMANENT_LOG_LEVEL` to filter, `--log-file` to redirect); `--tui` shows
the run in a dashboard instead.

## Method

The ingredients are those of Chen, Guo, Vigoda and Yang, "An Õ(n³)-time
FPRAS for the permanent" (arXiv:2609.20717). What differs is how samples
are produced: the paper draws every sample from nothing; here a population
of perfect matchings is carried from one temperature to the next, and the
paper's sampler is what cuts that population into independent pieces.

The modules named below are in `crates/permanent-core/src`.

1. **Scaling** (`scaling.rs`). A maximum-product matching and its dual
   potentials turn A into a row-stochastic A' whose matched entries are row
   maxima. Entries below tau = P*/(2 n!) are raised to tau, which makes
   every entry positive. The identity per(A') = per(A~) E[g] is exact.
2. **Annealing.** Activities are a~(u,v)^beta for beta from 0 (uniform over
   all n! permutations) to 1, so
   per(A) = n! · ∏ E_i[exp(-d_i H)] · E[g] / c.
3. **Learning** (`chain/far.rs`). The n anchor weights of each temperature are
   learned with the FAR chain, counted over cycles between perfect states
   (Kac's formula). The identity holds per cycle, so every particle of a
   population runs a few cycles instead of a few chains running many, and
   the runs are also what moves that population between temperatures.
4. **Population** (`estimator.rs`, `sampler.rs`, `chain/dynamics.rs`). P
   particles at one temperature are weighted by exp(-d H), resampled, and
   moved by the relaxed-assignment dynamics at the top activity, run until
   time t has been spent in perfect matchings. That move leaves the next
   temperature's distribution invariant for any weights and contracts it by
   exp(-t) (the paper's spectral gap of 1 carries over to the chain watched
   on perfect matchings).
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

## Backends

The population lives in an engine, an implementation of the `Engine` trait
of `permanent-core`. Each engine is a crate, built into the program by a
feature of the same name and chosen at run time with `--backend`:

| Backend | Crate | Population on |
|---|---|---|
| `rayon` (default feature) | `permanent-rayon` | the CPU thread pool |
| `cuda` | `permanent-cuda` | an NVIDIA GPU |

Both engines run the same chains, written once in
`crates/permanent-core/src/chain` and compiled for the host and for the
device.
Every probability decision is a comparison of random bits with an integer
threshold and FAR energies are fixed point; only the waiting times of the
move and of the sampler are floating point. Their type is the engine's type
parameter (`Rayon<T>`, `Cuda<T>`), chosen with `--precision f32|f64`: the
default is `f64` on rayon and `f32` on cuda. The estimator and the command
line are generic over the engine, so nothing is dispatched at run time
beyond the choice of backend.

The GPU draws from Philox4x32-10, keyed by a particle's seed; the rayon
engine draws from `SmallRng`, which makes it more than twice as fast, and
runs on Philox when asked (`Rayon<T, Stream>`), in which case the FAR chain
returns the same counts as on the GPU. A run with a given seed gives the
same result whatever the number of threads.

The kernels of the `cuda` backend are Rust, compiled with cuda-oxide
(https://github.com/NVIDIA/cuda-rust), which needs the toolchain pinned in
`rust-toolchain.toml`, LLVM 21+ and CUDA 13+. Build through `cargo oxide`
with the `cuda` profile (release without LTO, which the object carrying the
device code cannot take part in); a plain `cargo build --features cuda`
produces no device code.

    cargo oxide build -- --profile cuda --features cuda
    target/cuda/permanent data/dimer-32x32.mtx                  # cuda is the default once built in
    target/cuda/permanent data/dimer-32x32.mtx --backend rayon
    cargo oxide test -- --profile cuda -p permanent-cuda        # needs a GPU

Every population operation runs on the device. A particle's working state
is its matching, the inverse and a short list of collisions, staged in
shared memory while a warp's worth fits (n up to about 370) and kept in
global memory beyond. FAR chains have very uneven lengths, so a FAR launch
gives each chain a step budget and the few that overrun it are finished on
host threads.

## MPI

Built with `--features mpi`, every rank runs the same driver on its own
share of the work (and, with the `cuda` backend, its own device):

    cargo build --release --features mpi
    mpirun -n 4 target/release/permanent data/dimer-16x16.mtx

Ranks split the segments and islands of Phase 2, which share nothing, and
pool their learning statistics in Phase 1. Results do not depend on the
number of ranks beyond the split of the Phase 1 population.

## Nix

    nix build              # the program with the rayon backend, tests included
    nix build .#mpi        # with MPI
    nix develop            # cargo and rustc
    nix develop .#mpi      # plus an MPI and libclang for its bindings
    nix develop .#cuda     # the pinned toolchain and CUDA 13

The `cuda` backend is built inside the `cuda` shell with `cargo oxide`,
which the shell tells you how to install; the NVIDIA driver comes from the
host.

## Layout

    crates/permanent-core     the algorithm and the Engine trait
        src/problem.rs        Matrix Market input
        src/scaling.rs        Hungarian method, scaling, regularization
        src/tables.rs         per-temperature B matrix and alias tables
        src/sampler.rs        relaxed-assignment dynamics: activity schedule and caps
        src/chain/            the chains, for host and device: FAR, the move,
                              the annealing trial, a particle's memory and stream
        src/engine.rs         the Engine trait: population operations
        src/comm.rs           reductions across MPI ranks
        src/estimator.rs      schedule, planning, segments, islands
        src/exact.rs          exact subset DP for n <= 26
        src/stats.rs          ln n!, normal quantile, seeds of the random streams
    crates/permanent-rayon    engine on the CPU thread pool
    crates/permanent-cuda     engine on a GPU
        src/lib.rs            kernels (wrappers over the chains) and the host side
    crates/permanent-cli      the `permanent` program
        src/cli.rs            command line, report, logging
        src/tui.rs            dashboard
