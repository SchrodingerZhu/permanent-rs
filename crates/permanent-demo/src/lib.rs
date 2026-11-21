//! Pure-Rust demo problem generators, exact references, curated presets, and
//! interactive covering sampler for `permanent-rs`.
//!
//! Each generator family lives in its own module ([`dimer`], [`cycles`],
//! [`random`], [`presets`]), while [`sample`] provides a fast alternating-cycle
//! (worm) explorer over any built problem's support.

pub mod cycles;
pub mod dimer;
pub mod presets;
pub mod random;
pub mod sample;
pub mod types;

use permanent_core::problem::Problem;
use permanent_core::scaling::{Prepared, Scaled, prepare};
use permanent_core::stats::scientific;

pub use sample::sample_covering;
pub use types::{
    DemoPreset, GeneratedProblem, LatticeBond, MatrixEntry, ProblemGeometry, ProblemSpec, RandomMatrixFamily,
    SampleSnapshot,
};

/// Upper bound on generated matrix dimension `n` (matches the GPU engine limit
/// and avoids exhausting 32-bit WebAssembly memory).
pub const MAX_N: usize = 1024;

/// Maximum dimension `n` for computing an instant subset-DP exact reference
/// automatically during problem generation (`n <= 16` takes < 2 ms).
pub(crate) const INSTANT_EXACT_N: usize = 16;

/// Built matrix together with its scaling, metadata and visual geometry.
pub struct BuiltProblem {
    pub problem: Problem,
    /// `None` when the support has no perfect matching (the permanent is 0).
    pub scaled: Option<Scaled>,
    pub info: GeneratedProblem,
}

/// Build a [`BuiltProblem`] (matrix + scaling + metadata + visual geometry)
/// from a [`ProblemSpec`].
pub fn build_problem(spec: &ProblemSpec) -> anyhow::Result<BuiltProblem> {
    match spec {
        ProblemSpec::Dimer2d {
            rows,
            cols,
            periodic,
            weight_x,
            weight_y,
            disorder,
            vacancy_rate,
            ensure_matching,
            seed,
        } => dimer::build_2d(
            *rows,
            *cols,
            [*weight_x, *weight_y],
            dimer::LatticeOptions {
                periodic: *periodic,
                disorder: *disorder,
                vacancy_rate: *vacancy_rate,
                ensure_matching: *ensure_matching,
                seed: *seed,
            },
        ),
        ProblemSpec::Dimer3d { lx, ly, lz, periodic, weights, disorder, vacancy_rate, ensure_matching, seed } => {
            dimer::build_3d(
                [*lx, *ly, *lz],
                *weights,
                dimer::LatticeOptions {
                    periodic: *periodic,
                    disorder: *disorder,
                    vacancy_rate: *vacancy_rate,
                    ensure_matching: *ensure_matching,
                    seed: *seed,
                },
            )
        }
        ProblemSpec::DisjointCycles { cycle_sizes } => cycles::build(cycle_sizes),
        ProblemSpec::BundledPreset { name } => presets::build_bundled(name),
        ProblemSpec::RandomMatrix { n, family, density, log_spread, ensure_matching, seed } => {
            random::build(*n, *family, *density, *log_spread, *ensure_matching, *seed)
        }
        ProblemSpec::MatrixMarket { text } => presets::from_matrix_market(text),
    }
}

pub(crate) struct ProblemMeta {
    pub title: String,
    pub description: String,
    pub exact_ln: Option<f64>,
    pub exact_source: Option<String>,
    pub lattice_sites: Option<usize>,
    pub entropy_lower_bound: Option<f64>,
    pub geometry: ProblemGeometry,
}

pub(crate) fn finalize(problem: Problem, meta: ProblemMeta) -> anyhow::Result<BuiltProblem> {
    let n = problem.n;
    let is_zero_one = problem.is_zero_one();
    let mut triplets = Vec::new();
    for u in 0..n {
        for v in 0..n {
            let value = problem.a[u * n + v];
            if value > 0.0 {
                triplets.push(MatrixEntry { row: u as u32, col: v as u32, value });
            }
        }
    }
    let nnz = triplets.len();
    let scaled = match prepare(&problem) {
        Prepared::Zero => None,
        Prepared::Scaled(scaled) => Some(scaled),
    };
    let has_perfect_matching = scaled.is_some();
    let heaviest_matching = scaled.as_ref().map(|scaled| scaled.matching.iter().map(|&v| v as u32).collect());
    let (exact_ln, exact_scientific, exact_source) = if !has_perfect_matching {
        (Some(f64::NEG_INFINITY), Some("0".into()), Some("no perfect matching".into()))
    } else {
        let sci = meta.exact_ln.map(scientific);
        (meta.exact_ln, sci, meta.exact_source)
    };
    // JSON cannot serialize `-Infinity`; represent zero permanent via `has_perfect_matching: false`.
    let json_exact_ln = exact_ln.filter(|x| x.is_finite());

    let info = GeneratedProblem {
        title: meta.title,
        description: meta.description,
        n,
        nnz,
        is_zero_one,
        has_perfect_matching,
        heaviest_matching,
        exact_ln: json_exact_ln,
        exact_scientific,
        exact_source,
        lattice_sites: meta.lattice_sites,
        entropy_lower_bound: meta.entropy_lower_bound,
        geometry: meta.geometry,
        triplets,
    };
    Ok(BuiltProblem { problem, scaled, info })
}
