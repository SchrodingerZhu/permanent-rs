//! Serializable problem specification, geometry, preset, and matching types.

use serde::{Deserialize, Serialize};

/// Specification for generating or loading a matrix problem.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ProblemSpec {
    /// Two-dimensional square/rectangular lattice dimer model.
    Dimer2d {
        rows: usize,
        cols: usize,
        #[serde(default)]
        periodic: bool,
        #[serde(default = "default_one")]
        weight_x: f64,
        #[serde(default = "default_one")]
        weight_y: f64,
        /// Multiplicative log-uniform bond disorder half-width: each bond weight
        /// is multiplied by `exp(Uniform[-disorder, +disorder])`.
        #[serde(default)]
        disorder: f64,
        /// Fraction of non-protected bonds dropped uniformly at random in `[0, 0.8]`.
        #[serde(default)]
        vacancy_rate: f64,
        /// Preserve a canonical dimer covering when `vacancy_rate > 0`.
        #[serde(default = "default_true")]
        ensure_matching: bool,
        #[serde(default = "default_seed")]
        seed: u64,
    },
    /// Three-dimensional cubic lattice dimer model.
    Dimer3d {
        lx: usize,
        ly: usize,
        lz: usize,
        #[serde(default)]
        periodic: bool,
        #[serde(default = "default_weights_3d")]
        weights: [f64; 3],
        /// Multiplicative log-uniform bond disorder half-width.
        #[serde(default)]
        disorder: f64,
        /// Fraction of non-protected bonds dropped uniformly at random in `[0, 0.8]`.
        #[serde(default)]
        vacancy_rate: f64,
        /// Preserve a canonical dimer covering when `vacancy_rate > 0`.
        #[serde(default = "default_true")]
        ensure_matching: bool,
        #[serde(default = "default_seed")]
        seed: u64,
    },
    /// Separate bipartite cycles of lengths `2 * m_c`.
    DisjointCycles {
        /// Number of rows (and columns) in each cycle; each must be at least 2.
        cycle_sizes: Vec<usize>,
    },
    /// Curated matrix bundled from the repository's `data/` directory.
    BundledPreset { name: String },
    /// Parametric random nonnegative matrix generator.
    RandomMatrix {
        n: usize,
        #[serde(default)]
        family: RandomMatrixFamily,
        #[serde(default = "default_density")]
        density: f64,
        /// Half-width of the log-weight spread (entries span `exp(±log_spread)`).
        #[serde(default = "default_log_spread")]
        log_spread: f64,
        /// Plant a random permutation so the permanent is guaranteed positive.
        #[serde(default = "default_true")]
        ensure_matching: bool,
        #[serde(default = "default_seed")]
        seed: u64,
    },
    /// User-supplied Matrix Market coordinate text (`.mtx`).
    MatrixMarket { text: String },
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RandomMatrixFamily {
    #[default]
    SparseLognormal,
    ZeroOneBipartite,
    NearPermutation,
    WeaklyCoupledBlocks,
    Uniform,
}

/// Nonzero entry `(row, col, value)` in 0-based coordinates.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct MatrixEntry {
    pub row: u32,
    pub col: u32,
    pub value: f64,
}

/// A bond between an even-sublattice site (`row`) and an odd-sublattice site (`col`).
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct LatticeBond {
    pub row: u32,
    pub col: u32,
    /// Lattice axis: `0` = x, `1` = y, `2` = z.
    pub axis: u8,
    pub weight: f64,
    /// True when the bond wraps across a periodic boundary.
    pub periodic_wrap: bool,
}

/// Visual layout metadata for 2D/3D model viewers.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ProblemGeometry {
    Lattice2d {
        rows: usize,
        cols: usize,
        periodic: bool,
        /// Even-sublattice coordinates `[x, y]`, indexed by matrix row `u`.
        row_sites: Vec<[usize; 2]>,
        /// Odd-sublattice coordinates `[x, y]`, indexed by matrix col `v`.
        col_sites: Vec<[usize; 2]>,
        bonds: Vec<LatticeBond>,
    },
    Lattice3d {
        dims: [usize; 3],
        periodic: bool,
        /// Even-sublattice coordinates `[x, y, z]`, indexed by matrix row `u`.
        row_sites: Vec<[usize; 3]>,
        /// Odd-sublattice coordinates `[x, y, z]`, indexed by matrix col `v`.
        col_sites: Vec<[usize; 3]>,
        bonds: Vec<LatticeBond>,
    },
    Cycles {
        cycle_sizes: Vec<usize>,
    },
    Matrix,
}

/// Complete generated problem descriptor returned to front ends for display.
#[derive(Debug, Clone, Serialize)]
pub struct GeneratedProblem {
    pub title: String,
    pub description: String,
    pub n: usize,
    pub nnz: usize,
    pub is_zero_one: bool,
    pub has_perfect_matching: bool,
    /// Maximum-product matching from Hungarian scaling (`None` when permanent is 0).
    pub heaviest_matching: Option<Vec<u32>>,
    /// Known closed-form or cheap exact `ln(permanent)`, when available immediately.
    pub exact_ln: Option<f64>,
    pub exact_scientific: Option<String>,
    pub exact_source: Option<String>,
    /// Total lattice sites (`2 * n` for dimer models), for entropy per site.
    pub lattice_sites: Option<usize>,
    /// Known theoretical lower bound on entropy per site (e.g. Schrijver 0.4401).
    pub entropy_lower_bound: Option<f64>,
    pub geometry: ProblemGeometry,
    pub triplets: Vec<MatrixEntry>,
}

/// Snapshot of a single matching (row `u` matched to column `matching[u]`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SampleSnapshot {
    pub beta: f64,
    pub matching: Vec<u32>,
    pub valid: bool,
}

/// One entry in the curated demo catalog (frontend-agnostic).
#[derive(Debug, Clone, Serialize)]
pub struct DemoPreset {
    pub id: String,
    pub title: String,
    pub subtitle: String,
    pub spec: ProblemSpec,
    /// Whether an exact subset-DP reference is cheap enough to request by default.
    pub compute_exact: bool,
}

fn default_one() -> f64 {
    1.0
}

fn default_true() -> bool {
    true
}

fn default_seed() -> u64 {
    1
}

fn default_weights_3d() -> [f64; 3] {
    [1.0, 1.0, 1.0]
}

fn default_density() -> f64 {
    0.45
}

fn default_log_spread() -> f64 {
    3.0
}
