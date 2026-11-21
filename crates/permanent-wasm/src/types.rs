//! Serializable request, geometry, configuration, and telemetry types shared
//! across the Rust/Wasm boundary.

use serde::{Deserialize, Serialize};

/// Specification for generating or loading a matrix problem.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ProblemSpec {
    /// Two-dimensional square/rectangular lattice dimer model (PDF Examples 1 & 4).
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
    /// Three-dimensional cubic lattice dimer model (PDF Example 5).
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
    /// Separate bipartite cycles of lengths `2 * m_c` (PDF Example 2).
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

/// Visual layout metadata for the 3D/2D model viewers.
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

/// Complete generated problem descriptor returned to the UI for immediate display.
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

/// Which engine backend to run in the browser.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BackendKind {
    /// Prefer WebGPU (`permanent-wgpu`), falling back to CPU (`permanent-rayon`) if unavailable.
    #[default]
    Auto,
    /// Require WebGPU (`permanent-wgpu`).
    Wgpu,
    /// Run on the CPU in WebAssembly (`permanent-rayon`, single-threaded).
    Cpu,
}

/// Full estimator and backend configuration exposed to the web UI.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct WebRunConfig {
    pub backend: BackendKind,
    pub epsilon: f64,
    pub delta: f64,
    pub seed: u64,
    pub plan_only: bool,
    pub compute_exact: bool,
    pub capture_samples: bool,
    pub save_schedule: bool,
    pub rigorous_schedule: bool,
    pub rigorous_sampler: bool,
    pub density: f64,
    pub mutation_time: f64,
    pub learn_mutation_time: f64,
    pub learn_particles: Option<usize>,
    pub particles: Option<usize>,
    pub islands: Option<usize>,
    pub segments: Option<usize>,
    /// Working set limit in MiB for the `wgpu` backend (default: 32).
    pub wgpu_working_set_mb: Option<usize>,
    /// Per-dispatch event cap for `wgpu` mutation/annealing loops (default: 65536).
    pub wgpu_budget: Option<u32>,
}

impl Default for WebRunConfig {
    fn default() -> Self {
        Self {
            backend: BackendKind::Auto,
            epsilon: 0.1,
            delta: 0.05,
            seed: 1,
            plan_only: false,
            compute_exact: false,
            capture_samples: true,
            save_schedule: false,
            rigorous_schedule: false,
            rigorous_sampler: false,
            density: 32.0,
            mutation_time: 1.5,
            learn_mutation_time: 0.0,
            learn_particles: None,
            particles: None,
            islands: None,
            segments: None,
            wgpu_working_set_mb: None,
            wgpu_budget: None,
        }
    }
}

/// Snapshot of a single matching (row `u` matched to column `matching[u]`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SampleSnapshot {
    pub beta: f64,
    pub matching: Vec<u32>,
    pub valid: bool,
}

/// Result of the segment-boundary consistency check.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum VerdictInfo {
    Consistent { max_gap_se: f64 },
    Inconsistent { max_gap_se: f64 },
    Unchecked,
}

/// Structured Phase 2 estimation result.
#[derive(Debug, Clone, Serialize)]
pub struct EstimateSummary {
    pub ln_permanent: f64,
    pub std_error: f64,
    pub scientific: String,
    pub ci_low_scientific: String,
    pub ci_high_scientific: String,
    pub ci_level_pct: f64,
    pub rel_half_width_pct: f64,
    pub met_target: bool,
    pub rounds: usize,
    pub islands: usize,
    pub particles: usize,
    pub island_relvar: f64,
    pub mean_gap: f64,
    pub verdict: VerdictInfo,
    pub mutate_events: u64,
    pub sampler_events: u64,
    pub over_budget: u64,
    pub seconds: f64,
    pub rel_error_pct: Option<f64>,
    pub z_score: Option<f64>,
    pub entropy_per_site: Option<f64>,
    pub entropy_std_error: Option<f64>,
    pub entropy_lower_bound: Option<f64>,
}

/// Final summary emitted on `WebEvent::Done` and returned from `run`.
#[derive(Debug, Clone, Serialize)]
pub struct RunSummary {
    pub n: usize,
    pub nnz: usize,
    pub is_zero_one: bool,
    pub engine: String,
    pub seed: u64,
    pub plan_only: bool,
    pub zero_permanent: bool,
    pub temperatures: usize,
    pub phase1_seconds: f64,
    pub learn_particles: usize,
    pub learn_iterations: u64,
    pub far_steps: u64,
    pub worst_first_correction_factor: f64,
    pub check_min: f64,
    pub check_max: f64,
    pub exact_ln: Option<f64>,
    pub exact_scientific: Option<String>,
    pub exact_source: Option<String>,
    pub estimate: Option<EstimateSummary>,
    pub final_sample: Option<SampleSnapshot>,
    pub schedule_json: Option<String>,
    pub lines: Vec<String>,
}

/// Live telemetry event stream mirroring `permanent-cli`'s TUI dashboard.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum WebEvent {
    Init {
        n: usize,
        nnz: usize,
        is_zero_one: bool,
        engine: String,
        exact_ln: Option<f64>,
        exact_source: Option<String>,
    },
    Level {
        index: usize,
        beta: f64,
        step: f64,
        relvar: f64,
        log_w: Vec<f64>,
    },
    Sample {
        beta: f64,
        matching: Vec<u32>,
        valid: bool,
    },
    Plan {
        segments: Vec<[usize; 2]>,
        ln_scale: f64,
        particles: usize,
        islands: usize,
        relvar_sum: f64,
        mixing_factor: f64,
        island_relvar: f64,
        stage_time: f64,
        sigma_target: f64,
        z: f64,
    },
    Sweep {
        round: usize,
        done: u64,
        total: u64,
        segment: usize,
        level: usize,
        islands: usize,
        ln_mean: f64,
        ln_mean_square: f64,
    },
    Round {
        round: usize,
        islands: usize,
        ln_permanent: f64,
        std_error: f64,
    },
    Line {
        text: String,
    },
    Done {
        summary: Box<RunSummary>,
    },
    Cancelled,
    Failed {
        message: String,
    },
}

/// Curated demo entry matching Part 4 of `permanent-rs-explained`.
#[derive(Debug, Clone, Serialize)]
pub struct PresetInfo {
    pub id: String,
    pub title: String,
    pub subtitle: String,
    pub pdf_reference: String,
    pub spec: ProblemSpec,
    pub recommended_config: WebRunConfig,
}

/// Browser WebGPU & backend capability report.
#[derive(Debug, Clone, Serialize)]
pub struct Capabilities {
    pub webgpu_available: bool,
    pub adapter_name: Option<String>,
    pub max_storage_buffer_binding_size: Option<u64>,
    pub max_buffer_size: Option<u64>,
    pub backends: Vec<BackendKind>,
    pub browser_max_exact_n: usize,
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
