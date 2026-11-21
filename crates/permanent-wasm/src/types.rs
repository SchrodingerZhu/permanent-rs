//! Serializable configuration, capability, and telemetry types shared across
//! the Rust/Wasm boundary.

use serde::{Deserialize, Serialize};

pub use permanent_demo::types::{
    GeneratedProblem, LatticeBond, MatrixEntry, ProblemGeometry, ProblemSpec, RandomMatrixFamily, SampleSnapshot,
};

/// Estimator and engine configuration exposed to the web UI. Runs are on WebGPU (`permanent-wgpu`).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct WebRunConfig {
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
    /// Working set limit in MiB (default: 32).
    pub wgpu_working_set_mb: Option<usize>,
    /// Per-dispatch event cap for the mutation/annealing loops (default: 65536).
    pub wgpu_budget: Option<u32>,
}

impl Default for WebRunConfig {
    fn default() -> Self {
        Self {
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

/// One entry of the curated demo catalog.
#[derive(Debug, Clone, Serialize)]
pub struct PresetInfo {
    pub id: String,
    pub title: String,
    pub subtitle: String,
    pub spec: ProblemSpec,
    pub recommended_config: WebRunConfig,
}

/// Browser WebGPU capability report.
#[derive(Debug, Clone, Serialize)]
pub struct Capabilities {
    pub webgpu_available: bool,
    pub adapter_name: Option<String>,
    pub max_storage_buffer_binding_size: Option<u64>,
    pub max_buffer_size: Option<u64>,
    pub browser_max_exact_n: usize,
}
