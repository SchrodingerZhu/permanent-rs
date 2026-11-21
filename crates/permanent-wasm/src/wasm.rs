//! Browser `wasm-bindgen` exports for Dedicated Web Workers and Window contexts.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Once};

use js_sys::{Function, JSON, Promise, Reflect};
use serde::Serialize;
use serde::de::DeserializeOwned;
use wasm_bindgen::JsCast;
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;

use crate::generators::{self, BROWSER_MAX_EXACT_N};
use crate::runner;
use crate::sample;
use crate::types::{BackendKind, Capabilities, ProblemSpec, WebRunConfig};

#[wasm_bindgen(typescript_custom_section)]
const TS_TYPES: &str = r#"
export type RandomMatrixFamily =
  | "sparse_lognormal"
  | "zero_one_bipartite"
  | "near_permutation"
  | "weakly_coupled_blocks"
  | "uniform";

export type ProblemSpec =
  | {
      kind: "dimer2d";
      rows: number;
      cols: number;
      periodic?: boolean;
      weight_x?: number;
      weight_y?: number;
      disorder?: number;
      vacancy_rate?: number;
      ensure_matching?: boolean;
      seed?: number;
    }
  | {
      kind: "dimer3d";
      lx: number;
      ly: number;
      lz: number;
      periodic?: boolean;
      weights?: [number, number, number];
      disorder?: number;
      vacancy_rate?: number;
      ensure_matching?: boolean;
      seed?: number;
    }
  | { kind: "disjoint_cycles"; cycle_sizes: number[] }
  | { kind: "bundled_preset"; name: string }
  | {
      kind: "random_matrix";
      n: number;
      family?: RandomMatrixFamily;
      density?: number;
      log_spread?: number;
      ensure_matching?: boolean;
      seed?: number;
    }
  | { kind: "matrix_market"; text: string };

export type BackendKind = "auto" | "wgpu" | "cpu";

export interface WebRunConfig {
  backend?: BackendKind;
  epsilon?: number;
  delta?: number;
  seed?: number;
  plan_only?: boolean;
  compute_exact?: boolean;
  capture_samples?: boolean;
  save_schedule?: boolean;
  rigorous_schedule?: boolean;
  rigorous_sampler?: boolean;
  density?: number;
  mutation_time?: number;
  learn_mutation_time?: number;
  learn_particles?: number | null;
  particles?: number | null;
  islands?: number | null;
  segments?: number | null;
  wgpu_working_set_mb?: number | null;
  wgpu_budget?: number | null;
}

export interface Capabilities {
  webgpu_available: boolean;
  adapter_name: string | null;
  max_storage_buffer_binding_size: number | null;
  max_buffer_size: number | null;
  backends: BackendKind[];
  browser_max_exact_n: number;
}

export interface MatrixEntry {
  row: number;
  col: number;
  value: number;
}

export interface LatticeBond {
  row: number;
  col: number;
  axis: number;
  weight: number;
  periodic_wrap: boolean;
}

export type ProblemGeometry =
  | {
      kind: "lattice2d";
      rows: number;
      cols: number;
      periodic: boolean;
      row_sites: [number, number][];
      col_sites: [number, number][];
      bonds: LatticeBond[];
    }
  | {
      kind: "lattice3d";
      dims: [number, number, number];
      periodic: boolean;
      row_sites: [number, number, number][];
      col_sites: [number, number, number][];
      bonds: LatticeBond[];
    }
  | { kind: "cycles"; cycle_sizes: number[] }
  | { kind: "matrix" };

export interface GeneratedProblem {
  title: string;
  description: string;
  n: number;
  nnz: number;
  is_zero_one: boolean;
  has_perfect_matching: boolean;
  heaviest_matching: number[] | null;
  exact_ln: number | null;
  exact_scientific: string | null;
  exact_source: string | null;
  lattice_sites: number | null;
  entropy_lower_bound: number | null;
  geometry: ProblemGeometry;
  triplets: MatrixEntry[];
}

export interface SampleSnapshot {
  beta: number;
  matching: number[];
  valid: boolean;
}

export type VerdictInfo =
  | { status: "consistent"; max_gap_se: number }
  | { status: "inconsistent"; max_gap_se: number }
  | { status: "unchecked" };

export interface EstimateSummary {
  ln_permanent: number;
  std_error: number;
  scientific: string;
  ci_low_scientific: string;
  ci_high_scientific: string;
  ci_level_pct: number;
  rel_half_width_pct: number;
  met_target: boolean;
  rounds: number;
  islands: number;
  particles: number;
  island_relvar: number;
  mean_gap: number;
  verdict: VerdictInfo;
  mutate_events: number;
  sampler_events: number;
  over_budget: number;
  seconds: number;
  rel_error_pct: number | null;
  z_score: number | null;
  entropy_per_site: number | null;
  entropy_std_error: number | null;
  entropy_lower_bound: number | null;
}

export interface RunSummary {
  n: number;
  nnz: number;
  is_zero_one: boolean;
  engine: string;
  seed: number;
  plan_only: boolean;
  zero_permanent: boolean;
  temperatures: number;
  phase1_seconds: number;
  learn_particles: number;
  learn_iterations: number;
  far_steps: number;
  worst_first_correction_factor: number;
  check_min: number;
  check_max: number;
  exact_ln: number | null;
  exact_scientific: string | null;
  exact_source: string | null;
  estimate: EstimateSummary | null;
  final_sample: SampleSnapshot | null;
  schedule_json: string | null;
  lines: string[];
}

export type WebEvent =
  | {
      type: "init";
      n: number;
      nnz: number;
      is_zero_one: boolean;
      engine: string;
      exact_ln: number | null;
      exact_source: string | null;
    }
  | {
      type: "level";
      index: number;
      beta: number;
      step: number;
      relvar: number;
      log_w: number[];
    }
  | {
      type: "sample";
      beta: number;
      matching: number[];
      valid: boolean;
    }
  | {
      type: "plan";
      segments: [number, number][];
      ln_scale: number;
      particles: number;
      islands: number;
      relvar_sum: number;
      mixing_factor: number;
      island_relvar: number;
      stage_time: number;
      sigma_target: number;
      z: number;
    }
  | {
      type: "sweep";
      round: number;
      done: number;
      total: number;
      segment: number;
      level: number;
      islands: number;
      ln_mean: number;
      ln_mean_square: number;
    }
  | {
      type: "round";
      round: number;
      islands: number;
      ln_permanent: number;
      std_error: number;
    }
  | { type: "line"; text: string }
  | { type: "done"; summary: RunSummary }
  | { type: "cancelled" }
  | { type: "failed"; message: string };

export interface PresetInfo {
  id: string;
  title: string;
  subtitle: string;
  pdf_reference: string;
  spec: ProblemSpec;
  recommended_config: WebRunConfig;
}
"#;

static PANIC_HOOK: Once = Once::new();

/// Install a panic hook that logs Rust panic messages to `console.error`.
#[wasm_bindgen]
pub fn init_panic_hook() {
    PANIC_HOOK.call_once(|| {
        std::panic::set_hook(Box::new(|info| {
            web_sys::console::error_1(&JsValue::from_str(&info.to_string()));
        }));
    });
}

fn to_js<T: Serialize>(value: &T) -> Result<JsValue, JsValue> {
    let text = serde_json::to_string(value).map_err(|e| JsValue::from_str(&format!("serialize error: {e}")))?;
    JSON::parse(&text)
}

fn from_js<T: DeserializeOwned>(value: JsValue) -> Result<T, JsValue> {
    let text =
        JSON::stringify(&value)?.as_string().ok_or_else(|| JsValue::from_str("expected a JSON-serializable value"))?;
    serde_json::from_str(&text).map_err(|e| JsValue::from_str(&format!("invalid input: {e}")))
}

fn js_err(error: anyhow::Error) -> JsValue {
    JsValue::from_str(&format!("{error:#}"))
}

/// Probe WebGPU adapter availability and device limits from either a Window or
/// a Dedicated Web Worker.
#[wasm_bindgen]
pub async fn probe_capabilities() -> Result<JsValue, JsValue> {
    init_panic_hook();
    let caps = probe_webgpu().await;
    to_js(&caps)
}

async fn probe_webgpu() -> Capabilities {
    let mut caps = Capabilities {
        webgpu_available: false,
        adapter_name: None,
        max_storage_buffer_binding_size: None,
        max_buffer_size: None,
        backends: vec![BackendKind::Cpu],
        browser_max_exact_n: BROWSER_MAX_EXACT_N,
    };

    let global = js_sys::global();
    let Ok(navigator) = Reflect::get(&global, &JsValue::from_str("navigator")) else {
        return caps;
    };
    let Ok(gpu) = Reflect::get(&navigator, &JsValue::from_str("gpu")) else {
        return caps;
    };
    if gpu.is_undefined() || gpu.is_null() {
        return caps;
    }
    let Ok(req_fn) = Reflect::get(&gpu, &JsValue::from_str("requestAdapter")) else {
        return caps;
    };
    let Ok(req_fn) = req_fn.dyn_into::<Function>() else {
        return caps;
    };
    let Ok(promise) = req_fn.call0(&gpu) else {
        return caps;
    };
    let Ok(promise) = promise.dyn_into::<Promise>() else {
        return caps;
    };
    let Ok(adapter) = JsFuture::from(promise).await else {
        return caps;
    };
    if adapter.is_undefined() || adapter.is_null() {
        return caps;
    }

    caps.webgpu_available = true;
    caps.backends = vec![BackendKind::Wgpu, BackendKind::Cpu];

    if let Ok(info) = Reflect::get(&adapter, &JsValue::from_str("info"))
        && !info.is_undefined()
        && !info.is_null()
    {
        let parts: Vec<String> = ["vendor", "architecture", "device", "description"]
            .iter()
            .filter_map(|field| {
                Reflect::get(&info, &JsValue::from_str(field))
                    .ok()
                    .and_then(|v| v.as_string())
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
            })
            .collect();
        if !parts.is_empty() {
            caps.adapter_name = Some(parts.join(" "));
        }
    }

    if let Ok(limits) = Reflect::get(&adapter, &JsValue::from_str("limits"))
        && !limits.is_undefined()
        && !limits.is_null()
    {
        caps.max_storage_buffer_binding_size = Reflect::get(&limits, &JsValue::from_str("maxStorageBufferBindingSize"))
            .ok()
            .and_then(|v| v.as_f64())
            .map(|n| n as u64);
        caps.max_buffer_size =
            Reflect::get(&limits, &JsValue::from_str("maxBufferSize")).ok().and_then(|v| v.as_f64()).map(|n| n as u64);
    }

    caps
}

/// Return the curated demo catalog matching Part 4 of `permanent-rs-explained`.
#[wasm_bindgen]
pub fn list_presets() -> Result<JsValue, JsValue> {
    init_panic_hook();
    to_js(&generators::list_presets())
}

/// Generate a problem (matrix, exact reference when cheap, Hungarian matching,
/// and 2D/3D visual geometry) from a `ProblemSpec`.
#[wasm_bindgen]
pub fn generate_problem(spec: JsValue) -> Result<JsValue, JsValue> {
    init_panic_hook();
    let spec: ProblemSpec = from_js(spec)?;
    let built = generators::build_problem(&spec).map_err(js_err)?;
    to_js(&built.info)
}

/// Quickly draw a valid perfect matching / dimer covering on the support of
/// `spec` using alternating-cycle (worm) moves for interactive visualization.
///
/// `seed` is passed as an ordinary JS number (nonnegative integer) to avoid
/// `BigInt` conversion errors at the JS boundary.
#[wasm_bindgen]
pub fn sample_covering(spec: JsValue, seed: f64, steps: u32) -> Result<JsValue, JsValue> {
    init_panic_hook();
    if !seed.is_finite() || seed < 0.0 {
        return Err(JsValue::from_str("seed must be a finite nonnegative integer"));
    }
    let spec: ProblemSpec = from_js(spec)?;
    let built = generators::build_problem(&spec).map_err(js_err)?;
    let snapshot = sample::sample_covering(&built.problem, seed as u64, steps).map_err(js_err)?;
    to_js(&snapshot)
}

struct RunningGuard(Arc<AtomicBool>);

impl Drop for RunningGuard {
    fn drop(&mut self) {
        self.0.store(false, Ordering::Release);
    }
}

/// Handle controlling one asynchronous estimation run with cooperative
/// cancellation.
#[wasm_bindgen]
pub struct RunHandle {
    cancel: Arc<AtomicBool>,
    running: Arc<AtomicBool>,
}

impl Default for RunHandle {
    fn default() -> Self {
        Self::new()
    }
}

#[wasm_bindgen]
impl RunHandle {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        init_panic_hook();
        Self { cancel: Arc::new(AtomicBool::new(false)), running: Arc::new(AtomicBool::new(false)) }
    }

    /// Signal the active run to stop cooperatively at the next checkpoint.
    ///
    /// Takes `&self` so JavaScript can invoke it while `run()` is awaiting.
    pub fn cancel(&self) {
        self.cancel.store(true, Ordering::Relaxed);
    }

    /// Execute a full estimation run, invoking `on_event(event)` with each
    /// live telemetry message. Resolves to `RunSummary` or `null` if cancelled.
    ///
    /// Rejects overlapping calls on the same `RunHandle` until the active run
    /// finishes or is cancelled.
    pub async fn run(&self, spec: JsValue, config: JsValue, on_event: Function) -> Result<JsValue, JsValue> {
        if self.running.compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire).is_err() {
            return Err(JsValue::from_str("a run is already in progress on this RunHandle"));
        }
        let _guard = RunningGuard(Arc::clone(&self.running));
        self.cancel.store(false, Ordering::Relaxed);
        let spec: ProblemSpec = from_js(spec)?;
        let config: WebRunConfig =
            if config.is_undefined() || config.is_null() { WebRunConfig::default() } else { from_js(config)? };
        let summary = runner::run_problem(&spec, &config, &self.cancel, |event| {
            if let Ok(js_event) = to_js(&event) {
                let _ = on_event.call1(&JsValue::NULL, &js_event);
            }
        })
        .await
        .map_err(js_err)?;
        to_js(&summary)
    }
}
