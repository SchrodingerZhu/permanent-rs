//! Browser `wasm-bindgen` exports and typed WebGPU capability probing for
//! Dedicated Web Workers and Window contexts.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Once};

use js_sys::{Function, JSON, Promise, Reflect};
use permanent_demo::{BuiltProblem, ProblemSpec, build_problem};
use serde::Serialize;
use serde::de::DeserializeOwned;
use wasm_bindgen::JsCast;
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;

use crate::runner::{self, BROWSER_MAX_EXACT_N};
use crate::types::{Capabilities, WebRunConfig};

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

export interface WebRunConfig {
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

export interface Phase1Summary {
  temperatures: number;
  seconds: number;
  learn_particles: number;
  learn_iterations: number;
  far_steps: number;
  mutate_events: number;
  mutations: number;
  mutate_time: number;
  learn_mutation_time: number;
  learn_capped_chains: number;
  mutate_over_budget: number;
  worst_first_correction_factor: number;
  check_min: number;
  check_max: number;
}

export interface PlanSummary {
  segments: [number, number][];
  ln_scale: number;
  particles: number;
  islands: number;
  relvar_sum: number;
  mixing_factor: number;
  island_relvar: number;
  stage_time: number;
  predicted_events: number;
  sigma_target: number;
  z: number;
}

export type VerdictInfo =
  | { status: "consistent"; max_gap_se: number }
  | { status: "inconsistent"; max_gap_se: number }
  | { status: "unchecked" };

export interface EstimateSummary {
  ln_permanent: number;
  std_error: number;
  scientific: string;
  ln_ci_low: number;
  ln_ci_high: number;
  ci_low_scientific: string;
  ci_high_scientific: string;
  ci_level_pct: number;
  rel_half_width_pct: number;
  met_target: boolean;
  rounds: number;
  islands: number;
  particles: number;
  island_relvar: number;
  planned_island_relvar: number;
  mean_gap: number;
  verdict: VerdictInfo;
  mutate_events: number;
  sampler_events: number;
  mutate_over_budget: number;
  sampler_over_budget: number;
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
  epsilon: number;
  delta: number;
  seed: number;
  plan_only: boolean;
  zero_permanent: boolean;
  exact_ln: number | null;
  exact_scientific: string | null;
  exact_source: string | null;
  exact_skipped_too_large: boolean;
  phase1: Phase1Summary | null;
  plan: PlanSummary | null;
  estimate: EstimateSummary | null;
  final_sample: SampleSnapshot | null;
  schedule_json: string | null;
}

export type WebEvent =
  | {
      type: "init";
      n: number;
      nnz: number;
      is_zero_one: boolean;
      engine: string;
      epsilon: number;
      delta: number;
      seed: number;
      exact_ln: number | null;
      exact_source: string | null;
      exact_skipped_too_large: boolean;
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
  | ({ type: "phase1" } & Phase1Summary)
  | ({ type: "plan" } & PlanSummary)
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
      mutate_events: number;
      sampler_events: number;
      seconds: number;
    }
  | { type: "done"; summary: RunSummary }
  | { type: "cancelled" }
  | { type: "failed"; message: string };

export interface PresetInfo {
  id: string;
  title: string;
  subtitle: string;
  spec: ProblemSpec;
  recommended_config: WebRunConfig;
}
"#;

unsafe extern "C" {
    fn __wasm_call_ctors();
}

/// Run the module's static constructors once, when it is instantiated.
///
/// CubeCL registers its tables through constructors. Unless something calls
/// `__wasm_call_ctors`, the linker makes every exported function run all of
/// them first, including the helpers the JS glue calls for each value that
/// crosses the boundary.
#[wasm_bindgen(start)]
fn start() {
    // SAFETY: the linker defines this function, and nothing else calls it.
    unsafe { __wasm_call_ctors() }
}

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
/// a Dedicated Web Worker (JS-facing export).
#[wasm_bindgen]
pub async fn probe_capabilities() -> Result<JsValue, JsValue> {
    init_panic_hook();
    let caps = probe_webgpu().await;
    to_js(&caps)
}

/// Probe WebGPU adapter availability and device limits, returning a typed Rust
/// [`Capabilities`] struct directly (for Rust/Dioxus front ends).
pub async fn probe_webgpu() -> Capabilities {
    let mut caps = Capabilities {
        webgpu_available: false,
        adapter_name: None,
        max_storage_buffer_binding_size: None,
        max_buffer_size: None,
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

/// Return the curated demo catalog.
#[wasm_bindgen]
pub fn list_presets() -> Result<JsValue, JsValue> {
    init_panic_hook();
    to_js(&crate::list_presets())
}

/// A problem built once and kept in the module, so that a page can read it
/// and draw coverings from it without building it again.
#[wasm_bindgen]
pub struct Problem(BuiltProblem);

#[wasm_bindgen]
impl Problem {
    /// Build the problem (matrix, exact reference when cheap, Hungarian
    /// matching, and 2D/3D visual geometry) that `spec` describes.
    #[wasm_bindgen(constructor)]
    pub fn new(spec: JsValue) -> Result<Problem, JsValue> {
        init_panic_hook();
        let spec: ProblemSpec = from_js(spec)?;
        build_problem(&spec).map(Problem).map_err(js_err)
    }

    /// The `GeneratedProblem` to display.
    pub fn info(&self) -> Result<JsValue, JsValue> {
        to_js(&self.0.info)
    }

    /// Quickly draw a valid perfect matching / dimer covering on the support
    /// using alternating-cycle (worm) moves for interactive visualization.
    ///
    /// `seed` is passed as an ordinary JS number (nonnegative integer) to avoid
    /// `BigInt` conversion errors at the JS boundary.
    pub fn sample_covering(&self, seed: f64, steps: u32) -> Result<JsValue, JsValue> {
        if !seed.is_finite() || seed < 0.0 {
            return Err(JsValue::from_str("seed must be a finite nonnegative integer"));
        }
        let snapshot = permanent_demo::sample_covering(&self.0, seed as u64, steps).map_err(js_err)?;
        to_js(&snapshot)
    }
}

/// `new Problem(spec).info()`, for a page that needs the problem once.
#[wasm_bindgen]
pub fn generate_problem(spec: JsValue) -> Result<JsValue, JsValue> {
    Problem::new(spec)?.info()
}

/// `new Problem(spec).sample_covering(seed, steps)`, for a page that draws once.
#[wasm_bindgen]
pub fn sample_covering(spec: JsValue, seed: f64, steps: u32) -> Result<JsValue, JsValue> {
    Problem::new(spec)?.sample_covering(seed, steps)
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

    /// Execute a full estimation run on WebGPU, invoking `on_event(event)` with
    /// each live telemetry message. Resolves to `RunSummary` or `null` if cancelled.
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

#[cfg(test)]
mod tests {
    //! Run by the wasm test harness in a runtime with WebGPU.

    use std::cell::RefCell;
    use std::rc::Rc;

    use serde_json::Value;
    use wasm_bindgen_test::wasm_bindgen_test;

    use super::*;

    fn js(json: &str) -> JsValue {
        JSON::parse(json).unwrap()
    }

    fn json(value: &JsValue) -> Value {
        serde_json::from_str(&JSON::stringify(value).unwrap().as_string().unwrap()).unwrap()
    }

    /// A handle for a run. The test harness does not call the start function,
    /// so the constructors are run here.
    fn handle() -> RunHandle {
        static CONSTRUCTORS: Once = Once::new();
        CONSTRUCTORS.call_once(start);
        RunHandle::new()
    }

    /// Run on `handle` and return what `run` resolved to with the type of every event.
    async fn run(handle: &RunHandle, spec: &str, config: &str) -> (Value, Vec<String>) {
        let events = Rc::new(RefCell::new(Vec::new()));
        let seen = Rc::clone(&events);
        let on_event = Closure::<dyn FnMut(JsValue)>::new(move |event: JsValue| {
            seen.borrow_mut().push(json(&event)["type"].as_str().unwrap().to_owned());
        });
        let summary = handle.run(js(spec), js(config), on_event.as_ref().clone().unchecked_into()).await;
        (json(&summary.unwrap()), events.take())
    }

    #[wasm_bindgen_test]
    fn a_kept_problem_gives_its_description_and_coverings() {
        let problem = Problem::new(js(r#"{"kind": "dimer2d", "rows": 4, "cols": 6}"#)).unwrap();
        assert_eq!(json(&problem.info().unwrap())["n"], 12);
        let covering = json(&problem.sample_covering(1.0, 64).unwrap());
        assert_eq!(covering["matching"].as_array().unwrap().len(), 12);
    }

    #[wasm_bindgen_test]
    async fn probe_finds_webgpu() {
        let capabilities = json(&probe_capabilities().await.unwrap());
        assert_eq!(capabilities["webgpu_available"], true);
    }

    #[wasm_bindgen_test]
    async fn run_streams_events_and_agrees_with_the_exact_value() {
        // an open 2 x 2 x 2 box has 9 dimer coverings
        let (summary, events) = run(
            &handle(),
            r#"{"kind": "dimer3d", "lx": 2, "ly": 2, "lz": 2}"#,
            r#"{"epsilon": 0.4, "seed": 3, "learn_particles": 128, "particles": 16, "islands": 16, "segments": 2,
                "save_schedule": true}"#,
        )
        .await;
        for kind in ["init", "level", "sample", "phase1", "plan", "sweep", "round"] {
            assert!(events.iter().any(|event| event == kind), "no {kind} event");
        }
        assert_eq!(events.last().unwrap(), "done");

        let exact = 9.0_f64.ln();
        assert!((summary["exact_ln"].as_f64().unwrap() - exact).abs() < 1e-12);
        assert!(summary["phase1"]["temperatures"].as_u64().unwrap() >= 2);
        assert!(summary["plan"]["islands"].as_u64().unwrap() >= 1);
        let estimate = &summary["estimate"];
        let distance = (estimate["ln_permanent"].as_f64().unwrap() - exact) / estimate["std_error"].as_f64().unwrap();
        assert!(distance.abs() < 4.0, "{distance} standard errors from the exact value");
        assert!(estimate["entropy_per_site"].is_f64());
        assert!(summary["schedule_json"].as_str().unwrap().contains("\"levels\""));
    }

    #[wasm_bindgen_test]
    async fn plan_only_stops_before_phase_2_and_a_zero_permanent_needs_no_run() {
        let config = r#"{"plan_only": true, "learn_particles": 128}"#;
        let handle = handle();
        let (summary, events) = run(&handle, r#"{"kind": "bundled_preset", "name": "weighted-8"}"#, config).await;
        assert_eq!(summary["plan_only"], true);
        assert!(summary["phase1"].is_object());
        assert!(summary["plan"].is_object());
        assert!(summary["estimate"].is_null());
        assert!(events.iter().any(|event| event == "phase1"));
        assert!(events.iter().any(|event| event == "plan"));
        assert!(!events.iter().any(|event| event == "sweep"));

        // column 3 is empty
        let zero = r#"{"kind": "matrix_market",
            "text": "%%MatrixMarket matrix coordinate real general\n3 3 4\n1 1 1\n2 1 2\n3 1 1\n3 2 1\n"}"#;
        let (summary, _) = run(&handle, zero, config).await;
        assert_eq!(summary["zero_permanent"], true);
        assert!(summary["phase1"].is_null());
        assert!(summary["plan"].is_null());
    }

    /// `cancel()` called from a timer, as a page would on a click, ends the run.
    #[wasm_bindgen_test]
    async fn cancel_from_a_timer_stops_the_run() {
        let handle = Rc::new(handle());
        let stop = Rc::clone(&handle);
        let global = js_sys::global();
        let set_timeout: Function = Reflect::get(&global, &"setTimeout".into()).unwrap().unchecked_into();
        set_timeout.call2(&global, &Closure::once_into_js(move || stop.cancel()), &0.into()).unwrap();

        let (summary, events) = run(&handle, r#"{"kind": "bundled_preset", "name": "dimer-8x8"}"#, "{}").await;
        assert!(summary.is_null());
        assert_eq!(events.last().unwrap(), "cancelled");
    }
}
