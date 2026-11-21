//! Interactive Dioxus 0.7 web application for `permanent-rs`.

#[cfg(all(target_family = "wasm", target_os = "unknown", not(target_feature = "atomics")))]
#[global_allocator]
static ALLOCATOR: talc::wasm::WasmDynamicTalc = talc::wasm::new_wasm_dynamic_allocator();

mod components;
mod interop;
mod state;
mod worker;

use std::cell::RefCell;
use std::rc::Rc;

use dioxus::prelude::*;
use permanent_demo::{BuiltProblem, ProblemSpec, build_problem, sample_covering};
use permanent_wasm::list_presets;
use permanent_wasm::types::{GeneratedProblem, PresetInfo, SampleSnapshot, WebEvent, WebRunConfig};

use crate::components::controls_panel::{ControlsPanel, SAMPLE_MTX_WEIGHTED};
use crate::components::footer::Footer;
use crate::components::hero::Hero;
use crate::components::navbar::Navbar;
use crate::components::preset_strip::PresetStrip;
use crate::components::telemetry_panel::TelemetryPanel;
use crate::components::viewer_stage::ViewerStage;
use crate::state::{ActiveTab, TelemetryState, WorkerStatus};
use crate::worker::GpuWorkerClient;

/// Dimension threshold below which interactive preview builds and covering draws
/// can execute synchronously in < 2 ms before the background WebGPU worker has
/// finished instantiating.
const FAST_LOCAL_MAX_N: usize = 64;

/// Draw an illustrative initial covering for lattice and cycle demos so the
/// opening view displays multi-axis blue/gold structure rather than a trivial
/// single-axis canonical matching. Weighted general matrices keep their unique
/// heaviest Hungarian matching.
fn initial_sample_for(spec: &ProblemSpec, built: &BuiltProblem) -> Option<SampleSnapshot> {
    let want_mixed =
        matches!(spec, ProblemSpec::Dimer3d { .. } | ProblemSpec::Dimer2d { .. } | ProblemSpec::DisjointCycles { .. });
    if !want_mixed || !built.info.has_perfect_matching {
        return None;
    }
    sample_covering(built, 7, 160).ok().flatten()
}

fn main() {
    dioxus::launch(App);
}

#[component]
fn App() -> Element {
    // Theme management (System / Light / Dark)
    let mut theme_mode = use_signal(interop::load_theme_mode);
    let resolved_theme = use_memo(move || interop::resolve_theme(theme_mode()));
    use_effect(move || {
        interop::apply_theme_to_dom(theme_mode(), resolved_theme());
    });

    // Curated presets from permanent-wasm (zero duplication)
    let presets = use_hook(list_presets);
    let default_preset = presets.first().cloned().expect("preset catalog must be non-empty");

    let mut active_preset_id = use_signal(move || Some(default_preset.id.clone()));
    let mut active_spec = use_signal({
        let s = default_preset.spec.clone();
        move || s
    });
    let mut run_config = use_signal({
        let c = default_preset.recommended_config.clone();
        move || c
    });

    // Local fast-preview BuiltProblem cache (instant frame-1 3D render!)
    let local_built: Rc<RefCell<Option<BuiltProblem>>> = use_hook(|| Rc::new(RefCell::new(None)));

    // Build initial default 4x4x4 cube immediately (< 0.3 ms) so first paint is populated!
    let initial_info = use_hook({
        let spec0 = default_preset.spec.clone();
        let cache = Rc::clone(&local_built);
        move || match build_problem(&spec0) {
            Ok(built) => {
                let info = built.info.clone();
                let init_sample = initial_sample_for(&spec0, &built);
                *cache.borrow_mut() = Some(built);
                (Some(info), init_sample, None)
            }
            Err(err) => (None, None, Some(format!("{err:#}"))),
        }
    });

    let mut problem_info = use_signal(move || initial_info.0.clone());
    let mut displayed_sample = use_signal(move || initial_info.1.clone());
    let mut build_error = use_signal(move || initial_info.2.clone());
    let mut showing_live_sample = use_signal(|| false);
    let mut sample_seed = use_signal(|| 1u64);
    let mut sampling_busy = use_signal(|| false);
    let mut build_gen = use_signal(|| 0u32);

    // Right panel tab & live typed telemetry
    let mut active_tab = use_signal(|| ActiveTab::Configure);
    let mut telemetry = use_signal(TelemetryState::default);

    // Background Dedicated WebGPU Worker (lazily boots without blocking first paint)
    let worker_client = use_hook(|| GpuWorkerClient::spawn().ok());
    let mut worker_status = use_signal(|| WorkerStatus::Loading);

    use_effect({
        let worker = worker_client.clone();
        move || {
            let Some(client) = worker.clone() else {
                worker_status.set(WorkerStatus::Error("Web Workers unavailable in this browser".into()));
                return;
            };
            spawn(async move {
                match client.probe().await {
                    Ok(caps) => worker_status.set(WorkerStatus::Ready(caps)),
                    Err(err) => worker_status.set(WorkerStatus::Error(err)),
                }
            });
        }
    });

    // Rebuild model whenever `active_spec` changes (fast local for tiny problems,
    // off-thread worker for large matrices or once worker is ready).
    let rebuild_problem = {
        let worker = worker_client.clone();
        let cache = Rc::clone(&local_built);
        move |spec: ProblemSpec| {
            let generation = build_gen.peek().wrapping_add(1);
            build_gen.set(generation);
            displayed_sample.set(None);
            showing_live_sample.set(false);
            build_error.set(None);

            // Fast synchronous build for small interactive specs so UI sliders feel instantaneous.
            let try_local = match &spec {
                ProblemSpec::Dimer3d { lx, ly, lz, .. } => (lx * ly * lz) / 2 <= FAST_LOCAL_MAX_N,
                ProblemSpec::Dimer2d { rows, cols, .. } => (rows * cols) / 2 <= FAST_LOCAL_MAX_N,
                ProblemSpec::DisjointCycles { cycle_sizes } => cycle_sizes.iter().sum::<usize>() <= FAST_LOCAL_MAX_N,
                ProblemSpec::RandomMatrix { n, .. } => *n <= FAST_LOCAL_MAX_N,
                ProblemSpec::BundledPreset { name } => {
                    matches!(name.as_str(), "weighted-8" | "weighted-12" | "weighted-16" | "dimer-8x8")
                }
                ProblemSpec::MatrixMarket { text } => text.len() < 4096,
            };

            if try_local {
                match build_problem(&spec) {
                    Ok(built) => {
                        let init_sample = initial_sample_for(&spec, &built);
                        problem_info.set(Some(built.info.clone()));
                        displayed_sample.set(init_sample);
                        *cache.borrow_mut() = Some(built);
                        // Also warm the worker cache asynchronously if ready.
                        if let Some(client) = worker.clone()
                            && matches!(*worker_status.peek(), WorkerStatus::Ready(_))
                        {
                            spawn(async move {
                                let _ = client.build_problem(&spec).await;
                            });
                        }
                    }
                    Err(err) => {
                        *cache.borrow_mut() = None;
                        build_error.set(Some(format!("{err:#}")));
                    }
                }
                return;
            }

            // Large problem: build asynchronously inside the Dedicated Worker so 3D UI never stalls.
            *cache.borrow_mut() = None;
            if let Some(client) = worker.clone() {
                let want_mixed = matches!(
                    spec,
                    ProblemSpec::Dimer3d { .. } | ProblemSpec::Dimer2d { .. } | ProblemSpec::DisjointCycles { .. }
                );
                spawn(async move {
                    match client.build_problem(&spec).await {
                        Ok(info) => {
                            if *build_gen.peek() == generation {
                                let has_pm = info.has_perfect_matching;
                                problem_info.set(Some(info));
                                build_error.set(None);
                                if want_mixed
                                    && has_pm
                                    && let Ok(Some(snap)) = client.sample_covering(&spec, 7, 160).await
                                    && *build_gen.peek() == generation
                                {
                                    displayed_sample.set(Some(snap));
                                }
                            }
                        }
                        Err(err) => {
                            if *build_gen.peek() == generation {
                                build_error.set(Some(err));
                            }
                        }
                    }
                });
            } else {
                // Fallback if Worker API is unavailable
                match build_problem(&spec) {
                    Ok(built) => {
                        problem_info.set(Some(built.info.clone()));
                        *cache.borrow_mut() = Some(built);
                    }
                    Err(err) => build_error.set(Some(format!("{err:#}"))),
                }
            }
        }
    };

    let on_select_preset = {
        let mut rebuild = rebuild_problem.clone();
        move |preset: PresetInfo| {
            if telemetry.peek().is_running() {
                return;
            }
            active_preset_id.set(Some(preset.id.clone()));
            active_spec.set(preset.spec.clone());
            run_config.set(preset.recommended_config.clone());
            telemetry.set(TelemetryState::default());
            rebuild(preset.spec);
        }
    };

    let on_select_custom_mtx = {
        let mut rebuild = rebuild_problem.clone();
        move |()| {
            if telemetry.peek().is_running() {
                return;
            }
            let spec = ProblemSpec::MatrixMarket { text: SAMPLE_MTX_WEIGHTED.to_owned() };
            active_preset_id.set(Some("custom-mtx".into()));
            active_spec.set(spec.clone());
            run_config.set(WebRunConfig { compute_exact: true, ..Default::default() });
            telemetry.set(TelemetryState::default());
            active_tab.set(ActiveTab::Configure);
            rebuild(spec);
        }
    };

    let on_change_spec = {
        let mut rebuild = rebuild_problem.clone();
        move |spec: ProblemSpec| {
            active_spec.set(spec.clone());
            rebuild(spec);
        }
    };

    let on_random_covering = {
        let worker = worker_client.clone();
        let cache = Rc::clone(&local_built);
        move |()| {
            let next_seed = sample_seed.peek().wrapping_add(1);
            sample_seed.set(next_seed);
            showing_live_sample.set(false);

            // Sub-millisecond draw directly on cached local BuiltProblem when available.
            if let Some(ref built) = *cache.borrow() {
                if let Ok(Some(snap)) = sample_covering(built, next_seed, 96) {
                    displayed_sample.set(Some(snap));
                }
                return;
            }

            // Otherwise draw asynchronously from the worker's kept Problem handle.
            let Some(client) = worker.clone() else { return };
            let spec = active_spec.peek().clone();
            sampling_busy.set(true);
            spawn(async move {
                if let Ok(Some(snap)) = client.sample_covering(&spec, next_seed, 96).await {
                    displayed_sample.set(Some(snap));
                }
                sampling_busy.set(false);
            });
        }
    };

    let on_heaviest_matching = move |()| {
        showing_live_sample.set(false);
        displayed_sample.set(None);
    };

    let on_start_run = {
        let worker = worker_client.clone();
        move |_| {
            let Some(client) = worker.clone() else { return };
            let spec = active_spec.peek().clone();
            let cfg = run_config.peek().clone();

            telemetry.write().reset_for_run();
            active_tab.set(ActiveTab::Telemetry);
            showing_live_sample.set(true);

            spawn(async move {
                let res = client
                    .run_estimation(&spec, &cfg, move |event| {
                        if let WebEvent::Sample { beta, ref matching, valid } = event {
                            displayed_sample.set(Some(SampleSnapshot { beta, matching: matching.clone(), valid }));
                            showing_live_sample.set(true);
                        }
                        telemetry.write().apply_event(event);
                    })
                    .await;

                if let Err(err) = res {
                    let mut t = telemetry.write();
                    if t.error.is_none() {
                        t.apply_event(WebEvent::Failed { message: err });
                    }
                }
            });
        }
    };

    let on_cancel_run = {
        let worker = worker_client.clone();
        move |_| {
            if let Some(client) = worker.as_ref() {
                client.cancel();
            }
        }
    };

    let running = telemetry.read().is_running();
    let webgpu_ready = matches!(worker_status(), WorkerStatus::Ready(ref c) if c.webgpu_available);
    let webgpu_missing = matches!(worker_status(), WorkerStatus::Ready(ref c) if !c.webgpu_available);
    let worker_err = match worker_status() {
        WorkerStatus::Error(e) => Some(e),
        _ => None,
    };

    let current_problem: Option<GeneratedProblem> = problem_info();

    rsx! {
        Navbar {
            theme_mode: theme_mode(),
            resolved_theme: resolved_theme(),
            worker_status: worker_status(),
            on_cycle_theme: move |_| theme_mode.set(theme_mode().next()),
        }

        Hero { resolved_theme: resolved_theme() }

        main { id: "demos", class: "workbench-section",
            div { class: "section-header",
                div {
                    h2 { class: "section-title", "Interactive Permanent Workbench" }
                    p { class: "section-sub",
                        "Select a worked benchmark or generate a custom lattice/matrix, explore valid coverings instantaneously, and run live WebGPU estimation."
                    }
                }
            }

            PresetStrip {
                presets: presets.clone(),
                active_preset_id: active_preset_id(),
                on_select_preset,
                on_select_custom_mtx,
            }

            div { class: "workbench-grid",
                // Left: Monograph visualization stage
                ViewerStage {
                    problem: current_problem,
                    displayed_sample: displayed_sample(),
                    showing_live_sample: showing_live_sample(),
                    sampling_busy: sampling_busy(),
                    build_error: build_error(),
                    resolved_theme: resolved_theme(),
                    on_random_covering,
                    on_heaviest_matching,
                }

                // Right: Run bar + Configure / Live Telemetry tabs
                div { class: "panel-frame",
                    div { class: "run-bar",
                        div { class: "run-bar-row",
                            if running {
                                button {
                                    class: "btn-cancel",
                                    r#type: "button",
                                    onclick: on_cancel_run,
                                    "Cancel Run"
                                }
                            } else {
                                button {
                                    class: "btn-run",
                                    r#type: "button",
                                    disabled: !webgpu_ready || build_error().is_some(),
                                    onclick: on_start_run,
                                    if matches!(worker_status(), WorkerStatus::Loading) {
                                        "Initializing WebGPU…"
                                    } else {
                                        "Run on WebGPU"
                                    }
                                }
                            }
                        }

                        if webgpu_missing {
                            div { class: "notice-banner warn",
                                "WebGPU is unavailable in this browser. Estimation requires WebGPU (WGSL), while model generation, 3D/2D inspection, random coverings, and exact small references remain fully interactive."
                            }
                        }
                        if let Some(err) = worker_err {
                            div { class: "notice-banner err", "Worker error: {err}" }
                        }
                    }

                    div { class: "tab-strip", role: "tablist",
                        button {
                            r#type: "button",
                            role: "tab",
                            "aria-selected": active_tab() == ActiveTab::Configure,
                            class: if active_tab() == ActiveTab::Configure { "tab-btn active" } else { "tab-btn" },
                            onclick: move |_| active_tab.set(ActiveTab::Configure),
                            "Configure"
                        }
                        button {
                            r#type: "button",
                            role: "tab",
                            "aria-selected": active_tab() == ActiveTab::Telemetry,
                            class: if active_tab() == ActiveTab::Telemetry { "tab-btn active" } else { "tab-btn" },
                            onclick: move |_| active_tab.set(ActiveTab::Telemetry),
                            "Live Telemetry"
                            if running {
                                span { class: "gpu-dot pending" }
                            }
                        }
                    }

                    if active_tab() == ActiveTab::Configure {
                        ControlsPanel {
                            spec: active_spec(),
                            config: run_config(),
                            running,
                            on_change_spec,
                            on_change_config: move |cfg| run_config.set(cfg),
                        }
                    } else {
                        TelemetryPanel { telemetry: telemetry() }
                    }
                }
            }
        }

        Footer {}
    }
}
