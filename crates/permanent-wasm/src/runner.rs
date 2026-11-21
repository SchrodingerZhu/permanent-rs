//! Asynchronous WebGPU estimation driver, streaming purely typed `WebEvent`
//! telemetry directly to Rust/Dioxus or JS front ends.

use std::cell::RefCell;
use std::sync::atomic::AtomicBool;

use permanent_core::comm::Solo;
use permanent_core::engine::AsyncEngine;
use permanent_core::estimator::{AsyncSession, Cancelled, Config, Progress, Verdict};
use permanent_core::exact;
use permanent_core::scaling::Scaled;
use permanent_core::stats::{ln_factorial, scientific};
use permanent_demo::{BuiltProblem, ProblemSpec, SampleSnapshot, build_problem};
use permanent_wgpu::Wgpu;

use crate::types::{EstimateSummary, Phase1Summary, PlanSummary, RunSummary, VerdictInfo, WebEvent, WebRunConfig};

/// Maximum `n` for general subset-DP exact permanent in the browser
/// (`2^20 * 8` bytes = 8 MiB). Closed-form references work at any size.
pub const BROWSER_MAX_EXACT_N: usize = 20;

struct Sink<F: FnMut(WebEvent)> {
    emit: F,
    last_sample: Option<SampleSnapshot>,
}

impl<F: FnMut(WebEvent)> Sink<F> {
    fn new(emit: F) -> Self {
        Self { emit, last_sample: None }
    }

    fn send(&mut self, event: WebEvent) {
        (self.emit)(event);
    }
}

/// Build `spec` and run the full permanent estimation pipeline on WebGPU,
/// streaming structured telemetry events to `on_event`.
///
/// Returns `Ok(Some(summary))` on completion, `Ok(None)` if cooperatively
/// cancelled via `cancel`, or `Err` on invalid inputs / runtime errors (after
/// emitting `WebEvent::Failed`).
pub async fn run_problem<F: FnMut(WebEvent)>(
    spec: &ProblemSpec,
    config: &WebRunConfig,
    cancel: &AtomicBool,
    on_event: F,
) -> anyhow::Result<Option<RunSummary>> {
    let sink = RefCell::new(Sink::new(on_event));
    match build_problem(spec) {
        Ok(built) => run_built_inner(&built, config, cancel, &sink).await,
        Err(error) => {
            sink.borrow_mut().send(WebEvent::Failed { message: format!("{error:#}") });
            Err(error)
        }
    }
}

/// Run the full permanent estimation pipeline on WebGPU for an already-built
/// [`BuiltProblem`], avoiding repeating matrix construction and `prepare()`.
///
/// Intended for Rust front ends (such as Dioxus) that already hold a
/// [`BuiltProblem`] for interactive visualization and covering sampling.
pub async fn run_built_problem<F: FnMut(WebEvent)>(
    built: &BuiltProblem,
    config: &WebRunConfig,
    cancel: &AtomicBool,
    on_event: F,
) -> anyhow::Result<Option<RunSummary>> {
    let sink = RefCell::new(Sink::new(on_event));
    run_built_inner(built, config, cancel, &sink).await
}

async fn run_built_inner<F: FnMut(WebEvent)>(
    built: &BuiltProblem,
    config: &WebRunConfig,
    cancel: &AtomicBool,
    sink: &RefCell<Sink<F>>,
) -> anyhow::Result<Option<RunSummary>> {
    match run_inner(built, config, cancel, sink).await {
        Ok(outcome) => Ok(outcome),
        Err(error) => {
            if error.is::<Cancelled>() {
                sink.borrow_mut().send(WebEvent::Cancelled);
                return Ok(None);
            }
            let message = format!("{error:#}");
            sink.borrow_mut().send(WebEvent::Failed { message });
            Err(error)
        }
    }
}

async fn run_inner<F: FnMut(WebEvent)>(
    built: &BuiltProblem,
    config: &WebRunConfig,
    cancel: &AtomicBool,
    sink: &RefCell<Sink<F>>,
) -> anyhow::Result<Option<RunSummary>> {
    let n = built.problem.n;
    let nnz = built.info.nnz;
    let is_zero_one = built.info.is_zero_one;

    let (exact_ln, exact_source, exact_skipped_too_large) = resolve_exact(built, config);

    let Some(scaled) = &built.scaled else {
        sink.borrow_mut().send(WebEvent::Init {
            n,
            nnz,
            is_zero_one,
            engine: "none".into(),
            epsilon: config.epsilon,
            delta: config.delta,
            seed: config.seed,
            exact_ln: None,
            exact_source: Some("no perfect matching".into()),
            exact_skipped_too_large: false,
        });
        let summary = RunSummary {
            n,
            nnz,
            is_zero_one,
            engine: "none".into(),
            epsilon: config.epsilon,
            delta: config.delta,
            seed: config.seed,
            plan_only: config.plan_only,
            zero_permanent: true,
            exact_ln: None,
            exact_scientific: Some("0".into()),
            exact_source: Some("no perfect matching".into()),
            exact_skipped_too_large: false,
            phase1: None,
            plan: None,
            estimate: None,
            final_sample: None,
            schedule_json: None,
        };
        sink.borrow_mut().send(WebEvent::Done { summary: Box::new(summary.clone()) });
        return Ok(Some(summary));
    };

    let core_config = Config {
        epsilon: config.epsilon,
        delta: config.delta,
        seed: config.seed,
        rigorous_schedule: config.rigorous_schedule,
        rigorous_sampler: config.rigorous_sampler,
        density: config.density,
        mutation_time: config.mutation_time,
        learn_mutation_time: config.learn_mutation_time,
        learn_particles: config.learn_particles,
        particles: config.particles,
        islands: config.islands,
        segments: config.segments,
    };

    let engine = init_wgpu(scaled, config).await?;
    run_with_engine(
        built,
        scaled,
        &core_config,
        config,
        exact_ln,
        exact_source,
        exact_skipped_too_large,
        engine,
        cancel,
        sink,
    )
    .await
}

async fn init_wgpu<'a>(scaled: &'a Scaled, config: &WebRunConfig) -> anyhow::Result<Wgpu<'a>> {
    #[cfg(target_family = "wasm")]
    let mut engine = Wgpu::new(scaled).await?;
    #[cfg(not(target_family = "wasm"))]
    let mut engine = Wgpu::new(scaled)?;
    if let Some(mb) = config.wgpu_working_set_mb {
        anyhow::ensure!((1..=512).contains(&mb), "wgpu_working_set_mb must be in 1..=512 (got {mb})");
        engine.working_set = mb << 20;
    }
    if let Some(budget) = config.wgpu_budget {
        anyhow::ensure!(budget >= 256, "wgpu_budget must be at least 256 (got {budget})");
        engine.budget = budget;
    }
    Ok(engine)
}

fn resolve_exact(built: &BuiltProblem, config: &WebRunConfig) -> (Option<f64>, Option<String>, bool) {
    if let Some(exact_ln) = built.info.exact_ln {
        return (Some(exact_ln), built.info.exact_source.clone(), false);
    }
    if !config.compute_exact {
        return (None, None, false);
    }
    if built.problem.n > BROWSER_MAX_EXACT_N {
        return (None, None, true);
    }
    match exact::ln_permanent(&built.problem) {
        Some(v) if v.is_finite() => (Some(v), Some("subset DP".into()), false),
        _ => (None, None, false),
    }
}

#[allow(clippy::too_many_arguments)]
async fn run_with_engine<E: AsyncEngine, F: FnMut(WebEvent)>(
    built: &BuiltProblem,
    scaled: &Scaled,
    core_config: &Config,
    web_config: &WebRunConfig,
    exact_ln: Option<f64>,
    exact_source: Option<String>,
    exact_skipped_too_large: bool,
    engine: E,
    cancel: &AtomicBool,
    sink: &RefCell<Sink<F>>,
) -> anyhow::Result<Option<RunSummary>> {
    let n = scaled.n;
    let nnz = built.info.nnz;
    let is_zero_one = built.info.is_zero_one;

    let mut session = AsyncSession::new(scaled, core_config, engine, &Solo)?;
    let engine_name = session.engine_name();
    session.cancel_flag(cancel);
    session.capture_samples(web_config.capture_samples);

    sink.borrow_mut().send(WebEvent::Init {
        n,
        nnz,
        is_zero_one,
        engine: engine_name.clone(),
        epsilon: web_config.epsilon,
        delta: web_config.delta,
        seed: web_config.seed,
        exact_ln,
        exact_source: exact_source.clone(),
        exact_skipped_too_large,
    });

    session.observe(|progress| match progress {
        Progress::Level { index, level } => {
            sink.borrow_mut().send(WebEvent::Level {
                index,
                beta: level.beta,
                step: level.step,
                relvar: level.relvar,
                log_w: level.log_w.clone(),
            });
        }
        Progress::Sample { beta, matching, valid } => {
            let snap = SampleSnapshot { beta, matching: matching.to_vec(), valid };
            let mut s = sink.borrow_mut();
            s.last_sample = Some(snap.clone());
            s.send(WebEvent::Sample { beta: snap.beta, matching: snap.matching, valid: snap.valid });
        }
        Progress::Sweep { round, done, total, partial } => {
            sink.borrow_mut().send(WebEvent::Sweep {
                round,
                done,
                total,
                segment: partial.segment,
                level: partial.level,
                islands: partial.islands,
                ln_mean: partial.ln_mean,
                ln_mean_square: partial.ln_mean_square,
            });
        }
        Progress::Round { index, estimate: e } => {
            sink.borrow_mut().send(WebEvent::Round {
                round: index,
                islands: e.islands,
                ln_permanent: e.ln_permanent,
                std_error: e.std_error,
                mutate_events: e.mutate.events,
                sampler_events: e.sampler.events,
                seconds: e.seconds,
            });
        }
    });

    let levels = session.learn_schedule().await?;
    let d = session.diagnostics.clone();

    let phase1 = Phase1Summary {
        temperatures: levels.len(),
        seconds: d.phase1_seconds,
        learn_particles: d.learn_particles,
        learn_iterations: d.learn_iterations,
        far_steps: d.far_steps,
        mutate_events: d.mutate.events,
        mutations: d.mutations,
        mutate_time: d.mutate.time,
        learn_mutation_time: core_config.learn_mutation_time,
        learn_capped_chains: d.learn_capped_chains,
        mutate_over_budget: d.mutate.over_budget,
        worst_first_correction_factor: d.worst_first_correction.exp(),
        check_min: if d.check_min.is_finite() { d.check_min } else { 1.0 },
        check_max: if d.check_max.is_finite() { d.check_max } else { 1.0 },
    };
    sink.borrow_mut().send(WebEvent::Phase1(phase1.clone()));

    let schedule_json = if web_config.save_schedule {
        let payload = serde_json::json!({
            "n": n,
            "mutation_time": web_config.mutation_time,
            "energy": scaled.energy,
            "log_gap": scaled.log_gap.iter().map(|g| g.max(f64::MIN)).collect::<Vec<_>>(),
            "log_scale": scaled.log_scale,
            "levels": levels,
        });
        Some(serde_json::to_string(&payload)?)
    } else {
        None
    };

    let plan = session.plan(&levels);
    let ln_scale = ln_factorial(n) - scaled.log_scale;
    let plan_summary = PlanSummary {
        segments: plan.segments.iter().map(|r| [r.start, r.end]).collect(),
        ln_scale,
        particles: plan.particles,
        islands: plan.islands,
        relvar_sum: plan.relvar_sum,
        mixing_factor: plan.mixing_factor,
        island_relvar: plan.island_relvar,
        stage_time: plan.stage_time,
        predicted_events: plan.predicted_events,
        sigma_target: session.sigma_target,
        z: session.z,
    };
    sink.borrow_mut().send(WebEvent::Plan(plan_summary.clone()));

    let exact_scientific = exact_ln.map(scientific);
    if web_config.plan_only {
        drop(session);
        let final_sample = sink.borrow().last_sample.clone();
        let summary = RunSummary {
            n,
            nnz,
            is_zero_one,
            engine: engine_name,
            epsilon: web_config.epsilon,
            delta: web_config.delta,
            seed: web_config.seed,
            plan_only: true,
            zero_permanent: false,
            exact_ln,
            exact_scientific,
            exact_source,
            exact_skipped_too_large,
            phase1: Some(phase1),
            plan: Some(plan_summary),
            estimate: None,
            final_sample,
            schedule_json,
        };
        sink.borrow_mut().send(WebEvent::Done { summary: Box::new(summary.clone()) });
        return Ok(Some(summary));
    }

    let est = session.estimate(&levels, &plan).await?;
    let accuracy = est.accuracy(session.z, core_config, exact_ln);
    drop(session);

    let verdict = match est.verdict {
        Verdict::Consistent(gap) => VerdictInfo::Consistent { max_gap_se: gap },
        Verdict::Inconsistent(gap) => VerdictInfo::Inconsistent { max_gap_se: gap },
        Verdict::Unchecked => VerdictInfo::Unchecked,
    };

    let (entropy_per_site, entropy_std_error) = if let Some(sites) = built.info.lattice_sites {
        (Some(est.ln_permanent / sites as f64), Some(est.std_error / sites as f64))
    } else {
        (None, None)
    };

    let final_sample = sink.borrow().last_sample.clone();
    let summary = RunSummary {
        n,
        nnz,
        is_zero_one,
        engine: engine_name,
        epsilon: web_config.epsilon,
        delta: web_config.delta,
        seed: web_config.seed,
        plan_only: false,
        zero_permanent: false,
        exact_ln,
        exact_scientific,
        exact_source,
        exact_skipped_too_large,
        phase1: Some(phase1),
        plan: Some(plan_summary),
        estimate: Some(EstimateSummary {
            ln_permanent: est.ln_permanent,
            std_error: est.std_error,
            scientific: scientific(est.ln_permanent),
            ln_ci_low: accuracy.ln_low,
            ln_ci_high: accuracy.ln_high,
            ci_low_scientific: scientific(accuracy.ln_low),
            ci_high_scientific: scientific(accuracy.ln_high),
            ci_level_pct: accuracy.level_percent,
            rel_half_width_pct: accuracy.half_width_percent,
            met_target: est.met_target,
            rounds: est.rounds,
            islands: est.islands,
            particles: est.particles,
            island_relvar: est.island_relvar,
            planned_island_relvar: plan.island_relvar * plan.particles as f64 / est.particles as f64,
            mean_gap: est.mean_gap,
            verdict,
            mutate_events: est.mutate.events,
            sampler_events: est.sampler.events,
            mutate_over_budget: est.mutate.over_budget,
            sampler_over_budget: est.sampler.over_budget,
            seconds: est.seconds,
            rel_error_pct: accuracy.against_exact.map(|exact| exact.error_percent),
            z_score: accuracy.against_exact.and_then(|exact| exact.std_errors),
            entropy_per_site,
            entropy_std_error,
            entropy_lower_bound: built.info.entropy_lower_bound,
        }),
        final_sample,
        schedule_json,
    };
    sink.borrow_mut().send(WebEvent::Done { summary: Box::new(summary.clone()) });
    Ok(Some(summary))
}
