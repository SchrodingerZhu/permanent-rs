//! Asynchronous WebGPU estimation driver for the browser, streaming structured
//! `WebEvent`s (1:1 with `permanent-cli`'s TUI dashboard) and report lines.

use std::cell::RefCell;
use std::sync::atomic::AtomicBool;

use permanent_core::comm::Solo;
use permanent_core::engine::AsyncEngine;
use permanent_core::estimator::{AsyncSession, Cancelled, Config, Progress, Verdict};
use permanent_core::exact;
use permanent_core::report::{self, Accuracy};
use permanent_core::scaling::Scaled;
use permanent_core::stats::{ln_factorial, scientific};
use permanent_demo::{BuiltProblem, ProblemSpec, SampleSnapshot, build_problem};
use permanent_wgpu::Wgpu;

use crate::types::{EstimateSummary, RunSummary, VerdictInfo, WebEvent, WebRunConfig};

/// Maximum `n` for general subset-DP exact permanent in the browser
/// (`2^20 * 8` bytes = 8 MiB). Closed-form references work at any size.
pub const BROWSER_MAX_EXACT_N: usize = 20;

struct Sink<F: FnMut(WebEvent)> {
    emit: F,
    lines: Vec<String>,
    last_sample: Option<SampleSnapshot>,
}

impl<F: FnMut(WebEvent)> Sink<F> {
    fn new(emit: F) -> Self {
        Self { emit, lines: Vec::new(), last_sample: None }
    }

    fn send(&mut self, event: WebEvent) {
        (self.emit)(event);
    }

    fn say(&mut self, text: String) {
        self.lines.push(text.clone());
        (self.emit)(WebEvent::Line { text });
    }
}

macro_rules! say {
    ($sink:expr, $($arg:tt)*) => {
        $sink.borrow_mut().say(format!($($arg)*))
    };
}

/// Run the full permanent estimation pipeline on WebGPU for `spec` and
/// `config`, streaming live telemetry events to `on_event`.
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
    match run_inner(spec, config, cancel, &sink).await {
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
    spec: &ProblemSpec,
    config: &WebRunConfig,
    cancel: &AtomicBool,
    sink: &RefCell<Sink<F>>,
) -> anyhow::Result<Option<RunSummary>> {
    let built = build_problem(spec)?;
    let n = built.problem.n;
    let nnz = built.info.nnz;
    let is_zero_one = built.info.is_zero_one;

    say!(sink, "matrix: {n} x {n}, {nnz} nonzero entries{}", if is_zero_one { " (0/1)" } else { " (weighted)" });

    let (exact_ln, exact_source) = resolve_exact(&built, config, sink);

    let Some(scaled) = &built.scaled else {
        say!(sink, "permanent = 0 exactly (the support has no perfect matching)");
        sink.borrow_mut().send(WebEvent::Init {
            n,
            nnz,
            is_zero_one,
            engine: "none".into(),
            exact_ln: None,
            exact_source: Some("no perfect matching".into()),
        });
        let lines = sink.borrow().lines.clone();
        let summary = RunSummary {
            n,
            nnz,
            is_zero_one,
            engine: "none".into(),
            seed: config.seed,
            plan_only: config.plan_only,
            zero_permanent: true,
            temperatures: 0,
            phase1_seconds: 0.0,
            learn_particles: 0,
            learn_iterations: 0,
            far_steps: 0,
            worst_first_correction_factor: 1.0,
            check_min: 1.0,
            check_max: 1.0,
            exact_ln: None,
            exact_scientific: Some("0".into()),
            exact_source: Some("no perfect matching".into()),
            estimate: None,
            final_sample: None,
            schedule_json: None,
            lines,
        };
        sink.borrow_mut().send(WebEvent::Done { summary: Box::new(summary.clone()) });
        return Ok(Some(summary));
    };

    say!(
        sink,
        "target: relative error {} with probability {}; seed {}",
        config.epsilon,
        1.0 - config.delta,
        config.seed
    );

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
    run_with_engine(&built, scaled, &core_config, config, exact_ln, exact_source, engine, cancel, sink).await
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

fn resolve_exact<F: FnMut(WebEvent)>(
    built: &BuiltProblem,
    config: &WebRunConfig,
    sink: &RefCell<Sink<F>>,
) -> (Option<f64>, Option<String>) {
    if let Some(exact_ln) = built.info.exact_ln {
        return (Some(exact_ln), built.info.exact_source.clone());
    }
    if !config.compute_exact {
        return (None, None);
    }
    let n = built.problem.n;
    if n > BROWSER_MAX_EXACT_N {
        say!(sink, "exact: skipped in browser, n > {BROWSER_MAX_EXACT_N}");
        return (None, None);
    }
    match exact::ln_permanent(&built.problem) {
        Some(v) if v.is_finite() => (Some(v), Some("subset DP".into())),
        _ => (None, None),
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
        exact_ln,
        exact_source: exact_source.clone(),
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
            let mut s = sink.borrow_mut();
            s.say(report::round(e));
            s.send(WebEvent::Round {
                round: index,
                islands: e.islands,
                ln_permanent: e.ln_permanent,
                std_error: e.std_error,
            });
        }
    });

    let levels = session.learn_schedule().await?;
    let d = session.diagnostics.clone();

    report::phase1(&d, levels.len(), &engine_name, core_config, |line| sink.borrow_mut().say(line));

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
    sink.borrow_mut().send(WebEvent::Plan {
        segments: plan.segments.iter().map(|r| [r.start, r.end]).collect(),
        ln_scale,
        particles: plan.particles,
        islands: plan.islands,
        relvar_sum: plan.relvar_sum,
        mixing_factor: plan.mixing_factor,
        island_relvar: plan.island_relvar,
        stage_time: plan.stage_time,
        sigma_target: session.sigma_target,
        z: session.z,
    });

    report::plan(&plan, session.sigma_target, session.z, |line| sink.borrow_mut().say(line));

    let exact_scientific = exact_ln.map(scientific);
    if web_config.plan_only {
        drop(session);
        let (final_sample, lines) = {
            let s = sink.borrow();
            (s.last_sample.clone(), s.lines.clone())
        };
        let summary = RunSummary {
            n,
            nnz,
            is_zero_one,
            engine: engine_name,
            seed: web_config.seed,
            plan_only: true,
            zero_permanent: false,
            temperatures: levels.len(),
            phase1_seconds: d.phase1_seconds,
            learn_particles: d.learn_particles,
            learn_iterations: d.learn_iterations,
            far_steps: d.far_steps,
            worst_first_correction_factor: d.worst_first_correction.exp(),
            check_min: if d.check_min.is_finite() { d.check_min } else { 1.0 },
            check_max: if d.check_max.is_finite() { d.check_max } else { 1.0 },
            exact_ln,
            exact_scientific,
            exact_source,
            estimate: None,
            final_sample,
            schedule_json,
            lines,
        };
        sink.borrow_mut().send(WebEvent::Done { summary: Box::new(summary.clone()) });
        return Ok(Some(summary));
    }

    say!(sink, "phase 2 (estimate):");
    let est = session.estimate(&levels, &plan).await?;
    let accuracy = Accuracy::new(&est, session.z, core_config, exact_ln);
    drop(session);
    report::estimate(&est, &plan, &accuracy, |line| sink.borrow_mut().say(line));

    let verdict = match est.verdict {
        Verdict::Consistent(gap) => VerdictInfo::Consistent { max_gap_se: gap },
        Verdict::Inconsistent(gap) => VerdictInfo::Inconsistent { max_gap_se: gap },
        Verdict::Unchecked => VerdictInfo::Unchecked,
    };

    let (entropy_per_site, entropy_std_error) = if let Some(sites) = built.info.lattice_sites {
        let s = est.ln_permanent / sites as f64;
        let se = est.std_error / sites as f64;
        say!(sink, "entropy per site = {s:.5} ± {se:.5} ({sites} sites)");
        (Some(s), Some(se))
    } else {
        (None, None)
    };

    let (final_sample, lines) = {
        let s = sink.borrow();
        (s.last_sample.clone(), s.lines.clone())
    };
    let summary = RunSummary {
        n,
        nnz,
        is_zero_one,
        engine: engine_name,
        seed: web_config.seed,
        plan_only: false,
        zero_permanent: false,
        temperatures: levels.len(),
        phase1_seconds: d.phase1_seconds,
        learn_particles: d.learn_particles,
        learn_iterations: d.learn_iterations,
        far_steps: d.far_steps,
        worst_first_correction_factor: d.worst_first_correction.exp(),
        check_min: if d.check_min.is_finite() { d.check_min } else { 1.0 },
        check_max: if d.check_max.is_finite() { d.check_max } else { 1.0 },
        exact_ln,
        exact_scientific,
        exact_source,
        estimate: Some(EstimateSummary {
            ln_permanent: est.ln_permanent,
            std_error: est.std_error,
            scientific: scientific(est.ln_permanent),
            ci_low_scientific: scientific(accuracy.ln_low),
            ci_high_scientific: scientific(accuracy.ln_high),
            ci_level_pct: accuracy.level_percent,
            rel_half_width_pct: accuracy.half_width_percent,
            met_target: est.met_target,
            rounds: est.rounds,
            islands: est.islands,
            particles: est.particles,
            island_relvar: est.island_relvar,
            mean_gap: est.mean_gap,
            verdict,
            mutate_events: est.mutate.events,
            sampler_events: est.sampler.events,
            over_budget: est.mutate.over_budget + est.sampler.over_budget,
            seconds: est.seconds,
            rel_error_pct: accuracy.against_exact.map(|exact| exact.error_percent),
            z_score: accuracy.against_exact.and_then(|exact| exact.std_errors),
            entropy_per_site,
            entropy_std_error,
            entropy_lower_bound: built.info.entropy_lower_bound,
        }),
        final_sample,
        schedule_json,
        lines,
    };
    sink.borrow_mut().send(WebEvent::Done { summary: Box::new(summary.clone()) });
    Ok(Some(summary))
}
