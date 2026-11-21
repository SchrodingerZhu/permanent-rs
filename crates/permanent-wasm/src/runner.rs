//! Asynchronous estimation driver for the browser, streaming structured
//! `WebEvent`s (1:1 with `permanent-cli`'s TUI dashboard) and report lines.

use std::cell::RefCell;
use std::sync::atomic::AtomicBool;

use permanent_core::comm::Solo;
use permanent_core::engine::{AsyncEngine, Blocking};
use permanent_core::estimator::{AsyncSession, Cancelled, Config, Progress, Verdict};
use permanent_core::exact;
use permanent_core::scaling::{Prepared, Scaled, prepare};
use permanent_core::stats::{ln_factorial, scientific};
use permanent_rayon::Rayon;
use permanent_wgpu::Wgpu;
use rand::rngs::SmallRng;

use crate::generators::{BROWSER_MAX_EXACT_N, BuiltProblem, build_problem};
use crate::types::{
    BackendKind, EstimateSummary, ProblemSpec, RunSummary, SampleSnapshot, VerdictInfo, WebEvent, WebRunConfig,
};

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

/// Run the full permanent estimation pipeline for `spec` and `config`, streaming
/// live telemetry events to `on_event`.
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
    let problem = &built.problem;
    let n = problem.n;
    let nnz = built.info.nnz;
    let is_zero_one = built.info.is_zero_one;

    say!(sink, "matrix: {n} x {n}, {nnz} nonzero entries{}", if is_zero_one { " (0/1)" } else { " (weighted)" });

    let (exact_ln, exact_source) = resolve_exact(&built, config, sink);

    let scaled = match prepare(problem) {
        Prepared::Zero => {
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
        }
        Prepared::Scaled(scaled) => scaled,
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

    match config.backend {
        BackendKind::Cpu => {
            let engine = Blocking(Rayon::<f64, SmallRng>::new(&scaled));
            run_with_engine(&built, &scaled, &core_config, config, exact_ln, exact_source, engine, cancel, sink).await
        }
        BackendKind::Wgpu => {
            let engine = init_wgpu(&scaled, config).await?;
            run_with_engine(&built, &scaled, &core_config, config, exact_ln, exact_source, engine, cancel, sink).await
        }
        BackendKind::Auto => match init_wgpu(&scaled, config).await {
            Ok(engine) => {
                run_with_engine(&built, &scaled, &core_config, config, exact_ln, exact_source, engine, cancel, sink)
                    .await
            }
            Err(error) => {
                say!(sink, "WebGPU unavailable ({error:#}); falling back to CPU Wasm engine");
                let engine = Blocking(Rayon::<f64, SmallRng>::new(&scaled));
                run_with_engine(&built, &scaled, &core_config, config, exact_ln, exact_source, engine, cancel, sink)
                    .await
            }
        },
    }
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
            s.say(format!(
                "  {:>6} islands   ln(per) = {:.5} ± {:.5}   {:.3e} mutation + {:.3e} sampler events, {:.1} s",
                e.islands, e.ln_permanent, e.std_error, e.mutate.events as f64, e.sampler.events as f64, e.seconds,
            ));
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

    say!(sink, "phase 1 (learn weights, build schedule): {:.1} s", d.phase1_seconds);
    say!(sink, "  temperatures:                {}", levels.len());
    say!(sink, "  population:                  {} particles on {engine_name}", d.learn_particles);
    say!(
        sink,
        "  work:                        {} learning passes, {:.3e} mutation events, {:.3e} FAR steps",
        d.learn_iterations,
        d.mutate.events as f64,
        d.far_steps as f64
    );
    if d.mutations > 0 {
        say!(
            sink,
            "  mutation (t = {}):         {:.0} events each, {:.3} of the time in perfect matchings",
            web_config.learn_mutation_time,
            d.mutate.events as f64 / d.mutations as f64,
            web_config.learn_mutation_time * d.mutations as f64 / d.mutate.time
        );
    }
    say!(sink, "  worst first-pass correction: x{:.3}", d.worst_first_correction.exp());
    say!(sink, "  weight check (should be ~1): {:.3} to {:.3}", d.check_min, d.check_max);
    if d.learn_capped_chains + d.mutate.over_budget > 0 {
        say!(
            sink,
            "  WARNING: {} capped FAR chains, {} mutations over budget",
            d.learn_capped_chains,
            d.mutate.over_budget
        );
    }

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

    say!(sink, "plan:");
    say!(
        sink,
        "  segments:                    {} (independent; each starts from fresh sampler draws)",
        plan.segments.len()
    );
    say!(sink, "  islands x particles:         {} x {} per segment", plan.islands, plan.particles);
    say!(
        sink,
        "  variance per island:         planned {:.4} (sum of step variances {:.3}, margin 2); bound factor {:.2}",
        plan.island_relvar,
        plan.relvar_sum,
        plan.mixing_factor
    );
    say!(sink, "  target std error of ln(per): {:.4} (interval z = {:.3})", session.sigma_target, session.z);
    say!(sink, "  sampler stage time:          {:.4}", plan.stage_time);
    if plan.predicted_events > 0.0 {
        say!(sink, "  predicted:                   ~{:.3e} mutation events", plan.predicted_events);
    }

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
    let z = session.z;
    drop(session);

    say!(
        sink,
        "  measured variance per island: {:.4} with {} particles (planned {:.4})",
        est.island_relvar,
        est.particles,
        plan.island_relvar * plan.particles as f64 / est.particles as f64
    );
    let verdict = match est.verdict {
        Verdict::Consistent(gap) => {
            say!(
                sink,
                "  boundary check: carried and fresh populations agree (largest difference {gap:+.2} standard errors)"
            );
            VerdictInfo::Consistent { max_gap_se: gap }
        }
        Verdict::Inconsistent(gap) => {
            say!(
                sink,
                "  WARNING: carried and fresh populations differ by {gap:+.2} standard errors at a segment \
                 boundary; raise mutation time or enable rigorous sampler"
            );
            VerdictInfo::Inconsistent { max_gap_se: gap }
        }
        Verdict::Unchecked => {
            say!(sink, "  boundary check: none (one segment)");
            VerdictInfo::Unchecked
        }
    };
    let over_budget = est.mutate.over_budget + est.sampler.over_budget;
    if over_budget > 0 {
        say!(
            sink,
            "  WARNING: {} mutations and {} sampler trials hit their event cap",
            est.mutate.over_budget,
            est.sampler.over_budget
        );
    }
    say!(sink, "  regularization factor E[g]:  {:.4}", est.mean_gap);

    let half = z * est.std_error;
    let sci = scientific(est.ln_permanent);
    let ci_low_scientific = scientific(est.ln_permanent - half);
    let ci_high_scientific = scientific(est.ln_permanent + half);
    let ci_level_pct = 100.0 * (1.0 - web_config.delta / 2.0);
    let rel_half_width_pct = 100.0 * (half.exp() - 1.0);
    say!(sink, "permanent ≈ {sci}");
    say!(sink, "  ln(permanent) = {:.6} ± {:.6} (std error)", est.ln_permanent, est.std_error);
    say!(
        sink,
        "  {ci_level_pct:.1}% statistical interval: [{ci_low_scientific}, {ci_high_scientific}]  \
         (relative half-width {rel_half_width_pct:.2}%)"
    );
    if !est.met_target {
        say!(sink, "  WARNING: the measured error did not reach the target within the top-up limit");
    }

    let (rel_error_pct, z_score) = if let Some(exact) = exact_ln {
        let rel = 100.0 * ((est.ln_permanent - exact).exp() - 1.0);
        let zs = (est.std_error > 0.0).then(|| (est.ln_permanent - exact) / est.std_error);
        if let Some(zs) = zs {
            say!(sink, "exact = {}; relative error {rel:+.3}% ({zs:+.2} std errors)", scientific(exact));
        } else {
            say!(sink, "exact = {}; relative error {rel:+.3}%", scientific(exact));
        }
        (Some(rel), zs)
    } else {
        (None, None)
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
            scientific: sci,
            ci_low_scientific,
            ci_high_scientific,
            ci_level_pct,
            rel_half_width_pct,
            met_target: est.met_target,
            rounds: est.rounds,
            islands: est.islands,
            particles: est.particles,
            island_relvar: est.island_relvar,
            mean_gap: est.mean_gap,
            verdict,
            mutate_events: est.mutate.events,
            sampler_events: est.sampler.events,
            over_budget,
            seconds: est.seconds,
            rel_error_pct,
            z_score,
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::Ordering;
    use std::task::{Context, Poll, Waker};

    fn poll_now<F: Future>(future: F) -> F::Output {
        let mut future = std::pin::pin!(future);
        let mut cx = Context::from_waker(Waker::noop());
        match future.as_mut().poll(&mut cx) {
            Poll::Ready(out) => out,
            Poll::Pending => panic!("CPU backend should resolve immediately"),
        }
    }

    #[test]
    fn cpu_runner_streams_full_tui_telemetry_and_exact_comparison() {
        let spec = ProblemSpec::Dimer3d {
            lx: 2,
            ly: 2,
            lz: 2,
            periodic: false,
            weights: [1.0, 1.0, 1.0],
            disorder: 0.0,
            vacancy_rate: 0.0,
            ensure_matching: true,
            seed: 1,
        };
        let config = WebRunConfig {
            backend: BackendKind::Cpu,
            epsilon: 0.4,
            delta: 0.05,
            seed: 3,
            learn_particles: Some(128),
            particles: Some(16),
            islands: Some(16),
            segments: Some(2),
            capture_samples: true,
            save_schedule: true,
            ..Default::default()
        };
        let cancel = AtomicBool::new(false);
        let mut events = Vec::new();
        let summary = poll_now(run_problem(&spec, &config, &cancel, |ev| events.push(ev))).unwrap().unwrap();

        assert!(events.iter().any(|e| matches!(e, WebEvent::Init { .. })));
        assert!(events.iter().any(|e| matches!(e, WebEvent::Level { .. })));
        assert!(events.iter().any(|e| matches!(e, WebEvent::Sample { .. })));
        assert!(events.iter().any(|e| matches!(e, WebEvent::Plan { .. })));
        assert!(events.iter().any(|e| matches!(e, WebEvent::Sweep { .. })));
        assert!(events.iter().any(|e| matches!(e, WebEvent::Round { .. })));
        assert!(matches!(events.last(), Some(WebEvent::Done { .. })));

        let est = summary.estimate.as_ref().unwrap();
        let exact = 9.0_f64.ln();
        assert!((summary.exact_ln.unwrap() - exact).abs() < 1e-12);
        assert!(((est.ln_permanent - exact) / est.std_error).abs() < 4.0);
        assert!(est.entropy_per_site.is_some());
        assert!(summary.schedule_json.as_ref().unwrap().contains("\"levels\""));
    }

    #[test]
    fn runner_handles_plan_only_zero_permanent_and_cancellation() {
        // Plan-only run
        let spec = ProblemSpec::BundledPreset { name: "weighted-8".into() };
        let config = WebRunConfig {
            backend: BackendKind::Cpu,
            plan_only: true,
            learn_particles: Some(128),
            ..Default::default()
        };
        let cancel = AtomicBool::new(false);
        let mut events = Vec::new();
        let summary = poll_now(run_problem(&spec, &config, &cancel, |ev| events.push(ev))).unwrap().unwrap();
        assert!(summary.plan_only);
        assert!(summary.estimate.is_none());
        assert!(events.iter().any(|e| matches!(e, WebEvent::Plan { .. })));
        assert!(!events.iter().any(|e| matches!(e, WebEvent::Sweep { .. })));

        // Zero permanent matrix
        let zero = ProblemSpec::MatrixMarket {
            text: "%%MatrixMarket matrix coordinate real general\n3 3 4\n1 1 1\n2 1 2\n3 1 1\n3 2 1\n".into(),
        };
        let mut events = Vec::new();
        let summary = poll_now(run_problem(&zero, &config, &cancel, |ev| events.push(ev))).unwrap().unwrap();
        assert!(summary.zero_permanent);

        // Cancellation
        let cancel = AtomicBool::new(false);
        let mut saw_cancelled = false;
        let out = poll_now(run_problem(&spec, &config, &cancel, |ev| {
            if matches!(ev, WebEvent::Level { .. }) {
                cancel.store(true, Ordering::Relaxed);
            }
            if matches!(ev, WebEvent::Cancelled) {
                saw_cancelled = true;
            }
        }))
        .unwrap();
        assert!(out.is_none());
        assert!(saw_cancelled);
    }
}
