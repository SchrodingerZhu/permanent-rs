//! The command line: reads the matrix, builds the engine `--backend` names
//! and reports the run.

use std::sync::Mutex;
use std::sync::mpsc::{Sender, channel};
use std::time::{Duration, Instant};

use clap::{ArgAction, Parser, ValueEnum};
use permanent_core::comm::{Comm, Solo};
use permanent_core::engine::Engine;
use permanent_core::estimator::{Config, Progress, Session, Verdict};
use permanent_core::exact;
use permanent_core::problem::Problem;
use permanent_core::scaling::{Prepared, Scaled, prepare};
use permanent_core::stats::scientific;
use tracing::level_filters::LevelFilter;
use tracing::{debug, info};
use tracing_subscriber::EnvFilter;

use crate::tui::{self, TuiEvent};

/// Where the population lives. The choices are the engines built in.
#[derive(ValueEnum, Debug, Clone, Copy, PartialEq, Eq)]
enum Backend {
    /// The CPU thread pool.
    #[cfg(feature = "rayon")]
    Rayon,
    /// An NVIDIA GPU; under MPI rank r uses device `--device` + r, modulo
    /// the number of devices on its node.
    #[cfg(feature = "cuda")]
    Cuda,
}

impl Backend {
    #[cfg(feature = "cuda")]
    const DEFAULT: Backend = Backend::Cuda;
    #[cfg(all(feature = "rayon", not(feature = "cuda")))]
    const DEFAULT: Backend = Backend::Rayon;
}

/// Type of the waiting times in the chains.
#[derive(ValueEnum, Debug, Clone, Copy, PartialEq, Eq)]
enum Precision {
    F32,
    F64,
}

/// Estimate the permanent of a nonnegative matrix to a requested accuracy.
///
/// Give the matrix and the accuracy; the temperatures, the population and
/// the number of independent runs are derived, not supplied.
#[derive(Parser, Debug)]
#[command(version)]
pub struct Cli {
    /// The matrix, in Matrix Market coordinate format (.mtx).
    input: std::path::PathBuf,
    /// Engine that holds the population.
    #[arg(short, long, value_enum, default_value_t = Backend::DEFAULT)]
    backend: Backend,
    /// Target relative error of the estimate.
    #[arg(short, long, default_value_t = 0.1)]
    epsilon: f64,
    /// Probability with which the estimate may miss that target.
    #[arg(short, long, default_value_t = 0.05)]
    delta: f64,
    /// Show the run in a live terminal dashboard.
    #[arg(long)]
    tui: bool,
    /// Learn the schedule and print the plan (temperatures, population,
    /// cost) without running the estimation phase.
    #[arg(long)]
    plan_only: bool,
    /// Also compute the exact permanent (n <= 26) and compare.
    #[arg(long)]
    exact: bool,
    /// Segments the temperature ladder is cut into. Each starts from fresh
    /// sampler draws and runs independently, so segments run concurrently
    /// across MPI ranks. Default: the number of ranks, at least 4.
    #[arg(long)]
    segments: Option<usize>,
    /// Start segments with the sampler run for the paper's explicit time
    /// per stage. Much slower starts; without it the boundary check between
    /// segments covers the sampler's run length.
    #[arg(long)]
    rigorous_sampler: bool,
    /// Use only the paper's guaranteed temperature steps instead of the
    /// adaptive schedule. Many more temperatures; no step relies on pilot
    /// statistics.
    #[arg(long)]
    rigorous_schedule: bool,
    /// Seed; drawn from system entropy and printed when omitted.
    #[arg(long)]
    seed: Option<u64>,
    /// Worker threads (default: all).
    #[arg(short, long)]
    threads: Option<usize>,
    /// Type of the waiting times in the chains. Every probability decision
    /// is an integer comparison whatever this is. Default: f64 on rayon,
    /// f32 on cuda.
    #[arg(long, value_enum)]
    precision: Option<Precision>,
    /// First GPU ordinal of the cuda backend.
    #[cfg(feature = "cuda")]
    #[arg(long, default_value_t = 0)]
    device: usize,
    /// Log more: -v logs every temperature and round. The level can also be
    /// set with PERMANENT_LOG_LEVEL (e.g. `warn`, `permanent=trace`).
    #[arg(short, long, action = ArgAction::Count)]
    verbose: u8,
    /// Write the log to this file instead of stderr. With --tui the log is
    /// written only when this is given.
    #[arg(long)]
    log_file: Option<std::path::PathBuf>,
    /// Write what Phase 1 fixed (temperatures, anchor weights) and the
    /// scaled matrix to this JSON file.
    #[arg(long)]
    save_schedule: Option<std::path::PathBuf>,
    /// Expert: time a mutation spends in perfect matchings.
    #[arg(long, default_value_t = 1.5, hide = true)]
    mutation_time: f64,
    /// Expert: time of the mutation Phase 1 applies at each temperature.
    #[arg(long, default_value_t = 0.0, hide = true)]
    learn_mutation_time: f64,
    /// Expert: particles per island.
    #[arg(long, hide = true)]
    particles: Option<usize>,
    /// Expert: islands to start with.
    #[arg(long, hide = true)]
    islands: Option<usize>,
    /// Expert: particles of the Phase 1 population.
    #[arg(long, hide = true)]
    learn_particles: Option<usize>,
    /// Expert: activity-schedule density constant K of the sampler.
    #[arg(long, default_value_t = 32.0, hide = true)]
    density: f64,
}

/// The report goes to stdout, or to the dashboard when one is running.
/// Under MPI only rank 0 reports.
struct Reporter {
    dashboard: Option<Sender<TuiEvent>>,
    quiet: bool,
}

impl Reporter {
    fn say(&self, line: String) {
        match &self.dashboard {
            Some(sender) => drop(sender.send(TuiEvent::Line(line))),
            None if !self.quiet => println!("{line}"),
            None => {}
        }
    }

    fn send(&self, event: impl FnOnce() -> TuiEvent) {
        if let Some(sender) = &self.dashboard {
            let _ = sender.send(event());
        }
    }
}

macro_rules! say {
    ($reporter:expr, $($arg:tt)*) => { $reporter.say(format!($($arg)*)) };
}

fn init_logging(cli: &Cli, root: bool) -> anyhow::Result<()> {
    let default = match cli.verbose {
        _ if !root => LevelFilter::WARN,
        0 => LevelFilter::INFO,
        1 => LevelFilter::DEBUG,
        _ => LevelFilter::TRACE,
    };
    let filter = EnvFilter::builder()
        .with_default_directive(default.into())
        .with_env_var("PERMANENT_LOG_LEVEL")
        .from_env_lossy();
    let builder = tracing_subscriber::fmt().with_env_filter(filter);
    match &cli.log_file {
        Some(path) if root => {
            let file = std::fs::File::create(path)
                .map_err(|error| anyhow::anyhow!("creating {}: {error}", path.display()))?;
            builder.with_ansi(false).with_writer(Mutex::new(file)).init();
        }
        // the dashboard owns the terminal
        None if cli.tui => {}
        _ => builder.with_writer(std::io::stderr).init(),
    }
    Ok(())
}

/// Minimum spacing of the progress lines in the log.
const LOG_EVERY: Duration = Duration::from_secs(2);

fn run<C: Comm>(cli: &Cli, problem: &Problem, comm: &C, reporter: &Reporter) -> anyhow::Result<()> {
    let n = problem.n;
    say!(
        reporter,
        "matrix: {n} x {n}, {} nonzero entries{}",
        problem.a.iter().filter(|&&x| x > 0.0).count(),
        if problem.is_zero_one() { " (0/1)" } else { " (weighted)" }
    );
    let exact = if cli.exact && comm.rank() == 0 {
        let value = exact::ln_permanent(problem);
        if value.is_none() {
            say!(reporter, "exact: skipped, n > {}", exact::MAX_EXACT_N);
        }
        value
    } else {
        None
    };
    let scaled = match prepare(problem) {
        Prepared::Zero => {
            say!(reporter, "permanent = 0 exactly (the support has no perfect matching)");
            return Ok(());
        }
        Prepared::Scaled(scaled) => scaled,
    };
    if n == 1 {
        say!(reporter, "permanent = {} exactly", problem.a[0]);
        return Ok(());
    }
    // every rank must use the same seed
    let seed = cli.seed.unwrap_or_else(|| comm.gather_f64(&[rand::random::<u32>() as f64])[0] as u64);
    say!(reporter, "target: relative error {} with probability {}; seed {seed}", cli.epsilon, 1.0 - cli.delta);
    let config = Config {
        epsilon: cli.epsilon,
        delta: cli.delta,
        seed,
        rigorous_schedule: cli.rigorous_schedule,
        rigorous_sampler: cli.rigorous_sampler,
        density: cli.density,
        mutation_time: cli.mutation_time,
        learn_mutation_time: cli.learn_mutation_time,
        learn_particles: cli.learn_particles,
        particles: cli.particles,
        islands: cli.islands,
        segments: cli.segments,
    };
    // one engine per rank
    match cli.backend {
        #[cfg(feature = "rayon")]
        Backend::Rayon => {
            // the default generator: see `Rayon`
            type Rayon<'a, T> = permanent_rayon::Rayon<'a, T, rand::rngs::SmallRng>;
            match cli.precision.unwrap_or(Precision::F64) {
                Precision::F32 => estimate(cli, &scaled, &config, exact, Rayon::<f32>::new(&scaled), comm, reporter),
                Precision::F64 => estimate(cli, &scaled, &config, exact, Rayon::<f64>::new(&scaled), comm, reporter),
            }
        }
        #[cfg(feature = "cuda")]
        Backend::Cuda => {
            use permanent_cuda::Cuda;
            let devices = permanent_cuda::device_count()?;
            anyhow::ensure!(devices > 0, "no CUDA device");
            // one GPU per rank, wrapping around when there are fewer GPUs
            let device = (cli.device + comm.rank()) % devices;
            match cli.precision.unwrap_or(Precision::F32) {
                Precision::F32 => estimate(cli, &scaled, &config, exact, Cuda::<f32>::new(&scaled, device)?, comm, reporter),
                Precision::F64 => estimate(cli, &scaled, &config, exact, Cuda::<f64>::new(&scaled, device)?, comm, reporter),
            }
        }
    }
}

/// Run both phases on `engine` and report.
fn estimate<E: Engine, C: Comm>(
    cli: &Cli,
    scaled: &Scaled,
    config: &Config,
    exact: Option<f64>,
    engine: E,
    comm: &C,
    reporter: &Reporter,
) -> anyhow::Result<()> {
    let n = scaled.n;
    let seed = config.seed;
    let mut session = Session::new(scaled, config, engine, comm)?;
    let engine = session.engine_name();
    let ranks = comm.size();
    info!(n, seed, epsilon = cli.epsilon, delta = cli.delta, %engine, ranks, "starting");
    reporter.send(|| TuiEvent::Init { n, engine: engine.clone(), exact });

    let mut last_log = Instant::now();
    let mut due = move || {
        let due = last_log.elapsed() >= LOG_EVERY;
        if due {
            last_log = Instant::now();
        }
        due
    };
    session.observe(move |progress| match progress {
        Progress::Level { index, level } => {
            if due() {
                info!(temperatures = index + 1, beta = level.beta, "phase 1");
            }
            reporter.send(|| TuiEvent::Level {
                index,
                beta: level.beta,
                relvar: level.relvar,
                log_w: level.log_w.clone(),
            });
        }
        Progress::Sweep { round, done, total } => {
            if due() {
                info!(round = round + 1, done, total, "phase 2 temperature sweeps");
            }
            reporter.send(|| TuiEvent::Sweep { round, done, total });
        }
        Progress::Round { index, estimate: e } => {
            info!(
                round = index + 1,
                islands = e.islands,
                ln_permanent = e.ln_permanent,
                std_error = e.std_error,
                events = e.mutate.events + e.sampler.events,
                seconds = e.seconds,
                "phase 2 round finished"
            );
            say!(
                reporter,
                "  {:>6} islands   ln(per) = {:.5} ± {:.5}   {:.3e} mutation + {:.3e} sampler events, {:.1} s",
                e.islands,
                e.ln_permanent,
                e.std_error,
                e.mutate.events as f64,
                e.sampler.events as f64,
                e.seconds,
            );
            reporter.send(|| TuiEvent::Round {
                islands: e.islands,
                ln_permanent: e.ln_permanent,
                std_error: e.std_error,
            });
        }
    });

    let levels = session.learn_schedule()?;
    let d = session.diagnostics.clone();
    debug!(?d, "phase 1 diagnostics");
    let t = cli.mutation_time;
    say!(reporter, "phase 1 (learn weights, build schedule): {:.1} s", d.phase1_seconds);
    say!(reporter, "  temperatures:                {}", levels.len());
    say!(reporter, "  population:                  {} particles on {engine}", d.learn_particles);
    say!(
        reporter,
        "  work:                        {} learning passes, {:.3e} mutation events, {:.3e} FAR steps",
        d.learn_iterations,
        d.mutate.events as f64,
        d.far_steps as f64
    );
    // Phase 1 mutates only when --learn-mutation-time asks for it
    if d.mutations > 0 {
        say!(
            reporter,
            "  mutation (t = {}):         {:.0} events each, {:.3} of the time in perfect matchings",
            cli.learn_mutation_time,
            d.mutate.events as f64 / d.mutations as f64,
            cli.learn_mutation_time * d.mutations as f64 / d.mutate.time
        );
    }
    say!(reporter, "  worst first-pass correction: x{:.3}", d.worst_first_correction.exp());
    say!(reporter, "  weight check (should be ~1): {:.3} to {:.3}", d.check_min, d.check_max);
    if d.learn_capped_chains + d.mutate.over_budget > 0 {
        say!(
            reporter,
            "  WARNING: {} capped FAR chains, {} mutations over budget",
            d.learn_capped_chains,
            d.mutate.over_budget
        );
    }
    if let Some(path) = cli.save_schedule.as_ref().filter(|_| comm.rank() == 0) {
        let schedule = serde_json::json!({
            "n": n,
            "mutation_time": t,
            "energy": scaled.energy,
            "log_gap": scaled.log_gap.iter().map(|g| g.max(f64::MIN)).collect::<Vec<_>>(),
            "log_scale": scaled.log_scale,
            "levels": levels,
        });
        std::fs::write(path, serde_json::to_string(&schedule)?)
            .map_err(|error| anyhow::anyhow!("writing {}: {error}", path.display()))?;
    }
    let plan = session.plan(&levels);
    say!(reporter, "plan:");
    say!(
        reporter,
        "  segments:                    {} (independent; each starts from fresh sampler draws)",
        plan.segments.len()
    );
    say!(
        reporter,
        "  islands x particles:         {} x {} per segment",
        plan.islands,
        plan.particles
    );
    say!(
        reporter,
        "  variance per island:         planned {:.4} (sum of step variances {:.3}, margin 2); bound factor {:.2}",
        plan.island_relvar,
        plan.relvar_sum,
        plan.mixing_factor
    );
    say!(
        reporter,
        "  target std error of ln(per): {:.4} (interval z = {:.3})",
        session.sigma_target,
        session.z
    );
    say!(reporter, "  sampler stage time:          {:.4}", plan.stage_time);
    if plan.predicted_events > 0.0 {
        say!(reporter, "  predicted:                   ~{:.3e} mutation events", plan.predicted_events);
    }
    if cli.plan_only {
        return Ok(());
    }
    say!(reporter, "phase 2 (estimate, {ranks} rank{}):", if ranks == 1 { "" } else { "s" });
    let estimate = session.estimate(&levels, &plan)?;
    say!(
        reporter,
        "  measured variance per island: {:.4} with {} particles (planned {:.4})",
        estimate.island_relvar,
        estimate.particles,
        plan.island_relvar * plan.particles as f64 / estimate.particles as f64
    );
    match estimate.verdict {
        Verdict::Consistent(gap) => say!(
            reporter,
            "  boundary check: carried and fresh populations agree (largest difference {gap:+.2} standard errors)"
        ),
        Verdict::Inconsistent(gap) => say!(
            reporter,
            "  WARNING: carried and fresh populations differ by {gap:+.2} standard errors at a segment \
             boundary; raise --mutation-time or use --rigorous-sampler"
        ),
        Verdict::Unchecked => say!(reporter, "  boundary check: none (one segment)"),
    }
    if estimate.mutate.over_budget + estimate.sampler.over_budget > 0 {
        say!(
            reporter,
            "  WARNING: {} mutations and {} sampler trials hit their event cap",
            estimate.mutate.over_budget,
            estimate.sampler.over_budget
        );
    }
    say!(reporter, "  regularization factor E[g]:  {:.4}", estimate.mean_gap);
    let half = session.z * estimate.std_error;
    say!(reporter, "permanent ≈ {}", scientific(estimate.ln_permanent));
    say!(reporter, "  ln(permanent) = {:.6} ± {:.6} (std error)", estimate.ln_permanent, estimate.std_error);
    say!(
        reporter,
        "  {:.1}% statistical interval: [{}, {}]  (relative half-width {:.2}%)",
        100.0 * (1.0 - cli.delta / 2.0), // half of delta is statistical
        scientific(estimate.ln_permanent - half),
        scientific(estimate.ln_permanent + half),
        100.0 * (half.exp() - 1.0)
    );
    if !estimate.met_target {
        say!(reporter, "  WARNING: the measured error did not reach the target within the top-up limit");
    }
    if let Some(exact) = exact {
        say!(
            reporter,
            "exact = {}; relative error {:+.3}% ({:+.2} std errors)",
            scientific(exact),
            100.0 * ((estimate.ln_permanent - exact).exp() - 1.0),
            (estimate.ln_permanent - exact) / estimate.std_error
        );
    }
    info!(ln_permanent = estimate.ln_permanent, std_error = estimate.std_error, "finished");
    Ok(())
}

/// Parse the command line and run.
pub fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    #[cfg(feature = "mpi")]
    let comm = permanent_core::comm::Mpi::new().ok_or_else(|| anyhow::anyhow!("MPI could not be initialized"))?;
    #[cfg(not(feature = "mpi"))]
    let comm = Solo;
    let root = comm.rank() == 0;
    init_logging(&cli, root)?;
    if let Some(threads) = cli.threads {
        rayon::ThreadPoolBuilder::new().num_threads(threads).build_global()?;
    }
    let problem = Problem::load(&cli.input)?;
    if !cli.tui || !root {
        return run(&cli, &problem, &comm, &Reporter { dashboard: None, quiet: !root });
    }
    anyhow::ensure!(comm.size() == 1, "--tui runs in a single process; drop it under mpirun");
    let (sender, receiver) = channel();
    // Detached on purpose: quitting the dashboard ends the process without
    // waiting for the run.
    std::thread::spawn(move || {
        let reporter = Reporter { dashboard: Some(sender), quiet: false };
        let event = match run(&cli, &problem, &Solo, &reporter) {
            Ok(()) => TuiEvent::Done,
            Err(error) => TuiEvent::Failed(format!("{error:#}")),
        };
        reporter.send(|| event);
    });
    let outcome = tui::run(receiver)?;
    for line in &outcome.lines {
        println!("{line}");
    }
    match outcome.end {
        tui::End::Done => Ok(()),
        tui::End::Failed(message) => anyhow::bail!(message),
        tui::End::Quit => {
            println!("stopped before the run completed");
            Ok(())
        }
    }
}
