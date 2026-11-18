use clap::Parser;
use permanent::estimator::{Config, Session, Verdict};
use permanent::problem::Problem;
use permanent::scaling::{Prepared, prepare};
use permanent::exact;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

/// Estimate the permanent of a nonnegative matrix to a requested accuracy.
///
/// Give the matrix and the accuracy; the number of temperatures, the samples
/// at each, and the sampler's run lengths are derived, not supplied.
#[derive(Parser, Debug)]
#[command(version)]
struct Cli {
    /// JSON input: {"matrix": [[..],..]}, {"size": n, "entries": [[i,j,w],..]},
    /// or the 0/1 form {"size": n, "edges": [[cols of row 0],..]}.
    input: std::path::PathBuf,
    /// Target relative error of the estimate.
    #[arg(short, long, default_value_t = 0.1)]
    epsilon: f64,
    /// Probability with which the estimate may miss that target.
    #[arg(short, long, default_value_t = 0.05)]
    delta: f64,
    /// Learn the schedule and print the plan (temperatures, samples, cost)
    /// without running the estimation phase.
    #[arg(long)]
    plan_only: bool,
    /// Also compute the exact permanent (n <= 26) and compare.
    #[arg(long)]
    exact: bool,
    /// Run the sampler for the paper's explicit time per stage. Hundreds of
    /// times slower than the default, which instead doubles the time until
    /// two independent estimates agree.
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
    /// Print one line per temperature during Phase 1.
    #[arg(short, long)]
    verbose: bool,
    /// Expert: activity-schedule density constant K of the sampler.
    #[arg(long, default_value_t = 32.0, hide = true)]
    density: f64,
    /// Expert: force a sampler time per stage and skip validation.
    #[arg(long, hide = true)]
    stage_time: Option<f64>,
}

/// Format exp(ln_value) as mantissa x 10^exponent without overflow.
fn scientific(ln_value: f64) -> String {
    if ln_value == f64::NEG_INFINITY {
        return "0".into();
    }
    let log10 = ln_value / std::f64::consts::LN_10;
    let exponent = log10.floor();
    format!("{:.6}e{:+}", 10f64.powf(log10 - exponent), exponent as i64)
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    if let Some(threads) = cli.threads {
        rayon::ThreadPoolBuilder::new().num_threads(threads).build_global()?;
    }
    let problem = Problem::load(&cli.input)?;
    let n = problem.n;
    println!(
        "matrix: {n} x {n}, {} nonzero entries{}",
        problem.a.iter().filter(|&&x| x > 0.0).count(),
        if problem.is_zero_one() { " (0/1)" } else { " (weighted)" }
    );
    let exact = if cli.exact {
        let value = exact::ln_permanent(&problem);
        if value.is_none() {
            println!("exact: skipped, n > {}", exact::MAX_EXACT_N);
        }
        value
    } else {
        None
    };
    let scaled = match prepare(&problem) {
        Prepared::Zero => {
            println!("permanent = 0 exactly (the support has no perfect matching)");
            return Ok(());
        }
        Prepared::Scaled(scaled) => scaled,
    };
    if n == 1 {
        println!("permanent = {} exactly", problem.a[0]);
        return Ok(());
    }
    let seed = cli.seed.unwrap_or_else(rand::random);
    println!("target: relative error {} with probability {}; seed {seed}", cli.epsilon, 1.0 - cli.delta);
    let config = Config {
        epsilon: cli.epsilon,
        delta: cli.delta,
        seed,
        rigorous_schedule: cli.rigorous_schedule,
        rigorous_sampler: cli.rigorous_sampler,
        density: cli.density,
        stage_time: cli.stage_time,
        verbose: cli.verbose,
    };
    let mut session = Session::new(&scaled, &config)?;
    let levels = session.learn_schedule()?;
    let d = session.diagnostics.clone();
    println!(
        "phase 1 (learn weights, build schedule): {:.1} s",
        d.phase1_seconds
    );
    println!("  temperatures (rounds):       {}", levels.len());
    println!(
        "  sampler:                     {} activity stages, acceptance {:.3}",
        session.stages.stages(),
        d.phase1.accepted as f64 / d.phase1.trials.max(1) as f64
    );
    println!(
        "  work:                        {} learning passes, {:.3e} sampler events, {:.3e} FAR steps",
        d.learn_iterations, d.phase1.events as f64, d.phase1_far_steps as f64
    );
    println!("  worst first-pass correction: x{:.3}", d.worst_first_correction.exp());
    println!("  weight check (should be ~1): {:.3} to {:.3}", d.check_min, d.check_max);
    if d.learn_capped_chains + d.learn_failed_starts + d.phase1.over_budget > 0 {
        println!(
            "  WARNING: {} capped FAR chains, {} failed starts, {} over-budget trials",
            d.learn_capped_chains, d.learn_failed_starts, d.phase1.over_budget
        );
    }
    let first = session.plan(&Session::pilot_relvar(&levels), session.learn_stage_time());
    let (min, max) = first.samples.iter().fold((u64::MAX, 0), |(lo, hi), &s| (lo.min(s), hi.max(s)));
    println!("plan:");
    println!("  samples per temperature:     {min} to {max} (total {})", first.total_samples);
    println!(
        "  target std error of ln(per): {:.4} (interval z = {:.3})",
        session.sigma_target, session.z
    );
    if cli.rigorous_sampler {
        let t = session.rigorous_stage_time(first.total_samples);
        let plan = session.plan(&Session::pilot_relvar(&levels), t);
        println!(
            "  one run at the paper's stage time {t:.1}: ~{:.3e} events, ~{:.0} s",
            plan.predicted_events, plan.predicted_seconds
        );
    } else if cli.stage_time.is_none() {
        println!(
            "  first two runs (stage time {:.4} and {:.4}): ~{:.3e} events, ~{:.0} s",
            first.stage_time,
            2.0 * first.stage_time,
            3.0 * first.predicted_events,
            3.0 * first.predicted_seconds
        );
    }
    if cli.plan_only {
        return Ok(());
    }
    println!("phase 2 (estimate):");
    let (attempts, verdict) = session.estimate_validated(&levels, |attempt| {
        let e = &attempt.estimate;
        println!(
            "  stage time {:<8.4} ln(per) = {:.5} ± {:.5}   {} samples, {:.3e} events, {:.1} s, acceptance {:.3}{}",
            attempt.plan.stage_time,
            e.ln_permanent,
            e.std_error,
            e.samples,
            e.cost.events as f64,
            e.seconds,
            e.cost.accepted as f64 / e.cost.trials.max(1) as f64,
            if e.failed_samples + e.cost.over_budget > 0 {
                format!("  WARNING: {} failed samples, {} over-budget trials", e.failed_samples, e.cost.over_budget)
            } else {
                String::new()
            }
        );
    });
    let estimate = &attempts.last().unwrap().estimate;
    match verdict {
        Verdict::Rigorous => println!("  sampler setting: the paper's explicit stage time"),
        Verdict::Validated(gap) => println!(
            "  sampler setting: validated; halving the stage time moves the estimate by {gap:+.2} standard errors"
        ),
        Verdict::Unchecked => println!("  sampler setting: forced by --stage-time, not checked"),
        Verdict::NotValidated => println!(
            "  WARNING: the estimates never stabilized under doubling; treat the result as unreliable"
        ),
    }
    println!("  regularization factor E[g]:  {:.4}", estimate.mean_gap);
    let half = session.z * estimate.std_error;
    println!("permanent ≈ {}", scientific(estimate.ln_permanent));
    println!("  ln(permanent) = {:.6} ± {:.6} (std error)", estimate.ln_permanent, estimate.std_error);
    println!(
        "  {:.1}% statistical interval: [{}, {}]  (relative half-width {:.2}%)",
        100.0 * (1.0 - cli.delta / 2.0), // half of delta is statistical
        scientific(estimate.ln_permanent - half),
        scientific(estimate.ln_permanent + half),
        100.0 * (half.exp() - 1.0)
    );
    if !estimate.met_target {
        println!("  WARNING: the measured error did not reach the target within the top-up limit");
    }
    if let Some(exact) = exact {
        println!(
            "exact = {}; relative error {:+.3}% ({:+.2} std errors)",
            scientific(exact),
            100.0 * ((estimate.ln_permanent - exact).exp() - 1.0),
            (estimate.ln_permanent - exact) / estimate.std_error
        );
    }
    Ok(())
}
