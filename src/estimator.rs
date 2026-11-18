//! The driver: from a matrix and a target (epsilon, delta) to an estimate
//! with an error bar. Nothing here is a tuning knob the caller has to guess;
//! the number of temperatures and the number of samples at each are derived.
//!
//! Phase 1 (learn and plan) walks beta from 0 to 1. At each temperature it
//! learns the n anchor weights with the FAR chain, then picks the next beta.
//! Phase 2 (estimate) draws independent perfect matchings at every
//! temperature and multiplies the ratio estimates:
//!
//!   per(A) = n! * prod_i E_{mu_i}[exp(-d_i H)] * E_{mu_L}[g] / c_sc.
//!
//! Because every Phase 2 sample is an independent draw, the variance of the
//! estimate is measured directly from the samples, and sampling continues
//! until the measured error meets the target.

use rand::SeedableRng;
use rand::rngs::SmallRng;
use rayon::prelude::*;
use std::time::Instant;

use crate::far;
use crate::sampler::{SampleCost, Scratch, StageSchedule, sample_perfect};
use crate::scaling::Scaled;
use crate::stats::{ln_factorial, log_sum_exp, normal_quantile, stream_seed};
use crate::tables::Tables;

#[derive(Debug, Clone)]
pub struct Config {
    /// Target relative error.
    pub epsilon: f64,
    /// Allowed failure probability.
    pub delta: f64,
    pub seed: u64,
    /// Use only the paper's non-adaptive temperature steps (Lemma 50).
    pub rigorous_schedule: bool,
    /// Run the sampler for the paper's explicit time per stage instead of
    /// the validated (doubling) setting.
    pub rigorous_sampler: bool,
    /// Activity-schedule density constant K (see `StageSchedule::new`).
    pub density: f64,
    /// Expert override: a fixed sampler time per stage, with no validation.
    pub stage_time: Option<f64>,
    pub verbose: bool,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            epsilon: 0.1,
            delta: 0.05,
            seed: 0,
            rigorous_schedule: false,
            rigorous_sampler: false,
            density: 32.0,
            stage_time: None,
            verbose: false,
        }
    }
}

/// One temperature of the schedule, as fixed by Phase 1.
#[derive(Debug, Clone)]
pub struct Level {
    pub beta: f64,
    pub log_w: Vec<f64>,
    /// Step to the next temperature (0 at the last one).
    pub step: f64,
    /// Energy offset used to keep exp(-step * (H - shift)) near 1.
    pub shift: f64,
    /// Pilot estimate of the relative variance of the ratio term.
    pub pilot_relvar: f64,
}

#[derive(Debug, Clone, Default)]
pub struct Diagnostics {
    pub learn_iterations: u64,
    pub learn_capped_chains: u64,
    pub learn_failed_starts: u64,
    /// Largest |ln| correction the first learning pass applied to a
    /// predicted weight; above ln 2 means a step outran the weights.
    pub worst_first_correction: f64,
    /// Range of the weight-normalization check, which is 1 in expectation.
    pub check_min: f64,
    pub check_max: f64,
    pub phase1: SampleCost,
    pub phase1_far_steps: u64,
    pub phase1_seconds: f64,
    /// CPU seconds Phase 1 spent inside the sampler, for cost prediction.
    pub phase1_sampler_cpu: f64,
}

#[derive(Debug, Clone)]
pub struct Plan {
    pub samples: Vec<u64>,
    pub total_samples: u64,
    pub stage_time: f64,
    pub predicted_events: f64,
    pub predicted_seconds: f64,
}

#[derive(Debug, Clone)]
pub struct Estimate {
    pub ln_permanent: f64,
    /// Standard error of ln_permanent measured from the samples.
    pub std_error: f64,
    pub met_target: bool,
    pub rounds: usize,
    pub samples: u64,
    pub mean_gap: f64,
    /// Measured relative variance of each temperature's term.
    pub relvar: Vec<f64>,
    pub cost: SampleCost,
    pub failed_samples: u64,
    pub seconds: f64,
}

#[derive(Debug, Clone)]
pub struct Attempt {
    pub plan: Plan,
    pub estimate: Estimate,
}

/// Why the reported estimate's sampler setting is trusted.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Verdict {
    /// The sampler ran for the paper's explicit time.
    Rigorous,
    /// The estimate agrees, within noise, with an independent one from a
    /// sampler run half as long; the value is their difference in standard
    /// errors of the difference.
    Validated(f64),
    /// A fixed stage time was forced; nothing was checked.
    Unchecked,
    /// Doubling hit its limit without two consecutive settings agreeing.
    NotValidated,
}

const PHASE_LEARN: u64 = 1;
const PHASE_ESTIMATE: u64 = 1000;
const LEARN_CHAINS: usize = 32;
const LEARN_MAX_ITERATIONS: usize = 8;
const LEARN_MAX_ROUNDS: usize = 7;
/// Target relative standard error of each anchor weight.
const LEARN_RSE: f64 = 0.15;
/// A learning pass whose corrections are all below this is accepted.
const LEARN_ACCEPT: f64 = 0.2624; // ln 1.3
/// Relative variance aimed for when choosing a temperature step.
const STEP_RELVAR: f64 = 0.25;
const MIN_SAMPLES: u64 = 32;
const MAX_TOPUP_ROUNDS: usize = 8;
/// Two settings agree when they differ by at most this many standard
/// errors of their difference.
const AGREEMENT: f64 = 2.0;
const MAX_DOUBLINGS: usize = 12;

struct Learned {
    log_w: Vec<f64>,
    pilot: Vec<f64>,
    /// E[H | N(r,v)] - E[H | P], the beta-derivative of ln w*(r,v).
    drift: Vec<f64>,
}

fn relvar_of_step(pilot: &[f64], step: f64) -> f64 {
    let min = pilot.iter().copied().fold(f64::INFINITY, f64::min);
    let (mut m1, mut m2) = (0.0, 0.0);
    for &h in pilot {
        let x = (-step * (h - min)).exp();
        m1 += x;
        m2 += x * x;
    }
    let k = pilot.len() as f64;
    (m2 / k) / ((m1 / k) * (m1 / k)) - 1.0
}

/// Largest step in (0, cap] whose pilot relative variance stays below target.
fn step_for_relvar(pilot: &[f64], cap: f64, target: f64) -> f64 {
    if relvar_of_step(pilot, cap) <= target {
        return cap;
    }
    let (mut lo, mut hi) = (0.0, cap);
    for _ in 0..60 {
        let mid = 0.5 * (lo + hi);
        if relvar_of_step(pilot, mid) <= target { lo = mid } else { hi = mid }
    }
    lo
}

pub struct Session<'a> {
    scaled: &'a Scaled,
    config: &'a Config,
    pub stages: StageSchedule,
    anchor: usize,
    pub diagnostics: Diagnostics,
    /// Target standard error of ln(estimate).
    pub sigma_target: f64,
    /// Normal quantile of the reported interval.
    pub z: f64,
}

impl<'a> Session<'a> {
    pub fn new(scaled: &'a Scaled, config: &'a Config) -> anyhow::Result<Self> {
        anyhow::ensure!(config.epsilon > 0.0 && config.epsilon < 1.0, "epsilon must be in (0, 1)");
        anyhow::ensure!(config.delta > 0.0 && config.delta < 1.0, "delta must be in (0, 1)");
        anyhow::ensure!(scaled.n >= 2, "n must be at least 2");
        // half of delta covers statistical error; the rest is left for the
        // sampler (its tolerance in rigorous mode, the check otherwise)
        let z = normal_quantile(1.0 - config.delta / 4.0);
        Ok(Session {
            scaled,
            config,
            stages: StageSchedule::new(scaled.n, config.density),
            anchor: 0,
            diagnostics: Diagnostics { check_min: f64::INFINITY, check_max: 0.0, ..Default::default() },
            sigma_target: (1.0 + config.epsilon).ln() / z,
            z,
        })
    }

    /// Sampler time per stage used while learning weights.
    pub fn learn_stage_time(&self) -> f64 {
        if let Some(t) = self.config.stage_time {
            t
        } else if self.config.rigorous_sampler {
            self.stages.rigorous_stage_time(1.0 / 16.0)
        } else {
            self.stages.initial_stage_time(self.scaled.n)
        }
    }

    /// The paper's time per stage for a run of `samples` draws: the total
    /// variation budget delta/2 is shared by all of them.
    pub fn rigorous_stage_time(&self, samples: u64) -> f64 {
        self.stages.rigorous_stage_time((self.config.delta / 2.0) / (2.0 * samples as f64))
    }

    /// Learn the anchor weights at `beta`, starting from the guess `log_w`.
    fn learn(&mut self, level: usize, beta: f64, mut log_w: Vec<f64>) -> anyhow::Result<Learned> {
        let scaled = self.scaled;
        let n = scaled.n;
        let anchor = self.anchor;
        let stage_time = self.learn_stage_time();
        let event_cap = self.stages.event_cap(n, stage_time);
        let mut pilot = Vec::new();
        let mut drift = vec![0.0; n];
        for iteration in 0..LEARN_MAX_ITERATIONS {
            self.diagnostics.learn_iterations += 1;
            let tables = Tables::build(scaled, beta, &log_w, anchor);
            let mut cycles = 2 * (n * n) as u64;
            // per-chain (cycles, visits) records for the error estimate
            let mut records: Vec<(u64, Vec<u64>)> = Vec::new();
            let mut visit_energy = vec![0.0; n];
            let mut perfect_energy = 0.0;
            let mut total_cycles = 0u64;
            let mut totals = vec![0u64; n];
            for round in 0..LEARN_MAX_ROUNDS {
                let stages = &self.stages;
                let seed = self.config.seed;
                let results: Vec<(SampleCost, f64, Option<far::FarStats>)> = (0..LEARN_CHAINS)
                    .into_par_iter()
                    .map(|chain| {
                        let index = ((iteration as u64) << 40) | ((round as u64) << 20) | chain as u64;
                        let mut rng = SmallRng::seed_from_u64(stream_seed(seed, PHASE_LEARN, level as u64, index));
                        let mut scratch = Scratch::new(n);
                        let mut cost = SampleCost::default();
                        let clock = Instant::now();
                        let ok = sample_perfect(&tables, stages, stage_time, event_cap, 48, &mut scratch, &mut rng, &mut cost);
                        let cpu = clock.elapsed().as_secs_f64();
                        if !ok {
                            return (cost, cpu, None);
                        }
                        let stats = far::run(
                            &tables,
                            &scaled.energy,
                            &scratch.sigma,
                            cycles,
                            256 * cycles + 100_000,
                            &mut rng,
                        );
                        (cost, cpu, Some(stats))
                    })
                    .collect();
                for (cost, cpu, stats) in results {
                    self.diagnostics.phase1.add(cost);
                    self.diagnostics.phase1_sampler_cpu += cpu;
                    let Some(stats) = stats else {
                        self.diagnostics.learn_failed_starts += 1;
                        continue;
                    };
                    self.diagnostics.phase1_far_steps += stats.steps;
                    if stats.capped {
                        self.diagnostics.learn_capped_chains += 1;
                    }
                    total_cycles += stats.perfect_steps;
                    perfect_energy += stats.perfect_energy;
                    for v in 0..n {
                        totals[v] += stats.visits[v];
                        visit_energy[v] += stats.visit_energy[v];
                    }
                    pilot.extend_from_slice(&stats.pilot);
                    records.push((stats.perfect_steps, stats.visits));
                }
                if records.len() < 8 {
                    anyhow::bail!(
                        "the sampler could not produce perfect matchings at beta = {beta:.6} \
                         (level {level}); the anchor weights are out of range"
                    );
                }
                // relative standard error of each class total, from the
                // spread across independent chains (ratio-estimator form)
                let chains = records.len() as f64;
                let mut worst: f64 = 0.0;
                for v in 0..n {
                    if totals[v] == 0 {
                        worst = f64::INFINITY;
                        break;
                    }
                    let rate = totals[v] as f64 / total_cycles as f64;
                    let ss: f64 = records
                        .iter()
                        .map(|(k, visits)| (visits[v] as f64 - rate * *k as f64).powi(2))
                        .sum();
                    let rse = (ss * chains / (chains - 1.0)).sqrt() / totals[v] as f64;
                    worst = worst.max(rse);
                }
                if worst <= LEARN_RSE {
                    break;
                }
                cycles *= 2;
            }
            // w~(v) ∝ w(v) W(r,v) / visits(v), then normalize with
            // sum_z lambda(r,z)/w(r,z) = 1, which the ideal weights satisfy.
            let mut updated = vec![0.0; n];
            let mut terms = vec![0.0; n];
            for v in 0..n {
                updated[v] = log_w[v] + tables.hole_weight(anchor, v).ln() - (totals[v] as f64).max(0.5).ln();
                terms[v] = -beta * scaled.energy[anchor * n + v] - updated[v];
            }
            let norm = log_sum_exp(&terms);
            let mut worst_correction: f64 = 0.0;
            for v in 0..n {
                let previous = log_w[v] + tables.log_anchor_norm;
                let correction = (updated[v] + norm - previous).clamp(-8f64.ln(), 8f64.ln());
                worst_correction = worst_correction.max(correction.abs());
                log_w[v] = previous + correction;
            }
            // Unnormalized check: (n^2 / k) sum_z B(r,z) visits_z / W(r,z) = 1.
            let check: f64 = (0..n)
                .map(|z| tables.b[anchor * n + z] * totals[z] as f64 / tables.hole_weight(anchor, z))
                .sum::<f64>()
                * (n * n) as f64
                / total_cycles as f64;
            self.diagnostics.check_min = self.diagnostics.check_min.min(check);
            self.diagnostics.check_max = self.diagnostics.check_max.max(check);
            if iteration == 0 {
                self.diagnostics.worst_first_correction =
                    self.diagnostics.worst_first_correction.max(worst_correction);
            }
            let mean_perfect = perfect_energy / total_cycles as f64;
            for v in 0..n {
                drift[v] = if totals[v] > 0 { visit_energy[v] / totals[v] as f64 - mean_perfect } else { 0.0 };
            }
            if worst_correction <= LEARN_ACCEPT {
                break;
            }
        }
        Ok(Learned { log_w, pilot, drift })
    }

    /// Phase 1: build the temperature schedule and learn weights along it.
    pub fn learn_schedule(&mut self) -> anyhow::Result<Vec<Level>> {
        let scaled = self.scaled;
        let n = scaled.n;
        let h_max = scaled.h_max();
        let spread = 2.0 * scaled.log_inv_tau;
        let mut levels: Vec<Level> = Vec::new();
        let mut beta = 0.0f64;
        // at beta = 0 every activity is 1, so w*(r,v) = n!/(n-1)! = n exactly
        let mut guess = vec![(n as f64).ln(); n];
        let mut previous_step: Option<f64> = None;
        let start = Instant::now();
        loop {
            let learned = self.learn(levels.len(), beta, guess)?;
            let shift = learned.pilot.iter().sum::<f64>() / learned.pilot.len() as f64;
            if beta >= 1.0 {
                levels.push(Level { beta, log_w: learned.log_w, step: 0.0, shift, pilot_relvar: 0.0 });
                break;
            }
            let remaining = 1.0 - beta;
            // the paper's guaranteed step (Lemma 50)
            let floor = 0.25 * (1.0 / h_max).max(beta / spread);
            let step = if self.config.rigorous_schedule {
                floor
            } else {
                let by_variance = step_for_relvar(&learned.pilot, remaining, STEP_RELVAR);
                let (lo, hi) = learned
                    .drift
                    .iter()
                    .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &x| (lo.min(x), hi.max(x)));
                let by_drift = if hi > lo { 4f64.ln() / (hi - lo) } else { f64::INFINITY };
                let by_growth = previous_step.map_or(f64::INFINITY, |s| 4.0 * s);
                floor.max(by_variance.min(by_drift).min(by_growth))
            }
            .min(remaining);
            // avoid a sliver of a last step
            let step = if remaining - step < 0.25 * step { remaining } else { step };
            let pilot_relvar = relvar_of_step(&learned.pilot, step).max(1e-9);
            if self.config.verbose {
                eprintln!(
                    "level {:>5}  beta {:.6e}  step {:.3e}  relvar {:.3}  pilot {}",
                    levels.len(),
                    beta,
                    step,
                    pilot_relvar,
                    learned.pilot.len()
                );
            }
            // first-order prediction of the next weights
            guess = (0..n).map(|v| learned.log_w[v] + step * learned.drift[v]).collect();
            levels.push(Level { beta, log_w: learned.log_w, step, shift, pilot_relvar });
            previous_step = Some(step);
            beta = if step >= remaining { 1.0 } else { beta + step };
        }
        self.diagnostics.phase1_seconds = start.elapsed().as_secs_f64();
        Ok(levels)
    }

    /// Sample counts for one estimation run. `relvar` is the per-temperature
    /// relative variance to plan with: the pilot values, or those measured
    /// by an earlier run.
    pub fn plan(&self, relvar: &[f64], stage_time: f64) -> Plan {
        let bounded: Vec<f64> = relvar.iter().map(|v| v.clamp(1e-9, 1e6)).collect();
        let samples = allocate(&bounded, self.sigma_target, 1.1);
        let total_samples: u64 = samples.iter().sum();
        // extrapolate from Phase 1: events scale with the time per stage
        let d = &self.diagnostics;
        let scale = stage_time / self.learn_stage_time();
        let events_per_sample = d.phase1.events as f64 / d.phase1.accepted.max(1) as f64 * scale;
        let predicted_events = total_samples as f64 * events_per_sample;
        let cpu_per_event = d.phase1_sampler_cpu / d.phase1.events.max(1) as f64;
        let predicted_seconds = predicted_events * cpu_per_event / rayon::current_num_threads() as f64;
        Plan { samples, total_samples, stage_time, predicted_events, predicted_seconds }
    }

    /// Pilot relative variances from Phase 1. The last term is g, which lies
    /// in [0,1] with mean >= 2/3, so its relative variance is at most 1/2.
    pub fn pilot_relvar(levels: &[Level]) -> Vec<f64> {
        let last = levels.len() - 1;
        levels.iter().enumerate().map(|(i, l)| if i == last { 0.5 } else { l.pilot_relvar }).collect()
    }

    /// One estimation run: draw the planned samples, then top up until the
    /// measured standard error meets the target. `attempt` selects an
    /// independent random stream.
    pub fn estimate(&self, levels: &[Level], plan: &Plan, attempt: u64) -> Estimate {
        let start = Instant::now();
        let scaled = self.scaled;
        let n = scaled.n;
        let last = levels.len() - 1;
        let stage_time = plan.stage_time;
        let event_cap = self.stages.event_cap(n, stage_time);
        let max_trials = 96;
        let mut count = vec![0u64; levels.len()];
        let mut drawn = vec![0u64; levels.len()];
        let mut sum = vec![0.0f64; levels.len()];
        let mut sum_sq = vec![0.0f64; levels.len()];
        let mut want = plan.samples.clone();
        let sigma2 = self.sigma_target * self.sigma_target;
        let mut rounds = 0;
        let mut variance = f64::INFINITY;
        let mut relvar = vec![0.0; levels.len()];
        let mut total_cost = SampleCost::default();
        let mut failed_samples = 0u64;
        for round in 0..MAX_TOPUP_ROUNDS {
            rounds = round + 1;
            for (i, level) in levels.iter().enumerate() {
                if want[i] <= drawn[i] {
                    continue;
                }
                let tables = Tables::build(scaled, level.beta, &level.log_w, self.anchor);
                let stages = &self.stages;
                let seed = self.config.seed;
                let (c, s, ss, cost, failed) = (drawn[i]..want[i])
                    .into_par_iter()
                    .map_init(
                        || Scratch::new(n),
                        |scratch, index| {
                            let mut rng = SmallRng::seed_from_u64(stream_seed(
                                seed,
                                PHASE_ESTIMATE + attempt,
                                i as u64,
                                index,
                            ));
                            let mut cost = SampleCost::default();
                            if !sample_perfect(&tables, stages, stage_time, event_cap, max_trials, scratch, &mut rng, &mut cost) {
                                return (0u64, 0.0, 0.0, cost, 1u64);
                            }
                            let x = if i == last {
                                scaled.log_gap_of(&scratch.sigma).exp()
                            } else {
                                (-level.step * (scaled.energy_of(&scratch.sigma) - level.shift)).exp()
                            };
                            (1, x, x * x, cost, 0)
                        },
                    )
                    .reduce(
                        || (0, 0.0, 0.0, SampleCost::default(), 0),
                        |a, b| {
                            let mut cost = a.3;
                            cost.add(b.3);
                            (a.0 + b.0, a.1 + b.1, a.2 + b.2, cost, a.4 + b.4)
                        },
                    );
                count[i] += c;
                sum[i] += s;
                sum_sq[i] += ss;
                drawn[i] = want[i];
                total_cost.add(cost);
                failed_samples += failed;
            }
            variance = 0.0;
            for i in 0..levels.len() {
                let k = count[i] as f64;
                let mean = sum[i] / k;
                let var = ((sum_sq[i] / k - mean * mean) * k / (k - 1.0)).max(0.0);
                relvar[i] = if mean > 0.0 { var / (mean * mean) } else { f64::INFINITY };
                variance += relvar[i] / k;
            }
            if variance <= sigma2 * 1.02 {
                break;
            }
            let bounded: Vec<f64> = relvar.iter().map(|v| v.clamp(1e-9, 1e6)).collect();
            let fresh = allocate(&bounded, self.sigma_target, 1.05);
            let mut grew = false;
            for i in 0..levels.len() {
                if fresh[i] > want[i] {
                    want[i] = fresh[i];
                    grew = true;
                }
            }
            if !grew {
                // sampling noise in the variance estimate: add 10% everywhere
                for w in want.iter_mut() {
                    *w += (*w / 10).max(8);
                }
            }
        }
        let mut ln_ratio = 0.0;
        for (i, level) in levels.iter().enumerate().take(last) {
            ln_ratio += -level.step * level.shift + (sum[i] / count[i] as f64).ln();
        }
        let mean_gap = sum[last] / count[last] as f64;
        let ln_permanent = ln_factorial(n) + ln_ratio + mean_gap.ln() - scaled.log_scale;
        Estimate {
            ln_permanent,
            std_error: variance.sqrt(),
            met_target: variance <= sigma2 * 1.02,
            rounds,
            samples: count.iter().sum(),
            mean_gap,
            relvar,
            cost: total_cost,
            failed_samples,
            seconds: start.elapsed().as_secs_f64(),
        }
    }

    /// Phase 2 with the sampler setting chosen automatically.
    ///
    /// Rigorous mode runs once at the paper's stage time. Otherwise the run
    /// is repeated with the stage time doubled until two consecutive,
    /// independent estimates agree within noise; the longer of the agreeing
    /// pair is the answer. `observe` sees every attempt as it completes.
    pub fn estimate_validated(
        &self,
        levels: &[Level],
        mut observe: impl FnMut(&Attempt),
    ) -> (Vec<Attempt>, Verdict) {
        let mut relvar = Self::pilot_relvar(levels);
        let mut attempts: Vec<Attempt> = Vec::new();
        if self.config.rigorous_sampler || self.config.stage_time.is_some() {
            let total = self.plan(&relvar, 1.0).total_samples;
            let stage_time = self.config.stage_time.unwrap_or_else(|| self.rigorous_stage_time(total));
            let plan = self.plan(&relvar, stage_time);
            let estimate = self.estimate(levels, &plan, 0);
            attempts.push(Attempt { plan, estimate });
            observe(&attempts[0]);
            let verdict = if self.config.stage_time.is_some() { Verdict::Unchecked } else { Verdict::Rigorous };
            return (attempts, verdict);
        }
        let mut stage_time = self.learn_stage_time();
        for attempt in 0..MAX_DOUBLINGS {
            let total = self.plan(&relvar, stage_time).total_samples;
            let ceiling = self.rigorous_stage_time(total);
            let at_ceiling = stage_time >= ceiling;
            let plan = self.plan(&relvar, stage_time.min(ceiling));
            let estimate = self.estimate(levels, &plan, attempt as u64);
            relvar = estimate.relvar.clone();
            attempts.push(Attempt { plan, estimate });
            observe(attempts.last().unwrap());
            if at_ceiling {
                return (attempts, Verdict::Rigorous);
            }
            if attempts.len() >= 2 {
                let a = &attempts[attempts.len() - 2].estimate;
                let b = &attempts[attempts.len() - 1].estimate;
                let gap = (b.ln_permanent - a.ln_permanent)
                    / (a.std_error * a.std_error + b.std_error * b.std_error).sqrt();
                if gap.abs() <= AGREEMENT {
                    return (attempts, Verdict::Validated(gap));
                }
            }
            stage_time *= 2.0;
        }
        (attempts, Verdict::NotValidated)
    }
}

/// Sample counts minimizing the total subject to sum relvar_i / N_i equal to
/// sigma^2: N_i proportional to sqrt(relvar_i).
fn allocate(relvar: &[f64], sigma: f64, safety: f64) -> Vec<u64> {
    let total: f64 = relvar.iter().map(|v| v.sqrt()).sum();
    relvar
        .iter()
        .map(|v| ((safety * v.sqrt() * total / (sigma * sigma)).ceil() as u64).max(MIN_SAMPLES))
        .collect()
}
