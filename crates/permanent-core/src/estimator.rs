//! The driver: from a matrix and a target (epsilon, delta) to an estimate
//! with an error bar.
//!
//! With mu_i ∝ exp(-beta_i H) on the perfect matchings and
//! 0 = beta_0 < ... < beta_L = 1,
//!
//!   per(A) = n! * prod_i E_{mu_i}[exp(-d_i H)] * E_{mu_L}[g] / c_sc.
//!
//! The paper (arXiv:2609.20717) draws every sample of every mu_i from
//! nothing. Here a population of perfect matchings is carried along the
//! temperatures instead, and the paper's sampler is what lets that
//! population be cut into independent pieces.
//!
//! Population. P particles at mu_i are weighted by G_i = exp(-d_i H),
//! resampled in proportion, and moved by the mutation kernel K_t of
//! `sampler.rs`, which leaves mu_{i+1} invariant for any anchor weights and
//! contracts L2(mu_{i+1}) by exp(-t). The product of the mean weights is an
//! unbiased estimate of Z_j / Z_i (Del Moral 2004, Section 7.4), for every P
//! and every t, provided the first particles are exact samples and the
//! schedule and weights were fixed beforehand. Its relative variance is
//! asymptotically v / P with
//!
//!   v <= (1 - sqrt(g) exp(-t))^-2 * sum_i relvar_i,
//!
//! where relvar_i is the relative variance of G_i under mu_i and g bounds
//! G_i / E[G_i]. So the error falls as 1/sqrt(P) at a fixed number of
//! sweeps: a wider machine buys accuracy, not just speed. Measured, v has
//! stayed within 1.2 of sum_i relvar_i, far below the bound, so P is sized
//! from that sum and corrected by what the first islands measure.
//!
//! Segments. The ladder is cut into S segments. A segment that starts at
//! beta = 0 takes uniform permutations; any other takes fresh draws from
//! the paper's annealing sampler, which needs only the n anchor weights of
//! its first temperature. Segments share nothing, so their estimates are
//! independent and multiply without bias, and they run at the same time: on
//! different ranks, and within a rank as many as its engine moves at once. One sweep is sequential in temperature; S segments make
//! it S times shorter at the price of S P sampler calls.
//!
//! Islands. Each segment is run R times independently. The R values are
//! i.i.d. and unbiased, so the error bar is their measured spread. R
//! starts small and grows until the measured error meets the target.
//!
//! Check. At a segment boundary the particles carried there and the fresh
//! draws of the next segment target the same distribution. Their mean
//! weights are compared, the carried side as an unbiased ratio over
//! islands; a disagreement means the sampler's run length is too short.
//!
//! Phase 1 fixes the schedule and learns the anchor weights with one
//! population of its own: every particle runs a few FAR cycles
//! (`chain/far.rs`), which also move it.
//!
//! misc/theory.md derives all of this.

use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::{RngExt, SeedableRng};
use rayon::prelude::*;
use std::ops::Range;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc;
use std::thread;
use std::time::Instant;
use tracing::{debug, warn};

use crate::comm::Comm;
use crate::engine::{Engine, Stream};
use crate::sampler::{MutateCost, SampleCost, StageSchedule};
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
    /// Start segments with the sampler run for the paper's explicit time
    /// per stage instead of the default, which the boundary check covers.
    pub rigorous_sampler: bool,
    /// Activity-schedule density constant K (see `StageSchedule::new`).
    pub density: f64,
    /// Time t a mutation spends in perfect matchings.
    pub mutation_time: f64,
    /// Time of the mutation Phase 1 applies when it enters a temperature;
    /// 0 leaves the moving to the FAR runs.
    pub learn_mutation_time: f64,
    /// Particles of the Phase 1 population, over all ranks; the engine's
    /// preference per rank when absent.
    pub learn_particles: Option<usize>,
    /// Particles per island; derived when absent.
    pub particles: Option<usize>,
    /// Islands to start with; derived when absent.
    pub islands: Option<usize>,
    /// Segments the ladder is cut into; max(ranks, 4) when absent.
    pub segments: Option<usize>,
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
            mutation_time: 1.5,
            learn_mutation_time: 0.0,
            learn_particles: None,
            particles: None,
            islands: None,
            segments: None,
        }
    }
}

/// One temperature of the schedule, as fixed by Phase 1.
#[derive(Debug, Clone, serde::Serialize)]
pub struct Level {
    pub beta: f64,
    pub log_w: Vec<f64>,
    /// Step to the next temperature (0 at the last one).
    pub step: f64,
    /// Energy offset used to keep exp(-step * (H - shift)) near 1.
    pub shift: f64,
    /// Relative variance of the weight exp(-step * H), from the Phase 1
    /// population.
    pub relvar: f64,
    /// Largest weight over the mean weight in that population.
    pub weight_max: f64,
}

#[derive(Debug, Clone, Default)]
pub struct Diagnostics {
    pub learn_particles: usize,
    pub learn_iterations: u64,
    pub learn_capped_chains: u64,
    /// Largest |ln| correction the first learning pass applied to a
    /// predicted weight; above ln 2 means a step outran the weights.
    pub worst_first_correction: f64,
    /// Range of the weight-normalization check, which is 1 in expectation.
    pub check_min: f64,
    pub check_max: f64,
    pub far_steps: u64,
    pub mutate: MutateCost,
    /// Mutations applied (particles times calls).
    pub mutations: u64,
    pub phase1_seconds: f64,
    /// Relative variance of the regularization factor g at beta = 1.
    pub gap_relvar: f64,
}

#[derive(Debug, Clone)]
pub struct Plan {
    /// Segment s multiplies the ratios of steps `segments[s]`, starting
    /// from a population at level `segments[s].start`.
    pub segments: Vec<Range<usize>>,
    pub particles: usize,
    pub islands: usize,
    /// Sum of the per-step relative variances, g included.
    pub relvar_sum: f64,
    /// The bound's factor (1 - sqrt(g) exp(-t))^-2.
    pub mixing_factor: f64,
    /// Relative variance one island is planned to have.
    pub island_relvar: f64,
    /// Most particles a restart may grow an island to: what the bound asks.
    pub particles_bound: usize,
    /// Sampler time per stage for the segment starts.
    pub stage_time: f64,
    /// Mutation events of the first round, extrapolated from Phase 1's
    /// mutations; 0 when Phase 1 made none.
    pub predicted_events: f64,
}

#[derive(Debug, Clone)]
pub struct Estimate {
    pub ln_permanent: f64,
    /// Standard error of ln_permanent, from the spread of the islands.
    pub std_error: f64,
    pub met_target: bool,
    pub rounds: usize,
    pub islands: usize,
    /// Particles per island, after any restart.
    pub particles: usize,
    /// Measured relative variance of one island (all segments).
    pub island_relvar: f64,
    pub mean_gap: f64,
    pub mutate: MutateCost,
    pub mutations: u64,
    /// Cost of the sampler calls that started segments.
    pub sampler: SampleCost,
    pub verdict: Verdict,
    pub seconds: f64,
}

/// What the segment-boundary check found.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Verdict {
    /// Carried and fresh populations agree at every boundary; the value is
    /// the largest difference in standard errors.
    Consistent(f64),
    /// They disagree at some boundary by this many standard errors.
    Inconsistent(f64),
    /// One segment: there is no boundary to check.
    Unchecked,
}

/// What the driver reports while it runs (see `Session::observe`).
pub enum Progress<'a> {
    /// Phase 1 fixed one more temperature.
    Level { index: usize, level: &'a Level },
    /// Phase 2 finished `done` of this rank's `total` temperature sweeps.
    Sweep { round: usize, done: u64, total: u64 },
    /// One round of islands completed.
    Round { index: usize, estimate: &'a Estimate },
}

type Observer<'a> = Option<Box<dyn FnMut(Progress<'_>) + 'a>>;

const PHASE_LEARN_INIT: u64 = 1;
const PHASE_LEARN: u64 = 2;
const PHASE_LEARN_RESAMPLE: u64 = 3;
const PHASE_INIT: u64 = 1000;
const PHASE_ANNEAL: u64 = 1001;
const PHASE_MUTATE: u64 = 1002;
const PHASE_RESAMPLE: u64 = 1003;

/// Particles are pooled per rank into this many groups, whose spread gives
/// the error of the learned weights.
const LEARN_GROUPS: usize = 64;
/// FAR cycles of one learning round, in units of n^2.
const LEARN_CYCLES: u64 = 64;
const LEARN_MAX_ITERATIONS: usize = 8;
const LEARN_MAX_ROUNDS: usize = 7;
/// Target relative standard error of each anchor weight.
const LEARN_RSE: f64 = 0.15;
/// A learning pass whose corrections are all below this is accepted.
const LEARN_ACCEPT: f64 = 0.2624; // ln 1.3
/// Relative variance aimed for when choosing a temperature step.
const STEP_RELVAR: f64 = 0.25;
/// Relative variance one island is sized for.
const ISLAND_RELVAR: f64 = 0.25;
/// Islands are sized as if their variance were this many times the sum of
/// the step variances, which is what perfect mixing would give.
const MIXING_MARGIN: f64 = 2.0;
/// What an island's variance has measured, in the same units.
const EXPECTED_MIXING: f64 = 1.2;
/// First islands noisier than this are thrown away and rerun larger.
const RESTART_RELVAR: f64 = 1.0;
const MIN_PARTICLES: usize = 64;
const MIN_ISLANDS: usize = 32;
const DEFAULT_SEGMENTS: usize = 4;
/// Default sampler time per stage, in units of `initial_stage_time`.
const SAMPLER_TIME: f64 = 4.0;
const SAMPLER_TRIALS: u64 = 96;
const MAX_TOPUP_ROUNDS: usize = 8;
/// Carried and fresh populations agree when they differ by at most this
/// many standard errors of their difference.
const AGREEMENT: f64 = 3.5;

struct Learned {
    log_w: Vec<f64>,
    /// E[H | N(r,v)] - E[H | P], the beta-derivative of ln w*(r,v).
    drift: Vec<f64>,
}

/// Relative variance of exp(-step H) over the pilot energies, and the
/// largest such weight over their mean.
fn weight_stats(pilot: &[f64], step: f64) -> (f64, f64) {
    let min = pilot.iter().copied().fold(f64::INFINITY, f64::min);
    // summed by fixed chunks, in order, so the result does not depend on
    // how the pool schedules them
    let sums: Vec<(f64, f64)> = pilot
        .par_chunks(4096)
        .map(|chunk| {
            chunk.iter().fold((0.0, 0.0), |(m1, m2), &h| {
                let x = (-step * (h - min)).exp();
                (m1 + x, m2 + x * x)
            })
        })
        .collect();
    let (m1, m2) = sums.iter().fold((0.0, 0.0), |a, b| (a.0 + b.0, a.1 + b.1));
    let k = pilot.len() as f64;
    let mean = m1 / k;
    ((m2 / k) / (mean * mean) - 1.0, 1.0 / mean)
}

/// Largest step in (0, cap] whose pilot relative variance stays below target.
fn step_for_relvar(pilot: &[f64], cap: f64, target: f64) -> f64 {
    if weight_stats(pilot, cap).0 <= target {
        return cap;
    }
    let (mut lo, mut hi) = (0.0, cap);
    for _ in 0..60 {
        let mid = 0.5 * (lo + hi);
        if weight_stats(pilot, mid).0 <= target { lo = mid } else { hi = mid }
    }
    lo
}

/// `count` uniform random permutations; number `p` uses the stream of
/// particle `first_id + p`.
fn uniform_permutations(n: usize, count: usize, seed: u64, phase: u64, first_id: u64) -> Vec<u32> {
    let mut perms = vec![0u32; count * n];
    perms.par_chunks_mut(n).enumerate().for_each(|(p, row)| {
        let mut rng = SmallRng::seed_from_u64(stream_seed(seed, phase, 0, first_id + p as u64));
        for (u, slot) in row.iter_mut().enumerate() {
            *slot = u as u32;
        }
        row.shuffle(&mut rng);
    });
    perms
}

/// Multinomial resampling: `weights.len()` ancestors, each index drawn with
/// probability proportional to its weight, offset by `base`.
fn multinomial(weights: &[f64], base: usize, rng: &mut SmallRng, ancestors: &mut Vec<u32>) {
    let mut cumulative = Vec::with_capacity(weights.len());
    let mut total = 0.0;
    for &w in weights {
        total += w;
        cumulative.push(total);
    }
    let ancestor = |x: f64| (base + cumulative.partition_point(|&c| c <= x).min(weights.len() - 1)) as u32;
    // The draws are taken in the generator's order either way; a large
    // population then looks them up on the pool, which a small one is not
    // worth waking for.
    if weights.len() < 1 << 14 {
        ancestors.extend((0..weights.len()).map(|_| ancestor(rng.random::<f64>() * total)));
    } else {
        let draws: Vec<f64> = (0..weights.len()).map(|_| rng.random::<f64>() * total).collect();
        ancestors.par_extend(draws.par_iter().map(|&x| ancestor(x)));
    }
}

/// Mean, and squared standard error of the mean.
fn mean_and_error(values: &[f64]) -> (f64, f64) {
    let k = values.len() as f64;
    let mean = values.iter().sum::<f64>() / k;
    let ss: f64 = values.iter().map(|x| (x - mean) * (x - mean)).sum();
    (mean, ss / (k * (k - 1.0)))
}

/// The part of `islands` of each segment that `rank` runs. The islands of
/// all segments, one segment after another, are cut into one contiguous run
/// per rank. A rank then sweeps as few segments as it can, each with as many
/// islands as it can: on a GPU a sweep takes about as long for a few islands
/// as for many.
fn rank_share(segments: usize, islands: Range<u64>, rank: usize, ranks: usize) -> Vec<(usize, Range<u64>)> {
    let count = islands.end - islands.start;
    let cut = |rank: usize| segments as u64 * count * rank as u64 / ranks as u64;
    let (from, to) = (cut(rank), cut(rank + 1));
    (0..segments)
        .filter_map(|s| {
            let base = s as u64 * count;
            let lo = from.clamp(base, base + count) - base;
            let hi = to.clamp(base, base + count) - base;
            (lo < hi).then(|| (s, islands.start + lo..islands.start + hi))
        })
        .collect()
}

/// One island's run of one segment.
#[derive(Debug, Clone, Copy)]
struct SegmentRun {
    segment: usize,
    /// ln of the product of mean weights (and of the mean of g at the end).
    ln_ratio: f64,
    /// Mean weight of the fresh population at the segment's first level.
    fresh: f64,
    /// Mean weight of the carried population one level past the segment,
    /// or the mean of g for the last segment.
    carried: f64,
}

impl SegmentRun {
    const FIELDS: usize = 4;

    fn pack(&self, out: &mut Vec<f64>) {
        out.extend([self.segment as f64, self.ln_ratio, self.fresh, self.carried]);
    }

    fn unpack(fields: &[f64]) -> Self {
        SegmentRun { segment: fields[0] as usize, ln_ratio: fields[1], fresh: fields[2], carried: fields[3] }
    }
}

/// What a segment's sweep reads from its session.
#[derive(Clone, Copy)]
struct Ladder<'s> {
    scaled: &'s Scaled,
    config: &'s Config,
    stages: &'s StageSchedule,
    anchor: usize,
    levels: &'s [Level],
    plan: &'s Plan,
}

/// One sweep of a segment: its islands' runs and what they cost.
struct Sweep {
    runs: Vec<SegmentRun>,
    mutate: MutateCost,
    sampler: SampleCost,
}

/// Run islands `islands` of segment `segment` on `engine`. `tick` is called
/// once per temperature.
fn run_segment<E: Engine>(
    ladder: &Ladder<'_>,
    engine: &mut E,
    segment: usize,
    islands: Range<u64>,
    tick: &mut dyn FnMut(),
) -> anyhow::Result<Sweep> {
    let Ladder { scaled, config, stages, anchor, levels, plan } = *ladder;
    let n = scaled.n;
    let seed = config.seed;
    let particles = plan.particles;
    let count = (islands.end - islands.start) as usize;
    let last = levels.len() - 1;
    let Range { start: first, end } = plan.segments[segment].clone();
    let first_id = islands.start * particles as u64;
    let tables = |level: &Level| Tables::<E::Counter>::build(scaled, level.beta, &level.log_w, anchor);
    let mut mutate = MutateCost::default();
    let mut sampler = SampleCost::default();

    if first == 0 {
        engine.load(&uniform_permutations(n, count * particles, seed, PHASE_INIT, first_id), first_id)?;
    } else {
        let stream = Stream { seed, phase: PHASE_ANNEAL, level: first as u64 };
        sampler = engine.anneal(
            &tables(&levels[first]),
            stages,
            plan.stage_time,
            SAMPLER_TRIALS,
            count * particles,
            first_id,
            stream,
        )?;
    }
    let mut runs = vec![SegmentRun { segment, ln_ratio: 0.0, fresh: f64::NAN, carried: f64::NAN }; count];
    let mut weights = vec![0.0f64; particles];
    let mut ancestors = Vec::with_capacity(count * particles);
    for (i, level) in levels.iter().enumerate().take(end + 1).skip(first) {
        if i > first {
            let stream = Stream { seed, phase: PHASE_MUTATE, level: i as u64 };
            mutate.add(engine.mutate(&tables(level), config.mutation_time, stream)?);
        }
        tick();
        if i == last {
            let gaps = engine.log_gaps()?;
            for (run, gaps) in runs.iter_mut().zip(gaps.chunks(particles)) {
                let mean = gaps.iter().map(|g| g.exp()).sum::<f64>() / particles as f64;
                run.ln_ratio += mean.ln();
                run.carried = mean;
            }
            break;
        }
        let energies = engine.energies()?;
        ancestors.clear();
        for (island, (run, energies)) in runs.iter_mut().zip(energies.chunks(particles)).enumerate() {
            for (w, h) in weights.iter_mut().zip(energies) {
                *w = (-level.step * (h - level.shift)).exp();
            }
            let mean = weights.iter().sum::<f64>() / particles as f64;
            if i == first {
                run.fresh = mean;
            }
            if i == end {
                // one level past the segment: only for the boundary check
                run.carried = mean;
                continue;
            }
            run.ln_ratio += -level.step * level.shift + mean.ln();
            let id = islands.start + island as u64;
            let mut rng = SmallRng::seed_from_u64(stream_seed(seed, PHASE_RESAMPLE, i as u64, id));
            multinomial(&weights, island * particles, &mut rng, &mut ancestors);
        }
        if i < end {
            engine.resample(&ancestors)?;
        }
    }
    Ok(Sweep { runs, mutate, sampler })
}

pub struct Session<'a, E: Engine, C: Comm> {
    scaled: &'a Scaled,
    config: &'a Config,
    engine: E,
    comm: &'a C,
    observer: Observer<'a>,
    pub stages: StageSchedule,
    anchor: usize,
    pub diagnostics: Diagnostics,
    /// Target standard error of ln(estimate).
    pub sigma_target: f64,
    /// Normal quantile of the reported interval.
    pub z: f64,
    /// Gives every Phase 1 engine call its own streams.
    calls: u64,
    seconds: LearnSeconds,
}

/// Seconds spent in each part of Phase 1, for the debug log.
#[derive(Debug, Default)]
struct LearnSeconds {
    tables: f64,
    far: f64,
    energies: f64,
    step: f64,
    resample: f64,
}

impl<'a, E: Engine, C: Comm> Session<'a, E, C> {
    pub fn new(scaled: &'a Scaled, config: &'a Config, engine: E, comm: &'a C) -> anyhow::Result<Self> {
        anyhow::ensure!(config.epsilon > 0.0 && config.epsilon < 1.0, "epsilon must be in (0, 1)");
        anyhow::ensure!(config.delta > 0.0 && config.delta < 1.0, "delta must be in (0, 1)");
        anyhow::ensure!(config.mutation_time > 0.0, "the mutation time must be positive");
        anyhow::ensure!(scaled.n >= 2, "n must be at least 2");
        // half of delta covers statistical error; the rest is left for the
        // segment starts (their tolerance in rigorous mode, the check otherwise)
        let z = normal_quantile(1.0 - config.delta / 4.0);
        Ok(Session {
            scaled,
            config,
            engine,
            comm,
            observer: None,
            stages: StageSchedule::new(scaled.n, config.density),
            anchor: 0,
            diagnostics: Diagnostics { check_min: f64::INFINITY, check_max: 0.0, ..Default::default() },
            sigma_target: (1.0 + config.epsilon).ln() / z,
            z,
            calls: 0,
            seconds: LearnSeconds::default(),
        })
    }

    /// Receive a `Progress` event at every step of both phases.
    pub fn observe(&mut self, observer: impl FnMut(Progress<'_>) + 'a) {
        self.observer = Some(Box::new(observer));
    }

    pub fn engine_name(&self) -> String {
        self.engine.name()
    }

    fn learn_stream(&mut self) -> Stream {
        self.calls += 1;
        Stream { seed: self.config.seed, phase: PHASE_LEARN, level: self.calls }
    }

    /// Learn the anchor weights at `beta`, starting from the guess `log_w`.
    /// The population is at mu_beta on entry (exactly, when `exact`) and on
    /// exit.
    fn learn(&mut self, level: usize, beta: f64, mut log_w: Vec<f64>, exact: bool) -> anyhow::Result<Learned> {
        let scaled = self.scaled;
        let n = scaled.n;
        let anchor = self.anchor;
        let local = self.diagnostics.learn_particles;
        let total = (local * self.comm.size()) as u64;
        let groups = local.min(LEARN_GROUPS);
        let mut drift = vec![0.0; n];
        for iteration in 0..LEARN_MAX_ITERATIONS {
            self.diagnostics.learn_iterations += 1;
            let clock = Instant::now();
            let tables = Tables::<E::Counter>::build(scaled, beta, &log_w, anchor);
            self.seconds.tables += clock.elapsed().as_secs_f64();
            // Once per temperature, to spread out the copies resampling
            // made; after that the FAR runs themselves move the particles.
            if iteration == 0 && !exact && self.config.learn_mutation_time > 0.0 {
                let stream = self.learn_stream();
                let cost = self.engine.mutate(&tables, self.config.learn_mutation_time, stream)?;
                self.diagnostics.mutate.add(cost);
                self.diagnostics.mutations += local as u64;
            }
            let mut cycles = (LEARN_CYCLES * (n * n) as u64).div_ceil(total).max(1);
            let mut group_cycles = vec![0u64; groups];
            let mut group_visits = vec![vec![0u64; n]; groups];
            // visit_energy[v] for each v, then perfect_energy
            let mut energy_sums = vec![0.0; n + 1];
            let mut totals = vec![0u64; n];
            let mut total_cycles = 0u64;
            for _round in 0..LEARN_MAX_ROUNDS {
                let stream = self.learn_stream();
                let clock = Instant::now();
                let far = self.engine.far(&tables, cycles, 256 * cycles + 100_000, groups, stream)?;
                self.seconds.far += clock.elapsed().as_secs_f64();
                self.diagnostics.far_steps += far.steps;
                self.diagnostics.learn_capped_chains += far.capped;
                for g in 0..groups {
                    group_cycles[g] += far.cycles[g];
                    for (total, visits) in group_visits[g].iter_mut().zip(&far.visits[g]) {
                        *total += visits;
                    }
                }
                for (total, energy) in energy_sums.iter_mut().zip(&far.visit_energy) {
                    *total += energy.to_num::<f64>();
                }
                energy_sums[n] += far.perfect_energy.to_num::<f64>();
                // Over all ranks: cycles, groups, visits_v; and the second
                // moments sum k^2, sum visits_v^2, sum visits_v k of the
                // groups, for the ratio-estimator error of each class.
                let mut counts = vec![0u64; 2 + n];
                let mut moments = vec![0.0f64; 1 + 2 * n];
                for g in 0..groups {
                    let k = group_cycles[g];
                    counts[0] += k;
                    counts[1] += 1;
                    moments[0] += (k as f64) * (k as f64);
                    for v in 0..n {
                        let visits = group_visits[g][v];
                        counts[2 + v] += visits;
                        moments[1 + v] += (visits as f64) * (visits as f64);
                        moments[1 + n + v] += visits as f64 * k as f64;
                    }
                }
                self.comm.sum_u64(&mut counts);
                self.comm.sum_f64(&mut moments);
                total_cycles = counts[0];
                let chains = counts[1] as f64;
                totals.copy_from_slice(&counts[2..]);
                let mut worst: f64 = 0.0;
                for v in 0..n {
                    if totals[v] == 0 {
                        worst = f64::INFINITY;
                        break;
                    }
                    let rate = totals[v] as f64 / total_cycles as f64;
                    let ss = moments[1 + v] - 2.0 * rate * moments[1 + n + v] + rate * rate * moments[0];
                    let rse = (ss.max(0.0) * chains / (chains - 1.0)).sqrt() / totals[v] as f64;
                    worst = worst.max(rse);
                }
                if worst <= LEARN_RSE {
                    break;
                }
                cycles *= 2;
            }
            self.comm.sum_f64(&mut energy_sums);
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
                self.diagnostics.worst_first_correction = self.diagnostics.worst_first_correction.max(worst_correction);
            }
            let mean_perfect = energy_sums[n] / total_cycles as f64;
            for v in 0..n {
                drift[v] = if totals[v] > 0 { energy_sums[v] / totals[v] as f64 - mean_perfect } else { 0.0 };
            }
            debug!(level, iteration, worst_correction, check, total_cycles, "learning pass");
            if worst_correction <= LEARN_ACCEPT {
                break;
            }
        }
        Ok(Learned { log_w, drift })
    }

    /// Phase 1: build the temperature schedule and learn weights along it,
    /// with one population carried from beta = 0 to 1.
    pub fn learn_schedule(&mut self) -> anyhow::Result<Vec<Level>> {
        let scaled = self.scaled;
        let n = scaled.n;
        let seed = self.config.seed;
        let rank = self.comm.rank();
        let ranks = self.comm.size();
        let h_max = scaled.h_max();
        let spread = 2.0 * scaled.log_inv_tau;
        let start = Instant::now();

        let wanted = self.config.learn_particles.unwrap_or_else(|| self.engine.learn_particles() * ranks);
        let local = wanted.div_ceil(ranks).min(self.engine.capacity());
        anyhow::ensure!(local * ranks >= 2 * LEARN_GROUPS, "Phase 1 needs at least {} particles", 2 * LEARN_GROUPS);
        self.diagnostics.learn_particles = local;
        let first_id = (rank * local) as u64;
        // at beta = 0 every activity is 1: mu_0 is uniform over permutations
        self.engine.load(&uniform_permutations(n, local, seed, PHASE_LEARN_INIT, first_id), first_id)?;

        let mut levels: Vec<Level> = Vec::new();
        let mut beta = 0.0f64;
        // and w*(r,v) = n!/(n-1)! = n exactly
        let mut guess = vec![(n as f64).ln(); n];
        let mut previous_step: Option<f64> = None;
        loop {
            let learned = self.learn(levels.len(), beta, guess, levels.is_empty())?;
            let clock = Instant::now();
            let energies = self.engine.energies()?;
            self.seconds.energies += clock.elapsed().as_secs_f64();
            let pilot = self.comm.gather_f64(&energies);
            let shift = pilot.iter().sum::<f64>() / pilot.len() as f64;
            if beta >= 1.0 {
                let gaps = self.comm.gather_f64(&self.engine.log_gaps()?);
                let (mut m1, mut m2) = (0.0, 0.0);
                for gap in &gaps {
                    let g = gap.exp();
                    m1 += g;
                    m2 += g * g;
                }
                let k = gaps.len() as f64;
                // g lies in [0,1] with mean >= 2/3: relative variance at most 1/2
                self.diagnostics.gap_relvar =
                    if m1 > 0.0 { ((m2 / k) / ((m1 / k) * (m1 / k)) - 1.0).min(0.5) } else { 0.5 };
                levels.push(Level { beta, log_w: learned.log_w, step: 0.0, shift, relvar: 0.0, weight_max: 1.0 });
                self.notify_level(&levels);
                break;
            }
            let clock = Instant::now();
            let remaining = 1.0 - beta;
            // the paper's guaranteed step (Lemma 50)
            let floor = 0.25 * (1.0 / h_max).max(beta / spread);
            let step = if self.config.rigorous_schedule {
                floor
            } else {
                let by_variance = step_for_relvar(&pilot, remaining, STEP_RELVAR);
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
            let (relvar, weight_max) = weight_stats(&pilot, step);
            let relvar = relvar.max(1e-9);
            self.seconds.step += clock.elapsed().as_secs_f64();
            debug!(level = levels.len(), beta, step, relvar, weight_max, "temperature fixed");
            // first-order prediction of the next weights
            guess = (0..n).map(|v| learned.log_w[v] + step * learned.drift[v]).collect();
            levels.push(Level { beta, log_w: learned.log_w, step, shift, relvar, weight_max });
            self.notify_level(&levels);
            // carry this rank's particles to the next temperature
            let clock = Instant::now();
            let weights: Vec<f64> = energies.iter().map(|h| (-step * (h - shift)).exp()).collect();
            let mut rng =
                SmallRng::seed_from_u64(stream_seed(seed, PHASE_LEARN_RESAMPLE, levels.len() as u64, rank as u64));
            let mut ancestors = Vec::with_capacity(local);
            multinomial(&weights, 0, &mut rng, &mut ancestors);
            self.engine.resample(&ancestors)?;
            self.seconds.resample += clock.elapsed().as_secs_f64();
            previous_step = Some(step);
            beta = if step >= remaining { 1.0 } else { beta + step };
        }
        // totals over all ranks
        let d = &mut self.diagnostics;
        let mut counts =
            [d.learn_capped_chains, d.far_steps, d.mutations, d.mutate.events, d.mutate.over_budget, local as u64];
        let mut time = [d.mutate.time];
        self.comm.sum_u64(&mut counts);
        self.comm.sum_f64(&mut time);
        d.learn_capped_chains = counts[0];
        d.far_steps = counts[1];
        d.mutations = counts[2];
        d.mutate = MutateCost { events: counts[3], time: time[0], over_budget: counts[4] };
        d.learn_particles = counts[5] as usize;
        d.phase1_seconds = start.elapsed().as_secs_f64();
        debug!(seconds = ?self.seconds, total = d.phase1_seconds, passes = d.learn_iterations, "Phase 1 time");
        Ok(levels)
    }

    fn notify_level(&mut self, levels: &[Level]) {
        if let Some(observer) = self.observer.as_mut() {
            let index = levels.len() - 1;
            observer(Progress::Level { index, level: &levels[index] });
        }
    }

    /// Segments, particles and islands for Phase 2.
    pub fn plan(&self, levels: &[Level]) -> Plan {
        let steps = levels.len() - 1;
        let wanted = self.config.segments.unwrap_or(self.comm.size().max(DEFAULT_SEGMENTS));
        let count = wanted.clamp(1, steps.max(1));
        let segments: Vec<Range<usize>> = (0..count).map(|s| s * steps / count..(s + 1) * steps / count).collect();

        let relvar_sum = levels.iter().map(|l| l.relvar).sum::<f64>() + self.diagnostics.gap_relvar;
        let weight_max = levels.iter().map(|l| l.weight_max).fold(1.0, f64::max);
        let q = weight_max.sqrt() * (-self.config.mutation_time).exp();
        let mixing_factor = if q < 0.9 { 1.0 / ((1.0 - q) * (1.0 - q)) } else { 100.0 };
        // The bound's factor is far above anything measured, so islands are
        // sized from a fixed margin over perfect mixing; `estimate` grows
        // them, up to what the bound asks, if the first ones come out noisy.
        let sized = |factor: f64| ((factor * relvar_sum / ISLAND_RELVAR).ceil() as usize).max(MIN_PARTICLES);
        let capacity = self.engine.capacity();
        let particles = self.config.particles.unwrap_or(sized(MIXING_MARGIN)).min(capacity);
        let particles_bound = match self.config.particles {
            Some(fixed) => fixed.min(capacity),
            None => sized(mixing_factor.max(MIXING_MARGIN)).min(capacity),
        };
        let island_relvar = MIXING_MARGIN * relvar_sum / particles as f64;
        // Start with the islands the target is expected to take, so that
        // one round usually does and fills the engine: measured island
        // variances run about 1.2 times the perfect-mixing value. Never
        // fewer than measure their own spread.
        let sigma2 = self.sigma_target * self.sigma_target;
        let expected = EXPECTED_MIXING * relvar_sum / particles as f64;
        let islands = self.config.islands.unwrap_or(((1.1 * expected / sigma2).ceil() as usize).max(MIN_ISLANDS));
        let fresh = (count - 1) * particles * islands;
        let stage_time = if self.config.rigorous_sampler {
            // the total-variation budget delta/2 is shared by every fresh
            // draw, with room for the islands a top-up may add
            self.stages.rigorous_stage_time((self.config.delta / 2.0) / (8.0 * fresh.max(1) as f64))
        } else {
            SAMPLER_TIME * self.stages.initial_stage_time(self.scaled.n)
        };
        let d = &self.diagnostics;
        // a mutation's cost grows in proportion to its time
        let per_mutation = d.mutate.events as f64 / d.mutations.max(1) as f64 * self.config.mutation_time
            / self.config.learn_mutation_time.max(f64::MIN_POSITIVE);
        let predicted_events = per_mutation * (particles * islands * (steps + count - 1)) as f64;
        Plan {
            segments,
            particles,
            islands,
            relvar_sum,
            mixing_factor,
            island_relvar,
            particles_bound,
            stage_time,
            predicted_events,
        }
    }

    /// Sweep `jobs` (a segment and a batch of its islands each) and return
    /// their sweeps in the order of the jobs. As many run at the same time
    /// as the engine says it can move at once, each on an engine of its
    /// own; a job's result does not depend on what runs beside it.
    fn sweep(
        &mut self,
        levels: &[Level],
        plan: &Plan,
        jobs: &[(usize, Range<u64>)],
        round: usize,
    ) -> anyhow::Result<Vec<Sweep>> {
        let ladder = Ladder {
            scaled: self.scaled,
            config: self.config,
            stages: &self.stages,
            anchor: self.anchor,
            levels,
            plan,
        };
        let total: u64 = jobs.iter().map(|(segment, _)| plan.segments[*segment].len() as u64 + 1).sum();
        let largest = jobs.iter().map(|(_, islands)| islands.end - islands.start).max().unwrap_or(0) as usize;
        let lanes = self.engine.concurrency(largest * plan.particles).min(jobs.len());
        let mut swept = 0u64;
        let observer = &mut self.observer;
        let mut tick = || {
            swept += 1;
            if let Some(observer) = observer.as_mut() {
                observer(Progress::Sweep { round, done: swept, total });
            }
        };
        if lanes <= 1 {
            return jobs
                .iter()
                .map(|(segment, islands)| run_segment(&ladder, &mut self.engine, *segment, islands.clone(), &mut tick))
                .collect();
        }
        let mut forks = (1..lanes).map(|_| self.engine.fork()).collect::<anyhow::Result<Vec<E>>>()?;
        let next = AtomicUsize::new(0);
        let (ticks, ticked) = mpsc::channel::<()>();
        let mut sweeps: Vec<Option<Sweep>> = jobs.iter().map(|_| None).collect();
        thread::scope(|scope| {
            let workers: Vec<_> = std::iter::once(&mut self.engine)
                .chain(&mut forks)
                .map(|engine| {
                    let (ladder, next, ticks) = (&ladder, &next, ticks.clone());
                    scope.spawn(move || {
                        let mut done = Vec::new();
                        loop {
                            let job = next.fetch_add(1, Ordering::Relaxed);
                            let Some((segment, islands)) = jobs.get(job) else {
                                break;
                            };
                            let mut tick = || {
                                let _ = ticks.send(());
                            };
                            done.push((job, run_segment(ladder, engine, *segment, islands.clone(), &mut tick)?));
                        }
                        anyhow::Ok(done)
                    })
                })
                .collect();
            drop(ticks);
            for () in ticked {
                tick();
            }
            for worker in workers {
                for (job, sweep) in worker.join().unwrap_or_else(|panic| std::panic::resume_unwind(panic))? {
                    sweeps[job] = Some(sweep);
                }
            }
            anyhow::Ok(())
        })?;
        Ok(sweeps.into_iter().map(|sweep| sweep.expect("every job is swept once")).collect())
    }

    /// Phase 2: run islands of every segment until the measured standard
    /// error meets the target. Each round is reported as `Progress::Round`.
    pub fn estimate(&mut self, levels: &[Level], plan: &Plan) -> anyhow::Result<Estimate> {
        let start = Instant::now();
        let mut plan = plan.clone();
        let scaled = self.scaled;
        let n = scaled.n;
        let rank = self.comm.rank();
        let ranks = self.comm.size();
        let segments = plan.segments.len();
        let sigma2 = self.sigma_target * self.sigma_target;
        let mut mutate = MutateCost::default();
        let mut sampler = SampleCost::default();
        let mut packed: Vec<f64> = Vec::new();
        // islands first..done are in use; earlier ones were thrown away
        let mut first = 0u64;
        let mut done = 0u64;
        let mut want = plan.islands as u64;
        let mut estimate = None;
        for round in 0..MAX_TOPUP_ROUNDS {
            let capacity = (self.engine.capacity() / plan.particles).max(1) as u64;
            let mine = rank_share(segments, done..want, rank, ranks);
            let jobs: Vec<(usize, Range<u64>)> = mine
                .into_iter()
                .flat_map(|(s, islands)| {
                    (islands.start..islands.end)
                        .step_by(capacity as usize)
                        .map(move |from| (s, from..(from + capacity).min(islands.end)))
                })
                .collect();
            for sweep in self.sweep(levels, &plan, &jobs, round)? {
                mutate.add(sweep.mutate);
                sampler.add(sweep.sampler);
                for run in sweep.runs {
                    run.pack(&mut packed);
                }
            }
            done = want;

            let all = self.comm.gather_f64(&packed);
            let mut runs: Vec<Vec<SegmentRun>> = vec![Vec::new(); segments];
            for fields in all.chunks(SegmentRun::FIELDS) {
                let run = SegmentRun::unpack(fields);
                runs[run.segment].push(run);
            }
            // Each segment's estimate is the mean of its islands' ratios;
            // the relative variances of independent segments add.
            let mut ln_ratio = 0.0;
            let mut variance = 0.0;
            for segment in &runs {
                let values: Vec<f64> = segment.iter().map(|run| run.ln_ratio).collect();
                let ln_mean = log_sum_exp(&values) - (values.len() as f64).ln();
                let relative: Vec<f64> = values.iter().map(|v| (v - ln_mean).exp()).collect();
                ln_ratio += ln_mean;
                variance += mean_and_error(&relative).1;
            }
            let mut worst: f64 = 0.0;
            for boundary in 1..segments {
                // An island's carried mean weight is biased by O(1/P), but
                // its product with the island's own estimate is unbiased
                // for one more ratio. So the carried side is the ratio of
                // two means over islands, sum(Z c) / sum(Z), whose error
                // comes from the residuals Z (c - ratio) / mean(Z).
                let previous = &runs[boundary - 1];
                let top = previous.iter().map(|run| run.ln_ratio).fold(f64::NEG_INFINITY, f64::max);
                let estimates: Vec<f64> = previous.iter().map(|run| (run.ln_ratio - top).exp()).collect();
                let total: f64 = estimates.iter().sum();
                let a = previous.iter().zip(&estimates).map(|(run, z)| z * run.carried).sum::<f64>() / total;
                let scale = previous.len() as f64 / total;
                let residuals: Vec<f64> =
                    previous.iter().zip(&estimates).map(|(run, z)| z * (run.carried - a) * scale).collect();
                let va = mean_and_error(&residuals).1;
                let fresh: Vec<f64> = runs[boundary].iter().map(|run| run.fresh).collect();
                let (b, vb) = mean_and_error(&fresh);
                // When every particle has the same weight (a 0/1 matrix at
                // high beta) both sides are equal up to rounding, and so are
                // their errors; the floor keeps that from reading as a gap.
                let rounding = 1e-12 * a.abs().max(b.abs());
                let gap = (a - b) / (va + vb + rounding * rounding).sqrt();
                debug!(boundary, level = plan.segments[boundary].start, gap, "boundary check");
                if gap.abs() > worst.abs() {
                    worst = gap;
                }
            }
            let verdict = if segments == 1 {
                Verdict::Unchecked
            } else if worst.abs() <= AGREEMENT {
                Verdict::Consistent(worst)
            } else {
                Verdict::Inconsistent(worst)
            };
            let mut counts = [
                mutate.events,
                mutate.over_budget,
                sampler.trials,
                sampler.accepted,
                sampler.events,
                sampler.over_budget,
            ];
            let mut time = [mutate.time];
            self.comm.sum_u64(&mut counts);
            self.comm.sum_f64(&mut time);
            let steps = (levels.len() - 1 + segments - 1) as u64;
            let islands = (done - first) as usize;
            let gaps: Vec<f64> = runs[segments - 1].iter().map(|run| run.carried).collect();
            let current = Estimate {
                ln_permanent: ln_factorial(n) + ln_ratio - scaled.log_scale,
                std_error: variance.sqrt(),
                met_target: variance <= sigma2 * 1.02,
                rounds: round + 1,
                islands,
                particles: plan.particles,
                island_relvar: variance * islands as f64,
                mean_gap: mean_and_error(&gaps).0,
                mutate: MutateCost { events: counts[0], time: time[0], over_budget: counts[1] },
                mutations: steps * (islands * plan.particles) as u64,
                sampler: SampleCost {
                    trials: counts[2],
                    accepted: counts[3],
                    events: counts[4],
                    over_budget: counts[5],
                },
                verdict,
                seconds: start.elapsed().as_secs_f64(),
            };
            debug!(round, islands, std_error = current.std_error, target = self.sigma_target, "round of islands");
            if current.mutate.over_budget + current.sampler.over_budget > 0 {
                warn!(
                    mutations = current.mutate.over_budget,
                    sampler_trials = current.sampler.over_budget,
                    "some moves hit their event cap"
                );
            }
            if let Some(observer) = self.observer.as_mut() {
                observer(Progress::Round { index: round, estimate: &current });
            }
            let met = current.met_target;
            let island_relvar = current.island_relvar;
            estimate = Some(current);
            if met {
                break;
            }
            // Noisy islands have skewed values, and their spread is then a
            // poor error bar: start over with larger ones. Fresh island
            // numbers give the new islands streams of their own.
            if island_relvar > RESTART_RELVAR && plan.particles < plan.particles_bound {
                let grown = (plan.particles as f64 * island_relvar / ISLAND_RELVAR).ceil() as usize;
                warn!(
                    particles = plan.particles,
                    grown = grown.min(plan.particles_bound),
                    island_relvar,
                    "islands too noisy; starting over with larger ones"
                );
                plan.particles = grown.min(plan.particles_bound);
                packed.clear();
                first = done;
                want = done + plan.islands as u64;
                continue;
            }
            // islands are i.i.d., so the error falls as 1/sqrt(islands)
            let used = done - first;
            let needed = (1.1 * used as f64 * variance / sigma2).ceil() as u64;
            want = first + needed.max(used + used / 10 + 1);
        }
        Ok(estimate.unwrap())
    }
}

#[cfg(test)]
mod tests {
    use super::rank_share;

    #[test]
    fn ranks_share_every_island_once_and_evenly() {
        for (segments, count, ranks) in [(4, 91, 1), (4, 91, 2), (4, 91, 3), (8, 16, 8), (4, 5, 7), (1, 3, 8)] {
            let mut runs = vec![vec![0; count as usize]; segments];
            let mut loads = Vec::new();
            for rank in 0..ranks {
                let mine = rank_share(segments, 10..10 + count, rank, ranks);
                loads.push(mine.iter().map(|(_, islands)| islands.end - islands.start).sum::<u64>());
                for (s, islands) in mine {
                    for island in islands {
                        runs[s][(island - 10) as usize] += 1;
                    }
                }
            }
            assert!(runs.iter().flatten().all(|&times| times == 1), "{segments} segments, {ranks} ranks");
            assert!(loads.iter().max().unwrap() - loads.iter().min().unwrap() <= 1, "{loads:?}");
        }
    }
}
