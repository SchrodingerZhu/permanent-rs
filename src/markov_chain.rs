use crate::chain::JsvChain;
use crate::cooling_schedule::CoolingSchedule;
use crate::cooling_state::{Matrix, State};
use crate::graph;
use crate::graph::Match;
use rand::SeedableRng;
use rand::rngs::SmallRng;
use rayon::iter::{IntoParallelRefMutIterator, ParallelIterator};
use tracing::{info, warn};

/// Sizing for the final perfect-fraction round.
///
/// The estimator's last factor is Pr[sample is a perfect matching of G],
/// measured by counting hits, so its relative standard error is about
/// `1/sqrt(hits)`. A fixed sample count therefore overspends on dense graphs
/// and underspends badly on sparse ones: the 2x64 ladder sits near a 0.3% hit
/// rate, where a default budget buys only a few hundred hits and ~4% error,
/// while a dense graph reaches the same precision in a fraction of the work.
/// Sizing by hits makes the precision the constant and the cost the variable.
pub mod final_round {
    /// Hits to accumulate before the fraction is trusted (~3% relative error).
    pub const TARGET_HITS: usize = 1024;
    /// Cap on rounds, so a graph whose perfect matchings are effectively
    /// unreachable terminates instead of sampling forever.
    pub const MAX_ROUNDS: usize = 6;

    /// Samples per chain for the next top-up round, from the rate observed so
    /// far, or `None` once the target is met.
    ///
    /// The count is fixed before each round runs, so this is two-stage sizing
    /// rather than a sequential stopping rule; the residual bias is O(1/hits)
    /// against O(1/sqrt(hits)) sampling noise. The result never drops below
    /// the configured round, so this can only add work, never remove it.
    pub fn next_per_chain(
        hits: usize,
        total: usize,
        chains: usize,
        configured: usize,
    ) -> Option<usize> {
        if hits >= TARGET_HITS {
            return None;
        }
        let missing = (TARGET_HITS - hits) as f64;
        let rate = hits as f64 / total.max(1) as f64;
        let wanted = if rate > 0.0 {
            (missing / rate / chains.max(1) as f64).ceil() as usize
        } else {
            // No hits yet: nothing to extrapolate from, so escalate blindly.
            configured.saturating_mul(4)
        };
        let floor = configured.max(1);
        Some(wanted.clamp(floor, floor.saturating_mul(32)))
    }

    #[cfg(test)]
    mod test {
        use super::*;

        #[test]
        fn stops_once_the_hit_target_is_met() {
            assert_eq!(next_per_chain(TARGET_HITS, 10_000, 64, 16), None);
            assert_eq!(next_per_chain(TARGET_HITS + 1, 10_000, 64, 16), None);
        }

        #[test]
        fn sizes_the_next_round_from_the_observed_rate() {
            // 64 hits from 64k samples over 64 chains is a 0.1% rate, so the 960
            // still missing need ~960/0.001 = 960k more samples, 15000 per chain.
            // The configured round is large enough here that the cap does not bind.
            assert_eq!(next_per_chain(64, 64_000, 64, 1000), Some(15_000));
        }

        #[test]
        fn the_cap_binds_when_the_rate_is_very_low() {
            // Same 0.1% rate against a small configured round: the request is
            // 15000 per chain but 32x the configured 16 caps it at 512, so a
            // sparse graph escalates over several rounds rather than in one leap.
            assert_eq!(next_per_chain(64, 64_000, 64, 16), Some(512));
        }

        #[test]
        fn escalates_blindly_when_nothing_has_hit_yet() {
            assert_eq!(next_per_chain(0, 10_000, 64, 16), Some(64));
        }

        #[test]
        fn never_draws_less_than_the_configured_round() {
            // A high hit rate would ask for a tiny round; the floor keeps it at
            // the configured size so this can only ever add work.
            assert_eq!(next_per_chain(1000, 2000, 64, 16), Some(16));
        }

        #[test]
        fn caps_the_escalation() {
            // An extremely low rate would ask for an unbounded round.
            assert_eq!(next_per_chain(1, 10_000_000, 1, 16), Some(16 * 32));
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Config {
    /// number of chains
    pub num_of_chains: usize,
    /// potential mixing time of initial runs
    pub warmup_times: usize,
    /// potential relaxation time of the chain between per-step samples
    pub weight_sample_intervals: usize,
    /// potential relaxation time of the chain in the final sampling round
    pub estimator_sample_intervals: usize,
    /// per-chain samples per cooling step; the first half bootstraps the
    /// hole-weight table, the second half feeds the ratio estimator
    pub num_of_weight_estimations: usize,
    /// minimum per-chain samples in the final perfect-fraction round
    pub num_of_estimator_estimations: usize,
}

/// Result of one `evolve` step.
struct EvolveStats {
    /// estimate of Z(beta', w') / Z(beta, w)
    ratio: f64,
    /// ratio-phase samples that were perfect matchings of the real graph
    perfect_active_samples: usize,
    /// total ratio-phase samples
    total_samples: usize,
}

/// Per-step statistics reported to `cooling_evolve_with` observers.
pub struct StepStats {
    pub step: usize,
    pub total_steps: usize,
    pub beta: f64,
    pub ratio: f64,
    pub estimator: f64,
    /// count of ratio-phase samples that were perfect matchings of the real
    /// graph (the observable the final estimator is built from)
    pub accepted_samples: f64,
    /// total ratio-phase samples
    pub attempted_samples: usize,
}

/// Per-class occupancy histogram over the n^2 hole classes plus the perfect
/// class, merged across chains.
struct SampleCounts {
    size: usize,
    data: Vec<usize>,
    perfect: usize,
}

impl SampleCounts {
    pub fn new(size: usize) -> Self {
        SampleCounts {
            size,
            data: vec![0; size * size],
            perfect: 0,
        }
    }
    pub fn record(&mut self, hole: Option<(usize, usize)>) {
        match hole {
            Some((u, v)) => self.data[u * self.size + v] += 1,
            None => self.perfect += 1,
        }
    }
    pub fn merge(mut self, other: Self) -> Self {
        for (left, right) in self.data.iter_mut().zip(other.data) {
            *left += right;
        }
        self.perfect += other.perfect;
        self
    }
}

/// Ratio-phase accumulator.
struct RatioSum {
    sum: f64,
    perfect_active: usize,
    samples: usize,
}

impl RatioSum {
    fn new() -> Self {
        RatioSum {
            sum: 0.0,
            perfect_active: 0,
            samples: 0,
        }
    }
    fn merge(self, other: Self) -> Self {
        RatioSum {
            sum: self.sum + other.sum,
            perfect_active: self.perfect_active + other.perfect_active,
            samples: self.samples + other.samples,
        }
    }
}

pub struct MCState {
    #[allow(dead_code)]
    graph: graph::Graph,
    size: usize,
    config: Config,
    pub global_state: State,
    chains: Vec<JsvChain>,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            num_of_chains: 1024,
            warmup_times: 16384,
            weight_sample_intervals: 8,
            estimator_sample_intervals: 128,
            num_of_weight_estimations: 2048,
            num_of_estimator_estimations: 16,
        }
    }
}

impl MCState {
    pub fn new(graph: graph::Graph, config: Config) -> Self {
        let global_state = State::from(&graph);
        let size = graph.size;
        let chains = (0..config.num_of_chains)
            .map(|_| JsvChain::from_permutation(&Match::random(size), &global_state))
            .collect();
        MCState {
            graph,
            config,
            global_state,
            chains,
            size,
        }
    }

    pub fn warmup(&mut self) {
        self.chains.par_iter_mut().for_each(|x| {
            let mut rng = SmallRng::from_rng(&mut rand::rng());
            x.transit_n_times(&self.global_state, self.config.warmup_times, &mut rng);
        });
    }

    /// One occupancy-sampling pass over all chains at the current (beta, w).
    fn occupancy_pass(&mut self, samples_per_chain: usize) -> SampleCounts {
        let size = self.size;
        self.chains
            .par_iter_mut()
            .fold(
                || SampleCounts::new(size),
                |mut local, x| {
                    let mut rng = SmallRng::from_rng(&mut rand::rng());
                    for _ in 0..samples_per_chain {
                        x.transit_n_times(
                            &self.global_state,
                            self.config.weight_sample_intervals,
                            &mut rng,
                        );
                        local.record(x.hole());
                    }
                    local
                },
            )
            .reduce(|| SampleCounts::new(size), |left, right| left.merge(right))
    }

    /// One cooling step at the current (beta, w):
    ///
    /// 1. With the first half of the sample budget, histogram class
    ///    occupancies and bootstrap the next hole-weight table w' for the
    ///    current beta (BSVV weight update).
    /// 2. With the second, independent half, estimate the per-sample mean
    ///    of `e^{(beta - beta') inactive(M)} * w'(M)/w(M)`, which equals
    ///    Z(beta', w') / Z(beta, w) in expectation. Using a batch collected
    ///    after w' is fixed keeps the ratio estimator from being correlated
    ///    with the table it references, and giving it half the full budget
    ///    keeps the telescoping product's per-step ln-bias (~ Var/2N by
    ///    Jensen) from accumulating.
    /// 3. Install w'; the caller installs beta'.
    fn evolve(&mut self, next_beta: f64) -> EvolveStats {
        let size = self.size;
        let first_half = self.config.num_of_weight_estimations / 2;
        let second_half = self.config.num_of_weight_estimations - first_half;
        // BSVV's weight update (and through it the whole mixing argument)
        // assumes the sampled occupancies reflect the stationary
        // distribution, under which the perfect class holds ~1/(n^2+1) of
        // the mass when the weights are factor-2 correct. If the measured
        // perfect share falls far outside that band, the chains are lagging
        // the schedule; committing an update computed from lagged samples
        // can start a feedback spiral (every hole class slashed in lockstep,
        // ratio estimates collapsing with them). Stirring more and
        // resampling — the "slow with a warning" failure mode — is the
        // faithful response.
        let classes = size * size + 1;
        let expected_perfect = (self.config.num_of_chains * first_half) as f64 / classes as f64;
        let mut counts = self.occupancy_pass(first_half);
        const MAX_EQUILIBRATION_RETRIES: usize = 3;
        for retry in 0..MAX_EQUILIBRATION_RETRIES {
            let perfect = counts.perfect as f64;
            if perfect >= expected_perfect / 2.0 && perfect <= expected_perfect * 2.0 {
                break;
            }
            warn!(
                "perfect-class occupancy {} outside [{:.1}, {:.1}] at beta {:.4}; \
                 re-equilibrating (retry {}/{MAX_EQUILIBRATION_RETRIES})",
                counts.perfect,
                expected_perfect / 2.0,
                expected_perfect * 2.0,
                self.global_state.beta(),
                retry + 1,
            );
            counts = self.occupancy_pass(first_half);
        }
        let (next_weight, diagnostics) = Matrix::hole_weights_from_counts(
            &self.global_state.weight,
            &counts.data,
            counts.perfect,
        );
        if diagnostics.clamped_classes * 4 > size * size {
            warn!(
                "hole-weight update clamped {} of {} classes (occupancy {}..{}); \
                 the factor-2 weight invariant is likely violated — increase \
                 weight samples or slow the schedule",
                diagnostics.clamped_classes,
                size * size,
                diagnostics.min_count,
                diagnostics.max_count,
            );
        }

        let diff = self.global_state.beta() - next_beta;
        let ratio_sum = self
            .chains
            .par_iter_mut()
            .fold(RatioSum::new, |mut local, x| {
                let mut rng = SmallRng::from_rng(&mut rand::rng());
                for _ in 0..second_half {
                    x.transit_n_times(
                        &self.global_state,
                        self.config.weight_sample_intervals,
                        &mut rng,
                    );
                    let mut term = (diff * x.inactive_count() as f64).exp();
                    if let Some((u, v)) = x.hole() {
                        term *= next_weight.get(u, v) / self.global_state.weight.get(u, v);
                    }
                    local.sum += term;
                    local.samples += 1;
                    if x.is_fully_active_perfect() {
                        local.perfect_active += 1;
                    }
                }
                local
            })
            .reduce(RatioSum::new, |left, right| left.merge(right));

        self.global_state.weight = next_weight;
        EvolveStats {
            ratio: ratio_sum.sum / ratio_sum.samples.max(1) as f64,
            perfect_active_samples: ratio_sum.perfect_active,
            total_samples: ratio_sum.samples,
        }
    }

    /// Fraction of stationary samples that are perfect matchings of the real
    /// graph. Since such matchings have activity 1 at every beta,
    /// per(G) = Z * Pr_pi[M perfect and fully active] exactly; this measures
    /// that probability. The sample count is scaled to the n^2 + 1 classes so
    /// the target class (occupancy ~ 1/(n^2+1) under good weights) is hit
    /// often enough for a low-variance estimate.
    /// Draw one round of `per_chain` samples per chain, returning
    /// (fully-active-perfect hits, samples).
    fn perfect_fraction_round(&mut self, per_chain: usize) -> (usize, usize) {
        let (hits, samples) = self
            .chains
            .par_iter_mut()
            .fold(
                || (0usize, 0usize),
                |(mut hits, mut samples), x| {
                    let mut rng = SmallRng::from_rng(&mut rand::rng());
                    for _ in 0..per_chain {
                        x.transit_n_times(
                            &self.global_state,
                            self.config.estimator_sample_intervals,
                            &mut rng,
                        );
                        samples += 1;
                        if x.is_fully_active_perfect() {
                            hits += 1;
                        }
                    }
                    (hits, samples)
                },
            )
            .reduce(|| (0, 0), |a, b| (a.0 + b.0, a.1 + b.1));
        (hits, samples)
    }

    /// Fraction of stationary samples that are perfect matchings of the real
    /// graph, sampled until the hit count justifies the precision rather than
    /// for a fixed number of draws. See [`final_round`].
    fn estimate_perfect_fraction(&mut self) -> f64 {
        let configured = self
            .config
            .num_of_estimator_estimations
            .max((64 * (self.size * self.size + 1)).div_ceil(self.config.num_of_chains));
        let chains = self.config.num_of_chains;
        let (mut hits, mut samples) = self.perfect_fraction_round(configured);
        let mut rounds = 1;
        while rounds < final_round::MAX_ROUNDS {
            let Some(extra) = final_round::next_per_chain(hits, samples, chains, configured) else {
                break;
            };
            info!(
                "final round: {hits} of {samples} hits so far, short of \
                 {}; drawing {extra} more per chain",
                final_round::TARGET_HITS
            );
            let (more_hits, more_samples) = self.perfect_fraction_round(extra);
            hits += more_hits;
            samples += more_samples;
            rounds += 1;
        }
        info!(
            "final round: {hits} of {samples} samples were perfect matchings \
             of the graph ({rounds} round(s), ~{:.1}% relative error)",
            100.0 / (hits.max(1) as f64).sqrt()
        );
        if hits == 0 {
            warn!(
                "no perfect matching of the graph was ever sampled; the \
                 estimate is 0 and the chain has almost surely not mixed"
            );
        } else if hits < final_round::TARGET_HITS {
            warn!(
                "final round reached only {hits} hits after {rounds} rounds \
                 (target {}); the perfect-fraction factor is the dominant \
                 error term here",
                final_round::TARGET_HITS
            );
        }
        hits as f64 / samples.max(1) as f64
    }

    /// ln Z at beta = 0 with the exact initial table w = n: the perfect
    /// class contributes n! and each of the n^2 hole classes contributes
    /// (n-1)! * n = n!, hence Z_0 = (n^2 + 1) * n!.
    fn ln_z0(&self) -> f64 {
        ((self.size * self.size + 1) as f64).ln()
            + (1..=self.size).map(|k| (k as f64).ln()).sum::<f64>()
    }

    /// Run the cooling schedule, invoking `observer` after every step with
    /// the step statistics and the current global state. The observer returns
    /// whether to continue; returning false stops the annealing early (used
    /// by the TUI when the user quits).
    pub fn cooling_evolve_with<F>(&mut self, sequence: CoolingSchedule, mut observer: F) -> f64
    where
        F: FnMut(&StepStats, &State) -> bool,
    {
        let total_steps = sequence.total_steps();
        let mut ln_z = self.ln_z0();
        for (index, beta) in sequence.skip(1).enumerate() {
            let stats = self.evolve(beta);
            if stats.ratio < 0.3 {
                // A single-step ratio this small means the sampled ensemble
                // says Z lost most of its mass in one schedule tick — with a
                // sound schedule that is a symptom of chains lagging the
                // schedule, and it poisons the telescoping product.
                warn!(
                    "step {} (beta {:.4} -> {:.4}): ratio collapsed to {:.3e} \
                     ({} of {} samples fully-active perfect)",
                    index + 1,
                    self.global_state.beta(),
                    beta,
                    stats.ratio,
                    stats.perfect_active_samples,
                    stats.total_samples,
                );
            }
            ln_z += stats.ratio.max(f64::MIN_POSITIVE).ln();
            self.global_state.set_beta(beta);
            // Z * Pr[perfect and fully active] estimates per(G) at *every*
            // beta; it is 0 early (the event is never sampled) and converges
            // as the Gibbs mass concentrates on the real graph's matchings.
            let estimator = ln_z.exp() * stats.perfect_active_samples as f64
                / stats.total_samples.max(1) as f64;
            let step = StepStats {
                step: index + 1,
                total_steps,
                beta,
                ratio: stats.ratio,
                estimator,
                accepted_samples: stats.perfect_active_samples as f64,
                attempted_samples: stats.total_samples,
            };
            if !observer(&step, &self.global_state) {
                return estimator;
            }
        }
        ln_z.exp() * self.estimate_perfect_fraction()
    }

    pub fn cooling_evolve(&mut self, sequence: CoolingSchedule) -> f64 {
        self.cooling_evolve_with(sequence, |step, _| {
            info!(
                "beta = {:.5}, ratio: {:.5}, estimator: {:.5e}",
                step.beta, step.ratio, step.estimator
            );
            true
        })
    }
}

#[cfg(test)]
mod test {
    use std::{num::NonZeroUsize, path::PathBuf, time::Instant};

    use crate::{cooling_schedule::CoolingConfig, graph::Graph};

    fn estimate(name: &str, config: super::Config) -> (f64, f64) {
        let path: PathBuf = env!("CARGO_MANIFEST_DIR").into();
        let path = path.join("data").join(name);
        let graph = Graph::load(path).unwrap();
        let exact = crate::exact::to_f64(&crate::exact::permanent(&graph));
        let mut state = super::MCState::new(graph, config);
        state.warmup();
        let cooling_cfg = CoolingConfig {
            n: NonZeroUsize::new(state.size).unwrap(),
            additive_ratio: NonZeroUsize::new(4).unwrap(),
            multiplicative_ratio: NonZeroUsize::new(4).unwrap(),
        };
        let schedule = crate::cooling_schedule::CoolingSchedule::from(cooling_cfg);
        let estimator = state.cooling_evolve(schedule);
        (estimator, exact)
    }

    fn light_config() -> super::Config {
        super::Config {
            num_of_chains: 256,
            warmup_times: 2048,
            weight_sample_intervals: 8,
            estimator_sample_intervals: 16,
            num_of_weight_estimations: 256,
            num_of_estimator_estimations: 16,
        }
    }

    /// Manual, ignored phase probe used while evaluating accelerator designs.
    /// Environment variables make it possible to isolate warmup, weight
    /// estimation, and estimator sampling without changing production CLI
    /// behavior. Run with `cargo test --release profile_phases -- --ignored
    /// --nocapture`.
    #[test]
    #[ignore]
    fn profile_phases() {
        fn env_usize(name: &str, default: usize) -> usize {
            std::env::var(name)
                .ok()
                .and_then(|value| value.parse().ok())
                .unwrap_or(default)
        }

        let graph_path =
            std::env::var("PROFILE_GRAPH").unwrap_or_else(|_| "data/grid-8x8.json".to_owned());
        let graph = Graph::load(graph_path).unwrap();
        let config = super::Config {
            num_of_chains: env_usize("PROFILE_CHAINS", 2048),
            warmup_times: env_usize("PROFILE_WARMUP", 0),
            weight_sample_intervals: env_usize("PROFILE_WEIGHT_INTERVAL", 16),
            estimator_sample_intervals: env_usize("PROFILE_ESTIMATOR_INTERVAL", 128),
            num_of_weight_estimations: env_usize("PROFILE_WEIGHT_SAMPLES", 0),
            num_of_estimator_estimations: env_usize("PROFILE_ESTIMATOR_SAMPLES", 0),
        };
        let n = graph.size;
        let profile_steps = env_usize("PROFILE_STEPS", 1);
        let mut state = super::MCState::new(graph, config);

        let started = Instant::now();
        state.warmup();
        let warmup_elapsed = started.elapsed();

        let schedule = super::CoolingSchedule::from(CoolingConfig {
            n: NonZeroUsize::new(n).unwrap(),
            additive_ratio: NonZeroUsize::new(1).unwrap(),
            multiplicative_ratio: NonZeroUsize::new(1).unwrap(),
        });
        let started = Instant::now();
        let mut observed_steps = 0usize;
        let mut perfect_fraction_sum = 0.0;
        state.cooling_evolve_with(schedule, |step, _| {
            if step.attempted_samples > 0 {
                perfect_fraction_sum += step.accepted_samples / step.attempted_samples as f64;
            }
            observed_steps += 1;
            observed_steps < profile_steps
        });
        let evolve_elapsed = started.elapsed();

        println!(
            "n={n} config={config:?} steps={observed_steps} warmup={warmup_elapsed:?} \
             evolve={evolve_elapsed:?} mean_perfect_fraction={:.6}",
            perfect_fraction_sum / observed_steps.max(1) as f64
        );
    }

    #[test]
    fn four_cycles_example() {
        let (estimator, exact) = estimate("4-cycles.json", light_config());
        println!("estimator: {estimator}, exact: {exact}");
        assert!(
            (estimator / exact).ln().abs() < 0.5f64.ln().abs(),
            "estimator {estimator} too far from exact {exact}"
        );
    }

    #[test]
    fn box_example() {
        let (estimator, exact) = estimate("box.json", light_config());
        println!("estimator: {estimator}, exact: {exact}");
        assert!(
            (estimator / exact).ln().abs() < 0.5f64.ln().abs(),
            "estimator {estimator} too far from exact {exact}"
        );
    }

    /// A single 16-cycle has exactly two perfect matchings, which differ on
    /// every edge. This is the family where a swap-only walker on full
    /// pairings freezes: moving between the two matchings requires opening a
    /// hole and sliding it around the cycle, which only the JSV move set can
    /// do. The old transposition chain had no reliable path here; the full
    /// scheme must handle it. n = 16 with only two matchings needs more
    /// stirring than the tiny n = 4 and n = 8 examples, hence the heavier
    /// config.
    #[test]
    fn cycle_example() {
        let config = super::Config {
            num_of_chains: 512,
            warmup_times: 4096,
            weight_sample_intervals: 16,
            estimator_sample_intervals: 64,
            num_of_weight_estimations: 512,
            num_of_estimator_estimations: 32,
        };
        let (estimator, exact) = estimate("cycle.json", config);
        println!("estimator: {estimator}, exact: {exact}");
        assert!(
            (estimator / exact).ln().abs() < 0.5f64.ln().abs(),
            "estimator {estimator} too far from exact {exact}"
        );
    }
}
