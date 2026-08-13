use crate::chain::AugmentedMatch;
use crate::cooling_schedule::CoolingSchedule;
use crate::cooling_state::{Matrix, State};
use crate::graph;
use crate::graph::Match;
use rand::SeedableRng;
use rand::rngs::SmallRng;
use rayon::iter::{IntoParallelRefMutIterator, ParallelIterator};
use tracing::info;

#[derive(Debug, Clone, Copy)]
pub struct Config {
    /// number of chains
    pub num_of_chains: usize,
    /// potential mixing time of initial runs
    pub warmup_times: usize,
    /// potential relaxation time of the chain
    pub weight_sample_intervals: usize,
    /// potential relaxation time of the chain
    pub estimator_sample_intervals: usize,
    /// number of samples to from each chain for weight estimation
    pub num_of_weight_estimations: usize,
    /// number of samples to from each chain for estimator estimation
    pub num_of_estimator_estimations: usize,
}

/// Result of one `evolve` step.
struct EvolveStats {
    /// estimate of Z(beta') / Z(beta)
    ratio: f64,
    /// importance-weighted count of matchings that survived rejection
    /// sampling (with zero penalty this is a plain count)
    accepted_samples: f64,
    /// number of rejection-sampling attempts across all chains
    attempted_samples: usize,
}

/// Per-step statistics reported to `cooling_evolve_with` observers.
pub struct StepStats {
    pub step: usize,
    pub total_steps: usize,
    pub beta: f64,
    pub ratio: f64,
    pub estimator: f64,
    pub accepted_samples: f64,
    pub attempted_samples: usize,
}

struct SampleCounts {
    size: usize,
    data: Vec<usize>,
}

impl SampleCounts {
    pub fn new(size: usize) -> Self {
        SampleCounts {
            size,
            data: vec![0; size * size],
        }
    }
    pub fn inc(&mut self, u: usize, v: usize) {
        self.data[u * self.size + v] += 1;
    }
    pub fn merge(mut self, other: Self) -> Self {
        for (left, right) in self.data.iter_mut().zip(other.data) {
            *left += right;
        }
        self
    }
    pub fn finish(self, state: &State) -> Matrix {
        Matrix::from_sample_counts(state, &self.data)
    }
}

pub struct MCState {
    #[allow(dead_code)]
    graph: graph::Graph,
    size: usize,
    config: Config,
    pub global_state: State,
    chains: Vec<AugmentedMatch>,
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

struct EvolveSum {
    counts: SampleCounts,
    accepted: f64,
    estimate: f64,
    attempts: usize,
}

impl EvolveSum {
    fn new(size: usize) -> Self {
        EvolveSum {
            counts: SampleCounts::new(size),
            accepted: 0.0,
            estimate: 0.0,
            attempts: 0,
        }
    }

    fn merge(self, other: Self) -> Self {
        EvolveSum {
            counts: self.counts.merge(other.counts),
            accepted: self.accepted + other.accepted,
            estimate: self.estimate + other.estimate,
            attempts: self.attempts + other.attempts,
        }
    }
}

impl MCState {
    pub fn new(graph: graph::Graph, config: Config) -> Self {
        let global_state = State::from(&graph);
        let size = graph.size;
        let chains = (0..config.num_of_chains)
            .map(|_| {
                let matching = Match::random(graph.size);
                let weight = global_state.weight_of_match(&matching);
                let active_count = global_state.active_count_of_match(&matching);
                AugmentedMatch::new(matching, weight, active_count)
            })
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
    fn evolve(&mut self, next_beta: f64, penalty: f64) -> EvolveStats {
        let diff = self.global_state.beta() - next_beta;
        let weight_lower_bound = self.global_state.matching_weight_lower_bound();
        let global_sum = self
            .chains
            .par_iter_mut()
            .fold(
                || EvolveSum::new(self.size),
                |mut local, x| {
                    let mut rng = SmallRng::from_rng(&mut rand::rng());
                    // The weight matrix was replaced at the end of the previous
                    // evolve step, so the cached per-chain weight must be
                    // refreshed here: both the Metropolis acceptance and the L/W
                    // rejection probability rely on x.weight being exactly
                    // W(matching) under the *current* matrix. Keeping the stale
                    // cache and patching it incrementally lets a per-chain offset
                    // accumulate, which biases the rejection sampler (W >= 1 no
                    // longer holds).
                    x.set_weight(self.global_state.weight_of_match(&x.matching));
                    for _ in 0..self.config.num_of_weight_estimations {
                        x.transit_n_times(
                            &self.global_state,
                            self.config.weight_sample_intervals,
                            &mut rng,
                        );
                        let sample = x.choose_weighted_edge(&self.global_state, &mut rng);
                        local.counts.inc(sample.0, sample.1);
                    }
                    for _ in 0..self.config.num_of_estimator_estimations {
                        let (sample, attempts) = x.rejection_sample(
                            &self.global_state,
                            weight_lower_bound,
                            self.config.estimator_sample_intervals,
                            &mut rng,
                        );
                        local.attempts += attempts;
                        if let Some(sample) = sample {
                            let importance = (x.active_count as f64 * penalty).exp();
                            local.accepted += importance;
                            local.estimate += (diff * sample as f64).exp() * importance;
                        }
                    }
                    local
                },
            )
            .reduce(
                || EvolveSum::new(self.size),
                |left, right| left.merge(right),
            );
        let EvolveSum {
            counts,
            accepted,
            estimate,
            attempts,
        } = global_sum;
        self.global_state.weight = counts.finish(&self.global_state);
        let ratio = if estimate >= accepted {
            1.0
        } else {
            estimate / accepted
        };
        EvolveStats {
            ratio,
            accepted_samples: accepted,
            attempted_samples: attempts,
        }
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
        // accumulate in f64: usize overflows for n >= 21
        let factorial = (1..=self.size).map(|x| x as f64).product::<f64>();
        let mut estimator = factorial;
        for (index, i) in sequence.skip(1).enumerate() {
            let stats = self.evolve(i, 0.0);
            estimator *= stats.ratio;
            self.global_state.set_beta(i);
            let step = StepStats {
                step: index + 1,
                total_steps,
                beta: i,
                ratio: stats.ratio,
                estimator,
                accepted_samples: stats.accepted_samples,
                attempted_samples: stats.attempted_samples,
            };
            if !observer(&step, &self.global_state) {
                break;
            }
        }
        estimator
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
        let exact = crate::exact::permanent(&graph);
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
        let mut observed = None;
        let mut observed_steps = 0usize;
        let mut acceptance_sum = 0.0;
        let mut acceptance_min = f64::INFINITY;
        let mut acceptance_max = 0.0f64;
        let mut acceptance_min_step = 0usize;
        state.cooling_evolve_with(schedule, |step, _| {
            observed = Some((step.accepted_samples, step.attempted_samples));
            if step.attempted_samples > 0 {
                let acceptance = step.accepted_samples / step.attempted_samples as f64;
                acceptance_sum += acceptance;
                if acceptance < acceptance_min {
                    acceptance_min = acceptance;
                    acceptance_min_step = step.step;
                }
                acceptance_max = acceptance_max.max(acceptance);
            }
            observed_steps += 1;
            observed_steps < profile_steps
        });
        let evolve_elapsed = started.elapsed();
        let acceptance_mean = if observed_steps == 0 || config.num_of_estimator_estimations == 0 {
            0.0
        } else {
            acceptance_sum / observed_steps as f64
        };

        println!(
            "n={n} config={config:?} steps={observed_steps} warmup={warmup_elapsed:?} evolve={evolve_elapsed:?} acceptance=min:{acceptance_min:.6}@{acceptance_min_step},mean:{acceptance_mean:.6},max:{acceptance_max:.6} accepted={observed:?}"
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
}
