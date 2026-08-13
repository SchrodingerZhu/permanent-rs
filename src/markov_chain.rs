use crate::chain::AugmentedMatch;
use crate::cooling_schedule::CoolingSchedule;
use crate::cooling_state::{Matrix, State};
use crate::graph;
use crate::graph::Match;
use rand::SeedableRng;
use rand::rngs::SmallRng;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator};
use std::iter::Sum;
use std::sync::atomic::AtomicUsize;
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

struct AtomicMatrix {
    size: usize,
    data: Vec<AtomicUsize>,
}

impl AtomicMatrix {
    pub fn new(size: usize) -> Self {
        AtomicMatrix {
            size,
            data: (0..size * size).map(|_| AtomicUsize::new(0)).collect(),
        }
    }
    pub fn inc(&self, u: usize, v: usize) {
        self.data[u * self.size + v].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    pub fn finish(self, state: &State) -> Matrix {
        std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
        let mut matrix = Matrix::new(self.size, 0.0);
        let sum = matrix
            .par_mut_rows()
            .enumerate()
            .map(|(i, row)| {
                let mut sum = 0.0;
                for (j, item) in row.iter_mut().enumerate() {
                    let value = self.data[i * self.size + j]
                        .load(std::sync::atomic::Ordering::Relaxed)
                        .max(1) as f64;
                    let value = value / state.weight_of_edge(i, j);
                    *item = value;
                    sum += value;
                }
                sum
            })
            .sum::<f64>();
        // Cap the weights well below the scale where f64 addition starts
        // dropping the other summands of W(M) (ulp(1e12) ~ 2e-4): the chain
        // tracks W incrementally, and a cap near f64::MAX makes W collapse to
        // 0 by cancellation the moment a capped edge leaves the matching,
        // which turns the 1/W rejection probability into infinity. The cap
        // only limits how strongly unlikely edges are boosted for mixing; any
        // finite weight matrix keeps the estimator unbiased.
        const WEIGHT_CAP: f64 = 1e12;
        let n = self.size as f64;
        // algebraically 1 / (x * (n / sum)), with one fewer rounding step
        matrix.transform(|x| (sum / (x * n)).min(WEIGHT_CAP));
        matrix
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

struct AddPair(f64, f64);
impl Sum for AddPair {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(|x, y| AddPair(x.0 + y.0, x.1 + y.1))
            .unwrap_or(AddPair(0.0, 0.0))
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
        let matrix = AtomicMatrix::new(self.size);
        let diff = self.global_state.beta() - next_beta;
        let global_sum = self
            .chains
            .par_iter_mut()
            .map(|x| {
                let mut rng = SmallRng::from_rng(&mut rand::rng());
                // The weight matrix was replaced at the end of the previous
                // evolve step, so the cached per-chain weight must be
                // refreshed here: both the Metropolis acceptance and the 1/W
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
                    matrix.inc(sample.0, sample.1);
                }
                let mut local_sample_count = 0.0;
                let mut local_sum = 0.0;
                for _ in 0..self.config.num_of_estimator_estimations {
                    if let Some(sample) = x.rejection_sample(
                        &self.global_state,
                        self.config.estimator_sample_intervals,
                        &mut rng,
                    ) {
                        let importance = (x.active_count as f64 * penalty).exp();
                        local_sample_count += importance;
                        local_sum += (diff * sample as f64).exp() * importance;
                    }
                }
                AddPair(local_sample_count, local_sum)
            })
            .sum::<AddPair>();
        self.global_state.weight = matrix.finish(&self.global_state);
        let ratio = if global_sum.1 >= global_sum.0 {
            1.0
        } else {
            global_sum.1 / global_sum.0
        };
        EvolveStats {
            ratio,
            accepted_samples: global_sum.0,
            attempted_samples: self.config.num_of_estimator_estimations * self.config.num_of_chains,
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
    use std::{num::NonZeroUsize, path::PathBuf};

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
