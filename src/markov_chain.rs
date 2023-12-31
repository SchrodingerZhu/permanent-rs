use crate::cooling_schedule::CoolingSchedule;
use crate::cooling_state::{Matrix, State};
use crate::filter::{AugmentedMatch, MetropolisFilter};
use crate::graph;
use crate::graph::Match;
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
        let scale = self.size as f64 / sum;
        matrix.transform(|x| (1.0 / (x * scale)).min(f64::MAX / ((2 * self.size) as f64)));
        matrix
    }
}

pub struct MCState<T: MetropolisFilter> {
    #[allow(dead_code)]
    graph: graph::Graph,
    size: usize,
    config: Config,
    pub global_state: State,
    chains: Vec<AugmentedMatch<T>>,
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

impl<T: MetropolisFilter + 'static + Send + Sync> MCState<T> {
    pub fn new(graph: graph::Graph, config: Config) -> Self {
        let global_state = State::from(&graph);
        let size = graph.size;
        let chains = (0..config.num_of_chains)
            .map(|_| {
                let matching = Match::random(graph.size);
                let attr = T::initial_attr(&matching, &global_state);
                let weight = global_state.weight_of_match(&matching);
                let active_count = global_state.active_count_of_match(&matching);
                AugmentedMatch {
                    matching,
                    attr,
                    weight,
                    active_count,
                }
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
            x.transit_n_times(&self.global_state, self.config.warmup_times);
        });
    }
    fn evolve(&mut self, next_beta: f64, recompute: bool, penalty: f64) -> f64 {
        let matrix = AtomicMatrix::new(self.size);
        let diff = self.global_state.beta - next_beta;
        let global_sum = self
            .chains
            .par_iter_mut()
            .map(|x| {
                if recompute {
                    x.weight = self.global_state.weight_of_match(&x.matching);
                    x.attr = T::initial_attr(&x.matching, &self.global_state);
                }
                for _ in 0..self.config.num_of_weight_estimations {
                    x.transit_n_times(&self.global_state, self.config.weight_sample_intervals);
                    let sample = x.choose_weighted_edge(&self.global_state);
                    matrix.inc(sample.0, sample.1);
                }
                let mut local_sample_count = 0.0;
                let mut local_sum = 0.0;
                for _ in 0..self.config.num_of_estimator_estimations {
                    if let Some(sample) = x.rejection_sample(
                        &self.global_state,
                        self.config.estimator_sample_intervals,
                    ) {
                        let importance = (x.active_count as f64 * penalty).exp();
                        local_sample_count += importance;
                        local_sum += (diff * sample as f64).exp() * importance as f64;
                    }
                }
                AddPair(local_sample_count, local_sum)
            })
            .sum::<AddPair>();
        self.global_state.weight = matrix.finish(&self.global_state);
        if global_sum.1 >= global_sum.0 {
            1.0
        } else {
            global_sum.1 / global_sum.0
        }
    }
    pub fn cooling_evolve(&mut self, mut sequence: CoolingSchedule, recompute: bool) -> f64 {
        let factorial = (1..=self.size).product::<usize>() as f64;
        let mut estimator = factorial;
        sequence.next();
        for i in sequence {
            let ratio = self.evolve(i, recompute, 0.0);
            info!(
                "beta = {:.5}, estimator: {:.5}, ratio: {:.5}",
                self.global_state.beta, estimator, ratio
            );
            estimator *= ratio;
            self.global_state.beta = i;
        }
        estimator
    }
}

#[cfg(test)]
mod test {
    use std::{num::NonZeroUsize, path::PathBuf};

    use crate::{cooling_schedule::CoolingConfig, graph::Graph};

    #[test]
    fn box_example() {
        let path: PathBuf = env!("PWD").into();
        let path = path.join("data").join("4-cycles.json");
        let graph = Graph::load(path).unwrap();
        println!("{:?}", graph);
        let config = super::Config::default();
        let mut state = super::MCState::<crate::filter::Additive>::new(graph, config);
        state.warmup();
        println!("warmup done");
        let size = state.size;
        let cooling_cfg = CoolingConfig {
            n: NonZeroUsize::new(size).unwrap(),
            additive_ratio: NonZeroUsize::new(16).unwrap(),
            multiplicative_ratio: NonZeroUsize::new(16).unwrap(),
        };
        let schedule = crate::cooling_schedule::CoolingSchedule::from(cooling_cfg);
        state.cooling_evolve(schedule, false);
        for i in 0..size {
            for j in 0..size {
                // print state.global_state.weight.get(i, j)
                print!("{:.2} ", 1.0 / state.global_state.weight.get(i, j));
            }
            println!();
        }
    }
}
