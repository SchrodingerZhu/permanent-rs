use crate::cooling_schedule::CoolingSchedule;
use crate::cooling_state::{Matrix, State};
use crate::filter::{AugmentedMatch, MetropolisFilter};
use crate::graph::Match;
use crate::{cooling_schedule, cooling_state, graph};
use rayon::iter::{IntoParallelRefMutIterator, ParallelIterator};
use std::sync::atomic::AtomicUsize;

pub struct Config {
    /// number of chains
    pub num_of_chains: usize,
    /// potential mixing time of initial runs
    pub warmup_times: usize,
    /// potential relaxation time of the chain
    pub sample_intervals: usize,
    /// number of samples to from each chain for each cooling step
    pub num_of_samples: usize,
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
        let mut sum = 0.0;
        let mut matrix = Matrix::new(self.size, 0.0);
        for i in 0..self.size {
            for j in 0..self.size {
                let value = self.data[i * self.size + j]
                    .load(std::sync::atomic::Ordering::Relaxed)
                    .max(1) as f64;
                let value = value / state.weight_of_edge(i, j);
                matrix.set(i, j, value);
                sum += value;
            }
        }
        let scale = self.size as f64 / sum;
        matrix.transform(|x| {
            (1.0 / (x * scale)).min(f64::MAX / (4 * self.size * self.size) as f64 - f64::EPSILON)
        });
        matrix
    }
}

pub struct MCState<T: MetropolisFilter> {
    size: usize,
    config: Config,
    global_state: State,
    chains: Vec<AugmentedMatch<T>>,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            num_of_chains: 1024,
            warmup_times: 16384,
            sample_intervals: 8,
            num_of_samples: 2048,
        }
    }
}

impl<T: MetropolisFilter + 'static + Send + Sync> MCState<T> {
    pub fn new(graph: &graph::Graph, config: Config) -> Self {
        let global_state = State::from(graph);
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
            config,
            global_state,
            chains,
            size: graph.size,
        }
    }
    pub fn warmup(&mut self) {
        self.chains.par_iter_mut().for_each(|x| {
            x.transit_n_times(&self.global_state, self.config.warmup_times);
        });
    }
    fn evolve(&mut self, next_beta: f64, recompute: bool) -> f64 {
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
                let mut local_sum = 0.0;
                for _ in 0..self.config.num_of_samples {
                    x.transit_n_times(&self.global_state, self.config.sample_intervals);
                    let sample = x.choose_weighted_edge(&self.global_state);
                    matrix.inc(sample.0, sample.1);
                    let non_edges = self.size - x.active_count;
                    local_sum += (diff * non_edges as f64).exp();
                }
                local_sum
            })
            .sum::<f64>();
        self.global_state.weight = matrix.finish(&self.global_state);
        global_sum / self.config.num_of_chains as f64 / self.config.num_of_samples as f64
    }
    pub fn cooling_evolve(&mut self, mut sequence: CoolingSchedule, recompute: bool) -> f64 {
        let mut estimator = (1..=self.size).product::<usize>() as f64;
        sequence.next();
        for i in sequence {
            println!("beta = {}", self.global_state.beta);
            let ratio = self.evolve(i, recompute);
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
        let path = path.join("data").join("choice.json");
        let graph = Graph::load(path).unwrap();
        println!("{:?}", graph);
        let config = super::Config::default();
        let mut state = super::MCState::<crate::filter::Additive>::new(&graph, config);
        state.warmup();
        println!("warmup done");
        let cooling_cfg = CoolingConfig {
            n: NonZeroUsize::new(graph.size).unwrap(),
            additive_ratio: NonZeroUsize::new(4).unwrap(),
            multiplicative_ratio: NonZeroUsize::new(4).unwrap(),
        };
        let schedule = crate::cooling_schedule::CoolingSchedule::from(cooling_cfg);
        state.cooling_evolve(schedule, false);
        for i in 0..graph.size {
            for j in 0..graph.size {
                // print state.global_state.weight.get(i, j)
                print!("{:.2} ", 1.0/state.global_state.weight.get(i, j));
            }
            println!();
        }
    }
}
