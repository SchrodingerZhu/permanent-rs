use crate::cooling_state;
use crate::cooling_state::Matrix;
use crate::filter::{AugmentedMatch, MetropolisFilter};

pub struct Config {
    /// number of chains
    pub num_of_chains: usize,
    /// potential mixing time of initial runs
    pub warmup_times: usize,
    /// potential relaxation time of the chain
    pub sample_intervals: usize,
    /// number of samples to from each chain for each cooling step
    pub num_of_samples: usize,
    /// learning rate of the cooling schedule
    pub learning_rate: f64,
}

struct WeightEstimation<'a> {
    state: &'a cooling_state::State,
    matrix: Matrix,
    sum: f64,
}

impl<'a> WeightEstimation<'a> {
    pub fn pick(&mut self, u: usize, v: usize) {
        let w = 1.0 / self.state.weight_of_edge(u, v);
        self.matrix.add(u, v, w);
        self.sum += w;
    }
    pub fn finish(mut self) -> Matrix {
        let scale = self.matrix.dimension() as f64 / self.sum;
        self.matrix.scale(scale);
        self.matrix
    }
}
pub struct MCState<T: MetropolisFilter> {
    global_state: cooling_state::State,
    chains: Vec<AugmentedMatch<T>>,
}
