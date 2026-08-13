use std::time::Instant;

use cubecl::prelude::*;
use cubecl::server::Handle;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use tracing::info;

use crate::cooling_schedule::CoolingSchedule;
use crate::cooling_state::{Matrix, State};
use crate::graph::{Graph, Match};
use crate::markov_chain::{Config, StepStats};

const CUBE_UNITS: u32 = 32;
const LOWER_BOUND_SAFETY: f32 = 1.0 - 2.0e-6;
const MAX_ESTIMATOR_ATTEMPTS_PER_DISPATCH: usize = 64;

#[derive(CubeType, Clone, Copy)]
struct CompensatedF32 {
    sum: f32,
    correction: f32,
}

#[cube]
impl CompensatedF32 {
    fn value(&self) -> f32 {
        self.sum + self.correction
    }

    fn add(&mut self, value: f32) {
        let next = self.sum + value;
        self.correction += if self.sum.abs() >= value.abs() {
            (self.sum - next) + value
        } else {
            (value - next) + self.sum
        };
        self.sum = next;
    }
}

#[cube]
fn xorshift32(mut value: u32) -> u32 {
    value = value ^ (value << 13);
    value = value ^ (value >> 17);
    value ^ (value << 5)
}

#[cube]
fn uniform_f32(value: u32) -> f32 {
    // Keep the result in [0, 1). Casting all 32 bits can round u32::MAX to
    // 2^32 in f32 and produce exactly 1.
    f32::cast_from(value >> 8) * (1.0 / 16_777_216.0)
}

/// One invocation owns one chain. Transitions within a chain are dependent
/// and therefore serial; independent chains supply GPU parallelism.
#[cube(launch_unchecked)]
fn warmup_kernel(
    permutations: &mut Array<u32>,
    weights: &Array<f32>,
    adjacency: &Array<u32>,
    exp_beta: &Array<f32>,
    totals: &mut Array<f32>,
    corrections: &mut Array<f32>,
    active_counts: &mut Array<u32>,
    rng_states: &mut Array<u32>,
    n: usize,
    iterations: usize,
) {
    let chain = ABSOLUTE_POS;
    if chain >= totals.len() {
        terminate!();
    }

    let base = chain * n;
    let mut total = CompensatedF32 {
        sum: totals[chain],
        correction: corrections[chain],
    };
    let mut active_count = active_counts[chain];
    let mut rng = rng_states[chain];

    for _ in 0..iterations {
        rng = xorshift32(rng);
        let first = (rng as usize) % n;
        rng = xorshift32(rng);
        let mut second = (rng as usize) % (n - 1);
        if second >= first {
            second += 1;
        }

        let v1 = permutations[base + first] as usize;
        let v2 = permutations[base + second] as usize;
        let old1 = first * n + v1;
        let old2 = second * n + v2;
        let new1 = first * n + v2;
        let new2 = second * n + v1;
        let weight = total.value();
        let mut next_total = total;
        next_total.add(-weights[old1]);
        next_total.add(-weights[old2]);
        next_total.add(weights[new1]);
        next_total.add(weights[new2]);
        let next_active =
            active_count - adjacency[old1] - adjacency[old2] + adjacency[new1] + adjacency[new2];
        let delta = next_active as i32 - active_count as i32;
        let probability = (next_total.value() / weight) * exp_beta[(delta + 2) as usize];

        rng = xorshift32(rng);
        if probability >= 1.0 || uniform_f32(rng) < probability {
            permutations[base + first] = v2 as u32;
            permutations[base + second] = v1 as u32;
            total.sum = next_total.sum;
            total.correction = next_total.correction;
            active_count = next_active;
        }
    }

    totals[chain] = total.sum;
    corrections[chain] = total.correction;
    active_counts[chain] = active_count;
    rng_states[chain] = rng;
}

/// One chunk of a cooling step. The first launch collects the weight samples;
/// subsequent launches pass `weight_samples = 0` and continue the rejection
/// sampler. Chunking avoids a data-dependent shader `break` (which currently
/// crashes CubeCL's SPIR-V lowering on RADV) without making an accepted chain
/// spin through the full O(n^2) rejection cap.
#[cube(launch_unchecked)]
fn evolve_kernel(
    permutations: &mut Array<u32>,
    weights: &Array<f32>,
    adjacency: &Array<u32>,
    exp_beta: &Array<f32>,
    ratio_terms: &Array<f32>,
    totals: &mut Array<f32>,
    corrections: &mut Array<f32>,
    active_counts: &mut Array<u32>,
    rng_states: &mut Array<u32>,
    histogram: &mut Array<Atomic<u32>>,
    accepted_output: &mut Array<u32>,
    estimate_output: &mut Array<f32>,
    estimate_correction_output: &mut Array<f32>,
    attempted_output: &mut Array<u32>,
    completed_output: &mut Array<u32>,
    current_attempt_output: &mut Array<u32>,
    n: usize,
    weight_samples: usize,
    weight_interval: usize,
    estimator_samples: usize,
    estimator_interval: usize,
    estimator_attempt_budget: usize,
    max_attempts: usize,
    lower_bound: f32,
) {
    let chain = ABSOLUTE_POS;
    if chain >= totals.len() {
        terminate!();
    }

    let base = chain * n;
    // The matrix changes after every cooling step, so refresh W(M) from the
    // new f32 matrix before using incremental updates again.
    let mut total = CompensatedF32 {
        sum: 0.0,
        correction: 0.0,
    };
    for row in 0..n {
        total.add(weights[row * n + permutations[base + row] as usize]);
    }
    let mut active_count = active_counts[chain];
    let mut rng = rng_states[chain];

    for _ in 0..weight_samples {
        for _ in 0..weight_interval {
            rng = xorshift32(rng);
            let first = (rng as usize) % n;
            rng = xorshift32(rng);
            let mut second = (rng as usize) % (n - 1);
            if second >= first {
                second += 1;
            }

            let v1 = permutations[base + first] as usize;
            let v2 = permutations[base + second] as usize;
            let old1 = first * n + v1;
            let old2 = second * n + v2;
            let new1 = first * n + v2;
            let new2 = second * n + v1;
            let weight = total.value();
            let mut next_total = total;
            next_total.add(-weights[old1]);
            next_total.add(-weights[old2]);
            next_total.add(weights[new1]);
            next_total.add(weights[new2]);
            let next_active = active_count - adjacency[old1] - adjacency[old2]
                + adjacency[new1]
                + adjacency[new2];
            let delta = next_active as i32 - active_count as i32;
            let probability = (next_total.value() / weight) * exp_beta[(delta + 2) as usize];

            rng = xorshift32(rng);
            if probability >= 1.0 || uniform_f32(rng) < probability {
                permutations[base + first] = v2 as u32;
                permutations[base + second] = v1 as u32;
                total.sum = next_total.sum;
                total.correction = next_total.correction;
                active_count = next_active;
            }
        }

        rng = xorshift32(rng);
        let mut target = uniform_f32(rng) * total.value();
        let last_row = n - 1;
        let mut selected = last_row * n + permutations[base + last_row] as usize;
        let mut found = false;
        for row in 0..n {
            let index = row * n + permutations[base + row] as usize;
            target -= weights[index];
            if !found && target <= 0.0 {
                selected = index;
                found = true;
            }
        }
        histogram[selected].fetch_add(1);
    }

    let mut accepted = accepted_output[chain];
    let mut attempted = attempted_output[chain];
    let mut completed = completed_output[chain];
    let mut current_attempt = current_attempt_output[chain];
    let mut estimate = CompensatedF32 {
        sum: estimate_output[chain],
        correction: estimate_correction_output[chain],
    };
    for _ in 0..estimator_attempt_budget {
        if (completed as usize) < estimator_samples {
            for _ in 0..estimator_interval {
                rng = xorshift32(rng);
                let first = (rng as usize) % n;
                rng = xorshift32(rng);
                let mut second = (rng as usize) % (n - 1);
                if second >= first {
                    second += 1;
                }

                let v1 = permutations[base + first] as usize;
                let v2 = permutations[base + second] as usize;
                let old1 = first * n + v1;
                let old2 = second * n + v2;
                let new1 = first * n + v2;
                let new2 = second * n + v1;
                let weight = total.value();
                let mut next_total = total;
                next_total.add(-weights[old1]);
                next_total.add(-weights[old2]);
                next_total.add(weights[new1]);
                next_total.add(weights[new2]);
                let next_active = active_count - adjacency[old1] - adjacency[old2]
                    + adjacency[new1]
                    + adjacency[new2];
                let delta = next_active as i32 - active_count as i32;
                let probability = (next_total.value() / weight) * exp_beta[(delta + 2) as usize];

                rng = xorshift32(rng);
                if probability >= 1.0 || uniform_f32(rng) < probability {
                    permutations[base + first] = v2 as u32;
                    permutations[base + second] = v1 as u32;
                    total.sum = next_total.sum;
                    total.correction = next_total.correction;
                    active_count = next_active;
                }
            }

            attempted += 1;
            current_attempt += 1;
            rng = xorshift32(rng);
            if uniform_f32(rng) < lower_bound / total.value() {
                accepted += 1;
                completed += 1;
                current_attempt = 0;
                estimate.add(ratio_terms[n - active_count as usize]);
            } else if current_attempt as usize == max_attempts {
                // Match the CPU backend's finite rejection cap: this requested
                // sample contributes no observation, then the next one starts.
                completed += 1;
                current_attempt = 0;
            }
        }
    }

    totals[chain] = total.sum;
    corrections[chain] = total.correction;
    active_counts[chain] = active_count;
    rng_states[chain] = rng;
    accepted_output[chain] = accepted;
    estimate_output[chain] = estimate.sum;
    estimate_correction_output[chain] = estimate.correction;
    attempted_output[chain] = attempted;
    completed_output[chain] = completed;
    current_attempt_output[chain] = current_attempt;
}

struct GpuEvolveStats {
    ratio: f64,
    accepted_samples: f64,
    attempted_samples: usize,
}

/// GPU implementation of the same annealing estimator as `MCState`.
///
/// Permutations, RNGs, active counts, and compensated weights stay resident on
/// the device. The host retains the small global weight matrix so the existing
/// observer/TUI and Hungarian rejection bound remain shared with the CPU path.
pub struct GpuMCState {
    size: usize,
    config: Config,
    pub global_state: State,
    client: ComputeClient<WgpuRuntime>,
    permutations: Handle,
    adjacency: Handle,
    totals: Handle,
    corrections: Handle,
    active_counts: Handle,
    rng_states: Handle,
}

impl GpuMCState {
    pub fn new(graph: Graph, config: Config) -> Self {
        let size = graph.size;
        assert!(size >= 2, "GPU Markov chains require n >= 2");
        assert!(config.num_of_chains > 0, "at least one chain is required");
        assert!(size <= u32::MAX as usize);
        assert!(
            config.num_of_chains <= u32::MAX as usize
                && config.num_of_chains.div_ceil(CUBE_UNITS as usize) <= u16::MAX as usize,
            "too many chains for one GPU dispatch"
        );
        assert!(
            config
                .num_of_estimator_estimations
                .checked_mul(2 * size * size)
                .is_some_and(|attempts| attempts <= u32::MAX as usize),
            "per-chain rejection-attempt counter would overflow u32"
        );
        assert!(
            config
                .num_of_chains
                .checked_mul(config.num_of_weight_estimations)
                .is_some_and(|samples| samples <= u32::MAX as usize),
            "GPU histogram counters could overflow u32"
        );

        let global_state = State::from(&graph);
        let mut adjacency_values = vec![0u32; size * size];
        for (row, edges) in graph.edges.iter().enumerate() {
            for &column in edges.iter() {
                adjacency_values[row * size + column] = 1;
            }
        }

        let mut permutations = Vec::with_capacity(config.num_of_chains * size);
        let mut totals = Vec::with_capacity(config.num_of_chains);
        let mut active_counts = Vec::with_capacity(config.num_of_chains);
        for _ in 0..config.num_of_chains {
            let matching = Match::random(size);
            permutations.extend(matching.edges.iter().map(|&(_, column)| column as u32));
            totals.push(global_state.weight_of_match(&matching) as f32);
            active_counts.push(global_state.active_count_of_match(&matching) as u32);
        }
        let rng_states = (0..config.num_of_chains)
            .map(|chain| (chain as u32 + 1).wrapping_mul(0x9e37_79b9))
            .collect::<Vec<_>>();

        let started = Instant::now();
        let client = WgpuRuntime::client(&WgpuDevice::DefaultDevice);
        info!(
            "GPU runtime {} initialized in {:?}: {:#?}",
            WgpuRuntime::name(&client),
            started.elapsed(),
            client.properties().hardware
        );

        GpuMCState {
            size,
            config,
            permutations: client.create_from_slice(u32::as_bytes(&permutations)),
            adjacency: client.create_from_slice(u32::as_bytes(&adjacency_values)),
            totals: client.create_from_slice(f32::as_bytes(&totals)),
            corrections: client.create_from_slice(f32::as_bytes(&vec![0.0; config.num_of_chains])),
            active_counts: client.create_from_slice(u32::as_bytes(&active_counts)),
            rng_states: client.create_from_slice(u32::as_bytes(&rng_states)),
            client,
            global_state,
        }
    }

    fn cube_count(&self) -> CubeCount {
        CubeCount::Static(
            (self.config.num_of_chains as u32).div_ceil(CUBE_UNITS),
            1,
            1,
        )
    }

    fn weight_values(&self) -> Vec<f32> {
        (0..self.size * self.size)
            .map(|index| {
                self.global_state
                    .weight_of_edge(index / self.size, index % self.size) as f32
            })
            .collect()
    }

    fn exp_beta_values(&self) -> Vec<f32> {
        let beta = self.global_state.beta() as f32;
        (-2..=2).map(|delta| (beta * delta as f32).exp()).collect()
    }

    fn lower_bound_for_f32_weights(&self, weights: &[f32]) -> f32 {
        let mut matrix = Matrix::new(self.size, 0.0);
        for (index, &weight) in weights.iter().enumerate() {
            matrix.set(index / self.size, index % self.size, weight as f64);
        }
        let exact = matrix.matching_weight_lower_bound();
        let mut rounded = exact as f32;
        if rounded as f64 > exact {
            rounded = rounded.next_down();
        }
        rounded * LOWER_BOUND_SAFETY
    }

    pub fn warmup(&mut self) {
        if self.config.warmup_times == 0 {
            return;
        }
        let weights = self.weight_values();
        let exp_beta = self.exp_beta_values();
        let weights = self.client.create_from_slice(f32::as_bytes(&weights));
        let exp_beta = self.client.create_from_slice(f32::as_bytes(&exp_beta));
        unsafe {
            warmup_kernel::launch_unchecked(
                &self.client,
                self.cube_count(),
                CubeDim::new_1d(CUBE_UNITS),
                ArrayArg::from_raw_parts(
                    self.permutations.clone(),
                    self.config.num_of_chains * self.size,
                ),
                ArrayArg::from_raw_parts(weights, self.size * self.size),
                ArrayArg::from_raw_parts(self.adjacency.clone(), self.size * self.size),
                ArrayArg::from_raw_parts(exp_beta, 5),
                ArrayArg::from_raw_parts(self.totals.clone(), self.config.num_of_chains),
                ArrayArg::from_raw_parts(self.corrections.clone(), self.config.num_of_chains),
                ArrayArg::from_raw_parts(self.active_counts.clone(), self.config.num_of_chains),
                ArrayArg::from_raw_parts(self.rng_states.clone(), self.config.num_of_chains),
                self.size,
                self.config.warmup_times,
            );
        }
        // Make `warmup` synchronous, matching the CPU API and its progress log.
        self.client.read_one_unchecked(self.totals.clone());
    }

    fn evolve(&mut self, next_beta: f64) -> GpuEvolveStats {
        let weights = self.weight_values();
        let lower_bound = self.lower_bound_for_f32_weights(&weights);
        let exp_beta = self.exp_beta_values();
        let diff = (self.global_state.beta() - next_beta) as f32;
        let ratio_terms = (0..=self.size)
            .map(|missing| (diff * missing as f32).exp())
            .collect::<Vec<_>>();

        let weights = self.client.create_from_slice(f32::as_bytes(&weights));
        let exp_beta = self.client.create_from_slice(f32::as_bytes(&exp_beta));
        let ratio_terms = self.client.create_from_slice(f32::as_bytes(&ratio_terms));
        let histogram = self
            .client
            .create_from_slice(u32::as_bytes(&vec![0; self.size * self.size]));
        let accepted = self
            .client
            .create_from_slice(u32::as_bytes(&vec![0; self.config.num_of_chains]));
        let estimate = self
            .client
            .create_from_slice(f32::as_bytes(&vec![0.0; self.config.num_of_chains]));
        let estimate_correction = self
            .client
            .create_from_slice(f32::as_bytes(&vec![0.0; self.config.num_of_chains]));
        let attempted =
            self.client
                .create_from_slice(u32::as_bytes(&vec![0; self.config.num_of_chains]));
        let completed =
            self.client
                .create_from_slice(u32::as_bytes(&vec![0; self.config.num_of_chains]));
        let current_attempt =
            self.client
                .create_from_slice(u32::as_bytes(&vec![0; self.config.num_of_chains]));

        let max_attempts = 2 * self.size * self.size;
        let estimator_attempt_budget = if self.config.num_of_estimator_estimations == 0 {
            0
        } else {
            self.config
                .num_of_estimator_estimations
                .clamp(8, MAX_ESTIMATOR_ATTEMPTS_PER_DISPATCH)
        };
        let max_dispatches = if estimator_attempt_budget == 0 {
            1
        } else {
            (self.config.num_of_estimator_estimations * max_attempts)
                .div_ceil(estimator_attempt_budget)
        };
        let completed_goal = self.config.num_of_estimator_estimations as u32;
        let mut weight_samples = self.config.num_of_weight_estimations;
        for dispatch in 0..max_dispatches {
            unsafe {
                evolve_kernel::launch_unchecked(
                    &self.client,
                    self.cube_count(),
                    CubeDim::new_1d(CUBE_UNITS),
                    ArrayArg::from_raw_parts(
                        self.permutations.clone(),
                        self.config.num_of_chains * self.size,
                    ),
                    ArrayArg::from_raw_parts(weights.clone(), self.size * self.size),
                    ArrayArg::from_raw_parts(self.adjacency.clone(), self.size * self.size),
                    ArrayArg::from_raw_parts(exp_beta.clone(), 5),
                    ArrayArg::from_raw_parts(ratio_terms.clone(), self.size + 1),
                    ArrayArg::from_raw_parts(self.totals.clone(), self.config.num_of_chains),
                    ArrayArg::from_raw_parts(self.corrections.clone(), self.config.num_of_chains),
                    ArrayArg::from_raw_parts(self.active_counts.clone(), self.config.num_of_chains),
                    ArrayArg::from_raw_parts(self.rng_states.clone(), self.config.num_of_chains),
                    ArrayArg::from_raw_parts(histogram.clone(), self.size * self.size),
                    ArrayArg::from_raw_parts(accepted.clone(), self.config.num_of_chains),
                    ArrayArg::from_raw_parts(estimate.clone(), self.config.num_of_chains),
                    ArrayArg::from_raw_parts(
                        estimate_correction.clone(),
                        self.config.num_of_chains,
                    ),
                    ArrayArg::from_raw_parts(attempted.clone(), self.config.num_of_chains),
                    ArrayArg::from_raw_parts(completed.clone(), self.config.num_of_chains),
                    ArrayArg::from_raw_parts(current_attempt.clone(), self.config.num_of_chains),
                    self.size,
                    weight_samples,
                    self.config.weight_sample_intervals,
                    self.config.num_of_estimator_estimations,
                    self.config.estimator_sample_intervals,
                    estimator_attempt_budget,
                    max_attempts,
                    lower_bound,
                );
            }
            weight_samples = 0;

            if estimator_attempt_budget == 0 {
                break;
            }
            let completed_bytes = self.client.read_one_unchecked(completed.clone());
            let completed_values = u32::from_bytes(&completed_bytes);
            if completed_values
                .iter()
                .all(|&value| value == completed_goal)
            {
                break;
            }
            assert!(
                dispatch + 1 < max_dispatches,
                "GPU rejection sampler exceeded its finite attempt cap"
            );
        }

        let outputs = self.client.read(vec![
            histogram,
            accepted,
            estimate,
            estimate_correction,
            attempted,
            completed,
        ]);
        let histogram = u32::from_bytes(&outputs[0]);
        let accepted = u32::from_bytes(&outputs[1]);
        let estimates = f32::from_bytes(&outputs[2]);
        let estimate_corrections = f32::from_bytes(&outputs[3]);
        let attempted = u32::from_bytes(&outputs[4]);
        let completed = u32::from_bytes(&outputs[5]);
        assert_eq!(
            histogram.iter().map(|&value| value as u64).sum::<u64>(),
            (self.config.num_of_chains * self.config.num_of_weight_estimations) as u64,
            "GPU weight-sample histogram lost updates"
        );

        let counts = histogram
            .iter()
            .map(|&value| value as usize)
            .collect::<Vec<_>>();
        self.global_state.weight = Matrix::from_sample_counts(&self.global_state, &counts);
        assert!(
            completed.iter().all(|&value| value == completed_goal),
            "GPU rejection sampler did not finish every requested sample"
        );
        let accepted_samples = accepted.iter().map(|&value| value as f64).sum::<f64>();
        let estimate_sum = estimates
            .iter()
            .zip(estimate_corrections)
            .map(|(&sum, &correction)| sum as f64 + correction as f64)
            .sum::<f64>();
        let attempted_samples = attempted.iter().map(|&value| value as usize).sum::<usize>();
        let ratio = if estimate_sum >= accepted_samples {
            1.0
        } else {
            estimate_sum / accepted_samples
        };
        GpuEvolveStats {
            ratio,
            accepted_samples,
            attempted_samples,
        }
    }

    pub fn cooling_evolve_with<F>(&mut self, sequence: CoolingSchedule, mut observer: F) -> f64
    where
        F: FnMut(&StepStats, &State) -> bool,
    {
        let total_steps = sequence.total_steps();
        let factorial = (1..=self.size).map(|value| value as f64).product::<f64>();
        let mut estimator = factorial;
        for (index, beta) in sequence.skip(1).enumerate() {
            let stats = self.evolve(beta);
            estimator *= stats.ratio;
            self.global_state.set_beta(beta);
            let step = StepStats {
                step: index + 1,
                total_steps,
                beta,
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
mod tests {
    use std::{num::NonZeroUsize, time::Instant};

    use super::*;
    use crate::cooling_schedule::CoolingConfig;

    fn env_usize(name: &str, default: usize) -> usize {
        std::env::var(name)
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(default)
    }

    /// Manual end-to-end crossover probe. It is ignored because normal test
    /// runs must not require a Vulkan adapter.
    #[test]
    #[ignore = "requires a Vulkan GPU"]
    fn profile_gpu_phases() {
        let graph_path =
            std::env::var("PROFILE_GRAPH").unwrap_or_else(|_| "data/grid-8x8.json".to_owned());
        let graph = Graph::load(graph_path).unwrap();
        let config = Config {
            num_of_chains: env_usize("PROFILE_CHAINS", 2048),
            warmup_times: env_usize("PROFILE_WARMUP", 0),
            weight_sample_intervals: env_usize("PROFILE_WEIGHT_INTERVAL", 16),
            estimator_sample_intervals: env_usize("PROFILE_ESTIMATOR_INTERVAL", 128),
            num_of_weight_estimations: env_usize("PROFILE_WEIGHT_SAMPLES", 2048),
            num_of_estimator_estimations: env_usize("PROFILE_ESTIMATOR_SAMPLES", 64),
        };
        let n = graph.size;
        let profile_steps = env_usize("PROFILE_STEPS", 10);

        let started = Instant::now();
        let mut state = GpuMCState::new(graph, config);
        let initialize_elapsed = started.elapsed();
        let started = Instant::now();
        state.warmup();
        let warmup_elapsed = started.elapsed();

        let schedule = CoolingSchedule::from(CoolingConfig {
            n: NonZeroUsize::new(n).unwrap(),
            additive_ratio: NonZeroUsize::new(1).unwrap(),
            multiplicative_ratio: NonZeroUsize::new(1).unwrap(),
        });
        let started = Instant::now();
        let mut observed_steps = 0usize;
        let mut acceptance_sum = 0.0;
        let mut acceptance_min = f64::INFINITY;
        let mut acceptance_max = 0.0f64;
        state.cooling_evolve_with(schedule, |step, _| {
            if step.attempted_samples > 0 {
                let acceptance = step.accepted_samples / step.attempted_samples as f64;
                acceptance_sum += acceptance;
                acceptance_min = acceptance_min.min(acceptance);
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
            "gpu n={n} config={config:?} steps={observed_steps} init={initialize_elapsed:?} warmup={warmup_elapsed:?} evolve={evolve_elapsed:?} per_step={:?} acceptance=min:{acceptance_min:.6},mean:{acceptance_mean:.6},max:{acceptance_max:.6}",
            evolve_elapsed / observed_steps as u32
        );
    }

    /// Deterministic GPU RNG streams make this a useful manual regression
    /// check against a small graph whose permanent is known exactly.
    #[test]
    #[ignore = "requires a Vulkan GPU"]
    fn estimates_four_cycles() {
        let graph = Graph::load("data/4-cycles.json").unwrap();
        let exact = crate::exact::permanent(&graph);
        let config = Config {
            num_of_chains: 1024,
            warmup_times: 2048,
            weight_sample_intervals: 8,
            estimator_sample_intervals: 16,
            num_of_weight_estimations: 512,
            num_of_estimator_estimations: 64,
        };
        let mut state = GpuMCState::new(graph, config);
        state.warmup();
        let schedule = CoolingSchedule::from(CoolingConfig {
            n: NonZeroUsize::new(8).unwrap(),
            additive_ratio: NonZeroUsize::new(1).unwrap(),
            multiplicative_ratio: NonZeroUsize::new(1).unwrap(),
        });
        let estimate = state.cooling_evolve(schedule);
        let relative_error = (estimate - exact).abs() / exact;
        assert!(
            relative_error < 0.05,
            "GPU estimate {estimate} differs from exact {exact} by {:.2}%",
            relative_error * 100.0
        );
    }
}
