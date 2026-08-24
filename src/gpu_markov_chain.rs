use std::time::Instant;

use cubecl::prelude::*;
use cubecl::server::Handle;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use tracing::{info, warn};

use crate::cooling_schedule::CoolingSchedule;
use crate::cooling_state::{Matrix, State};
use crate::graph::{Graph, Match};
use crate::markov_chain::{Config, StepStats};

const CUBE_UNITS: u32 = 32;
/// sentinel for the per-chain hole registers: the matching is perfect
const NO_HOLE: u32 = u32::MAX;

#[derive(CubeType, Clone, Copy)]
struct CompensatedF32 {
    sum: f32,
    correction: f32,
}

#[cube]
impl CompensatedF32 {
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

/// Per-chain registers of the JSV walker that live outside the matching
/// arrays; carried through the transit helper by value.
#[derive(CubeType, Clone, Copy)]
struct ChainRegs {
    hole_u: u32,
    hole_v: u32,
    active: u32,
    rng: u32,
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

/// One menu-based Metropolis–Hastings proposal of the JSV walker, the exact
/// GPU mirror of `JsvChain::transit`: a perfect matching draws one of its n
/// removals, a near-perfect one draws from its 2n-1 menu (add across the
/// holes, or slide either hole), and the asymmetric menu sizes enter as the
/// Hastings factor n/(2n-1) on removals and its inverse on adds.
/// `exp_beta[delta + 2]` caches e^{beta * delta}. The modulo range reduction
/// has bias ~n/2^32; the proposal distribution over menu slots is
/// state-size-dependent only, which keeps reversibility intact.
#[cube]
impl ChainRegs {
    #[allow(clippy::too_many_arguments)]
    fn transit(
        &mut self,
        row_match: &mut Array<u32>,
        col_match: &mut Array<u32>,
        weights: &Array<f32>,
        adjacency: &Array<u32>,
        exp_beta: &Array<f32>,
        base: usize,
        n: u32,
    ) {
        let nn = n as usize;
    if self.hole_u == NO_HOLE {
        // remove one of the n matched edges; Hastings factor n/(2n-1)
        self.rng = xorshift32(self.rng);
        let u = self.rng % n;
        let v = row_match[base + u as usize];
        let index = u as usize * nn + v as usize;
        let activity = adjacency[index];
        let probability = weights[index]
            * exp_beta[(3 - activity) as usize]
            * (f32::cast_from(n) / f32::cast_from(2 * n - 1));
        self.rng = xorshift32(self.rng);
        if probability >= 1.0 || uniform_f32(self.rng) < probability {
            row_match[base + u as usize] = NO_HOLE;
            col_match[base + v as usize] = NO_HOLE;
            self.hole_u = u;
            self.hole_v = v;
            self.active -= activity;
        }
    } else {
        self.rng = xorshift32(self.rng);
        let slot = self.rng % (2 * n - 1);
        if slot == 0 {
            // add across the holes; Hastings factor (2n-1)/n
            let index = self.hole_u as usize * nn + self.hole_v as usize;
            let activity = adjacency[index];
            let probability = exp_beta[(activity + 1) as usize] / weights[index]
                * (f32::cast_from(2 * n - 1) / f32::cast_from(n));
            self.rng = xorshift32(self.rng);
            if probability >= 1.0 || uniform_f32(self.rng) < probability {
                row_match[base + self.hole_u as usize] = self.hole_v;
                col_match[base + self.hole_v as usize] = self.hole_u;
                self.active += activity;
                self.hole_u = NO_HOLE;
                self.hole_v = NO_HOLE;
            }
        } else if slot < n {
            // slide onto the row hole: column v != hole_v, matched to row z
            let pick = slot - 1;
            let v = if pick >= self.hole_v { pick + 1 } else { pick };
            let z = col_match[base + v as usize];
            let gained_index = self.hole_u as usize * nn + v as usize;
            let lost_index = z as usize * nn + v as usize;
            let gained = adjacency[gained_index];
            let lost = adjacency[lost_index];
            let probability = exp_beta[(gained + 2 - lost) as usize]
                * weights[z as usize * nn + self.hole_v as usize]
                / weights[self.hole_u as usize * nn + self.hole_v as usize];
            self.rng = xorshift32(self.rng);
            if probability >= 1.0 || uniform_f32(self.rng) < probability {
                row_match[base + self.hole_u as usize] = v;
                col_match[base + v as usize] = self.hole_u;
                row_match[base + z as usize] = NO_HOLE;
                self.hole_u = z;
                self.active = self.active + gained - lost;
            }
        } else {
            // slide onto the column hole: row u != hole_u, matched to col z
            let pick = slot - n;
            let u = if pick >= self.hole_u { pick + 1 } else { pick };
            let z = row_match[base + u as usize];
            let gained_index = u as usize * nn + self.hole_v as usize;
            let lost_index = u as usize * nn + z as usize;
            let gained = adjacency[gained_index];
            let lost = adjacency[lost_index];
            let probability = exp_beta[(gained + 2 - lost) as usize]
                * weights[self.hole_u as usize * nn + z as usize]
                / weights[self.hole_u as usize * nn + self.hole_v as usize];
            self.rng = xorshift32(self.rng);
            if probability >= 1.0 || uniform_f32(self.rng) < probability {
                row_match[base + u as usize] = self.hole_v;
                col_match[base + self.hole_v as usize] = u;
                col_match[base + z as usize] = NO_HOLE;
                self.hole_v = z;
                self.active = self.active + gained - lost;
            }
        }
    }
}
}

/// One invocation owns one chain. Transitions within a chain are dependent
/// and therefore serial; independent chains supply GPU parallelism.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn warmup_kernel(
    row_match: &mut Array<u32>,
    col_match: &mut Array<u32>,
    weights: &Array<f32>,
    adjacency: &Array<u32>,
    exp_beta: &Array<f32>,
    holes_u: &mut Array<u32>,
    holes_v: &mut Array<u32>,
    active_counts: &mut Array<u32>,
    rng_states: &mut Array<u32>,
    n: u32,
    iterations: usize,
) {
    let chain = ABSOLUTE_POS;
    if chain >= holes_u.len() {
        terminate!();
    }
    let base = chain * n as usize;
    let mut regs = ChainRegs {
        hole_u: holes_u[chain],
        hole_v: holes_v[chain],
        active: active_counts[chain],
        rng: rng_states[chain],
    };
    for _ in 0..iterations {
        regs.transit(row_match, col_match, weights, adjacency, exp_beta, base, n);
    }
    holes_u[chain] = regs.hole_u;
    holes_v[chain] = regs.hole_v;
    active_counts[chain] = regs.active;
    rng_states[chain] = regs.rng;
}

/// Occupancy pass: `samples` per chain, `interval` proposals apart, into a
/// histogram over the n^2 hole classes plus the perfect class (last slot).
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn occupancy_kernel(
    row_match: &mut Array<u32>,
    col_match: &mut Array<u32>,
    weights: &Array<f32>,
    adjacency: &Array<u32>,
    exp_beta: &Array<f32>,
    holes_u: &mut Array<u32>,
    holes_v: &mut Array<u32>,
    active_counts: &mut Array<u32>,
    rng_states: &mut Array<u32>,
    histogram: &mut Array<Atomic<u32>>,
    n: u32,
    samples: usize,
    interval: usize,
) {
    let chain = ABSOLUTE_POS;
    if chain >= holes_u.len() {
        terminate!();
    }
    let base = chain * n as usize;
    let mut regs = ChainRegs {
        hole_u: holes_u[chain],
        hole_v: holes_v[chain],
        active: active_counts[chain],
        rng: rng_states[chain],
    };
    for _ in 0..samples {
        for _ in 0..interval {
            regs.transit(row_match, col_match, weights, adjacency, exp_beta, base, n);
        }
        let slot = if regs.hole_u == NO_HOLE {
            n as usize * n as usize
        } else {
            regs.hole_u as usize * n as usize + regs.hole_v as usize
        };
        histogram[slot].fetch_add(1);
    }
    holes_u[chain] = regs.hole_u;
    holes_v[chain] = regs.hole_v;
    active_counts[chain] = regs.active;
    rng_states[chain] = regs.rng;
}

/// Ratio pass: accumulate per chain the telescoping terms
/// e^{(beta - beta') inactive} * w'(M)/w(M) (via the precomputed
/// `ratio_terms[inactive]` table) and count fully-active perfect samples.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn ratio_kernel(
    row_match: &mut Array<u32>,
    col_match: &mut Array<u32>,
    weights: &Array<f32>,
    next_weights: &Array<f32>,
    adjacency: &Array<u32>,
    exp_beta: &Array<f32>,
    ratio_terms: &Array<f32>,
    holes_u: &mut Array<u32>,
    holes_v: &mut Array<u32>,
    active_counts: &mut Array<u32>,
    rng_states: &mut Array<u32>,
    sums: &mut Array<f32>,
    corrections: &mut Array<f32>,
    perfect_active: &mut Array<u32>,
    n: u32,
    samples: usize,
    interval: usize,
) {
    let chain = ABSOLUTE_POS;
    if chain >= holes_u.len() {
        terminate!();
    }
    let base = chain * n as usize;
    let mut regs = ChainRegs {
        hole_u: holes_u[chain],
        hole_v: holes_v[chain],
        active: active_counts[chain],
        rng: rng_states[chain],
    };
    let mut acc = CompensatedF32 {
        sum: 0.0,
        correction: 0.0,
    };
    let mut hits = 0u32;
    for _ in 0..samples {
        for _ in 0..interval {
            regs.transit(row_match, col_match, weights, adjacency, exp_beta, base, n);
        }
        if regs.hole_u == NO_HOLE {
            let inactive = n - regs.active;
            acc.add(ratio_terms[inactive as usize]);
            if inactive == 0 {
                hits += 1;
            }
        } else {
            let index = regs.hole_u as usize * n as usize + regs.hole_v as usize;
            let inactive = n - 1 - regs.active;
            acc.add(ratio_terms[inactive as usize] * next_weights[index] / weights[index]);
        }
    }
    holes_u[chain] = regs.hole_u;
    holes_v[chain] = regs.hole_v;
    active_counts[chain] = regs.active;
    rng_states[chain] = regs.rng;
    sums[chain] = acc.sum;
    corrections[chain] = acc.correction;
    perfect_active[chain] = hits;
}

struct GpuEvolveStats {
    ratio: f64,
    perfect_active_samples: usize,
    total_samples: usize,
}

/// GPU implementation of the same BSVV annealing estimator as `MCState`.
///
/// Matchings, hole registers, RNGs, and active counts stay resident on the
/// device. The host retains the hole-weight matrix (in f64) so the weight
/// bootstrap, the occupancy-invariant guard, and the observer/TUI plumbing
/// are shared with the CPU path; the device works from an f32 copy uploaded
/// each step, which the [1e-30, 1e30] weight cap keeps representable.
pub struct GpuMCState {
    size: usize,
    config: Config,
    pub global_state: State,
    client: ComputeClient<WgpuRuntime>,
    row_match: Handle,
    col_match: Handle,
    holes_u: Handle,
    holes_v: Handle,
    adjacency: Handle,
    active_counts: Handle,
    rng_states: Handle,
}

impl GpuMCState {
    pub fn new(graph: Graph, config: Config) -> Self {
        let size = graph.size;
        assert!(size >= 2, "GPU Markov chains require n >= 2");
        assert!(config.num_of_chains > 0, "at least one chain is required");
        assert!(size < u32::MAX as usize);
        assert!(
            config.num_of_chains <= u32::MAX as usize
                && config.num_of_chains.div_ceil(CUBE_UNITS as usize) <= u16::MAX as usize,
            "too many chains for one GPU dispatch"
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

        let mut row_match = Vec::with_capacity(config.num_of_chains * size);
        let mut col_match = vec![0u32; config.num_of_chains * size];
        let mut active_counts = Vec::with_capacity(config.num_of_chains);
        for chain in 0..config.num_of_chains {
            let matching = Match::random(size);
            for &(row, column) in matching.edges.iter() {
                col_match[chain * size + column] = row as u32;
            }
            row_match.extend(matching.edges.iter().map(|&(_, column)| column as u32));
            active_counts.push(global_state.active_count_of_match(&matching) as u32);
        }
        let holes = vec![NO_HOLE; config.num_of_chains];
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
            row_match: client.create_from_slice(u32::as_bytes(&row_match)),
            col_match: client.create_from_slice(u32::as_bytes(&col_match)),
            holes_u: client.create_from_slice(u32::as_bytes(&holes)),
            holes_v: client.create_from_slice(u32::as_bytes(&holes)),
            adjacency: client.create_from_slice(u32::as_bytes(&adjacency_values)),
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

    /// One occupancy pass on the device; returns the merged histogram with
    /// the perfect-class count in the last slot.
    fn occupancy_pass(&self, weights: &Handle, exp_beta: &Handle, samples: usize) -> Vec<u32> {
        let histogram = self
            .client
            .create_from_slice(u32::as_bytes(&vec![0u32; self.size * self.size + 1]));
        unsafe {
            occupancy_kernel::launch_unchecked(
                &self.client,
                self.cube_count(),
                CubeDim::new_1d(CUBE_UNITS),
                ArrayArg::from_raw_parts(
                    self.row_match.clone(),
                    self.config.num_of_chains * self.size,
                ),
                ArrayArg::from_raw_parts(
                    self.col_match.clone(),
                    self.config.num_of_chains * self.size,
                ),
                ArrayArg::from_raw_parts(weights.clone(), self.size * self.size),
                ArrayArg::from_raw_parts(self.adjacency.clone(), self.size * self.size),
                ArrayArg::from_raw_parts(exp_beta.clone(), 5),
                ArrayArg::from_raw_parts(self.holes_u.clone(), self.config.num_of_chains),
                ArrayArg::from_raw_parts(self.holes_v.clone(), self.config.num_of_chains),
                ArrayArg::from_raw_parts(self.active_counts.clone(), self.config.num_of_chains),
                ArrayArg::from_raw_parts(self.rng_states.clone(), self.config.num_of_chains),
                ArrayArg::from_raw_parts(histogram.clone(), self.size * self.size + 1),
                self.size as u32,
                samples,
                self.config.weight_sample_intervals,
            );
        }
        let bytes = self.client.read_one_unchecked(histogram);
        u32::from_bytes(&bytes).to_vec()
    }

    /// One ratio pass on the device; returns (term sum, fully-active-perfect
    /// hits, total samples).
    fn ratio_pass(
        &self,
        weights: &Handle,
        next_weights: &Handle,
        exp_beta: &Handle,
        ratio_terms: &[f32],
        samples: usize,
        interval: usize,
    ) -> (f64, usize, usize) {
        let ratio_terms = self.client.create_from_slice(f32::as_bytes(ratio_terms));
        let sums = self
            .client
            .create_from_slice(f32::as_bytes(&vec![0.0f32; self.config.num_of_chains]));
        let corrections = self
            .client
            .create_from_slice(f32::as_bytes(&vec![0.0f32; self.config.num_of_chains]));
        let perfect_active = self
            .client
            .create_from_slice(u32::as_bytes(&vec![0u32; self.config.num_of_chains]));
        unsafe {
            ratio_kernel::launch_unchecked(
                &self.client,
                self.cube_count(),
                CubeDim::new_1d(CUBE_UNITS),
                ArrayArg::from_raw_parts(
                    self.row_match.clone(),
                    self.config.num_of_chains * self.size,
                ),
                ArrayArg::from_raw_parts(
                    self.col_match.clone(),
                    self.config.num_of_chains * self.size,
                ),
                ArrayArg::from_raw_parts(weights.clone(), self.size * self.size),
                ArrayArg::from_raw_parts(next_weights.clone(), self.size * self.size),
                ArrayArg::from_raw_parts(self.adjacency.clone(), self.size * self.size),
                ArrayArg::from_raw_parts(exp_beta.clone(), 5),
                ArrayArg::from_raw_parts(ratio_terms, self.size + 1),
                ArrayArg::from_raw_parts(self.holes_u.clone(), self.config.num_of_chains),
                ArrayArg::from_raw_parts(self.holes_v.clone(), self.config.num_of_chains),
                ArrayArg::from_raw_parts(self.active_counts.clone(), self.config.num_of_chains),
                ArrayArg::from_raw_parts(self.rng_states.clone(), self.config.num_of_chains),
                ArrayArg::from_raw_parts(sums.clone(), self.config.num_of_chains),
                ArrayArg::from_raw_parts(corrections.clone(), self.config.num_of_chains),
                ArrayArg::from_raw_parts(perfect_active.clone(), self.config.num_of_chains),
                self.size as u32,
                samples,
                interval,
            );
        }
        let outputs = self.client.read(vec![sums, corrections, perfect_active]);
        let sums = f32::from_bytes(&outputs[0]);
        let corrections = f32::from_bytes(&outputs[1]);
        let hits = u32::from_bytes(&outputs[2]);
        let total = sums
            .iter()
            .zip(corrections)
            .map(|(&sum, &correction)| sum as f64 + correction as f64)
            .sum::<f64>();
        let hits = hits.iter().map(|&value| value as usize).sum::<usize>();
        (total, hits, self.config.num_of_chains * samples)
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
                    self.row_match.clone(),
                    self.config.num_of_chains * self.size,
                ),
                ArrayArg::from_raw_parts(
                    self.col_match.clone(),
                    self.config.num_of_chains * self.size,
                ),
                ArrayArg::from_raw_parts(weights, self.size * self.size),
                ArrayArg::from_raw_parts(self.adjacency.clone(), self.size * self.size),
                ArrayArg::from_raw_parts(exp_beta, 5),
                ArrayArg::from_raw_parts(self.holes_u.clone(), self.config.num_of_chains),
                ArrayArg::from_raw_parts(self.holes_v.clone(), self.config.num_of_chains),
                ArrayArg::from_raw_parts(self.active_counts.clone(), self.config.num_of_chains),
                ArrayArg::from_raw_parts(self.rng_states.clone(), self.config.num_of_chains),
                self.size as u32,
                self.config.warmup_times,
            );
        }
        // Make `warmup` synchronous, matching the CPU API and its progress log.
        self.client.read_one_unchecked(self.holes_u.clone());
    }

    /// Same step structure and invariant guard as `MCState::evolve`; see the
    /// CPU implementation for the reasoning.
    fn evolve(&mut self, next_beta: f64) -> GpuEvolveStats {
        let first_half = self.config.num_of_weight_estimations / 2;
        let second_half = self.config.num_of_weight_estimations - first_half;
        let weights_f32 = self.weight_values();
        let weights = self.client.create_from_slice(f32::as_bytes(&weights_f32));
        let exp_beta = self.exp_beta_values();
        let exp_beta = self.client.create_from_slice(f32::as_bytes(&exp_beta));

        let classes = self.size * self.size + 1;
        let expected_perfect =
            (self.config.num_of_chains * first_half) as f64 / classes as f64;
        let mut histogram = self.occupancy_pass(&weights, &exp_beta, first_half);
        const MAX_EQUILIBRATION_RETRIES: usize = 3;
        for retry in 0..MAX_EQUILIBRATION_RETRIES {
            let perfect = histogram[self.size * self.size] as f64;
            if perfect >= expected_perfect / 2.0 && perfect <= expected_perfect * 2.0 {
                break;
            }
            warn!(
                "perfect-class occupancy {} outside [{:.1}, {:.1}] at beta {:.4}; \
                 re-equilibrating (retry {}/{MAX_EQUILIBRATION_RETRIES})",
                histogram[self.size * self.size],
                expected_perfect / 2.0,
                expected_perfect * 2.0,
                self.global_state.beta(),
                retry + 1,
            );
            histogram = self.occupancy_pass(&weights, &exp_beta, first_half);
        }
        let counts = histogram[..self.size * self.size]
            .iter()
            .map(|&value| value as usize)
            .collect::<Vec<_>>();
        let perfect_count = histogram[self.size * self.size] as usize;
        let (next_weight, _) = Matrix::hole_weights_from_counts(
            &self.global_state.weight,
            &counts,
            perfect_count,
        );

        let next_weights_f32 = (0..self.size * self.size)
            .map(|index| next_weight.get(index / self.size, index % self.size) as f32)
            .collect::<Vec<_>>();
        let next_weights = self
            .client
            .create_from_slice(f32::as_bytes(&next_weights_f32));
        let diff = (self.global_state.beta() - next_beta) as f32;
        let ratio_terms = (0..=self.size)
            .map(|missing| (diff * missing as f32).exp())
            .collect::<Vec<_>>();
        let (sum, hits, total) = self.ratio_pass(
            &weights,
            &next_weights,
            &exp_beta,
            &ratio_terms,
            second_half,
            self.config.weight_sample_intervals,
        );

        self.global_state.weight = next_weight;
        GpuEvolveStats {
            ratio: sum / total.max(1) as f64,
            perfect_active_samples: hits,
            total_samples: total,
        }
    }

    /// Fraction of stationary samples that are perfect matchings of the
    /// real graph, measured with the identity weight table (all ratio terms
    /// 1). See `MCState::estimate_perfect_fraction`.
    fn estimate_perfect_fraction(&mut self) -> f64 {
        let per_chain = self
            .config
            .num_of_estimator_estimations
            .max((64 * (self.size * self.size + 1)).div_ceil(self.config.num_of_chains));
        let weights_f32 = self.weight_values();
        let weights = self.client.create_from_slice(f32::as_bytes(&weights_f32));
        let exp_beta = self.exp_beta_values();
        let exp_beta = self.client.create_from_slice(f32::as_bytes(&exp_beta));
        let ratio_terms = vec![1.0f32; self.size + 1];
        let (_, hits, total) = self.ratio_pass(
            &weights,
            &weights,
            &exp_beta,
            &ratio_terms,
            per_chain,
            self.config.estimator_sample_intervals,
        );
        info!("final round: {hits} of {total} samples were perfect matchings of the graph");
        if hits == 0 {
            warn!(
                "no perfect matching of the graph was ever sampled; the \
                 estimate is 0 and the chain has almost surely not mixed"
            );
        }
        hits as f64 / total.max(1) as f64
    }

    fn ln_z0(&self) -> f64 {
        ((self.size * self.size + 1) as f64).ln()
            + (1..=self.size).map(|k| (k as f64).ln()).sum::<f64>()
    }

    pub fn cooling_evolve_with<F>(&mut self, sequence: CoolingSchedule, mut observer: F) -> f64
    where
        F: FnMut(&StepStats, &State) -> bool,
    {
        let total_steps = sequence.total_steps();
        let mut ln_z = self.ln_z0();
        for (index, beta) in sequence.skip(1).enumerate() {
            let stats = self.evolve(beta);
            if stats.ratio < 0.3 {
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
        state.cooling_evolve_with(schedule, |_, _| {
            observed_steps += 1;
            observed_steps < profile_steps
        });
        let evolve_elapsed = started.elapsed();

        println!(
            "gpu n={n} config={config:?} steps={observed_steps} init={initialize_elapsed:?} \
             warmup={warmup_elapsed:?} evolve={evolve_elapsed:?} per_step={:?}",
            evolve_elapsed / observed_steps.max(1) as u32
        );
    }

    /// Deterministic GPU RNG streams make this a useful manual regression
    /// check against a small graph whose permanent is known exactly.
    #[test]
    #[ignore = "requires a Vulkan GPU"]
    fn estimates_four_cycles() {
        let graph = Graph::load("data/4-cycles.json").unwrap();
        let exact = crate::exact::to_f64(&crate::exact::permanent(&graph));
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
            additive_ratio: NonZeroUsize::new(4).unwrap(),
            multiplicative_ratio: NonZeroUsize::new(4).unwrap(),
        });
        let estimate = state.cooling_evolve(schedule);
        let relative_error = (estimate - exact).abs() / exact;
        assert!(
            relative_error < 0.30,
            "GPU estimate {estimate} differs from exact {exact} by {:.2}%",
            relative_error * 100.0
        );
    }
}
