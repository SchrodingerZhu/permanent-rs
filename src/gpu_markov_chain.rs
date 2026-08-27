use std::marker::PhantomData;
use std::time::Instant;

#[cfg(feature = "cuda")]
use cubecl::cuda::{CudaDevice, CudaRuntime as GpuRuntime};
use cubecl::prelude::*;
use cubecl::server::Handle;
#[cfg(not(feature = "cuda"))]
use cubecl::wgpu::{WgpuDevice, WgpuRuntime as GpuRuntime};
use tracing::{info, warn};

use crate::cooling_schedule::CoolingSchedule;
use crate::cooling_state::{Matrix, State};
use crate::graph::{Graph, Match};
use crate::markov_chain::{Config, StepStats, final_round};

const CUBE_UNITS: u32 = 32;

#[cfg(feature = "cuda")]
fn default_device() -> CudaDevice {
    CudaDevice::default()
}
#[cfg(not(feature = "cuda"))]
fn default_device() -> WgpuDevice {
    WgpuDevice::DefaultDevice
}
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
struct ChainRegs<R: ChainRng> {
    hole_u: u32,
    hole_v: u32,
    active: u32,
    rng: R,
}

/// The random source a chain draws from, so the kernels below can be written
/// once and instantiated per generator. `states` holds two u32 per chain; a
/// generator that needs less simply ignores the second slot.
// `Send + Sync + 'static` are what the generated launch wrappers require of a
// kernel's generic parameters.
#[cube]
trait ChainRng: CubeType + Send + Sync + 'static {
    fn load(states: &Array<u32>, chain: usize) -> Self;
    fn store(&self, states: &mut Array<u32>, chain: usize);
    fn next_u32(&mut self) -> u32;
}

/// The 32-bit xorshift this backend used before Philox: cheap, but it fails
/// TestU01 Crush and its period is only 2^32-1, so chains seeded nearby walk
/// correlated stretches of one cycle. Kept as a speed baseline, not a default.
#[derive(CubeType, Clone, Copy)]
struct Xorshift32 {
    state: u32,
}

#[cube]
impl ChainRng for Xorshift32 {
    fn load(states: &Array<u32>, chain: usize) -> Xorshift32 {
        Xorshift32 {
            state: states[2 * chain],
        }
    }

    fn store(&self, states: &mut Array<u32>, chain: usize) {
        states[2 * chain] = self.state;
    }

    fn next_u32(&mut self) -> u32 {
        let mut value = self.state;
        value = value ^ (value << 13);
        value = value ^ (value >> 17);
        value = value ^ (value << 5);
        self.state = value;
        value
    }
}

const PHILOX_M0: u32 = 0xD251_1F53;
const PHILOX_M1: u32 = 0xCD9E_8D57;
const PHILOX_W0: u32 = 0x9E37_79B9;
const PHILOX_W1: u32 = 0xBB67_AE85;
const PHILOX_ROUNDS: u32 = 10;

/// Philox4x32-10 (Salmon et al., "Parallel Random Numbers: As Easy as
/// 1, 2, 3", SC'11), the counter-based RNG used by CUDA's cuRAND and by
/// most GPU tensor libraries. The 128-bit counter is laid out as
/// [block_lo, block_hi, stream, 0] with the chain index as the stream
/// word, so every chain owns 2^64 disjoint blocks of four outputs by
/// construction, and only the 64-bit block counter has to round-trip
/// through device memory between kernel launches.
#[derive(CubeType, Clone, Copy)]
struct Philox {
    block_lo: u32,
    block_hi: u32,
    stream: u32,
    lane_0: u32,
    lane_1: u32,
    lane_2: u32,
    lane_3: u32,
    remaining: u32,
}

/// High 32 bits of the 32x32 widening product. CUDA/NVRTC has a native
/// 64-bit type, so the limb decomposition below is only needed for WGSL.
#[cube]
fn mul_wide_hi(a: u32, b: u32) -> u32 {
    u32::cast_from((u64::cast_from(a) * u64::cast_from(b)) >> 32)
}

/// High 32 bits of the 32x32 widening product, via 16-bit limbs: WGSL has
/// no u64 and `mulhi` is not exposed, and no intermediate here can wrap.
#[cube]
#[allow(dead_code)]
fn mul_wide_hi_limbs(a: u32, b: u32) -> u32 {
    let a_lo = a & 0xFFFF;
    let a_hi = a >> 16;
    let b_lo = b & 0xFFFF;
    let b_hi = b >> 16;
    let mid_0 = a_hi * b_lo + ((a_lo * b_lo) >> 16);
    let mid_1 = a_lo * b_hi + (mid_0 & 0xFFFF);
    a_hi * b_hi + (mid_0 >> 16) + (mid_1 >> 16)
}

#[cube]
impl Philox {
    /// Encrypt the current counter block into four fresh outputs and
    /// advance the block counter.
    fn refill(&mut self) {
        let mut c0 = self.block_lo;
        let mut c1 = self.block_hi;
        let mut c2 = self.stream;
        let mut c3 = 0u32;
        let mut k0 = 0u32;
        let mut k1 = 0u32;
        for _ in 0..PHILOX_ROUNDS {
            let hi0 = mul_wide_hi(PHILOX_M0, c0);
            let lo0 = PHILOX_M0 * c0;
            let hi1 = mul_wide_hi(PHILOX_M1, c2);
            let lo1 = PHILOX_M1 * c2;
            c0 = hi1 ^ c1 ^ k0;
            c1 = lo1;
            c2 = hi0 ^ c3 ^ k1;
            c3 = lo0;
            k0 += PHILOX_W0;
            k1 += PHILOX_W1;
        }
        self.lane_0 = c0;
        self.lane_1 = c1;
        self.lane_2 = c2;
        self.lane_3 = c3;
        self.remaining = 4;
        self.block_lo += 1;
        if self.block_lo == 0 {
            self.block_hi += 1;
        }
    }
}

#[cube]
impl ChainRng for Philox {
    fn load(states: &Array<u32>, chain: usize) -> Philox {
        Philox {
            block_lo: states[2 * chain],
            block_hi: states[2 * chain + 1],
            stream: chain as u32,
            lane_0: 0,
            lane_1: 0,
            lane_2: 0,
            lane_3: 0,
            remaining: 0,
        }
    }

    /// Only the block counter persists; up to three buffered outputs are
    /// discarded at a launch boundary, which skips (never reuses) them.
    fn store(&self, states: &mut Array<u32>, chain: usize) {
        states[2 * chain] = self.block_lo;
        states[2 * chain + 1] = self.block_hi;
    }

    fn next_u32(&mut self) -> u32 {
        if self.remaining == 0 {
            self.refill();
        }
        let out = self.lane_0;
        self.lane_0 = self.lane_1;
        self.lane_1 = self.lane_2;
        self.lane_2 = self.lane_3;
        self.remaining -= 1;
        out
    }
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
impl<R: ChainRng> ChainRegs<R> {
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
            let u = self.rng.next_u32() % n;
            let v = row_match[base + u as usize];
            let index = u as usize * nn + v as usize;
            let activity = adjacency[index];
            let probability = weights[index]
                * exp_beta[(3 - activity) as usize]
                * (f32::cast_from(n) / f32::cast_from(2 * n - 1));
            if probability >= 1.0 || uniform_f32(self.rng.next_u32()) < probability {
                row_match[base + u as usize] = NO_HOLE;
                col_match[base + v as usize] = NO_HOLE;
                self.hole_u = u;
                self.hole_v = v;
                self.active -= activity;
            }
        } else {
            let slot = self.rng.next_u32() % (2 * n - 1);
            if slot == 0 {
                // add across the holes; Hastings factor (2n-1)/n
                let index = self.hole_u as usize * nn + self.hole_v as usize;
                let activity = adjacency[index];
                let probability = exp_beta[(activity + 1) as usize] / weights[index]
                    * (f32::cast_from(2 * n - 1) / f32::cast_from(n));
                if probability >= 1.0 || uniform_f32(self.rng.next_u32()) < probability {
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
                if probability >= 1.0 || uniform_f32(self.rng.next_u32()) < probability {
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
                if probability >= 1.0 || uniform_f32(self.rng.next_u32()) < probability {
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
fn warmup_kernel<R: ChainRng>(
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
    let mut regs = ChainRegs::<R> {
        hole_u: holes_u[chain],
        hole_v: holes_v[chain],
        active: active_counts[chain],
        rng: R::load(rng_states, chain),
    };
    for _ in 0..iterations {
        regs.transit(row_match, col_match, weights, adjacency, exp_beta, base, n);
    }
    holes_u[chain] = regs.hole_u;
    holes_v[chain] = regs.hole_v;
    active_counts[chain] = regs.active;
    regs.rng.store(rng_states, chain);
}

/// Occupancy pass: `samples` per chain, `interval` proposals apart, into a
/// histogram over the n^2 hole classes plus the perfect class (last slot).
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn occupancy_kernel<R: ChainRng>(
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
    let mut regs = ChainRegs::<R> {
        hole_u: holes_u[chain],
        hole_v: holes_v[chain],
        active: active_counts[chain],
        rng: R::load(rng_states, chain),
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
    regs.rng.store(rng_states, chain);
}

/// Ratio pass: accumulate per chain the telescoping terms
/// e^{(beta - beta') inactive} * w'(M)/w(M) (via the precomputed
/// `ratio_terms[inactive]` table) and count fully-active perfect samples.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn ratio_kernel<R: ChainRng>(
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
    let mut regs = ChainRegs::<R> {
        hole_u: holes_u[chain],
        hole_v: holes_v[chain],
        active: active_counts[chain],
        rng: R::load(rng_states, chain),
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
    regs.rng.store(rng_states, chain);
    sums[chain] = acc.sum;
    corrections[chain] = acc.correction;
    perfect_active[chain] = hits;
}

struct GpuEvolveStats {
    ratio: f64,
    perfect_active_samples: usize,
    total_samples: usize,
}

/// cuRAND device generators, available only through the native CUDA backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CurandRng {
    /// `curandStatePhilox4_32_10_t`, cuRAND's tuned Philox4x32-10.
    Philox,
    /// `curandStateXORWOW_t`, cuRAND's default generator.
    Xorwow,
    /// `curandStateMRG32k3a_t`.
    Mrg32k3a,
}

impl CurandRng {
    pub fn as_str(self) -> &'static str {
        match self {
            CurandRng::Philox => "philox",
            CurandRng::Xorwow => "xorwow",
            CurandRng::Mrg32k3a => "mrg32k3a",
        }
    }
}

/// Host-side starting state for the chains; every backend uploads it verbatim.
pub struct InitialChains {
    pub size: usize,
    pub num_chains: usize,
    pub adjacency: Vec<u32>,
    pub row_match: Vec<u32>,
    pub col_match: Vec<u32>,
    pub holes: Vec<u32>,
    pub active_counts: Vec<u32>,
}

impl InitialChains {
    fn build(graph: &Graph, config: &Config, global_state: &State, seed: u64) -> Self {
        let size = graph.size;
        let mut adjacency = vec![0u32; size * size];
        for (row, edges) in graph.edges.iter().enumerate() {
            for &column in edges.iter() {
                adjacency[row * size + column] = 1;
            }
        }
        let mut row_match = Vec::with_capacity(config.num_of_chains * size);
        let mut col_match = vec![0u32; config.num_of_chains * size];
        let mut active_counts = Vec::with_capacity(config.num_of_chains);
        // The CLI promises that runs with the same seed and configuration
        // reproduce each other, so the initial matchings must come from the
        // seed as well - an entropy-seeded RNG here silently broke that.
        // XOR-folded so the host stream is domain-separated from the device
        // generators, which consume the seed directly.
        let mut host_rng =
            <rand::rngs::StdRng as rand::SeedableRng>::seed_from_u64(seed ^ 0x9E37_79B9_7F4A_7C15);
        for chain in 0..config.num_of_chains {
            let matching = Match::random_with(size, &mut host_rng);
            for &(row, column) in matching.edges.iter() {
                col_match[chain * size + column] = row as u32;
            }
            row_match.extend(matching.edges.iter().map(|&(_, column)| column as u32));
            active_counts.push(global_state.active_count_of_match(&matching) as u32);
        }
        InitialChains {
            size,
            num_chains: config.num_of_chains,
            adjacency,
            row_match,
            col_match,
            holes: vec![NO_HOLE; config.num_of_chains],
            active_counts,
        }
    }
}

/// The device half of the BSVV annealing loop.
///
/// Only these three passes touch the GPU; the weight bootstrap, the ratio
/// estimator and the cooling schedule are backend-independent and live in
/// [`GpuMCState`]. A backend owns its resident chain state between calls.
pub trait JsvDevice: Send {
    /// Backend name, for logging.
    fn name(&self) -> String;

    /// Upload the per-step weight and `exp(beta * delta)` tables. They stay
    /// current until the next call.
    fn begin_step(&mut self, weights: &[f32], exp_beta: &[f32]);

    /// Advance every chain by `iterations` proposals. Synchronous on return.
    fn warmup_pass(&mut self, iterations: usize);

    /// Draw `samples` per chain, `interval` proposals apart. Returns the
    /// merged histogram over the `n^2` hole classes with the perfect class
    /// in the last slot.
    fn occupancy_pass(&mut self, samples: usize, interval: usize) -> Vec<u32>;

    /// Accumulate the telescoping ratio terms. Returns
    /// (term sum, fully-active-perfect hits, total samples).
    fn ratio_pass(
        &mut self,
        next_weights: &[f32],
        ratio_terms: &[f32],
        samples: usize,
        interval: usize,
    ) -> (f64, usize, usize);
}

/// Generators the CubeCL kernels carry. cuRAND's generators are not here:
/// `#[cube]` bodies are transpiled, so an external device library is out of
/// reach - that is what the native CUDA backend exists for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CubeclRng {
    Xorshift32,
    Philox,
}

impl CubeclRng {
    pub fn as_str(self) -> &'static str {
        match self {
            CubeclRng::Xorshift32 => "xorshift32",
            CubeclRng::Philox => "philox",
        }
    }
}

/// Finger-mixer used to spread a seed across chains for the generators that
/// need a full-entropy starting state.
fn splitmix32(mut value: u32) -> u32 {
    value = value.wrapping_add(0x9E37_79B9);
    let mut z = value;
    z = (z ^ (z >> 16)).wrapping_mul(0x21F0_AAAD);
    z = (z ^ (z >> 15)).wrapping_mul(0x735A_2D97);
    z ^ (z >> 15)
}

/// Two u32 of starting state per chain.
fn initial_rng_states(rng: CubeclRng, num_chains: usize, seed: u64) -> Vec<u32> {
    let mut states = vec![0u32; num_chains * 2];
    for chain in 0..num_chains {
        match rng {
            // Philox separates chains by the stream word inside the counter
            // block, so every chain can share one seed-derived counter origin.
            CubeclRng::Philox => {
                states[2 * chain] = seed as u32;
                states[2 * chain + 1] = (seed >> 32) as u32;
            }
            // xorshift32 has no stream concept, so chains must be spread over
            // the single cycle by hashing. Zero is the absorbing state.
            CubeclRng::Xorshift32 => {
                let mixed = splitmix32((seed as u32) ^ splitmix32(chain as u32));
                states[2 * chain] = if mixed == 0 { 0x9E37_79B9 } else { mixed };
            }
        }
    }
    states
}

/// CubeCL backend: one `#[cube]` kernel source retargeted to SPIR-V/Vulkan by
/// default, or to CUDA/NVRTC under the `cuda` feature.
struct CubeclDevice<R: ChainRng> {
    size: usize,
    num_chains: usize,
    client: ComputeClient<GpuRuntime>,
    row_match: Handle,
    col_match: Handle,
    holes_u: Handle,
    holes_v: Handle,
    adjacency: Handle,
    active_counts: Handle,
    rng_states: Handle,
    weights: Option<Handle>,
    exp_beta: Option<Handle>,
    rng_kind: CubeclRng,
    _rng: PhantomData<R>,
}

impl<R: ChainRng> CubeclDevice<R> {
    fn new(init: &InitialChains, rng_kind: CubeclRng, seed: u64) -> Self {
        let started = Instant::now();
        let client = GpuRuntime::client(&default_device());
        info!(
            "GPU runtime {} initialized in {:?}: {:#?}",
            GpuRuntime::name(&client),
            started.elapsed(),
            client.properties().hardware
        );
        let rng_states = initial_rng_states(rng_kind, init.num_chains, seed);
        CubeclDevice {
            size: init.size,
            num_chains: init.num_chains,
            row_match: client.create_from_slice(u32::as_bytes(&init.row_match)),
            col_match: client.create_from_slice(u32::as_bytes(&init.col_match)),
            holes_u: client.create_from_slice(u32::as_bytes(&init.holes)),
            holes_v: client.create_from_slice(u32::as_bytes(&init.holes)),
            adjacency: client.create_from_slice(u32::as_bytes(&init.adjacency)),
            active_counts: client.create_from_slice(u32::as_bytes(&init.active_counts)),
            rng_states: client.create_from_slice(u32::as_bytes(&rng_states)),
            client,
            weights: None,
            exp_beta: None,
            rng_kind,
            _rng: PhantomData,
        }
    }

    fn cube_count(&self) -> CubeCount {
        CubeCount::Static((self.num_chains as u32).div_ceil(CUBE_UNITS), 1, 1)
    }

    fn step_tables(&self) -> (&Handle, &Handle) {
        (
            self.weights
                .as_ref()
                .expect("begin_step must precede a device pass"),
            self.exp_beta
                .as_ref()
                .expect("begin_step must precede a device pass"),
        )
    }
}

impl<R: ChainRng> JsvDevice for CubeclDevice<R> {
    fn name(&self) -> String {
        format!(
            "cubecl/{}/{}",
            GpuRuntime::name(&self.client),
            self.rng_kind.as_str()
        )
    }

    fn begin_step(&mut self, weights: &[f32], exp_beta: &[f32]) {
        self.weights = Some(self.client.create_from_slice(f32::as_bytes(weights)));
        self.exp_beta = Some(self.client.create_from_slice(f32::as_bytes(exp_beta)));
    }

    fn warmup_pass(&mut self, iterations: usize) {
        let (weights, exp_beta) = self.step_tables();
        unsafe {
            warmup_kernel::launch_unchecked::<R, GpuRuntime>(
                &self.client,
                self.cube_count(),
                CubeDim::new_1d(CUBE_UNITS),
                ArrayArg::from_raw_parts(self.row_match.clone(), self.num_chains * self.size),
                ArrayArg::from_raw_parts(self.col_match.clone(), self.num_chains * self.size),
                ArrayArg::from_raw_parts(weights.clone(), self.size * self.size),
                ArrayArg::from_raw_parts(self.adjacency.clone(), self.size * self.size),
                ArrayArg::from_raw_parts(exp_beta.clone(), 5),
                ArrayArg::from_raw_parts(self.holes_u.clone(), self.num_chains),
                ArrayArg::from_raw_parts(self.holes_v.clone(), self.num_chains),
                ArrayArg::from_raw_parts(self.active_counts.clone(), self.num_chains),
                ArrayArg::from_raw_parts(self.rng_states.clone(), self.num_chains * 2),
                self.size as u32,
                iterations,
            );
        }
        // Make the pass synchronous, matching the CPU API and its progress log.
        self.client.read_one_unchecked(self.holes_u.clone());
    }

    fn occupancy_pass(&mut self, samples: usize, interval: usize) -> Vec<u32> {
        let (weights, exp_beta) = self.step_tables();
        let histogram =
            self.client
                .create_from_slice(u32::as_bytes(&vec![0u32; self.size * self.size + 1]));
        unsafe {
            occupancy_kernel::launch_unchecked::<R, GpuRuntime>(
                &self.client,
                self.cube_count(),
                CubeDim::new_1d(CUBE_UNITS),
                ArrayArg::from_raw_parts(self.row_match.clone(), self.num_chains * self.size),
                ArrayArg::from_raw_parts(self.col_match.clone(), self.num_chains * self.size),
                ArrayArg::from_raw_parts(weights.clone(), self.size * self.size),
                ArrayArg::from_raw_parts(self.adjacency.clone(), self.size * self.size),
                ArrayArg::from_raw_parts(exp_beta.clone(), 5),
                ArrayArg::from_raw_parts(self.holes_u.clone(), self.num_chains),
                ArrayArg::from_raw_parts(self.holes_v.clone(), self.num_chains),
                ArrayArg::from_raw_parts(self.active_counts.clone(), self.num_chains),
                ArrayArg::from_raw_parts(self.rng_states.clone(), self.num_chains * 2),
                ArrayArg::from_raw_parts(histogram.clone(), self.size * self.size + 1),
                self.size as u32,
                samples,
                interval,
            );
        }
        let bytes = self.client.read_one_unchecked(histogram);
        u32::from_bytes(&bytes).to_vec()
    }

    fn ratio_pass(
        &mut self,
        next_weights: &[f32],
        ratio_terms: &[f32],
        samples: usize,
        interval: usize,
    ) -> (f64, usize, usize) {
        let (weights, exp_beta) = self.step_tables();
        let next_weights = self.client.create_from_slice(f32::as_bytes(next_weights));
        let ratio_terms = self.client.create_from_slice(f32::as_bytes(ratio_terms));
        let sums = self
            .client
            .create_from_slice(f32::as_bytes(&vec![0.0f32; self.num_chains]));
        let corrections = self
            .client
            .create_from_slice(f32::as_bytes(&vec![0.0f32; self.num_chains]));
        let perfect_active = self
            .client
            .create_from_slice(u32::as_bytes(&vec![0u32; self.num_chains]));
        unsafe {
            ratio_kernel::launch_unchecked::<R, GpuRuntime>(
                &self.client,
                self.cube_count(),
                CubeDim::new_1d(CUBE_UNITS),
                ArrayArg::from_raw_parts(self.row_match.clone(), self.num_chains * self.size),
                ArrayArg::from_raw_parts(self.col_match.clone(), self.num_chains * self.size),
                ArrayArg::from_raw_parts(weights.clone(), self.size * self.size),
                ArrayArg::from_raw_parts(next_weights, self.size * self.size),
                ArrayArg::from_raw_parts(self.adjacency.clone(), self.size * self.size),
                ArrayArg::from_raw_parts(exp_beta.clone(), 5),
                ArrayArg::from_raw_parts(ratio_terms, self.size + 1),
                ArrayArg::from_raw_parts(self.holes_u.clone(), self.num_chains),
                ArrayArg::from_raw_parts(self.holes_v.clone(), self.num_chains),
                ArrayArg::from_raw_parts(self.active_counts.clone(), self.num_chains),
                ArrayArg::from_raw_parts(self.rng_states.clone(), self.num_chains * 2),
                ArrayArg::from_raw_parts(sums.clone(), self.num_chains),
                ArrayArg::from_raw_parts(corrections.clone(), self.num_chains),
                ArrayArg::from_raw_parts(perfect_active.clone(), self.num_chains),
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
        (total, hits, self.num_chains * samples)
    }
}

/// Build a CubeCL device for `rng`, hiding the monomorphised generator type.
pub fn cubecl_device(init: &InitialChains, rng: CubeclRng, seed: u64) -> Box<dyn JsvDevice> {
    match rng {
        CubeclRng::Xorshift32 => Box::new(CubeclDevice::<Xorshift32>::new(init, rng, seed)),
        CubeclRng::Philox => Box::new(CubeclDevice::<Philox>::new(init, rng, seed)),
    }
}

/// GPU implementation of the same BSVV annealing estimator as `MCState`.
///
/// Matchings, hole registers, RNGs, and active counts stay resident on the
/// device behind [`JsvDevice`]. The host retains the hole-weight matrix (in
/// f64) so the weight bootstrap, the occupancy-invariant guard, and the
/// observer/TUI plumbing are shared with the CPU path; the device works from
/// an f32 copy uploaded each step, which the [1e-30, 1e30] weight cap keeps
/// representable.
pub struct GpuMCState {
    size: usize,
    config: Config,
    pub global_state: State,
    device: Box<dyn JsvDevice>,
}

impl GpuMCState {
    /// Build the shared host state and hand the initial chains to `device`.
    /// `seed` covers the host-side initial matchings; the device generators
    /// are seeded by the backend itself. Fallible so a backend that cannot
    /// start (no driver, no visible GPU) reports it here rather than from
    /// inside the annealing loop.
    pub fn try_with_device<F>(
        graph: Graph,
        config: Config,
        seed: u64,
        device: F,
    ) -> anyhow::Result<Self>
    where
        F: FnOnce(&InitialChains) -> anyhow::Result<Box<dyn JsvDevice>>,
    {
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
        let init = InitialChains::build(&graph, &config, &global_state, seed);
        let device = device(&init)?;
        info!("GPU backend: {}", device.name());
        Ok(GpuMCState {
            size,
            config,
            global_state,
            device,
        })
    }

    /// The CubeCL backend, kept as the default so existing callers and tests
    /// are unaffected.
    pub fn new(graph: Graph, config: Config) -> Self {
        Self::cubecl(graph, config, CubeclRng::Philox, 0)
    }

    /// The CubeCL backend with an explicit generator and seed.
    pub fn cubecl(graph: Graph, config: Config, rng: CubeclRng, seed: u64) -> Self {
        Self::try_with_device(graph, config, seed, |init| {
            Ok(cubecl_device(init, rng, seed))
        })
        .expect("the CubeCL backend cannot fail to construct")
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

    pub fn warmup(&mut self) {
        if self.config.warmup_times == 0 {
            return;
        }
        let weights = self.weight_values();
        let exp_beta = self.exp_beta_values();
        self.device.begin_step(&weights, &exp_beta);
        self.device.warmup_pass(self.config.warmup_times);
    }

    /// Same step structure and invariant guard as `MCState::evolve`; see the
    /// CPU implementation for the reasoning.
    fn evolve(&mut self, next_beta: f64) -> GpuEvolveStats {
        let first_half = self.config.num_of_weight_estimations / 2;
        let second_half = self.config.num_of_weight_estimations - first_half;
        let weights = self.weight_values();
        let exp_beta = self.exp_beta_values();
        self.device.begin_step(&weights, &exp_beta);

        let classes = self.size * self.size + 1;
        let expected_perfect = (self.config.num_of_chains * first_half) as f64 / classes as f64;
        let mut histogram = self
            .device
            .occupancy_pass(first_half, self.config.weight_sample_intervals);
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
            histogram = self
                .device
                .occupancy_pass(first_half, self.config.weight_sample_intervals);
        }
        let counts = histogram[..self.size * self.size]
            .iter()
            .map(|&value| value as usize)
            .collect::<Vec<_>>();
        let perfect_count = histogram[self.size * self.size] as usize;
        let (next_weight, _) =
            Matrix::hole_weights_from_counts(&self.global_state.weight, &counts, perfect_count);

        let next_weights_f32 = (0..self.size * self.size)
            .map(|index| next_weight.get(index / self.size, index % self.size) as f32)
            .collect::<Vec<_>>();
        let diff = (self.global_state.beta() - next_beta) as f32;
        let ratio_terms = (0..=self.size)
            .map(|missing| (diff * missing as f32).exp())
            .collect::<Vec<_>>();
        let (sum, hits, total) = self.device.ratio_pass(
            &next_weights_f32,
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

    /// Fraction of stationary samples that are perfect matchings of the real
    /// graph, measured with the identity weight table (all ratio terms 1).
    /// Sampled until the hit count justifies the precision rather than for a
    /// fixed number of draws; see [`final_round`].
    fn estimate_perfect_fraction(&mut self) -> f64 {
        let configured = self
            .config
            .num_of_estimator_estimations
            .max((64 * (self.size * self.size + 1)).div_ceil(self.config.num_of_chains));
        let chains = self.config.num_of_chains;
        let weights = self.weight_values();
        let exp_beta = self.exp_beta_values();
        self.device.begin_step(&weights, &exp_beta);
        let ratio_terms = vec![1.0f32; self.size + 1];
        let interval = self.config.estimator_sample_intervals;

        let (_, mut hits, mut total) =
            self.device
                .ratio_pass(&weights, &ratio_terms, configured, interval);
        let mut rounds = 1;
        while rounds < final_round::MAX_ROUNDS {
            let Some(extra) = final_round::next_per_chain(hits, total, chains, configured) else {
                break;
            };
            info!(
                "final round: {hits} of {total} hits so far, short of {}; \
                 drawing {extra} more per chain",
                final_round::TARGET_HITS
            );
            let (_, more_hits, more_total) =
                self.device
                    .ratio_pass(&weights, &ratio_terms, extra, interval);
            hits += more_hits;
            total += more_total;
            rounds += 1;
        }

        info!(
            "final round: {hits} of {total} samples were perfect matchings of \
             the graph ({rounds} round(s), ~{:.1}% relative error)",
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

    /// Runs the annealer on a small graph whose permanent is known exactly and
    /// returns (estimate, exact).
    fn four_cycles_estimate<F>(device: F) -> (f64, f64)
    where
        F: FnOnce(&InitialChains) -> anyhow::Result<Box<dyn JsvDevice>>,
    {
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
        let mut state = GpuMCState::try_with_device(graph, config, 0, device).unwrap();
        state.warmup();
        let schedule = CoolingSchedule::from(CoolingConfig {
            n: NonZeroUsize::new(8).unwrap(),
            additive_ratio: NonZeroUsize::new(4).unwrap(),
            multiplicative_ratio: NonZeroUsize::new(4).unwrap(),
        });
        (state.cooling_evolve(schedule), exact)
    }

    fn assert_close(label: &str, estimate: f64, exact: f64) {
        let relative_error = (estimate - exact).abs() / exact;
        assert!(
            relative_error < 0.30,
            "{label}: estimate {estimate} differs from exact {exact} by {:.2}%",
            relative_error * 100.0
        );
    }

    /// Deterministic device RNG streams make this a useful manual regression
    /// check for every generator the CubeCL kernels carry.
    #[test]
    #[ignore = "requires a Vulkan GPU"]
    fn estimates_four_cycles_cubecl() {
        for rng in [CubeclRng::Philox, CubeclRng::Xorshift32] {
            let (estimate, exact) =
                four_cycles_estimate(|init| Ok(cubecl_device(init, rng, 0x5eed)));
            assert_close(rng.as_str(), estimate, exact);
        }
    }

    /// The same check for the cuRAND generators reachable only from the
    /// hand-written CUDA kernels.
    #[cfg(feature = "native-cuda")]
    #[test]
    #[ignore = "requires an NVIDIA GPU and driver"]
    fn estimates_four_cycles_curand() {
        use crate::cuda_backend::NativeCudaDevice;
        for rng in [CurandRng::Philox, CurandRng::Xorwow, CurandRng::Mrg32k3a] {
            let (estimate, exact) = four_cycles_estimate(|init| {
                Ok(Box::new(NativeCudaDevice::new(init, rng, 0x5eed, 0)?) as Box<dyn JsvDevice>)
            });
            assert_close(rng.as_str(), estimate, exact);
        }
    }
}
