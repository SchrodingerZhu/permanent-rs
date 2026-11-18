//! `permanent` with the population resident on an NVIDIA GPU.
//!
//! The kernels are Rust, compiled for the device by cuda-oxide. One GPU
//! thread owns one particle. Between launches only the particles' matchings
//! (n 16-bit words each) stay in global memory; a launch stages each
//! particle's working arrays in the block's shared memory, so the event
//! loop touches global memory for the alias table alone.
//!
//! The arithmetic differs from the CPU engine, so the two are statistically
//! equivalent, not draw-for-draw identical:
//!
//!  - Every probability decision is an integer comparison. Ring rates are
//!    exact integers (the collision penalty J = 32 n^2 is one), the alias
//!    tables carry 48-bit thresholds and the unassign probability a 64-bit
//!    one, so no decision is coarser than the CPU's 53-bit uniforms.
//!  - Waiting times are f32; nothing in the event loop is f64.
//!  - The generator is Philox4x32-10, keyed by the particle's stream seed.
//!
//! The FAR learner still runs on the CPU: the engine moves the matchings to
//! the host for it.

use std::sync::Arc;

use anyhow::{Context, Result, ensure};
use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig, Stream as CudaStream};
use cuda_device::{DisjointSlice, DynamicSharedArray, device, kernel, thread};
use cuda_host::cuda_module;
use permanent::engine::{Cpu, Engine, FarRound, Stream};
use permanent::sampler::{MutateCost, SampleCost, StageSchedule, mutate_event_cap, penalty};
use permanent::scaling::Scaled;
use permanent::stats::{mix, stream_seed};
use permanent::tables::Tables;
use tracing::info;

#[path = "../../src/philox.rs"]
mod philox;
use philox::{Philox, philox_new, philox_next, philox_next_u64};

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

/// The five working arrays of a particle, each n words (see
/// `permanent::sampler::Scratch`).
const SIGMA: usize = 0;
const DEG: usize = 1;
const OCC: usize = 2;
const LIST: usize = 3;
const POS: usize = 4;
const ARRAYS: usize = 5;
const NONE: u32 = 0xFFFF;
/// Set in a particle's event count when its move hit the event cap.
const OVER_BUDGET: u64 = 1 << 63;
/// Set in a trial's event count when it ended on a perfect matching.
const PERFECT: u64 = 1 << 62;
/// Stage time of the sampler, in ticks.
const STAGE_TICKS: u64 = 1 << 40;

/// One particle's working arrays. Element i of array a lives at
/// `base + (a n + i) stride`: stride 1 in global memory, the block size in
/// shared memory, where it puts the lanes of a warp in distinct banks.
#[derive(Clone, Copy)]
struct Work {
    base: *mut u16,
    stride: usize,
    n: usize,
    n_conf: u32,
    unassigned: u32,
}

#[device]
unsafe fn get(work: &Work, array: usize, index: u32) -> u32 {
    unsafe { *work.base.add((array * work.n + index as usize) * work.stride) as u32 }
}

#[device]
unsafe fn set(work: &Work, array: usize, index: u32, value: u32) {
    unsafe { *work.base.add((array * work.n + index as usize) * work.stride) = value as u16 }
}

#[device]
unsafe fn swap(work: &Work, i: u32, j: u32) {
    unsafe {
        let a = get(work, LIST, i);
        let b = get(work, LIST, j);
        set(work, LIST, i, b);
        set(work, LIST, j, a);
        set(work, POS, a, j);
        set(work, POS, b, i);
    }
}

#[device]
unsafe fn mark_conflicted(work: &mut Work, u: u32) {
    unsafe {
        let p = get(work, POS, u);
        if p >= work.n_conf {
            swap(work, p, work.n_conf);
            work.n_conf += 1;
        }
    }
}

#[device]
unsafe fn mark_clear(work: &mut Work, u: u32) {
    unsafe {
        let p = get(work, POS, u);
        if p < work.n_conf {
            work.n_conf -= 1;
            swap(work, p, work.n_conf);
        }
    }
}

#[device]
unsafe fn remove(work: &mut Work, u: u32) {
    unsafe {
        let v = get(work, SIGMA, u);
        if v == NONE {
            return;
        }
        set(work, SIGMA, u, NONE);
        work.unassigned += 1;
        let degree = get(work, DEG, v) - 1;
        set(work, DEG, v, degree);
        // 16-bit wrapping sum of the vertices on v
        let rest = get(work, OCC, v).wrapping_sub(u) & 0xFFFF;
        set(work, OCC, v, rest);
        if degree >= 1 {
            mark_clear(work, u);
            if degree == 1 {
                mark_clear(work, rest);
            }
        }
    }
}

#[device]
unsafe fn place(work: &mut Work, u: u32, v: u32) {
    unsafe {
        set(work, SIGMA, u, v);
        work.unassigned -= 1;
        let degree = get(work, DEG, v) + 1;
        set(work, DEG, v, degree);
        let occupants = get(work, OCC, v);
        if degree == 2 {
            mark_conflicted(work, occupants);
            mark_conflicted(work, u);
        } else if degree > 2 {
            mark_conflicted(work, u);
        }
        set(work, OCC, v, occupants.wrapping_add(u) & 0xFFFF);
    }
}

/// A uniform integer below `bound` from 32 random bits.
#[device]
fn below(bits: u32, bound: u32) -> u32 {
    ((bits as u64 * bound as u64) >> 32) as u32
}

/// High 64 bits of the 64x64 product, by 32-bit limbs.
#[device]
fn mul_high(a: u64, b: u64) -> u64 {
    let (a0, a1) = (a & 0xFFFF_FFFF, a >> 32);
    let (b0, b1) = (b & 0xFFFF_FFFF, b >> 32);
    let p01 = a0 * b1;
    let p10 = a1 * b0;
    let middle = ((a0 * b0) >> 32) + (p01 & 0xFFFF_FFFF) + (p10 & 0xFFFF_FFFF);
    a1 * b1 + (p01 >> 32) + (p10 >> 32) + (middle >> 32)
}

/// One ring of the dynamics: pick the vertex whose clock rang, unassign it,
/// and redraw it from its row unless it stays unassigned.
#[device]
unsafe fn ring(
    work: &mut Work,
    rng: &mut Philox,
    table: *const u64,
    rate: u64,
    conflicted_rate: u64,
    unassign_below: u64,
) {
    let n = work.n as u32;
    let conflicted = mul_high(philox_next_u64(rng), rate) < conflicted_rate;
    let bits = philox_next(rng);
    let position = if conflicted { below(bits, work.n_conf) } else { work.n_conf + below(bits, n - work.n_conf) };
    unsafe {
        let u = get(work, LIST, position);
        remove(work, u);
        if philox_next_u64(rng) >= unassign_below {
            let i = below(philox_next(rng), n);
            let entry = *table.add(u as usize * work.n + i as usize);
            let draw = philox_next_u64(rng) >> 16;
            let v = if draw < (entry >> 16) { i } else { (entry & 0xFFFF) as u32 };
            place(work, u, v);
        }
    }
}

/// This thread's working arrays: a lane of the block's shared memory when
/// `staged`, else its slot of `scratch`.
#[device]
unsafe fn bind(scratch: *mut u16, slot: usize, n: usize, staged: u32) -> Work {
    unsafe {
        if staged != 0 {
            let shared: *mut u16 = DynamicSharedArray::<u16>::get();
            Work {
                base: shared.add(thread::threadIdx_x() as usize),
                stride: thread::blockDim_x() as usize,
                n,
                n_conf: 0,
                unassigned: 0,
            }
        } else {
            Work { base: scratch.add(slot * ARRAYS * n), stride: 1, n, n_conf: 0, unassigned: 0 }
        }
    }
}

/// An exponential waiting time of rate 1 from 32 random bits.
#[device]
fn exponential(bits: u32) -> f32 {
    -((bits as f32 + 0.5) * 2.328_306_4e-10).ln()
}

#[cuda_module]
mod kernels {
    use super::*;

    /// The mutation kernel K_t of `permanent::sampler::mutate` on every
    /// particle: relaxed-assignment dynamics at activity J until
    /// `perfect_time` has been spent in perfect matchings.
    ///
    /// alias[u n + i]: (48-bit keep threshold << 16) | alias column.
    /// unassign_below: P(unassign) 2^64. fast: 1 + J.
    #[kernel(unchecked_indexing)]
    #[allow(clippy::too_many_arguments)]
    pub fn mutate(
        alias: &[u64],
        seeds: &[u64],
        mut sigma: DisjointSlice<u16>,
        mut scratch: DisjointSlice<u16>,
        mut events_out: DisjointSlice<u64>,
        mut time_out: DisjointSlice<f32>,
        n: u32,
        count: u32,
        staged: u32,
        perfect_time: f32,
        fast: u64,
        unassign_below: u64,
        event_cap: u64,
    ) {
        let index = thread::index_1d();
        let slot = index.get();
        if slot >= count as usize {
            return;
        }
        let size = n as usize;
        let mut work = unsafe { bind(scratch.as_mut_ptr(), slot, size, staged) };
        let row = unsafe { sigma.as_mut_ptr().add(slot * size) };
        let mut u = 0u32;
        while u < n {
            unsafe {
                let v = *row.add(u as usize) as u32;
                set(&work, SIGMA, u, v);
                set(&work, DEG, u, 1);
                set(&work, OCC, v, u);
                set(&work, LIST, u, u);
                set(&work, POS, u, u);
            }
            u += 1;
        }

        let mut rng: Philox = philox_new(seeds[slot]);
        let table = alias.as_ptr();
        let mut remaining = perfect_time;
        let mut time = 0.0f32;
        let mut events = 0u64;
        let mut over_budget = false;
        loop {
            let conflicted_rate = work.n_conf as u64 * fast;
            let rate = (n - work.n_conf) as u64 + conflicted_rate;
            let dt = exponential(philox_next(&mut rng)) / rate as f32;
            if work.n_conf == 0 && work.unassigned == 0 {
                if dt >= remaining {
                    time += remaining;
                    break;
                }
                remaining -= dt;
            }
            time += dt;
            events += 1;
            if events > event_cap {
                over_budget = true;
                break;
            }
            unsafe { ring(&mut work, &mut rng, table, rate, conflicted_rate, unassign_below) };
        }

        if !over_budget {
            let mut u = 0u32;
            while u < n {
                unsafe { *row.add(u as usize) = get(&work, SIGMA, u) as u16 };
                u += 1;
            }
        }
        if let Some(out) = events_out.get_mut(index) {
            *out = if over_budget { events | OVER_BUDGET } else { events };
        }
        if let Some(out) = time_out.get_mut(thread::index_1d()) {
            *out = time;
        }
    }

    /// One trial of the annealing sampler (`permanent::sampler::run_trial`)
    /// per slot: from the all-unassigned state through every activity
    /// stage. A trial that ends on a perfect matching writes it to particle
    /// `targets[slot]` and sets PERFECT in its event count.
    ///
    /// unassign[stage]: P(unassign) 2^64 at that stage's activity.
    /// ticks_per_time: STAGE_TICKS / stage time.
    #[kernel(unchecked_indexing)]
    #[allow(clippy::too_many_arguments)]
    pub fn anneal(
        alias: &[u64],
        unassign: &[u64],
        seeds: &[u64],
        targets: &[u32],
        mut sigma: DisjointSlice<u16>,
        mut scratch: DisjointSlice<u16>,
        mut events_out: DisjointSlice<u64>,
        n: u32,
        count: u32,
        staged: u32,
        stages: u32,
        ticks_per_time: f32,
        fast: u64,
        event_cap: u64,
    ) {
        let index = thread::index_1d();
        let slot = index.get();
        if slot >= count as usize {
            return;
        }
        let size = n as usize;
        let mut work = unsafe { bind(scratch.as_mut_ptr(), slot, size, staged) };
        work.unassigned = n;
        let mut u = 0u32;
        while u < n {
            unsafe {
                set(&work, SIGMA, u, NONE);
                set(&work, DEG, u, 0);
                set(&work, OCC, u, 0);
                set(&work, LIST, u, u);
                set(&work, POS, u, u);
            }
            u += 1;
        }

        let mut rng: Philox = philox_new(seeds[slot]);
        let table = alias.as_ptr();
        // One flat loop over events and stage changes, so the threads of a
        // warp stay in the same code whatever stage each has reached.
        let mut stage = 0u32;
        let mut ticks = 0u64;
        let mut unassign_below = unassign[0];
        let mut events = 0u64;
        let mut outcome = 0u64;
        loop {
            let conflicted_rate = work.n_conf as u64 * fast;
            let rate = (n - work.n_conf) as u64 + conflicted_rate;
            ticks += (exponential(philox_next(&mut rng)) * (ticks_per_time / rate as f32)) as u64;
            if ticks > STAGE_TICKS {
                stage += 1;
                if stage == stages {
                    break;
                }
                ticks = 0;
                unassign_below = unassign[stage as usize];
                continue;
            }
            events += 1;
            if events > event_cap {
                outcome = OVER_BUDGET;
                break;
            }
            unsafe { ring(&mut work, &mut rng, table, rate, conflicted_rate, unassign_below) };
        }

        if outcome == 0 && work.n_conf == 0 && work.unassigned == 0 {
            outcome = PERFECT;
            let row = unsafe { sigma.as_mut_ptr().add(targets[slot] as usize * size) };
            let mut u = 0u32;
            while u < n {
                unsafe { *row.add(u as usize) = get(&work, SIGMA, u) as u16 };
                u += 1;
            }
        }
        if let Some(out) = events_out.get_mut(index) {
            *out = events | outcome;
        }
    }

    /// Energy and regularization log-gap of every particle.
    #[kernel(unchecked_indexing)]
    pub fn measure(
        energy: &[f64],
        log_gap: &[f64],
        sigma: &[u16],
        mut energy_out: DisjointSlice<f64>,
        mut gap_out: DisjointSlice<f64>,
        n: u32,
        count: u32,
    ) {
        let index = thread::index_1d();
        let slot = index.get();
        if slot >= count as usize {
            return;
        }
        let size = n as usize;
        let mut h = 0.0f64;
        let mut gap = 0.0f64;
        let mut u = 0usize;
        while u < size {
            let cell = u * size + sigma[slot * size + u] as usize;
            h += energy[cell];
            gap += log_gap[cell];
            u += 1;
        }
        if let Some(out) = energy_out.get_mut(index) {
            *out = h;
        }
        if let Some(out) = gap_out.get_mut(thread::index_1d()) {
            *out = gap;
        }
    }

    /// Resampling: particle p of `next` becomes a copy of particle
    /// `ancestors[p]` of `sigma`.
    #[kernel(unchecked_indexing)]
    pub fn gather(sigma: &[u16], ancestors: &[u32], mut next: DisjointSlice<u16>, n: u32, count: u32) {
        let slot = thread::index_1d().get();
        if slot >= count as usize {
            return;
        }
        let size = n as usize;
        let from = ancestors[slot] as usize * size;
        let to = unsafe { next.as_mut_ptr().add(slot * size) };
        let mut u = 0usize;
        while u < size {
            unsafe { *to.add(u) = sigma[from + u] };
            u += 1;
        }
    }
}

/// Threads per block of the particle kernels: one warp, so blocks spread
/// evenly over the device.
const BLOCK: u32 = 32;
/// Shared memory a block may use without opting in to more.
const SHARED_LIMIT: usize = 48 * 1024;
/// Most particles resident at once, and the device memory they may take.
const MAX_PARTICLES: usize = 1 << 20;
const MEMORY_BUDGET: usize = 8 << 30;
/// The kernels index vertices with 16-bit words; 0xFFFF marks "unassigned".
const MAX_N: usize = 0xFFFE;

struct Cuda<'a> {
    scaled: &'a Scaled,
    name: String,
    stream: Arc<CudaStream>,
    module: kernels::LoadedModule,
    /// The CPU engine, for the operations that have no kernel yet.
    cpu: Cpu<'a>,
    first_id: u64,
    count: usize,
    /// Working arrays fit a block's shared memory.
    staged: bool,
    energy: DeviceBuffer<f64>,
    log_gap: DeviceBuffer<f64>,
    sigma: DeviceBuffer<u16>,
    /// Destination of the next resampling; swapped with `sigma` after it.
    next: DeviceBuffer<u16>,
    /// Working arrays when they do not fit shared memory.
    scratch: DeviceBuffer<u16>,
    events: DeviceBuffer<u64>,
    time: DeviceBuffer<f32>,
    energy_out: DeviceBuffer<f64>,
    gap_out: DeviceBuffer<f64>,
}

impl<'a> Cuda<'a> {
    fn new(scaled: &'a Scaled, ordinal: usize) -> Result<Self> {
        let n = scaled.n;
        ensure!(n <= MAX_N, "the CUDA engine supports n up to {MAX_N}");
        let ctx = CudaContext::new(ordinal).with_context(|| format!("opening CUDA device {ordinal}"))?;
        let stream = ctx.default_stream();
        let module = kernels::load(&ctx).context("loading the kernels")?;
        let device = ctx.device_name()?;
        let staged = BLOCK as usize * ARRAYS * n * size_of::<u16>() <= SHARED_LIMIT;
        info!(%device, staged, "CUDA engine ready");
        Ok(Cuda {
            scaled,
            name: format!("cuda:{ordinal} ({device})"),
            module,
            cpu: Cpu::new(scaled),
            first_id: 0,
            count: 0,
            staged,
            energy: DeviceBuffer::from_host(&stream, &scaled.energy)?,
            log_gap: DeviceBuffer::from_host(&stream, &scaled.log_gap)?,
            sigma: DeviceBuffer::zeroed(&stream, 1)?,
            next: DeviceBuffer::zeroed(&stream, 1)?,
            scratch: DeviceBuffer::zeroed(&stream, 1)?,
            events: DeviceBuffer::zeroed(&stream, 1)?,
            time: DeviceBuffer::zeroed(&stream, 1)?,
            energy_out: DeviceBuffer::zeroed(&stream, 1)?,
            gap_out: DeviceBuffer::zeroed(&stream, 1)?,
            stream,
        })
    }

    /// Launch shape for `threads` particle threads; `staged` asks for the
    /// block's shared memory.
    fn config(&self, threads: usize, staged: bool) -> LaunchConfig {
        let shared = if staged { BLOCK as usize * ARRAYS * self.scaled.n * size_of::<u16>() } else { 0 };
        LaunchConfig {
            grid_dim: ((threads as u32).div_ceil(BLOCK), 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: shared as u32,
        }
    }

    /// The row alias tables in the kernels' packed form.
    fn pack_alias(&self, tables: &Tables) -> Result<DeviceBuffer<u64>> {
        let (prob, alias) = tables.row_alias();
        const ONE: f64 = (1u64 << 48) as f64;
        let packed: Vec<u64> = prob
            .iter()
            .zip(alias)
            .map(|(&p, &a)| (((p * ONE) as u64).min((1 << 48) - 1) << 16) | a as u64)
            .collect();
        Ok(DeviceBuffer::from_host(&self.stream, &packed)?)
    }

    /// Size the per-particle buffers for `count` particles.
    fn resize(&mut self, count: usize) -> Result<()> {
        if count == self.count {
            return Ok(());
        }
        let n = self.scaled.n;
        self.count = count;
        self.sigma = DeviceBuffer::zeroed(&self.stream, count * n)?;
        self.next = DeviceBuffer::zeroed(&self.stream, count * n)?;
        let scratch = if self.staged { 1 } else { count * ARRAYS * n };
        self.scratch = DeviceBuffer::zeroed(&self.stream, scratch)?;
        self.events = DeviceBuffer::zeroed(&self.stream, count)?;
        self.time = DeviceBuffer::zeroed(&self.stream, count)?;
        self.energy_out = DeviceBuffer::zeroed(&self.stream, count)?;
        self.gap_out = DeviceBuffer::zeroed(&self.stream, count)?;
        Ok(())
    }

    /// Bring the device population to the host engine.
    fn to_host(&mut self) -> Result<()> {
        let words = self.sigma.to_host_vec(&self.stream)?;
        let perms: Vec<u32> = words.iter().map(|&w| w as u32).collect();
        self.cpu.load(&perms, self.first_id)
    }

    /// Replace the device population by the host engine's.
    fn from_host(&mut self) -> Result<()> {
        let perms = self.cpu.particles().to_vec();
        self.upload(&perms)
    }

    fn upload(&mut self, perms: &[u32]) -> Result<()> {
        self.resize(perms.len() / self.scaled.n)?;
        let words: Vec<u16> = perms.iter().map(|&v| v as u16).collect();
        self.sigma = DeviceBuffer::from_host(&self.stream, &words)?;
        Ok(())
    }

    fn measure(&mut self) -> Result<()> {
        let config = self.config(self.count, false);
        // SAFETY: the arguments match the kernel, and every per-particle
        // buffer holds `count` entries.
        unsafe {
            self.module.measure(
                self.stream.as_ref(),
                config,
                &self.energy,
                &self.log_gap,
                &self.sigma,
                &mut self.energy_out,
                &mut self.gap_out,
                self.scaled.n as u32,
                self.count as u32,
            )
        }?;
        Ok(())
    }
}

impl Engine for Cuda<'_> {
    fn name(&self) -> String {
        self.name.clone()
    }

    fn capacity(&self) -> usize {
        let n = self.scaled.n;
        let per_particle = (2 + if self.staged { 0 } else { ARRAYS }) * n * size_of::<u16>() + 32;
        MAX_PARTICLES.min(MEMORY_BUDGET / per_particle)
    }

    fn load(&mut self, perms: &[u32], first_id: u64) -> Result<()> {
        self.first_id = first_id;
        self.upload(perms)
    }

    fn anneal(
        &mut self,
        tables: &Tables,
        stages: &StageSchedule,
        stage_time: f64,
        max_trials: u64,
        count: usize,
        first_id: u64,
        stream: Stream,
    ) -> Result<SampleCost> {
        let n = self.scaled.n;
        self.first_id = first_id;
        self.resize(count)?;
        let alias = self.pack_alias(tables)?;
        // P(unassign) = 1 / (1 + a) as a 64-bit threshold; a > 0 keeps it below 2^64
        let unassign: Vec<u64> =
            stages.activities.iter().map(|a| (18446744073709551616.0 / (1.0 + a)) as u64).collect();
        let unassign = DeviceBuffer::from_host(&self.stream, &unassign)?;
        let ticks_per_time = (STAGE_TICKS as f64 / stage_time) as f32;
        let event_cap = stages.event_cap(n, stage_time);
        let mut cost = SampleCost::default();
        // A trial that misses is run again with a fresh stream, like the
        // CPU sampler; only the particles still missing are relaunched.
        let mut pending: Vec<u32> = (0..count as u32).collect();
        for trial in 0..max_trials {
            if pending.is_empty() {
                break;
            }
            let seeds: Vec<u64> = pending
                .iter()
                .map(|&p| {
                    let sample = stream_seed(stream.seed, stream.phase, stream.level, first_id + p as u64);
                    mix(sample.wrapping_add(trial))
                })
                .collect();
            let seeds = DeviceBuffer::from_host(&self.stream, &seeds)?;
            let targets = DeviceBuffer::from_host(&self.stream, &pending)?;
            let config = self.config(pending.len(), self.staged);
            // SAFETY: the arguments match the kernel; the targets are
            // distinct particles below `count`, and every per-slot buffer
            // holds at least `pending.len()` entries.
            unsafe {
                self.module.anneal(
                    self.stream.as_ref(),
                    config,
                    &alias,
                    &unassign,
                    &seeds,
                    &targets,
                    &mut self.sigma,
                    &mut self.scratch,
                    &mut self.events,
                    n as u32,
                    pending.len() as u32,
                    self.staged as u32,
                    stages.stages() as u32,
                    ticks_per_time,
                    1 + stages.penalty as u64,
                    event_cap,
                )
            }?;
            let outcomes = self.events.to_host_vec(&self.stream)?;
            let mut missing = Vec::new();
            for (&particle, outcome) in pending.iter().zip(outcomes) {
                cost.trials += 1;
                cost.events += outcome & !(OVER_BUDGET | PERFECT);
                if outcome & PERFECT != 0 {
                    cost.accepted += 1;
                } else {
                    cost.over_budget += (outcome & OVER_BUDGET != 0) as u64;
                    missing.push(particle);
                }
            }
            pending = missing;
        }
        ensure!(
            pending.is_empty(),
            "the sampler could not produce {} of {count} perfect matchings; the anchor weights are out of range",
            pending.len()
        );
        Ok(cost)
    }

    fn mutate(&mut self, tables: &Tables, perfect_time: f64, stream: Stream) -> Result<MutateCost> {
        let n = self.scaled.n;
        let alias = self.pack_alias(tables)?;
        let seeds: Vec<u64> = (0..self.count as u64)
            .map(|p| stream_seed(stream.seed, stream.phase, stream.level, self.first_id + p))
            .collect();
        let seeds = DeviceBuffer::from_host(&self.stream, &seeds)?;
        let penalty = penalty(n);
        // P(unassign) = 1 / (1 + J) as a 64-bit threshold
        let unassign_below = (18446744073709551616.0 / (1.0 + penalty)) as u64;
        let config = self.config(self.count, self.staged);
        // SAFETY: the arguments match the kernel; `sigma` holds count rows
        // of n, `scratch` the working arrays when they are not staged, and
        // the dynamic shared memory the block's working arrays when they are.
        unsafe {
            self.module.mutate(
                self.stream.as_ref(),
                config,
                &alias,
                &seeds,
                &mut self.sigma,
                &mut self.scratch,
                &mut self.events,
                &mut self.time,
                n as u32,
                self.count as u32,
                self.staged as u32,
                perfect_time as f32,
                1 + penalty as u64,
                unassign_below,
                mutate_event_cap(n, perfect_time),
            )
        }?;
        let events = self.events.to_host_vec(&self.stream)?;
        let time = self.time.to_host_vec(&self.stream)?;
        let mut cost = MutateCost::default();
        for (events, time) in events.iter().zip(time) {
            cost.events += events & !OVER_BUDGET;
            cost.over_budget += (events & OVER_BUDGET != 0) as u64;
            cost.time += time as f64;
        }
        Ok(cost)
    }

    fn energies(&mut self) -> Result<Vec<f64>> {
        self.measure()?;
        Ok(self.energy_out.to_host_vec(&self.stream)?)
    }

    fn log_gaps(&mut self) -> Result<Vec<f64>> {
        self.measure()?;
        Ok(self.gap_out.to_host_vec(&self.stream)?)
    }

    fn resample(&mut self, ancestors: &[u32]) -> Result<()> {
        let ancestors = DeviceBuffer::from_host(&self.stream, ancestors)?;
        let config = self.config(self.count, false);
        // SAFETY: the arguments match the kernel; both populations hold
        // count rows of n and every ancestor is below count.
        unsafe {
            self.module.gather(
                self.stream.as_ref(),
                config,
                &self.sigma,
                &ancestors,
                &mut self.next,
                self.scaled.n as u32,
                self.count as u32,
            )
        }?;
        std::mem::swap(&mut self.sigma, &mut self.next);
        Ok(())
    }

    fn far(
        &mut self,
        tables: &Tables,
        cycles: u64,
        max_steps: u64,
        groups: usize,
        stream: Stream,
    ) -> Result<FarRound> {
        self.to_host()?;
        let round = self.cpu.far(tables, cycles, max_steps, groups, stream)?;
        self.from_host()?;
        Ok(round)
    }
}

fn main() -> Result<()> {
    permanent::cli::main(|scaled, cli, comm| {
        // one GPU per rank, wrapping around when there are fewer GPUs
        let devices = CudaContext::device_count()? as usize;
        ensure!(devices > 0, "no CUDA device");
        Ok(Box::new(Cuda::new(scaled, (cli.device + comm.rank()) % devices)?))
    })
}

#[cfg(test)]
mod test {
    use super::*;
    use permanent::problem::Problem;
    use permanent::scaling::{Prepared, prepare};

    fn instance() -> Scaled {
        let problem = Problem {
            n: 4,
            a: vec![
                3.0, 1.0, 0.0, 0.5, //
                0.2, 2.0, 1.0, 0.0, //
                0.0, 0.7, 4.0, 1.0, //
                1.0, 0.0, 0.3, 2.5,
            ],
        };
        match prepare(&problem) {
            Prepared::Scaled(scaled) => scaled,
            Prepared::Zero => unreachable!(),
        }
    }

    fn permutations(n: usize) -> Vec<Vec<u32>> {
        fn extend(n: usize, current: &mut Vec<u32>, out: &mut Vec<Vec<u32>>) {
            if current.len() == n {
                out.push(current.clone());
                return;
            }
            for v in 0..n as u32 {
                if !current.contains(&v) {
                    current.push(v);
                    extend(n, current, out);
                    current.pop();
                }
            }
        }
        let mut out = Vec::new();
        extend(n, &mut Vec::new(), &mut out);
        out
    }

    /// The population's energies must be distributed as under
    /// lambda(M) / sum lambda; matchings are told apart by their energy.
    fn check(scaled: &Scaled, beta: f64, energies: &[f64]) {
        let all: Vec<f64> = permutations(scaled.n).iter().map(|p| scaled.energy_of(p)).collect();
        let total: f64 = all.iter().map(|h| (-beta * h).exp()).sum();
        let mut bins: Vec<(f64, f64, u64)> = Vec::new();
        for h in &all {
            let mass = (-beta * h).exp() / total;
            match bins.iter_mut().find(|bin| (bin.0 - h).abs() < 1e-9) {
                Some(bin) => bin.1 += mass,
                None => bins.push((*h, mass, 0)),
            }
        }
        for h in energies {
            bins.iter_mut().find(|bin| (bin.0 - h).abs() < 1e-9).expect("energy of no matching").2 += 1;
        }
        let draws = energies.len() as f64;
        for (h, expected, observed) in bins {
            let observed = observed as f64 / draws;
            let sigma = (expected * (1.0 - expected) / draws).sqrt();
            assert!(
                (observed - expected).abs() < 5.0 * sigma + 1e-5,
                "energy {h:.4}: expected {expected:.5}, observed {observed:.5}"
            );
        }
    }

    const BETA: f64 = 0.6;
    // deliberately far from ideal: the target may not depend on the weights
    const LOG_W: [f64; 4] = [0.3, -0.4, 1.0, 0.0];
    const COUNT: usize = 200_000;

    /// From one fixed matching, a long mutation lands on the target.
    #[test]
    fn a_long_mutation_reaches_the_target_distribution() {
        let scaled = instance();
        let tables = Tables::build(&scaled, BETA, &LOG_W, 0);
        let mut cuda = Cuda::new(&scaled, 0).unwrap();
        let start: Vec<u32> = (0..COUNT).flat_map(|_| [0, 1, 2, 3]).collect();
        cuda.load(&start, 0).unwrap();
        let cost = cuda.mutate(&tables, 24.0, Stream { seed: 5, phase: 1, level: 0 }).unwrap();
        assert_eq!(cost.over_budget, 0);
        check(&scaled, BETA, &cuda.energies().unwrap());
    }

    /// The sampler's accepted trials follow the target, and a short
    /// mutation of exact samples keeps it.
    #[test]
    fn the_sampler_and_a_short_mutation_keep_the_target_distribution() {
        let scaled = instance();
        let tables = Tables::build(&scaled, BETA, &LOG_W, 0);
        let stages = StageSchedule::new(scaled.n, 32.0);
        let stage_time = stages.rigorous_stage_time(1e-3);
        let mut cuda = Cuda::new(&scaled, 0).unwrap();
        let cost = cuda
            .anneal(&tables, &stages, stage_time, 64, COUNT, 0, Stream { seed: 5, phase: 2, level: 0 })
            .unwrap();
        assert_eq!(cost.over_budget, 0);
        assert!(COUNT as f64 / cost.trials as f64 > 0.2);
        check(&scaled, BETA, &cuda.energies().unwrap());
        let cost = cuda.mutate(&tables, 0.3, Stream { seed: 5, phase: 3, level: 0 }).unwrap();
        assert_eq!(cost.over_budget, 0);
        check(&scaled, BETA, &cuda.energies().unwrap());
    }

    /// Resampling copies whole particles.
    #[test]
    fn resampling_copies_particles() {
        let scaled = instance();
        let perms = permutations(scaled.n);
        let flat: Vec<u32> = perms.iter().flatten().copied().collect();
        let mut cuda = Cuda::new(&scaled, 0).unwrap();
        cuda.load(&flat, 0).unwrap();
        let before = cuda.energies().unwrap();
        let ancestors: Vec<u32> = (0..perms.len() as u32).rev().collect();
        cuda.resample(&ancestors).unwrap();
        let after = cuda.energies().unwrap();
        let expected: Vec<f64> = before.iter().rev().copied().collect();
        assert_eq!(after, expected);
        assert_eq!(before, perms.iter().map(|p| scaled.energy_of(p)).collect::<Vec<_>>());
    }
}
