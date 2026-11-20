//! The population resident on an NVIDIA GPU.
//!
//! The kernels are Rust, compiled for the device by cuda-oxide. One GPU
//! thread owns one particle. Between launches only the particles' matchings
//! (n 16-bit words each) stay in global memory; a launch stages each
//! particle's working state in the block's shared memory, so the event
//! loop touches global memory for the alias table alone.
//!
//! How many particles a multiprocessor runs at once is set by how much
//! shared memory each needs, so the working state is kept to the matching,
//! its inverse, and a short list of the vertices in a collision. The
//! penalty J keeps collisions rare at every activity, which is what makes a
//! short list enough; a particle that overflows it is treated like one that
//! hit its event cap. Once a warp's worth of state no longer fits shared
//! memory (n above about 370), the state lives in global memory instead:
//! it is small enough to sit in the device's cache, and that measures
//! faster than staging fewer particles per block.
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
//!  - The FAR learner tracks a matching's energy in 2^-24 fixed point, so
//!    the running sum never drifts.
//!
//! A launch lasts as long as its longest thread. FAR cycles have a long
//! tail, so a FAR launch gives every chain a step budget, and the few
//! chains that overrun it are finished on the host; every cycle still
//! completes, which is what the learner's identity needs.

use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result, ensure};
use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig, Stream as CudaStream};
use cuda_device::atomic::{AtomicOrdering, DeviceAtomicF64, DeviceAtomicU64};
use cuda_device::{DisjointSlice, DynamicSharedArray, device, kernel, thread};
use cuda_host::cuda_module;
use permanent_core::engine::{Engine, FarRound, Stream};
use permanent_core::far;
use permanent_core::sampler::{MutateCost, SampleCost, StageSchedule, mutate_event_cap, penalty};
use permanent_core::scaling::Scaled;
use permanent_core::stats::{mix, stream_seed};
use permanent_core::tables::Tables;
use rand::SeedableRng;
use rand::rngs::SmallRng;
use rayon::prelude::*;
use tracing::{debug, info};

mod philox;
use philox::{Philox, philox_new, philox_next, philox_next_u64};

/// A particle's working state: the matching sigma (row u to column, or
/// NONE), its inverse (column v to its row, NONE when free, MULTI when two
/// or more rows are on it), then the rows in a collision.
const SIGMA: usize = 0;
const COL: usize = 1;
/// The FAR chain's matching, in the same two arrays.
const ROW: usize = 0;
/// Rows in a collision that the list can hold.
const CONFLICTS: usize = 32;
/// Fixed-point scale of the FAR chain's energies.
const ENERGY_ONE: f64 = (1u64 << 24) as f64;
const NONE: u32 = 0xFFFF;
const MULTI: u32 = 0xFFFE;
/// Set in a particle's event count when its move hit the event cap.
const OVER_BUDGET: u64 = 1 << 63;
/// Set in a trial's event count when it ended on a perfect matching.
const PERFECT: u64 = 1 << 62;
/// Stage time of the sampler, in ticks.
const STAGE_TICKS: u64 = 1 << 40;

/// Words of working state per particle.
const fn lane_words(n: usize) -> usize {
    2 * n + CONFLICTS
}

/// One particle's working state. Element i of array a lives at
/// `base + (a n + i) stride`, and entry i of the collision list at
/// `base + (2 n + i) stride`: stride 1 in global memory, the block size in
/// shared memory, where it puts the lanes of a warp in distinct banks.
#[derive(Clone, Copy)]
struct Work {
    base: *mut u16,
    stride: usize,
    n: usize,
    n_conf: u32,
    unassigned: u32,
    /// The collision list was full when a row had to be added.
    overflow: bool,
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
unsafe fn conflict(work: &Work, index: u32) -> u32 {
    unsafe { get(work, 2, index) }
}

#[device]
unsafe fn is_conflicted(work: &Work, u: u32) -> bool {
    let mut i = 0u32;
    while i < work.n_conf {
        if unsafe { conflict(work, i) } == u {
            return true;
        }
        i += 1;
    }
    false
}

#[device]
unsafe fn add_conflict(work: &mut Work, u: u32) {
    if work.n_conf as usize == CONFLICTS {
        work.overflow = true;
        return;
    }
    unsafe { set(work, 2, work.n_conf, u) };
    work.n_conf += 1;
}

#[device]
unsafe fn drop_conflict(work: &mut Work, u: u32) {
    let mut i = 0u32;
    while i < work.n_conf {
        unsafe {
            if conflict(work, i) == u {
                work.n_conf -= 1;
                set(work, 2, i, conflict(work, work.n_conf));
                return;
            }
        }
        i += 1;
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
        if get(work, COL, v) != MULTI {
            // u was alone on v
            set(work, COL, v, NONE);
            return;
        }
        drop_conflict(work, u);
        // if one row is left on v, it is no longer in a collision
        let mut left = 0u32;
        let mut last = NONE;
        let mut i = 0u32;
        while i < work.n_conf {
            let w = conflict(work, i);
            if get(work, SIGMA, w) == v {
                left += 1;
                last = w;
            }
            i += 1;
        }
        if left == 1 {
            drop_conflict(work, last);
            set(work, COL, v, last);
        }
    }
}

#[device]
unsafe fn place(work: &mut Work, u: u32, v: u32) {
    unsafe {
        set(work, SIGMA, u, v);
        work.unassigned -= 1;
        let occupant = get(work, COL, v);
        if occupant == NONE {
            set(work, COL, v, u);
        } else if occupant == MULTI {
            add_conflict(work, u);
        } else {
            set(work, COL, v, MULTI);
            add_conflict(work, occupant);
            add_conflict(work, u);
        }
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

/// Draw from line `line` of a packed alias table.
#[device]
unsafe fn alias_draw(table: *const u64, line: u32, n: u32, rng: &mut Philox) -> u32 {
    let i = below(philox_next(rng), n);
    let entry = unsafe { *table.add(line as usize * n as usize + i as usize) };
    let draw = philox_next_u64(rng) >> 16;
    if draw < (entry >> 16) { i } else { (entry & 0xFFFF) as u32 }
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
    unsafe {
        let mut u;
        if conflicted {
            u = conflict(work, below(philox_next(rng), work.n_conf));
        } else {
            // uniform over the rows outside the list, by rejection
            loop {
                u = below(philox_next(rng), n);
                if !is_conflicted(work, u) {
                    break;
                }
            }
        }
        remove(work, u);
        if philox_next_u64(rng) >= unassign_below {
            let v = alias_draw(table, u, n, rng);
            place(work, u, v);
        }
    }
}

/// This thread's working state: a lane of the block's shared memory when
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
                overflow: false,
            }
        } else {
            Work { base: scratch.add(slot * lane_words(n)), stride: 1, n, n_conf: 0, unassigned: 0, overflow: false }
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

    /// The mutation kernel K_t of `permanent_core::sampler::mutate` on every
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
                set(&work, COL, v, u);
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
            if work.overflow {
                over_budget = true;
                break;
            }
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

    /// One trial of the annealing sampler (`permanent_core::sampler::run_trial`)
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
                set(&work, COL, u, NONE);
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
            if work.overflow {
                outcome = OVER_BUDGET;
                break;
            }
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

    /// The FAR chain of `permanent_core::far::run` on every particle: `cycles`
    /// cycles from its perfect matching, or `budget` steps if that comes
    /// first. The particle is replaced by where the chain stopped: a perfect
    /// matching, or one with NONE at its left hole. `status_out` holds
    /// steps | left hole << 32 | right hole << 48, with NONE for no holes.
    /// Particle p adds its visits to N(anchor, v) to
    /// `visits[(p % groups) n + v]` and their energies to `visit_energy[v]`.
    ///
    /// moves[2 (u n + v)], moves[2 (u n + v) + 1]: with holes (u, v), the
    /// chain adds the edge below the first threshold and slides left below
    /// the second, as fractions of 2^64; a right slide is as likely as a
    /// left one. energy: 2^-24 fixed point.
    #[kernel(unchecked_indexing)]
    #[allow(clippy::too_many_arguments)]
    pub fn far(
        row_alias: &[u64],
        col_alias: &[u64],
        moves: &[u64],
        energy: &[i64],
        seeds: &[u64],
        visits: &[DeviceAtomicU64],
        visit_energy: &[DeviceAtomicF64],
        mut sigma: DisjointSlice<u16>,
        mut scratch: DisjointSlice<u16>,
        mut cycles_out: DisjointSlice<u64>,
        mut status_out: DisjointSlice<u64>,
        mut energy_out: DisjointSlice<i64>,
        n: u32,
        count: u32,
        staged: u32,
        anchor: u32,
        groups: u32,
        cycles: u64,
        budget: u64,
    ) {
        let index = thread::index_1d();
        let slot = index.get();
        if slot >= count as usize {
            return;
        }
        let size = n as usize;
        let work = unsafe { bind(scratch.as_mut_ptr(), slot, size, staged) };
        let stored = unsafe { sigma.as_mut_ptr().add(slot * size) };
        let mut h = 0i64;
        let mut u = 0u32;
        while u < n {
            unsafe {
                let v = *stored.add(u as usize) as u32;
                set(&work, ROW, u, v);
                set(&work, COL, v, u);
                h += energy[u as usize * size + v as usize];
            }
            u += 1;
        }

        let mut rng: Philox = philox_new(seeds[slot]);
        let group = (slot as u32 % groups) as usize;
        let mut hole_u = NONE;
        let mut hole_v = 0u32;
        let mut perfect_steps = 0u64;
        let mut perfect_energy = 0i64;
        let mut steps = 0u64;
        loop {
            if steps >= budget {
                break;
            }
            if hole_u == NONE {
                if perfect_steps == cycles {
                    break;
                }
                perfect_steps += 1;
                perfect_energy += h;
                steps += 1;
                if philox_next(&mut rng) & 1 == 0 {
                    let u = below(philox_next(&mut rng), n);
                    unsafe {
                        let v = get(&work, ROW, u);
                        set(&work, ROW, u, NONE);
                        set(&work, COL, v, NONE);
                        h -= energy[u as usize * size + v as usize];
                        hole_u = u;
                        hole_v = v;
                    }
                }
            } else {
                if hole_u == anchor {
                    visits[group * size + hole_v as usize].fetch_add(1, AtomicOrdering::Relaxed);
                    visit_energy[hole_v as usize].fetch_add(h as f64 / ENERGY_ONE, AtomicOrdering::Relaxed);
                }
                steps += 1;
                let cell = hole_u as usize * size + hole_v as usize;
                let add_below = moves[2 * cell];
                let left_below = moves[2 * cell + 1];
                let x = philox_next_u64(&mut rng);
                unsafe {
                    if x < add_below {
                        set(&work, ROW, hole_u, hole_v);
                        set(&work, COL, hole_v, hole_u);
                        h += energy[cell];
                        hole_u = NONE;
                    } else if x < left_below {
                        // left slide: the left hole takes over a matched column
                        let z = alias_draw(row_alias.as_ptr(), hole_u, n, &mut rng);
                        if z != hole_v {
                            let y = get(&work, COL, z);
                            set(&work, ROW, y, NONE);
                            set(&work, ROW, hole_u, z);
                            set(&work, COL, z, hole_u);
                            h += energy[hole_u as usize * size + z as usize] - energy[y as usize * size + z as usize];
                            hole_u = y;
                        }
                    } else if x - left_below < left_below - add_below {
                        // right slide: the right hole takes over a matched row
                        let y = alias_draw(col_alias.as_ptr(), hole_v, n, &mut rng);
                        if y != hole_u {
                            let z = get(&work, ROW, y);
                            set(&work, ROW, y, hole_v);
                            set(&work, COL, hole_v, y);
                            set(&work, COL, z, NONE);
                            h += energy[y as usize * size + hole_v as usize] - energy[y as usize * size + z as usize];
                            hole_v = z;
                        }
                    }
                }
            }
        }

        let mut u = 0u32;
        while u < n {
            unsafe { *stored.add(u as usize) = get(&work, ROW, u) as u16 };
            u += 1;
        }
        if let Some(out) = cycles_out.get_mut(index) {
            *out = perfect_steps;
        }
        if let Some(out) = status_out.get_mut(thread::index_1d()) {
            *out = steps | (hole_u as u64) << 32 | (hole_v as u64) << 48;
        }
        if let Some(out) = energy_out.get_mut(thread::index_1d()) {
            *out = perfect_energy;
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

    /// Row p of `rows` replaces particle `targets[p]` of `sigma`.
    #[kernel(unchecked_indexing)]
    pub fn scatter(rows: &[u16], targets: &[u32], mut sigma: DisjointSlice<u16>, n: u32, count: u32) {
        let slot = thread::index_1d().get();
        if slot >= count as usize {
            return;
        }
        let size = n as usize;
        let to = unsafe { sigma.as_mut_ptr().add(targets[slot] as usize * size) };
        let mut u = 0usize;
        while u < size {
            unsafe { *to.add(u) = rows[slot * size + u] };
            u += 1;
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

/// Whether a block of particles with `words` words of working state each
/// can keep that state in shared memory.
fn fits_shared(words: usize) -> bool {
    BLOCK as usize * words * size_of::<u16>() <= SHARED_LIMIT
}
/// Most particles resident at once, and the device memory they may take.
const MAX_PARTICLES: usize = 1 << 20;
const MEMORY_BUDGET: usize = 8 << 30;
/// The kernels index vertices with 16-bit words; the two highest values
/// mark "unassigned" and "more than one row here".
const MAX_N: usize = 0xFFFD;

pub struct Cuda<'a> {
    scaled: &'a Scaled,
    name: String,
    stream: Arc<CudaStream>,
    module: kernels::LoadedModule,
    first_id: u64,
    count: usize,
    energy: DeviceBuffer<f64>,
    log_gap: DeviceBuffer<f64>,
    /// The energies in the FAR kernel's fixed point.
    energy_fixed: DeviceBuffer<i64>,
    /// `Tables::id` of the row alias table on the device, and of the FAR
    /// tables (column alias, move thresholds); 0 when there is none.
    row_id: u64,
    far_id: u64,
    row_alias: DeviceBuffer<u64>,
    col_alias: DeviceBuffer<u64>,
    moves: DeviceBuffer<u64>,
    sigma: DeviceBuffer<u16>,
    /// Destination of the next resampling; swapped with `sigma` after it.
    next: DeviceBuffer<u16>,
    /// Working arrays when they do not fit shared memory.
    scratch: DeviceBuffer<u16>,
    events: DeviceBuffer<u64>,
    time: DeviceBuffer<f32>,
    energy_out: DeviceBuffer<f64>,
    gap_out: DeviceBuffer<f64>,
    cycles_out: DeviceBuffer<u64>,
    fixed_out: DeviceBuffer<i64>,
    /// Steps per cycle a FAR launch allows before a chain is handed to the
    /// host; tuned so that a few percent of the chains overrun.
    far_budget: f64,
    /// Seconds spent in each part of the FAR calls, for the debug log.
    far_seconds: FarSeconds,
}

#[derive(Debug, Default)]
struct FarSeconds {
    calls: u64,
    late_chains: u64,
    setup: f64,
    kernel: f64,
    readback: f64,
    host_chains: f64,
}

impl Drop for Cuda<'_> {
    fn drop(&mut self) {
        if self.far_seconds.calls > 0 {
            debug!(far = ?self.far_seconds, budget_per_cycle = self.far_budget, "CUDA engine FAR calls");
        }
    }
}

/// Alias tables in the kernels' packed form:
/// (48-bit keep threshold << 16) | alias index.
fn pack_alias((prob, alias): (&[f64], &[u32])) -> Vec<u64> {
    const ONE: f64 = (1u64 << 48) as f64;
    prob.iter().zip(alias).map(|(&p, &a)| (((p * ONE) as u64).min((1 << 48) - 1) << 16) | a as u64).collect()
}

/// A probability as a fraction of 2^64.
fn fraction(p: f64) -> u64 {
    (p * 18446744073709551616.0) as u64
}

impl<'a> Cuda<'a> {
    /// Number of CUDA devices on this host.
    pub fn device_count() -> Result<usize> {
        Ok(CudaContext::device_count()? as usize)
    }

    /// The engine on device `ordinal`.
    pub fn new(scaled: &'a Scaled, ordinal: usize) -> Result<Self> {
        let n = scaled.n;
        ensure!(n <= MAX_N, "the CUDA engine supports n up to {MAX_N}");
        let ctx = CudaContext::new(ordinal).with_context(|| format!("opening CUDA device {ordinal}"))?;
        let stream = ctx.default_stream();
        let module = kernels::load(&ctx).context("loading the kernels; device code is built by `cargo oxide`")?;
        let device = ctx.device_name()?;
        info!(%device, staged = fits_shared(lane_words(n)), far_staged = fits_shared(2 * n), "CUDA engine ready");
        Ok(Cuda {
            scaled,
            name: format!("cuda:{ordinal} ({device})"),
            module,
            first_id: 0,
            count: 0,
            energy: DeviceBuffer::from_host(&stream, &scaled.energy)?,
            log_gap: DeviceBuffer::from_host(&stream, &scaled.log_gap)?,
            energy_fixed: DeviceBuffer::from_host(
                &stream,
                &scaled.energy.iter().map(|e| (e * ENERGY_ONE).round() as i64).collect::<Vec<_>>(),
            )?,
            row_id: 0,
            far_id: 0,
            row_alias: DeviceBuffer::zeroed(&stream, 1)?,
            col_alias: DeviceBuffer::zeroed(&stream, 1)?,
            moves: DeviceBuffer::zeroed(&stream, 1)?,
            sigma: DeviceBuffer::zeroed(&stream, 1)?,
            next: DeviceBuffer::zeroed(&stream, 1)?,
            scratch: DeviceBuffer::zeroed(&stream, 1)?,
            events: DeviceBuffer::zeroed(&stream, 1)?,
            time: DeviceBuffer::zeroed(&stream, 1)?,
            energy_out: DeviceBuffer::zeroed(&stream, 1)?,
            gap_out: DeviceBuffer::zeroed(&stream, 1)?,
            cycles_out: DeviceBuffer::zeroed(&stream, 1)?,
            fixed_out: DeviceBuffer::zeroed(&stream, 1)?,
            far_budget: 64.0,
            far_seconds: FarSeconds::default(),
            stream,
        })
    }

    /// Launch shape for `threads` particle threads of `words` words of
    /// working state each, and whether that state is staged in shared
    /// memory (else it lives in `scratch`). 0 words is a kernel with no
    /// working state.
    fn config(&self, threads: usize, words: usize) -> (LaunchConfig, u32) {
        let staged = words > 0 && fits_shared(words);
        let shared = if staged { BLOCK as usize * words * size_of::<u16>() } else { 0 };
        let config = LaunchConfig {
            grid_dim: ((threads as u32).div_ceil(BLOCK), 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: shared as u32,
        };
        (config, staged as u32)
    }

    /// Whether some kernel keeps its working state in `scratch`.
    fn needs_scratch(&self) -> bool {
        !fits_shared(lane_words(self.scaled.n))
    }

    /// Put the row alias table of `tables` on the device, unless it is there.
    fn sync_row(&mut self, tables: &Tables) -> Result<()> {
        if self.row_id != tables.id {
            self.row_alias = DeviceBuffer::from_host(&self.stream, &pack_alias(tables.row_alias()))?;
            self.row_id = tables.id;
        }
        Ok(())
    }

    /// Likewise the tables only the FAR kernel reads.
    fn sync_far(&mut self, tables: &Tables) -> Result<()> {
        if self.far_id == tables.id {
            return Ok(());
        }
        let n = self.scaled.n;
        self.col_alias = DeviceBuffer::from_host(&self.stream, &pack_alias(tables.col_alias()))?;
        let mut moves = Vec::with_capacity(2 * n * n);
        for u in 0..n {
            for v in 0..n {
                // the move probabilities of far.rs
                let b = tables.b[u * n + v];
                let half_inv_w = 0.5 / tables.hole_weight(u, v);
                let add = n as f64 * b * half_inv_w;
                let slide = tables.s[v] * half_inv_w;
                moves.push(fraction(add));
                moves.push(fraction(add + slide));
            }
        }
        self.moves = DeviceBuffer::from_host(&self.stream, &moves)?;
        self.far_id = tables.id;
        Ok(())
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
        let scratch = if self.needs_scratch() { count * lane_words(n) } else { 1 };
        self.scratch = DeviceBuffer::zeroed(&self.stream, scratch)?;
        self.events = DeviceBuffer::zeroed(&self.stream, count)?;
        self.time = DeviceBuffer::zeroed(&self.stream, count)?;
        self.energy_out = DeviceBuffer::zeroed(&self.stream, count)?;
        self.gap_out = DeviceBuffer::zeroed(&self.stream, count)?;
        self.cycles_out = DeviceBuffer::zeroed(&self.stream, count)?;
        self.fixed_out = DeviceBuffer::zeroed(&self.stream, count)?;
        Ok(())
    }

    fn upload(&mut self, perms: &[u32]) -> Result<()> {
        self.resize(perms.len() / self.scaled.n)?;
        let words: Vec<u16> = perms.iter().map(|&v| v as u16).collect();
        self.sigma = DeviceBuffer::from_host(&self.stream, &words)?;
        Ok(())
    }

    fn measure(&mut self) -> Result<()> {
        let (config, _) = self.config(self.count, 0);
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
        let scratch = if self.needs_scratch() { lane_words(n) } else { 0 };
        let per_particle = (2 * n + scratch) * size_of::<u16>() + 48;
        MAX_PARTICLES.min(MEMORY_BUDGET / per_particle)
    }

    fn learn_particles(&self) -> usize {
        // enough chains that a learning round is about 16 cycles for each
        let n = self.scaled.n;
        (4 * n * n).clamp(4096, 1 << 18).min(self.capacity())
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
        self.sync_row(tables)?;
        // P(unassign) = 1 / (1 + a) as a 64-bit threshold; a > 0 keeps it below 2^64
        let unassign: Vec<u64> =
            stages.activities.iter().map(|a| fraction(1.0 / (1.0 + a))).collect();
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
            let (config, staged) = self.config(pending.len(), lane_words(n));
            // SAFETY: the arguments match the kernel; the targets are
            // distinct particles below `count`, and every per-slot buffer
            // holds at least `pending.len()` entries.
            unsafe {
                self.module.anneal(
                    self.stream.as_ref(),
                    config,
                    &self.row_alias,
                    &unassign,
                    &seeds,
                    &targets,
                    &mut self.sigma,
                    &mut self.scratch,
                    &mut self.events,
                    n as u32,
                    pending.len() as u32,
                    staged,
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
        self.sync_row(tables)?;
        let seeds: Vec<u64> = (0..self.count as u64)
            .map(|p| stream_seed(stream.seed, stream.phase, stream.level, self.first_id + p))
            .collect();
        let seeds = DeviceBuffer::from_host(&self.stream, &seeds)?;
        let penalty = penalty(n);
        let unassign_below = fraction(1.0 / (1.0 + penalty));
        let (config, staged) = self.config(self.count, lane_words(n));
        // SAFETY: the arguments match the kernel; `sigma` holds count rows
        // of n, `scratch` the working state when it is not staged, and the
        // dynamic shared memory the block's working state when it is.
        unsafe {
            self.module.mutate(
                self.stream.as_ref(),
                config,
                &self.row_alias,
                &seeds,
                &mut self.sigma,
                &mut self.scratch,
                &mut self.events,
                &mut self.time,
                n as u32,
                self.count as u32,
                staged,
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
        let (config, _) = self.config(self.count, 0);
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
        let n = self.scaled.n;
        let clock = Instant::now();
        self.sync_row(tables)?;
        self.sync_far(tables)?;
        let seeds: Vec<u64> = (0..self.count as u64)
            .map(|p| stream_seed(stream.seed, stream.phase, stream.level, self.first_id + p))
            .collect();
        let seeds = DeviceBuffer::from_host(&self.stream, &seeds)?;
        let visits = DeviceBuffer::<u64>::zeroed(&self.stream, groups * n)?.wrap_elem::<DeviceAtomicU64>();
        let visit_energy = DeviceBuffer::<f64>::zeroed(&self.stream, n)?.wrap_elem::<DeviceAtomicF64>();
        let budget = (self.far_budget * cycles as f64).ceil() as u64 + 32;
        // the FAR chain keeps a matching and its inverse, and no list
        let (config, staged) = self.config(self.count, 2 * n);
        self.far_seconds.calls += 1;
        self.far_seconds.setup += clock.elapsed().as_secs_f64();
        let clock = Instant::now();
        // SAFETY: the arguments match the kernel; the tables are n x n,
        // `visits` holds groups rows of n, and every per-particle buffer
        // holds `count` entries. Threads share only the atomic counters.
        unsafe {
            self.module.far(
                self.stream.as_ref(),
                config,
                &self.row_alias,
                &self.col_alias,
                &self.moves,
                &self.energy_fixed,
                &seeds,
                &visits,
                &visit_energy,
                &mut self.sigma,
                &mut self.scratch,
                &mut self.cycles_out,
                &mut self.events,
                &mut self.fixed_out,
                n as u32,
                self.count as u32,
                staged,
                tables.anchor as u32,
                groups as u32,
                cycles,
                budget,
            )
        }?;
        self.stream.synchronize()?;
        self.far_seconds.kernel += clock.elapsed().as_secs_f64();
        let clock = Instant::now();
        let done = self.cycles_out.to_host_vec(&self.stream)?;
        let status = self.events.to_host_vec(&self.stream)?;
        let chain_energy = self.fixed_out.to_host_vec(&self.stream)?;
        let visits = visits.peel_elem::<u64>().to_host_vec(&self.stream)?;
        let mut round = FarRound {
            cycles: vec![0; groups],
            visits: visits.chunks(n).map(|row| row.to_vec()).collect(),
            visit_energy: visit_energy.peel_elem::<f64>().to_host_vec(&self.stream)?,
            perfect_energy: 0.0,
            steps: 0,
            capped: 0,
        };
        let hole_of = |status: u64| ((status >> 32) & 0xFFFF, status >> 48);
        let mut late: Vec<u32> = Vec::new();
        for (particle, ((&done, &status), energy)) in done.iter().zip(&status).zip(chain_energy).enumerate() {
            round.cycles[particle % groups] += done;
            round.steps += status & 0xFFFF_FFFF;
            round.perfect_energy += energy as f64 / ENERGY_ONE;
            if done < cycles || hole_of(status).0 != NONE as u64 {
                late.push(particle as u32);
            }
        }
        // aim for about half a percent of late chains: fewer makes the
        // launch wait for the slowest, more moves the work to the host
        let share = late.len() as f64 / self.count as f64;
        if share > 0.01 {
            self.far_budget *= 1.5;
        } else if share < 0.002 {
            self.far_budget = (self.far_budget / 1.2).max(8.0);
        }
        self.far_seconds.readback += clock.elapsed().as_secs_f64();
        self.far_seconds.late_chains += late.len() as u64;
        if late.is_empty() {
            return Ok(round);
        }
        let clock = Instant::now();

        // Finish the late chains on the host, from where they stopped.
        let targets = DeviceBuffer::from_host(&self.stream, &late)?;
        let mut rows = DeviceBuffer::<u16>::zeroed(&self.stream, late.len() * n)?;
        let (config, _) = self.config(late.len(), 0);
        // SAFETY: the arguments match the kernel; `rows` holds one row per
        // late chain and every target is below `count`.
        unsafe {
            self.module.gather(self.stream.as_ref(), config, &self.sigma, &targets, &mut rows, n as u32, late.len() as u32)
        }?;
        let mut host_rows: Vec<u32> = rows
            .to_host_vec(&self.stream)?
            .iter()
            .map(|&v| if v as u32 == NONE { permanent_core::sampler::NONE } else { v as u32 })
            .collect();
        let scaled = self.scaled;
        let first_id = self.first_id;
        let finished: Vec<(u32, far::FarStats)> = host_rows
            .par_chunks_mut(n)
            .zip(&late)
            .map(|(row, &particle)| {
                let (hole_u, hole_v) = hole_of(status[particle as usize]);
                let hole = (hole_u != NONE as u64).then_some((hole_u as usize, hole_v as usize));
                let seed = stream_seed(stream.seed, stream.phase, stream.level, first_id + particle as u64);
                // a stream of its own, apart from the kernel's Philox one
                let mut rng = SmallRng::seed_from_u64(mix(!seed));
                let left = cycles - done[particle as usize];
                (particle, far::resume(tables, &scaled.energy, row, hole, left, max_steps, &mut rng))
            })
            .collect();
        for (particle, stats) in finished {
            let group = particle as usize % groups;
            round.cycles[group] += stats.perfect_steps;
            for v in 0..n {
                round.visits[group][v] += stats.visits[v];
                round.visit_energy[v] += stats.visit_energy[v];
            }
            round.perfect_energy += stats.perfect_energy;
            round.steps += stats.steps;
            round.capped += stats.capped as u64;
        }
        let words: Vec<u16> = host_rows.iter().map(|&v| v as u16).collect();
        let rows = DeviceBuffer::from_host(&self.stream, &words)?;
        // SAFETY: as for the gather above; the targets are distinct.
        unsafe {
            self.module.scatter(self.stream.as_ref(), config, &rows, &targets, &mut self.sigma, n as u32, late.len() as u32)
        }?;
        self.far_seconds.host_chains += clock.elapsed().as_secs_f64();
        Ok(round)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use permanent_core::problem::Problem;
    use permanent_core::scaling::{Prepared, prepare};

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

    /// From exact starts, the cycle counts recover the ideal anchor weights
    /// lambda(P) / lambda(N(r, v)), computed here by enumeration.
    #[test]
    fn far_cycle_counts_recover_the_ideal_weights() {
        let scaled = instance();
        let n = scaled.n;
        let (beta, anchor) = (0.8, 1);
        let log_w = [0.5, 0.0, -0.7, 0.2];
        let tables = Tables::build(&scaled, beta, &log_w, anchor);
        let lambda = |u: usize, v: usize| (-beta * scaled.energy[u * n + v]).exp();
        let perms = permutations(n);
        let weights: Vec<f64> = perms.iter().map(|p| (-beta * scaled.energy_of(p)).exp()).collect();
        let z_perfect: f64 = weights.iter().sum();
        let mut z_hole = vec![0.0; n];
        for (p, w) in perms.iter().zip(&weights) {
            let v = p[anchor] as usize;
            z_hole[v] += w / lambda(anchor, v);
        }
        let stages = StageSchedule::new(n, 32.0);
        let mut cuda = Cuda::new(&scaled, 0).unwrap();
        cuda.anneal(&tables, &stages, stages.rigorous_stage_time(1e-3), 64, 8192, 0, Stream { seed: 9, phase: 1, level: 0 })
            .unwrap();
        let groups = 16;
        let round = cuda.far(&tables, 200, u64::MAX >> 1, groups, Stream { seed: 9, phase: 2, level: 0 }).unwrap();
        // a budget this short hands most chains to the host part-way
        cuda.far_budget = 0.5;
        let short = cuda.far(&tables, 200, u64::MAX >> 1, groups, Stream { seed: 9, phase: 3, level: 0 }).unwrap();
        assert_eq!(short.cycles.iter().sum::<u64>(), 8192 * 200);
        for v in 0..n {
            let (a, b): (u64, u64) = (round.visits.iter().map(|g| g[v]).sum(), short.visits.iter().map(|g| g[v]).sum());
            assert!((a as f64 / b as f64 - 1.0).abs() < 0.03, "v={v}: {a} visits in full launches, {b} with hand-off");
        }
        assert_eq!(round.capped, 0);
        let cycles: u64 = round.cycles.iter().sum();
        assert_eq!(cycles, 8192 * 200);
        let anchor_norm = tables.log_anchor_norm.exp();
        let mut visit_energy = 0.0;
        for v in 0..n {
            let visits: u64 = round.visits.iter().map(|group| group[v]).sum();
            let ideal = z_perfect / z_hole[v];
            let learned = log_w[v].exp() * anchor_norm * tables.hole_weight(anchor, v) * cycles as f64
                / ((n * n) as f64 * visits as f64);
            assert!((learned / ideal - 1.0).abs() < 0.03, "v={v}: learned {learned:.5}, ideal {ideal:.5}");
            visit_energy += round.visit_energy[v];
        }
        // the population is still on the target, and the energy sums are sane
        check(&scaled, beta, &cuda.energies().unwrap());
        let mean: f64 = perms.iter().zip(&weights).map(|(p, w)| scaled.energy_of(p) * w / z_perfect).sum();
        assert!((round.perfect_energy / cycles as f64 / mean - 1.0).abs() < 0.02);
        assert!(visit_energy > 0.0);
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
