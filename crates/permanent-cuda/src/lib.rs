//! The population resident on an NVIDIA GPU.
//!
//! The kernels are Rust, compiled for the device by cuda-oxide. One GPU
//! thread works on one particle at a time. Between launches only the
//! particles' matchings (n 16-bit words each) stay in global memory; a
//! launch stages each particle's working state in the block's shared
//! memory, so the event loop touches global memory for the alias table
//! alone.
//!
//! How many particles a multiprocessor runs at once is set by how much
//! shared memory each needs, so the working state of the dynamics is kept
//! to one array over the columns and a short list of the rows in a
//! collision (`permanent_core::chain::matching`). The penalty J keeps
//! collisions rare at every activity, which is what makes a short list
//! enough; a particle that overflows it is treated like one that hit its
//! event cap. Once a warp's worth of state no longer fits the shared memory
//! every device gives a block (n above about 670 for the dynamics, 380 for
//! FAR, which keeps a matching and its inverse), the mutation kernel opts
//! in to what this device allows and takes the block size that keeps the
//! most particles on a multiprocessor; the other kernels, and a mutation
//! whose state fits no block, keep the state in global memory instead.
//!
//! The chains are those of `permanent_core::chain`, the same code the rayon
//! engine runs, compiled here for the device: every probability decision
//! is an integer comparison, the generator is Philox4x32-10 keyed by the
//! particle's stream seed, FAR energies are fixed point, and only waiting
//! times are in the real type `T` of `Cuda<T>`. With the same seeds the
//! FAR chain returns the same counts on both engines.
//!
//! A launch lasts as long as its longest thread. FAR cycles have a long
//! tail, so a FAR launch gives every chain a step budget, and the few
//! chains that overrun it are continued on the host from where they
//! stopped, on the same stream; every cycle still completes, which is what
//! the learner's identity needs.

use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result, ensure};
use cuda_core::{CudaContext, DeviceBuffer, DeviceCopy, LaunchConfig, Stream as CudaStream};
use cuda_device::atomic::{AtomicOrdering, DeviceAtomicI64, DeviceAtomicU64};
use cuda_device::{DisjointSlice, DynamicSharedArray, device, kernel, thread};
use cuda_host::cuda_module;
use num_traits::{AsPrimitive, Float};
use permanent_core::chain::dynamics::{self, STAGE_TICKS};
use permanent_core::chain::far::{self as far_chain, FarChain, FarTables, Tally};
use permanent_core::chain::lane::{Lane, MAX_N, NONE};
use permanent_core::chain::matching::{Matching, words as lane_words};
use permanent_core::chain::stream::Stream as Stream32;
use permanent_core::chain::{Energy, EnergySum};
use permanent_core::engine::{Engine, FarRound, Stream, VisitList};
use permanent_core::sampler::{MutateCost, SampleCost, StageSchedule, mutate_event_cap, penalty};
use permanent_core::scaling::Scaled;
use permanent_core::stats::{mix, stream_seed};
use permanent_core::tables::{Tables, fraction};
use rayon::prelude::*;
use tracing::{debug, info};

/// Set in a particle's event count when its move hit the event cap.
const OVER_BUDGET: u64 = 1 << 63;
/// Set in a trial's event count when it ended on a perfect matching.
const PERFECT: u64 = 1 << 62;

/// Words of a particle's slot of `scratch`, which takes the working state of
/// any kernel: a relaxed assignment, or the two arrays of a FAR chain.
fn slot_words(n: usize) -> usize {
    lane_words(n).max(2 * n)
}

/// This thread's lane: one of its block's shared memory when `staged`, where
/// the stride puts the lanes of a warp in distinct banks, else its slot of
/// `scratch`.
#[device]
unsafe fn lane(scratch: *mut u16, slot: usize, n: usize, staged: u32) -> Lane {
    unsafe {
        if staged != 0 {
            let shared: *mut u16 = DynamicSharedArray::<u16>::get();
            Lane::new(shared.add(thread::threadIdx_x() as usize), thread::blockDim_x() as usize, n)
        } else {
            Lane::new(scratch.add(slot * slot_words(n)), 1, n)
        }
    }
}

/// Counts a chain's visits into the launch's shared sums.
struct DeviceTally<'k> {
    /// The row of this chain's group.
    visits: &'k [DeviceAtomicU64],
    first: usize,
    visit_energy: &'k [DeviceAtomicI64],
}

impl Tally for DeviceTally<'_> {
    #[inline(always)]
    fn visit(&mut self, v: u32, energy: Energy) {
        // SAFETY: the kernel is launched with groups rows of n counters and n sums
        unsafe {
            self.visits.get_unchecked(self.first + v as usize).fetch_add(1, AtomicOrdering::Relaxed);
            self.visit_energy.get_unchecked(v as usize).fetch_add(energy.to_bits(), AtomicOrdering::Relaxed);
        }
    }
}

#[cuda_module]
mod kernels {
    use super::*;

    /// The mutation kernel K_t (`permanent_core::chain::dynamics::mutate`)
    /// on every particle: relaxed-assignment dynamics at activity J until
    /// `perfect_time` has been spent in perfect matchings.
    ///
    /// A thread is not bound to one particle: it takes the next unmoved one
    /// from `cursor` until none is left, so a launch may have fewer threads
    /// than particles. A particle's move depends on its seed alone, not on
    /// the thread that makes it.
    ///
    /// alias: the packed row alias table. unassign_below: P(unassign) 2^64.
    /// fast: 1 + J. cursor: one counter, zero at launch.
    #[kernel(unchecked_indexing)]
    #[allow(clippy::too_many_arguments)]
    pub fn mutate<T: Float + AsPrimitive<u64>>(
        alias: &[u64],
        seeds: &[u64],
        cursor: &[DeviceAtomicU64],
        mut sigma: DisjointSlice<u16>,
        mut scratch: DisjointSlice<u16>,
        mut events_out: DisjointSlice<u64>,
        mut time_out: DisjointSlice<T>,
        n: u32,
        count: u32,
        staged: u32,
        perfect_time: T,
        fast: u64,
        unassign_below: u64,
        event_cap: u64,
    ) {
        let thread = thread::index_1d().get();
        if thread >= count as usize {
            return;
        }
        let size = n as usize;
        // SAFETY: the lane is this thread's own.
        let mut matching = unsafe { Matching::new(lane(scratch.as_mut_ptr(), thread, size, staged)) };
        loop {
            let slot = cursor[0].fetch_add(1, AtomicOrdering::Relaxed) as usize;
            if slot >= count as usize {
                break;
            }
            let mut rng = Stream32::new(seeds[slot]);
            // SAFETY: `sigma` holds `count` perfect matchings on n rows, the
            // lane a relaxed assignment, the table is n x n, and the cursor
            // gives each slot to one thread.
            unsafe {
                let row = sigma.as_mut_ptr().add(slot * size);
                matching.load(row);
                let moved = dynamics::mutate::<u64, T, _>(
                    &mut matching,
                    n,
                    alias,
                    perfect_time,
                    fast,
                    unassign_below,
                    event_cap,
                    &mut rng,
                );
                if !moved.over_budget {
                    matching.store(row);
                }
                *events_out.as_mut_ptr().add(slot) =
                    if moved.over_budget { moved.events | OVER_BUDGET } else { moved.events };
                *time_out.as_mut_ptr().add(slot) = moved.time;
            }
        }
    }

    /// One trial of the annealing sampler
    /// (`permanent_core::chain::dynamics::trial`) per slot: from the
    /// all-unmatched state through every activity stage. A trial that ends
    /// on a perfect matching writes it to particle `targets[slot]` and sets
    /// PERFECT in its event count.
    ///
    /// unassign[stage]: P(unassign) 2^64 at that stage's activity.
    /// ticks_per_time: STAGE_TICKS / stage time.
    #[kernel(unchecked_indexing)]
    #[allow(clippy::too_many_arguments)]
    pub fn anneal<T: Float + AsPrimitive<u64>>(
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
        ticks_per_time: T,
        fast: u64,
        event_cap: u64,
    ) {
        let index = thread::index_1d();
        let slot = index.get();
        if slot >= count as usize {
            return;
        }
        let size = n as usize;
        let mut rng = Stream32::new(seeds[slot]);
        // SAFETY: the lane holds a relaxed assignment, the table is n x n,
        // and the targets are distinct particles of `sigma`.
        let (trial, perfect) = unsafe {
            let mut matching = Matching::new(lane(scratch.as_mut_ptr(), slot, size, staged));
            let trial = dynamics::trial::<u64, T, _>(
                &mut matching,
                n,
                alias,
                unassign,
                ticks_per_time,
                fast,
                event_cap,
                &mut rng,
            );
            let perfect = !trial.over_budget && matching.is_perfect();
            if perfect {
                matching.store(sigma.as_mut_ptr().add(targets[slot] as usize * size));
            }
            (trial, perfect)
        };
        if let Some(out) = events_out.get_mut(index) {
            *out = trial.events
                | if perfect {
                    PERFECT
                } else if trial.over_budget {
                    OVER_BUDGET
                } else {
                    0
                };
        }
    }

    /// The FAR chain (`permanent_core::chain::far`) on every particle:
    /// `cycles` cycles from its perfect matching in `sigma`, or `budget`
    /// steps if that comes first. Where the chain stopped goes to `next`: a
    /// perfect matching, or one with NONE at its row hole. `status_out`
    /// holds steps | row hole << 32 | column hole << 48, with NONE for no
    /// holes, and `position_out` the stream's blocks << 3 | outputs left.
    /// Particle p adds its visits to N(anchor, v) to
    /// `visits[(p % groups) n + v]` and their energies to `visit_energy[v]`.
    ///
    /// As in `mutate`, a thread takes particles from `cursor` (one counter,
    /// zero at launch) until none is left, and a chain depends on its
    /// particle's seed alone.
    #[kernel(unchecked_indexing)]
    #[allow(clippy::too_many_arguments)]
    pub fn far(
        row_alias: &[u64],
        col_alias: &[u64],
        moves: &[u64],
        energy: &[i64],
        seeds: &[u64],
        cursor: &[DeviceAtomicU64],
        visits: &[DeviceAtomicU64],
        visit_energy: &[DeviceAtomicI64],
        sigma: &[u16],
        mut next: DisjointSlice<u16>,
        mut scratch: DisjointSlice<u16>,
        mut cycles_out: DisjointSlice<u64>,
        mut status_out: DisjointSlice<u64>,
        mut energy_out: DisjointSlice<i64>,
        mut position_out: DisjointSlice<u64>,
        n: u32,
        count: u32,
        staged: u32,
        anchor: u32,
        groups: u32,
        cycles: u64,
        budget: u64,
    ) {
        let thread = thread::index_1d().get();
        if thread >= count as usize {
            return;
        }
        let size = n as usize;
        let tables = FarTables { row_alias, col_alias, moves, energy };
        // SAFETY: the lane is this thread's own.
        let lane = unsafe { lane(scratch.as_mut_ptr(), thread, size, staged) };
        loop {
            let slot = cursor[0].fetch_add(1, AtomicOrdering::Relaxed) as usize;
            if slot >= count as usize {
                break;
            }
            let mut rng = Stream32::new(seeds[slot]);
            let mut tally = DeviceTally { visits, first: (slot as u32 % groups) as usize * size, visit_energy };
            // SAFETY: `sigma` and `next` hold `count` rows of n, the lane
            // two arrays of n, the tables are n x n, every per-particle
            // buffer holds `count` entries, and the cursor gives each slot
            // to one thread.
            unsafe {
                let mut chain = far_chain::begin(&lane, sigma.as_ptr().add(slot * size), &tables);
                far_chain::run(&lane, &tables, anchor, cycles, budget, &mut rng, &mut chain, &mut tally);
                far_chain::store(&lane, next.as_mut_ptr().add(slot * size));
                *cycles_out.as_mut_ptr().add(slot) = chain.perfect_steps;
                *status_out.as_mut_ptr().add(slot) =
                    chain.steps | (chain.hole_u as u64) << 32 | (chain.hole_v as u64) << 48;
                *energy_out.as_mut_ptr().add(slot) = chain.perfect_energy.to_bits();
                let (blocks, left) = rng.position();
                *position_out.as_mut_ptr().add(slot) = blocks << 3 | left as u64;
            }
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

/// The share of its warps' time a launch left idle, taking a particle's
/// time as its event count: a warp runs as long as its longest particle,
/// and the lanes of the others wait for it. `events` is per particle in
/// launch order, with `flags` set aside.
fn warp_idle(events: &[u64], flags: u64) -> f64 {
    let mut used = 0u64;
    let mut held = 0u64;
    for warp in events.chunks(BLOCK as usize) {
        let longest = warp.iter().map(|&e| e & !flags).max().unwrap_or(0);
        used += warp.iter().map(|&e| e & !flags).sum::<u64>();
        held += longest * BLOCK as u64;
    }
    if held == 0 { 0.0 } else { 1.0 - used as f64 / held as f64 }
}

/// Most threads of a mutation launch whose working state is in global
/// memory, which is where it is when one particle's state exceeds the shared
/// memory a block may opt in to. With more, the state of the threads running
/// at once outgrows the device's cache: at n = 512 a GH200 moves 2^18
/// particles 1.7 times slower with a thread each than 2^16 particles.
const UNSTAGED_THREADS: usize = 1 << 16;
/// Most particles resident at once, and the device memory they may take.
const MAX_PARTICLES: usize = 1 << 20;
const MEMORY_BUDGET: usize = 8 << 30;
/// Populations swept at once even when one of them fills the device. A
/// launch ends with its longest particle, which runs several times as long
/// as the mean one, so for the later part of every launch most of the
/// threads a population occupied are free; that is a property of the
/// chains, not of the device.
const PAIRED: usize = 2;
/// Particles a sampler launch that retries may expect to leave missing.
/// It weighs one more launch against trials that are not needed, both in
/// units of one trial's duration, so it does not depend on the device.
const EXPECTED_MISSING: f64 = 0.5;

/// What a device reports about its multiprocessors.
#[derive(Debug, Clone, Copy)]
struct Multiprocessors {
    count: usize,
    /// Most threads one of them runs at once.
    threads: usize,
    /// Shared memory of one of them, taken as the most a block may opt in
    /// to: all of it but a reserved kilobyte on the devices so far.
    shared: usize,
}

impl Multiprocessors {
    fn of(context: &CudaContext) -> Result<Self> {
        Ok(Multiprocessors {
            count: context.multiprocessor_count()? as usize,
            threads: context.max_threads_per_multiprocessor()? as usize,
            shared: context.max_opt_in_shared_memory_per_block()? as usize,
        })
    }
}

/// Blocks that stage their working state in shared memory.
#[derive(Debug, Clone, Copy)]
struct Staged {
    /// Threads per block.
    block: u32,
    /// Shared memory of a block, in bytes.
    shared: u32,
    /// Threads of such blocks a multiprocessor runs at once.
    resident: u32,
}

/// Number of CUDA devices on this host.
pub fn device_count() -> Result<usize> {
    Ok(CudaContext::device_count()? as usize)
}

/// The population on one GPU. `T` is the type of the chains' waiting times.
pub struct Cuda<'a, T: DeviceCopy> {
    scaled: &'a Scaled,
    name: String,
    context: Arc<CudaContext>,
    multiprocessors: Multiprocessors,
    /// A stream of this engine's own, so that engines of one context run
    /// their launches at the same time.
    stream: Arc<CudaStream>,
    /// The kernels, loaded once for all the engines of a context.
    module: Arc<kernels::LoadedModule>,
    /// How the mutation kernel stages its working state; `None` when the
    /// state of one particle exceeds the shared memory of a block.
    mutation: Option<Staged>,
    first_id: u64,
    count: usize,
    energy: DeviceBuffer<f64>,
    log_gap: DeviceBuffer<f64>,
    /// The energies in the chains' fixed point, on the host and the device.
    energy_bits: Vec<i64>,
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
    time: DeviceBuffer<T>,
    energy_out: DeviceBuffer<f64>,
    gap_out: DeviceBuffer<f64>,
    cycles_out: DeviceBuffer<u64>,
    fixed_out: DeviceBuffer<i64>,
    position_out: DeviceBuffer<u64>,
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

impl<T: DeviceCopy> Drop for Cuda<'_, T> {
    fn drop(&mut self) {
        if self.far_seconds.calls > 0 {
            debug!(far = ?self.far_seconds, budget_per_cycle = self.far_budget, "CUDA engine FAR calls");
        }
    }
}

impl<'a, T> Cuda<'a, T>
where
    T: Float + AsPrimitive<u64> + DeviceCopy + Send + Sync,
{
    /// The engine on device `ordinal`.
    pub fn new(scaled: &'a Scaled, ordinal: usize) -> Result<Self> {
        let n = scaled.n;
        ensure!(n <= MAX_N, "the CUDA engine supports n up to {MAX_N}");
        let context = CudaContext::new(ordinal).with_context(|| format!("opening CUDA device {ordinal}"))?;
        let device = context.device_name()?;
        let multiprocessors = Multiprocessors::of(&context)?;
        let name = format!("cuda:{ordinal} ({device}, {})", std::any::type_name::<T>());
        let module = kernels::load(&context).context("loading the kernels; device code is built by `cargo oxide`")?;
        let mutation = Self::stage_mutation(&module, multiprocessors.shared, lane_words(n))?;
        let engine = Self::on(scaled, context, multiprocessors, Arc::new(module), mutation, name)?;
        info!(
            %device,
            staged = fits_shared(lane_words(n)),
            mutation = ?engine.mutation,
            far_staged = fits_shared(2 * n),
            threads_at_once = engine.threads_at_once(lane_words(n)),
            "CUDA engine ready"
        );
        Ok(engine)
    }

    /// An engine with no particles in `context`.
    fn on(
        scaled: &'a Scaled,
        context: Arc<CudaContext>,
        multiprocessors: Multiprocessors,
        module: Arc<kernels::LoadedModule>,
        mutation: Option<Staged>,
        name: String,
    ) -> Result<Self> {
        let energy_bits = scaled.fixed_energy();
        let stream = context.new_stream()?;
        Ok(Cuda {
            scaled,
            name,
            context,
            multiprocessors,
            module,
            mutation,
            first_id: 0,
            count: 0,
            energy: DeviceBuffer::from_host(&stream, &scaled.energy)?,
            log_gap: DeviceBuffer::from_host(&stream, &scaled.log_gap)?,
            energy_fixed: DeviceBuffer::from_host(&stream, &energy_bits)?,
            energy_bits,
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
            position_out: DeviceBuffer::zeroed(&stream, 1)?,
            far_budget: 64.0,
            far_seconds: FarSeconds::default(),
            stream,
        })
    }

    /// The blocks of the mutation kernel for `words` words of working state
    /// per particle, where a block may opt in to `limit` bytes of shared
    /// memory: of the block sizes whose state fits, the one that keeps the
    /// most threads on a multiprocessor, as the driver computes it for this
    /// kernel. Shared memory is what bounds that number once the state is
    /// large, and a thread takes any particle, so nothing is lost by a
    /// block of less than a warp. Of two sizes that keep as many threads
    /// the larger wins: it puts them in fewer warps, and a multiprocessor
    /// issues instructions by the warp.
    fn stage_mutation(module: &kernels::LoadedModule, limit: usize, words: usize) -> Result<Option<Staged>> {
        let function = module.as_cuda_module().load_function(kernels::mutate_ptx_name::<T>())?;
        let bytes = |block: u32| block as usize * words * size_of::<u16>();
        let sizes = || (0..=BLOCK.ilog2()).rev().map(|log| 1 << log).filter(|&block| bytes(block) <= limit);
        // opt in before asking the driver about blocks that need it
        match sizes().next() {
            Some(block) if bytes(block) > SHARED_LIMIT => {
                function.set_max_dynamic_shared_memory_bytes(bytes(block) as u32)?
            }
            _ => {}
        }
        let mut best: Option<Staged> = None;
        for block in sizes() {
            let shared = bytes(block) as u32;
            let resident = block * function.max_active_blocks_per_multiprocessor(block, shared)?;
            if best.is_none_or(|staged| resident > staged.resident) {
                best = Some(Staged { block, shared, resident });
            }
        }
        Ok(best)
    }

    /// `data` on the device. The memory is allocated and freed in the order
    /// of this engine's stream; a synchronous allocation would wait for the
    /// launches of the other engines of the device.
    fn device<D: DeviceCopy>(&self, data: &[D]) -> Result<DeviceBuffer<D>> {
        // SAFETY: the copy fills the buffer before anything reads it
        let mut buffer = unsafe { DeviceBuffer::uninitialized_async(&self.stream, data.len()) }?;
        buffer.copy_from_host(&self.stream, data)?;
        Ok(buffer)
    }

    /// `len` zeros on the device, allocated like `device`.
    fn zeros<D: DeviceCopy>(&self, len: usize) -> Result<DeviceBuffer<D>> {
        // SAFETY: the buffer is zeroed on the stream that later reads it
        let mut buffer = unsafe { DeviceBuffer::uninitialized_async(&self.stream, len) }?;
        buffer.zero_async(&self.stream)?;
        Ok(buffer)
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

    /// Particle threads of `words` words of working state each that the
    /// device runs at once, over all the launches in flight: every
    /// multiprocessor takes blocks until its threads or its shared memory
    /// are used up, and further blocks wait for one of those to finish.
    fn threads_at_once(&self, words: usize) -> usize {
        let (config, staged) = self.config(1, words);
        let block = config.block_dim.0 as usize;
        let mut blocks = self.multiprocessors.threads / block;
        if staged != 0 {
            blocks = blocks.min(self.multiprocessors.shared / config.shared_mem_bytes as usize);
        }
        self.multiprocessors.count * blocks.max(1) * block
    }

    /// Whether some kernel keeps its working state in `scratch`.
    fn needs_scratch(&self) -> bool {
        !fits_shared(slot_words(self.scaled.n))
    }

    /// Put the row alias table of `tables` on the device, unless it is there.
    fn sync_row(&mut self, tables: &Tables<u64>) -> Result<()> {
        if self.row_id != tables.id {
            self.row_alias = self.device(&tables.packed_rows)?;
            self.row_id = tables.id;
        }
        Ok(())
    }

    /// Likewise the tables only the FAR kernel reads.
    fn sync_far(&mut self, tables: &Tables<u64>) -> Result<()> {
        if self.far_id != tables.id {
            self.col_alias = self.device(&tables.packed_cols)?;
            self.moves = self.device(&tables.far_moves)?;
            self.far_id = tables.id;
        }
        Ok(())
    }

    /// Size the per-particle buffers for `count` particles.
    fn resize(&mut self, count: usize) -> Result<()> {
        if count == self.count {
            return Ok(());
        }
        let n = self.scaled.n;
        self.count = count;
        self.sigma = self.zeros(count * n)?;
        self.next = self.zeros(count * n)?;
        let scratch = if self.needs_scratch() { count * slot_words(n) } else { 1 };
        self.scratch = self.zeros(scratch)?;
        self.events = self.zeros(count)?;
        self.time = self.zeros(count)?;
        self.energy_out = self.zeros(count)?;
        self.gap_out = self.zeros(count)?;
        self.cycles_out = self.zeros(count)?;
        self.fixed_out = self.zeros(count)?;
        self.position_out = self.zeros(count)?;
        Ok(())
    }

    fn upload(&mut self, perms: &[u32]) -> Result<()> {
        self.resize(perms.len() / self.scaled.n)?;
        let words: Vec<u16> = perms.iter().map(|&v| v as u16).collect();
        self.sigma = self.device(&words)?;
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

impl<T> Engine for Cuda<'_, T>
where
    T: Float + AsPrimitive<u64> + DeviceCopy + Send + Sync,
{
    type Data = T;
    type Counter = u64;

    fn name(&self) -> String {
        self.name.clone()
    }

    fn capacity(&self) -> usize {
        let n = self.scaled.n;
        let scratch = if self.needs_scratch() { slot_words(n) } else { 0 };
        let per_particle = (2 * n + scratch) * size_of::<u16>() + 48;
        MAX_PARTICLES.min(MEMORY_BUDGET / per_particle)
    }

    fn learn_particles(&self) -> usize {
        // enough chains that a learning round is about 16 cycles for each
        let n = self.scaled.n;
        (4 * n * n).clamp(4096, 1 << 18).min(self.capacity())
    }

    fn concurrency(&self, particles: usize) -> usize {
        // As many populations as fill the device with mutation threads.
        // The last may fit only in part: some of its blocks then wait,
        // which still costs less than sweeping it after the others.
        let resident = match self.mutation {
            Some(staged) => self.multiprocessors.count * staged.resident as usize,
            None => self.threads_at_once(lane_words(self.scaled.n)),
        };
        let filling = resident.div_ceil(particles.max(1));
        // A population that already fills the device still gains from a
        // second one beside it, as long as the engine may hold both.
        let paired = if PAIRED * particles <= self.capacity() { PAIRED } else { 1 };
        filling.max(paired)
    }

    fn fork(&self) -> Result<Self> {
        Self::on(
            self.scaled,
            self.context.clone(),
            self.multiprocessors,
            self.module.clone(),
            self.mutation,
            self.name.clone(),
        )
    }

    fn load(&mut self, perms: &[u32], first_id: u64) -> Result<()> {
        self.first_id = first_id;
        self.upload(perms)
    }

    fn anneal(
        &mut self,
        tables: &Tables<u64>,
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
        let unassign = self.device(&stages.unassign::<u64>())?;
        let ticks_per_time = T::from(STAGE_TICKS as f64 / stage_time).unwrap();
        let event_cap = stages.event_cap(n, stage_time);
        let mut cost = SampleCost::default();
        // A trial that misses is run again on a stream of its own. A launch
        // lasts as long as one trial however few it holds, so once the miss
        // rate is known a launch runs the next few trials of every particle
        // still missing, each into a row of its own: enough to expect
        // EXPECTED_MISSING left. A particle takes its first hit, the
        // sample it would get from running its trials one at a time.
        let mut pending: Vec<u32> = (0..count as u32).collect();
        let mut tried = 0u64;
        while !pending.is_empty() && tried < max_trials {
            let clock = Instant::now();
            let wanted = if cost.trials == 0 {
                1
            } else {
                let miss = 1.0 - cost.accepted as f64 / cost.trials as f64;
                ((pending.len() as f64 / EXPECTED_MISSING).ln() / -miss.ln()).ceil() as u64
            };
            let most = ((self.threads_at_once(lane_words(n)) / pending.len()) as u64).clamp(1, max_trials - tried);
            let attempts = wanted.clamp(1, most);
            let slots = pending.len() * attempts as usize;
            let seeds: Vec<u64> = pending
                .iter()
                .flat_map(|&p| {
                    let sample = stream_seed(stream.seed, stream.phase, stream.level, first_id + p as u64);
                    (tried..tried + attempts).map(move |trial| mix(sample.wrapping_add(trial)))
                })
                .collect();
            let seeds = self.device(&seeds)?;
            let own_row: Vec<u32> = (0..slots as u32).collect();
            let own_row = self.device(&own_row)?;
            let mut rows = self.zeros::<u16>(slots * n)?;
            let scratch = if self.needs_scratch() { slots * slot_words(n) } else { 1 };
            let mut scratch = self.zeros::<u16>(scratch)?;
            let mut events = self.zeros::<u64>(slots)?;
            let (config, staged) = self.config(slots, lane_words(n));
            // SAFETY: the arguments match the kernel; every slot writes its
            // own row of `rows`, and every per-slot buffer holds `slots`
            // entries.
            unsafe {
                self.module.anneal::<T>(
                    self.stream.as_ref(),
                    config,
                    &self.row_alias,
                    &unassign,
                    &seeds,
                    &own_row,
                    &mut rows,
                    &mut scratch,
                    &mut events,
                    n as u32,
                    slots as u32,
                    staged,
                    ticks_per_time,
                    stages.fast(),
                    event_cap,
                )
            }?;
            let setup = clock.elapsed().as_secs_f64();
            self.stream.synchronize()?;
            let launched = clock.elapsed().as_secs_f64();
            let outcomes = events.to_host_vec(&self.stream)?;
            debug!(
                particles = pending.len(),
                trials = slots,
                setup,
                kernel = launched - setup,
                readback = clock.elapsed().as_secs_f64() - launched,
                warp_idle = warp_idle(&outcomes, OVER_BUDGET | PERFECT),
                "CUDA sampler launch"
            );
            // trials after a particle's first hit never happened
            let mut missing = Vec::new();
            let mut hit_rows: Vec<u32> = Vec::new();
            let mut hit_particles: Vec<u32> = Vec::new();
            for ((index, &particle), outcomes) in pending.iter().enumerate().zip(outcomes.chunks(attempts as usize)) {
                let mut hit = false;
                for (attempt, outcome) in outcomes.iter().enumerate() {
                    cost.trials += 1;
                    cost.events += outcome & !(OVER_BUDGET | PERFECT);
                    cost.over_budget += (outcome & OVER_BUDGET != 0) as u64;
                    if outcome & PERFECT != 0 {
                        cost.accepted += 1;
                        hit_rows.push((index * attempts as usize + attempt) as u32);
                        hit_particles.push(particle);
                        hit = true;
                        break;
                    }
                }
                if !hit {
                    missing.push(particle);
                }
            }
            if !hit_rows.is_empty() {
                let hits = hit_rows.len();
                let hit_rows = self.device(&hit_rows)?;
                let hit_particles = self.device(&hit_particles)?;
                let mut samples = self.zeros::<u16>(hits * n)?;
                let (config, _) = self.config(hits, 0);
                // SAFETY: the arguments match the kernels; `samples` holds
                // one row per hit, the hit rows are below `slots`, and the
                // particles are distinct and below `count`.
                unsafe {
                    self.module.gather(
                        self.stream.as_ref(),
                        config,
                        &rows,
                        &hit_rows,
                        &mut samples,
                        n as u32,
                        hits as u32,
                    )?;
                    self.module.scatter(
                        self.stream.as_ref(),
                        config,
                        &samples,
                        &hit_particles,
                        &mut self.sigma,
                        n as u32,
                        hits as u32,
                    )?;
                }
            }
            pending = missing;
            tried += attempts;
        }
        ensure!(
            pending.is_empty(),
            "the sampler could not produce {} of {count} perfect matchings; the anchor weights are out of range",
            pending.len()
        );
        Ok(cost)
    }

    fn mutate(&mut self, tables: &Tables<u64>, perfect_time: f64, stream: Stream) -> Result<MutateCost> {
        let n = self.scaled.n;
        let clock = Instant::now();
        self.sync_row(tables)?;
        let seeds: Vec<u64> = (0..self.count as u64)
            .map(|p| stream_seed(stream.seed, stream.phase, stream.level, self.first_id + p))
            .collect();
        let seeds = self.device(&seeds)?;
        let cursor = self.zeros::<u64>(1)?.wrap_elem::<DeviceAtomicU64>();
        let penalty = penalty(n);
        let unassign_below: u64 = fraction(1.0 / (1.0 + penalty));
        let (config, staged) = match self.mutation {
            Some(Staged { block, shared, .. }) => {
                let grid = (self.count as u32).div_ceil(block);
                (LaunchConfig { grid_dim: (grid, 1, 1), block_dim: (block, 1, 1), shared_mem_bytes: shared }, 1)
            }
            None => self.config(self.count.min(UNSTAGED_THREADS), lane_words(n)),
        };
        // SAFETY: the arguments match the kernel; `sigma` holds count rows
        // of n, `scratch` the working state when it is not staged, and the
        // dynamic shared memory the block's working state when it is.
        unsafe {
            self.module.mutate::<T>(
                self.stream.as_ref(),
                config,
                &self.row_alias,
                &seeds,
                &cursor,
                &mut self.sigma,
                &mut self.scratch,
                &mut self.events,
                &mut self.time,
                n as u32,
                self.count as u32,
                staged,
                T::from(perfect_time).unwrap(),
                1 + penalty as u64,
                unassign_below,
                mutate_event_cap(n, perfect_time),
            )
        }?;
        let setup = clock.elapsed().as_secs_f64();
        self.stream.synchronize()?;
        let launched = clock.elapsed().as_secs_f64();
        let events = self.events.to_host_vec(&self.stream)?;
        let time = self.time.to_host_vec(&self.stream)?;
        let mut cost = MutateCost::default();
        for (events, time) in events.iter().zip(time) {
            cost.events += events & !OVER_BUDGET;
            cost.over_budget += (events & OVER_BUDGET != 0) as u64;
            cost.time += time.to_f64().unwrap();
        }
        debug!(
            particles = self.count,
            events = cost.events,
            longest = events.iter().map(|&e| e & !OVER_BUDGET).max().unwrap_or(0),
            setup,
            kernel = launched - setup,
            readback = clock.elapsed().as_secs_f64() - launched,
            "CUDA mutation call"
        );
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
        let ancestors = self.device(ancestors)?;
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
        tables: &Tables<u64>,
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
        let device_seeds = self.device(&seeds)?;
        let cursor = self.zeros::<u64>(1)?.wrap_elem::<DeviceAtomicU64>();
        let visits = self.zeros::<u64>(groups * n)?.wrap_elem::<DeviceAtomicU64>();
        let visit_energy = self.zeros::<i64>(n)?.wrap_elem::<DeviceAtomicI64>();
        let budget = ((self.far_budget * cycles as f64).ceil() as u64 + 32).min(max_steps);
        // The FAR chain keeps a matching and its inverse, and no list. A
        // launch lasts as long as a thread's share of the chains, each a
        // few hundred steps, so unlike a mutation it has a thread for every
        // particle wherever the state lives.
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
                &device_seeds,
                &cursor,
                &visits,
                &visit_energy,
                &self.sigma,
                &mut self.next,
                &mut self.scratch,
                &mut self.cycles_out,
                &mut self.events,
                &mut self.fixed_out,
                &mut self.position_out,
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
        let visit_energy = visit_energy.peel_elem::<i64>().to_host_vec(&self.stream)?;
        // the device adds the energies of a class into 64 bits
        let h_max = self.scaled.h_max();
        for v in 0..n {
            let class: u64 = visits.iter().skip(v).step_by(n).sum();
            ensure!(
                class as f64 * h_max < Energy::MAX.to_num::<f64>(),
                "the energy sum of hole class {v} can overflow; use fewer cycles per call"
            );
        }
        let mut round = FarRound {
            cycles: vec![0; groups],
            visits: visits.chunks(n).map(|row| row.to_vec()).collect(),
            visit_energy: visit_energy.iter().map(|&bits| EnergySum::from(Energy::from_bits(bits))).collect(),
            perfect_energy: EnergySum::ZERO,
            steps: 0,
            capped: 0,
        };
        let steps_of = |status: u64| status & 0xFFFF_FFFF;
        let hole_of = |status: u64| (((status >> 32) & 0xFFFF) as u32, (status >> 48) as u32);
        let mut late: Vec<u32> = Vec::new();
        for (particle, ((&done, &status), &energy)) in done.iter().zip(&status).zip(&chain_energy).enumerate() {
            round.cycles[particle % groups] += done;
            round.steps += steps_of(status);
            round.perfect_energy += EnergySum::from(Energy::from_bits(energy));
            if done < cycles || hole_of(status).0 != NONE {
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
            std::mem::swap(&mut self.sigma, &mut self.next);
            return Ok(round);
        }
        let clock = Instant::now();

        // Continue the late chains on the host: the same chain on the same
        // stream, from where the device stopped it.
        let targets = self.device(&late)?;
        let positions = self.position_out.to_host_vec(&self.stream)?;
        let (config, _) = self.config(late.len(), 0);
        let gather = |from: &DeviceBuffer<u16>, targets: &DeviceBuffer<u32>| -> Result<Vec<u16>> {
            let count = targets.len();
            let mut rows = self.zeros::<u16>(count * n)?;
            // SAFETY: the arguments match the kernel; `rows` holds one row
            // per target and every target is below `count`.
            unsafe {
                self.module.gather(self.stream.as_ref(), config, from, targets, &mut rows, n as u32, count as u32)
            }?;
            Ok(rows.to_host_vec(&self.stream)?)
        };
        let mut rows = gather(&self.next, &targets)?;
        let far_tables = tables.far(&self.energy_bits);
        let anchor = tables.anchor as u32;
        // one small record per chain: a round's tables are too large to
        // build for every task of the pool
        let finished: Vec<(FarChain, VisitList)> = rows
            .par_chunks_mut(n)
            .zip(&late)
            .map_init(
                || vec![0u16; 2 * n],
                |words, (row, &particle)| {
                    let particle = particle as usize;
                    let (hole_u, hole_v) = hole_of(status[particle]);
                    let position = positions[particle];
                    let mut rng = Stream32::at(seeds[particle], position >> 3, (position & 7) as u32);
                    let lane = Lane::on(words, n);
                    let mut tally = VisitList::default();
                    // SAFETY: `row` holds n words, the lane two arrays of n,
                    // and the tables are n x n.
                    let chain = unsafe {
                        let mut chain = far_chain::begin(&lane, row.as_ptr(), &far_tables);
                        chain.hole_u = hole_u;
                        chain.hole_v = hole_v;
                        chain.perfect_steps = done[particle];
                        chain.steps = steps_of(status[particle]);
                        far_chain::run(&lane, &far_tables, anchor, cycles, max_steps, &mut rng, &mut chain, &mut tally);
                        far_chain::store(&lane, row.as_mut_ptr());
                        chain
                    };
                    (chain, tally)
                },
            )
            .collect();
        // a chain that ran out of steps leaves its particle where it was
        let mut capped: Vec<u32> = Vec::new();
        for (index, ((chain, tally), &particle)) in finished.iter().zip(&late).enumerate() {
            let particle = particle as usize;
            let complete = chain.hole_u == NONE && chain.perfect_steps == cycles;
            round.add(particle % groups, tally);
            round.close(particle % groups, chain, done[particle], steps_of(status[particle]), !complete);
            if !complete {
                capped.push(index as u32);
            }
        }
        if !capped.is_empty() {
            let from: Vec<u32> = capped.iter().map(|&index| late[index as usize]).collect();
            let starts = gather(&self.sigma, &self.device(&from)?)?;
            for (start, &index) in starts.chunks(n).zip(&capped) {
                rows[index as usize * n..][..n].copy_from_slice(start);
            }
        }
        let rows = self.device(&rows)?;
        // SAFETY: as for the gather above; the targets are distinct.
        unsafe {
            self.module.scatter(
                self.stream.as_ref(),
                config,
                &rows,
                &targets,
                &mut self.next,
                n as u32,
                late.len() as u32,
            )
        }?;
        std::mem::swap(&mut self.sigma, &mut self.next);
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
        long_mutation::<f32>();
        long_mutation::<f64>();
    }

    fn long_mutation<T: Float + AsPrimitive<u64> + DeviceCopy + Send + Sync>() {
        let scaled = instance();
        let tables = Tables::<u64>::build(&scaled, BETA, &LOG_W, 0);
        let mut cuda = Cuda::<T>::new(&scaled, 0).unwrap();
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
        sampler_and_short_mutation::<f32>();
        sampler_and_short_mutation::<f64>();
    }

    fn sampler_and_short_mutation<T: Float + AsPrimitive<u64> + DeviceCopy + Send + Sync>() {
        let scaled = instance();
        let tables = Tables::<u64>::build(&scaled, BETA, &LOG_W, 0);
        let stages = StageSchedule::new(scaled.n, 32.0);
        let stage_time = stages.rigorous_stage_time(1e-3);
        let mut cuda = Cuda::<T>::new(&scaled, 0).unwrap();
        let cost =
            cuda.anneal(&tables, &stages, stage_time, 64, COUNT, 0, Stream { seed: 5, phase: 2, level: 0 }).unwrap();
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
        let tables = Tables::<u64>::build(&scaled, beta, &log_w, anchor);
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
        let mut cuda = Cuda::<f32>::new(&scaled, 0).unwrap();
        cuda.anneal(
            &tables,
            &stages,
            stages.rigorous_stage_time(1e-3),
            64,
            8192,
            0,
            Stream { seed: 9, phase: 1, level: 0 },
        )
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
            visit_energy += round.visit_energy[v].to_num::<f64>();
        }
        // the population is still on the target, and the energy sums are sane
        check(&scaled, beta, &cuda.energies().unwrap());
        let mean: f64 = perms.iter().zip(&weights).map(|(p, w)| scaled.energy_of(p) * w / z_perfect).sum();
        assert!((round.perfect_energy.to_num::<f64>() / cycles as f64 / mean - 1.0).abs() < 0.02);
        assert!(visit_energy > 0.0);
    }

    /// Resampling copies whole particles.
    #[test]
    fn resampling_copies_particles() {
        let scaled = instance();
        let perms = permutations(scaled.n);
        let flat: Vec<u32> = perms.iter().flatten().copied().collect();
        let mut cuda = Cuda::<f32>::new(&scaled, 0).unwrap();
        cuda.load(&flat, 0).unwrap();
        let before = cuda.energies().unwrap();
        let ancestors: Vec<u32> = (0..perms.len() as u32).rev().collect();
        cuda.resample(&ancestors).unwrap();
        let after = cuda.energies().unwrap();
        let expected: Vec<f64> = before.iter().rev().copied().collect();
        assert_eq!(after, expected);
        assert_eq!(before, perms.iter().map(|p| scaled.energy_of(p)).collect::<Vec<_>>());
    }

    /// FAR is integer arithmetic on a seeded stream, so the rayon engine, a
    /// full launch and a launch that hands most chains to the host must
    /// return the same round and leave the same population.
    #[test]
    fn far_matches_the_rayon_engine_exactly() {
        use permanent_rayon::Rayon;
        let n = 24;
        let mut a = vec![0.0; n * n];
        for u in 0..n {
            for v in 0..n {
                if (u * 5 + v * 3) % 4 != 1 || u == v {
                    a[u * n + v] = (((u * 7 + v * 11) % 13) as f64 - 6.0).exp();
                }
            }
        }
        let Prepared::Scaled(scaled) = prepare(&Problem { n, a }) else { unreachable!() };
        let log_w: Vec<f64> = (0..n).map(|v| (n as f64).ln() + 0.1 * (v % 5) as f64).collect();
        let tables = Tables::<u64>::build(&scaled, 0.6, &log_w, 3);
        let identity: Vec<u32> = (0..4096).flat_map(|_| 0..n as u32).collect();
        let stream = Stream { seed: 11, phase: 4, level: 2 };
        let mut rayon = Rayon::<f32, Stream32>::new(&scaled);
        rayon.load(&identity, 100).unwrap();
        // 40 steps per cycle is too few for some chains, which are then capped
        let expected = rayon.far(&tables, 50, 2000, 16, stream).unwrap();
        assert!(expected.capped > 0 && expected.capped < 4096);
        for budget in [64.0, 0.5] {
            let mut cuda = Cuda::<f32>::new(&scaled, 0).unwrap();
            cuda.load(&identity, 100).unwrap();
            cuda.far_budget = budget;
            assert_eq!(cuda.far(&tables, 50, 2000, 16, stream).unwrap(), expected, "device budget {budget}");
            assert_eq!(cuda.energies().unwrap(), rayon.energies().unwrap());
        }
    }
}
