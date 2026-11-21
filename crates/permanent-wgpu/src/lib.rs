//! The population on a GPU through wgpu: Vulkan, Metal or DirectX on a
//! desktop, WebGPU in a browser.
//!
//! The kernels (`kernels.rs`) are written with CubeCL and compiled to WGSL,
//! so they use only what WebGPU has: 32-bit integers and `f32`. They follow
//! the chains of `permanent_core::chain` draw for draw; the FAR kernel
//! returns the same counts as the other engines for the same seeds.
//!
//! A dispatch runs a bounded number of steps and leaves every chain's state
//! on the device, and the host dispatches again until all chains are over.
//! A browser resets a device whose dispatch runs too long, so nothing here
//! depends on a dispatch finishing its chains.
//!
//! Every operation is an `async fn`, because a browser cannot wait for the
//! device. On other targets the [`Engine`] implementation waits for them.

pub mod kernels;

use anyhow::{Context, Result, anyhow, ensure};
// the browser target has no clock in std
use cubecl::prelude::*;
use cubecl::server::Handle;
use fixed::types::I32F32;
use permanent_core::chain::dynamics::{self, STAGE_TICKS};
use permanent_core::chain::far::{self as far_chain, FarChain};
use permanent_core::chain::lane::{Lane, MAX_N, NONE};
use permanent_core::chain::matching::{Matching, words};
use permanent_core::chain::stream::Stream as Philox;
use permanent_core::chain::{Energy, EnergySum};
use permanent_core::engine::{FarRound, Stream, VisitList};
use permanent_core::sampler::{MutateCost, SampleCost, StageSchedule, mutate_event_cap, penalty};
use permanent_core::scaling::Scaled;
use permanent_core::stats::{mix, stream_seed};
use permanent_core::tables::{Tables, fraction};
use rayon::prelude::*;
// std's clock, or the browser's where std has none
use web_time::Instant;

use kernels::{
    AHEAD, CAPPED, CAPPED_CHAINS, CLASSES, ENERGY_TOTAL, EVENTS, FLAGS, HANDED_BLOCKS, HANDED_CYCLES, HANDED_ENERGY,
    HANDED_HOLE_U, HANDED_HOLE_V, HANDED_KEY, HANDED_PARTICLE, HANDED_POSITION, HANDED_ROW, HANDED_STEPS, LATE,
    MEASURED, OVER, PERFECT, POSITION, STATE, STEPS_TOTAL, handed, lane_words,
};

/// Threads of a workgroup.
const GROUP: u32 = 64;
/// Most particles resident at once.
const MAX_PARTICLES: usize = 1 << 20;
/// Most chains a FAR call hands to the host.
const HANDED_MAX: usize = 1 << 14;
/// Particles still without a perfect matching that the host takes over
/// from the device's sampler.
const HOST_TRIALS: usize = 512;

/// The population on one wgpu device.
pub struct Wgpu<'a> {
    scaled: &'a Scaled,
    client: Client,
    name: String,
    /// The energies in the FAR chain's fixed point.
    energy_bits: Vec<i64>,
    /// Entry energies and log-gaps for `kernels::measure`.
    measures: Handle,
    first_id: u64,
    count: usize,
    /// The particles: `count` rows of n words.
    sigma: Handle,
    /// `Tables::id` of the row alias table on the device, and the table.
    rows: (u64, Handle),
    /// Likewise the FAR tables.
    far: (u64, Handle),
    /// The FAR chains' working memory, kept from call to call, with the
    /// words it holds.
    scratch: (usize, Handle),
    /// Events one dispatch runs a move or an annealing trial for.
    pub budget: u32,
    /// Bytes of working memory a dispatch may touch. Past the size of the
    /// device's cache, more chains at once make all of them slower: on an
    /// RTX 4090 throughput peaks between 32 and 80 MB and falls by a fifth
    /// at 128 MB.
    pub working_set: usize,
    /// Steps per cycle a FAR dispatch allows before a chain is handed to
    /// the host; tuned so that about half a percent of the chains overrun.
    pub far_budget: f64,
    /// Seconds spent in each part of the FAR calls, for the debug log.
    far_seconds: FarSeconds,
}

#[derive(Debug, Default)]
struct FarSeconds {
    calls: u64,
    dispatches: u64,
    late_chains: u64,
    setup: f64,
    device: f64,
    host_chains: f64,
}

impl Drop for Wgpu<'_> {
    fn drop(&mut self) {
        if self.far_seconds.calls > 0 {
            tracing::debug!(far = ?self.far_seconds, budget_per_cycle = self.far_budget, "wgpu engine FAR calls");
        }
    }
}

/// A buffer of `len` words as a kernel argument.
fn buffer(handle: &Handle, len: usize) -> BufferArg {
    // SAFETY: every handle here is allocated with the length it is used at
    unsafe { BufferArg::from_raw_parts(handle.clone(), len) }
}

fn halves(x: u64) -> (u32, u32) {
    (x as u32, (x >> 32) as u32)
}

fn whole(low: u32, high: u32) -> u64 {
    (high as u64) << 32 | low as u64
}

impl<'a> Wgpu<'a> {
    /// The engine on the default wgpu adapter.
    #[cfg(not(target_family = "wasm"))]
    pub fn new(scaled: &'a Scaled) -> Result<Self> {
        use cubecl::Device;
        use cubecl::device::WgpuDeviceKind;
        let device = Device::wgpu(WgpuDeviceKind::default()).map_err(|error| anyhow!("no wgpu device: {error:?}"))?;
        Self::on(scaled, device.client())
    }

    /// The engine on the browser's WebGPU adapter.
    #[cfg(target_family = "wasm")]
    pub async fn new(scaled: &'a Scaled) -> Result<Self> {
        use cubecl::Device;
        use cubecl::device::WgpuDeviceKind;
        use cubecl::wgpu::{WebGpu, WgpuDevice, init_setup_async};
        use std::sync::atomic::{AtomicBool, Ordering};
        // A browser hands out its adapter asynchronously, and CubeCL takes
        // one setup per device; once it is done, the client below finds it.
        static SET_UP: AtomicBool = AtomicBool::new(false);
        if !SET_UP.swap(true, Ordering::Relaxed) {
            init_setup_async::<WebGpu>(&WgpuDevice::default(), Default::default()).await;
        }
        let device = Device::wgpu(WgpuDeviceKind::default()).map_err(|error| anyhow!("no WebGPU device: {error:?}"))?;
        Self::on(scaled, device.client())
    }

    /// The engine on the device behind `client`.
    pub fn on(scaled: &'a Scaled, client: Client) -> Result<Self> {
        let n = scaled.n;
        ensure!(n <= MAX_N, "the wgpu engine supports n up to {MAX_N}");
        // 32 fraction bits: a matching's energy is off by less than n 2^-33
        let fixed = |x: f64| if x == f64::NEG_INFINITY { i64::MIN } else { I32F32::from_num(x).to_bits() };
        let measures: Vec<i64> =
            scaled.energy.iter().map(|&e| fixed(e)).chain(scaled.log_gap.iter().map(|&g| fixed(g))).collect();
        let name = format!("wgpu ({})", client.properties().identity.name);
        let placeholder = client.empty(4);
        Ok(Wgpu {
            scaled,
            name,
            energy_bits: scaled.fixed_energy(),
            measures: client.create_from_slice(i64::as_bytes(&measures)),
            first_id: 0,
            count: 0,
            sigma: placeholder.clone(),
            rows: (0, placeholder.clone()),
            far: (0, placeholder.clone()),
            scratch: (0, placeholder),
            far_budget: 64.0,
            working_set: 48 << 20,
            far_seconds: FarSeconds::default(),
            // long enough that most chains finish in one dispatch, short
            // enough that a dispatch stays well under a second
            budget: 1 << 16,
            client,
        })
    }

    pub fn name(&self) -> String {
        self.name.clone()
    }

    pub fn capacity(&self) -> usize {
        // the largest buffer is the working memory, and a device caps the size of one
        let per_particle = lane_words(self.scaled.n) * size_of::<u32>();
        MAX_PARTICLES.min(self.client.properties().memory.max_page_size as usize / per_particle)
    }

    pub fn learn_particles(&self) -> usize {
        // Every chain of a call pays for loading its matching, so fewer
        // chains with more cycles each are faster: about 64 cycles for each
        // in a learning round. On an RTX 4090 the 512 x 512 dimer matrix
        // learns in 124 s with 65536 chains and in 194 s with 262144.
        let n = self.scaled.n;
        (n * n).clamp(4096, 1 << 16).min(self.capacity())
    }

    fn grid(&self) -> CubeCount {
        CubeCount::Static((self.count as u32).div_ceil(GROUP), 1, 1)
    }

    /// Particles a dispatch runs when each takes `words` of working memory.
    fn batch(&self, words: usize) -> usize {
        (self.working_set / (words * size_of::<u32>())).clamp(GROUP as usize, self.count.max(GROUP as usize))
    }

    /// Replace the particles `targets` by `rows`, one row of n for each.
    fn scatter(&self, rows: &[u32], targets: &[u32]) {
        if targets.is_empty() {
            return;
        }
        let n = self.scaled.n;
        let device_rows = self.client.create_from_slice(u32::as_bytes(rows));
        let device_targets = self.client.create_from_slice(u32::as_bytes(targets));
        // SAFETY: `rows` holds one row per target, and the targets are
        // distinct particles.
        unsafe {
            kernels::scatter::launch_unchecked(
                &self.client,
                CubeCount::Static((targets.len() as u32).div_ceil(GROUP), 1, 1),
                CubeDim::new_1d(GROUP),
                buffer(&device_rows, rows.len()),
                buffer(&device_targets, targets.len()),
                buffer(&self.sigma, self.count * n),
                n as u32,
                targets.len() as u32,
            );
        }
    }

    /// `words` of working memory, grown when more are needed.
    fn working(&mut self, words: usize) -> Handle {
        if self.scratch.0 < words {
            self.scratch = (words, self.client.empty(words * size_of::<u32>()));
        }
        self.scratch.1.clone()
    }

    async fn read(&self, handle: &Handle) -> Result<Vec<u32>> {
        let bytes = self.client.read_async(vec![handle.clone()]).await.map_err(|error| anyhow!("{error:?}"))?;
        Ok(u32::from_bytes(&bytes[0]).to_vec())
    }

    /// A fresh chain state for every particle that `seed` gives a stream
    /// to; the others are marked over.
    fn states(&self, seed: impl Fn(usize) -> Option<u64>) -> Handle {
        let mut state = vec![0u32; self.count * STATE];
        for (particle, state) in state.chunks_mut(STATE).enumerate() {
            match seed(particle) {
                Some(seed) => {
                    (state[0], state[1]) = halves(seed);
                    state[POSITION] = AHEAD;
                }
                None => state[FLAGS] = OVER,
            }
        }
        self.client.create_from_slice(u32::as_bytes(&state))
    }

    fn seed(&self, stream: Stream, particle: usize) -> u64 {
        stream_seed(stream.seed, stream.phase, stream.level, self.first_id + particle as u64)
    }

    /// Put the row alias table of `tables` on the device, unless it is there.
    fn sync_rows(&mut self, tables: &Tables<u64>) {
        if self.rows.0 != tables.id {
            self.rows = (tables.id, self.client.create_from_slice(u64::as_bytes(&tables.packed_rows)));
        }
    }

    pub fn load(&mut self, perms: &[u32], first_id: u64) {
        self.count = perms.len() / self.scaled.n;
        self.first_id = first_id;
        self.sigma = self.client.create_from_slice(u32::as_bytes(perms));
    }

    /// The particles, as rows of n.
    pub async fn particles(&self) -> Result<Vec<u32>> {
        self.read(&self.sigma).await
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn anneal(
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
        self.count = count;
        self.first_id = first_id;
        self.sigma = self.client.empty(count * n * size_of::<u32>());
        // the row alias table, then each stage's probability of staying unmatched
        let mut entries = tables.packed_rows.clone();
        entries.extend(stages.unassign::<u64>());
        let device_tables = self.client.create_from_slice(u64::as_bytes(&entries));
        let scratch = self.client.empty(count * lane_words(n) * size_of::<u32>());
        let ticks_per_time = (STAGE_TICKS as f64 / stage_time) as f32;
        let (fast_low, fast_high) = halves(stages.fast());
        let (cap_low, cap_high) = halves(stages.event_cap(n, stage_time));
        let mut cost = SampleCost::default();
        let mut pending = vec![true; count];
        for trial in 0..max_trials {
            if !pending.contains(&true) {
                break;
            }
            // a trial that misses is run again on a stream of its own
            let state =
                self.states(|particle| pending[particle].then(|| mix(self.seed(stream, particle).wrapping_add(trial))));
            loop {
                let tally = self.client.create_from_slice(u32::as_bytes(&[0]));
                // SAFETY: the buffers have the lengths the kernel indexes
                // them with: an n x n table and a threshold per stage,
                // `count` particles of n words, their working memory and state.
                unsafe {
                    kernels::anneal::launch_unchecked(
                        &self.client,
                        self.grid(),
                        CubeDim::new_1d(GROUP),
                        buffer(&device_tables, 2 * entries.len()),
                        buffer(&self.sigma, count * n),
                        buffer(&scratch, count * lane_words(n)),
                        buffer(&state, count * STATE),
                        buffer(&tally, 1),
                        n as u32,
                        count as u32,
                        stages.stages() as u32,
                        ticks_per_time,
                        fast_low,
                        fast_high,
                        cap_low,
                        cap_high,
                        self.budget,
                    );
                }
                if self.read(&tally).await?[0] == 0 {
                    break;
                }
            }
            let state = self.read(&state).await?;
            for (pending, state) in pending.iter_mut().zip(state.chunks(STATE)).filter(|(pending, _)| **pending) {
                cost.trials += 1;
                cost.events += whole(state[EVENTS], state[EVENTS + 1]);
                if state[FLAGS] & PERFECT != 0 {
                    cost.accepted += 1;
                    *pending = false;
                } else {
                    cost.over_budget += (state[FLAGS] & CAPPED != 0) as u64;
                }
            }
            // A trial lasts as long on the device for a few particles as
            // for all of them, so the last ones are finished on the host.
            let left: Vec<(usize, u64)> = (0..count)
                .filter(|&particle| pending[particle])
                .map(|particle| (particle, self.seed(stream, particle)))
                .collect();
            if left.len() <= HOST_TRIALS {
                let unassign = stages.unassign::<u64>();
                let (fast, cap) = (stages.fast(), stages.event_cap(n, stage_time));
                let finished: Vec<(SampleCost, Option<Vec<u32>>)> = left
                    .par_iter()
                    .map_init(
                        || (vec![0u16; words(n)], vec![0u16; n]),
                        |(memory, row), &(_, seed)| {
                            let mut matching = Matching::on(memory, n);
                            let mut cost = SampleCost::default();
                            for trial in trial + 1..max_trials {
                                let mut rng = Philox::new(mix(seed.wrapping_add(trial)));
                                // SAFETY: the lane holds words(n) words and the tables are n x n
                                let outcome = unsafe {
                                    dynamics::trial::<u64, f32, _>(
                                        &mut matching,
                                        n as u32,
                                        &tables.packed_rows,
                                        &unassign,
                                        ticks_per_time,
                                        fast,
                                        cap,
                                        &mut rng,
                                    )
                                };
                                cost.trials += 1;
                                cost.events += outcome.events;
                                if !outcome.over_budget && matching.is_perfect() {
                                    cost.accepted += 1;
                                    // SAFETY: `row` holds n words
                                    unsafe { matching.store(row.as_mut_ptr()) };
                                    return (cost, Some(row.iter().map(|&v| v as u32).collect()));
                                }
                                cost.over_budget += outcome.over_budget as u64;
                            }
                            (cost, None)
                        },
                    )
                    .collect();
                let mut moved: Vec<u32> = Vec::new();
                let mut rows: Vec<u32> = Vec::new();
                for (&(particle, _), (trials, row)) in left.iter().zip(finished) {
                    cost.add(trials);
                    if let Some(row) = row {
                        pending[particle] = false;
                        moved.push(particle as u32);
                        rows.extend(row);
                    }
                }
                self.scatter(&rows, &moved);
                break;
            }
        }
        let missing = pending.iter().filter(|&&pending| pending).count();
        ensure!(
            missing == 0,
            "the sampler could not produce {missing} of {count} perfect matchings; the anchor weights are out of range"
        );
        Ok(cost)
    }

    pub async fn mutate(&mut self, tables: &Tables<u64>, perfect_time: f64, stream: Stream) -> Result<MutateCost> {
        let n = self.scaled.n;
        let count = self.count;
        self.sync_rows(tables);
        let state = self.states(|particle| Some(self.seed(stream, particle)));
        let scratch = self.client.empty(count * lane_words(n) * size_of::<u32>());
        let times = self.client.empty(count * 2 * size_of::<f32>());
        let penalty = penalty(n);
        let (fast_low, fast_high) = halves(1 + penalty as u64);
        let (unassign_low, unassign_high) = halves(fraction(1.0 / (1.0 + penalty)));
        let (cap_low, cap_high) = halves(mutate_event_cap(n, perfect_time));
        loop {
            let tally = self.client.create_from_slice(u32::as_bytes(&[0]));
            // SAFETY: the buffers have the lengths the kernel indexes them
            // with: an n x n table, `count` particles of n words, their
            // working memory, state and two times each.
            unsafe {
                kernels::mutate::launch_unchecked(
                    &self.client,
                    self.grid(),
                    CubeDim::new_1d(GROUP),
                    buffer(&self.rows.1, 2 * n * n),
                    buffer(&self.sigma, count * n),
                    buffer(&scratch, count * lane_words(n)),
                    buffer(&state, count * STATE),
                    buffer(&times, count * 2),
                    buffer(&tally, 1),
                    n as u32,
                    count as u32,
                    perfect_time as f32,
                    fast_low,
                    fast_high,
                    unassign_low,
                    unassign_high,
                    cap_low,
                    cap_high,
                    self.budget,
                );
            }
            if self.read(&tally).await?[0] == 0 {
                break;
            }
        }
        let state = self.read(&state).await?;
        let bytes = self.client.read_async(vec![times]).await.map_err(|error| anyhow!("{error:?}"))?;
        let mut cost = MutateCost::default();
        for (state, times) in state.chunks(STATE).zip(f32::from_bytes(&bytes[0]).chunks(2)) {
            cost.events += whole(state[EVENTS], state[EVENTS + 1]);
            cost.over_budget += (state[FLAGS] & CAPPED != 0) as u64;
            cost.time += times[1] as f64;
        }
        Ok(cost)
    }

    /// Energy and regularization log-gap of every particle.
    async fn measure(&self) -> Result<(Vec<f64>, Vec<f64>)> {
        let n = self.scaled.n;
        let out = self.client.empty(self.count * MEASURED * size_of::<u32>());
        // SAFETY: the tables hold two n x n arrays of 64-bit entries, and
        // `sigma` and `out` one row per particle.
        unsafe {
            kernels::measure::launch_unchecked(
                &self.client,
                self.grid(),
                CubeDim::new_1d(GROUP),
                buffer(&self.measures, 4 * n * n),
                buffer(&self.sigma, self.count * n),
                buffer(&out, self.count * MEASURED),
                n as u32,
                self.count as u32,
            );
        }
        let out = self.read(&out).await?;
        let value = |low: u32, high: u32| I32F32::from_bits(whole(low, high) as i64).to_num::<f64>();
        let energies = out.chunks(MEASURED).map(|m| value(m[0], m[1])).collect();
        let gaps = out.chunks(MEASURED).map(|m| if m[4] > 0 { f64::NEG_INFINITY } else { value(m[2], m[3]) }).collect();
        Ok((energies, gaps))
    }

    pub async fn energies(&mut self) -> Result<Vec<f64>> {
        Ok(self.measure().await?.0)
    }

    pub async fn log_gaps(&mut self) -> Result<Vec<f64>> {
        Ok(self.measure().await?.1)
    }

    pub fn resample(&mut self, ancestors: &[u32]) {
        let n = self.scaled.n;
        let count = ancestors.len();
        let next = self.client.empty(count * n * size_of::<u32>());
        let device_ancestors = self.client.create_from_slice(u32::as_bytes(ancestors));
        // SAFETY: every ancestor is a particle of `sigma`, and `next` holds
        // one row per ancestor.
        unsafe {
            kernels::gather::launch_unchecked(
                &self.client,
                CubeCount::Static((count as u32).div_ceil(GROUP), 1, 1),
                CubeDim::new_1d(GROUP),
                buffer(&self.sigma, self.count * n),
                buffer(&device_ancestors, count),
                buffer(&next, count * n),
                n as u32,
                count as u32,
            );
        }
        self.sigma = next;
        self.count = count;
    }

    pub async fn far(
        &mut self,
        tables: &Tables<u64>,
        cycles: u64,
        max_steps: u64,
        groups: usize,
        stream: Stream,
    ) -> Result<FarRound> {
        let n = self.scaled.n;
        let count = self.count;
        let clock = Instant::now();
        let cycles32 = u32::try_from(cycles).context("too many cycles for one call")?;
        // steps are counted in a word; a chain that long is capped anyway
        let max_steps = max_steps.min(u32::MAX as u64 - 1);
        if self.far.0 != tables.id {
            // every entry is 64 bits: two words, low first
            let mut entries: Vec<u64> = Vec::with_capacity(5 * n * n);
            entries.extend_from_slice(&tables.packed_rows);
            entries.extend_from_slice(&tables.packed_cols);
            entries.extend_from_slice(&tables.far_moves);
            entries.extend(self.energy_bits.iter().map(|&bits| bits as u64));
            // handing the vector over saves a copy of 40 n^2 bytes
            self.far = (tables.id, self.client.create(cubecl::bytes::Bytes::from_elems(entries)));
        }
        let seeds: Vec<u64> = (0..count).map(|particle| self.seed(stream, particle)).collect();
        let device_seeds = self.client.create_from_slice(u64::as_bytes(&seeds));
        // the chains run a batch at a time, in one working buffer
        let batch = self.batch(n);
        let scratch = self.working(batch * n);
        // Chains stop at a budget and the few that overran are continued on
        // the host. A group's visits to a class are counted in a word.
        let budget = ((self.far_budget * cycles as f64).ceil() as u64 + 32)
            .min(max_steps)
            .min((u32::MAX as usize / count.div_ceil(groups)) as u64) as u32;
        let capacity = (count / 16).clamp(1024.min(count), HANDED_MAX);
        let record = handed(n);
        let chains = self.client.empty(capacity * record * size_of::<u32>());
        let classes = CLASSES + 2 * n;
        let ended = classes + groups * n;
        let tally_len = ended + groups;
        let tally = self.client.create_from_slice(u32::as_bytes(&vec![0u32; tally_len]));
        self.far_seconds.calls += 1;
        self.far_seconds.setup += clock.elapsed().as_secs_f64();
        let clock = Instant::now();
        // the batches follow one another on the device; nothing is read
        // back between them
        for first in (0..count).step_by(batch) {
            let len = batch.min(count - first);
            // SAFETY: the buffers have the lengths the kernel indexes them
            // with: n x n tables, `count` particles, working memory for a
            // batch, `groups` rows of visits and `capacity` records.
            unsafe {
                kernels::far::launch_unchecked(
                    &self.client,
                    CubeCount::Static((len as u32).div_ceil(GROUP), 1, 1),
                    CubeDim::new_1d(GROUP),
                    buffer(&self.far.1, 10 * n * n),
                    buffer(&self.sigma, count * n),
                    buffer(&scratch, batch * n),
                    buffer(&tally, tally_len),
                    buffer(&device_seeds, 2 * count),
                    buffer(&chains, capacity * record),
                    n as u32,
                    len as u32,
                    tables.anchor as u32,
                    groups as u32,
                    cycles32,
                    budget,
                    max_steps as u32,
                    capacity as u32,
                    first as u32,
                );
            }
            self.far_seconds.dispatches += 1;
        }
        let tally = self.read(&tally).await?;
        let mut round = FarRound::new(n, groups);
        round.capped = tally[CAPPED_CHAINS] as u64;
        round.steps = whole(tally[STEPS_TOTAL], tally[STEPS_TOTAL + 1]);
        let bits = whole(tally[ENERGY_TOTAL], tally[ENERGY_TOTAL + 1]) as i64;
        round.perfect_energy = EnergySum::from(Energy::from_bits(bits));
        for v in 0..n {
            let bits = whole(tally[CLASSES + 2 * v], tally[CLASSES + 2 * v + 1]) as i64;
            round.visit_energy[v] = EnergySum::from(Energy::from_bits(bits));
        }
        for (group, visits) in tally[classes..ended].chunks(n).enumerate() {
            for (total, &visits) in round.visits[group].iter_mut().zip(visits) {
                *total = visits as u64;
            }
        }
        for (total, &cycles) in round.cycles.iter_mut().zip(&tally[ended..]) {
            *total = cycles as u64;
        }
        // chains that asked for a record beyond the last one ran to their end
        let late = (tally[LATE] as usize).min(capacity);
        // aim for about half a percent of late chains: fewer makes a
        // dispatch wait for the slowest, more moves the work to the host
        let share = late as f64 / count as f64;
        if share > 0.01 {
            self.far_budget *= 1.5;
        } else if share < 0.002 {
            self.far_budget = (self.far_budget / 1.2).max(8.0);
        }
        self.far_seconds.device += clock.elapsed().as_secs_f64();
        self.far_seconds.late_chains += late as u64;
        if late == 0 {
            return Ok(round);
        }
        let clock = Instant::now();

        // Continue the late chains on the host: the same chain on the same
        // stream, from where the device stopped it.
        let unused = ((capacity - late) * record * size_of::<u32>()) as u64;
        let records = self.read(&chains.offset_end(unused)).await?;
        let far_tables = tables.far(&self.energy_bits);
        let anchor = tables.anchor as u32;
        // one small record per chain: a round's tables are too large to
        // build for every task of the pool
        let finished: Vec<(FarChain, VisitList, Option<Vec<u32>>)> = records
            .par_chunks(record)
            .map_init(
                || (vec![0u16; n], vec![0u16; 2 * n]),
                |(stored, memory), chain| {
                    for (word, &v) in stored.iter_mut().zip(&chain[HANDED_ROW..]) {
                        *word = v as u16;
                    }
                    // the device keeps `AHEAD - position` generated words unread
                    let generated = whole(chain[HANDED_BLOCKS], chain[HANDED_BLOCKS + 1]);
                    let drawn = 4 * generated - (AHEAD - chain[HANDED_POSITION]) as u64;
                    let blocks = drawn.div_ceil(4);
                    let seed = whole(chain[HANDED_KEY], chain[HANDED_KEY + 1]);
                    let mut rng = Philox::at(seed, blocks, (4 * blocks - drawn) as u32);
                    let lane = Lane::on(memory, n);
                    let mut tally = VisitList::default();
                    // SAFETY: `stored` holds n words, the lane two arrays of
                    // n, and the tables are n x n.
                    let run = unsafe {
                        let mut run = far_chain::begin(&lane, stored.as_ptr(), &far_tables);
                        run.hole_u = chain[HANDED_HOLE_U];
                        run.hole_v = chain[HANDED_HOLE_V];
                        run.perfect_steps = chain[HANDED_CYCLES] as u64;
                        run.steps = chain[HANDED_STEPS] as u64;
                        far_chain::run(&lane, &far_tables, anchor, cycles, max_steps, &mut rng, &mut run, &mut tally);
                        far_chain::store(&lane, stored.as_mut_ptr());
                        run
                    };
                    // the device had already summed part of the perfect energy
                    let before = Energy::from_bits(whole(chain[HANDED_ENERGY], chain[HANDED_ENERGY + 1]) as i64);
                    let run = FarChain { perfect_energy: run.perfect_energy + before, ..run };
                    let complete = run.hole_u == NONE && run.perfect_steps == cycles;
                    (run, tally, complete.then(|| stored.iter().map(|&v| v as u32).collect()))
                },
            )
            .collect();
        // a chain that ran out of steps leaves its particle where it was
        let mut moved: Vec<u32> = Vec::new();
        let mut rows: Vec<u32> = Vec::new();
        for ((run, tally, row), chain) in finished.iter().zip(records.chunks(record)) {
            let particle = chain[HANDED_PARTICLE];
            let group = particle as usize % groups;
            round.add(group, tally);
            round.close(group, run, 0, 0, row.is_none());
            if let Some(row) = row {
                moved.push(particle);
                rows.extend_from_slice(row);
            }
        }
        self.scatter(&rows, &moved);
        self.far_seconds.host_chains += clock.elapsed().as_secs_f64();
        Ok(round)
    }
}

#[cfg(not(target_family = "wasm"))]
mod blocking {
    use super::*;
    use cubecl::future::block_on;
    use permanent_core::engine::Engine;

    impl Engine for Wgpu<'_> {
        type Data = f32;
        type Counter = u64;

        fn name(&self) -> String {
            Wgpu::name(self)
        }

        fn capacity(&self) -> usize {
            Wgpu::capacity(self)
        }

        fn learn_particles(&self) -> usize {
            Wgpu::learn_particles(self)
        }

        fn fork(&self) -> Result<Self> {
            Wgpu::on(self.scaled, self.client.clone())
        }

        fn load(&mut self, perms: &[u32], first_id: u64) -> Result<()> {
            Wgpu::load(self, perms, first_id);
            Ok(())
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
            block_on(Wgpu::anneal(self, tables, stages, stage_time, max_trials, count, first_id, stream))
        }

        fn mutate(&mut self, tables: &Tables<u64>, perfect_time: f64, stream: Stream) -> Result<MutateCost> {
            block_on(Wgpu::mutate(self, tables, perfect_time, stream))
        }

        fn energies(&mut self) -> Result<Vec<f64>> {
            block_on(Wgpu::energies(self))
        }

        fn log_gaps(&mut self) -> Result<Vec<f64>> {
            block_on(Wgpu::log_gaps(self))
        }

        fn resample(&mut self, ancestors: &[u32]) -> Result<()> {
            Wgpu::resample(self, ancestors);
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
            block_on(Wgpu::far(self, tables, cycles, max_steps, groups, stream))
        }
    }
}

#[cfg(test)]
mod test {
    //! Each test body is an `async fn` over the engine's own operations, run
    //! to completion natively and by the wasm test harness in a runtime
    //! with WebGPU.

    use super::*;
    use permanent_core::chain::stream::Stream as Philox;
    use permanent_core::engine::Engine;
    use permanent_core::problem::Problem;
    use permanent_core::scaling::{Prepared, prepare};
    use permanent_rayon::Rayon;

    #[cfg(target_family = "wasm")]
    use wasm_bindgen_test::wasm_bindgen_test;

    /// Declare a test that runs `$body` natively and in wasm.
    macro_rules! on_both {
        ($(#[$doc:meta])* $name:ident, $body:expr) => {
            $(#[$doc])*
            #[cfg(not(target_family = "wasm"))]
            #[test]
            fn $name() {
                if !has_gpu() {
                    eprintln!("skipped: no GPU adapter on this machine");
                    return;
                }
                cubecl::future::block_on($body);
            }

            $(#[$doc])*
            #[cfg(target_family = "wasm")]
            #[wasm_bindgen_test]
            async fn $name() {
                $body.await;
            }
        };
    }

    /// Whether a hardware adapter exists. The software ones lack features CubeCL asks for.
    #[cfg(not(target_family = "wasm"))]
    fn has_gpu() -> bool {
        use cubecl::wgpu::wgpu::{Backends, DeviceType, Instance, InstanceDescriptor};
        let instance = Instance::new(InstanceDescriptor::new_without_display_handle());
        cubecl::future::block_on(instance.enumerate_adapters(Backends::PRIMARY))
            .iter()
            .any(|adapter| adapter.get_info().device_type != DeviceType::Cpu)
    }

    async fn engine(scaled: &Scaled) -> Wgpu<'_> {
        #[cfg(not(target_family = "wasm"))]
        return Wgpu::new(scaled).unwrap();
        #[cfg(target_family = "wasm")]
        return Wgpu::new(scaled).await.unwrap();
    }

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
            match bins.iter_mut().find(|bin| (bin.0 - h).abs() < 1e-6) {
                Some(bin) => bin.1 += mass,
                None => bins.push((*h, mass, 0)),
            }
        }
        for h in energies {
            bins.iter_mut().find(|bin| (bin.0 - h).abs() < 1e-6).expect("energy of no matching").2 += 1;
        }
        let draws = energies.len() as f64;
        for (h, expected, observed) in bins {
            let observed = observed as f64 / draws;
            let sigma = (expected * (1.0 - expected) / draws).sqrt();
            assert!(
                (observed - expected).abs() < 5.0 * sigma + 1e-4,
                "energy {h}: expected {expected:.5}, observed {observed:.5}"
            );
        }
    }

    const BETA: f64 = 0.6;
    const LOG_W: [f64; 4] = [0.3, -0.4, 1.0, 0.0];
    const COUNT: usize = 200_000;

    on_both!(
        /// From one fixed matching, a long mutation lands on the target, also
        /// when a dispatch is far too short for it.
        a_long_mutation_reaches_the_target_distribution,
        async {
            let scaled = instance();
            let tables = Tables::<u64>::build(&scaled, BETA, &LOG_W, 0);
            for budget in [1 << 16, 7] {
                let mut wgpu = engine(&scaled).await;
                wgpu.budget = budget;
                let start: Vec<u32> = (0..COUNT).flat_map(|_| [0, 1, 2, 3]).collect();
                wgpu.load(&start, 0);
                let cost = wgpu.mutate(&tables, 24.0, Stream { seed: 5, phase: 1, level: 0 }).await.unwrap();
                assert_eq!(cost.over_budget, 0);
                check(&scaled, BETA, &wgpu.energies().await.unwrap());
            }
        }
    );

    on_both!(
        /// The sampler's accepted trials follow the target, and a short
        /// mutation of exact samples keeps it.
        the_sampler_and_a_short_mutation_keep_the_target_distribution,
        async {
            let scaled = instance();
            let tables = Tables::<u64>::build(&scaled, BETA, &LOG_W, 0);
            let stages = StageSchedule::new(scaled.n, 32.0);
            let stage_time = stages.rigorous_stage_time(1e-3);
            let mut wgpu = engine(&scaled).await;
            let stream = Stream { seed: 5, phase: 2, level: 0 };
            let cost = wgpu.anneal(&tables, &stages, stage_time, 64, COUNT, 0, stream).await.unwrap();
            assert_eq!(cost.over_budget, 0);
            assert!(COUNT as f64 / cost.trials as f64 > 0.2);
            check(&scaled, BETA, &wgpu.energies().await.unwrap());
            let cost = wgpu.mutate(&tables, 0.3, Stream { seed: 5, phase: 3, level: 0 }).await.unwrap();
            assert_eq!(cost.over_budget, 0);
            check(&scaled, BETA, &wgpu.energies().await.unwrap());
        }
    );

    on_both!(
        /// Resampling copies whole particles.
        resampling_copies_particles,
        async {
            let scaled = instance();
            let perms = permutations(scaled.n);
            let flat: Vec<u32> = perms.iter().flatten().copied().collect();
            let mut wgpu = engine(&scaled).await;
            wgpu.load(&flat, 0);
            let ancestors: Vec<u32> = (0..100).map(|i| (i * 7 % perms.len()) as u32).collect();
            wgpu.resample(&ancestors);
            let rows = wgpu.particles().await.unwrap();
            for (row, &ancestor) in rows.chunks(scaled.n).zip(&ancestors) {
                assert_eq!(row, perms[ancestor as usize]);
            }
        }
    );

    on_both!(
        /// The FAR kernel follows the same integer arithmetic and the same
        /// streams as the generic chain, so it must return the same round and
        /// leave the same population as the rayon engine on those streams,
        /// however short a dispatch is.
        far_matches_the_rayon_engine_exactly,
        async {
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
            let mut rayon = Rayon::<f32, Philox>::new(&scaled);
            rayon.load(&identity, 100).unwrap();
            // 40 steps per cycle is too few for some chains, which are then capped
            let expected = rayon.far(&tables, 50, 2000, 16, stream).unwrap();
            assert!(expected.capped > 0 && expected.capped < 4096);
            let energies = rayon.energies().unwrap();
            // dispatches that finish most chains, and ones that hand most to
            // the host; all particles at once, and in batches of a few hundred
            for (budget, working_set) in [(64.0, 64 << 20), (0.5, 100_000)] {
                let mut wgpu = engine(&scaled).await;
                wgpu.far_budget = budget;
                wgpu.working_set = working_set;
                wgpu.load(&identity, 100);
                assert_eq!(wgpu.far(&tables, 50, 2000, 16, stream).await.unwrap(), expected, "budget {budget}");
                let measured = wgpu.energies().await.unwrap();
                for (measured, exact) in measured.iter().zip(&energies) {
                    assert!((measured - exact).abs() < 1e-6, "{measured} against {exact}");
                }
            }
        }
    );
}
