//! Where the population lives and moves. The engines are crates of their
//! own: `permanent-rayon` (the CPU thread pool) and `permanent-cuda` (a GPU).
//!
//! An engine holds perfect matchings (the particles) and applies whole-
//! population operations to them. The particles stay resident in the engine
//! from the first temperature to the last; the estimator only sees their
//! energies and decides who survives a resampling.
//!
//! Particle p of a call draws from the stream (seed, phase, level,
//! first id + p), so results do not depend on threads, batches or ranks.

use num_traits::{Float, PrimInt, Unsigned};

use crate::chain::far::{FarChain, Tally};
use crate::chain::{Energy, EnergySum};
use crate::sampler::{MutateCost, SampleCost, StageSchedule};
use crate::tables::Tables;

/// The random streams of one engine call.
#[derive(Debug, Clone, Copy)]
pub struct Stream {
    pub seed: u64,
    pub phase: u64,
    pub level: u64,
}

/// FAR statistics of one call, pooled over the particles of each group.
#[derive(Debug, Clone, PartialEq)]
pub struct FarRound {
    /// Per group: cycles run, then visits to N(anchor, v) for each v.
    pub cycles: Vec<u64>,
    pub visits: Vec<Vec<u64>>,
    /// Sum of energies over the visits to N(anchor, v).
    pub visit_energy: Vec<EnergySum>,
    /// Sum of energies over perfect steps.
    pub perfect_energy: EnergySum,
    pub steps: u64,
    /// Chains that hit the step cap; their particles did not move.
    pub capped: u64,
}

impl FarRound {
    pub fn new(n: usize, groups: usize) -> Self {
        FarRound {
            cycles: vec![0; groups],
            visits: vec![vec![0; n]; groups],
            visit_energy: vec![EnergySum::ZERO; n],
            perfect_energy: EnergySum::ZERO,
            steps: 0,
            capped: 0,
        }
    }

    /// Count what `chain` did since it had taken `cycles` perfect steps
    /// and `steps` steps with a perfect-energy sum of zero; `capped` if it
    /// ran out of steps.
    pub fn close(&mut self, group: usize, chain: &FarChain, cycles: u64, steps: u64, capped: bool) {
        self.cycles[group] += chain.perfect_steps - cycles;
        self.perfect_energy += EnergySum::from(chain.perfect_energy);
        self.steps += chain.steps - steps;
        self.capped += capped as u64;
    }

    /// Add the visits of a chain of `group`.
    pub fn add(&mut self, group: usize, visits: &VisitList) {
        for &(v, energy) in &visits.0 {
            self.visits[group][v as usize] += 1;
            self.visit_energy[v as usize] += EnergySum::from(energy);
        }
    }

    pub fn merge(mut self, other: FarRound) -> FarRound {
        for (a, b) in self.cycles.iter_mut().zip(&other.cycles) {
            *a += b;
        }
        for (a, b) in self.visits.iter_mut().zip(&other.visits) {
            for (x, y) in a.iter_mut().zip(b) {
                *x += y;
            }
        }
        for (a, b) in self.visit_energy.iter_mut().zip(&other.visit_energy) {
            *a += b;
        }
        self.perfect_energy += other.perfect_energy;
        self.steps += other.steps;
        self.capped += other.capped;
        self
    }
}

/// The visits of one chain, counted into its group of a round.
pub struct GroupTally<'r> {
    pub round: &'r mut FarRound,
    pub group: usize,
}

impl Tally for GroupTally<'_> {
    #[inline]
    fn visit(&mut self, v: u32, energy: Energy) {
        self.round.visits[self.group][v as usize] += 1;
        self.round.visit_energy[v as usize] += EnergySum::from(energy);
    }
}

/// The visits of one chain, in order, kept apart until it is added to a
/// round. A chain is at the anchor row's hole classes for few of its steps,
/// so the list is short where a count for every class would be mostly zeros.
#[derive(Default)]
pub struct VisitList(Vec<(u32, Energy)>);

impl Tally for VisitList {
    #[inline]
    fn visit(&mut self, v: u32, energy: Energy) {
        self.0.push((v, energy));
    }
}

/// Default Phase 1 population size for engines that do not override
/// [`Engine::learn_particles`] / [`AsyncEngine::learn_particles`].
pub const DEFAULT_LEARN_PARTICLES: usize = 4096;

pub trait Engine: Send {
    /// Type of the waiting times in the move and the annealing sampler.
    type Data: Float;
    /// Type of the random thresholds the chains compare against.
    type Counter: PrimInt + Unsigned + Send + Sync;

    fn name(&self) -> String;

    /// Most particles the engine can hold.
    fn capacity(&self) -> usize;

    /// The Phase 1 population this engine handles well: FAR learning needs
    /// a fixed total of cycles, so more particles mean fewer cycles each.
    fn learn_particles(&self) -> usize {
        DEFAULT_LEARN_PARTICLES
    }

    /// How many populations of `particles` the engine moves at the same
    /// time, each on an engine of its own (`fork`), in about the time of
    /// one.
    fn concurrency(&self, _particles: usize) -> usize {
        1
    }

    /// An empty engine of the same kind on the same device or pool.
    fn fork(&self) -> anyhow::Result<Self>
    where
        Self: Sized;

    /// Replace the population by the perfect matchings in `perms` (rows of
    /// n); `first_id` is the global id of the first one.
    fn load(&mut self, perms: &[u32], first_id: u64) -> anyhow::Result<()>;

    /// Replace the population by `count` perfect matchings drawn from
    /// nothing with the annealing sampler of `sampler.rs`.
    #[allow(clippy::too_many_arguments)]
    fn anneal(
        &mut self,
        tables: &Tables<Self::Counter>,
        stages: &StageSchedule,
        stage_time: f64,
        max_trials: u64,
        count: usize,
        first_id: u64,
        stream: Stream,
    ) -> anyhow::Result<SampleCost>;

    /// Apply the mutation kernel K_t of `sampler.rs` to every particle.
    fn mutate(
        &mut self,
        tables: &Tables<Self::Counter>,
        perfect_time: f64,
        stream: Stream,
    ) -> anyhow::Result<MutateCost>;

    /// Energy of every particle.
    fn energies(&mut self) -> anyhow::Result<Vec<f64>>;

    /// Regularization log-gap of every particle.
    fn log_gaps(&mut self) -> anyhow::Result<Vec<f64>>;

    /// Particle p becomes a copy of the former particle `ancestors[p]`.
    fn resample(&mut self, ancestors: &[u32]) -> anyhow::Result<()>;

    /// Run every particle through `cycles` FAR cycles (`chain/far.rs`),
    /// leaving it on the perfect matching its chain stops on. Particle p
    /// reports into group p % groups.
    fn far(
        &mut self,
        tables: &Tables<Self::Counter>,
        cycles: u64,
        max_steps: u64,
        groups: usize,
        stream: Stream,
    ) -> anyhow::Result<FarRound>;

    /// Up to `count` matchings from the start of the current population,
    /// flattened as rows of n, for live inspection.
    fn head(&mut self, _count: usize) -> anyhow::Result<Vec<u32>> {
        Ok(Vec::new())
    }
}

/// Asynchronous counterpart of [`Engine`], for backends such as WebGPU in a
/// browser where device readbacks cannot block.
///
/// Methods mirror [`Engine`]. Any synchronous [`Engine`] adapts via
/// [`Blocking`]. Returned futures are not required to be `Send`, since browser
/// WebGPU promises live on a single thread.
pub trait AsyncEngine {
    type Data: Float;
    type Counter: PrimInt + Unsigned + Send + Sync;

    fn name(&self) -> String;
    fn capacity(&self) -> usize;

    fn learn_particles(&self) -> usize {
        DEFAULT_LEARN_PARTICLES
    }

    fn load(&mut self, perms: &[u32], first_id: u64) -> anyhow::Result<()>;

    #[allow(clippy::too_many_arguments)]
    fn anneal(
        &mut self,
        tables: &Tables<Self::Counter>,
        stages: &StageSchedule,
        stage_time: f64,
        max_trials: u64,
        count: usize,
        first_id: u64,
        stream: Stream,
    ) -> impl Future<Output = anyhow::Result<SampleCost>>;

    fn mutate(
        &mut self,
        tables: &Tables<Self::Counter>,
        perfect_time: f64,
        stream: Stream,
    ) -> impl Future<Output = anyhow::Result<MutateCost>>;

    fn energies(&mut self) -> impl Future<Output = anyhow::Result<Vec<f64>>>;
    fn log_gaps(&mut self) -> impl Future<Output = anyhow::Result<Vec<f64>>>;
    fn resample(&mut self, ancestors: &[u32]) -> anyhow::Result<()>;

    fn far(
        &mut self,
        tables: &Tables<Self::Counter>,
        cycles: u64,
        max_steps: u64,
        groups: usize,
        stream: Stream,
    ) -> impl Future<Output = anyhow::Result<FarRound>>;

    /// See [`Engine::head`].
    fn head(&mut self, _count: usize) -> impl Future<Output = anyhow::Result<Vec<u32>>> {
        core::future::ready(Ok(Vec::new()))
    }
}

/// Adapts a synchronous [`Engine`] to [`AsyncEngine`] using immediately ready
/// futures.
pub struct Blocking<E>(pub E);

impl<E: Engine> AsyncEngine for Blocking<E> {
    type Data = E::Data;
    type Counter = E::Counter;

    fn name(&self) -> String {
        self.0.name()
    }

    fn capacity(&self) -> usize {
        self.0.capacity()
    }

    fn learn_particles(&self) -> usize {
        self.0.learn_particles()
    }

    fn load(&mut self, perms: &[u32], first_id: u64) -> anyhow::Result<()> {
        self.0.load(perms, first_id)
    }

    fn anneal(
        &mut self,
        tables: &Tables<Self::Counter>,
        stages: &StageSchedule,
        stage_time: f64,
        max_trials: u64,
        count: usize,
        first_id: u64,
        stream: Stream,
    ) -> impl Future<Output = anyhow::Result<SampleCost>> {
        core::future::ready(self.0.anneal(tables, stages, stage_time, max_trials, count, first_id, stream))
    }

    fn mutate(
        &mut self,
        tables: &Tables<Self::Counter>,
        perfect_time: f64,
        stream: Stream,
    ) -> impl Future<Output = anyhow::Result<MutateCost>> {
        core::future::ready(self.0.mutate(tables, perfect_time, stream))
    }

    fn energies(&mut self) -> impl Future<Output = anyhow::Result<Vec<f64>>> {
        core::future::ready(self.0.energies())
    }

    fn log_gaps(&mut self) -> impl Future<Output = anyhow::Result<Vec<f64>>> {
        core::future::ready(self.0.log_gaps())
    }

    fn resample(&mut self, ancestors: &[u32]) -> anyhow::Result<()> {
        self.0.resample(ancestors)
    }

    fn far(
        &mut self,
        tables: &Tables<Self::Counter>,
        cycles: u64,
        max_steps: u64,
        groups: usize,
        stream: Stream,
    ) -> impl Future<Output = anyhow::Result<FarRound>> {
        core::future::ready(self.0.far(tables, cycles, max_steps, groups, stream))
    }

    fn head(&mut self, count: usize) -> impl Future<Output = anyhow::Result<Vec<u32>>> {
        core::future::ready(self.0.head(count))
    }
}
