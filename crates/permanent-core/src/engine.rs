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
    pub fn add(&mut self, group: usize, tally: &ChainTally) {
        for (total, visits) in self.visits[group].iter_mut().zip(&tally.visits) {
            *total += visits;
        }
        for (total, energy) in self.visit_energy.iter_mut().zip(&tally.visit_energy) {
            *total += *energy;
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

/// The visits of one chain, kept apart until it is added to a round.
pub struct ChainTally {
    visits: Vec<u64>,
    visit_energy: Vec<EnergySum>,
}

impl ChainTally {
    pub fn new(n: usize) -> Self {
        ChainTally { visits: vec![0; n], visit_energy: vec![EnergySum::ZERO; n] }
    }
}

impl Tally for ChainTally {
    #[inline]
    fn visit(&mut self, v: u32, energy: Energy) {
        self.visits[v as usize] += 1;
        self.visit_energy[v as usize] += EnergySum::from(energy);
    }
}

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
        4096
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
    fn mutate(&mut self, tables: &Tables<Self::Counter>, perfect_time: f64, stream: Stream) -> anyhow::Result<MutateCost>;

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
}
