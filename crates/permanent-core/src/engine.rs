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

use crate::far::FarStats;
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
#[derive(Debug, Clone)]
pub struct FarRound {
    /// Per group: cycles run, then visits to N(anchor, v) for each v.
    pub cycles: Vec<u64>,
    pub visits: Vec<Vec<u64>>,
    /// Sum of energies over the visits to N(anchor, v).
    pub visit_energy: Vec<f64>,
    /// Sum of energies over perfect steps.
    pub perfect_energy: f64,
    pub steps: u64,
    /// Chains that hit the step cap (their particle did not move).
    pub capped: u64,
}

impl FarRound {
    pub fn new(n: usize, groups: usize) -> Self {
        FarRound {
            cycles: vec![0; groups],
            visits: vec![vec![0; n]; groups],
            visit_energy: vec![0.0; n],
            perfect_energy: 0.0,
            steps: 0,
            capped: 0,
        }
    }

    /// Add one chain's statistics, as a member of `group`.
    pub fn add(&mut self, group: usize, stats: &FarStats) {
        self.cycles[group] += stats.perfect_steps;
        for (total, visits) in self.visits[group].iter_mut().zip(&stats.visits) {
            *total += visits;
        }
        for (total, energy) in self.visit_energy.iter_mut().zip(&stats.visit_energy) {
            *total += energy;
        }
        self.perfect_energy += stats.perfect_energy;
        self.steps += stats.steps;
        self.capped += stats.capped as u64;
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

pub trait Engine {
    fn name(&self) -> String;

    /// Most particles the engine can hold.
    fn capacity(&self) -> usize;

    /// The Phase 1 population this engine handles well: FAR learning needs
    /// a fixed total of cycles, so more particles mean fewer cycles each.
    fn learn_particles(&self) -> usize {
        4096
    }

    /// Replace the population by the perfect matchings in `perms` (rows of
    /// n); `first_id` is the global id of the first one.
    fn load(&mut self, perms: &[u32], first_id: u64) -> anyhow::Result<()>;

    /// Replace the population by `count` perfect matchings drawn from
    /// nothing with the annealing sampler of `sampler.rs`.
    #[allow(clippy::too_many_arguments)]
    fn anneal(
        &mut self,
        tables: &Tables,
        stages: &StageSchedule,
        stage_time: f64,
        max_trials: u64,
        count: usize,
        first_id: u64,
        stream: Stream,
    ) -> anyhow::Result<SampleCost>;

    /// Apply the mutation kernel K_t of `sampler.rs` to every particle.
    fn mutate(&mut self, tables: &Tables, perfect_time: f64, stream: Stream) -> anyhow::Result<MutateCost>;

    /// Energy of every particle.
    fn energies(&mut self) -> anyhow::Result<Vec<f64>>;

    /// Regularization log-gap of every particle.
    fn log_gaps(&mut self) -> anyhow::Result<Vec<f64>>;

    /// Particle p becomes a copy of the former particle `ancestors[p]`.
    fn resample(&mut self, ancestors: &[u32]) -> anyhow::Result<()>;

    /// Run every particle through `cycles` FAR cycles (`far.rs`), leaving
    /// it on the perfect matching its chain stops on. Particle p reports
    /// into group p % groups.
    fn far(
        &mut self,
        tables: &Tables,
        cycles: u64,
        max_steps: u64,
        groups: usize,
        stream: Stream,
    ) -> anyhow::Result<FarRound>;
}
