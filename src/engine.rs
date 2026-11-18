//! Where the population lives and moves: the CPU thread pool here, or a GPU.
//!
//! An engine holds perfect matchings (the particles) and applies whole-
//! population operations to them. The particles stay resident in the engine
//! from the first temperature to the last; the estimator only sees their
//! energies and decides who survives a resampling.
//!
//! Particle p of a call draws from the stream (seed, phase, level,
//! first id + p), so results do not depend on threads, batches or ranks.

use rand::SeedableRng;
use rand::rngs::SmallRng;
use rayon::prelude::*;

use crate::far;
use crate::sampler::{MutateCost, SampleCost, Scratch, StageSchedule, mutate, mutate_event_cap, sample_perfect};
use crate::scaling::Scaled;
use crate::stats::stream_seed;
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
    fn new(n: usize, groups: usize) -> Self {
        FarRound {
            cycles: vec![0; groups],
            visits: vec![vec![0; n]; groups],
            visit_energy: vec![0.0; n],
            perfect_energy: 0.0,
            steps: 0,
            capped: 0,
        }
    }

    fn merge(mut self, other: FarRound) -> FarRound {
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

/// The population on the rayon pool, one particle per task.
pub struct Cpu<'a> {
    scaled: &'a Scaled,
    sigma: Vec<u32>,
    first_id: u64,
}

impl<'a> Cpu<'a> {
    pub fn new(scaled: &'a Scaled) -> Self {
        Cpu { scaled, sigma: Vec::new(), first_id: 0 }
    }

    /// The particles, as rows of n.
    pub fn particles(&self) -> &[u32] {
        &self.sigma
    }

    fn rng(&self, stream: Stream, particle: usize) -> SmallRng {
        SmallRng::seed_from_u64(stream_seed(stream.seed, stream.phase, stream.level, self.first_id + particle as u64))
    }
}

impl Engine for Cpu<'_> {
    fn name(&self) -> String {
        format!("cpu ({} threads)", rayon::current_num_threads())
    }

    fn capacity(&self) -> usize {
        // bounded by host memory only; 2^31 keeps particle indices in u32
        1 << 31
    }

    fn load(&mut self, perms: &[u32], first_id: u64) -> anyhow::Result<()> {
        self.sigma = perms.to_vec();
        self.first_id = first_id;
        Ok(())
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
    ) -> anyhow::Result<SampleCost> {
        let n = self.scaled.n;
        let cap = stages.event_cap(n, stage_time);
        self.first_id = first_id;
        let this = &*self;
        let mut sigma = vec![0u32; count * n];
        let (cost, failed) = sigma
            .par_chunks_mut(n)
            .enumerate()
            .map_init(
                || Scratch::new(n),
                |scratch, (particle, row)| {
                    let mut rng = this.rng(stream, particle);
                    let mut cost = SampleCost::default();
                    let accepted =
                        sample_perfect(tables, stages, stage_time, cap, max_trials, scratch, &mut rng, &mut cost);
                    row.copy_from_slice(&scratch.sigma);
                    (cost, !accepted as u64)
                },
            )
            .reduce(
                || (SampleCost::default(), 0),
                |mut a, b| {
                    a.0.add(b.0);
                    (a.0, a.1 + b.1)
                },
            );
        anyhow::ensure!(
            failed == 0,
            "the sampler could not produce {failed} of {count} perfect matchings; the anchor weights are out of range"
        );
        self.sigma = sigma;
        Ok(cost)
    }

    fn mutate(&mut self, tables: &Tables, perfect_time: f64, stream: Stream) -> anyhow::Result<MutateCost> {
        let n = self.scaled.n;
        let cap = mutate_event_cap(n, perfect_time);
        let this = &*self;
        let costs: Vec<MutateCost> = this
            .sigma
            .par_chunks(n)
            .enumerate()
            .map_init(
                || Scratch::new(n),
                |scratch, (particle, row)| {
                    let mut rng = this.rng(stream, particle);
                    scratch.load(row);
                    let cost = mutate(tables, perfect_time, cap, scratch, &mut rng);
                    (cost, (cost.over_budget == 0).then(|| scratch.sigma.clone()))
                },
            )
            .collect::<Vec<_>>()
            .into_iter()
            .enumerate()
            .map(|(particle, (cost, moved))| {
                if let Some(moved) = moved {
                    self.sigma[particle * n..(particle + 1) * n].copy_from_slice(&moved);
                }
                cost
            })
            .collect();
        let mut total = MutateCost::default();
        for cost in costs {
            total.add(cost);
        }
        Ok(total)
    }

    fn energies(&mut self) -> anyhow::Result<Vec<f64>> {
        let scaled = self.scaled;
        Ok(self.sigma.par_chunks(scaled.n).map(|row| scaled.energy_of(row)).collect())
    }

    fn log_gaps(&mut self) -> anyhow::Result<Vec<f64>> {
        let scaled = self.scaled;
        Ok(self.sigma.par_chunks(scaled.n).map(|row| scaled.log_gap_of(row)).collect())
    }

    fn resample(&mut self, ancestors: &[u32]) -> anyhow::Result<()> {
        let n = self.scaled.n;
        let mut next = vec![0u32; ancestors.len() * n];
        next.par_chunks_mut(n).zip(ancestors).for_each(|(row, &ancestor)| {
            let ancestor = ancestor as usize;
            row.copy_from_slice(&self.sigma[ancestor * n..(ancestor + 1) * n]);
        });
        self.sigma = next;
        Ok(())
    }

    fn far(
        &mut self,
        tables: &Tables,
        cycles: u64,
        max_steps: u64,
        groups: usize,
        stream: Stream,
    ) -> anyhow::Result<FarRound> {
        let scaled = self.scaled;
        let n = scaled.n;
        let first_id = self.first_id;
        let round = self
            .sigma
            .par_chunks_mut(n)
            .enumerate()
            .fold(
                || FarRound::new(n, groups),
                |mut local, (particle, row)| {
                    let seed = stream_seed(stream.seed, stream.phase, stream.level, first_id + particle as u64);
                    let mut rng = SmallRng::seed_from_u64(seed);
                    let stats = far::run(tables, &scaled.energy, row, cycles, max_steps, &mut rng);
                    let group = particle % groups;
                    local.cycles[group] += stats.perfect_steps;
                    for v in 0..n {
                        local.visits[group][v] += stats.visits[v];
                        local.visit_energy[v] += stats.visit_energy[v];
                    }
                    local.perfect_energy += stats.perfect_energy;
                    local.steps += stats.steps;
                    local.capped += stats.capped as u64;
                    local
                },
            )
            .reduce(|| FarRound::new(n, groups), FarRound::merge);
        Ok(round)
    }
}
