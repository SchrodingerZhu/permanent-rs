//! The population on the rayon thread pool.

use permanent_core::engine::{Engine, FarRound, Stream};
use permanent_core::far;
use permanent_core::sampler::{
    MutateCost, SampleCost, Scratch, StageSchedule, mutate, mutate_event_cap, sample_perfect,
};
use permanent_core::scaling::Scaled;
use permanent_core::stats::stream_seed;
use permanent_core::tables::Tables;
use rand::SeedableRng;
use rand::rngs::SmallRng;
use rayon::prelude::*;

/// The population on the rayon pool, one particle per task.
pub struct Rayon<'a> {
    scaled: &'a Scaled,
    sigma: Vec<u32>,
    first_id: u64,
}

impl<'a> Rayon<'a> {
    pub fn new(scaled: &'a Scaled) -> Self {
        Rayon { scaled, sigma: Vec::new(), first_id: 0 }
    }

    /// The particles, as rows of n.
    pub fn particles(&self) -> &[u32] {
        &self.sigma
    }

    fn rng(&self, stream: Stream, particle: usize) -> SmallRng {
        SmallRng::seed_from_u64(stream_seed(stream.seed, stream.phase, stream.level, self.first_id + particle as u64))
    }
}

impl Engine for Rayon<'_> {
    fn name(&self) -> String {
        format!("rayon ({} threads)", rayon::current_num_threads())
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
