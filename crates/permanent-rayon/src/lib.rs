//! The population on the rayon thread pool.

use std::marker::PhantomData;

use num_traits::{AsPrimitive, Float};
use permanent_core::chain::dynamics::{self, Moved, STAGE_TICKS};
use permanent_core::chain::far;
use permanent_core::chain::lane::{Lane, NONE};
use permanent_core::chain::matching::{Matching, words};
use permanent_core::engine::{Engine, FarRound, GroupTally, Stream};
use permanent_core::sampler::{MutateCost, SampleCost, StageSchedule, mutate_event_cap, penalty};
use permanent_core::scaling::Scaled;
use permanent_core::stats::{mix, stream_seed};
use permanent_core::tables::{Tables, fraction};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

/// Fewest FAR chains a task runs. Every task tallies into a round of its
/// own, which costs about as much to set up and merge as a few short chains
/// cost to run, so the pool must not split a round down to single chains.
const FAR_CHAINS_PER_TASK: usize = 32;

/// The population on the rayon pool, one particle per task. `T` is the
/// type of the chains' waiting times and `G` the generator of a particle's
/// stream.
///
/// `SmallRng` makes this engine more than twice as fast as the GPU's
/// counter-based generator would. With `permanent_core::chain::stream::Stream`
/// for `G` the chains follow the same streams as on the GPU.
pub struct Rayon<'a, T = f64, G = SmallRng> {
    scaled: &'a Scaled,
    /// The energies in the chains' fixed point.
    energy: Vec<i64>,
    /// The particles, as rows of n.
    sigma: Vec<u16>,
    first_id: u64,
    types: PhantomData<(T, G)>,
}

impl<'a, T, G> Rayon<'a, T, G> {
    pub fn new(scaled: &'a Scaled) -> Self {
        Rayon { scaled, energy: scaled.fixed_energy(), sigma: Vec::new(), first_id: 0, types: PhantomData }
    }

    fn seed(&self, stream: Stream, particle: usize) -> u64 {
        stream_seed(stream.seed, stream.phase, stream.level, self.first_id + particle as u64)
    }
}

impl<T, G> Engine for Rayon<'_, T, G>
where
    T: Float + AsPrimitive<u64> + Send + Sync,
    G: Rng + SeedableRng + Send + Sync,
{
    type Data = T;
    type Counter = u64;

    fn fork(&self) -> anyhow::Result<Self> {
        Ok(Rayon::new(self.scaled))
    }

    fn name(&self) -> String {
        format!("rayon ({} threads, {})", rayon::current_num_threads(), std::any::type_name::<T>())
    }

    fn capacity(&self) -> usize {
        // bounded by host memory only; 2^31 keeps particle indices in u32
        1 << 31
    }

    fn load(&mut self, perms: &[u32], first_id: u64) -> anyhow::Result<()> {
        self.sigma = perms.iter().map(|&v| v as u16).collect();
        self.first_id = first_id;
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
    ) -> anyhow::Result<SampleCost> {
        let n = self.scaled.n;
        let cap = stages.event_cap(n, stage_time);
        let unassign = stages.unassign::<u64>();
        let ticks_per_time = T::from(STAGE_TICKS as f64 / stage_time).unwrap();
        self.first_id = first_id;
        let this = &*self;
        let mut sigma = vec![0u16; count * n];
        let (cost, failed) = sigma
            .par_chunks_mut(n)
            .enumerate()
            .map_init(
                || vec![0u16; words(n)],
                |memory, (particle, row)| {
                    let mut matching = Matching::new(Lane::on(memory, n));
                    let seed = this.seed(stream, particle);
                    let mut cost = SampleCost::default();
                    // a trial that misses is run again on a stream of its own
                    for trial in 0..max_trials {
                        let mut rng = G::seed_from_u64(mix(seed.wrapping_add(trial)));
                        // SAFETY: the lane holds words(n) words and the tables are n x n
                        let outcome = unsafe {
                            dynamics::trial::<u64, T, _>(
                                &mut matching,
                                n as u32,
                                &tables.packed_rows,
                                &unassign,
                                ticks_per_time,
                                stages.fast(),
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
                            return (cost, 0);
                        }
                        cost.over_budget += outcome.over_budget as u64;
                    }
                    (cost, 1)
                },
            )
            .reduce(
                || (SampleCost::default(), 0u64),
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

    fn mutate(&mut self, tables: &Tables<u64>, perfect_time: f64, stream: Stream) -> anyhow::Result<MutateCost> {
        let n = self.scaled.n;
        let cap = mutate_event_cap(n, perfect_time);
        let penalty = penalty(n);
        let unassign_below: u64 = fraction(1.0 / (1.0 + penalty));
        let first_id = self.first_id;
        let moves: Vec<Moved<T>> = self
            .sigma
            .par_chunks_mut(n)
            .enumerate()
            .map_init(
                || vec![0u16; words(n)],
                |memory, (particle, row)| {
                    let mut matching = Matching::new(Lane::on(memory, n));
                    let seed = stream_seed(stream.seed, stream.phase, stream.level, first_id + particle as u64);
                    let mut rng = G::seed_from_u64(seed);
                    // SAFETY: `row` is a perfect matching on n rows, the lane
                    // holds words(n) words and the tables are n x n
                    unsafe {
                        matching.load(row.as_ptr());
                        let moved = dynamics::mutate::<u64, T, _>(
                            &mut matching,
                            n as u32,
                            &tables.packed_rows,
                            T::from(perfect_time).unwrap(),
                            1 + penalty as u64,
                            unassign_below,
                            cap,
                            &mut rng,
                        );
                        // a move that ran out of events leaves its particle where it was
                        if !moved.over_budget {
                            matching.store(row.as_mut_ptr());
                        }
                        moved
                    }
                },
            )
            .collect();
        let mut cost = MutateCost::default();
        for moved in moves {
            cost.events += moved.events;
            cost.time += moved.time.to_f64().unwrap();
            cost.over_budget += moved.over_budget as u64;
        }
        Ok(cost)
    }

    fn energies(&mut self) -> anyhow::Result<Vec<f64>> {
        let (n, energy) = (self.scaled.n, &self.scaled.energy);
        Ok(self
            .sigma
            .par_chunks(n)
            .map(|row| row.iter().enumerate().map(|(u, &v)| energy[u * n + v as usize]).sum())
            .collect())
    }

    fn log_gaps(&mut self) -> anyhow::Result<Vec<f64>> {
        let (n, log_gap) = (self.scaled.n, &self.scaled.log_gap);
        Ok(self
            .sigma
            .par_chunks(n)
            .map(|row| row.iter().enumerate().map(|(u, &v)| log_gap[u * n + v as usize]).sum())
            .collect())
    }

    fn resample(&mut self, ancestors: &[u32]) -> anyhow::Result<()> {
        let n = self.scaled.n;
        let mut next = vec![0u16; ancestors.len() * n];
        next.par_chunks_mut(n).zip(ancestors).for_each(|(row, &ancestor)| {
            let ancestor = ancestor as usize;
            row.copy_from_slice(&self.sigma[ancestor * n..(ancestor + 1) * n]);
        });
        self.sigma = next;
        Ok(())
    }

    fn far(
        &mut self,
        tables: &Tables<u64>,
        cycles: u64,
        max_steps: u64,
        groups: usize,
        stream: Stream,
    ) -> anyhow::Result<FarRound> {
        let n = self.scaled.n;
        let first_id = self.first_id;
        let far_tables = tables.far(&self.energy);
        let anchor = tables.anchor as u32;
        let round = self
            .sigma
            .par_chunks_mut(n)
            .enumerate()
            .with_min_len(FAR_CHAINS_PER_TASK)
            .fold(
                // the round of this task and one chain's lane
                || (FarRound::new(n, groups), vec![0u16; 2 * n]),
                |(mut local, mut memory), (particle, row)| {
                    let lane = Lane::on(&mut memory, n);
                    let group = particle % groups;
                    let seed = stream_seed(stream.seed, stream.phase, stream.level, first_id + particle as u64);
                    let mut rng = G::seed_from_u64(seed);
                    let mut tally = GroupTally { round: &mut local, group };
                    // SAFETY: `row` holds a perfect matching on n rows, the
                    // lane two arrays of n, and the tables are n x n.
                    let chain = unsafe {
                        let mut chain = far::begin(&lane, row.as_ptr(), &far_tables);
                        far::run(&lane, &far_tables, anchor, cycles, max_steps, &mut rng, &mut chain, &mut tally);
                        chain
                    };
                    // a chain that ran out of steps leaves its particle where it was
                    let complete = chain.hole_u == NONE && chain.perfect_steps == cycles;
                    if complete {
                        // SAFETY: `row` holds n words
                        unsafe { far::store(&lane, row.as_mut_ptr()) };
                    }
                    local.close(group, &chain, 0, 0, !complete);
                    (local, memory)
                },
            )
            .map(|(round, _)| round)
            .reduce(|| FarRound::new(n, groups), FarRound::merge);
        Ok(round)
    }
}
