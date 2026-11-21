//! The reductions the estimator needs across processes.
//!
//! Every rank runs the same driver on its own share of the population (its
//! own GPU). Nothing a particle does depends on another rank; ranks only
//! combine sums (learning statistics) and concatenate small vectors (pilot
//! energies, one number per island). All ranks see the same combined
//! values, so they take the same decisions without a coordinator.

pub trait Comm {
    fn rank(&self) -> usize;
    fn size(&self) -> usize;
    /// This rank's index among the ranks on its node.
    fn node_rank(&self) -> usize;
    /// Replace each entry by its sum over all ranks.
    fn sum_f64(&self, values: &mut [f64]);
    fn sum_u64(&self, values: &mut [u64]);
    /// Every rank's values concatenated in rank order.
    fn gather_f64(&self, values: &[f64]) -> Vec<f64>;
    /// End the run on every rank. A rank that fails must not just return:
    /// the others would wait for it in their next reduction.
    fn abort(&self) -> !;
}

/// A single process.
pub struct Solo;

impl Comm for Solo {
    fn rank(&self) -> usize {
        0
    }
    fn size(&self) -> usize {
        1
    }
    fn node_rank(&self) -> usize {
        0
    }
    fn sum_f64(&self, _values: &mut [f64]) {}
    fn sum_u64(&self, _values: &mut [u64]) {}
    fn gather_f64(&self, values: &[f64]) -> Vec<f64> {
        values.to_vec()
    }
    fn abort(&self) -> ! {
        std::process::exit(1)
    }
}

#[cfg(feature = "mpi")]
pub use with_mpi::Mpi;

#[cfg(feature = "mpi")]
mod with_mpi {
    use super::Comm;
    use mpi::collective::SystemOperation;
    use mpi::datatype::PartitionMut;
    use mpi::environment::Universe;
    use mpi::topology::SimpleCommunicator;
    use mpi::traits::*;

    /// MPI_COMM_WORLD. Dropping it finalizes MPI.
    pub struct Mpi {
        world: SimpleCommunicator,
        node_rank: usize,
        _universe: Universe,
    }

    impl Mpi {
        /// None if MPI was already initialized.
        pub fn new() -> Option<Self> {
            let universe = mpi::initialize()?;
            let world = universe.world();
            // the ranks that share this node's memory, in rank order
            let node_rank = world.split_shared(world.rank()).rank() as usize;
            Some(Mpi { world, node_rank, _universe: universe })
        }
    }

    impl Comm for Mpi {
        fn rank(&self) -> usize {
            self.world.rank() as usize
        }
        fn size(&self) -> usize {
            self.world.size() as usize
        }
        fn node_rank(&self) -> usize {
            self.node_rank
        }
        fn sum_f64(&self, values: &mut [f64]) {
            let local = values.to_vec();
            self.world.all_reduce_into(&local[..], values, SystemOperation::sum());
        }
        fn sum_u64(&self, values: &mut [u64]) {
            let local = values.to_vec();
            self.world.all_reduce_into(&local[..], values, SystemOperation::sum());
        }
        fn gather_f64(&self, values: &[f64]) -> Vec<f64> {
            let mut counts = vec![0i32; self.size()];
            self.world.all_gather_into(&(values.len() as i32), &mut counts[..]);
            let displacements: Vec<i32> = counts
                .iter()
                .scan(0, |offset, &count| {
                    let start = *offset;
                    *offset += count;
                    Some(start)
                })
                .collect();
            let mut all = vec![0.0f64; counts.iter().sum::<i32>() as usize];
            let mut partition = PartitionMut::new(&mut all[..], counts, &displacements[..]);
            self.world.all_gather_varcount_into(values, &mut partition);
            all
        }
        fn abort(&self) -> ! {
            self.world.abort(1)
        }
    }
}
