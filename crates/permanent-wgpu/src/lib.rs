//! The population on a GPU through wgpu. (under construction)

pub mod kernels;

use anyhow::{Result, ensure};
use cubecl::prelude::*;
use cubecl::server::Handle;
use permanent_core::chain::{Energy, EnergySum};
use permanent_core::engine::{FarRound, Stream};
use permanent_core::scaling::Scaled;
use permanent_core::stats::stream_seed;
use permanent_core::tables::Tables;

use kernels::{CAPPED, CYCLES, PERFECT_ENERGY, STATE, STEPS};

/// Threads of a workgroup.
const GROUP: u32 = 64;

pub struct Wgpu<'a> {
    scaled: &'a Scaled,
    client: Client,
    /// The energies in the chains' fixed point.
    energy_bits: Vec<i64>,
    first_id: u64,
    count: usize,
    /// The particles: `count` rows of n words.
    sigma: Handle,
}

/// A buffer of `len` words as a kernel argument.
fn words(handle: &Handle, len: usize) -> BufferArg {
    // SAFETY: every handle here is allocated with the length it is used at
    unsafe { BufferArg::from_raw_parts(handle.clone(), len) }
}

impl<'a> Wgpu<'a> {
    pub fn new(scaled: &'a Scaled, client: Client) -> Self {
        let sigma = client.empty(4);
        Wgpu { scaled, client, energy_bits: scaled.fixed_energy(), first_id: 0, count: 0, sigma }
    }

    pub fn load(&mut self, perms: &[u32], first_id: u64) {
        self.count = perms.len() / self.scaled.n;
        self.first_id = first_id;
        self.sigma = self.client.create_from_slice(u32::as_bytes(perms));
    }

    pub fn particles(&self) -> Vec<u32> {
        u32::from_bytes(&self.client.read_one_unchecked(self.sigma.clone())).to_vec()
    }

    fn grid(&self) -> CubeCount {
        CubeCount::Static((self.count as u32).div_ceil(GROUP), 1, 1)
    }

    pub fn far(
        &mut self,
        tables: &Tables<u64>,
        cycles: u64,
        max_steps: u64,
        groups: usize,
        stream: Stream,
    ) -> Result<FarRound> {
        let n = self.scaled.n;
        let count = self.count;
        ensure!(cycles < u32::MAX as u64, "too many cycles for one call");
        let max_steps = max_steps.min(u32::MAX as u64 - 1) as u32;
        // every table entry is 64 bits: two words, low first
        let mut entries: Vec<u64> = Vec::with_capacity(5 * n * n);
        entries.extend_from_slice(&tables.packed_rows);
        entries.extend_from_slice(&tables.packed_cols);
        entries.extend_from_slice(&tables.far_moves);
        entries.extend(self.energy_bits.iter().map(|&bits| bits as u64));
        let device_tables = self.client.create_from_slice(u64::as_bytes(&entries));
        let mut state = vec![0u32; count * STATE];
        for (particle, state) in state.chunks_mut(STATE).enumerate() {
            let seed = stream_seed(stream.seed, stream.phase, stream.level, self.first_id + particle as u64);
            state[0] = seed as u32;
            state[1] = (seed >> 32) as u32;
        }
        let state = self.client.create_from_slice(u32::as_bytes(&state));
        let scratch = self.client.empty(count * 2 * n * 4);
        let tally_len = 1 + 2 * n + groups * n;
        let budget = (64 * cycles + 32).min(max_steps as u64) as u32;
        let mut round = FarRound::new(n, groups);
        loop {
            let tally = self.client.create_from_slice(u32::as_bytes(&vec![0u32; tally_len]));
            // SAFETY: the buffers have the lengths the kernel indexes them
            // with: n x n tables, `count` particles, `groups` rows of visits.
            unsafe {
                kernels::far::launch_unchecked(
                    &self.client,
                    self.grid(),
                    CubeDim::new_1d(GROUP),
                    words(&device_tables, 10 * n * n),
                    words(&self.sigma, count * n),
                    words(&scratch, count * 2 * n),
                    words(&state, count * STATE),
                    words(&tally, tally_len),
                    n as u32,
                    count as u32,
                    tables.anchor as u32,
                    groups as u32,
                    cycles as u32,
                    budget,
                    max_steps,
                );
            }
            let bytes = self.client.read_one_unchecked(tally);
            let tally = u32::from_bytes(&bytes);
            for v in 0..n {
                let bits = (tally[2 + 2 * v] as u64) << 32 | tally[1 + 2 * v] as u64;
                round.visit_energy[v] += EnergySum::from(Energy::from_bits(bits as i64));
            }
            for (group, visits) in tally[1 + 2 * n..].chunks(n).enumerate() {
                for (total, &visits) in round.visits[group].iter_mut().zip(visits) {
                    *total += visits as u64;
                }
            }
            if tally[0] == 0 {
                break;
            }
        }
        let bytes = self.client.read_one_unchecked(state);
        for (particle, state) in u32::from_bytes(&bytes).chunks(STATE).enumerate() {
            round.cycles[particle % groups] += state[CYCLES] as u64;
            round.steps += state[STEPS] as u64;
            let bits = (state[PERFECT_ENERGY + 1] as u64) << 32 | state[PERFECT_ENERGY] as u64;
            round.perfect_energy += EnergySum::from(Energy::from_bits(bits as i64));
            round.capped += (state[kernels::FLAGS] & CAPPED != 0) as u64;
        }
        Ok(round)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use cubecl::Device;
    use cubecl::device::WgpuDeviceKind;
    use permanent_core::chain::stream::Stream as Philox;
    use permanent_core::engine::Engine;
    use permanent_core::problem::Problem;
    use permanent_core::scaling::{Prepared, prepare};
    use permanent_rayon::Rayon;

    /// The FAR kernel follows the same integer arithmetic and the same
    /// streams as the generic chain, so it must return the same round and
    /// leave the same population as the rayon engine on those streams.
    #[test]
    fn far_matches_the_rayon_engine_exactly() {
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
        let client = Device::wgpu(WgpuDeviceKind::default()).unwrap().client();
        let mut wgpu = Wgpu::new(&scaled, client);
        wgpu.load(&identity, 100);
        assert_eq!(wgpu.far(&tables, 50, 2000, 16, stream).unwrap(), expected);
        let expected: Vec<f64> = rayon.energies().unwrap();
        let energies: Vec<f64> = wgpu.particles().chunks(n).map(|row| scaled.energy_of(row)).collect();
        assert_eq!(energies, expected);
    }
}
