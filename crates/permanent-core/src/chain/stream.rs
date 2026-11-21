//! A particle's random stream: the blocks of Philox4x32-10 (Salmon et al.,
//! "Parallel Random Numbers: As Easy as 1, 2, 3", SC'11) under the key
//! `seed`, in counter order. Streams of different seeds are disjoint by
//! construction, and a stream's position is two integers, so a chain
//! stopped on one device can be continued on another.

use rand_core::{Infallible, SeedableRng, TryRng, utils};
use rand_philox::philox4x32_10;

#[derive(Clone, Copy)]
pub struct Stream {
    key: [u32; 2],
    /// Blocks generated so far.
    block: u64,
    lanes: [u32; 4],
    /// Outputs of the last block not yet handed out.
    left: u32,
}

impl Stream {
    #[inline(always)]
    pub fn new(seed: u64) -> Self {
        Stream { key: [seed as u32, (seed >> 32) as u32], block: 0, lanes: [0; 4], left: 0 }
    }

    /// Blocks generated, and outputs of the last one not yet handed out.
    #[inline(always)]
    pub fn position(&self) -> (u64, u32) {
        (self.block, self.left)
    }

    /// The stream of `seed` at a position another copy reported.
    pub fn at(seed: u64, block: u64, left: u32) -> Self {
        let mut stream = Stream::new(seed);
        if left > 0 {
            stream.block = block - 1;
            stream.refill();
            stream.left = left;
        } else {
            stream.block = block;
        }
        stream
    }

    #[inline(always)]
    fn refill(&mut self) {
        self.lanes = philox4x32_10([self.block as u32, (self.block >> 32) as u32, 0, 0], self.key);
        self.block += 1;
        self.left = 4;
    }
}

impl TryRng for Stream {
    type Error = Infallible;

    #[inline(always)]
    fn try_next_u32(&mut self) -> Result<u32, Infallible> {
        if self.left == 0 {
            self.refill();
        }
        self.left -= 1;
        Ok(self.lanes[3 - self.left as usize])
    }

    #[inline(always)]
    fn try_next_u64(&mut self) -> Result<u64, Infallible> {
        let Ok(high) = self.try_next_u32();
        let Ok(low) = self.try_next_u32();
        Ok((high as u64) << 32 | low as u64)
    }

    fn try_fill_bytes(&mut self, dst: &mut [u8]) -> Result<(), Infallible> {
        utils::fill_bytes_via_next_word(dst, || self.try_next_u32())
    }
}

impl SeedableRng for Stream {
    type Seed = [u8; 8];

    fn from_seed(seed: [u8; 8]) -> Self {
        Stream::new(u64::from_le_bytes(seed))
    }

    /// The stream keyed by `seed` itself, as [`Stream::new`].
    fn seed_from_u64(seed: u64) -> Self {
        Stream::new(seed)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use rand_core::Rng;

    #[test]
    fn hands_out_each_block_once() {
        let mut stream = Stream::new(0);
        let first: Vec<u32> = (0..4).map(|_| stream.next_u32()).collect();
        // the first known answer of the Random123 distribution
        assert_eq!(first, [0x6627_e8d5, 0xe169_c58d, 0xbc57_ac4c, 0x9b00_dbd8]);
        assert_eq!(stream.next_u32(), philox4x32_10([1, 0, 0, 0], [0, 0])[0]);
    }

    #[test]
    fn continues_from_a_reported_position() {
        let seed = 0x1234_5678_9abc_def0;
        for skip in 0..9 {
            let mut stream = Stream::new(seed);
            for _ in 0..skip {
                stream.next_u32();
            }
            let (block, left) = stream.position();
            let mut copy = Stream::at(seed, block, left);
            for _ in 0..9 {
                assert_eq!(copy.next_u64(), stream.next_u64());
            }
        }
    }
}
