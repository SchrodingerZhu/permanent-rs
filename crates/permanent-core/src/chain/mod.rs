//! The per-particle chains, written once for every engine that runs Rust:
//! the rayon engine calls them on host memory and the CUDA engine compiles
//! them for the device.
//!
//! Every probability decision is a comparison of random bits with an
//! integer threshold from the tables, and FAR energies are fixed point, so
//! a chain's path depends only on its seed. Waiting times are the one place
//! where a floating-point type appears.

use num_traits::{AsPrimitive, PrimInt};
use rand_core::Rng;

pub mod dynamics;
pub mod far;
pub mod lane;
pub mod matching;
pub mod stream;

/// Energies in the chains: 2^-24 resolution, so sums do not depend on order.
pub type Energy = fixed::types::I40F24;
/// Totals of [`Energy`] over a population.
pub type EnergySum = fixed::types::I104F24;

/// A uniform integer below `bound` from 32 random bits.
#[inline(always)]
pub fn below(bits: u32, bound: u32) -> u32 {
    ((bits as u64 * bound as u64) >> 32) as u32
}

/// High 64 bits of the 64 x 64 product, by 32-bit limbs.
#[inline(always)]
pub fn mul_high(a: u64, b: u64) -> u64 {
    let (a0, a1) = (a & 0xFFFF_FFFF, a >> 32);
    let (b0, b1) = (b & 0xFFFF_FFFF, b >> 32);
    let p01 = a0 * b1;
    let p10 = a1 * b0;
    let middle = ((a0 * b0) >> 32) + (p01 & 0xFFFF_FFFF) + (p10 & 0xFFFF_FFFF);
    a1 * b1 + (p01 >> 32) + (p10 >> 32) + (middle >> 32)
}

/// High 64 bits of the product of `a` 2^32 and `b`.
#[inline(always)]
pub fn mul_high_word(a: u32, b: u64) -> u64 {
    let a = a as u64;
    a * (b >> 32) + ((a * (b & 0xFFFF_FFFF)) >> 32)
}

/// Whether `mul_high(u, scale) < bound` for a uniform 64-bit u with high
/// word `high`. The low word moves the product by at most
/// `scale >> 32` + 1, so it is drawn only when `high` leaves the comparison
/// open.
#[inline(always)]
pub fn scaled_below<R: Rng>(high: u32, scale: u64, bound: u64, rng: &mut R) -> bool {
    let least = mul_high_word(high, scale);
    if least + (scale >> 32) + 1 < bound {
        true
    } else if least >= bound {
        false
    } else {
        mul_high((high as u64) << 32 | rng.next_u32() as u64, scale) < bound
    }
}

/// Whether a uniform `C` is at least `threshold`, read from its high word
/// down: a word is drawn only while the ones before it tie with the
/// threshold's.
#[inline(always)]
pub fn at_least<C, R>(threshold: C, rng: &mut R) -> bool
where
    C: PrimInt + AsPrimitive<u32>,
    R: Rng,
{
    let mut shift = 8 * core::mem::size_of::<C>();
    loop {
        shift -= 32;
        let word: u32 = (threshold >> shift).as_();
        let drawn = rng.next_u32();
        if drawn != word || shift == 0 {
            return drawn >= word;
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use rand_core::{Infallible, TryRng};

    /// Hands out the words it was given.
    struct Words<'a>(&'a [u32]);

    impl TryRng for Words<'_> {
        type Error = Infallible;

        fn try_next_u32(&mut self) -> Result<u32, Infallible> {
            let (&word, rest) = self.0.split_first().unwrap();
            self.0 = rest;
            Ok(word)
        }

        fn try_next_u64(&mut self) -> Result<u64, Infallible> {
            let Ok(high) = self.try_next_u32();
            let Ok(low) = self.try_next_u32();
            Ok((high as u64) << 32 | low as u64)
        }

        fn try_fill_bytes(&mut self, _: &mut [u8]) -> Result<(), Infallible> {
            unimplemented!()
        }
    }

    #[test]
    fn decisions_from_the_high_word_agree_with_the_whole_draw() {
        let scale = 3 * (1 << 40) + 12_345;
        // around the bound the high word does not decide
        let bound = mul_high(0x8000_0000_8000_0000, scale);
        let threshold = 0x8000_0000_8000_0000u64;
        for high in [0, 0x7FFF_FFFF, 0x8000_0000, 0x8000_0001, u32::MAX] {
            for low in [0, 0x7FFF_FFFF, 0x8000_0000, u32::MAX] {
                let u = (high as u64) << 32 | low as u64;
                assert_eq!(scaled_below(high, scale, bound, &mut Words(&[low])), mul_high(u, scale) < bound);
                assert_eq!(at_least(threshold, &mut Words(&[high, low])), u >= threshold);
            }
            assert_eq!(at_least(0x8000_0000u32, &mut Words(&[high])), high >= 0x8000_0000);
        }
    }
}
