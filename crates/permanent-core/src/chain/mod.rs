//! The per-particle chains, written once for every engine that runs Rust:
//! the rayon engine calls them on host memory and the CUDA engine compiles
//! them for the device.
//!
//! Every probability decision is a comparison of random bits with an
//! integer threshold from the tables, and FAR energies are fixed point, so
//! a chain's path depends only on its seed. Waiting times are the one place
//! where a floating-point type appears.

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
