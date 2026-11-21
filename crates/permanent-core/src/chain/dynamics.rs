//! The relaxed-assignment dynamics of `sampler.rs`, as the two chains the
//! engines run: the move at the top activity and one annealing trial.
//!
//! Which clock rings, whether the row stays unmatched and which column it
//! takes are comparisons of random bits with integers: the ring rates are
//! exact integers (the penalty J = 32 n^2 is one) and the probabilities are
//! thresholds from the tables. Only the waiting times are floating point,
//! in the engine's real type `T`.

use num_traits::{AsPrimitive, Float, PrimInt, Unsigned};
use rand::RngExt;
use rand::distr::{Distribution, StandardUniform};
use rand_core::Rng;

use super::below;
use super::far::alias_draw;
use super::matching::Matching;

/// Stage time of the annealing sampler, in ticks of its integer clock.
pub const STAGE_TICKS: u64 = 1 << 40;

/// An exponential waiting time of rate 1 from 32 random bits.
#[inline(always)]
pub fn exponential<T: Float>(bits: u32) -> T {
    // (bits + 1/2) 2^-32 is uniform on (0, 1)
    -((T::from(bits).unwrap() + T::from(0.5).unwrap()) * T::from(2.328_306_436_538_696_3e-10).unwrap()).ln()
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

/// One ring: pick the row whose clock rang, unmatch it, and redraw it from
/// its row of B unless it stays unmatched. Rows in a collision ring at rate
/// 1 + J each (`conflicted_rate` in all), the others at rate 1.
///
/// # Safety
/// `alias` must be the packed row alias table of an n x n matrix.
#[inline(always)]
unsafe fn ring<C, R>(
    matching: &mut Matching,
    n: u32,
    alias: &[C],
    rate: u64,
    conflicted_rate: u64,
    unassign_below: C,
    rng: &mut R,
) where
    C: PrimInt + Unsigned + AsPrimitive<u32>,
    R: Rng,
    StandardUniform: Distribution<C>,
{
    let conflicted = mul_high(rng.next_u64(), rate) < conflicted_rate;
    unsafe {
        let mut u;
        if conflicted {
            u = matching.conflict(below(rng.next_u32(), matching.conflicted));
        } else {
            // uniform over the rows outside the list, by rejection
            loop {
                u = below(rng.next_u32(), n);
                if !matching.is_conflicted(u) {
                    break;
                }
            }
        }
        matching.remove(u);
        if rng.random::<C>() >= unassign_below {
            let v = alias_draw(alias, u, n, rng);
            matching.place(u, v);
        }
    }
}

/// What a move did.
#[derive(Clone, Copy)]
pub struct Moved<T> {
    pub events: u64,
    /// Elapsed chain time, excursions included.
    pub time: T,
    /// The event cap was hit or the collision list overflowed; the state is
    /// not a sample.
    pub over_budget: bool,
}

/// The mutation kernel K_t: run the dynamics at the top activity from the
/// perfect matching in `matching` until `perfect_time` has been spent in
/// perfect matchings. `fast` is 1 + J and `unassign_below` the probability
/// 1 / (1 + J) as a fraction of the counter's range.
///
/// # Safety
/// `matching` must be loaded, on n rows, and `alias` the packed row alias
/// table of an n x n matrix.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub unsafe fn mutate<C, T, R>(
    matching: &mut Matching,
    n: u32,
    alias: &[C],
    perfect_time: T,
    fast: u64,
    unassign_below: C,
    event_cap: u64,
    rng: &mut R,
) -> Moved<T>
where
    C: PrimInt + Unsigned + AsPrimitive<u32>,
    T: Float,
    R: Rng,
    StandardUniform: Distribution<C>,
{
    let mut remaining = perfect_time;
    let mut time = T::zero();
    let mut events = 0u64;
    let mut over_budget = false;
    loop {
        let conflicted_rate = matching.conflicted as u64 * fast;
        let rate = (n - matching.conflicted) as u64 + conflicted_rate;
        let dt = exponential::<T>(rng.next_u32()) / T::from(rate).unwrap();
        if matching.is_perfect() {
            if dt >= remaining {
                time = time + remaining;
                break;
            }
            remaining = remaining - dt;
        }
        time = time + dt;
        events += 1;
        if events > event_cap {
            over_budget = true;
            break;
        }
        unsafe { ring(matching, n, alias, rate, conflicted_rate, unassign_below, rng) };
        if matching.overflow {
            over_budget = true;
            break;
        }
    }
    Moved { events, time, over_budget }
}

/// What an annealing trial did; it produced a sample when the matching it
/// leaves is perfect.
#[derive(Clone, Copy)]
pub struct Trial {
    pub events: u64,
    pub over_budget: bool,
}

/// One trial of the annealing sampler: from the all-unmatched state through
/// every activity stage, each for `STAGE_TICKS`. `unassign[stage]` is the
/// probability 1 / (1 + a) of staying unmatched at that stage's activity,
/// and `ticks_per_time` is `STAGE_TICKS` over the stage time.
///
/// # Safety
/// `matching` must be on n rows, and `alias` the packed row alias table of
/// an n x n matrix.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub unsafe fn trial<C, T, R>(
    matching: &mut Matching,
    n: u32,
    alias: &[C],
    unassign: &[C],
    ticks_per_time: T,
    fast: u64,
    event_cap: u64,
    rng: &mut R,
) -> Trial
where
    C: PrimInt + Unsigned + AsPrimitive<u32>,
    T: Float + AsPrimitive<u64>,
    R: Rng,
    StandardUniform: Distribution<C>,
{
    unsafe { matching.clear() };
    // One flat loop over events and stage changes, so the threads of a
    // warp stay in the same code whatever stage each has reached.
    let stages = unassign.len();
    let mut stage = 0;
    let mut ticks = 0u64;
    let mut unassign_below = unsafe { *unassign.get_unchecked(0) };
    let mut events = 0u64;
    let mut over_budget = false;
    loop {
        let conflicted_rate = matching.conflicted as u64 * fast;
        let rate = (n - matching.conflicted) as u64 + conflicted_rate;
        ticks += (exponential::<T>(rng.next_u32()) * (ticks_per_time / T::from(rate).unwrap())).as_();
        if ticks > STAGE_TICKS {
            stage += 1;
            if stage == stages {
                break;
            }
            ticks = 0;
            unassign_below = unsafe { *unassign.get_unchecked(stage) };
            continue;
        }
        events += 1;
        if events > event_cap {
            over_budget = true;
            break;
        }
        unsafe { ring(matching, n, alias, rate, conflicted_rate, unassign_below, rng) };
        if matching.overflow {
            over_budget = true;
            break;
        }
    }
    Trial { events, over_budget }
}
