//! The FAR chain, used to learn the anchor hole weights (arXiv:2609.20717,
//! Section 4).
//!
//! States are perfect matchings of K_{n,n} and near-perfect matchings with
//! one hole on each side. The stationary distribution is
//!
//!   pi(M) ∝ n^2 B(M)           for perfect M,
//!   pi(N) ∝ W(u,v) B(N)        for N with holes (u, v),
//!
//! with W(u,v) = 1 + n B(u,v) + s_v (1 - B(u,v)) + (s_v - B(u,v)). Moves,
//! with no accept/reject step:
//!
//!   perfect:        delete each edge with probability 1/(2n);
//!   holes (u, v):   add (u,v)                          w.p. n B(u,v) / (2W);
//!                   slide the row hole to (u,z), z != v    w.p. B(u,z) s_v / (2W);
//!                   slide the column hole to (y,v), y != u w.p. B(y,v) / (2W).
//!
//! Learning without a mixing-time assumption: the chain watched only at its
//! perfect states is itself a stationary chain with law mu = pi restricted
//! to perfect matchings. Started from a mu sample and run until its
//! (k+1)-th perfect step, the expected number of visits to a class S during
//! those k cycles is exactly k pi(S)/pi(P) (Kac's formula), for every
//! k >= 1. With S = the hole class N(r, v) of the anchor row this yields
//!
//!   lambda(P)/lambda(N(r,v)) = w(r,v) R_r W(r,v) k / (n^2 E[visits_v]).
//!
//! Because the identity holds per cycle, W chains of k cycles are worth one
//! chain of W k: the learner runs every particle of the population for a
//! few cycles instead of a few chains for many. The matching a chain stops
//! on is again a mu sample, so a run also moves its particle.
//!
//! The move probabilities are precomputed as integer thresholds
//! (`Tables::far_moves`) and energies are fixed point, so a chain's path
//! and its sums depend only on its seed.

use num_traits::{AsPrimitive, PrimInt, Unsigned};
use rand::RngExt;
use rand::distr::{Distribution, StandardUniform};
use rand_core::Rng;

use super::lane::{COL, Lane, NONE, ROW};
use super::{Energy, below, mul_high};

/// The integer tables of one temperature (`Tables::far`).
pub struct FarTables<'t, C> {
    /// Packed alias tables of the rows and of the columns of B.
    pub row_alias: &'t [C],
    pub col_alias: &'t [C],
    /// With holes (u, v): add the edge below `moves[2 (u n + v)]`, slide
    /// the row hole below `moves[2 (u n + v) + 1]`, as fractions of the
    /// counter's range; sliding the column hole is as likely as the row's.
    pub moves: &'t [C],
    /// Entry energies, as bits of [`Energy`].
    pub energy: &'t [i64],
}

/// Where a chain stands. It is all a chain needs, besides its matching and
/// its stream, to be continued elsewhere.
#[derive(Clone, Copy)]
pub struct FarChain {
    /// The unmatched row, or `NONE` on a perfect matching.
    pub hole_u: u32,
    pub hole_v: u32,
    /// Perfect steps taken, each of which closes a cycle.
    pub perfect_steps: u64,
    pub steps: u64,
    /// Energy of the current matching.
    pub energy: Energy,
    /// Sum of energies over the perfect steps.
    pub perfect_energy: Energy,
}

/// Receives the visits to the anchor row's hole classes.
pub trait Tally {
    /// The chain is at N(anchor, v), on a matching of this energy.
    fn visit(&mut self, v: u32, energy: Energy);
}

/// Draw from line `line` of a packed alias table: the counter's bits above
/// the low 16 are a cell's keep threshold, the low 16 its alias.
///
/// One 64-bit number makes both choices. The high word of its product with
/// n is the cell, and the low word is uniform within the cell (it takes
/// every n-th value), so it is compared with the keep threshold. A cell's
/// probability is 1/n to within 2^-64. The number's low 32 bits are drawn
/// only when its high 32 do not settle both choices.
///
/// # Safety
/// `table` must hold n lines of n entries.
#[inline(always)]
pub unsafe fn alias_draw<C, R>(table: &[C], line: u32, n: u32, rng: &mut R) -> u32
where
    C: PrimInt + Unsigned + AsPrimitive<u32>,
    R: Rng,
    u64: AsPrimitive<C>,
{
    // A 64-bit draw u gives the cell (the high half of u n) and the place
    // within it (the low half). Its high word alone gives both unless the
    // low word, which adds less than n 2^32 to the product, could change
    // the cell or the comparison.
    let shift = 64 - 8 * core::mem::size_of::<C>();
    let high = rng.next_u32();
    let product = high as u64 * n as u64;
    let least = product << 32;
    let most = least.wrapping_add(((n as u64) << 32) - 1);
    let mut i = (product >> 32) as u32;
    let mut entry = unsafe { *table.get_unchecked(line as usize * n as usize + i as usize) };
    let within: C = (least >> shift).as_();
    let within_most: C = (most >> shift).as_();
    let mut keep = within >> 16 < entry >> 16;
    if most < least || keep != (within_most >> 16 < entry >> 16) {
        let bits = (high as u64) << 32 | rng.next_u32() as u64;
        i = mul_high(bits, n as u64) as u32;
        let within: C = (bits.wrapping_mul(n as u64) >> shift).as_();
        entry = unsafe { *table.get_unchecked(line as usize * n as usize + i as usize) };
        keep = within >> 16 < entry >> 16;
    }
    if keep { i } else { entry.as_() & 0xFFFF }
}

/// Bits of the energy of entry (u, v).
#[inline(always)]
unsafe fn energy_at(energy: &[i64], n: usize, u: u32, v: u32) -> i64 {
    unsafe { *energy.get_unchecked(u as usize * n + v as usize) }
}

/// Put the matching stored at `stored` (n words, `NONE` at an unmatched
/// row) into `lane` and start a chain on it. For a matching with holes the
/// caller sets `hole_u` and `hole_v` afterwards.
///
/// The inverse is kept for matched columns only. The chain never reads the
/// entry of its free column, so that entry is left as it was, here and in
/// [`run`].
///
/// # Safety
/// `stored` must be valid for n words and `lane` hold two arrays.
#[inline(always)]
pub unsafe fn begin<C>(lane: &Lane, stored: *const u16, tables: &FarTables<C>) -> FarChain {
    let n = lane.n as u32;
    let mut energy = 0i64;
    unsafe {
        let mut u = 0;
        while u < n {
            let v = *stored.add(u as usize) as u32;
            lane.set(ROW, u, v);
            if v != NONE {
                lane.set(COL, v, u);
                energy += *tables.energy.get_unchecked(u as usize * lane.n + v as usize);
            }
            u += 1;
        }
    }
    FarChain {
        hole_u: NONE,
        hole_v: 0,
        perfect_steps: 0,
        steps: 0,
        energy: Energy::from_bits(energy),
        perfect_energy: Energy::ZERO,
    }
}

/// Write the lane's matching to `stored`, with `NONE` at the row hole.
///
/// # Safety
/// As for [`begin`].
#[inline(always)]
pub unsafe fn store(lane: &Lane, stored: *mut u16) {
    let mut u = 0;
    while u < lane.n as u32 {
        unsafe { *stored.add(u as usize) = lane.get(ROW, u) as u16 };
        u += 1;
    }
}

/// Run `chain` until it has taken `cycles` perfect steps or `budget` steps
/// in all, whichever comes first. It is complete when it stops on a perfect
/// matching with `perfect_steps == cycles`.
///
/// # Safety
/// `lane` must hold the chain's matching as [`begin`] left it, and the
/// tables must be those of an n x n matrix.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub unsafe fn run<C, R, V>(
    lane: &Lane,
    tables: &FarTables<C>,
    anchor: u32,
    cycles: u64,
    budget: u64,
    rng: &mut R,
    chain: &mut FarChain,
    tally: &mut V,
) where
    C: PrimInt + Unsigned + AsPrimitive<u32>,
    R: Rng,
    V: Tally,
    StandardUniform: Distribution<C>,
    u64: AsPrimitive<C>,
{
    let size = lane.n;
    let n = size as u32;
    let mut hole_u = chain.hole_u;
    let mut hole_v = chain.hole_v;
    let mut perfect_steps = chain.perfect_steps;
    let mut steps = chain.steps;
    let mut h = chain.energy.to_bits();
    let mut perfect_energy = chain.perfect_energy.to_bits();
    while steps < budget {
        if hole_u == NONE {
            if perfect_steps == cycles {
                break;
            }
            perfect_steps += 1;
            perfect_energy += h;
            steps += 1;
            if rng.next_u32() & 1 == 0 {
                let u = below(rng.next_u32(), n);
                unsafe {
                    let v = lane.get(ROW, u);
                    lane.set(ROW, u, NONE);
                    h -= energy_at(tables.energy, size, u, v);
                    hole_u = u;
                    hole_v = v;
                }
            }
        } else {
            if hole_u == anchor {
                tally.visit(hole_v, Energy::from_bits(h));
            }
            steps += 1;
            let cell = hole_u as usize * size + hole_v as usize;
            let (add_below, row_below) =
                unsafe { (*tables.moves.get_unchecked(2 * cell), *tables.moves.get_unchecked(2 * cell + 1)) };
            let x: C = rng.random();
            unsafe {
                if x < add_below {
                    lane.set(ROW, hole_u, hole_v);
                    lane.set(COL, hole_v, hole_u);
                    h += energy_at(tables.energy, size, hole_u, hole_v);
                    hole_u = NONE;
                } else if x < row_below {
                    // the row hole takes over a matched column
                    let z = alias_draw(tables.row_alias, hole_u, n, rng);
                    if z != hole_v {
                        let y = lane.get(COL, z);
                        lane.set(ROW, y, NONE);
                        lane.set(ROW, hole_u, z);
                        lane.set(COL, z, hole_u);
                        h += energy_at(tables.energy, size, hole_u, z) - energy_at(tables.energy, size, y, z);
                        hole_u = y;
                    }
                } else if x - row_below < row_below - add_below {
                    // the column hole takes over a matched row
                    let y = alias_draw(tables.col_alias, hole_v, n, rng);
                    if y != hole_u {
                        let z = lane.get(ROW, y);
                        lane.set(ROW, y, hole_v);
                        lane.set(COL, hole_v, y);
                        h += energy_at(tables.energy, size, y, hole_v) - energy_at(tables.energy, size, y, z);
                        hole_v = z;
                    }
                }
            }
        }
    }
    chain.hole_u = hole_u;
    chain.hole_v = hole_v;
    chain.perfect_steps = perfect_steps;
    chain.steps = steps;
    chain.energy = Energy::from_bits(h);
    chain.perfect_energy = Energy::from_bits(perfect_energy);
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::chain::stream::Stream;
    use crate::sampler::test::{permutations, test_instance};
    use crate::tables::Tables;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    struct Count(Vec<u64>);

    impl Tally for Count {
        fn visit(&mut self, v: u32, _: Energy) {
            self.0[v as usize] += 1;
        }
    }

    /// Starting from exact target samples, the cycle counts must reproduce
    /// the ideal anchor weights lambda(P)/lambda(N(r,v)), computed here by
    /// enumeration. This checks every move's probability and the Kac
    /// identity at once. Half the chains are stopped and continued from
    /// their reported state, which must not change anything.
    #[test]
    fn cycle_counts_recover_the_ideal_weights() {
        let scaled = test_instance();
        let n = scaled.n;
        let beta = 0.8;
        let anchor = 1;
        let log_w = [0.5, 0.0, -0.7, 0.2];
        let tables = Tables::<u64>::build(&scaled, beta, &log_w, anchor);
        let energy = scaled.fixed_energy();
        let far = tables.far(&energy);
        let lambda = |u: usize, v: usize| (-beta * scaled.energy[u * n + v]).exp();
        let perms = permutations(n);
        let weights: Vec<f64> = perms.iter().map(|p| (-beta * scaled.energy_of(p)).exp()).collect();
        let z_perfect: f64 = weights.iter().sum();
        // lambda(N(anchor, v)) = sum over perfect M through (anchor, v) of lambda(M)/lambda(anchor,v)
        let mut z_hole = vec![0.0; n];
        for (p, w) in perms.iter().zip(&weights) {
            let v = p[anchor] as usize;
            z_hole[v] += w / lambda(anchor, v);
        }
        let mut pick_rng = SmallRng::seed_from_u64(9);
        let mut count = Count(vec![0; n]);
        let mut cycles = 0u64;
        let chains = 4000u64;
        let per_chain = 400;
        for seed in 0..chains {
            // exact start: inverse-CDF over the enumerated permutations
            let mut x = pick_rng.random::<f64>() * z_perfect;
            let mut pick = perms.len() - 1;
            for (i, w) in weights.iter().enumerate() {
                if x < *w {
                    pick = i;
                    break;
                }
                x -= w;
            }
            let mut stored: Vec<u16> = perms[pick].iter().map(|&v| v as u16).collect();
            let mut words = vec![0u16; 2 * n];
            let lane = Lane::on(&mut words, n);
            let mut rng = Stream::new(seed);
            // SAFETY: `stored` holds n words, the lane two arrays, and the tables are n x n
            let chain = unsafe {
                let mut chain = begin(&lane, stored.as_ptr(), &far);
                if seed % 2 == 0 {
                    run(&lane, &far, anchor as u32, per_chain, u64::MAX, &mut rng, &mut chain, &mut count);
                } else {
                    // stop after 300 steps and continue elsewhere
                    run(&lane, &far, anchor as u32, per_chain, 300, &mut rng, &mut chain, &mut count);
                    store(&lane, stored.as_mut_ptr());
                    let (block, left) = rng.position();
                    let mut words = vec![0u16; 2 * n];
                    let lane = Lane::on(&mut words, n);
                    let mut rng = Stream::at(seed, block, left);
                    let mut resumed = begin(&lane, stored.as_ptr(), &far);
                    assert_eq!(resumed.energy, chain.energy);
                    resumed.hole_u = chain.hole_u;
                    resumed.hole_v = chain.hole_v;
                    resumed.perfect_steps = chain.perfect_steps;
                    resumed.steps = chain.steps;
                    run(&lane, &far, anchor as u32, per_chain, u64::MAX, &mut rng, &mut resumed, &mut count);
                    chain = resumed;
                }
                chain
            };
            assert_eq!((chain.hole_u, chain.perfect_steps), (NONE, per_chain));
            cycles += chain.perfect_steps;
        }
        let anchor_norm = tables.log_anchor_norm.exp();
        for v in 0..n {
            let ideal = z_perfect / z_hole[v];
            let learned = log_w[v].exp() * anchor_norm * tables.hole_weight(anchor, v) * cycles as f64
                / ((n * n) as f64 * count.0[v] as f64);
            assert!((learned / ideal - 1.0).abs() < 0.03, "v={v}: learned {learned:.5}, ideal {ideal:.5}");
        }
    }

    /// A chain stopped and continued follows the path of one run in one go.
    #[test]
    fn a_continued_chain_follows_the_same_path() {
        let scaled = test_instance();
        let n = scaled.n;
        let tables = Tables::<u64>::build(&scaled, 0.8, &[0.5, 0.0, -0.7, 0.2], 1);
        let energy = scaled.fixed_energy();
        let far = tables.far(&energy);
        let start: Vec<u16> = (0..n as u16).collect();
        let whole = |budgets: &[u64]| {
            let mut stored = start.clone();
            let mut count = Count(vec![0; n]);
            let mut position = (0, 0);
            let mut state: Option<FarChain> = None;
            for &budget in budgets {
                let mut words = vec![0u16; 2 * n];
                let lane = Lane::on(&mut words, n);
                let mut rng = Stream::at(7, position.0, position.1);
                // SAFETY: as in the test above
                unsafe {
                    let mut chain = begin(&lane, stored.as_ptr(), &far);
                    if let Some(previous) = state {
                        chain = FarChain { energy: chain.energy, ..previous };
                    }
                    run(&lane, &far, 1, 5000, budget, &mut rng, &mut chain, &mut count);
                    store(&lane, stored.as_mut_ptr());
                    state = Some(chain);
                }
                position = rng.position();
            }
            let chain = state.unwrap();
            (stored, count.0, chain.steps, chain.perfect_energy)
        };
        assert_eq!(whole(&[u64::MAX]), whole(&[17, 400, 401, 9000, u64::MAX]));
    }
}
