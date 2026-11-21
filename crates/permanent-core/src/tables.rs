//! Per-temperature tables shared by the sampler and the learner.
//!
//! With activities lambda(u,v) = exp(-beta * energy(u,v)) and anchor weights
//! w(r, v) for a fixed anchor row r, the rescaled matrix is
//!
//!   B(u,v) = (lambda(u,v) / w(r,v)) / sum_v' (lambda(u,v') / w(r,v')),
//!
//! which is row-stochastic and satisfies B(M) ∝ lambda(M) on perfect
//! matchings, for ANY positive weights. The weights only decide how
//! well-conditioned B is (how often the sampler ends on a perfect matching);
//! they never change which distribution is sampled.

use num_traits::{PrimInt, Unsigned};
use rayon::prelude::*;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::chain::far::FarTables;
use crate::scaling::Scaled;
use crate::stats::log_sum_exp;

static NEXT_ID: AtomicU64 = AtomicU64::new(1);

/// `C` is the integer type of the chains' random thresholds.
pub struct Tables<C> {
    /// Distinct for every table built in this process, so an engine can
    /// tell whether the copy it holds is still the current one.
    pub id: u64,
    pub n: usize,
    pub anchor: usize,
    pub b: Vec<f64>,
    /// Column sums s_v = sum_u B(u,v).
    pub s: Vec<f64>,
    /// ln R_r, R_r = sum_v lambda(r,v)/w(r,v): the anchor row's normalizer.
    pub log_anchor_norm: f64,
    /// The row and column alias tables as the chains read them: a cell's
    /// keep probability as a fraction of the bits above the low 16, and
    /// its alias in the low 16.
    pub packed_rows: Vec<C>,
    pub packed_cols: Vec<C>,
    /// The FAR chain's move thresholds: for holes (u, v), entry
    /// 2 (u n + v) is the probability of adding the edge and the next one
    /// that of adding it or sliding the row hole, as fractions of C's range.
    pub far_moves: Vec<C>,
}

/// Turns probabilities into fractions of 2^bits, in a counter of at least
/// that many bits; 1 becomes the largest fraction below it.
struct Fraction<C> {
    scale: f64,
    below_one: C,
}

impl<C: PrimInt> Fraction<C> {
    fn new(bits: u32) -> Self {
        let whole = bits == C::zero().count_zeros();
        let below_one = if whole { C::max_value() } else { (C::one() << bits as usize) - C::one() };
        Fraction { scale: 2f64.powi(bits as i32), below_one }
    }

    #[inline]
    fn of(&self, p: f64) -> C {
        C::from(p * self.scale).map_or(self.below_one, |x: C| x.min(self.below_one))
    }
}

/// A probability as a fraction of the counter's range, which is how the
/// chains compare it with random bits.
pub fn fraction<C: PrimInt>(p: f64) -> C {
    Fraction::new(C::zero().count_zeros()).of(p)
}

/// Vose's alias method for one distribution given by unnormalized weights.
fn build_alias(weights: &[f64], prob: &mut [f64], alias: &mut [u32]) {
    let n = weights.len();
    let total: f64 = weights.iter().sum();
    let mut small = Vec::with_capacity(n);
    let mut large = Vec::with_capacity(n);
    let mut scaled: Vec<f64> = weights.iter().map(|w| w * n as f64 / total).collect();
    for (i, &p) in scaled.iter().enumerate() {
        if p < 1.0 { small.push(i) } else { large.push(i) }
    }
    while let (Some(&s), Some(&l)) = (small.last(), large.last()) {
        small.pop();
        prob[s] = scaled[s];
        alias[s] = l as u32;
        scaled[l] = (scaled[l] + scaled[s]) - 1.0;
        if scaled[l] < 1.0 {
            large.pop();
            small.push(l);
        }
    }
    for i in large.into_iter().chain(small) {
        prob[i] = 1.0;
        alias[i] = i as u32;
    }
}

impl<C: PrimInt + Unsigned + Send + Sync> Tables<C> {
    pub fn build(scaled: &Scaled, beta: f64, log_w: &[f64], anchor: usize) -> Self {
        let n = scaled.n;
        assert!(n <= crate::chain::lane::MAX_N, "the chains hold vertices in 16-bit words");
        // Every row and every column is independent of the others, so a
        // large matrix is built on the pool. A small one is not worth
        // waking it for.
        let parallel = n * n >= 1 << 16;
        let fill = |(u, out): (usize, &mut [f64])| {
            for v in 0..n {
                out[v] = -beta * scaled.energy[u * n + v] - log_w[v];
            }
            let ls = log_sum_exp(out);
            for x in out.iter_mut() {
                *x = (*x - ls).exp();
            }
            ls
        };
        let mut b = vec![0.0; n * n];
        let norms: Vec<f64> = if parallel {
            b.par_chunks_mut(n).enumerate().map(fill).collect()
        } else {
            b.chunks_mut(n).enumerate().map(fill).collect()
        };
        let mut s = vec![0.0; n];
        for row in b.chunks(n) {
            for (total, x) in s.iter_mut().zip(row) {
                *total += x;
            }
        }
        let keep = Fraction::<C>::new(C::zero().count_zeros() - 16);
        // (the weights of a column, alias probabilities, aliases)
        let scratch = || (vec![0.0; n], vec![0.0; n], vec![0u32; n]);
        let pack = |weights: &[f64], prob: &mut [f64], alias: &mut [u32], out: &mut [C]| {
            build_alias(weights, prob, alias);
            for ((cell, &p), &a) in out.iter_mut().zip(prob.iter()).zip(alias.iter()) {
                *cell = keep.of(p) << 16 | C::from(a).unwrap();
            }
        };
        let pack_row = |(_, prob, alias): &mut (Vec<f64>, Vec<f64>, Vec<u32>), (out, row): (&mut [C], &[f64])| {
            pack(row, prob, alias, out)
        };
        let pack_col = |(column, prob, alias): &mut (Vec<f64>, Vec<f64>, Vec<u32>), (v, out): (usize, &mut [C])| {
            for (u, x) in column.iter_mut().enumerate() {
                *x = b[u * n + v];
            }
            pack(column, prob, alias, out)
        };
        let below = Fraction::<C>::new(C::zero().count_zeros());
        let moves = |(out, row): (&mut [C], &[f64])| {
            for v in 0..n {
                let cell = row[v];
                let half_inv_w = 0.5 / (1.0 + n as f64 * cell + s[v] * (1.0 - cell) + (s[v] - cell));
                let add = n as f64 * cell * half_inv_w;
                out[2 * v] = below.of(add);
                out[2 * v + 1] = below.of(add + s[v] * half_inv_w);
            }
        };
        let mut packed_rows = vec![C::zero(); n * n];
        let mut packed_cols = vec![C::zero(); n * n];
        let mut far_moves = vec![C::zero(); 2 * n * n];
        if parallel {
            packed_rows.par_chunks_mut(n).zip(b.par_chunks(n)).for_each_init(scratch, pack_row);
            packed_cols.par_chunks_mut(n).enumerate().for_each_init(scratch, pack_col);
            far_moves.par_chunks_mut(2 * n).zip(b.par_chunks(n)).for_each(moves);
        } else {
            let mut memory = scratch();
            packed_rows.chunks_mut(n).zip(b.chunks(n)).for_each(|item| pack_row(&mut memory, item));
            packed_cols.chunks_mut(n).enumerate().for_each(|item| pack_col(&mut memory, item));
            far_moves.chunks_mut(2 * n).zip(b.chunks(n)).for_each(moves);
        }
        let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
        Tables { id, n, anchor, b, s, log_anchor_norm: norms[anchor], packed_rows, packed_cols, far_moves }
    }

    /// What the FAR chain reads, with `energy` from `Scaled::fixed_energy`.
    pub fn far<'t>(&'t self, energy: &'t [i64]) -> FarTables<'t, C> {
        FarTables { row_alias: &self.packed_rows, col_alias: &self.packed_cols, moves: &self.far_moves, energy }
    }
}

impl<C> Tables<C> {
    /// FAR hole-class weight W(u,v) = 1 + n B + s_v (1 - B) + (s_v - B).
    #[inline]
    pub fn hole_weight(&self, u: usize, v: usize) -> f64 {
        let b = self.b[u * self.n + v];
        let s = self.s[v];
        1.0 + self.n as f64 * b + s * (1.0 - b) + (s - b)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::chain::far::alias_draw;
    use crate::chain::stream::Stream;

    #[test]
    fn alias_reproduces_the_distribution() {
        let weights = [0.5, 0.0, 3.0, 1.5, 1e-9];
        let mut prob = [0.0; 5];
        let mut alias = [0u32; 5];
        build_alias(&weights, &mut prob, &mut alias);
        let keep = Fraction::<u64>::new(48);
        let packed: Vec<u64> = prob.iter().zip(&alias).map(|(&p, &a)| keep.of(p) << 16 | a as u64).collect();
        let mut rng = Stream::new(3);
        let mut counts = [0usize; 5];
        let draws = 2_000_000;
        for _ in 0..draws {
            // SAFETY: one line of five entries
            counts[unsafe { alias_draw(&packed, 0, 5, &mut rng) } as usize] += 1;
        }
        let total: f64 = weights.iter().sum();
        for i in 0..5 {
            let expected = weights[i] / total;
            let observed = counts[i] as f64 / draws as f64;
            assert!((observed - expected).abs() < 2e-3, "{i}: {observed} vs {expected}");
        }
    }
}
