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

use rand::RngExt;
use rand::rngs::SmallRng;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::scaling::Scaled;
use crate::stats::log_sum_exp;

static NEXT_ID: AtomicU64 = AtomicU64::new(1);

pub struct Tables {
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
    row_prob: Vec<f64>,
    row_alias: Vec<u32>,
    col_prob: Vec<f64>,
    col_alias: Vec<u32>,
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

impl Tables {
    pub fn build(scaled: &Scaled, beta: f64, log_w: &[f64], anchor: usize) -> Self {
        let n = scaled.n;
        let mut b = vec![0.0; n * n];
        let mut row = vec![0.0; n];
        let mut log_anchor_norm = 0.0;
        for u in 0..n {
            for v in 0..n {
                row[v] = -beta * scaled.energy[u * n + v] - log_w[v];
            }
            let ls = log_sum_exp(&row);
            if u == anchor {
                log_anchor_norm = ls;
            }
            for v in 0..n {
                b[u * n + v] = (row[v] - ls).exp();
            }
        }
        let mut s = vec![0.0; n];
        for u in 0..n {
            for v in 0..n {
                s[v] += b[u * n + v];
            }
        }
        let mut row_prob = vec![0.0; n * n];
        let mut row_alias = vec![0u32; n * n];
        for u in 0..n {
            build_alias(
                &b[u * n..(u + 1) * n],
                &mut row_prob[u * n..(u + 1) * n],
                &mut row_alias[u * n..(u + 1) * n],
            );
        }
        let mut col_prob = vec![0.0; n * n];
        let mut col_alias = vec![0u32; n * n];
        let mut column = vec![0.0; n];
        for v in 0..n {
            for u in 0..n {
                column[u] = b[u * n + v];
            }
            build_alias(
                &column,
                &mut col_prob[v * n..(v + 1) * n],
                &mut col_alias[v * n..(v + 1) * n],
            );
        }
        let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
        Tables { id, n, anchor, b, s, log_anchor_norm, row_prob, row_alias, col_prob, col_alias }
    }

    #[inline]
    fn draw(prob: &[f64], alias: &[u32], rng: &mut SmallRng) -> usize {
        let n = prob.len();
        let i = ((rng.random::<u32>() as u64 * n as u64) >> 32) as usize;
        if rng.random::<f64>() < prob[i] { i } else { alias[i] as usize }
    }

    /// The row alias tables: for row u, entries u*n..(u+1)*n hold each
    /// cell's keep probability and its alias column.
    pub fn row_alias(&self) -> (&[f64], &[u32]) {
        (&self.row_prob, &self.row_alias)
    }

    /// The column alias tables: for column v, entries v*n..(v+1)*n hold
    /// each cell's keep probability and its alias row.
    pub fn col_alias(&self) -> (&[f64], &[u32]) {
        (&self.col_prob, &self.col_alias)
    }

    /// A column v with probability B(u, v).
    #[inline]
    pub fn sample_row(&self, u: usize, rng: &mut SmallRng) -> usize {
        let n = self.n;
        Self::draw(&self.row_prob[u * n..(u + 1) * n], &self.row_alias[u * n..(u + 1) * n], rng)
    }

    /// A row u with probability B(u, v) / s_v.
    #[inline]
    pub fn sample_col(&self, v: usize, rng: &mut SmallRng) -> usize {
        let n = self.n;
        Self::draw(&self.col_prob[v * n..(v + 1) * n], &self.col_alias[v * n..(v + 1) * n], rng)
    }

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
    use rand::SeedableRng;

    #[test]
    fn alias_reproduces_the_distribution() {
        let weights = [0.5, 0.0, 3.0, 1.5, 1e-9];
        let mut prob = [0.0; 5];
        let mut alias = [0u32; 5];
        build_alias(&weights, &mut prob, &mut alias);
        let mut rng = SmallRng::seed_from_u64(3);
        let mut counts = [0usize; 5];
        let draws = 2_000_000;
        for _ in 0..draws {
            counts[Tables::draw(&prob, &alias, &mut rng)] += 1;
        }
        let total: f64 = weights.iter().sum();
        for i in 0..5 {
            let expected = weights[i] / total;
            let observed = counts[i] as f64 / draws as f64;
            assert!((observed - expected).abs() < 2e-3, "{i}: {observed} vs {expected}");
        }
    }
}
