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
//!   holes (u, v):   add (u,v)                     w.p. n B(u,v) / (2W);
//!                   left slide to (u,z), z != v   w.p. B(u,z) s_v / (2W);
//!                   right slide to (y,v), y != u  w.p. B(y,v) / (2W).
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

use rand::RngExt;
use rand::rngs::SmallRng;

use crate::sampler::NONE;
use crate::tables::Tables;

pub struct FarStats {
    /// Perfect steps counted (k unless the step cap was hit).
    pub perfect_steps: u64,
    pub steps: u64,
    /// Visits to N(anchor, v).
    pub visits: Vec<u64>,
    /// Sum of energies over those visits.
    pub visit_energy: Vec<f64>,
    /// Sum of energies over perfect steps.
    pub perfect_energy: f64,
    /// The step cap was hit; `row` was left as it started.
    pub capped: bool,
}

/// Run `cycles` cycles from the perfect matching `row`, which is replaced
/// by the perfect matching the chain stops on.
pub fn run(
    tables: &Tables,
    energy: &[f64],
    row: &mut [u32],
    cycles: u64,
    max_steps: u64,
    rng: &mut SmallRng,
) -> FarStats {
    let n = tables.n;
    let nf = n as f64;
    let r = tables.anchor;
    let start: Vec<u32> = row.to_vec();
    let mut col = vec![NONE; n];
    let mut h = 0.0;
    for (u, &v) in row.iter().enumerate() {
        col[v as usize] = u as u32;
        h += energy[u * n + v as usize];
    }
    let mut hole: Option<(usize, usize)> = None;
    let mut stats = FarStats {
        perfect_steps: 0,
        steps: 0,
        visits: vec![0; n],
        visit_energy: vec![0.0; n],
        perfect_energy: 0.0,
        capped: false,
    };
    loop {
        match hole {
            None => {
                if stats.perfect_steps == cycles {
                    break;
                }
                stats.perfect_steps += 1;
                stats.perfect_energy += h;
                stats.steps += 1;
                if rng.random::<u32>() & 1 == 0 {
                    let u = ((rng.random::<u32>() as u64 * n as u64) >> 32) as usize;
                    let v = row[u] as usize;
                    row[u] = NONE;
                    col[v] = NONE;
                    h -= energy[u * n + v];
                    hole = Some((u, v));
                }
            }
            Some((u, v)) => {
                if u == r {
                    stats.visits[v] += 1;
                    stats.visit_energy[v] += h;
                }
                stats.steps += 1;
                if stats.steps > max_steps {
                    stats.capped = true;
                    row.copy_from_slice(&start);
                    break;
                }
                let b = tables.b[u * n + v];
                let s = tables.s[v];
                let half_inv_w = 0.5 / (1.0 + nf * b + s * (1.0 - b) + (s - b));
                let p_add = nf * b * half_inv_w;
                let p_slide = s * half_inv_w;
                let x = rng.random::<f64>();
                if x < p_add {
                    row[u] = v as u32;
                    col[v] = u as u32;
                    h += energy[u * n + v];
                    hole = None;
                } else if x < p_add + p_slide {
                    // left slide: the left hole takes over a matched column
                    let z = tables.sample_row(u, rng);
                    if z != v {
                        let y = col[z] as usize;
                        row[y] = NONE;
                        row[u] = z as u32;
                        col[z] = u as u32;
                        h += energy[u * n + z] - energy[y * n + z];
                        hole = Some((y, v));
                    }
                } else if x < p_add + 2.0 * p_slide {
                    // right slide: the right hole takes over a matched row
                    let y = tables.sample_col(v, rng);
                    if y != u {
                        let z = row[y] as usize;
                        row[y] = v as u32;
                        col[v] = y as u32;
                        col[z] = NONE;
                        h += energy[y * n + v] - energy[y * n + z];
                        hole = Some((u, z));
                    }
                }
            }
        }
    }
    stats
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::sampler::test::{permutations, test_instance};
    use rand::SeedableRng;

    /// Starting from exact target samples, the cycle counts must reproduce
    /// the ideal anchor weights lambda(P)/lambda(N(r,v)), computed here by
    /// enumeration. This checks every move's probability and the Kac
    /// identity at once.
    #[test]
    fn cycle_counts_recover_the_ideal_weights() {
        let scaled = test_instance();
        let n = scaled.n;
        let beta = 0.8;
        let anchor = 1;
        let log_w = [0.5, 0.0, -0.7, 0.2];
        let tables = Tables::build(&scaled, beta, &log_w, anchor);
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
        let mut rng = SmallRng::seed_from_u64(9);
        let mut visits = vec![0u64; n];
        let mut cycles = 0u64;
        let chains = 4000;
        let per_chain = 400;
        for _ in 0..chains {
            // exact start: inverse-CDF over the enumerated permutations
            let mut x = rng.random::<f64>() * z_perfect;
            let mut pick = perms.len() - 1;
            for (i, w) in weights.iter().enumerate() {
                if x < *w {
                    pick = i;
                    break;
                }
                x -= w;
            }
            let mut row = perms[pick].clone();
            let stats = run(&tables, &scaled.energy, &mut row, per_chain, u64::MAX, &mut rng);
            cycles += stats.perfect_steps;
            for v in 0..n {
                visits[v] += stats.visits[v];
            }
        }
        let anchor_norm = tables.log_anchor_norm.exp();
        for v in 0..n {
            let ideal = z_perfect / z_hole[v];
            let learned = log_w[v].exp() * anchor_norm * tables.hole_weight(anchor, v) * cycles as f64
                / ((n * n) as f64 * visits[v] as f64);
            assert!(
                (learned / ideal - 1.0).abs() < 0.03,
                "v={v}: learned {learned:.5}, ideal {ideal:.5}"
            );
        }
    }
}
