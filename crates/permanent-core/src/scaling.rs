//! Reduction of a general nonnegative matrix to the annealing instance
//! (Chen, Guo, Vigoda, Yang, "An O~(n^3)-time FPRAS for the permanent",
//! arXiv:2609.20717, Corollary 2).
//!
//! 1. Find a maximum-product perfect matching of the support (Hungarian
//!    method on -ln a). No matching means the permanent is exactly 0.
//! 2. The dual potentials give column factors y_j that make each matched
//!    entry the maximum of its row; dividing every row by its sum then gives
//!    a row-stochastic A' with per(A') = c_sc per(A) and a matched product
//!    P* >= n^-n.
//! 3. Regularize: a~ = max(a', tau). Every entry becomes positive, so the
//!    annealing family lambda_beta(u,v) = a~(u,v)^beta, beta in [0, 1], runs
//!    from the uniform distribution on all n! permutations to the target.
//!
//! The identity the estimator uses is exact for every tau > 0:
//!
//!   per(A') = per(A~) * E_mu[g],   g(M) = prod_{(u,v) in M} a'(u,v)/a~(u,v),
//!
//! with mu the distribution on permutations proportional to their A~ weight.
//! tau only controls how large E[g] is. We take tau = P* / (2 n!), which is
//! never smaller than the paper's n^-n / (2 n!) and still guarantees
//! E[g] >= 2/3: the matchings that touch a raised entry weigh at most
//! n! * tau = P*/2 in total, while per(A') >= P*.

use crate::problem::Problem;
use crate::stats::{ln_factorial, log_sum_exp};

pub struct Scaled {
    pub n: usize,
    /// energy(u,v) = -ln a~(u,v), finite and nonnegative.
    pub energy: Vec<f64>,
    /// ln a'(u,v) - ln a~(u,v) <= 0; -inf where a(u,v) = 0.
    pub log_gap: Vec<f64>,
    /// ln c_sc, where per(A') = c_sc * per(A).
    pub log_scale: f64,
    /// ln(1/tau).
    pub log_inv_tau: f64,
    /// A maximum-product perfect matching: row u is matched to matching[u].
    pub matching: Vec<usize>,
}

pub enum Prepared {
    /// The support has no perfect matching: per(A) = 0 exactly.
    Zero,
    Scaled(Scaled),
}

impl Scaled {
    /// Upper bound on the energy of any permutation.
    pub fn h_max(&self) -> f64 {
        self.n as f64 * self.log_inv_tau
    }
    /// The energies as bits of the chains' fixed-point type, to the nearest.
    pub fn fixed_energy(&self) -> Vec<i64> {
        self.energy.iter().map(|&e| crate::chain::Energy::from_num(e).to_bits()).collect()
    }
    pub fn energy_of(&self, perm: &[u32]) -> f64 {
        let n = self.n;
        perm.iter().enumerate().map(|(u, &v)| self.energy[u * n + v as usize]).sum()
    }
    pub fn log_gap_of(&self, perm: &[u32]) -> f64 {
        let n = self.n;
        perm.iter().enumerate().map(|(u, &v)| self.log_gap[u * n + v as usize]).sum()
    }
}

/// Hungarian method (shortest augmenting paths with potentials), O(n^3).
/// Returns (matching column of each row, column potentials v) with
/// u_i + v_j <= cost(i,j) and equality on the matching, or `None` when no
/// finite-cost perfect matching exists.
fn hungarian(n: usize, cost: &[f64]) -> Option<(Vec<usize>, Vec<f64>)> {
    let inf = f64::INFINITY;
    let mut u = vec![0.0; n + 1];
    let mut v = vec![0.0; n + 1];
    let mut p = vec![0usize; n + 1];
    let mut way = vec![0usize; n + 1];
    for i in 1..=n {
        p[0] = i;
        let mut j0 = 0usize;
        let mut minv = vec![inf; n + 1];
        let mut used = vec![false; n + 1];
        loop {
            used[j0] = true;
            let i0 = p[j0];
            let mut delta = inf;
            let mut j1 = 0usize;
            for j in 1..=n {
                if used[j] {
                    continue;
                }
                let cur = cost[(i0 - 1) * n + (j - 1)] - u[i0] - v[j];
                if cur < minv[j] {
                    minv[j] = cur;
                    way[j] = j0;
                }
                if minv[j] < delta {
                    delta = minv[j];
                    j1 = j;
                }
            }
            if !delta.is_finite() {
                return None;
            }
            for j in 0..=n {
                if used[j] {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
            if p[j0] == 0 {
                break;
            }
        }
        loop {
            let j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
            if j0 == 0 {
                break;
            }
        }
    }
    let mut matching = vec![0usize; n];
    for j in 1..=n {
        matching[p[j] - 1] = j - 1;
    }
    Some((matching, v[1..].to_vec()))
}

pub fn prepare(problem: &Problem) -> Prepared {
    let n = problem.n;
    let cost: Vec<f64> = problem
        .a
        .iter()
        .map(|&x| if x > 0.0 { -x.ln() } else { f64::INFINITY })
        .collect();
    let Some((matching, v)) = hungarian(n, &cost) else {
        return Prepared::Zero;
    };
    // ln a'(u,j) = ln a(u,j) + v_j - ln s_u
    let mut log_a = vec![f64::NEG_INFINITY; n * n];
    let mut log_scale: f64 = v.iter().sum();
    let mut log_best = 0.0;
    let mut row = vec![0.0; n];
    for u in 0..n {
        for j in 0..n {
            row[j] = -cost[u * n + j] + v[j];
        }
        let ls = log_sum_exp(&row);
        log_scale -= ls;
        let mut best = f64::NEG_INFINITY;
        for j in 0..n {
            let x = row[j] - ls;
            log_a[u * n + j] = x;
            best = best.max(x);
        }
        log_best += best;
    }
    let log_tau = log_best - std::f64::consts::LN_2 - ln_factorial(n);
    let mut energy = vec![0.0; n * n];
    let mut log_gap = vec![0.0; n * n];
    for index in 0..n * n {
        let raised = log_a[index].max(log_tau);
        // a~ <= 1 because A' is row-stochastic; clamp rounding noise.
        energy[index] = (-raised).max(0.0);
        log_gap[index] = log_a[index] - raised;
    }
    Prepared::Scaled(Scaled {
        n,
        energy,
        log_gap,
        log_scale,
        log_inv_tau: -log_tau,
        matching,
    })
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::exact;
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    fn random_problem(n: usize, density: f64, rng: &mut SmallRng) -> Problem {
        let mut a = vec![0.0; n * n];
        for x in a.iter_mut() {
            if rng.random::<f64>() < density {
                *x = (rng.random::<f64>() * 12.0 - 6.0).exp();
            }
        }
        for u in 0..n {
            a[u * n + u] = a[u * n + u].max(0.01);
        }
        Problem { n, a }
    }

    #[test]
    fn no_matching_is_zero() {
        // two rows supported on the same single column
        let p = Problem { n: 3, a: vec![1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 1.0, 1.0, 1.0] };
        assert!(matches!(prepare(&p), Prepared::Zero));
    }

    /// per(A~) * E[g] / c_sc must reproduce per(A); check the two exact
    /// pieces (per(A') against per(A), and the E[g] >= 2/3 guarantee).
    #[test]
    fn scaling_identities_hold() {
        let mut rng = SmallRng::seed_from_u64(11);
        for trial in 0..20 {
            let n = 3 + trial % 6;
            let p = random_problem(n, 0.6, &mut rng);
            let Prepared::Scaled(s) = prepare(&p) else { panic!("has the diagonal") };
            let a_prime = Problem {
                n,
                a: (0..n * n).map(|i| (-s.energy[i] + s.log_gap[i]).exp()).collect(),
            };
            let a_tilde = Problem { n, a: s.energy.iter().map(|e| (-e).exp()).collect() };
            for u in 0..n {
                let sum: f64 = a_prime.a[u * n..(u + 1) * n].iter().sum();
                assert!((sum - 1.0).abs() < 1e-9, "row {u} sums to {sum}");
                let matched = a_prime.a[u * n + s.matching[u]];
                let max = a_prime.a[u * n..(u + 1) * n].iter().copied().fold(0.0, f64::max);
                assert!(matched >= max * (1.0 - 1e-9));
            }
            let exact = exact::ln_permanent(&p).unwrap();
            let scaled = exact::ln_permanent(&a_prime).unwrap();
            assert!((scaled - s.log_scale - exact).abs() < 1e-8, "{scaled} {exact}");
            let tilde = exact::ln_permanent(&a_tilde).unwrap();
            let g = (scaled - tilde).exp();
            assert!((2.0 / 3.0..=1.0 + 1e-12).contains(&g), "E[g] = {g}");
        }
    }
}
