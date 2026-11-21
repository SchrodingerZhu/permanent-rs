//! Parametric random nonnegative matrix generators.

use anyhow::ensure;
use permanent_core::exact;
use permanent_core::problem::Problem;
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::{RngExt, SeedableRng};

use crate::types::{ProblemGeometry, RandomMatrixFamily};
use crate::{BuiltProblem, INSTANT_EXACT_N, MAX_N, ProblemMeta, finalize};

/// Generate a random `n x n` nonnegative matrix from one of the built-in
/// families.
pub fn build(
    n: usize,
    family: RandomMatrixFamily,
    density: f64,
    log_spread: f64,
    ensure_matching: bool,
    seed: u64,
) -> anyhow::Result<BuiltProblem> {
    ensure!((2..=MAX_N).contains(&n), "n must be in 2..={MAX_N} (got {n})");
    ensure!(density.is_finite() && density > 0.0 && density <= 1.0, "density must be in (0, 1]");
    ensure!(log_spread.is_finite() && (0.0..=15.0).contains(&log_spread), "log_spread must be in [0, 15]");

    let mut rng = SmallRng::seed_from_u64(seed);
    let mut a = vec![0.0; n * n];

    match family {
        RandomMatrixFamily::SparseLognormal => {
            for entry in &mut a {
                if rng.random::<f64>() < density {
                    let x = (rng.random::<f64>() * 2.0 - 1.0) * log_spread;
                    *entry = x.exp();
                }
            }
        }
        RandomMatrixFamily::ZeroOneBipartite => {
            for entry in &mut a {
                if rng.random::<f64>() < density {
                    *entry = 1.0;
                }
            }
        }
        RandomMatrixFamily::NearPermutation => {
            let off_scale = (-log_spread).exp();
            for u in 0..n {
                for v in 0..n {
                    if u == v {
                        a[u * n + v] = (rng.random::<f64>() * 0.5).exp();
                    } else if rng.random::<f64>() < density {
                        a[u * n + v] = off_scale * (rng.random::<f64>() * 2.0 - 1.0).exp();
                    }
                }
            }
        }
        RandomMatrixFamily::WeaklyCoupledBlocks => {
            let block = 4.min(n);
            let couple = (-log_spread).exp();
            for u in 0..n {
                for v in 0..n {
                    let same_block = u / block == v / block;
                    if same_block {
                        if rng.random::<f64>() < density.max(0.6) {
                            a[u * n + v] = (rng.random::<f64>() * 2.0 - 1.0).exp();
                        }
                    } else if rng.random::<f64>() < density * 0.25 {
                        a[u * n + v] = couple * (rng.random::<f64>() * 2.0 - 1.0).exp();
                    }
                }
            }
        }
        RandomMatrixFamily::Uniform => {
            for entry in &mut a {
                if rng.random::<f64>() < density {
                    *entry = 0.05 + 0.95 * rng.random::<f64>();
                }
            }
        }
    }

    if ensure_matching {
        let mut perm: Vec<usize> = (0..n).collect();
        perm.shuffle(&mut rng);
        for (u, &v) in perm.iter().enumerate() {
            if a[u * n + v] == 0.0 {
                a[u * n + v] = match family {
                    RandomMatrixFamily::ZeroOneBipartite => 1.0,
                    RandomMatrixFamily::Uniform => 0.5,
                    _ => ((rng.random::<f64>() * 2.0 - 1.0) * (log_spread * 0.5)).exp(),
                };
            }
        }
    }

    let problem = Problem { n, a };
    let (exact_ln, exact_source) =
        if n <= INSTANT_EXACT_N { (exact::ln_permanent(&problem), Some("subset DP".into())) } else { (None, None) };
    let family_name = match family {
        RandomMatrixFamily::SparseLognormal => "sparse lognormal",
        RandomMatrixFamily::ZeroOneBipartite => "0/1 bipartite",
        RandomMatrixFamily::NearPermutation => "near-permutation",
        RandomMatrixFamily::WeaklyCoupledBlocks => "weakly coupled blocks",
        RandomMatrixFamily::Uniform => "uniform random",
    };
    let title = format!("Random {n} × {n} ({family_name}, seed {seed})");
    let description =
        format!("Generated {family_name} matrix (density {:.0}%, spread ±{log_spread:.1}).", density * 100.0);

    finalize(
        problem,
        ProblemMeta {
            title,
            description,
            exact_ln,
            exact_source,
            lattice_sites: None,
            entropy_lower_bound: None,
            geometry: ProblemGeometry::Matrix,
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn random_matrices_respect_invariants() {
        for family in [
            RandomMatrixFamily::SparseLognormal,
            RandomMatrixFamily::ZeroOneBipartite,
            RandomMatrixFamily::NearPermutation,
            RandomMatrixFamily::WeaklyCoupledBlocks,
            RandomMatrixFamily::Uniform,
        ] {
            let built = build(12, family, 0.4, 3.0, true, 99).unwrap();
            assert_eq!(built.problem.n, 12);
            assert!(built.info.has_perfect_matching);
            assert!(built.scaled.is_some());
            assert!(built.problem.a.iter().all(|&x| x.is_finite() && x >= 0.0));
        }
    }
}
