//! Disjoint bipartite cycles generator (`per(A) = 2^k`).

use anyhow::{Context, ensure};
use permanent_core::problem::Problem;

use crate::types::ProblemGeometry;
use crate::{BuiltProblem, MAX_N, ProblemMeta, finalize};

/// Build a block-diagonal 0/1 matrix of disjoint bipartite cycles whose exact
/// permanent is `2^cycle_sizes.len()`.
pub fn build(cycle_sizes: &[usize]) -> anyhow::Result<BuiltProblem> {
    ensure!(!cycle_sizes.is_empty(), "specify at least one cycle");
    let mut n = 0usize;
    for &size in cycle_sizes {
        ensure!(size >= 2, "each cycle must have at least 2 rows and 2 columns (got {size})");
        n = n.checked_add(size).context("total size overflow")?;
    }
    ensure!(n <= MAX_N, "total dimension n = {n} exceeds limit {MAX_N}");

    let mut a = vec![0.0; n * n];
    let mut offset = 0usize;
    for &size in cycle_sizes {
        for i in 0..size {
            let row = offset + i;
            let col_a = offset + i;
            let col_b = offset + (i + 1) % size;
            a[row * n + col_a] = 1.0;
            a[row * n + col_b] = 1.0;
        }
        offset += size;
    }

    let k = cycle_sizes.len();
    let exact_ln = (k as f64) * std::f64::consts::LN_2;
    let title = format!("{k} separate cycles (n = {n})");
    let description = format!(
        "Disjoint 0/1 bipartite cycles of sizes {cycle_sizes:?}. Each cycle has 2 perfect matchings, so per(A) = 2^{k}."
    );

    finalize(
        Problem { n, a },
        ProblemMeta {
            title,
            description,
            exact_ln: Some(exact_ln),
            exact_source: Some(format!("exact 2^{k} disjoint cycles")),
            lattice_sites: None,
            entropy_lower_bound: None,
            geometry: ProblemGeometry::Cycles { cycle_sizes: cycle_sizes.to_vec() },
        },
    )
}

#[cfg(test)]
mod tests {
    use permanent_core::exact;

    use super::*;

    #[test]
    fn disjoint_cycles_has_exact_two_to_the_k() {
        let small = build(&[3, 4, 5]).unwrap();
        assert_eq!(small.problem.n, 12);
        let exact = exact::ln_permanent(&small.problem).unwrap();
        assert!((exact - 8.0_f64.ln()).abs() < 1e-12);
        assert!((small.info.exact_ln.unwrap() - exact).abs() < 1e-12);

        let four_cycles = build(&[16, 16, 16, 16]).unwrap();
        assert_eq!(four_cycles.problem.n, 64);
        assert_eq!(four_cycles.info.nnz, 128);
        assert!((four_cycles.info.exact_ln.unwrap() - 16.0_f64.ln()).abs() < 1e-12);
    }
}
