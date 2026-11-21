//! Fast standalone valid-covering explorer for interactive 3D/2D models.
//!
//! Uses a loop-erased alternating-cycle walk (worm move) directly on the
//! nonzero support of `A`, starting from the Hungarian matching. Every state is
//! guaranteed to be a valid perfect matching on the support of `A` (zero raised
//! floor edges), making it suitable for instant visual exploration between runs.
//! Exact population draws during estimation come from `AsyncSession` instead.

use anyhow::ensure;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

use crate::BuiltProblem;
use crate::types::SampleSnapshot;

/// Draw a valid perfect matching on the support of `built` by applying
/// `steps` alternating-cycle (worm) attempts starting from the heaviest matching.
///
/// Returns `Ok(None)` when `built` has no perfect matching (permanent = 0).
pub fn sample_covering(built: &BuiltProblem, seed: u64, steps: u32) -> anyhow::Result<Option<SampleSnapshot>> {
    let problem = &built.problem;
    let n = problem.n;
    ensure!(n >= 2, "matrix dimension must be at least 2");
    let Some(scaled) = &built.scaled else {
        return Ok(None);
    };

    let mut col_of = scaled.matching.clone();
    let mut row_of = vec![0usize; n];
    for (u, &v) in col_of.iter().enumerate() {
        row_of[v] = u;
    }

    let mut adj: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
    for (u, neighbors) in adj.iter_mut().enumerate() {
        for v in 0..n {
            let w = problem.a[u * n + v];
            if w > 0.0 {
                neighbors.push((v, w));
            }
        }
    }

    let mut rng = SmallRng::seed_from_u64(seed);
    let max_walk = (4 * n).clamp(16, 256);
    // Per-column stamp and index into the current alternating walk.
    let mut visited_stamp = vec![0u32; n];
    let mut col_pos = vec![0usize; n];
    let mut walk_rows = Vec::with_capacity(max_walk + 1);
    let mut walk_cols = Vec::with_capacity(max_walk + 1);
    let mut stamp = 0u32;

    for _ in 0..steps {
        stamp = stamp.wrapping_add(1);
        if stamp == 0 {
            visited_stamp.fill(0);
            stamp = 1;
        }
        walk_rows.clear();
        walk_cols.clear();

        let u0 = rng.random_range(0..n);
        let c0 = col_of[u0];
        walk_rows.push(u0);
        walk_cols.push(c0);
        visited_stamp[c0] = stamp;
        col_pos[c0] = 0;

        let mut cur_u = u0;
        for _ in 0..max_walk {
            let neighbors = &adj[cur_u];
            if neighbors.len() <= 1 {
                break;
            }
            // Pick a neighbor distinct from cur_u's current matched column.
            let current_c = col_of[cur_u];
            let mut chosen = current_c;
            for _ in 0..4 {
                let &(z, _) = &neighbors[rng.random_range(0..neighbors.len())];
                if z != current_c {
                    chosen = z;
                    break;
                }
            }
            if chosen == current_c {
                // Guaranteed to exist since neighbors.len() > 1 and current_c is in neighbors.
                let idx = rng.random_range(0..(neighbors.len() - 1));
                chosen = neighbors.iter().filter(|&&(z, _)| z != current_c).nth(idx).unwrap().0;
            }

            if visited_stamp[chosen] == stamp {
                // Closed a simple alternating cycle from index `start` through the walk end.
                let start = col_pos[chosen];
                let end = walk_rows.len() - 1;
                let mut log_ratio = 0.0;
                for i in start..=end {
                    let u = walk_rows[i];
                    let old_c = walk_cols[i];
                    let new_c = if i == end { chosen } else { walk_cols[i + 1] };
                    log_ratio += problem.a[u * n + new_c].ln() - problem.a[u * n + old_c].ln();
                }
                if log_ratio >= 0.0 || rng.random::<f64>() < log_ratio.exp() {
                    for i in start..=end {
                        let u = walk_rows[i];
                        let new_c = if i == end { chosen } else { walk_cols[i + 1] };
                        col_of[u] = new_c;
                        row_of[new_c] = u;
                    }
                }
                break;
            }

            let next_u = row_of[chosen];
            visited_stamp[chosen] = stamp;
            col_pos[chosen] = walk_cols.len();
            walk_rows.push(next_u);
            walk_cols.push(chosen);
            cur_u = next_u;
        }
    }

    let matching: Vec<u32> = col_of.into_iter().map(|v| v as u32).collect();
    Ok(Some(SampleSnapshot { beta: 1.0, matching, valid: true }))
}

#[cfg(test)]
mod tests {
    use permanent_core::problem::Problem;

    use super::*;
    use crate::build_problem;
    use crate::types::{ProblemSpec, RandomMatrixFamily};

    fn assert_valid_matching(problem: &Problem, matching: &[u32]) {
        let n = problem.n;
        assert_eq!(matching.len(), n);
        let mut seen = vec![false; n];
        for (u, &v) in matching.iter().enumerate() {
            let v = v as usize;
            assert!(v < n);
            assert!(!seen[v], "duplicate column {v}");
            seen[v] = true;
            assert!(problem.a[u * n + v] > 0.0, "zero-weight edge ({u}, {v})");
        }
    }

    #[test]
    fn worm_sampler_preserves_valid_matchings_and_mixes() {
        // 3D open 4x4x4 cube
        let cube = build_problem(&ProblemSpec::Dimer3d {
            lx: 4,
            ly: 4,
            lz: 4,
            periodic: false,
            weights: [1.0, 1.0, 1.0],
            disorder: 0.0,
            vacancy_rate: 0.0,
            ensure_matching: true,
            seed: 1,
        })
        .unwrap();
        let a = sample_covering(&cube, 1, 64).unwrap().unwrap();
        let b = sample_covering(&cube, 2, 64).unwrap().unwrap();
        assert_valid_matching(&cube.problem, &a.matching);
        assert_valid_matching(&cube.problem, &b.matching);
        assert_ne!(a.matching, b.matching, "distinct seeds should explore distinct coverings");

        // Disjoint cycles (length 16): worm walk should cleanly flip full cycles.
        let cycles = build_problem(&ProblemSpec::DisjointCycles { cycle_sizes: vec![16, 16] }).unwrap();
        let c = sample_covering(&cycles, 9, 32).unwrap().unwrap();
        assert_valid_matching(&cycles.problem, &c.matching);

        // Weighted sparse matrix.
        let weighted = build_problem(&ProblemSpec::RandomMatrix {
            n: 16,
            family: RandomMatrixFamily::SparseLognormal,
            density: 0.45,
            log_spread: 3.0,
            ensure_matching: true,
            seed: 5,
        })
        .unwrap();
        let w = sample_covering(&weighted, 11, 64).unwrap().unwrap();
        assert_valid_matching(&weighted.problem, &w.matching);
    }

    #[test]
    fn zero_permanent_returns_none() {
        // column 3 is empty
        let text = "%%MatrixMarket matrix coordinate real general\n3 3 4\n1 1 1\n2 1 2\n3 1 1\n3 2 1\n".into();
        let zero = build_problem(&ProblemSpec::MatrixMarket { text }).unwrap();
        assert!(sample_covering(&zero, 1, 16).unwrap().is_none());
    }
}
