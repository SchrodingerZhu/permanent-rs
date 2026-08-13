use crate::graph::Graph;
use rayon::prelude::*;

/// Exact permanent of the 0/1 bipartite adjacency matrix via Ryser's formula
/// with Gray-code subset enumeration, O(2^n * n). Only feasible for small n
/// (roughly n <= 24); used as ground truth to validate the Monte-Carlo
/// estimator.
pub fn permanent(graph: &Graph) -> f64 {
    let n = graph.size;
    assert!(n <= 24, "Ryser is only feasible for small graphs (n <= 24)");
    assert!(n > 0);
    // columns[j] = bitmask of rows i with A[i][j] = 1.
    let mut columns = vec![0u64; n];
    for (u, edges) in graph.edges.iter().enumerate() {
        for v in edges.iter().copied() {
            columns[v] |= 1 << u;
        }
    }
    // Split the Gray-code traversal into blocks. Each block pays O(n) once to
    // reconstruct its initial row sums, then retains the one-column update of
    // the sequential algorithm. Integer accumulation avoids the severe
    // cancellation that made the former f64 implementation inaccurate near
    // the n=24 limit. At n=24, even the sum of the absolute Ryser terms,
    // sum_k C(24,k) k^24, is below 2^120 and therefore fits in i128.
    const BLOCK_SIZE: u64 = 1 << 12;
    let subset_end = 1u64 << n;
    let block_count = (subset_end - 1).div_ceil(BLOCK_SIZE);
    let total = (0..block_count)
        .into_par_iter()
        .map(|block| {
            let start = 1 + block * BLOCK_SIZE;
            let end = (start + BLOCK_SIZE).min(subset_end);
            let previous = start - 1;
            let mut gray = previous ^ (previous >> 1);
            let mut row_sums = (0..n)
                .map(|row| {
                    columns
                        .iter()
                        .enumerate()
                        .filter(|(column, _)| gray & (1 << column) != 0)
                        .map(|(_, rows)| ((rows >> row) & 1) as u32)
                        .sum::<u32>()
                })
                .collect::<Vec<_>>();
            let mut subtotal = 0i128;

            for k in start..end {
                // Gray code of k differs from that of k-1 in exactly one bit.
                let next_gray = k ^ (k >> 1);
                let flip = (gray ^ next_gray).trailing_zeros() as usize;
                let added = next_gray & (1 << flip) != 0;
                gray = next_gray;
                for (row, row_sum) in row_sums.iter_mut().enumerate() {
                    let value = ((columns[flip] >> row) & 1) as u32;
                    if added {
                        *row_sum += value;
                    } else {
                        *row_sum -= value;
                    }
                }
                let product = row_sums
                    .iter()
                    .map(|value| *value as i128)
                    .product::<i128>();
                if (n as u32 - gray.count_ones()).is_multiple_of(2) {
                    subtotal += product;
                } else {
                    subtotal -= product;
                }
            }
            subtotal
        })
        .sum::<i128>();
    total as f64
}

#[cfg(test)]
mod test {
    use std::{path::PathBuf, time::Instant};

    fn load(name: &str) -> super::Graph {
        let path: PathBuf = env!("CARGO_MANIFEST_DIR").into();
        super::Graph::load(path.join("data").join(name)).unwrap()
    }

    #[test]
    fn complete_graph() {
        // permanent of the all-ones matrix is n!
        let graph = load("complete.json");
        let n = graph.size;
        let factorial = (1..=n).product::<usize>() as f64;
        assert_eq!(super::permanent(&graph), factorial);
    }

    #[test]
    fn cycle_graph() {
        // a single n-cycle circulant has exactly two cycle covers
        let graph = load("cycle.json");
        assert_eq!(super::permanent(&graph), 2.0);
    }

    #[test]
    fn four_cycles_graph() {
        // two independent 4-cycle blocks, permanent 2 * 2
        let graph = load("4-cycles.json");
        assert_eq!(super::permanent(&graph), 4.0);
    }

    #[test]
    #[ignore]
    fn profile_complete_graph() {
        let n = std::env::var("PROFILE_EXACT_N")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(24usize);
        let graph = super::Graph {
            size: n,
            edges: (0..n)
                .map(|_| (0..n).collect::<Vec<_>>().into_boxed_slice())
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        };
        let started = Instant::now();
        let result = super::permanent(&graph);
        println!("n={n} elapsed={:?} result={result:.9e}", started.elapsed());
    }
}
