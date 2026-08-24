use crate::graph::Graph;
use malachite::{
    Integer,
    base::{num::conversion::traits::RoundingFrom, rounding_modes::RoundingMode::Nearest},
};
use rayon::prelude::*;

#[inline]
fn nonzero_row_product(row_sums: &[u32]) -> Option<Integer> {
    if row_sums.contains(&0) {
        return None;
    }

    // Most sparse-graph terms fit in u128. Keep that fast path, but fall back
    // to Malachite as soon as a term outgrows it.
    Some(
        if let Some(product) = row_sums
            .iter()
            .try_fold(1u128, |acc, &value| acc.checked_mul(value as u128))
        {
            Integer::from(product)
        } else {
            row_sums.iter().fold(Integer::from(1), |mut acc, &value| {
                acc *= Integer::from(value);
                acc
            })
        },
    )
}

/// Exact permanent of the 0/1 bipartite adjacency matrix via Ryser's formula
/// with Gray-code subset enumeration, O(2^n * n). Malachite integers keep both
/// block subtotals and the final alternating sum exact. A u64 represents each
/// column subset, so n=64 is the representation limit; runtime becomes
/// impractical much earlier.
pub fn permanent(graph: &Graph) -> Integer {
    let n = graph.size;
    assert!(n <= 64, "Ryser's subset mask supports at most n=64");
    assert!(n > 0);
    // columns[j] = bitmask of rows i with A[i][j] = 1.
    let mut columns = vec![0u64; n];
    for (u, edges) in graph.edges.iter().enumerate() {
        for v in edges.iter().copied() {
            columns[v] |= 1u64 << u;
        }
    }
    // Split the Gray-code traversal into blocks. Each block pays O(n) once to
    // reconstruct its initial row sums, then retains the one-column update of
    // the sequential algorithm. u128 is used only to describe the exclusive
    // endpoint 2^64, which cannot itself be represented by the u64 Gray code.
    const BLOCK_SIZE: u64 = 1 << 12;
    let subset_end = 1u128 << n;
    let block_count = (subset_end - 1).div_ceil(BLOCK_SIZE as u128) as u64;
    (0..block_count)
        .into_par_iter()
        .map(|block| {
            let start_wide = 1 + block as u128 * BLOCK_SIZE as u128;
            let block_len = (start_wide + BLOCK_SIZE as u128).min(subset_end) - start_wide;
            let start = start_wide as u64;
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
            let mut subtotal = Integer::from(0);

            for offset in 0..block_len as u64 {
                let k = start + offset;
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
                let Some(product) = nonzero_row_product(&row_sums) else {
                    continue;
                };
                if (n as u32 - gray.count_ones()).is_multiple_of(2) {
                    subtotal += product;
                } else {
                    subtotal -= product;
                }
            }
            subtotal
        })
        .reduce(|| Integer::from(0), |left, right| left + right)
}

/// Rounded projection used only for comparing the stochastic estimate with an
/// exact integer in logs and the TUI.
pub fn to_f64(value: &Integer) -> f64 {
    f64::rounding_from(value, Nearest).0
}

#[cfg(test)]
mod test {
    use malachite::Integer;
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
        let mut factorial = Integer::from(1);
        for value in 1..=n {
            factorial *= Integer::from(value);
        }
        assert_eq!(super::permanent(&graph), factorial);
    }

    #[test]
    fn cycle_graph() {
        // a single n-cycle circulant has exactly two cycle covers
        let graph = load("cycle.json");
        assert_eq!(super::permanent(&graph), 2);
    }

    #[test]
    fn four_cycles_graph() {
        // two independent 4-cycle blocks, permanent 2 * 2
        let graph = load("4-cycles.json");
        assert_eq!(super::permanent(&graph), 4);
    }

    #[test]
    fn row_product_falls_back_to_malachite_past_u128() {
        let values = [u32::MAX; 5];
        let mut expected = Integer::from(1);
        for value in values {
            expected *= Integer::from(value);
        }

        assert_eq!(super::nonzero_row_product(&values), Some(expected));
        assert_eq!(super::nonzero_row_product(&[3, 0, 7]), None);
    }

    #[test]
    fn n64_block_geometry_reaches_the_last_subset() {
        const BLOCK_SIZE: u128 = 1 << 12;
        let subset_end = 1u128 << 64;
        let block_count = (subset_end - 1).div_ceil(BLOCK_SIZE);
        let last_start = 1 + (block_count - 1) * BLOCK_SIZE;
        let last_len = subset_end - last_start;

        assert_eq!(block_count, 1u128 << 52);
        assert_eq!(last_len, BLOCK_SIZE - 1);
        assert_eq!(last_start + last_len - 1, u64::MAX as u128);
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
        println!("n={n} elapsed={:?} result={result}", started.elapsed());
    }

    #[test]
    #[ignore = "exponential-time exact benchmark"]
    fn profile_graph() {
        let path = std::env::var("PROFILE_EXACT_GRAPH")
            .unwrap_or_else(|_| "data/grid-8x8.json".to_owned());
        let graph = super::Graph::load(&path).unwrap();
        let n = graph.size;
        let started = Instant::now();
        let result = super::permanent(&graph);
        println!(
            "graph={path} n={n} elapsed={:?} result={result}",
            started.elapsed()
        );
    }
}
