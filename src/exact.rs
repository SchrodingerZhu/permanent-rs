use crate::graph::Graph;

/// Exact permanent of the 0/1 bipartite adjacency matrix via Ryser's formula
/// with Gray-code subset enumeration, O(2^n * n). Only feasible for small n
/// (roughly n <= 24); used as ground truth to validate the Monte-Carlo
/// estimator.
pub fn permanent(graph: &Graph) -> f64 {
    let n = graph.size;
    assert!(n <= 24, "Ryser is only feasible for small graphs (n <= 24)");
    assert!(n > 0);
    // row_sums[i] accumulates sum_{j in S} A[i][j] for the current subset S.
    let mut row_sums = vec![0.0f64; n];
    // columns[j] = bitmask of rows i with A[i][j] = 1.
    let mut columns = vec![0u64; n];
    for (u, edges) in graph.edges.iter().enumerate() {
        for v in edges.iter().copied() {
            columns[v] |= 1 << u;
        }
    }
    let mut total = 0.0f64;
    let mut gray = 0u64;
    for k in 1u64..1 << n {
        // Gray code of k differs from that of k-1 in exactly bit `flip`.
        let next_gray = k ^ (k >> 1);
        let flip = (gray ^ next_gray).trailing_zeros() as usize;
        let added = next_gray & (1 << flip) != 0;
        gray = next_gray;
        for (i, row_sum) in row_sums.iter_mut().enumerate() {
            let a = ((columns[flip] >> i) & 1) as f64;
            if added {
                *row_sum += a;
            } else {
                *row_sum -= a;
            }
        }
        let product: f64 = row_sums.iter().product();
        let sign = if (n as u32 - gray.count_ones()).is_multiple_of(2) {
            1.0
        } else {
            -1.0
        };
        total += sign * product;
    }
    total
}

#[cfg(test)]
mod test {
    use std::path::PathBuf;

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
}
