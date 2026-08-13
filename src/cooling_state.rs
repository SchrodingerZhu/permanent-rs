use rayon::{
    iter::{IndexedParallelIterator, ParallelIterator},
    slice::{ChunksMut, ParallelSliceMut},
};

use crate::graph::{Graph, Match};

pub struct Matrix {
    size: usize,
    data: Box<[f64]>,
}

impl Matrix {
    pub fn new(size: usize, initial: f64) -> Self {
        Matrix {
            size,
            data: vec![initial; size * size].into_boxed_slice(),
        }
    }
    pub fn dimension(&self) -> usize {
        self.size
    }
    pub fn par_mut_rows(&mut self) -> ChunksMut<'_, f64> {
        self.data.par_chunks_mut(self.size)
    }
    pub fn get(&self, u: usize, v: usize) -> f64 {
        self.data[u * self.size + v]
    }
    pub fn set(&mut self, u: usize, v: usize, value: f64) {
        self.data[u * self.size + v] = value;
    }
    pub fn add(&mut self, u: usize, v: usize, value: f64) {
        self.data[u * self.size + v] += value;
    }
    pub fn transform(&mut self, f: impl Fn(f64) -> f64 + Sync) {
        self.par_mut_rows().for_each(|row| {
            for x in row.iter_mut() {
                *x = f(*x);
            }
        });
    }

    /// A tight lower bound on the minimum perfect-matching weight, computed
    /// from the feasible dual produced by the Hungarian algorithm. Accepting a
    /// matching M with probability `lower_bound / W(M)` removes the auxiliary
    /// W(M) factor while avoiding nearly all avoidable rejection.
    pub fn matching_weight_lower_bound(&self) -> f64 {
        let n = self.size;
        let mut row_potential = vec![0.0; n + 1];
        let mut column_potential = vec![0.0; n + 1];
        let mut matched_row = vec![0usize; n + 1];
        let mut predecessor = vec![0usize; n + 1];

        for row in 1..=n {
            matched_row[0] = row;
            let mut column = 0;
            let mut minimum_reduced_cost = vec![f64::INFINITY; n + 1];
            let mut used = vec![false; n + 1];

            loop {
                used[column] = true;
                let current_row = matched_row[column];
                let mut delta = f64::INFINITY;
                let mut next_column = 0;
                for candidate in 1..=n {
                    if used[candidate] {
                        continue;
                    }
                    let reduced_cost = self.get(current_row - 1, candidate - 1)
                        - row_potential[current_row]
                        - column_potential[candidate];
                    if reduced_cost < minimum_reduced_cost[candidate] {
                        minimum_reduced_cost[candidate] = reduced_cost;
                        predecessor[candidate] = column;
                    }
                    if minimum_reduced_cost[candidate] < delta {
                        delta = minimum_reduced_cost[candidate];
                        next_column = candidate;
                    }
                }

                for candidate in 0..=n {
                    if used[candidate] {
                        row_potential[matched_row[candidate]] += delta;
                        column_potential[candidate] -= delta;
                    } else {
                        minimum_reduced_cost[candidate] -= delta;
                    }
                }
                column = next_column;
                if matched_row[column] == 0 {
                    break;
                }
            }

            loop {
                let previous = predecessor[column];
                matched_row[column] = matched_row[previous];
                column = previous;
                if column == 0 {
                    break;
                }
            }
        }

        // Floating-point updates can violate a dual constraint by a few ulps.
        // Shift every row potential down by the largest observed violation;
        // this restores u_i + v_j <= w_ij and therefore makes the dual
        // objective a genuine lower bound rather than an approximate optimum.
        let mut violation = 0.0f64;
        for (row, &row_value) in row_potential.iter().enumerate().skip(1) {
            for (column, &column_value) in column_potential.iter().enumerate().skip(1) {
                violation = violation.max(row_value + column_value - self.get(row - 1, column - 1));
            }
        }
        let row_sum = row_potential[1..].iter().sum::<f64>();
        let column_sum = column_potential[1..].iter().sum::<f64>();
        let bound = row_sum + column_sum - violation * n as f64;
        let magnitude = row_potential[1..]
            .iter()
            .chain(&column_potential[1..])
            .map(|value| value.abs())
            .sum::<f64>();
        (bound - magnitude * f64::EPSILON * (4 * n + 8) as f64).max(0.0)
    }

    /// Construct the next adaptive weight matrix from an edge-sample
    /// histogram. Shared by the CPU and GPU chain backends so their annealing
    /// schedules differ only in transition arithmetic and random streams.
    pub(crate) fn from_sample_counts(state: &State, counts: &[usize]) -> Self {
        let size = state.weight.dimension();
        assert_eq!(counts.len(), size * size);
        let mut matrix = Matrix::new(size, 0.0);
        let sum = matrix
            .par_mut_rows()
            .enumerate()
            .map(|(row_index, row)| {
                let mut row_sum = 0.0;
                for (column, item) in row.iter_mut().enumerate() {
                    let value = counts[row_index * size + column].max(1) as f64
                        / state.weight_of_edge(row_index, column);
                    *item = value;
                    row_sum += value;
                }
                row_sum
            })
            .sum::<f64>();

        // Cap well below the scale where f64 addition starts dropping other
        // summands of W(M). The cap affects mixing, not estimator bias.
        const WEIGHT_CAP: f64 = 1e12;
        let n = size as f64;
        matrix.transform(|value| (sum / (value * n)).min(WEIGHT_CAP));
        matrix
    }
}

pub struct BitMatrix {
    size: usize,
    data: Box<[u64]>,
}

impl BitMatrix {
    pub fn new(size: usize) -> Self {
        BitMatrix {
            size,
            data: vec![0; size * size / 64 + 1].into_boxed_slice(),
        }
    }
    pub fn get(&self, u: usize, v: usize) -> bool {
        self.data[(u * self.size + v) / 64] & (1 << ((u * self.size + v) % 64)) != 0
    }
    pub fn set(&mut self, u: usize, v: usize, value: bool) {
        if value {
            self.data[(u * self.size + v) / 64] |= 1 << ((u * self.size + v) % 64);
        } else {
            self.data[(u * self.size + v) / 64] &= !(1 << ((u * self.size + v) % 64));
        }
    }
}

pub struct State {
    adjacency: BitMatrix,
    pub weight: Matrix,
    beta: f64,
    /// cached e^{beta * delta} for delta in -2..=2, the only values a
    /// transposition can change the active count by; avoids an exp() call in
    /// the innermost Metropolis loop
    exp_beta: [f64; 5],
}

impl<'a> From<&'a Graph> for State {
    fn from(graph: &'a Graph) -> Self {
        let mut adjacency = BitMatrix::new(graph.size);
        let weight = Matrix::new(graph.size, graph.size as f64);
        for (u, edges) in graph.edges.iter().enumerate() {
            for v in edges.iter().copied() {
                adjacency.set(u, v, true);
            }
        }
        State {
            adjacency,
            weight,
            beta: 0.0,
            exp_beta: [1.0; 5],
        }
    }
}

impl State {
    pub fn beta(&self) -> f64 {
        self.beta
    }
    pub fn set_beta(&mut self, beta: f64) {
        self.beta = beta;
        self.exp_beta = std::array::from_fn(|i| (beta * (i as f64 - 2.0)).exp());
    }
    pub fn exp_beta_delta(&self, delta: isize) -> f64 {
        self.exp_beta[(delta + 2) as usize]
    }
    pub fn activity_of_edge(&self, u: usize, v: usize) -> usize {
        // e ^ (-beta * (1 - A[u, v]))
        if self.adjacency.get(u, v) { 1 } else { 0 }
    }
    pub fn active_count_of_match(&self, matching: &Match) -> usize {
        matching
            .edges
            .iter()
            .filter(|x| self.adjacency.get(x.0, x.1))
            .count()
    }
    pub fn weight_of_edge(&self, u: usize, v: usize) -> f64 {
        self.weight.get(u, v)
    }
    pub fn weight_of_match(&self, matching: &Match) -> f64 {
        matching
            .edges
            .iter()
            .map(|x| self.weight.get(x.0, x.1))
            .sum()
    }

    pub fn matching_weight_lower_bound(&self) -> f64 {
        self.weight.matching_weight_lower_bound()
    }
}

#[cfg(test)]
mod test {
    use std::time::Instant;

    #[test]
    fn bitmatrix_test() {
        let mut diagnal = super::BitMatrix::new(10);
        for i in 0..10 {
            diagnal.set(i, i, true);
        }
        for i in 0..10 {
            for j in 0..10 {
                assert_eq!(diagnal.get(i, j), i == j);
            }
        }
    }

    #[test]
    fn matching_weight_lower_bound_is_valid() {
        let mut matrix = super::Matrix::new(3, 0.0);
        let values = [[1.0, 8.0, 9.0], [7.0, 2.0, 6.0], [4.0, 5.0, 3.0]];
        for (row, values) in values.into_iter().enumerate() {
            for (column, value) in values.into_iter().enumerate() {
                matrix.set(row, column, value);
            }
        }
        let bound = matrix.matching_weight_lower_bound();
        assert!(bound <= 6.0);
        assert!((bound - 6.0).abs() < 1e-12);
    }

    #[test]
    fn matching_weight_lower_bound_handles_conflicting_minima() {
        let mut matrix = super::Matrix::new(3, 0.0);
        let values = [[0.0, 100.0, 100.0], [0.0, 100.0, 100.0], [100.0, 0.0, 0.0]];
        for (row, values) in values.into_iter().enumerate() {
            for (column, value) in values.into_iter().enumerate() {
                matrix.set(row, column, value);
            }
        }
        let bound = matrix.matching_weight_lower_bound();
        assert!(bound <= 100.0);
        assert!((bound - 100.0).abs() < 1e-11);
    }

    #[test]
    fn matching_weight_lower_bound_agrees_with_brute_force() {
        fn next_permutation(values: &mut [usize]) -> bool {
            let Some(pivot) = (0..values.len() - 1)
                .rev()
                .find(|&index| values[index] < values[index + 1])
            else {
                return false;
            };
            let swap = (pivot + 1..values.len())
                .rev()
                .find(|&index| values[pivot] < values[index])
                .unwrap();
            values.swap(pivot, swap);
            values[pivot + 1..].reverse();
            true
        }

        for n in 2..=6 {
            for seed in 0..16usize {
                let mut matrix = super::Matrix::new(n, 0.0);
                for row in 0..n {
                    for column in 0..n {
                        let value =
                            1 + (row * 97 + column * 53 + seed * 29 + row * column * 11) % 101;
                        matrix.set(row, column, value as f64);
                    }
                }
                let mut permutation = (0..n).collect::<Vec<_>>();
                let mut exact = f64::INFINITY;
                loop {
                    exact = exact.min(
                        permutation
                            .iter()
                            .enumerate()
                            .map(|(row, &column)| matrix.get(row, column))
                            .sum::<f64>(),
                    );
                    if !next_permutation(&mut permutation) {
                        break;
                    }
                }
                let bound = matrix.matching_weight_lower_bound();
                assert!(bound <= exact, "bound {bound} exceeded optimum {exact}");
                assert!(
                    (bound - exact).abs() < exact * 1e-12,
                    "bound {bound} did not reach optimum {exact}"
                );
            }
        }
    }

    #[test]
    #[ignore]
    fn profile_matching_weight_lower_bound() {
        let n = std::env::var("PROFILE_BOUND_N")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(256usize);
        let mut matrix = super::Matrix::new(n, 0.0);
        for row in 0..n {
            for column in 0..n {
                matrix.set(
                    row,
                    column,
                    1.0 + ((row * 97 + column * 53 + row * column * 11) % 1009) as f64,
                );
            }
        }
        let started = Instant::now();
        let bound = matrix.matching_weight_lower_bound();
        println!("n={n} elapsed={:?} bound={bound}", started.elapsed());
    }
}
