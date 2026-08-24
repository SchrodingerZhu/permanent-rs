use crate::graph::{Graph, Match};

pub struct Matrix {
    size: usize,
    data: Box<[f64]>,
}

/// Health report of one hole-weight bootstrap step. Balanced occupancy
/// across the n^2 + 1 classes is the invariant the mixing guarantee rests
/// on; these numbers are the cheap observable proxy for it.
pub struct HoleWeightDiagnostics {
    /// classes whose per-step correction factor hit the clamp
    pub clamped_classes: usize,
    /// smallest per-class sample count (perfect class included)
    pub min_count: usize,
    /// largest per-class sample count (perfect class included)
    pub max_count: usize,
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
    pub fn get(&self, u: usize, v: usize) -> f64 {
        self.data[u * self.size + v]
    }
    pub fn set(&mut self, u: usize, v: usize, value: f64) {
        self.data[u * self.size + v] = value;
    }
    pub fn add(&mut self, u: usize, v: usize, value: f64) {
        self.data[u * self.size + v] += value;
    }


    /// Bootstrap the next hole-weight table from per-class occupancy counts,
    /// following BSVV: with the current table `w`, the chain's stationary
    /// occupancy of hole class (u, v) is proportional to
    /// `lambda(N(u,v)) * w(u,v)` while the perfect class carries
    /// `lambda(P)`, so
    ///
    ///   w*(u, v) = lambda(P) / lambda(N(u,v)) = w(u, v) * c_P / c(u, v)
    ///
    /// up to sampling noise. The BSVV schedule keeps consecutive ideal
    /// tables within a small factor of each other, so a genuine correction
    /// is always small; the per-step clamp bounds the damage a noisy or
    /// starved class count can do, and `clamped_classes` reports how often
    /// it fired (the factor-2 validity invariant is at risk when it does).
    pub(crate) fn hole_weights_from_counts(
        previous: &Matrix,
        counts: &[usize],
        perfect_count: usize,
    ) -> (Self, HoleWeightDiagnostics) {
        let size = previous.dimension();
        assert_eq!(counts.len(), size * size);
        // Weight magnitudes are capped so the table stays representable in
        // f32 for the GPU backend; a class whose ideal weight exceeds the
        // cap is merely under-visited (its true occupancy fades to zero as
        // beta grows), which affects mixing head-room, never stationarity.
        const WEIGHT_CAP: f64 = 1e30;
        // The BSVV schedule moves each ideal weight by at most a small
        // constant factor per step, so a tight clamp never blocks a genuine
        // correction. It also bounds the ratio-estimator term w'/w, whose
        // spikes on starved classes otherwise dominate the estimator's
        // variance (and through it the ln-space Jensen bias of the
        // telescoping product).
        const STEP_FACTOR_CAP: f64 = 4.0;
        let numerator = perfect_count.max(1) as f64;
        let mut matrix = Matrix::new(size, 0.0);
        let mut clamped = 0usize;
        for u in 0..size {
            for v in 0..size {
                let factor = (numerator / counts[u * size + v].max(1) as f64)
                    .clamp(1.0 / STEP_FACTOR_CAP, STEP_FACTOR_CAP);
                if factor == STEP_FACTOR_CAP || factor == 1.0 / STEP_FACTOR_CAP {
                    clamped += 1;
                }
                let value = (previous.get(u, v) * factor).clamp(1.0 / WEIGHT_CAP, WEIGHT_CAP);
                matrix.set(u, v, value);
            }
        }
        let (min_count, max_count) = counts
            .iter()
            .copied()
            .chain([perfect_count])
            .fold((usize::MAX, 0), |(lo, hi), c| (lo.min(c), hi.max(c)));
        (
            matrix,
            HoleWeightDiagnostics {
                clamped_classes: clamped,
                min_count,
                max_count,
            },
        )
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

}
