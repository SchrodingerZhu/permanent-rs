use rayon::{
    iter::ParallelIterator,
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
    pub fn par_mut_rows(&mut self) -> ChunksMut<f64> {
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
    pub beta: f64,
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
        }
    }
}

impl State {
    pub fn activity_of_edge(&self, u: usize, v: usize) -> usize {
        // e ^ (-beta * (1 - A[u, v]))
        if self.adjacency.get(u, v) {
            1
        } else {
            0
        }
    }
    pub fn active_count_of_match(&self, matching: &Match) -> usize {
        matching
            .edges
            .iter()
            .filter(|x| self.adjacency.get(x.0, x.1))
            .count()
    }
    // pub fn activity_of_match(&self, matching: &Match, beta: f64) -> f64 {
    //     let n = matching.size();
    //     let m = self.active_count_of_match(matching);
    //     (beta * (m - n) as f64).exp()
    // }
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
}

#[cfg(test)]
mod test {
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
