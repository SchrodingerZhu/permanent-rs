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
    pub fn get(&self, u: usize, v: usize) -> f64 {
        self.data[u * self.size + v]
    }
    pub fn set(&mut self, u: usize, v: usize, value: f64) {
        self.data[u * self.size + v] = value;
    }
    pub fn add(&mut self, u: usize, v: usize, value: f64) {
        self.data[u * self.size + v] += value;
    }
    pub fn combine(&mut self, other: &Self, ratio: f64) {
        for (i, j) in self.data.iter_mut().zip(other.data.iter()) {
            *i = *i * (1.0 - ratio) + *j * ratio;
        }
    }
    pub fn scale(&mut self, ratio: f64) {
        for i in self.data.iter_mut() {
            *i *= ratio;
        }
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
    weight: Matrix,
    pub beta: f64,
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
    // fn active_count_of_match(&self, matching: &Match) -> usize {
    //     matching
    //         .edges
    //         .iter()
    //         .filter(|x| self.adjacency.get(x.0, x.1))
    //         .count()
    // }
    // pub fn activity_of_match(&self, matching: &Match, beta: f64) -> f64 {
    //     let n = matching.size();
    //     let m = self.active_count_of_match(matching);
    //     (beta * (m - n) as f64).exp()
    // }
    pub fn weight_of_edge(&self, u: usize, v: usize) -> f64 {
        self.weight.get(u, v)
    }
    // pub fn weight_of_match(&self, matching: &Match) -> f64 {
    //     matching
    //         .edges
    //         .iter()
    //         .map(|x| self.weight.get(x.0, x.1))
    //         .sum()
    // }
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
