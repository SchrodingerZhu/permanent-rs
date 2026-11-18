//! Exact reference permanent for small matrices.
//!
//! A subset dynamic program over columns: after placing rows 0..k the state
//! is the set of used columns. All terms are nonnegative, so unlike Ryser's
//! alternating sum there is no cancellation and f64 is accurate. O(n 2^n)
//! time and 2^n memory.

use crate::problem::Problem;

pub const MAX_EXACT_N: usize = 26;

/// ln per(A), or `None` when n is too large. `-inf` when the permanent is 0.
pub fn ln_permanent(problem: &Problem) -> Option<f64> {
    let n = problem.n;
    if n > MAX_EXACT_N {
        return None;
    }
    // Rescale each row by its maximum so the table stays in range.
    let mut log_scale = 0.0;
    let mut rows = problem.a.clone();
    for u in 0..n {
        let max = rows[u * n..(u + 1) * n].iter().copied().fold(0.0, f64::max);
        if max == 0.0 {
            return Some(f64::NEG_INFINITY);
        }
        log_scale += max.ln();
        for x in &mut rows[u * n..(u + 1) * n] {
            *x /= max;
        }
    }
    let mut table = vec![0.0f64; 1usize << n];
    table[0] = 1.0;
    for mask in 0..(1usize << n) {
        let value = table[mask];
        if value == 0.0 {
            continue;
        }
        let row = mask.count_ones() as usize;
        if row == n {
            continue;
        }
        let mut free = !mask & ((1usize << n) - 1);
        while free != 0 {
            let v = free.trailing_zeros() as usize;
            free &= free - 1;
            let weight = rows[row * n + v];
            if weight != 0.0 {
                table[mask | (1 << v)] += value * weight;
            }
        }
    }
    Some(table[(1usize << n) - 1].ln() + log_scale)
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn all_ones_is_factorial() {
        let n = 7;
        let p = Problem { n, a: vec![1.0; n * n] };
        assert!((ln_permanent(&p).unwrap() - 5040f64.ln()).abs() < 1e-12);
    }

    #[test]
    fn small_weighted() {
        // per [[1,2],[3,4]] = 1*4 + 2*3 = 10
        let p = Problem { n: 2, a: vec![1.0, 2.0, 3.0, 4.0] };
        assert!((ln_permanent(&p).unwrap() - 10f64.ln()).abs() < 1e-12);
    }
}
