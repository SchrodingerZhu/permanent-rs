//! Input: a nonnegative n x n matrix in Matrix Market coordinate format.

use anyhow::{Context, bail, ensure};
use sprs::io::{IoError, read_matrix_market_from_bufread as read};
use sprs::num_kinds::{NumKind, Pattern};
use sprs::{TriMat, TriMatI};
use std::path::Path;

/// A nonnegative square matrix in row-major order.
#[derive(Debug, Clone)]
pub struct Problem {
    pub n: usize,
    pub a: Vec<f64>,
}

impl Problem {
    /// Entries given more than once add up.
    pub fn from_triplets(matrix: &TriMat<f64>) -> anyhow::Result<Self> {
        let (n, cols) = matrix.shape();
        ensure!(n == cols, "the matrix is {n} x {cols}, not square");
        ensure!(n > 0, "the matrix is empty");
        let mut a = vec![0.0; n * n];
        for (&x, (u, v)) in matrix.triplet_iter() {
            ensure!(x.is_finite() && x >= 0.0, "entry ({u}, {v}) = {x} is not a finite nonnegative number");
            a[u * n + v] += x;
        }
        Ok(Problem { n, a })
    }

    pub fn load(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let path = path.as_ref();
        let text = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
        Self::from_matrix_market(&text).with_context(|| format!("parsing {}", path.display()))
    }

    /// Parses a `real`, `integer` or `pattern` matrix; a pattern entry is 1.
    pub fn from_matrix_market(text: &[u8]) -> anyhow::Result<Self> {
        let matrix = match read::<f64, usize, _>(&mut &*text) {
            Err(IoError::MismatchedMatrixMarketRead(_, NumKind::Integer)) => {
                read::<i64, usize, _>(&mut &*text).map(|m| convert(&m, |&x| x as f64))
            }
            Err(IoError::MismatchedMatrixMarketRead(_, NumKind::Pattern)) => {
                read::<Pattern, usize, _>(&mut &*text).map(|m| convert(&m, |_| 1.0))
            }
            Err(IoError::MismatchedMatrixMarketRead(_, NumKind::Complex)) => {
                bail!("complex matrices are not supported")
            }
            other => other,
        }?;
        Self::from_triplets(&matrix)
    }

    pub fn is_zero_one(&self) -> bool {
        self.a.iter().all(|&x| x == 0.0 || x == 1.0)
    }
}

fn convert<N>(matrix: &TriMatI<N, usize>, value: impl Fn(&N) -> f64) -> TriMat<f64> {
    let mut out = TriMat::with_capacity(matrix.shape(), matrix.nnz());
    for (x, (u, v)) in matrix.triplet_iter() {
        out.add_triplet(u, v, value(x));
    }
    out
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn reads_real_entries_and_adds_duplicates() {
        let text = b"%%MatrixMarket matrix coordinate real general\n% a comment\n2 2 3\n1 1 0.5\n2 1 2\n1 1 0.25\n";
        let problem = Problem::from_matrix_market(text).unwrap();
        assert_eq!((problem.n, problem.a), (2, vec![0.75, 0.0, 2.0, 0.0]));
    }

    #[test]
    fn reads_integer_and_pattern_fields() {
        let integer = b"%%MatrixMarket matrix coordinate integer general\n2 2 2\n1 2 3\n2 1 4\n";
        assert_eq!(Problem::from_matrix_market(integer).unwrap().a, [0.0, 3.0, 4.0, 0.0]);
        let pattern = b"%%MatrixMarket matrix coordinate pattern symmetric\n2 2 2\n1 1\n2 1\n";
        let problem = Problem::from_matrix_market(pattern).unwrap();
        assert_eq!(problem.a, [1.0, 1.0, 1.0, 0.0]);
        assert!(problem.is_zero_one());
    }

    #[test]
    fn rejects_what_has_no_permanent() {
        let negative = b"%%MatrixMarket matrix coordinate real general\n1 1 1\n1 1 -1\n";
        assert!(Problem::from_matrix_market(negative).is_err());
        let rectangular = b"%%MatrixMarket matrix coordinate real general\n1 2 1\n1 1 1\n";
        assert!(Problem::from_matrix_market(rectangular).is_err());
        let complex = b"%%MatrixMarket matrix coordinate complex general\n1 1 1\n1 1 1 0\n";
        assert!(Problem::from_matrix_market(complex).is_err());
    }
}
