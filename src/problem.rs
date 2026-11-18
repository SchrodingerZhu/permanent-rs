//! Input: a nonnegative n x n matrix, dense or sparse, or a 0/1 bipartite
//! graph in the legacy `{size, edges}` form.

use anyhow::{Context, bail};
use serde::Deserialize;
use std::path::Path;

/// A nonnegative square matrix in row-major order.
#[derive(Debug, Clone)]
pub struct Problem {
    pub n: usize,
    pub a: Vec<f64>,
}

#[derive(Deserialize)]
struct Raw {
    size: Option<usize>,
    /// 0/1 adjacency lists: `edges[u]` lists the columns adjacent to row u.
    edges: Option<Vec<Vec<usize>>>,
    /// Dense rows.
    matrix: Option<Vec<Vec<f64>>>,
    /// Sparse triples `[row, column, weight]`; duplicates add up.
    entries: Option<Vec<(usize, usize, f64)>>,
}

impl Problem {
    pub fn from_dense(rows: &[Vec<f64>]) -> anyhow::Result<Self> {
        let n = rows.len();
        let mut a = Vec::with_capacity(n * n);
        for (u, row) in rows.iter().enumerate() {
            if row.len() != n {
                bail!("row {u} has {} entries, expected {n}", row.len());
            }
            a.extend_from_slice(row);
        }
        Self::checked(n, a)
    }

    pub fn from_edges(n: usize, edges: &[Vec<usize>]) -> anyhow::Result<Self> {
        if edges.len() != n {
            bail!("`edges` has {} rows, expected {n}", edges.len());
        }
        let mut a = vec![0.0; n * n];
        for (u, row) in edges.iter().enumerate() {
            for &v in row {
                if v >= n {
                    bail!("edge ({u}, {v}) is out of range for size {n}");
                }
                a[u * n + v] = 1.0;
            }
        }
        Self::checked(n, a)
    }

    fn checked(n: usize, a: Vec<f64>) -> anyhow::Result<Self> {
        if n == 0 {
            bail!("the matrix is empty");
        }
        for (index, &x) in a.iter().enumerate() {
            if !x.is_finite() || x < 0.0 {
                bail!(
                    "entry ({}, {}) = {x} is not a finite nonnegative number",
                    index / n,
                    index % n
                );
            }
        }
        Ok(Problem { n, a })
    }

    pub fn load(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("reading {}", path.display()))?;
        let raw: Raw =
            serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))?;
        let given = raw.edges.is_some() as u8 + raw.matrix.is_some() as u8 + raw.entries.is_some() as u8;
        if given != 1 {
            bail!("expected exactly one of `edges`, `matrix`, `entries`");
        }
        if let Some(rows) = raw.matrix {
            if let Some(size) = raw.size
                && size != rows.len()
            {
                bail!("`size` is {size} but `matrix` has {} rows", rows.len());
            }
            return Self::from_dense(&rows);
        }
        let n = raw.size.context("`size` is required with `edges` or `entries`")?;
        if let Some(edges) = raw.edges {
            return Self::from_edges(n, &edges);
        }
        let mut a = vec![0.0; n * n];
        for (u, v, w) in raw.entries.unwrap() {
            if u >= n || v >= n {
                bail!("entry ({u}, {v}) is out of range for size {n}");
            }
            a[u * n + v] += w;
        }
        Self::checked(n, a)
    }

    pub fn is_zero_one(&self) -> bool {
        self.a.iter().all(|&x| x == 0.0 || x == 1.0)
    }
}
