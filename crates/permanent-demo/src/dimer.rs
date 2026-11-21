//! 2D square and 3D cubic lattice dimer model generators and exact references.

use std::collections::HashMap;

use anyhow::{Context, ensure};
use permanent_core::problem::Problem;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

use crate::types::{LatticeBond, ProblemGeometry};
use crate::{BuiltProblem, MAX_N, ProblemMeta, finalize};

/// Shared boundary, disorder, and vacancy parameters for 2D and 3D lattices.
#[derive(Debug, Clone, Copy)]
pub struct LatticeOptions {
    pub periodic: bool,
    pub disorder: f64,
    pub vacancy_rate: f64,
    pub ensure_matching: bool,
    pub seed: u64,
}

impl Default for LatticeOptions {
    fn default() -> Self {
        Self { periodic: false, disorder: 0.0, vacancy_rate: 0.0, ensure_matching: true, seed: 1 }
    }
}

/// Schrijver's lower bound on entropy per site for 6-regular bipartite graphs
/// (3D periodic cubic lattice with unit bond weights): `0.5 * ln(5^5 / 6^4)`.
pub fn schrijver_cubic_lower_bound() -> f64 {
    0.5 * (3125.0_f64 / 1296.0).ln()
}

/// Kasteleyn's closed-form `ln Z` for the open `m x n` square lattice
/// (`m`, `n` even) with bond weights `a` along the side of length `m` and `b`
/// along the side of length `n`.
pub fn kasteleyn_open(m: usize, n: usize, a: f64, b: f64) -> f64 {
    let mut total = 0.0;
    for j in 1..=(m / 2) {
        let cx = (std::f64::consts::PI * j as f64 / (m + 1) as f64).cos();
        for k in 1..=(n / 2) {
            let cy = (std::f64::consts::PI * k as f64 / (n + 1) as f64).cos();
            total += (4.0 * a * a * cx * cx + 4.0 * b * b * cy * cy).ln();
        }
    }
    total
}

/// Exact number of dimer coverings of an open `lx x ly x lz` box with unit
/// weights, by cell-by-cell profile DP (ported from `misc/dimers.py`).
///
/// Requires the two smallest dimensions to satisfy `d0 * d1 <= 16`.
pub fn count_open_box(lx: usize, ly: usize, lz: usize) -> Option<u128> {
    let mut dims = [lx, ly, lz];
    dims.sort_unstable();
    let [a, b, c] = dims;
    let span = a * b;
    if span == 0 || span > 16 || (a * b * c) % 2 != 0 {
        return None;
    }
    let mut states: HashMap<u32, u128> = HashMap::from([(0u32, 1u128)]);
    for cell in 0..(a * b * c) {
        let x = cell % a;
        let y = (cell / a) % b;
        let z = cell / span;
        let mut next: HashMap<u32, u128> = HashMap::with_capacity(states.len());
        for (mask, ways) in states {
            if mask & 1 != 0 {
                *next.entry(mask >> 1).or_default() += ways;
                continue;
            }
            for (ok, bit) in [(x + 1 < a, 1usize), (y + 1 < b, a), (z + 1 < c, span)] {
                if ok && (mask & (1u32 << bit)) == 0 {
                    let m = (mask | (1u32 << bit)) >> 1;
                    *next.entry(m).or_default() += ways;
                }
            }
        }
        states = next;
    }
    Some(states.get(&0).copied().unwrap_or(0))
}

/// Build a 2D square/rectangular lattice dimer model.
pub fn build_2d(rows: usize, cols: usize, weights: [f64; 2], opts: LatticeOptions) -> anyhow::Result<BuiltProblem> {
    validate_lattice_dims(&[rows, cols], opts.periodic)?;
    validate_bond_params(&weights, opts.disorder, opts.vacancy_rate)?;

    let sites = rows * cols;
    let n = sites / 2;
    let mut row_sites = Vec::with_capacity(n);
    let mut odd_index = vec![0u32; sites];
    let mut col_sites = Vec::with_capacity(n);
    for x in 0..rows {
        for y in 0..cols {
            if (x + y) % 2 == 0 {
                row_sites.push([x, y]);
            } else {
                let idx = col_sites.len() as u32;
                odd_index[x * cols + y] = idx;
                col_sites.push([x, y]);
            }
        }
    }

    let protected_axis = if rows.is_multiple_of(2) { 0usize } else { 1usize };
    let mut rng = SmallRng::seed_from_u64(opts.seed);
    let mut a = vec![0.0; n * n];
    let mut bonds = Vec::with_capacity(n * 4);

    for (u, &[x, y]) in row_sites.iter().enumerate() {
        for (axis, (&dim, &base_w)) in [rows, cols].iter().zip(&weights).enumerate() {
            for step in [-1isize, 1isize] {
                let (nx, ny, wrap) = match axis {
                    0 => {
                        let Some((coord, wrap)) = step_coord(x, step, dim, opts.periodic) else { continue };
                        (coord, y, wrap)
                    }
                    _ => {
                        let Some((coord, wrap)) = step_coord(y, step, dim, opts.periodic) else { continue };
                        (x, coord, wrap)
                    }
                };
                let v = odd_index[nx * cols + ny] as usize;
                let protected = opts.ensure_matching && !wrap && is_canonical_pair([x, y], [nx, ny], protected_axis);
                if opts.vacancy_rate > 0.0 && !protected && rng.random::<f64>() < opts.vacancy_rate {
                    continue;
                }
                let mut w = base_w;
                if opts.disorder > 0.0 && w > 0.0 {
                    let jitter = (rng.random::<f64>() * 2.0 - 1.0) * opts.disorder;
                    w *= jitter.exp();
                }
                if w > 0.0 {
                    a[u * n + v] += w;
                    bonds.push(LatticeBond {
                        row: u as u32,
                        col: v as u32,
                        axis: axis as u8,
                        weight: w,
                        periodic_wrap: wrap,
                    });
                }
            }
        }
    }

    let [weight_x, weight_y] = weights;
    let (exact_ln, exact_source) = if !opts.periodic
        && rows.is_multiple_of(2)
        && cols.is_multiple_of(2)
        && opts.disorder == 0.0
        && opts.vacancy_rate == 0.0
    {
        (Some(kasteleyn_open(rows, cols, weight_x, weight_y)), Some("Kasteleyn formula".into()))
    } else {
        (None, None)
    };

    let boundary = if opts.periodic { "periodic" } else { "open" };
    let title = format!("{rows} × {cols} {boundary} square dimer lattice");
    let description = format!(
        "2D {boundary} lattice ({sites} sites, n = {n}); bond weights ({weight_x:.3}, {weight_y:.3}){}{}.",
        if opts.disorder > 0.0 { format!(", disorder ±{:.2}", opts.disorder) } else { String::new() },
        if opts.vacancy_rate > 0.0 { format!(", {:.0}% vacancies", opts.vacancy_rate * 100.0) } else { String::new() },
    );

    finalize(
        Problem { n, a },
        ProblemMeta {
            title,
            description,
            exact_ln,
            exact_source,
            lattice_sites: Some(sites),
            entropy_lower_bound: None,
            geometry: ProblemGeometry::Lattice2d { rows, cols, periodic: opts.periodic, row_sites, col_sites, bonds },
        },
    )
}

/// Build a 3D cubic lattice dimer model.
pub fn build_3d(dims: [usize; 3], weights: [f64; 3], opts: LatticeOptions) -> anyhow::Result<BuiltProblem> {
    validate_lattice_dims(&dims, opts.periodic)?;
    validate_bond_params(&weights, opts.disorder, opts.vacancy_rate)?;

    let [lx, ly, lz] = dims;
    let sites = lx * ly * lz;
    let n = sites / 2;
    let mut row_sites = Vec::with_capacity(n);
    let mut odd_index = vec![0u32; sites];
    let mut col_sites = Vec::with_capacity(n);
    for x in 0..lx {
        for y in 0..ly {
            for z in 0..lz {
                if (x + y + z) % 2 == 0 {
                    row_sites.push([x, y, z]);
                } else {
                    let idx = col_sites.len() as u32;
                    odd_index[(x * ly + y) * lz + z] = idx;
                    col_sites.push([x, y, z]);
                }
            }
        }
    }

    let protected_axis = dims.iter().position(|&d| d.is_multiple_of(2)).unwrap_or(0);
    let mut rng = SmallRng::seed_from_u64(opts.seed);
    let mut a = vec![0.0; n * n];
    let mut bonds = Vec::with_capacity(n * 6);

    for (u, &s) in row_sites.iter().enumerate() {
        for (axis, (&dim, &base_w)) in dims.iter().zip(&weights).enumerate() {
            for step in [-1isize, 1isize] {
                let Some((coord, wrap)) = step_coord(s[axis], step, dim, opts.periodic) else { continue };
                let mut t = s;
                t[axis] = coord;
                let v = odd_index[(t[0] * ly + t[1]) * lz + t[2]] as usize;
                let protected = opts.ensure_matching && !wrap && is_canonical_pair(s, t, protected_axis);
                if opts.vacancy_rate > 0.0 && !protected && rng.random::<f64>() < opts.vacancy_rate {
                    continue;
                }
                let mut w = base_w;
                if opts.disorder > 0.0 && w > 0.0 {
                    let jitter = (rng.random::<f64>() * 2.0 - 1.0) * opts.disorder;
                    w *= jitter.exp();
                }
                if w > 0.0 {
                    a[u * n + v] += w;
                    bonds.push(LatticeBond {
                        row: u as u32,
                        col: v as u32,
                        axis: axis as u8,
                        weight: w,
                        periodic_wrap: wrap,
                    });
                }
            }
        }
    }

    let uniform_unit =
        opts.disorder == 0.0 && opts.vacancy_rate == 0.0 && weights.iter().all(|&w| (w - 1.0).abs() < 1e-12);
    let (exact_ln, exact_source) = if !opts.periodic && uniform_unit {
        let mut sorted = dims;
        sorted.sort_unstable();
        if sorted == [4, 4, 4] {
            // Verified against count_open_box(4, 4, 4) in unit tests.
            (Some(5_051_532_105_f64.ln()), Some("open 4×4×4 profile DP".into()))
        } else if sorted[0] * sorted[1] <= 8 {
            // Instant (< 0.2 ms) for tiny open boxes such as 2×2×2 or 2×4×4.
            match count_open_box(lx, ly, lz) {
                Some(c) if c > 0 => (Some((c as f64).ln()), Some("3D profile DP".into())),
                _ => (None, None),
            }
        } else {
            (None, None)
        }
    } else {
        (None, None)
    };

    let entropy_lower_bound = (opts.periodic && uniform_unit).then(schrijver_cubic_lower_bound);
    let boundary = if opts.periodic { "periodic torus" } else { "open box" };
    let title = format!("{lx} × {ly} × {lz} {boundary} cubic dimer model");
    let description = format!(
        "3D {boundary} ({sites} sites, n = {n}); bond weights ({:.3}, {:.3}, {:.3}){}{}.",
        weights[0],
        weights[1],
        weights[2],
        if opts.disorder > 0.0 { format!(", disorder ±{:.2}", opts.disorder) } else { String::new() },
        if opts.vacancy_rate > 0.0 { format!(", {:.0}% vacancies", opts.vacancy_rate * 100.0) } else { String::new() },
    );

    finalize(
        Problem { n, a },
        ProblemMeta {
            title,
            description,
            exact_ln,
            exact_source,
            lattice_sites: Some(sites),
            entropy_lower_bound,
            geometry: ProblemGeometry::Lattice3d { dims, periodic: opts.periodic, row_sites, col_sites, bonds },
        },
    )
}

fn validate_lattice_dims(dims: &[usize], periodic: bool) -> anyhow::Result<()> {
    let mut sites = 1usize;
    for &d in dims {
        if periodic {
            ensure!(d >= 4 && d.is_multiple_of(2), "periodic lattice dimensions must be even and >= 4 (got {d})");
        } else {
            ensure!(d >= 2, "lattice dimensions must be at least 2 (got {d})");
        }
        sites = sites.checked_mul(d).context("lattice size overflow")?;
    }
    ensure!(sites.is_multiple_of(2), "total number of lattice sites must be even (got {sites})");
    let n = sites / 2;
    ensure!((2..=MAX_N).contains(&n), "matrix size n = {n} must be in 2..={MAX_N}");
    Ok(())
}

fn validate_bond_params(weights: &[f64], disorder: f64, vacancy_rate: f64) -> anyhow::Result<()> {
    for &w in weights {
        ensure!(w.is_finite() && w >= 0.0, "bond weights must be finite and nonnegative (got {w})");
    }
    ensure!(weights.iter().any(|&w| w > 0.0), "at least one bond weight must be positive");
    ensure!(disorder.is_finite() && (0.0..=8.0).contains(&disorder), "disorder must be in [0, 8]");
    ensure!(vacancy_rate.is_finite() && (0.0..=0.8).contains(&vacancy_rate), "vacancy_rate must be in [0, 0.8]");
    Ok(())
}

fn step_coord(coord: usize, step: isize, dim: usize, periodic: bool) -> Option<(usize, bool)> {
    let next = coord as isize + step;
    if next < 0 {
        periodic.then_some((dim - 1, true))
    } else if next >= dim as isize {
        periodic.then_some((0, true))
    } else {
        Some((next as usize, false))
    }
}

fn is_canonical_pair<const D: usize>(a: [usize; D], b: [usize; D], axis: usize) -> bool {
    let lo = a[axis].min(b[axis]);
    let hi = a[axis].max(b[axis]);
    lo.is_multiple_of(2) && hi == lo + 1
}

#[cfg(test)]
mod tests {
    use permanent_core::exact;

    use super::*;

    #[test]
    fn kasteleyn_matches_chessboard_and_anisotropic_exact() {
        let count = kasteleyn_open(8, 8, 1.0, 1.0).exp().round() as u64;
        assert_eq!(count, 12_988_816);

        // Verify orientation of weight_x vs weight_y against exact subset DP on 4x6.
        let built = build_2d(4, 6, [1.0, 0.5], LatticeOptions::default()).unwrap();
        let exact = exact::ln_permanent(&built.problem).unwrap();
        assert!((built.info.exact_ln.unwrap() - exact).abs() < 1e-12);
    }

    #[test]
    fn open_3d_cube_counts_match_exact() {
        assert_eq!(count_open_box(2, 2, 2), Some(9));
        assert_eq!(count_open_box(4, 4, 4), Some(5_051_532_105));

        let cube2 = build_3d([2, 2, 2], [1.0, 1.0, 1.0], LatticeOptions::default()).unwrap();
        assert_eq!(cube2.problem.n, 4);
        let exact2 = exact::ln_permanent(&cube2.problem).unwrap();
        assert!((exact2 - 9.0_f64.ln()).abs() < 1e-12);

        let cube4 = build_3d([4, 4, 4], [1.0, 1.0, 1.0], LatticeOptions::default()).unwrap();
        assert_eq!(cube4.problem.n, 32);
        assert_eq!(cube4.info.lattice_sites, Some(64));
        // An open 4x4x4 cube has 3 * 3 * 4 * 4 = 144 undirected bonds.
        assert_eq!(cube4.info.nnz, 144);
        assert!((cube4.info.exact_ln.unwrap() - 5_051_532_105_f64.ln()).abs() < 1e-12);
    }

    #[test]
    fn disordered_3d_lattice_preserves_matching() {
        let opts = LatticeOptions { periodic: true, disorder: 1.5, vacancy_rate: 0.35, ensure_matching: true, seed: 7 };
        let disordered = build_3d([4, 4, 4], [1.0, 1.0, 0.5], opts).unwrap();
        assert!(disordered.info.has_perfect_matching);
        assert!(disordered.scaled.is_some());
        assert!(disordered.info.exact_ln.is_none());
    }
}
