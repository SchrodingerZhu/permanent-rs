//! Pure-Rust problem generators, exact references, and curated demos matching
//! Part 4 of `permanent-rs-explained` and `misc/dimers.py`.

use std::collections::HashMap;

use anyhow::{Context, bail, ensure};
use permanent_core::exact;
use permanent_core::problem::Problem;
use permanent_core::scaling::{Prepared, prepare};
use permanent_core::stats::scientific;
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::{RngExt, SeedableRng};

use crate::types::{
    BackendKind, GeneratedProblem, LatticeBond, MatrixEntry, PresetInfo, ProblemGeometry, ProblemSpec,
    RandomMatrixFamily, WebRunConfig,
};

/// Upper bound on matrix dimension `n` generated in the browser to avoid
/// exhausting 32-bit Wasm address space.
pub const MAX_BROWSER_N: usize = 1024;

/// Maximum `n` for general subset-DP exact permanent in the browser
/// (`2^20 * 8` bytes = 8 MiB). Closed-form references work at any size.
pub const BROWSER_MAX_EXACT_N: usize = 20;

/// Schrijver's lower bound on entropy per site for 6-regular bipartite graphs
/// (3D periodic cubic lattice with unit bond weights): `0.5 * ln(5^5 / 6^4)`.
pub fn schrijver_cubic_lower_bound() -> f64 {
    0.5 * (3125.0_f64 / 1296.0).ln()
}

/// Built matrix together with metadata and visual geometry.
#[derive(Debug, Clone)]
pub struct BuiltProblem {
    pub problem: Problem,
    pub info: GeneratedProblem,
}

/// Build a [`BuiltProblem`] (matrix + metadata + geometry + Hungarian matching)
/// from a [`ProblemSpec`].
pub fn build_problem(spec: &ProblemSpec) -> anyhow::Result<BuiltProblem> {
    match spec {
        ProblemSpec::Dimer2d {
            rows,
            cols,
            periodic,
            weight_x,
            weight_y,
            disorder,
            vacancy_rate,
            ensure_matching,
            seed,
        } => build_dimer_2d(
            *rows,
            *cols,
            *periodic,
            *weight_x,
            *weight_y,
            *disorder,
            *vacancy_rate,
            *ensure_matching,
            *seed,
        ),
        ProblemSpec::Dimer3d { lx, ly, lz, periodic, weights, disorder, vacancy_rate, ensure_matching, seed } => {
            build_dimer_3d([*lx, *ly, *lz], *periodic, *weights, *disorder, *vacancy_rate, *ensure_matching, *seed)
        }
        ProblemSpec::DisjointCycles { cycle_sizes } => build_disjoint_cycles(cycle_sizes),
        ProblemSpec::BundledPreset { name } => build_bundled_preset(name),
        ProblemSpec::RandomMatrix { n, family, density, log_spread, ensure_matching, seed } => {
            build_random_matrix(*n, *family, *density, *log_spread, *ensure_matching, *seed)
        }
        ProblemSpec::MatrixMarket { text } => build_matrix_market(text),
    }
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

#[allow(clippy::too_many_arguments)]
fn build_dimer_2d(
    rows: usize,
    cols: usize,
    periodic: bool,
    weight_x: f64,
    weight_y: f64,
    disorder: f64,
    vacancy_rate: f64,
    ensure_matching: bool,
    seed: u64,
) -> anyhow::Result<BuiltProblem> {
    validate_lattice_dims(&[rows, cols], periodic)?;
    validate_bond_params(&[weight_x, weight_y], disorder, vacancy_rate)?;

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
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut a = vec![0.0; n * n];
    let mut bonds = Vec::with_capacity(n * 4);
    let weights = [weight_x, weight_y];

    for (u, &[x, y]) in row_sites.iter().enumerate() {
        for (axis, (&dim, &base_w)) in [rows, cols].iter().zip(&weights).enumerate() {
            for step in [-1isize, 1isize] {
                let (nx, ny, wrap) = match axis {
                    0 => {
                        let Some((coord, wrap)) = step_coord(x, step, dim, periodic) else { continue };
                        (coord, y, wrap)
                    }
                    _ => {
                        let Some((coord, wrap)) = step_coord(y, step, dim, periodic) else { continue };
                        (x, coord, wrap)
                    }
                };
                let v = odd_index[nx * cols + ny] as usize;
                let protected = ensure_matching && !wrap && is_canonical_pair([x, y], [nx, ny], protected_axis);
                if vacancy_rate > 0.0 && !protected && rng.random::<f64>() < vacancy_rate {
                    continue;
                }
                let mut w = base_w;
                if disorder > 0.0 && w > 0.0 {
                    let jitter = (rng.random::<f64>() * 2.0 - 1.0) * disorder;
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

    let (exact_ln, exact_source) =
        if !periodic && rows.is_multiple_of(2) && cols.is_multiple_of(2) && disorder == 0.0 && vacancy_rate == 0.0 {
            (Some(kasteleyn_open(rows, cols, weight_x, weight_y)), Some("Kasteleyn formula".into()))
        } else {
            (None, None)
        };

    let boundary = if periodic { "periodic" } else { "open" };
    let title = format!("{rows} × {cols} {boundary} square dimer lattice");
    let description = format!(
        "2D {boundary} lattice ({sites} sites, n = {n}); bond weights ({weight_x:.3}, {weight_y:.3}){}{}.",
        if disorder > 0.0 { format!(", disorder ±{disorder:.2}") } else { String::new() },
        if vacancy_rate > 0.0 { format!(", {:.0}% vacancies", vacancy_rate * 100.0) } else { String::new() },
    );

    finalize_problem(
        Problem { n, a },
        title,
        description,
        exact_ln,
        exact_source,
        Some(sites),
        None,
        ProblemGeometry::Lattice2d { rows, cols, periodic, row_sites, col_sites, bonds },
    )
}

#[allow(clippy::too_many_arguments)]
fn build_dimer_3d(
    dims: [usize; 3],
    periodic: bool,
    weights: [f64; 3],
    disorder: f64,
    vacancy_rate: f64,
    ensure_matching: bool,
    seed: u64,
) -> anyhow::Result<BuiltProblem> {
    validate_lattice_dims(&dims, periodic)?;
    validate_bond_params(&weights, disorder, vacancy_rate)?;

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
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut a = vec![0.0; n * n];
    let mut bonds = Vec::with_capacity(n * 6);

    for (u, &s) in row_sites.iter().enumerate() {
        for (axis, (&dim, &base_w)) in dims.iter().zip(&weights).enumerate() {
            for step in [-1isize, 1isize] {
                let Some((coord, wrap)) = step_coord(s[axis], step, dim, periodic) else { continue };
                let mut t = s;
                t[axis] = coord;
                let v = odd_index[(t[0] * ly + t[1]) * lz + t[2]] as usize;
                let protected = ensure_matching && !wrap && is_canonical_pair(s, t, protected_axis);
                if vacancy_rate > 0.0 && !protected && rng.random::<f64>() < vacancy_rate {
                    continue;
                }
                let mut w = base_w;
                if disorder > 0.0 && w > 0.0 {
                    let jitter = (rng.random::<f64>() * 2.0 - 1.0) * disorder;
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

    let uniform_unit = disorder == 0.0 && vacancy_rate == 0.0 && weights.iter().all(|&w| (w - 1.0).abs() < 1e-12);
    let (exact_ln, exact_source) = if !periodic && uniform_unit {
        let mut sorted = dims;
        sorted.sort_unstable();
        if sorted == [4, 4, 4] {
            // Verified against count_open_box(4, 4, 4) in unit tests (PDF p. 29).
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

    let entropy_lower_bound = (periodic && uniform_unit).then(schrijver_cubic_lower_bound);
    let boundary = if periodic { "periodic torus" } else { "open box" };
    let title = format!("{lx} × {ly} × {lz} {boundary} cubic dimer model");
    let description = format!(
        "3D {boundary} ({sites} sites, n = {n}); bond weights ({:.3}, {:.3}, {:.3}){}{}.",
        weights[0],
        weights[1],
        weights[2],
        if disorder > 0.0 { format!(", disorder ±{disorder:.2}") } else { String::new() },
        if vacancy_rate > 0.0 { format!(", {:.0}% vacancies", vacancy_rate * 100.0) } else { String::new() },
    );

    finalize_problem(
        Problem { n, a },
        title,
        description,
        exact_ln,
        exact_source,
        Some(sites),
        entropy_lower_bound,
        ProblemGeometry::Lattice3d { dims, periodic, row_sites, col_sites, bonds },
    )
}

fn build_disjoint_cycles(cycle_sizes: &[usize]) -> anyhow::Result<BuiltProblem> {
    ensure!(!cycle_sizes.is_empty(), "specify at least one cycle");
    let mut n = 0usize;
    for &size in cycle_sizes {
        ensure!(size >= 2, "each cycle must have at least 2 rows and 2 columns (got {size})");
        n = n.checked_add(size).context("total size overflow")?;
    }
    ensure!(n <= MAX_BROWSER_N, "total dimension n = {n} exceeds browser limit {MAX_BROWSER_N}");

    let mut a = vec![0.0; n * n];
    let mut offset = 0usize;
    for &size in cycle_sizes {
        for i in 0..size {
            let row = offset + i;
            let col_a = offset + i;
            let col_b = offset + (i + 1) % size;
            a[row * n + col_a] = 1.0;
            a[row * n + col_b] = 1.0;
        }
        offset += size;
    }

    let k = cycle_sizes.len();
    let exact_ln = (k as f64) * std::f64::consts::LN_2;
    let title = format!("{k} separate cycles (n = {n})");
    let description = format!(
        "Disjoint 0/1 bipartite cycles of sizes {cycle_sizes:?}. Each cycle has 2 perfect matchings, so per(A) = 2^{k}."
    );

    finalize_problem(
        Problem { n, a },
        title,
        description,
        Some(exact_ln),
        Some(format!("exact 2^{k} disjoint cycles")),
        None,
        None,
        ProblemGeometry::Cycles { cycle_sizes: cycle_sizes.to_vec() },
    )
}

fn build_bundled_preset(name: &str) -> anyhow::Result<BuiltProblem> {
    match name {
        "dimer-8x8" => build_dimer_2d(8, 8, false, 1.0, 0.5, 0.0, 0.0, true, 1),
        "dimer-16x16" => build_dimer_2d(16, 16, false, 1.0, 0.5, 0.0, 0.0, true, 1),
        "dimer-torus-6x6x6" => build_dimer_3d([6, 6, 6], true, [1.0, 1.0, 0.5], 0.0, 0.0, true, 1),
        "weighted-8" => load_bundled_mtx(
            include_bytes!("../../../data/weighted-8.mtx"),
            "data/weighted-8.mtx",
            "8 × 8 weighted matrix with exact reference.",
            None,
            None,
        ),
        "weighted-12" => load_bundled_mtx(
            include_bytes!("../../../data/weighted-12.mtx"),
            "data/weighted-12.mtx",
            "12 × 12 weighted matrix with exact reference.",
            None,
            None,
        ),
        "weighted-16" => load_bundled_mtx(
            include_bytes!("../../../data/weighted-16.mtx"),
            "data/weighted-16.mtx",
            "PDF Example 3: 16 × 16 weighted matrix with 106 nonzeros spanning 0.05 to 17.9.",
            None,
            None,
        ),
        "weighted-20" => load_bundled_mtx(
            include_bytes!("../../../data/weighted-20.mtx"),
            "data/weighted-20.mtx",
            "20 × 20 sparse weighted matrix with exact reference.",
            None,
            None,
        ),
        "weighted-blocks-48" => load_bundled_mtx(
            include_bytes!("../../../data/weighted-blocks-48.mtx"),
            "data/weighted-blocks-48.mtx",
            "48 × 48 weakly coupled weighted blocks.",
            Some(113.25630197721398),
            Some("bundled exact reference".into()),
        ),
        other => bail!("unknown bundled preset: {other}"),
    }
}

fn load_bundled_mtx(
    bytes: &[u8],
    title: &str,
    description: &str,
    known_exact: Option<f64>,
    known_source: Option<String>,
) -> anyhow::Result<BuiltProblem> {
    let problem = Problem::from_matrix_market(bytes).with_context(|| format!("parsing bundled {title}"))?;
    let (exact_ln, exact_source) = match known_exact {
        Some(v) => (Some(v), known_source),
        // Cheap instant exact for small bundled weighted matrices (n <= 16 takes < 2 ms).
        None if problem.n <= 16 => (exact::ln_permanent(&problem), Some("subset DP".into())),
        None => (None, None),
    };
    finalize_problem(
        problem,
        title.into(),
        description.into(),
        exact_ln,
        exact_source,
        None,
        None,
        ProblemGeometry::Matrix,
    )
}

fn build_random_matrix(
    n: usize,
    family: RandomMatrixFamily,
    density: f64,
    log_spread: f64,
    ensure_matching: bool,
    seed: u64,
) -> anyhow::Result<BuiltProblem> {
    ensure!((2..=MAX_BROWSER_N).contains(&n), "n must be in 2..={MAX_BROWSER_N} (got {n})");
    ensure!(density.is_finite() && density > 0.0 && density <= 1.0, "density must be in (0, 1]");
    ensure!(log_spread.is_finite() && (0.0..=15.0).contains(&log_spread), "log_spread must be in [0, 15]");

    let mut rng = SmallRng::seed_from_u64(seed);
    let mut a = vec![0.0; n * n];

    match family {
        RandomMatrixFamily::SparseLognormal => {
            for entry in &mut a {
                if rng.random::<f64>() < density {
                    let x = (rng.random::<f64>() * 2.0 - 1.0) * log_spread;
                    *entry = x.exp();
                }
            }
        }
        RandomMatrixFamily::ZeroOneBipartite => {
            for entry in &mut a {
                if rng.random::<f64>() < density {
                    *entry = 1.0;
                }
            }
        }
        RandomMatrixFamily::NearPermutation => {
            let off_scale = (-log_spread).exp();
            for u in 0..n {
                for v in 0..n {
                    if u == v {
                        a[u * n + v] = (rng.random::<f64>() * 0.5).exp();
                    } else if rng.random::<f64>() < density {
                        a[u * n + v] = off_scale * (rng.random::<f64>() * 2.0 - 1.0).exp();
                    }
                }
            }
        }
        RandomMatrixFamily::WeaklyCoupledBlocks => {
            let block = 4.min(n);
            let couple = (-log_spread).exp();
            for u in 0..n {
                for v in 0..n {
                    let same_block = u / block == v / block;
                    if same_block {
                        if rng.random::<f64>() < density.max(0.6) {
                            a[u * n + v] = (rng.random::<f64>() * 2.0 - 1.0).exp();
                        }
                    } else if rng.random::<f64>() < density * 0.25 {
                        a[u * n + v] = couple * (rng.random::<f64>() * 2.0 - 1.0).exp();
                    }
                }
            }
        }
        RandomMatrixFamily::Uniform => {
            for entry in &mut a {
                if rng.random::<f64>() < density {
                    *entry = 0.05 + 0.95 * rng.random::<f64>();
                }
            }
        }
    }

    if ensure_matching {
        let mut perm: Vec<usize> = (0..n).collect();
        perm.shuffle(&mut rng);
        for (u, &v) in perm.iter().enumerate() {
            if a[u * n + v] == 0.0 {
                a[u * n + v] = match family {
                    RandomMatrixFamily::ZeroOneBipartite => 1.0,
                    RandomMatrixFamily::Uniform => 0.5,
                    _ => ((rng.random::<f64>() * 2.0 - 1.0) * (log_spread * 0.5)).exp(),
                };
            }
        }
    }

    let problem = Problem { n, a };
    let (exact_ln, exact_source) =
        if n <= 16 { (exact::ln_permanent(&problem), Some("subset DP".into())) } else { (None, None) };
    let family_name = match family {
        RandomMatrixFamily::SparseLognormal => "sparse lognormal",
        RandomMatrixFamily::ZeroOneBipartite => "0/1 bipartite",
        RandomMatrixFamily::NearPermutation => "near-permutation",
        RandomMatrixFamily::WeaklyCoupledBlocks => "weakly coupled blocks",
        RandomMatrixFamily::Uniform => "uniform random",
    };
    let title = format!("Random {n} × {n} ({family_name}, seed {seed})");
    let description =
        format!("Generated {family_name} matrix (density {:.0}%, spread ±{log_spread:.1}).", density * 100.0);

    finalize_problem(problem, title, description, exact_ln, exact_source, None, None, ProblemGeometry::Matrix)
}

fn build_matrix_market(text: &str) -> anyhow::Result<BuiltProblem> {
    let problem = Problem::from_matrix_market(text.as_bytes()).context("parsing Matrix Market (.mtx) input")?;
    ensure!(problem.n >= 2, "matrix dimension must be at least 2 (got {})", problem.n);
    ensure!(problem.n <= MAX_BROWSER_N, "matrix dimension n = {} exceeds browser limit {MAX_BROWSER_N}", problem.n);
    let n = problem.n;
    let (exact_ln, exact_source) =
        if n <= 16 { (exact::ln_permanent(&problem), Some("subset DP".into())) } else { (None, None) };
    finalize_problem(
        problem,
        format!("Custom {n} × {n} Matrix Market"),
        "User-supplied Matrix Market coordinate matrix.".into(),
        exact_ln,
        exact_source,
        None,
        None,
        ProblemGeometry::Matrix,
    )
}

#[allow(clippy::too_many_arguments)]
fn finalize_problem(
    problem: Problem,
    title: String,
    description: String,
    exact_ln: Option<f64>,
    exact_source: Option<String>,
    lattice_sites: Option<usize>,
    entropy_lower_bound: Option<f64>,
    geometry: ProblemGeometry,
) -> anyhow::Result<BuiltProblem> {
    let n = problem.n;
    let is_zero_one = problem.is_zero_one();
    let mut triplets = Vec::new();
    for u in 0..n {
        for v in 0..n {
            let value = problem.a[u * n + v];
            if value > 0.0 {
                triplets.push(MatrixEntry { row: u as u32, col: v as u32, value });
            }
        }
    }
    let nnz = triplets.len();
    let (has_perfect_matching, heaviest_matching) = match prepare(&problem) {
        Prepared::Zero => (false, None),
        Prepared::Scaled(scaled) => (true, Some(scaled.matching.into_iter().map(|v| v as u32).collect())),
    };
    let (exact_ln, exact_scientific, exact_source) = if !has_perfect_matching {
        (Some(f64::NEG_INFINITY), Some("0".into()), Some("no perfect matching".into()))
    } else {
        let sci = exact_ln.map(scientific);
        (exact_ln, sci, exact_source)
    };
    // JSON cannot serialize `-Infinity`; represent zero permanent via `has_perfect_matching: false`.
    let json_exact_ln = exact_ln.filter(|x| x.is_finite());

    let info = GeneratedProblem {
        title,
        description,
        n,
        nnz,
        is_zero_one,
        has_perfect_matching,
        heaviest_matching,
        exact_ln: json_exact_ln,
        exact_scientific,
        exact_source,
        lattice_sites,
        entropy_lower_bound,
        geometry,
        triplets,
    };
    Ok(BuiltProblem { problem, info })
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
    ensure!((2..=MAX_BROWSER_N).contains(&n), "matrix size n = {n} must be in 2..={MAX_BROWSER_N}");
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

/// Curated demo catalog covering every worked example in Part 4 of
/// `permanent-rs-explained`.
pub fn list_presets() -> Vec<PresetInfo> {
    vec![
        PresetInfo {
            id: "cubic-open-4x4x4".into(),
            title: "3D Open Cube (4×4×4)".into(),
            subtitle: "64 sites, n = 32, exact 5,051,532,105 coverings".into(),
            pdf_reference: "Part 4, Example 5 (Fig. 14)".into(),
            spec: ProblemSpec::Dimer3d {
                lx: 4,
                ly: 4,
                lz: 4,
                periodic: false,
                weights: [1.0, 1.0, 1.0],
                disorder: 0.0,
                vacancy_rate: 0.0,
                ensure_matching: true,
                seed: 1,
            },
            recommended_config: WebRunConfig {
                backend: BackendKind::Auto,
                epsilon: 0.1,
                delta: 0.05,
                seed: 1,
                ..Default::default()
            },
        },
        PresetInfo {
            id: "cubic-torus-6x6x6".into(),
            title: "3D Periodic Torus (6×6×6)".into(),
            subtitle: "216 sites, n = 108, no closed form (entropy per site)".into(),
            pdf_reference: "Part 4, Example 5 (data/dimer-torus-6x6x6.mtx)".into(),
            spec: ProblemSpec::Dimer3d {
                lx: 6,
                ly: 6,
                lz: 6,
                periodic: true,
                weights: [1.0, 1.0, 0.5],
                disorder: 0.0,
                vacancy_rate: 0.0,
                ensure_matching: true,
                seed: 1,
            },
            recommended_config: WebRunConfig {
                backend: BackendKind::Auto,
                epsilon: 0.1,
                delta: 0.05,
                seed: 1,
                ..Default::default()
            },
        },
        PresetInfo {
            id: "chessboard-8x8".into(),
            title: "Chessboard Domino Tilings (8×8)".into(),
            subtitle: "32×32 matrix, exact 12,988,816 tilings (Kasteleyn)".into(),
            pdf_reference: "Part 4, Example 1 (Fig. 10)".into(),
            spec: ProblemSpec::Dimer2d {
                rows: 8,
                cols: 8,
                periodic: false,
                weight_x: 1.0,
                weight_y: 1.0,
                disorder: 0.0,
                vacancy_rate: 0.0,
                ensure_matching: true,
                seed: 1,
            },
            recommended_config: WebRunConfig {
                backend: BackendKind::Auto,
                epsilon: 0.1,
                delta: 0.05,
                seed: 1,
                ..Default::default()
            },
        },
        PresetInfo {
            id: "square-weighted-16x16".into(),
            title: "Weighted Square Dimers (16×16)".into(),
            subtitle: "128×128 matrix, bond weights 1 & 0.5, Kasteleyn ln Z = 35.018".into(),
            pdf_reference: "Part 4, Example 4 (Fig. 13)".into(),
            spec: ProblemSpec::Dimer2d {
                rows: 16,
                cols: 16,
                periodic: false,
                weight_x: 1.0,
                weight_y: 0.5,
                disorder: 0.0,
                vacancy_rate: 0.0,
                ensure_matching: true,
                seed: 1,
            },
            recommended_config: WebRunConfig {
                backend: BackendKind::Auto,
                epsilon: 0.1,
                delta: 0.05,
                seed: 1,
                ..Default::default()
            },
        },
        PresetInfo {
            id: "four-cycles-64".into(),
            title: "Four Separate Cycles (64×64)".into(),
            subtitle: "4 cycles of 16 rows & cols; exact permanent = 16".into(),
            pdf_reference: "Part 4, Example 2 (Fig. 11)".into(),
            spec: ProblemSpec::DisjointCycles { cycle_sizes: vec![16, 16, 16, 16] },
            recommended_config: WebRunConfig {
                backend: BackendKind::Auto,
                epsilon: 0.1,
                delta: 0.05,
                seed: 1,
                ..Default::default()
            },
        },
        PresetInfo {
            id: "weighted-16".into(),
            title: "Weighted Matrix (16×16)".into(),
            subtitle: "106 nonzeros in [0.05, 17.9]; exact 4.660518 × 10¹⁰".into(),
            pdf_reference: "Part 4, Example 3 (Fig. 12)".into(),
            spec: ProblemSpec::BundledPreset { name: "weighted-16".into() },
            recommended_config: WebRunConfig {
                backend: BackendKind::Auto,
                epsilon: 0.1,
                delta: 0.05,
                seed: 1,
                compute_exact: true,
                ..Default::default()
            },
        },
        PresetInfo {
            id: "random-sparse-16".into(),
            title: "Random Sparse Lognormal (16×16)".into(),
            subtitle: "Generate random matrices with adjustable density & spread".into(),
            pdf_reference: "Part 5 (Against exact permanents)".into(),
            spec: ProblemSpec::RandomMatrix {
                n: 16,
                family: RandomMatrixFamily::SparseLognormal,
                density: 0.45,
                log_spread: 3.0,
                ensure_matching: true,
                seed: 42,
            },
            recommended_config: WebRunConfig {
                backend: BackendKind::Auto,
                epsilon: 0.1,
                delta: 0.05,
                seed: 1,
                compute_exact: true,
                ..Default::default()
            },
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kasteleyn_matches_chessboard_and_anisotropic_exact() {
        let count = kasteleyn_open(8, 8, 1.0, 1.0).exp().round() as u64;
        assert_eq!(count, 12_988_816);

        // Verify orientation of weight_x vs weight_y against exact subset DP on 4x6.
        let built = build_dimer_2d(4, 6, false, 1.0, 0.5, 0.0, 0.0, true, 1).unwrap();
        let exact = exact::ln_permanent(&built.problem).unwrap();
        assert!((built.info.exact_ln.unwrap() - exact).abs() < 1e-12);
    }

    #[test]
    fn open_3d_cube_counts_match_pdf() {
        assert_eq!(count_open_box(2, 2, 2), Some(9));
        assert_eq!(count_open_box(4, 4, 4), Some(5_051_532_105));

        let cube2 = build_dimer_3d([2, 2, 2], false, [1.0, 1.0, 1.0], 0.0, 0.0, true, 1).unwrap();
        assert_eq!(cube2.problem.n, 4);
        let exact2 = exact::ln_permanent(&cube2.problem).unwrap();
        assert!((exact2 - 9.0_f64.ln()).abs() < 1e-12);

        let cube4 = build_dimer_3d([4, 4, 4], false, [1.0, 1.0, 1.0], 0.0, 0.0, true, 1).unwrap();
        assert_eq!(cube4.problem.n, 32);
        assert_eq!(cube4.info.lattice_sites, Some(64));
        // An open 4x4x4 cube has 3 * 3 * 4 * 4 = 144 undirected bonds.
        assert_eq!(cube4.info.nnz, 144);
        assert!((cube4.info.exact_ln.unwrap() - (5_051_532_105.0_f64).ln()).abs() < 1e-12);
    }

    #[test]
    fn bundled_dimers_match_repository_mtx_files() {
        let mtx_8x8 = Problem::from_matrix_market(include_bytes!("../../../data/dimer-8x8.mtx")).unwrap();
        let gen_8x8 = build_bundled_preset("dimer-8x8").unwrap().problem;
        assert_eq!(gen_8x8.n, mtx_8x8.n);
        assert_eq!(gen_8x8.a, mtx_8x8.a);

        let mtx_6x6x6 = Problem::from_matrix_market(include_bytes!("../../../data/dimer-torus-6x6x6.mtx")).unwrap();
        let gen_6x6x6 = build_bundled_preset("dimer-torus-6x6x6").unwrap().problem;
        assert_eq!(gen_6x6x6.n, mtx_6x6x6.n);
        assert_eq!(gen_6x6x6.a, mtx_6x6x6.a);
    }

    #[test]
    fn disjoint_cycles_has_exact_two_to_the_k() {
        let small = build_disjoint_cycles(&[3, 4, 5]).unwrap();
        assert_eq!(small.problem.n, 12);
        let exact = exact::ln_permanent(&small.problem).unwrap();
        assert!((exact - 8.0_f64.ln()).abs() < 1e-12);
        assert!((small.info.exact_ln.unwrap() - exact).abs() < 1e-12);

        let pdf_ex2 = build_disjoint_cycles(&[16, 16, 16, 16]).unwrap();
        assert_eq!(pdf_ex2.problem.n, 64);
        assert_eq!(pdf_ex2.info.nnz, 128);
        assert!((pdf_ex2.info.exact_ln.unwrap() - 16.0_f64.ln()).abs() < 1e-12);
    }

    #[test]
    fn random_matrices_and_disordered_lattices_respect_invariants() {
        for family in [
            RandomMatrixFamily::SparseLognormal,
            RandomMatrixFamily::ZeroOneBipartite,
            RandomMatrixFamily::NearPermutation,
            RandomMatrixFamily::WeaklyCoupledBlocks,
            RandomMatrixFamily::Uniform,
        ] {
            let built = build_random_matrix(12, family, 0.4, 3.0, true, 99).unwrap();
            assert_eq!(built.problem.n, 12);
            assert!(built.info.has_perfect_matching);
            assert!(built.problem.a.iter().all(|&x| x.is_finite() && x >= 0.0));
        }

        let disordered = build_dimer_3d([4, 4, 4], true, [1.0, 1.0, 0.5], 1.5, 0.35, true, 7).unwrap();
        assert!(disordered.info.has_perfect_matching);
        assert!(disordered.info.exact_ln.is_none());
    }

    #[test]
    fn all_curated_presets_build_cleanly() {
        for preset in list_presets() {
            let built = build_problem(&preset.spec).unwrap_or_else(|e| panic!("preset {}: {e:#}", preset.id));
            assert!(built.info.has_perfect_matching, "preset {} has no matching", preset.id);
        }
    }
}
