//! Bundled repository matrices, Matrix Market parser, and curated demo catalog.

use anyhow::{Context, bail, ensure};
use permanent_core::exact;
use permanent_core::problem::Problem;

use crate::dimer::{self, LatticeOptions};
use crate::types::{DemoPreset, ProblemGeometry, ProblemSpec, RandomMatrixFamily};
use crate::{BuiltProblem, INSTANT_EXACT_N, MAX_N, ProblemMeta, finalize};

/// Build one of the bundled repository presets by name.
pub fn build_bundled(name: &str) -> anyhow::Result<BuiltProblem> {
    match name {
        "dimer-8x8" => dimer::build_2d(8, 8, [1.0, 0.5], LatticeOptions::default()),
        "dimer-16x16" => dimer::build_2d(16, 16, [1.0, 0.5], LatticeOptions::default()),
        "dimer-torus-6x6x6" => {
            dimer::build_3d([6, 6, 6], [1.0, 1.0, 0.5], LatticeOptions { periodic: true, ..Default::default() })
        }
        "weighted-8" => load_bundled_mtx(
            include_bytes!("../../../data/weighted-8.mtx"),
            "8 × 8 weighted matrix",
            "8 × 8 weighted nonnegative matrix with exact reference.",
            None,
            None,
        ),
        "weighted-12" => load_bundled_mtx(
            include_bytes!("../../../data/weighted-12.mtx"),
            "12 × 12 weighted matrix",
            "12 × 12 weighted nonnegative matrix with exact reference.",
            None,
            None,
        ),
        "weighted-16" => load_bundled_mtx(
            include_bytes!("../../../data/weighted-16.mtx"),
            "16 × 16 weighted matrix",
            "16 × 16 weighted matrix with 106 nonzeros spanning 0.05 to 17.9.",
            None,
            None,
        ),
        "weighted-20" => load_bundled_mtx(
            include_bytes!("../../../data/weighted-20.mtx"),
            "20 × 20 sparse weighted matrix",
            "20 × 20 sparse weighted matrix with exact reference.",
            None,
            None,
        ),
        "weighted-blocks-48" => load_bundled_mtx(
            include_bytes!("../../../data/weighted-blocks-48.mtx"),
            "48 × 48 weakly coupled weighted blocks",
            "48 × 48 weakly coupled weighted blocks.",
            Some(113.25630197721398),
            Some("exact reference".into()),
        ),
        other => bail!("unknown bundled preset: {other}"),
    }
}

/// Parse user-supplied Matrix Market coordinate text into a [`BuiltProblem`].
pub fn from_matrix_market(text: &str) -> anyhow::Result<BuiltProblem> {
    let problem = Problem::from_matrix_market(text.as_bytes()).context("parsing Matrix Market (.mtx) input")?;
    ensure!(problem.n >= 2, "matrix dimension must be at least 2 (got {})", problem.n);
    ensure!(problem.n <= MAX_N, "matrix dimension n = {} exceeds limit {MAX_N}", problem.n);
    let n = problem.n;
    let (exact_ln, exact_source) =
        if n <= INSTANT_EXACT_N { (exact::ln_permanent(&problem), Some("subset DP".into())) } else { (None, None) };
    finalize(
        problem,
        ProblemMeta {
            title: format!("Custom {n} × {n} matrix"),
            description: "User-supplied Matrix Market coordinate matrix.".into(),
            exact_ln,
            exact_source,
            lattice_sites: None,
            entropy_lower_bound: None,
            geometry: ProblemGeometry::Matrix,
        },
    )
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
        None if problem.n <= INSTANT_EXACT_N => (exact::ln_permanent(&problem), Some("subset DP".into())),
        None => (None, None),
    };
    finalize(
        problem,
        ProblemMeta {
            title: title.into(),
            description: description.into(),
            exact_ln,
            exact_source,
            lattice_sites: None,
            entropy_lower_bound: None,
            geometry: ProblemGeometry::Matrix,
        },
    )
}

/// The curated demo catalog.
pub fn catalog() -> Vec<DemoPreset> {
    vec![
        DemoPreset {
            id: "cubic-open-4x4x4".into(),
            title: "3D Open Cube (4×4×4)".into(),
            subtitle: "64 sites, n = 32, exact 5,051,532,105 coverings".into(),
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
            compute_exact: false,
        },
        DemoPreset {
            id: "cubic-torus-6x6x6".into(),
            title: "3D Periodic Torus (6×6×6)".into(),
            subtitle: "216 sites, n = 108, no closed form (entropy per site)".into(),
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
            compute_exact: false,
        },
        DemoPreset {
            id: "chessboard-8x8".into(),
            title: "Chessboard Domino Tilings (8×8)".into(),
            subtitle: "32×32 matrix, exact 12,988,816 tilings (Kasteleyn)".into(),
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
            compute_exact: false,
        },
        DemoPreset {
            id: "square-weighted-16x16".into(),
            title: "Weighted Square Dimers (16×16)".into(),
            subtitle: "128×128 matrix, bond weights 1 & 0.5, Kasteleyn ln Z = 35.018".into(),
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
            compute_exact: false,
        },
        DemoPreset {
            id: "four-cycles-64".into(),
            title: "Four Separate Cycles (64×64)".into(),
            subtitle: "4 cycles of 16 rows & cols; exact permanent = 16".into(),
            spec: ProblemSpec::DisjointCycles { cycle_sizes: vec![16, 16, 16, 16] },
            compute_exact: false,
        },
        DemoPreset {
            id: "weighted-16".into(),
            title: "Weighted Matrix (16×16)".into(),
            subtitle: "106 nonzeros in [0.05, 17.9]; exact 4.660518 × 10¹⁰".into(),
            spec: ProblemSpec::BundledPreset { name: "weighted-16".into() },
            compute_exact: true,
        },
        DemoPreset {
            id: "random-sparse-16".into(),
            title: "Random Sparse Lognormal (16×16)".into(),
            subtitle: "Generate random matrices with adjustable density & spread".into(),
            spec: ProblemSpec::RandomMatrix {
                n: 16,
                family: RandomMatrixFamily::SparseLognormal,
                density: 0.45,
                log_spread: 3.0,
                ensure_matching: true,
                seed: 42,
            },
            compute_exact: true,
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::build_problem;

    #[test]
    fn bundled_dimers_match_repository_mtx_files() {
        let mtx_8x8 = Problem::from_matrix_market(include_bytes!("../../../data/dimer-8x8.mtx")).unwrap();
        let gen_8x8 = build_bundled("dimer-8x8").unwrap().problem;
        assert_eq!(gen_8x8.n, mtx_8x8.n);
        assert_eq!(gen_8x8.a, mtx_8x8.a);

        let mtx_6x6x6 = Problem::from_matrix_market(include_bytes!("../../../data/dimer-torus-6x6x6.mtx")).unwrap();
        let gen_6x6x6 = build_bundled("dimer-torus-6x6x6").unwrap().problem;
        assert_eq!(gen_6x6x6.n, mtx_6x6x6.n);
        assert_eq!(gen_6x6x6.a, mtx_6x6x6.a);
    }

    #[test]
    fn all_curated_presets_build_cleanly() {
        for preset in catalog() {
            let built = build_problem(&preset.spec).unwrap_or_else(|e| panic!("preset {}: {e:#}", preset.id));
            assert!(built.info.has_perfect_matching, "preset {} has no matching", preset.id);
        }
    }
}
