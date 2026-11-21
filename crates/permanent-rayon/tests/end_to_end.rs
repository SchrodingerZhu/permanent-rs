use permanent_core::comm::Solo;
use permanent_core::estimator::{Config, Session, Verdict};
use permanent_core::exact;
use permanent_core::problem::Problem;
use permanent_core::scaling::{Prepared, prepare};
use permanent_rayon::Rayon;
use rand::rngs::SmallRng;

fn check(problem: &Problem, seed: u64) {
    let exact = exact::ln_permanent(problem).unwrap();
    let Prepared::Scaled(scaled) = prepare(problem) else { panic!("permanent is zero") };
    let config = Config { epsilon: 0.1, delta: 0.05, seed, ..Default::default() };
    let mut session = Session::new(&scaled, &config, Rayon::<f64, SmallRng>::new(&scaled), &Solo).unwrap();
    let levels = session.learn_schedule().unwrap();
    let plan = session.plan(&levels);
    let estimate = session.estimate(&levels, &plan).unwrap();
    assert!(matches!(estimate.verdict, Verdict::Consistent(_)), "{:?}", estimate.verdict);
    assert!(estimate.met_target);
    // 4 standard errors: a fixed-seed regression check, not a coverage test
    let z = (estimate.ln_permanent - exact) / estimate.std_error;
    assert!(z.abs() < 4.0, "estimate {} vs exact {exact} ({z:+.2} se)", estimate.ln_permanent);
}

#[test]
fn weighted_sparse_matrix() {
    let n = 7;
    let mut a = vec![0.0; n * n];
    for u in 0..n {
        for v in 0..n {
            if (u * 5 + v * 3) % 4 != 1 || u == v {
                a[u * n + v] = (((u * 7 + v * 11) % 13) as f64 - 6.0).exp();
            }
        }
    }
    check(&Problem { n, a }, 1);
}

#[test]
fn zero_one_cycle_cover() {
    // rows u adjacent to columns u, u+1, u+3 (mod n)
    let n = 10;
    let mut a = vec![0.0; n * n];
    for u in 0..n {
        for v in [u, (u + 1) % n, (u + 3) % n] {
            a[u * n + v] = 1.0;
        }
    }
    check(&Problem { n, a }, 2);
}

/// The estimate is unbiased for every population size and mutation time,
/// so with both tiny the mean over many independent runs must still be the
/// exact permanent. Averaging in linear space is the point: ln of the
/// estimate is biased, the estimate is not.
#[test]
fn tiny_populations_are_unbiased() {
    let n = 6;
    let mut a = vec![0.0; n * n];
    for u in 0..n {
        for v in 0..n {
            if (u + 2 * v) % 5 != 3 || u == v {
                a[u * n + v] = (((u * 5 + v * 7) % 11) as f64 / 3.0 - 1.5).exp();
            }
        }
    }
    let problem = Problem { n, a };
    let exact = exact::ln_permanent(&problem).unwrap();
    let Prepared::Scaled(scaled) = prepare(&problem) else { panic!("permanent is zero") };
    // islands are the independent runs: 4096 of them, 4 particles each
    let config = Config {
        epsilon: 0.5,
        seed: 3,
        mutation_time: 0.2,
        particles: Some(4),
        islands: Some(4096),
        segments: Some(1),
        ..Default::default()
    };
    let mut session = Session::new(&scaled, &config, Rayon::<f64, SmallRng>::new(&scaled), &Solo).unwrap();
    let levels = session.learn_schedule().unwrap();
    let plan = session.plan(&levels);
    let estimate = session.estimate(&levels, &plan).unwrap();
    let z = (estimate.ln_permanent - exact) / estimate.std_error;
    assert!(z.abs() < 4.0, "estimate {} vs exact {exact} ({z:+.2} se)", estimate.ln_permanent);
    // far from the asymptotic regime, which is what makes this a test of
    // unbiasedness and not of the variance bound
    assert!(estimate.island_relvar > 0.01);
}
