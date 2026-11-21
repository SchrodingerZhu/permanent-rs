//! Relaxed-assignment dynamics (arXiv:2609.20717, Section 4), used two ways:
//! an annealing `trial` draws a perfect matching from nothing by annealing
//! the activity, and `mutate` moves an existing one at the top activity.
//!
//! A state assigns every left vertex u either a right vertex or
//! "unassigned". With cr = sum_v max(d_v - 1, 0) counting collisions and
//! `un` the number of unassigned vertices, the target at activity a is
//!
//!   nu_a(sigma) ∝ a^(n - un) (1 + J)^(-cr) prod_{u assigned} B(u, sigma_u).
//!
//! Perfect matchings are exactly the states with un = cr = 0, and nu_a
//! restricted to them is proportional to B(M), i.e. to lambda(M). So running
//! to the last activity and keeping the outcome only when it is a perfect
//! matching samples the target distribution; rejection never biases it.
//!
//! The dynamics is continuous-time Glauber with state-dependent rates: a
//! vertex in a collision rings at rate 1 + J, any other at rate 1, and on a
//! ring it is redrawn from q(*) = 1/(1+a), q(v) = a B(u,v)/(1+a). There is
//! no accept/reject step; reversibility follows because leaving a collision
//! removes one (1+J)^-1 penalty and the faster clock pays for it exactly.
//! The paper proves spectral gap >= 1 for every a, so the time to spend at
//! each activity is an explicit logarithm, not a guessed mixing time.
//!
//! `mutate` runs the dynamics at a = J from a perfect matching until the
//! time spent in perfect matchings reaches t. Cutting the excursions out of
//! a reversible process leaves a process on the perfect matchings that is
//! reversible for the restricted law mu, with a spectral gap no smaller
//! than the full one. So the kernel K_t that `mutate` applies satisfies
//!
//!   mu K_t = mu,   ||K_t f - mu f||_2 <= exp(-t) ||f - mu f||_2
//!
//! for every f and any anchor weights. A population carried from one
//! temperature to the next only needs `mutate`; trials are what start a
//! population at a temperature of its own (see `estimator.rs`).
//!
//! `trial` and `mutate` themselves are in `chain/dynamics.rs`; here are the
//! activity schedule and the caps they run under.

use num_traits::PrimInt;

use crate::tables::fraction;

/// Roughness the weights are assumed to satisfy (the paper's alpha = 2).
pub const ALPHA: f64 = 2.0;

/// The collision penalty J = J_alpha n^2 with J_alpha = 8 alpha^2, which is
/// also the top activity.
pub fn penalty(n: usize) -> f64 {
    8.0 * ALPHA * ALPHA * (n * n) as f64
}

/// Activities a_0 = 1/n < ... < a_m = J and the collision penalty J.
#[derive(Debug, Clone)]
pub struct StageSchedule {
    pub activities: Vec<f64>,
    pub penalty: f64,
}

impl StageSchedule {
    /// Lemma 10: a_{k+1} = min(J, (1 + 1/q(a_k)) a_k), where q(a) is the
    /// least power of two with q^2 >= 16 K (1 + min(n, n / sqrt(a))).
    ///
    /// J = J_alpha n^2 with J_alpha = 8 alpha^2. K = max(A_alpha, J_alpha);
    /// the paper does not give A_alpha explicitly, so `density` is exposed
    /// and defaults to J_alpha. Larger is more conservative.
    pub fn new(n: usize, density: f64) -> Self {
        let nf = n as f64;
        let penalty = penalty(n);
        let mut activities = vec![1.0 / nf];
        loop {
            let a = *activities.last().unwrap();
            if a >= penalty {
                break;
            }
            let need = 16.0 * density * (1.0 + nf.min(nf / a.sqrt()));
            let mut q = 1.0f64;
            while q * q < need {
                q *= 2.0;
            }
            activities.push(penalty.min(a * (1.0 + 1.0 / q)));
        }
        StageSchedule { activities, penalty }
    }

    /// 1 + J, the rate at which a row in a collision rings.
    pub fn fast(&self) -> u64 {
        1 + self.penalty as u64
    }

    /// For each stage, the probability 1 / (1 + a) that a ringing row stays
    /// unmatched, as a fraction of the counter's range.
    pub fn unassign<C: PrimInt>(&self) -> Vec<C> {
        self.activities.iter().map(|a| fraction(1.0 / (1.0 + a))).collect()
    }

    pub fn stages(&self) -> usize {
        self.activities.len()
    }

    /// Time per stage so that a sample is within total variation `eta` of
    /// the target: each stage contributes eta / (8 stages), and a gap of 1
    /// from a 3-warm start needs T = ln(2 / that). This is the paper's
    /// value; it is safe and, in practice, hundreds of times longer than
    /// needed.
    pub fn rigorous_stage_time(&self, eta: f64) -> f64 {
        (16.0 * self.stages() as f64 / eta).ln()
    }

    /// Unit of the default (not rigorous) setting: a total annealing time
    /// of 8 ln(2n), spread evenly over the stages.
    pub fn initial_stage_time(&self, n: usize) -> f64 {
        8.0 * (2.0 * n as f64).ln() / self.stages() as f64
    }

    /// Event cap for one trial. The paper bounds the events of stage k by
    /// C n (1 + sqrt(a_k)) (T + ln 2n)^2 with an unspecified C; this uses
    /// that shape with a generous constant. Hitting it fails the trial.
    pub fn event_cap(&self, n: usize, stage_time: f64) -> u64 {
        let nf = n as f64;
        let per = (stage_time + (2.0 * nf).ln()).powi(2);
        let sum: f64 = self.activities.iter().map(|a| 1.0 + a.sqrt()).sum();
        (64.0 * nf * sum * per) as u64 + 1_000_000
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct SampleCost {
    pub trials: u64,
    pub accepted: u64,
    pub events: u64,
    pub over_budget: u64,
}

impl SampleCost {
    pub fn add(&mut self, other: SampleCost) {
        self.trials += other.trials;
        self.accepted += other.accepted;
        self.events += other.events;
        self.over_budget += other.over_budget;
    }
}

/// Event cap of one `mutate` call. The paper bounds the events of time T at
/// activity a by C n (1 + sqrt(a)) (T + ln 2n)^2 with an unspecified C; this
/// uses that shape with a generous constant. Hitting it fails the call.
pub fn mutate_event_cap(n: usize, perfect_time: f64) -> u64 {
    let nf = n as f64;
    let per = (2.0 * perfect_time + (2.0 * nf).ln()).powi(2);
    (64.0 * nf * (1.0 + penalty(n).sqrt()) * per) as u64 + 1_000_000
}

/// What one `mutate` call cost.
#[derive(Debug, Clone, Copy, Default)]
pub struct MutateCost {
    pub events: u64,
    /// Elapsed chain time, excursions included; perfect_time / time
    /// estimates nu_J(perfect).
    pub time: f64,
    /// Calls that hit the event cap; their particle is left unchanged.
    pub over_budget: u64,
}

impl MutateCost {
    pub fn add(&mut self, other: MutateCost) {
        self.events += other.events;
        self.time += other.time;
        self.over_budget += other.over_budget;
    }
}

#[cfg(test)]
pub(crate) mod test {
    use super::*;
    use crate::problem::Problem;
    use crate::scaling::{Prepared, Scaled, prepare};
    use crate::chain::dynamics::{STAGE_TICKS, mutate, trial};
    use crate::chain::lane::Lane;
    use crate::chain::matching::{Matching, words};
    use crate::chain::stream::Stream;
    use crate::stats::mix;
    use crate::tables::Tables;
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};
    use std::collections::HashMap;

    pub fn permutations(n: usize) -> Vec<Vec<u32>> {
        fn rec(n: usize, cur: &mut Vec<u32>, used: &mut Vec<bool>, out: &mut Vec<Vec<u32>>) {
            if cur.len() == n {
                out.push(cur.clone());
                return;
            }
            for v in 0..n {
                if !used[v] {
                    used[v] = true;
                    cur.push(v as u32);
                    rec(n, cur, used, out);
                    cur.pop();
                    used[v] = false;
                }
            }
        }
        let mut out = vec![];
        rec(n, &mut vec![], &mut vec![false; n], &mut out);
        out
    }

    pub fn test_instance() -> Scaled {
        let p = Problem {
            n: 4,
            a: vec![
                3.0, 1.0, 0.0, 0.5, //
                0.2, 2.0, 1.0, 0.0, //
                0.0, 0.7, 4.0, 1.0, //
                1.0, 0.0, 0.3, 2.5,
            ],
        };
        match prepare(&p) {
            Prepared::Scaled(s) => s,
            Prepared::Zero => unreachable!(),
        }
    }

    fn assert_follows(histogram: &HashMap<Vec<u16>, usize>, perms: &[Vec<u32>], weights: &[f64], draws: usize) {
        let total: f64 = weights.iter().sum();
        for (perm, weight) in perms.iter().zip(weights) {
            let key: Vec<u16> = perm.iter().map(|&v| v as u16).collect();
            let expected = weight / total;
            let observed = *histogram.get(&key).unwrap_or(&0) as f64 / draws as f64;
            let sigma = (expected * (1.0 - expected) / draws as f64).sqrt();
            assert!(
                (observed - expected).abs() < 5.0 * sigma + 1e-4,
                "{perm:?}: expected {expected:.5}, observed {observed:.5}"
            );
        }
    }

    /// The accepted outputs must follow lambda(M) / sum lambda, for weights
    /// that are deliberately far from ideal: correctness may not depend on
    /// the weights.
    #[test]
    fn accepted_samples_follow_the_target_distribution() {
        let scaled = test_instance();
        let n = scaled.n;
        let beta = 0.6;
        let log_w = [0.3, -0.4, 1.0, 0.0];
        let tables = Tables::<u64>::build(&scaled, beta, &log_w, 0);
        let schedule = StageSchedule::new(n, 32.0);
        let stage_time = schedule.rigorous_stage_time(1e-3);
        let cap = schedule.event_cap(n, stage_time);
        let unassign = schedule.unassign::<u64>();
        let ticks_per_time = STAGE_TICKS as f64 / stage_time;
        let perms = permutations(n);
        let weights: Vec<f64> = perms.iter().map(|p| (-beta * scaled.energy_of(p)).exp()).collect();
        let mut memory = vec![0u16; words(n)];
        let mut matching = Matching::new(Lane::on(&mut memory, n));
        let mut histogram: HashMap<Vec<u16>, usize> = HashMap::new();
        let draws = 20_000;
        for draw in 0..draws as u64 {
            let mut accepted = false;
            for attempt in 0..64 {
                let mut rng = Stream::new(mix(mix(draw).wrapping_add(attempt)));
                // SAFETY: the lane holds words(n) words and the tables are n x n
                let outcome = unsafe {
                    trial(&mut matching, n as u32, &tables.packed_rows, &unassign, ticks_per_time, schedule.fast(), cap, &mut rng)
                };
                assert!(!outcome.over_budget);
                if matching.is_perfect() {
                    accepted = true;
                    break;
                }
            }
            assert!(accepted);
            let mut sample = vec![0u16; n];
            unsafe { matching.store(sample.as_mut_ptr()) };
            *histogram.entry(sample).or_default() += 1;
        }
        assert_follows(&histogram, &perms, &weights, draws);
    }

    /// Apply the mutation kernel to `draws` starts chosen by `start`, in
    /// the real type `T`, and compare the outputs with the target.
    fn check<T>(perfect_time: f64, draws: usize, mut start: impl FnMut(&mut SmallRng, &[f64]) -> usize)
    where
        T: num_traits::Float,
    {
        let scaled = test_instance();
        let n = scaled.n;
        let beta = 0.6;
        let log_w = [0.3, -0.4, 1.0, 0.0];
        let tables = Tables::<u64>::build(&scaled, beta, &log_w, 0);
        let cap = mutate_event_cap(n, perfect_time);
        let perms = permutations(n);
        let weights: Vec<f64> = perms.iter().map(|p| (-beta * scaled.energy_of(p)).exp()).collect();
        let fast = 1 + penalty(n) as u64;
        let unassign_below: u64 = fraction(1.0 / (1.0 + penalty(n)));
        let mut pick_rng = SmallRng::seed_from_u64(5);
        let mut memory = vec![0u16; words(n)];
        let mut matching = Matching::new(Lane::on(&mut memory, n));
        let mut histogram: HashMap<Vec<u16>, usize> = HashMap::new();
        for draw in 0..draws as u64 {
            let mut sample: Vec<u16> = perms[start(&mut pick_rng, &weights)].iter().map(|&v| v as u16).collect();
            let mut rng = Stream::new(mix(draw));
            // SAFETY: `sample` is a permutation of n, the lane holds
            // words(n) words and the tables are n x n
            let moved = unsafe {
                matching.load(sample.as_ptr());
                mutate::<u64, T, _>(
                    &mut matching,
                    n as u32,
                    &tables.packed_rows,
                    T::from(perfect_time).unwrap(),
                    fast,
                    unassign_below,
                    cap,
                    &mut rng,
                )
            };
            assert!(!moved.over_budget && matching.is_perfect());
            unsafe { matching.store(sample.as_mut_ptr()) };
            *histogram.entry(sample).or_default() += 1;
        }
        assert_follows(&histogram, &perms, &weights, draws);
    }

    /// K_t forgets its start: from one fixed matching, a long run lands on
    /// the target distribution.
    #[test]
    fn a_long_run_reaches_the_target_distribution() {
        check::<f64>(24.0, 40_000, |_, _| 0);
    }

    /// K_t leaves mu invariant for every t, however short: starting from
    /// exact samples, the output is still an exact sample. In single
    /// precision too, where only the waiting times are rounded.
    #[test]
    fn a_short_run_keeps_the_target_distribution() {
        let exact = |rng: &mut SmallRng, weights: &[f64]| {
            let total: f64 = weights.iter().sum();
            let mut x = rng.random::<f64>() * total;
            for (i, w) in weights.iter().enumerate() {
                if x < *w {
                    return i;
                }
                x -= w;
            }
            weights.len() - 1
        };
        check::<f64>(0.3, 200_000, exact);
        check::<f32>(0.3, 200_000, exact);
    }
}
