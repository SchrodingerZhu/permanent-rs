//! Perfect-matching sampler by annealed relaxed assignments (arXiv:2609.20717,
//! Lemmas 8, 10, 11, 49).
//!
//! A state assigns every left vertex u either a right vertex or "unassigned"
//! (`NONE`). With cr = sum_v max(d_v - 1, 0) counting collisions and `un` the
//! number of unassigned vertices, the target at activity a is
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

use rand::RngExt;
use rand::rngs::SmallRng;

use crate::tables::Tables;

pub const NONE: u32 = u32::MAX;

/// Roughness the weights are assumed to satisfy (the paper's alpha = 2).
pub const ALPHA: f64 = 2.0;

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
        let penalty = 8.0 * ALPHA * ALPHA * nf * nf;
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

    /// Starting point for the validated mode: a total annealing time of
    /// 8 ln(2n), spread evenly over the stages. The estimator doubles it
    /// until two consecutive settings agree.
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

/// Reusable per-thread working state for one trial.
pub struct Scratch {
    pub sigma: Vec<u32>,
    deg: Vec<u32>,
    /// Wrapping sum of the left vertices assigned to each right vertex; when
    /// exactly one remains this is its index.
    occ: Vec<u32>,
    /// Left vertices, those in a collision first.
    list: Vec<u32>,
    pos: Vec<u32>,
    n_conf: usize,
    unassigned: usize,
}

impl Scratch {
    pub fn new(n: usize) -> Self {
        Scratch {
            sigma: vec![NONE; n],
            deg: vec![0; n],
            occ: vec![0; n],
            list: (0..n as u32).collect(),
            pos: (0..n as u32).collect(),
            n_conf: 0,
            unassigned: n,
        }
    }

    fn reset(&mut self) {
        let n = self.sigma.len();
        self.sigma.fill(NONE);
        self.deg.fill(0);
        self.occ.fill(0);
        self.n_conf = 0;
        self.unassigned = n;
    }

    #[inline]
    fn swap(&mut self, i: usize, j: usize) {
        let (a, b) = (self.list[i], self.list[j]);
        self.list[i] = b;
        self.list[j] = a;
        self.pos[a as usize] = j as u32;
        self.pos[b as usize] = i as u32;
    }

    #[inline]
    fn mark_conflicted(&mut self, u: u32) {
        let p = self.pos[u as usize] as usize;
        if p >= self.n_conf {
            self.swap(p, self.n_conf);
            self.n_conf += 1;
        }
    }

    #[inline]
    fn mark_clear(&mut self, u: u32) {
        let p = self.pos[u as usize] as usize;
        if p < self.n_conf {
            self.n_conf -= 1;
            self.swap(p, self.n_conf);
        }
    }

    #[inline]
    fn remove(&mut self, u: u32) {
        let v = self.sigma[u as usize];
        if v == NONE {
            return;
        }
        let v = v as usize;
        self.sigma[u as usize] = NONE;
        self.unassigned += 1;
        self.deg[v] -= 1;
        self.occ[v] = self.occ[v].wrapping_sub(u);
        if self.deg[v] >= 1 {
            self.mark_clear(u);
            if self.deg[v] == 1 {
                self.mark_clear(self.occ[v]);
            }
        }
    }

    #[inline]
    fn place(&mut self, u: u32, v: usize) {
        self.sigma[u as usize] = v as u32;
        self.unassigned -= 1;
        self.deg[v] += 1;
        if self.deg[v] == 2 {
            self.mark_conflicted(self.occ[v]);
            self.mark_conflicted(u);
        } else if self.deg[v] > 2 {
            self.mark_conflicted(u);
        }
        self.occ[v] = self.occ[v].wrapping_add(u);
    }

    pub fn is_perfect(&self) -> bool {
        self.n_conf == 0 && self.unassigned == 0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Outcome {
    Perfect,
    Imperfect,
    OverBudget,
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

/// One pass over the whole activity schedule from the all-unassigned state.
pub fn run_trial(
    tables: &Tables,
    schedule: &StageSchedule,
    stage_time: f64,
    event_cap: u64,
    scratch: &mut Scratch,
    rng: &mut SmallRng,
) -> (Outcome, u64) {
    let n = tables.n;
    let nf = n as f64;
    scratch.reset();
    let fast = 1.0 + schedule.penalty;
    let mut events = 0u64;
    for &a in &schedule.activities {
        let p_unassign = 1.0 / (1.0 + a);
        let mut t = 0.0;
        loop {
            let nc = scratch.n_conf as f64;
            let conflicted_rate = nc * fast;
            let rate = (nf - nc) + conflicted_rate;
            t += -(1.0 - rng.random::<f64>()).ln() / rate;
            if t > stage_time {
                break;
            }
            events += 1;
            if events > event_cap {
                return (Outcome::OverBudget, events);
            }
            let x = rng.random::<f64>() * rate;
            let slot = if x < conflicted_rate {
                ((x / fast) as usize).min(scratch.n_conf - 1)
            } else {
                (scratch.n_conf + (x - conflicted_rate) as usize).min(n - 1)
            };
            let u = scratch.list[slot];
            scratch.remove(u);
            if rng.random::<f64>() >= p_unassign {
                let v = tables.sample_row(u as usize, rng);
                scratch.place(u, v);
            }
        }
    }
    let outcome = if scratch.is_perfect() { Outcome::Perfect } else { Outcome::Imperfect };
    (outcome, events)
}

/// Repeat trials until one ends on a perfect matching, which is then in
/// `scratch.sigma`. Returns false if `max_trials` all failed.
pub fn sample_perfect(
    tables: &Tables,
    schedule: &StageSchedule,
    stage_time: f64,
    event_cap: u64,
    max_trials: u64,
    scratch: &mut Scratch,
    rng: &mut SmallRng,
    cost: &mut SampleCost,
) -> bool {
    for _ in 0..max_trials {
        let (outcome, events) = run_trial(tables, schedule, stage_time, event_cap, scratch, rng);
        cost.trials += 1;
        cost.events += events;
        match outcome {
            Outcome::Perfect => {
                cost.accepted += 1;
                return true;
            }
            Outcome::Imperfect => {}
            Outcome::OverBudget => cost.over_budget += 1,
        }
    }
    false
}

#[cfg(test)]
pub(crate) mod test {
    use super::*;
    use crate::problem::Problem;
    use crate::scaling::{Prepared, Scaled, prepare};
    use rand::SeedableRng;
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

    /// The accepted outputs must follow lambda(M) / sum lambda, for weights
    /// that are deliberately far from ideal: correctness may not depend on
    /// the weights.
    #[test]
    fn accepted_samples_follow_the_target_distribution() {
        let scaled = test_instance();
        let n = scaled.n;
        let beta = 0.6;
        let log_w = [0.3, -0.4, 1.0, 0.0];
        let tables = Tables::build(&scaled, beta, &log_w, 0);
        let schedule = StageSchedule::new(n, 32.0);
        let stage_time = schedule.rigorous_stage_time(1e-3);
        let cap = schedule.event_cap(n, stage_time);
        let perms = permutations(n);
        let weights: Vec<f64> = perms.iter().map(|p| (-beta * scaled.energy_of(p)).exp()).collect();
        let total: f64 = weights.iter().sum();
        let mut rng = SmallRng::seed_from_u64(5);
        let mut scratch = Scratch::new(n);
        let mut cost = SampleCost::default();
        let mut histogram: HashMap<Vec<u32>, usize> = HashMap::new();
        let draws = 20_000;
        for _ in 0..draws {
            assert!(sample_perfect(&tables, &schedule, stage_time, cap, 64, &mut scratch, &mut rng, &mut cost));
            *histogram.entry(scratch.sigma.clone()).or_default() += 1;
        }
        assert_eq!(cost.over_budget, 0);
        for (perm, weight) in perms.iter().zip(&weights) {
            let expected = weight / total;
            let observed = *histogram.get(perm).unwrap_or(&0) as f64 / draws as f64;
            let sigma = (expected * (1.0 - expected) / draws as f64).sqrt();
            assert!(
                (observed - expected).abs() < 5.0 * sigma + 1e-4,
                "{perm:?}: expected {expected:.5}, observed {observed:.5}"
            );
        }
        // with these rough weights more than a fifth of the trials succeed
        assert!((draws as f64) / (cost.trials as f64) > 0.2);
    }
}
