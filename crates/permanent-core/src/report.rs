//! The text report of a run, so that every front end prints the same lines.
//! Each function hands its lines to `say`, one at a time.

use crate::estimator::{Config, Diagnostics, Estimate, Plan, Verdict};
use crate::stats::scientific;

/// The line for one finished round of Phase 2.
pub fn round(estimate: &Estimate) -> String {
    format!(
        "  {:>6} islands   ln(per) = {:.5} ± {:.5}   {:.3e} mutation + {:.3e} sampler events, {:.1} s",
        estimate.islands,
        estimate.ln_permanent,
        estimate.std_error,
        estimate.mutate.events as f64,
        estimate.sampler.events as f64,
        estimate.seconds,
    )
}

/// Phase 1: what it cost, and the checks on the weights it learned.
pub fn phase1(d: &Diagnostics, temperatures: usize, engine: &str, config: &Config, mut say: impl FnMut(String)) {
    say(format!("phase 1 (learn weights, build schedule): {:.1} s", d.phase1_seconds));
    say(format!("  temperatures:                {temperatures}"));
    say(format!("  population:                  {} particles on {engine}", d.learn_particles));
    say(format!(
        "  work:                        {} learning passes, {:.3e} mutation events, {:.3e} FAR steps",
        d.learn_iterations, d.mutate.events as f64, d.far_steps as f64
    ));
    // Phase 1 mutates only when the configuration asks for it
    if d.mutations > 0 {
        say(format!(
            "  mutation (t = {}):         {:.0} events each, {:.3} of the time in perfect matchings",
            config.learn_mutation_time,
            d.mutate.events as f64 / d.mutations as f64,
            config.learn_mutation_time * d.mutations as f64 / d.mutate.time
        ));
    }
    say(format!("  worst first-pass correction: x{:.3}", d.worst_first_correction.exp()));
    say(format!("  weight check (should be ~1): {:.3} to {:.3}", d.check_min, d.check_max));
    if d.learn_capped_chains + d.mutate.over_budget > 0 {
        say(format!(
            "  WARNING: {} capped FAR chains, {} mutations over budget",
            d.learn_capped_chains, d.mutate.over_budget
        ));
    }
}

/// The plan for Phase 2, with the session's target standard error and its
/// interval's normal quantile `z`.
pub fn plan(plan: &Plan, sigma_target: f64, z: f64, mut say: impl FnMut(String)) {
    say("plan:".into());
    say(format!(
        "  segments:                    {} (independent; each starts from fresh sampler draws)",
        plan.segments.len()
    ));
    say(format!("  islands x particles:         {} x {} per segment", plan.islands, plan.particles));
    say(format!(
        "  variance per island:         planned {:.4} (sum of step variances {:.3}, margin 2); bound factor {:.2}",
        plan.island_relvar, plan.relvar_sum, plan.mixing_factor
    ));
    say(format!("  target std error of ln(per): {sigma_target:.4} (interval z = {z:.3})"));
    say(format!("  sampler stage time:          {:.4}", plan.stage_time));
    if plan.predicted_events > 0.0 {
        say(format!("  predicted:                   ~{:.3e} mutation events", plan.predicted_events));
    }
}

/// The statistical interval of an estimate, and the estimate against an exact
/// value when one is known.
#[derive(Debug, Clone, Copy)]
pub struct Accuracy {
    /// Confidence level of the interval, in percent.
    pub level_percent: f64,
    /// ln of the two ends of the interval.
    pub ln_low: f64,
    pub ln_high: f64,
    /// Distance of the upper end from the estimate, relative to it, in percent.
    pub half_width_percent: f64,
    pub against_exact: Option<AgainstExact>,
}

#[derive(Debug, Clone, Copy)]
pub struct AgainstExact {
    pub ln_exact: f64,
    /// Relative error of the estimate, in percent.
    pub error_percent: f64,
    /// The same difference in standard errors; none when the standard error is 0.
    pub std_errors: Option<f64>,
}

impl Accuracy {
    /// For `estimate` from a session whose interval uses the normal quantile `z`.
    pub fn new(estimate: &Estimate, z: f64, config: &Config, ln_exact: Option<f64>) -> Self {
        let half = z * estimate.std_error;
        Accuracy {
            // half of delta is statistical
            level_percent: 100.0 * (1.0 - config.delta / 2.0),
            ln_low: estimate.ln_permanent - half,
            ln_high: estimate.ln_permanent + half,
            half_width_percent: 100.0 * (half.exp() - 1.0),
            against_exact: ln_exact.map(|ln_exact| AgainstExact {
                ln_exact,
                error_percent: 100.0 * ((estimate.ln_permanent - ln_exact).exp() - 1.0),
                std_errors: (estimate.std_error > 0.0).then(|| (estimate.ln_permanent - ln_exact) / estimate.std_error),
            }),
        }
    }
}

/// Phase 2: the checks on the run, then the estimate with its interval and
/// its comparison with the exact value.
pub fn estimate(estimate: &Estimate, plan: &Plan, accuracy: &Accuracy, mut say: impl FnMut(String)) {
    say(format!(
        "  measured variance per island: {:.4} with {} particles (planned {:.4})",
        estimate.island_relvar,
        estimate.particles,
        plan.island_relvar * plan.particles as f64 / estimate.particles as f64
    ));
    say(match estimate.verdict {
        Verdict::Consistent(gap) => format!(
            "  boundary check: carried and fresh populations agree (largest difference {gap:+.2} standard errors)"
        ),
        Verdict::Inconsistent(gap) => format!(
            "  WARNING: carried and fresh populations differ by {gap:+.2} standard errors at a segment \
             boundary; raise the mutation time or use the rigorous sampler"
        ),
        Verdict::Unchecked => "  boundary check: none (one segment)".into(),
    });
    if estimate.mutate.over_budget + estimate.sampler.over_budget > 0 {
        say(format!(
            "  WARNING: {} mutations and {} sampler trials hit their event cap",
            estimate.mutate.over_budget, estimate.sampler.over_budget
        ));
    }
    say(format!("  regularization factor E[g]:  {:.4}", estimate.mean_gap));
    say(format!("permanent ≈ {}", scientific(estimate.ln_permanent)));
    say(format!("  ln(permanent) = {:.6} ± {:.6} (std error)", estimate.ln_permanent, estimate.std_error));
    say(format!(
        "  {:.1}% statistical interval: [{}, {}]  (relative half-width {:.2}%)",
        accuracy.level_percent,
        scientific(accuracy.ln_low),
        scientific(accuracy.ln_high),
        accuracy.half_width_percent
    ));
    if !estimate.met_target {
        say("  WARNING: the measured error did not reach the target within the top-up limit".into());
    }
    if let Some(exact) = accuracy.against_exact {
        let distance = exact.std_errors.map(|z| format!(" ({z:+.2} std errors)")).unwrap_or_default();
        say(format!("exact = {}; relative error {:+.3}%{distance}", scientific(exact.ln_exact), exact.error_percent));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sampler::{MutateCost, SampleCost};

    #[test]
    fn accuracy_omits_std_errors_when_std_error_is_zero() {
        let estimate = Estimate {
            ln_permanent: 2.0,
            std_error: 0.0,
            met_target: true,
            rounds: 1,
            islands: 4,
            particles: 16,
            island_relvar: 0.0,
            mean_gap: 1.0,
            mutate: MutateCost::default(),
            mutations: 0,
            sampler: SampleCost::default(),
            verdict: Verdict::Unchecked,
            seconds: 0.1,
        };
        let accuracy = Accuracy::new(&estimate, 1.96, &Config::default(), Some(2.0));
        assert!(accuracy.against_exact.unwrap().std_errors.is_none());

        let plan = Plan {
            segments: std::iter::once(0..1).collect(),
            particles: 16,
            islands: 4,
            relvar_sum: 0.0,
            mixing_factor: 1.0,
            island_relvar: 0.0,
            particles_bound: 64,
            stage_time: 1.0,
            predicted_events: 0.0,
        };
        let mut lines = Vec::new();
        super::estimate(&estimate, &plan, &accuracy, |line| lines.push(line));
        let exact_line = lines.last().unwrap();
        assert!(exact_line.starts_with("exact = "));
        assert!(!exact_line.contains("std errors"));
    }
}
