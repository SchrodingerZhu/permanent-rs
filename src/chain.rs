use crate::{cooling_state::State, graph::Match};
use rand::RngExt;

/// A matching augmented with the incrementally-tracked quantities the chain
/// needs: `weight = W(M) = sum of w(e) for e in M` and the number of edges of
/// M that exist in the underlying graph.
///
/// The chain proposes a uniformly random transposition of two matched edges
/// and accepts with probability `(W'/W) * e^{beta * (a' - a)}`, giving the
/// stationary distribution `pi(M) ∝ W(M) e^{beta a(M)}`. The `L/W(M)`
/// rejection sampling in `rejection_sample`, for a shared lower bound `L`,
/// turns samples from `pi` into
/// samples from the Gibbs distribution `e^{beta a(M)}` required by the
/// telescoping-product estimator.
pub struct AugmentedMatch {
    pub matching: Match,
    /// primary accumulator of W(M)
    weight: f64,
    /// Neumaier compensation term: `weight + weight_c` is the correctly
    /// rounded running sum. Weights span [1/n, 1e12], so a plainly-tracked W
    /// loses the small summands entirely whenever the matching transiently
    /// holds a capped-weight edge (ulp(1e12) ~ 2e-4); the compensated update
    /// keeps the drift at one rounding regardless of how often that happens.
    weight_c: f64,
    pub active_count: usize,
}

impl AugmentedMatch {
    pub fn new(matching: Match, weight: f64, active_count: usize) -> Self {
        AugmentedMatch {
            matching,
            weight,
            weight_c: 0.0,
            active_count,
        }
    }

    /// The tracked total weight `W(M)` of the current matching.
    pub fn weight(&self) -> f64 {
        self.weight + self.weight_c
    }

    /// Reset the tracked weight to a freshly computed value.
    pub fn set_weight(&mut self, weight: f64) {
        self.weight = weight;
        self.weight_c = 0.0;
    }

    /// Neumaier compensated add: exact up to a single final rounding.
    fn weight_add(&mut self, v: f64) {
        let t = self.weight + v;
        self.weight_c += if self.weight.abs() >= v.abs() {
            (self.weight - t) + v
        } else {
            (v - t) + self.weight
        };
        self.weight = t;
    }

    /// Sample an edge of the matching with probability proportional to its
    /// weight, reusing the tracked total `W(M)` instead of re-summing.
    pub fn choose_weighted_edge(&self, state: &State, rng: &mut impl RngExt) -> (usize, usize) {
        let mut target = rng.random::<f64>() * self.weight();
        for &(u, v) in self.matching.edges.iter() {
            target -= state.weight_of_edge(u, v);
            if target <= 0.0 {
                return (u, v);
            }
        }
        // float rounding can leave a sliver of `target`; fall back to the
        // last edge
        *self.matching.edges.last().expect("empty matching")
    }

    /// Two distinct random edge positions from a single u64 draw, using
    /// multiply-shift range reduction (bias ~n/2^32, irrelevant here: the
    /// proposal distribution over *positions* is state-independent, so any
    /// fixed distribution with full support keeps the chain reversible — it
    /// only needs to not be adversarially skewed, not perfectly uniform).
    fn choose_edge_pair(&self, rng: &mut impl RngExt) -> (usize, usize) {
        let n = self.matching.edges.len() as u64;
        let r = rng.random::<u64>();
        let i = ((r >> 32) * n) >> 32;
        let j = ((r & 0xffff_ffff) * (n - 1)) >> 32;
        let (i, j) = (i as usize, j as usize);
        (i, if j >= i { j + 1 } else { j })
    }

    pub fn transit_n_times(&mut self, state: &State, n: usize, rng: &mut impl RngExt) {
        for _ in 0..n {
            let pair = self.choose_edge_pair(rng);
            self.transit(pair, state, rng);
        }
    }

    /// Accept the current matching with probability `L/W(M)`, where `L` is a
    /// shared lower bound on every matching's weight. This turns the chain's
    /// stationary distribution `W(M) e^{beta a(M)}` into the Gibbs
    /// distribution `e^{beta a(M)}`. Using the largest cheap lower bound
    /// available avoids needless rejection while preserving that distribution;
    /// the epsilon absorbs float rounding at the boundary.
    pub fn rejection_sample(
        &mut self,
        state: &State,
        weight_lower_bound: f64,
        n: usize,
        rng: &mut impl RngExt,
    ) -> (Option<usize>, usize) {
        let max_attempts = 2 * state.weight.dimension() * state.weight.dimension();
        for attempt in 1..=max_attempts {
            self.transit_n_times(state, n, rng);
            if rng.random::<f64>() < weight_lower_bound / self.weight() + 2.0 * f64::EPSILON {
                return (Some(state.weight.dimension() - self.active_count), attempt);
            }
        }
        (None, max_attempts)
    }

    pub fn transit(
        &mut self,
        position: (usize, usize),
        state: &State,
        rng: &mut impl RngExt,
    ) -> bool {
        let (u1, v1) = self.matching.edges[position.0];
        let (u2, v2) = self.matching.edges[position.1];
        let (a, b) = (state.weight_of_edge(u1, v1), state.weight_of_edge(u2, v2));
        let (c, d) = (state.weight_of_edge(u1, v2), state.weight_of_edge(u2, v1));
        let weight = self.weight();
        let next_weight = weight - a - b + c + d;
        let next_active_count =
            self.active_count - state.activity_of_edge(u1, v1) - state.activity_of_edge(u2, v2)
                + state.activity_of_edge(u1, v2)
                + state.activity_of_edge(u2, v1);
        let weight_ratio = next_weight / weight;
        let active_ratio =
            state.exp_beta_delta(next_active_count as isize - self.active_count as isize);
        let probability = weight_ratio * active_ratio;
        if probability >= 1.0 || rng.random::<f64>() < probability {
            self.matching.edges[position.0] = (u1, v2);
            self.matching.edges[position.1] = (u2, v1);
            self.weight_add(-a);
            self.weight_add(-b);
            self.weight_add(c);
            self.weight_add(d);
            self.active_count = next_active_count;
            true
        } else {
            false
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::graph::Graph;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    /// The compensated running W(M) must agree with a fresh recomputation to
    /// near machine precision even when the weight matrix mixes magnitudes as
    /// far apart as 1/n and the 1e12 cap, i.e. exactly the regime where a
    /// plainly-tracked sum loses the small weights to absorption whenever a
    /// capped edge transits in and out of the matching.
    #[test]
    fn compensated_weight_tracks_exactly() {
        // n chosen so that (smallest weight 1.1/n) / (largest running sum
        // ~ n * 1e12) exceeds 2^53: the plain-sum absorption regime. The 1.1
        // factor keeps the weights non-dyadic so every add actually rounds.
        let n = 512usize;
        let graph = Graph {
            size: n,
            edges: (0..n)
                .map(|u| [u, (u + 1) % n].into())
                .collect::<Vec<_>>()
                .into(),
        };
        let mut state = crate::cooling_state::State::from(&graph);
        // beta = ln(cap)/2: entering a capped non-edge (acceptance ratio
        // ~ cap * e^{-2 beta} = 1) and leaving it again (~ 1/cap * e^{2 beta}
        // = 1) are both routinely accepted, so the tracked W oscillates
        // between ~1 and ~1e12 — the mid-cooling regime where plain
        // summation strands absorption errors in W each round trip.
        state.set_beta(1e12f64.ln() / 2.0);
        // graph edges keep small weights, every non-edge sits at the cap
        for u in 0..n {
            for v in 0..n {
                let w = if v == u || v == (u + 1) % n {
                    1.1 / n as f64
                } else {
                    1e12
                };
                state.weight.set(u, v, w);
            }
        }
        let mut rng = SmallRng::seed_from_u64(7);
        let matching = crate::graph::Match {
            // A deterministic, well-scrambled permutation. Since 97 is odd,
            // it is coprime to this power-of-two n.
            edges: (0..n).map(|u| (u, (u * 97 + 13) % n)).collect(),
        };
        let weight = state.weight_of_match(&matching);
        let active_count = state.active_count_of_match(&matching);
        let mut chain = AugmentedMatch::new(matching, weight, active_count);
        // shadow-track W the naive way (plain f64 sum) for comparison
        let mut plain = chain.weight();
        let mut float_weight = chain.weight() as f32;
        let mut float_compensation = 0.0f32;
        let mut worst: f64 = 0.0;
        let mut worst_plain: f64 = 0.0;
        let mut worst_float = 0.0f64;
        for _ in 0..1_000_000 {
            let pair = {
                let n = chain.matching.edges.len() as u64;
                let r = rng.random::<u64>();
                let i = (((r >> 32) * n) >> 32) as usize;
                let j = (((r & 0xffff_ffff) * (n - 1)) >> 32) as usize;
                (i, if j >= i { j + 1 } else { j })
            };
            let (u1, v1) = chain.matching.edges[pair.0];
            let (u2, v2) = chain.matching.edges[pair.1];
            let (a, b) = (state.weight_of_edge(u1, v1), state.weight_of_edge(u2, v2));
            let (c, d) = (state.weight_of_edge(u1, v2), state.weight_of_edge(u2, v1));
            let delta = c + d - a - b;
            if chain.transit(pair, &state, &mut rng) {
                plain += delta;
                for value in [-a, -b, c, d] {
                    let value = value as f32;
                    let next = float_weight + value;
                    float_compensation += if float_weight.abs() >= value.abs() {
                        (float_weight - next) + value
                    } else {
                        (value - next) + float_weight
                    };
                    float_weight = next;
                }
            }
            let fresh = state.weight_of_match(&chain.matching);
            worst = worst.max(((chain.weight() - fresh) / fresh).abs());
            worst_plain = worst_plain.max(((plain - fresh) / fresh).abs());
            worst_float = worst_float
                .max((((float_weight + float_compensation) as f64 - fresh) / fresh).abs());
        }
        println!(
            "worst relative drift over 1e6 transits: f64 compensated {worst:.3e}, f64 plain {worst_plain:.3e}, f32 compensated {worst_float:.3e}"
        );
        assert!(worst < 1e-12, "tracked W drifted: {worst:.3e}");
        assert!(
            worst_float < 1e-5,
            "compensated f32 shadow drifted: {worst_float:.3e}"
        );
    }
}
