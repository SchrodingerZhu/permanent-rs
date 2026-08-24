use crate::{cooling_state::State, graph::Match};
use rand::RngExt;

/// Sentinel partner value marking an unmatched (hole) vertex.
const HOLE: u32 = u32::MAX;

/// State of one Jerrum–Sinclair–Vigoda walker: a perfect matching of the
/// complete bipartite graph K_{n,n}, or a near-perfect matching leaving
/// exactly one row hole and one column hole uncovered.
///
/// The chain proposes a uniformly random vertex pair (u, v) and applies the
/// JSV move set — remove (u,v) from a perfect matching, add (u,v) across the
/// two holes, or slide an edge onto a hole — with a Metropolis filter for the
/// stationary distribution
///
///   pi(M) ∝ e^{-beta * inactive(M)} * w(M),
///
/// where inactive(M) counts edges of M absent from the underlying graph,
/// w(M) = w(x, y) for the hole pair (x, y) of a near-perfect matching, and
/// w(M) = 1 for a perfect matching. With w near its ideal value (the ratio of
/// perfect to per-hole-class near-perfect activity) every hole class and the
/// perfect class carry equal stationary mass, which is the invariant the
/// BSVV/JSV rapid-mixing proof rests on.
pub struct JsvChain {
    /// column matched to each row, or `HOLE`
    row_match: Box<[u32]>,
    /// row matched to each column, or `HOLE`
    col_match: Box<[u32]>,
    /// (row hole, column hole); `None` when the matching is perfect
    hole: Option<(u32, u32)>,
    /// number of matched edges present in the underlying graph
    active_count: u32,
}

impl JsvChain {
    /// Start from a perfect matching (any permutation of columns).
    pub fn from_permutation(matching: &Match, state: &State) -> Self {
        let n = matching.size();
        let mut row_match = vec![HOLE; n].into_boxed_slice();
        let mut col_match = vec![HOLE; n].into_boxed_slice();
        let mut active_count = 0;
        for &(u, v) in matching.edges.iter() {
            row_match[u] = v as u32;
            col_match[v] = u as u32;
            active_count += state.activity_of_edge(u, v) as u32;
        }
        JsvChain {
            row_match,
            col_match,
            hole: None,
            active_count,
        }
    }

    /// The hole pair of a near-perfect state, or `None` for a perfect state.
    pub fn hole(&self) -> Option<(usize, usize)> {
        self.hole.map(|(u, v)| (u as usize, v as usize))
    }

    /// Number of matched edges that do not exist in the underlying graph.
    pub fn inactive_count(&self) -> u32 {
        let size = self.row_match.len() as u32 - self.hole.is_some() as u32;
        size - self.active_count
    }

    /// Perfect and using only real graph edges, i.e. a perfect matching of
    /// the underlying graph itself. `e^{-beta * inactive} = 1` for exactly
    /// these states at every beta, which is what makes
    /// `Z * Pr[fully active perfect]` equal the permanent exactly.
    pub fn is_fully_active_perfect(&self) -> bool {
        self.hole.is_none() && self.active_count as usize == self.row_match.len()
    }

    /// A uniform index in [0, bound) from a u32 draw via multiply-shift
    /// range reduction (bias ~bound/2^32; the proposal distribution over
    /// move slots is what carries the Hastings correction, so any fixed
    /// near-uniform distribution keeps the chain honest).
    fn choose_index(bound: u32, rng: &mut impl RngExt) -> u32 {
        (((rng.random::<u32>() as u64) * bound as u64) >> 32) as u32
    }

    pub fn transit_n_times(&mut self, state: &State, n: usize, rng: &mut impl RngExt) {
        for _ in 0..n {
            self.transit(state, rng);
        }
    }

    fn accept(probability: f64, rng: &mut impl RngExt) -> bool {
        probability >= 1.0 || rng.random::<f64>() < probability
    }

    /// One proposal + Metropolis–Hastings filter. Returns whether the move
    /// was accepted.
    ///
    /// Instead of a uniform vertex pair — of which only n out of n^2 touch a
    /// perfect state at all — the proposal draws uniformly from the *valid*
    /// moves of the current state: a perfect matching offers its n removals,
    /// a near-perfect one offers the add across its holes plus 2(n-1)
    /// slides. The asymmetric menu sizes (n versus 2n-1) enter the
    /// acceptance probability as the Hastings factor q(M'→M)/q(M→M'), so the
    /// stationary distribution is untouched while every proposal does work.
    pub fn transit(&mut self, state: &State, rng: &mut impl RngExt) -> bool {
        let n = self.row_match.len() as u32;
        match self.hole {
            None => {
                // remove one of the n matched edges; reverse move is the add
                // out of a 2n-1 menu, so the Hastings factor is n/(2n-1)
                let u = Self::choose_index(n, rng);
                let v = self.row_match[u as usize];
                let activity = state.activity_of_edge(u as usize, v as usize) as isize;
                // pi ratio: 1/lambda_e(u,v) * w(u,v) = e^{beta(1-A)} w(u,v)
                let probability = state.weight_of_edge(u as usize, v as usize)
                    * state.exp_beta_delta(1 - activity)
                    * (n as f64 / (2 * n - 1) as f64);
                if Self::accept(probability, rng) {
                    self.row_match[u as usize] = HOLE;
                    self.col_match[v as usize] = HOLE;
                    self.hole = Some((u, v));
                    self.active_count -= activity as u32;
                    true
                } else {
                    false
                }
            }
            Some((hole_u, hole_v)) => {
                // menu: slot 0 = add (hole_u, hole_v); slots 1..n = slide the
                // row hole to column c (skipping hole_v); slots n..2n-1 =
                // slide the column hole to row r (skipping hole_u)
                let slot = Self::choose_index(2 * n - 1, rng);
                if slot == 0 {
                    // add: reverse is a removal out of an n-menu, Hastings
                    // factor (2n-1)/n
                    let activity =
                        state.activity_of_edge(hole_u as usize, hole_v as usize) as isize;
                    let probability = state.exp_beta_delta(activity - 1)
                        / state.weight_of_edge(hole_u as usize, hole_v as usize)
                        * ((2 * n - 1) as f64 / n as f64);
                    if Self::accept(probability, rng) {
                        self.row_match[hole_u as usize] = hole_v;
                        self.col_match[hole_v as usize] = hole_u;
                        self.hole = None;
                        self.active_count += activity as u32;
                        true
                    } else {
                        false
                    }
                } else if slot < n {
                    // slide onto the row hole: pick column v != hole_v,
                    // matched to row z; replace (z, v) with (hole_u, v),
                    // holes become (z, hole_v). Reverse is another slide out
                    // of a 2n-1 menu: Hastings factor 1.
                    let v = if slot > hole_v { slot } else { slot - 1 };
                    let z = self.col_match[v as usize];
                    let gained = state.activity_of_edge(hole_u as usize, v as usize) as isize;
                    let lost = state.activity_of_edge(z as usize, v as usize) as isize;
                    let probability = state.exp_beta_delta(gained - lost)
                        * state.weight_of_edge(z as usize, hole_v as usize)
                        / state.weight_of_edge(hole_u as usize, hole_v as usize);
                    if Self::accept(probability, rng) {
                        self.row_match[hole_u as usize] = v;
                        self.col_match[v as usize] = hole_u;
                        self.row_match[z as usize] = HOLE;
                        self.hole = Some((z, hole_v));
                        self.active_count = (self.active_count as isize + gained - lost) as u32;
                        true
                    } else {
                        false
                    }
                } else {
                    // slide onto the column hole: pick row u != hole_u,
                    // matched to column z; replace (u, z) with (u, hole_v),
                    // holes become (hole_u, z). Hastings factor 1.
                    let pick = slot - n;
                    let u = if pick >= hole_u { pick + 1 } else { pick };
                    let z = self.row_match[u as usize];
                    let gained = state.activity_of_edge(u as usize, hole_v as usize) as isize;
                    let lost = state.activity_of_edge(u as usize, z as usize) as isize;
                    let probability = state.exp_beta_delta(gained - lost)
                        * state.weight_of_edge(hole_u as usize, z as usize)
                        / state.weight_of_edge(hole_u as usize, hole_v as usize);
                    if Self::accept(probability, rng) {
                        self.row_match[u as usize] = hole_v;
                        self.col_match[hole_v as usize] = u;
                        self.col_match[z as usize] = HOLE;
                        self.hole = Some((hole_u, z));
                        self.active_count = (self.active_count as isize + gained - lost) as u32;
                        true
                    } else {
                        false
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::graph::Graph;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;
    use std::collections::HashMap;

    /// Exhaustively enumerate every perfect and near-perfect matching of
    /// K_{n,n} with its unnormalized stationary weight, then check that the
    /// walker's empirical occupancy matches. This validates all four move
    /// cases (remove, add, both slides) and their Metropolis ratios at once:
    /// any error in a ratio or an update shifts the whole histogram.
    #[test]
    fn empirical_distribution_matches_stationary_distribution() {
        let n = 3usize;
        // asymmetric adjacency: edges of the underlying graph
        let graph = Graph {
            size: n,
            edges: vec![
                vec![0, 1].into_boxed_slice(),
                vec![1, 2].into_boxed_slice(),
                vec![2].into_boxed_slice(),
            ]
            .into_boxed_slice(),
        };
        let mut state = State::from(&graph);
        state.set_beta(0.7);
        // deliberately non-uniform hole weights
        for u in 0..n {
            for v in 0..n {
                state.weight.set(u, v, 1.0 + (u * 3 + v) as f64 * 0.5);
            }
        }

        // enumerate exact stationary weights, keyed by a canonical state id
        let key = |row_match: &[Option<usize>]| -> Vec<isize> {
            row_match
                .iter()
                .map(|entry| entry.map(|v| v as isize).unwrap_or(-1))
                .collect()
        };
        let mut exact: HashMap<Vec<isize>, f64> = HashMap::new();
        let perms: [[usize; 3]; 6] = [
            [0, 1, 2],
            [0, 2, 1],
            [1, 0, 2],
            [1, 2, 0],
            [2, 0, 1],
            [2, 1, 0],
        ];
        let activity = |u: usize, v: usize| state.activity_of_edge(u, v);
        for perm in perms {
            // perfect matching
            let inactive = (0..n).map(|u| 1 - activity(u, perm[u])).sum::<usize>();
            let weight = (-state.beta() * inactive as f64).exp();
            exact.insert(key(&perm.map(Some)), weight);
            // near-perfect: drop each edge in turn (every near-perfect
            // matching of K_{3,3} arises from a perfect one this way)
            for drop in 0..n {
                let mut row_match = perm.map(Some);
                row_match[drop] = None;
                let inactive = (0..n)
                    .filter(|&u| u != drop)
                    .map(|u| 1 - activity(u, perm[u]))
                    .sum::<usize>();
                let weight = (-state.beta() * inactive as f64).exp()
                    * state.weight_of_edge(drop, perm[drop]);
                exact.insert(key(&row_match), weight);
            }
        }
        let total: f64 = exact.values().sum();

        // run the walker and histogram its states
        let matching = Match {
            edges: (0..n).map(|u| (u, u)).collect(),
        };
        let mut chain = JsvChain::from_permutation(&matching, &state);
        let mut rng = SmallRng::seed_from_u64(42);
        let steps = 4_000_000usize;
        let mut histogram: HashMap<Vec<isize>, usize> = HashMap::new();
        for _ in 0..steps {
            chain.transit(&state, &mut rng);
            let row_match: Vec<Option<usize>> = chain
                .row_match
                .iter()
                .map(|&v| (v != HOLE).then_some(v as usize))
                .collect();
            *histogram.entry(key(&row_match)).or_default() += 1;
        }

        assert_eq!(histogram.len(), exact.len(), "walker missed states");
        for (state_key, weight) in &exact {
            let expected = weight / total;
            let observed = histogram[state_key] as f64 / steps as f64;
            assert!(
                (observed - expected).abs() < 0.15 * expected + 2e-3,
                "state {state_key:?}: expected {expected:.5}, observed {observed:.5}"
            );
        }
    }

    /// The tracked active count and hole bookkeeping must stay consistent
    /// with a from-scratch recomputation across a long random trajectory.
    #[test]
    fn incremental_bookkeeping_stays_consistent() {
        let n = 8usize;
        let graph = Graph {
            size: n,
            edges: (0..n)
                .map(|u| [u, (u + 1) % n].into())
                .collect::<Vec<_>>()
                .into(),
        };
        let mut state = State::from(&graph);
        state.set_beta(0.3);
        let matching = Match {
            edges: (0..n).map(|u| (u, (u * 3 + 1) % n)).collect(),
        };
        let mut chain = JsvChain::from_permutation(&matching, &state);
        let mut rng = SmallRng::seed_from_u64(7);
        for _ in 0..100_000 {
            chain.transit(&state, &mut rng);
            let mut active = 0u32;
            let mut row_holes = vec![];
            for (u, &v) in chain.row_match.iter().enumerate() {
                if v == HOLE {
                    row_holes.push(u as u32);
                } else {
                    assert_eq!(chain.col_match[v as usize], u as u32);
                    active += state.activity_of_edge(u, v as usize) as u32;
                }
            }
            let col_holes: Vec<u32> = (0..n as u32)
                .filter(|&v| chain.col_match[v as usize] == HOLE)
                .collect();
            assert_eq!(active, chain.active_count);
            match chain.hole {
                None => assert!(row_holes.is_empty() && col_holes.is_empty()),
                Some((u, v)) => {
                    assert_eq!(row_holes, vec![u]);
                    assert_eq!(col_holes, vec![v]);
                }
            }
        }
    }
}
