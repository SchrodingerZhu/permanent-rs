use crate::{cooling_state::State, graph::Match};
use rand::prelude::{IteratorRandom, SliceRandom};

pub(crate) struct Additive;

pub(crate) struct Multiplicative;

pub(crate) struct Constant;

#[derive(Clone, Copy)]
pub struct Proposal {
    u1: usize,
    v1: usize,
    u2: usize,
    v2: usize,
}

pub trait MetropolisFilter {
    type MatchAttr: Send;
    fn ratio(
        attr: &Self::MatchAttr,
        matching: &Match,
        proposal: &Proposal,
        state: &State,
    ) -> (f64, Self::MatchAttr);
    fn initial_attr(matching: &Match, state: &State) -> Self::MatchAttr;
}

impl MetropolisFilter for Constant {
    type MatchAttr = ();
    fn ratio(
        _attr: &Self::MatchAttr,
        _matching: &Match,
        _proposal: &Proposal,
        _state: &State,
    ) -> (f64, Self::MatchAttr) {
        (1.0, ())
    }

    fn initial_attr(matching: &Match, state: &State) -> Self::MatchAttr {
        ()
    }
}

impl MetropolisFilter for Additive {
    type MatchAttr = f64;
    fn ratio(
        attr: &Self::MatchAttr,
        _matching: &Match,
        proposal: &Proposal,
        state: &State,
    ) -> (f64, Self::MatchAttr) {
        let a = state.weight_of_edge(proposal.u1, proposal.v1);
        let b = state.weight_of_edge(proposal.u2, proposal.v2);
        let c = state.weight_of_edge(proposal.u1, proposal.v2);
        let d = state.weight_of_edge(proposal.u2, proposal.v1);
        let new_attr = *attr - a - b + c + d;
        (new_attr / attr * (c + d) / (a + b), new_attr)
    }

    fn initial_attr(matching: &Match, state: &State) -> Self::MatchAttr {
        matching
            .edges
            .iter()
            .map(|x| state.weight_of_edge(x.0, x.1))
            .sum()
    }
}

impl MetropolisFilter for Multiplicative {
    type MatchAttr = f64;
    fn ratio(
        attr: &Self::MatchAttr,
        matching: &Match,
        proposal: &Proposal,
        state: &State,
    ) -> (f64, Self::MatchAttr) {
        let a = state.weight_of_edge(proposal.u1, proposal.v1);
        let b = state.weight_of_edge(proposal.u2, proposal.v2);
        let c = state.weight_of_edge(proposal.u1, proposal.v2);
        let d = state.weight_of_edge(proposal.u2, proposal.v1);
        let mut new_attr = *attr;
        for (i, j) in matching.edges.iter().copied().filter(|(i, j)| {
            (*i != proposal.u1 || *j != proposal.v1) && (*i != proposal.u2 || *j != proposal.v2)
        }) {
            new_attr -= state.weight_of_edge(i, j) * a;
            new_attr -= state.weight_of_edge(i, j) * b;
            new_attr += state.weight_of_edge(i, j) * c;
            new_attr += state.weight_of_edge(i, j) * d;
        }
        new_attr += c * c + d * d - a * a - b * b;
        (new_attr / attr * (c * d) / (a * b), new_attr)
    }

    fn initial_attr(matching: &Match, state: &State) -> Self::MatchAttr {
        // sum of pairwise weight products
        let mut attr = 0.0;
        for (i, j) in matching.edges.iter().copied() {
            for (k, l) in matching.edges.iter().copied() {
                attr += state.weight_of_edge(i, j) * state.weight_of_edge(k, l);
            }
        }
        attr
    }
}

pub struct AugmentedMatch<T: MetropolisFilter> {
    pub matching: Match,
    pub attr: T::MatchAttr,
    pub weight: f64,
    pub active_count: usize,
}

impl<T: MetropolisFilter> AugmentedMatch<T> {
    pub fn choose_weighted_edge(&self, state: &State) -> (usize, usize) {
        let mut rng = rand::thread_rng();
        self.matching
            .edges
            .choose_weighted(&mut rng, |x| state.weight_of_edge(x.0, x.1))
            .copied()
            .expect("failed to choose weighted edge")
    }

    pub fn choose_edge_pairs(&self) -> (usize, usize) {
        let indices = (0..self.matching.edges.len()).choose_multiple(&mut rand::thread_rng(), 2);
        (indices[0], indices[1])
    }
    pub fn transit_n_times(&mut self, state: &State, n: usize) {
        for _ in 0..n {
            self.transit(self.choose_edge_pairs(), state);
        }
    }
    pub fn transit(&mut self, position: (usize, usize), state: &State) -> bool {
        let proposal = Proposal {
            u1: self.matching.edges[position.0].0,
            v1: self.matching.edges[position.0].1,
            u2: self.matching.edges[position.1].0,
            v2: self.matching.edges[position.1].1,
        };
        let (ratio, new_attr) = T::ratio(&self.attr, &self.matching, &proposal, state);
        let next_weight = self.weight
            - state.weight_of_edge(proposal.u1, proposal.v1)
            - state.weight_of_edge(proposal.u2, proposal.v2)
            + state.weight_of_edge(proposal.u1, proposal.v2)
            + state.weight_of_edge(proposal.u2, proposal.v1);
        let next_active_count = self.active_count
            - state.activity_of_edge(proposal.u1, proposal.v1)
            - state.activity_of_edge(proposal.u2, proposal.v2)
            + state.activity_of_edge(proposal.u1, proposal.v2)
            + state.activity_of_edge(proposal.u2, proposal.v1);
        let weight_ratio = next_weight / self.weight;
        let active_ratio =
            (state.beta * (next_active_count as isize - self.active_count as isize) as f64).exp();
        let probability = (ratio * weight_ratio * active_ratio).min(1.0);
        if rand::random::<f64>() < probability {
            self.matching.edges[position.0] = (proposal.u1, proposal.v2);
            self.matching.edges[position.1] = (proposal.u2, proposal.v1);
            self.attr = new_attr;
            self.weight = next_weight;
            self.active_count = next_active_count;
            true
        } else {
            false
        }
    }
}
