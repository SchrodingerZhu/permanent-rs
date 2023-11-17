use crate::{chain_state::State, graph::Match};

struct Additive;
struct Multiplicative;
struct Constant;

struct Proposal {
    u1: usize,
    v1: usize,
    u2: usize,
    v2: usize,
}

trait MetropolisFilter {
    type MatchAttr;
    fn ratio(
        &self,
        attr: &Self::MatchAttr,
        matching: &Match,
        proposal: &Proposal,
        state: &State,
    ) -> (f64, Self::MatchAttr);
}

impl MetropolisFilter for Constant {
    type MatchAttr = ();
    fn ratio(
        &self,
        _attr: &Self::MatchAttr,
        _matching: &Match,
        _proposal: &Proposal,
        _state: &State,
    ) -> (f64, Self::MatchAttr) {
        (1.0, ())
    }
}

impl MetropolisFilter for Additive {
    type MatchAttr = f64;
    fn ratio(
        &self,
        attr: &Self::MatchAttr,
        _matching: &Match,
        proposal: &Proposal,
        state: &State,
    ) -> (f64, Self::MatchAttr) {
        let a = state.weight_of_edge(proposal.u1, proposal.v1);
        let b = state.weight_of_edge(proposal.u2, proposal.v2);
        let c = state.weight_of_edge(proposal.u1, proposal.v2);
        let d = state.weight_of_edge(proposal.u2, proposal.v1);
        ((attr - a - b + c + d) / attr, attr - a - b + c + d)
    }
}

impl MetropolisFilter for Multiplicative {
    type MatchAttr = f64;
    fn ratio(
        &self,
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
        (new_attr / attr, new_attr)
    }
}
