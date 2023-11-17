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
}
