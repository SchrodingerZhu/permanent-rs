#!/usr/bin/env python3
"""Generate random bipartite graphs in the sampler's JSON format.

The model is G(n, n, p) with a planted random perfect matching: every
left vertex u is first matched to pi(u) for a uniformly random
permutation pi (so a perfect matching always exists and `find_match`
succeeds), then every other pair (u, v) is added independently with
probability p = (d - 1) / n, giving average degree ~d. The planted
permutation is random, so it is statistically indistinguishable from
the other edges.

Usage: gen_random_bipartite.py <n> <avg_degree> <seed> [out.json]
"""

import json
import random
import sys


def generate(n: int, avg_degree: float, seed: int) -> dict:
    rng = random.Random(seed)
    pi = list(range(n))
    rng.shuffle(pi)
    p = max(0.0, (avg_degree - 1.0) / n)
    edges = [set() for _ in range(n)]
    for u in range(n):
        edges[u].add(pi[u])
        for v in range(n):
            if v != pi[u] and rng.random() < p:
                edges[u].add(v)
    return {"size": n, "edges": [sorted(s) for s in edges]}


def main() -> None:
    n = int(sys.argv[1])
    d = float(sys.argv[2])
    seed = int(sys.argv[3])
    graph = generate(n, d, seed)
    degrees = [len(e) for e in graph["edges"]]
    out = sys.argv[4] if len(sys.argv) > 4 else f"random-{n}-d{d:g}.json"
    with open(out, "w") as f:
        json.dump(graph, f, separators=(",", ":"))
    print(
        f"n={n} planted-PM G(n,n,p) with p=({d}-1)/{n}: "
        f"avg deg {sum(degrees) / n:.2f}, min {min(degrees)}, max {max(degrees)} -> {out}"
    )


if __name__ == "__main__":
    main()
