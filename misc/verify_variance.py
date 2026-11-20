#!/usr/bin/env python3
"""Check the statements of misc/theory.md by enumeration on a small matrix.

    permanent small.mtx --segments 1 --save-schedule schedule.json
    python3 misc/verify_variance.py schedule.json

Builds the relaxed-assignment generator at the top activity on all
(n+1)^n states for every temperature of the schedule, cuts it down to the
perfect matchings, and reports

  * that the cut-down process is reversible for mu and its spectral gap,
  * the exact asymptotic variance of the population estimate, P * relvar,
    from the backward recursion h_i = G_i K_{i+1} h_{i+1},
  * the bound (1 - sqrt(g) e^-t)^-2 * sum relvar_i that sizes the population.

n = 4 takes a second, n = 5 a few minutes; larger n is out of reach.
"""
import itertools
import json
import sys

import numpy as np

ALPHA = 2.0


def generator(n, bmat, activity, penalty):
    """Generator of the dynamics on {0..n-1, *}^n, and the perfect states."""
    states = list(itertools.product(range(n + 1), repeat=n))
    index = {s: i for i, s in enumerate(states)}
    q = np.zeros((len(states), len(states)))
    weight = np.zeros(len(states))
    perfect = []
    for s in states:
        i = index[s]
        degree = [0] * n
        for v in s:
            if v < n:
                degree[v] += 1
        unassigned = sum(v == n for v in s)
        collisions = sum(max(d - 1, 0) for d in degree)
        w = activity ** (n - unassigned) * (1.0 + penalty) ** (-collisions)
        for u, v in enumerate(s):
            if v < n:
                w *= bmat[u, v]
        weight[i] = w
        if unassigned == 0 and collisions == 0:
            perfect.append(i)
        for u, v in enumerate(s):
            rate = 1.0 + penalty if v < n and degree[v] >= 2 else 1.0
            for new in range(n + 1):
                if new == v:
                    continue
                p = 1.0 / (1.0 + activity) if new == n else activity * bmat[u, new] / (1.0 + activity)
                j = index[s[:u] + (new,) + s[u + 1:]]
                q[i, j] += rate * p
    q[np.diag_indices_from(q)] = -q.sum(axis=1)
    return states, q, weight, np.array(perfect)


def symmetric_spectrum(q, pi):
    """Eigen-decomposition of a generator reversible for pi."""
    root = np.sqrt(pi)
    sym = root[:, None] * q / root[None, :]
    asymmetry = np.abs(sym - sym.T).max()
    values, vectors = np.linalg.eigh(0.5 * (sym + sym.T))
    return values, vectors, asymmetry


def main(path):
    data = json.load(open(path))
    n = data["n"]
    t = data["mutation_time"]
    energy = np.array(data["energy"]).reshape(n, n)
    # entries that were zero in the input are written as the lowest float
    log_gap = np.array(data["log_gap"]).reshape(n, n)
    log_gap = np.where(log_gap < -1e300, -np.inf, log_gap)
    levels = data["levels"]
    penalty = 8.0 * ALPHA * ALPHA * n * n

    kernels, mus, perms = [], [], None
    print(f"n = {n}, {len(levels)} temperatures, t = {t}, J = {penalty:g}")
    print(f"{'level':>5} {'beta':>9} {'full gap':>9} {'cut gap':>9} {'nu(perfect)':>11} {'reversibility':>13}")
    for i, level in enumerate(levels):
        rows = -level["beta"] * energy - np.array(level["log_w"])[None, :]
        bmat = np.exp(rows - rows.max(axis=1, keepdims=True))
        bmat /= bmat.sum(axis=1, keepdims=True)
        states, q, weight, perfect = generator(n, bmat, penalty, penalty)
        nu = weight / weight.sum()
        full_values, _, full_asym = symmetric_spectrum(q, nu)
        rest = np.setdiff1d(np.arange(len(states)), perfect)
        cut = q[np.ix_(perfect, perfect)] - q[np.ix_(perfect, rest)] @ np.linalg.solve(
            q[np.ix_(rest, rest)], q[np.ix_(rest, perfect)]
        )
        mu = nu[perfect] / nu[perfect].sum()
        if perms is None:
            perms = [states[p] for p in perfect]
        h = np.array([sum(energy[u, v] for u, v in enumerate(p)) for p in perms])
        target = np.exp(-level["beta"] * (h - h.min()))
        target /= target.sum()
        assert np.abs(mu - target).max() < 1e-9, "the cut-down law is not mu"
        values, vectors, asym = symmetric_spectrum(cut, mu)
        root = np.sqrt(mu)
        kernels.append((vectors * np.exp(t * values)[None, :]) @ vectors.T * root[None, :] / root[:, None])
        mus.append(mu)
        print(f"{i:5d} {level['beta']:9.5f} {-np.sort(full_values)[-2]:9.4f} {-np.sort(values)[-2]:9.4f} "
              f"{nu[perfect].sum():11.4f} {max(asym, full_asym):13.1e}")

    energies = np.array([sum(energy[u, v] for u, v in enumerate(p)) for p in perms])
    gaps = np.exp(np.array([sum(log_gap[u, v] for u, v in enumerate(p)) for p in perms]))
    last = len(levels) - 1
    potentials = [np.exp(-lv["step"] * (energies - lv["shift"])) for lv in levels[:last]] + [gaps]
    normalized = [g / (mu @ g) for g, mu in zip(potentials, mus)]
    relvar = [mu @ (g * g) - 1.0 for g, mu in zip(normalized, mus)]
    largest = max(g.max() for g in normalized)

    # h_i = G_i K_{i+1} h_{i+1}, each with mean 1 under mu_i
    h = normalized[last]
    variance = [mus[last] @ (h * h) - 1.0]
    for i in range(last - 1, -1, -1):
        h = normalized[i] * (kernels[i + 1] @ h)
        assert abs(mus[i] @ h - 1.0) < 1e-8
        variance.append(mus[i] @ (h * h) - 1.0)
    exact = sum(variance)
    contraction = np.sqrt(largest) * np.exp(-t)
    print(f"sum of step variances (perfect mixing): {sum(relvar):.5f}")
    print(f"exact asymptotic P * relvar:            {exact:.5f}")
    print(f"bound (g = {largest:.4f}, q = {contraction:.4f}):        {sum(relvar) / (1.0 - contraction) ** 2:.5f}")


if __name__ == "__main__":
    main(sys.argv[1])
