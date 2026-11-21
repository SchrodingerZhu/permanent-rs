# How the estimator is derived

This note derives what `crates/permanent-core/src/estimator.rs` does: which statements are
theorems, which are bounds that only size the work, and which are measured.
`misc/verify_variance.py` checks the statements of sections 1 to 3 by
enumeration on small matrices; its output is quoted in section 9.

The ingredients come from Chen, Guo, Vigoda and Yang, "An Õ(n³)-time FPRAS
for the permanent" (arXiv:2609.20717), called "the paper" below. The
population, the segments and the sizing rules are this program's.

## 0. Setting

After scaling and regularization (`scaling.rs`) the matrix has positive
entries exp(-energy(u,v)). Write H(M) for the energy of a perfect matching
M, and for temperatures 0 = β₀ < β₁ < … < β_L = 1

    λ_i(M) = exp(-β_i H(M)),   Z_i = Σ_M λ_i(M),   μ_i = λ_i / Z_i.

Z₀ = n!, and per(A) = Z_L · μ_L(g) / c, with g the regularization factor
and c the scaling constant. With δ_i = β_{i+1} − β_i define the weights

    G_i = exp(-δ_i H) for i < L,   G_L = g,

so that μ_i(G_i) = Z_{i+1}/Z_i for i < L and

    per(A) = (n!/c) · Π_{i=0..L} μ_i(G_i).                              (0.1)

For i < L, μ_i(G_i f) = μ_i(G_i) · μ_{i+1}(f) for every f: weighting by
G_i turns μ_i into μ_{i+1}. Write Ḡ_i = G_i / μ_i(G_i),

    v_i = Var_{μ_i}(Ḡ_i),   g_i = max_M Ḡ_i(M).

## 1. The mutation kernel

Fix a temperature and anchor weights; let B be the rescaled row-stochastic
matrix of `tables.rs`. The paper's relaxed-assignment process X lives on
Ω = {σ : rows → columns ∪ {∗}} and is reversible for

    ν(σ) ∝ a^(n − un(σ)) (1+J)^(−cr(σ)) Π_{u assigned} B(u, σ_u),

with ν(· | perfect) = μ for any positive anchor weights. Its spectral gap is
at least 1 at every activity a (the paper's Claim 14). The program runs it
at a = J.

Let A ⊂ Ω be the perfect matchings and B = Ω ∖ A. `mutate` starts in A,
runs X, and stops when the time spent in A reaches t. With
A_s = ∫₀ˢ 1_A(X_r) dr and τ_t = inf{s : A_s > t}, it returns
Y_t = X_{τ_t}. Y is the *trace* of X on A: a Markov process on A with
generator the Schur complement

    Q^A = Q_AA − Q_AB Q_BB⁻¹ Q_BA.

**Lemma 1.** Y is reversible for μ, and its spectral gap is at least the
gap γ of X. Hence the kernel K_t = exp(t Q^A) satisfies μ K_t = μ and

    ‖K_t f − μ(f)‖_{L²(μ)} ≤ e^(−γt) ‖f − μ(f)‖_{L²(μ)} ≤ e^(−t) ‖f − μ(f)‖_{L²(μ)}.

*Proof.* Let D be the diagonal matrix of ν. Reversibility of X says DQ is
symmetric: D_A Q_AA and D_B Q_BB are symmetric and D_A Q_AB = (D_B Q_BA)ᵀ.
Then

    D_A Q^A = D_A Q_AA + (D_A Q_AB)(−D_B Q_BB)⁻¹(D_B Q_BA)

is symmetric, which is reversibility of Y for ν restricted to A, that is
for μ.

For the gap, let E(g) = −⟨g, Qg⟩_ν be the Dirichlet form of X and E^A that
of Y under μ. The Schur complement is what remains of a quadratic form
after minimizing over the eliminated coordinates, so

    E^A(f) = (1/ν(A)) · min{E(g) : g = f on A},

attained at the harmonic extension g* of f. Since a variance is the least
mean square deviation from a constant,

    Var_μ(f) ≤ μ((f − ν(g*))²) = (1/ν(A)) Σ_{x∈A} ν(x)(g*(x) − ν(g*))²
             ≤ Var_ν(g*)/ν(A) ≤ E(g*)/(γ ν(A)) = E^A(f)/γ.  ∎

The cost of K_t: X spends a fraction ν(A) of its time in A, at least 1/2
for the paper's α-rough weights (measured: 0.96 to 0.98), so it runs for
about t/ν(A). The paper bounds the events of time T at activity a by
C n (1+√a)(T + ln 2n)², which is O(n² log² n) per unit time at a = J.
Measured, a mutation costs about 0.5 n² t events (section 7).

## 2. The population estimate is unbiased

Take any kernels K_1, …, K_L with μ_i K_i = μ_i (here K_i is the K_t of
section 1 at temperature i). One *island* of P particles does:

    X_0^1 … X_0^P  i.i.d. from μ_0
    for i = 0 … L:
        ρ̂_i = (1/P) Σ_p G_i(X_i^p)
        if i < L: draw ancestors a_1 … a_P i.i.d. with P(a = q) ∝ G_i(X_i^q),
                  and X_{i+1}^p from K_{i+1}(X_i^{a_p}, ·)
    Ẑ = Π_{i=0..L} ρ̂_i

**Theorem 2.** E[Ẑ] = Π_i μ_i(G_i), for every P ≥ 1 and every choice of
μ_i-invariant kernels made before the run.

*Proof.* Put φ_{L+1} = 1 and φ_i = G_i · K_{i+1} φ_{i+1} (with K_{L+1} the
identity). By the weighting identity and invariance,
μ_i(φ_i) = μ_i(G_i) μ_{i+1}(K_{i+1} φ_{i+1}) = μ_i(G_i) μ_{i+1}(φ_{i+1}), so
μ_0(φ_0) = Π_i μ_i(G_i). Let

    M_i = (Π_{j<i} ρ̂_j) · (1/P) Σ_p φ_i(X_i^p).

E[M_0] = μ_0(φ_0). Given the particles up to level i, each X_{i+1}^p is a
copy of X_i^q with probability G_i(X_i^q)/(P ρ̂_i), then moved by K_{i+1}:

    E[(1/P) Σ_p φ_{i+1}(X_{i+1}^p) | levels ≤ i]
        = Σ_q G_i(X_i^q) (K_{i+1}φ_{i+1})(X_i^q) / (P ρ̂_i)
        = (1/(P ρ̂_i)) Σ_q φ_i(X_i^q).

Multiplying by Π_{j≤i} ρ̂_j gives E[M_{i+1} | levels ≤ i] = M_i. So M is a
martingale and E[Ẑ] = E[M_{L+1}] = E[M_0].  ∎

This is the unbiasedness of unnormalized particle estimates (Del Moral,
*Feynman–Kac Formulae*, 2004, Section 7.4). Three things it does and does not
need:

- It needs exact first particles. At β = 0 they are uniform permutations.
- It needs the schedule and the anchor weights fixed before the run, which
  is why Phase 1 and Phase 2 are separate. It does not need them to be
  good: any positive weights give invariant kernels (section 1).
- It needs no mixing at all: t may be arbitrarily small and P may be 1.
  Mixing only decides the variance.

`tests/end_to_end.rs::tiny_populations_are_unbiased` checks this with 4
particles and t = 0.2: the mean over 4096 islands matches the exact
permanent.

## 3. Its variance

For multinomial resampling at every level, the estimate is asymptotically
normal (Del Moral 2004, Section 9.4; Chopin, *Ann. Statist.* 2004):

    √P (Ẑ/Π_i μ_i(G_i) − 1)  ⇒  N(0, σ²),
    σ² = Σ_{i=0..L} Var_{μ_i}(h_i),   h_i = φ_i / μ_i(φ_i).            (3.1)

The h_i obey h_L = Ḡ_L and h_i = Ḡ_i · K_{i+1} h_{i+1}, each with mean 1.
If the kernels mixed perfectly, K_{i+1} h_{i+1} = 1 and σ² = Σ_i v_i, the
variance of a product of independent sample means of P draws each.

**Proposition 4.** Let q = e^(−t) · max_i √g_i. If q < 1,

    σ² ≤ (1 − q)^(−2) Σ_i v_i.

*Proof.* Let a_i = ‖h_i − 1‖_{L²(μ_i)}, so σ² = Σ a_i². Write

    h_i − 1 = (Ḡ_i − 1) + Ḡ_i (K_{i+1} h_{i+1} − 1).

The first term has norm √v_i. For the second, Ḡ_i ≤ g_i, the weighting
identity and Lemma 1 give

    μ_i(Ḡ_i² (K_{i+1}h_{i+1} − 1)²) ≤ g_i μ_i(Ḡ_i (K_{i+1}h_{i+1} − 1)²)
        = g_i ‖K_{i+1}h_{i+1} − 1‖²_{L²(μ_{i+1})} ≤ g_i e^(−2t) a_{i+1}².

So a_i ≤ √v_i + q a_{i+1}, hence a_i ≤ Σ_{j≥i} q^(j−i) √v_j: the sequence
(a_i) is at most (√v_i) convolved with (q^j), whose sum is 1/(1−q). Young's
inequality gives Σ a_i² ≤ (1−q)^(−2) Σ v_i.  ∎

(3.1) is asymptotic in P. The program does not rely on it for the error
bar (section 5); the bound of Proposition 4 only caps the size of an
island.

## 4. Segments

Cut the levels into S consecutive segments. By (0.1) the product splits,

    Π_i μ_i(G_i) = Π_{s} Π_{i ∈ segment s} μ_i(G_i),

and an island run on one segment alone, started from exact samples of the
segment's first μ_k, is unbiased for that segment's factor by Theorem 2.
Islands of different segments share nothing, so the product of their
estimates is unbiased for the whole, and relative variances add:

    relvar(Π_s Ẑ_s) = Π_s (1 + V_s) − 1 ≈ Σ_s V_s,

each V_s bounded by Proposition 4 over its own levels.

What makes this possible is the paper's sampler. A JSV-type chain needs the
whole cooling history to reach temperature k; the relaxed-assignment
sampler reaches μ_k from nothing with only the n anchor weights of level k.
One sweep is sequential in temperature and cannot be shortened by adding
devices; S segments run at the same time on S ranks.

**Starts.** The sampler's output law μ̂_k is within total variation η of
μ_k when each stage runs for ln(16 (m+1)/η) (`rigorous_stage_time`; m+1
stages). With N fresh particles in a run, all N coincide with exact draws
except with probability N η. `--rigorous-sampler` sets η so that this is at
most δ/16, with room for top-ups. The default runs each stage for 4 units
of `initial_stage_time`, which is not backed by a theorem; the check below
covers it.

**Check.** Let segment s end at level e, where segment s+1 starts. Segment
s carries its particles one level further and records c = its mean of
G_e; segment s+1 records f, the mean of G_e over its fresh particles.
E[f] = μ_e(G_e) when the starts are exact. The carried mean c alone is
biased by O(1/P), but Ẑ_s · c is the estimate of one more level, so by
Theorem 2

    E[Ẑ_s c] / E[Ẑ_s] = μ_e(G_e).

The program compares the ratio of island means Σ_r Ẑ_{s,r} c_r / Σ_r Ẑ_{s,r}
with the mean of f over islands, in standard errors of the difference
(the ratio's error from the residuals Ẑ_r (c_r − ratio) / mean(Ẑ)), and
warns beyond 3.5. A too-short sampler shows up as a difference; so would a
defect in the mutation or the resampling, since the two sides reach level e
by different routes.

On a 0/1 matrix the check is empty at high β: every particle is then a
perfect matching of the graph, all with the same energy, and both sides
are exactly 1.

## 5. Islands and the sizing rules

Each segment is run on R independent islands. Their values Ẑ_{s,1…R} are
i.i.d. and unbiased, so the segment's estimate is their mean, its relative
variance is estimated by their sample variance over R, and
ln per(A) is reported with the standard error

    se² = Σ_s (sample relvar of Ẑ_{s,·}) / R.

Nothing about this error bar depends on the mixing bound, on P, or on
(3.1); it is the ordinary error of a mean of i.i.d. unbiased values.

**Target.** The request is a relative error ε with probability 1 − δ. Half
of δ is given to the statistical error, two-sided: z = Φ⁻¹(1 − δ/4), and
the target is se ≤ σ_t = ln(1+ε)/z.

**Particles.** Proposition 4 bounds an island's relative variance by
c Σ_i v_i / P with c = (1 − q)^(−2). That c is far from what is measured.
q uses the largest normalized weight g, and the largest weight in a
population grows with the population it is read from, so on larger problems
c reaches the cap of 100 the program puts on it, while the measured variance
stays within 1.2 Σ_i v_i / P (section 9). Sizing by c made a 256 × 256
problem do a hundred times the work its target needed. The program
therefore sizes for twice the perfect-mixing variance,

    P = ⌈ 2 Σ_i v_i / V₀ ⌉,   V₀ = 1/4,   P ≥ 64,

and lets the measurement correct it: if the first islands come out with a
relative variance above 1, they are thrown away and run again with P
multiplied by (measured / V₀), up to the P the bound asks for. V₀ = 1/4
keeps an island's relative standard deviation at most 1/2: small enough
that the island values are far from the heavy right tail an under-mixed
estimate develops, and that the mean of R ≥ 32 of them is close to normal.

**Islands.** R starts at the number the target is expected to take,
1.1 · 1.2 Σ_i v_i / (P σ_t²) and at least 32, so that one round usually
does and the islands run together. If se > σ_t afterwards, islands are
added to reach R' = 1.1 R (se/σ_t)², since the error of a mean of i.i.d.
values falls as 1/√R.

**Mutation time.** For a fixed target the work is proportional to
P · t ∝ t (1 − q₀ e^(−t))^(−2) with q₀ = max √g_i. Its minimum is at
e^t = q₀ (1 + 2t): t = 1.57 for g = 4/3 (the paper's step rule) and
t = 2.0 for g = 2.25. The default t = 1.5 sits at that minimum. It is
conservative: the true gap on perfect matchings exceeds 1 (section 9).

**Segments.** With c_m events per mutation and c_s per fresh draw, an
estimate costs

    P R [ (L + S − 1) c_m + (S − 1) c_s ]   events,

and on S or more ranks its wall-clock is that of the longest segment,
about (L/S) sweeps plus one batch of fresh draws. c_s is 14 to 20 times c_m
(section 7), so segments pay off in wall-clock while S ≲ L c_m / c_s, about
L/15. The default is S = max(ranks, 4): four segments give three boundary
checks at a cost of about 3 c_s / (L c_m), a fifth of the total at L ≈ 200.

## 6. Phase 1

Phase 1 fixes β_i and the anchor weights w_i. Its errors cannot bias the
estimate (Theorem 2); they only change v_i, g_i and ν(A), that is, the cost.

**Learning.** The FAR chain (`chain/far.rs`) is a discrete-time chain on perfect
and near-perfect matchings, reversible for π with π(· | perfect) = μ. Let
T_1 < T_2 < … be its perfect steps. The chain watched at those steps is
stationary when started from μ, and by Kac's formula the expected number of
visits to a set S between two consecutive perfect steps is π(S)/π(perfect).
So for a chain started from μ and every k ≥ 1,

    E[visits to S during k cycles] = k π(S)/π(perfect),

and summed over W chains, each started from a μ-distributed state (the
chains need not be independent), the expectation depends only on the total
number of cycles. With S = N(r, v) this gives the anchor weights.

The variance of W chains of k cycles is W · Var(N_k), and Var(N_k)/k does
not decrease in k when cycles are positively correlated, so for a fixed
total W k, many short chains are no worse than few long ones. The
program therefore runs every particle of a Phase 1 population for
⌈64 n²/W⌉ cycles per round, and estimates the error from the spread
between groups of particles.

**Passes.** A pass runs the chains under the current weights w, doubling
the cycles until the visit count of every class N(r, v) has a relative
standard error s_v ≤ 0.15, and replaces w by the weights w̃ those counts
give. Kac's formula holds for the chain of any positive w, so w̃ is a
consistent measurement of the ideal weights whatever w was; w only decides
how often each class is visited, and a class whose weight is far too small
is visited rarely. A pass is therefore repeated, at most 8 times, while it
shows that w was far off.

What a pass shows is the correction c_v = ln(w̃(v)/w(v)), the true
distance of w(v) from the ideal weight plus the sampling error of the visit
count, whose standard deviation is s_v. The largest of n such corrections
is of order s √(2 ln 2n) when w is ideal: 0.3 to 0.5 at n = 128, where s is
typically 0.09. Comparing it with a fixed tolerance of ln 1.3 = 0.26 asks
for pass after pass that only draw the sampling error again. The test is
instead made class by class, against the hypothesis that w(v) was within
the tolerance: a pass asks for another only if

    |c_v| > ln 1.3 + z s_v   for some v,   z = Φ⁻¹(1 − 0.05/n),

so that when no weight was beyond the tolerance all n classes pass
together with probability at least 95%, or if some class missed the
precision s_v ≤ 0.15. With every pass forced to run at n = 128, none of the
511 second and later passes exceeded the tolerance by more than 3.0 s_v
(z = 3.36), while two of the 73 first passes did, by 3.5 and 7.8 s_v: the
steps whose first-order prediction of the weights was off by factors 1.8
and 2.6. The weights a temperature ends on have the same error either way,
that of its last pass.

The matching a chain stops on is again μ-distributed (the chain watched on
perfect matchings is μ-reversible), so a FAR run also moves its particle.
Phase 1 uses that as its only move between temperatures: resampled copies
are spread out by the next FAR runs. A separate mutation changed neither
the number of temperatures nor the weight check and cost 40% more.

**Schedule.** The step is the largest δ with relative variance of
exp(−δH) at most 1/4 over the population's energies, capped so that the
anchor weights move by at most a factor 4 (from the measured drift) and
by 4 times the previous step, and never below the paper's guaranteed step
0.25 max(1/H_max, β/(2 ln(1/τ))). The cap v_i ≤ 1/4 is what keeps g_i,
and with it q, moderate.

## 7. Costs

Measured events per mutation at t = 1.5, divided by n² t:

| n | 16 | 20 | 48 | 64 | 128 |
|---|---|---|---|---|---|
| events / (n² t) | 0.90 | 0.81 | 0.56 | 0.53 | 0.50 |

A fresh draw from the annealing sampler at the default stage time costs
10.3k events at n = 20, 29.6k at n = 48 and 47k at n = 64: 20, 15 and 14
mutations. In a scheme that draws every sample from nothing, each sample
costs a fresh draw; here a particle costs one at the start of its segment
and a mutation at every level after.

On the GPU a particle's working state is its matching, the inverse, and a
list of up to 32 rows in a collision: 2n + 32 sixteen-bit words. Collisions
cost a factor 1/(1+J) each in the target, so the list is almost always
empty; a particle that overflows it is treated like one that hit its event
cap. A warp's state is staged in shared memory while it fits (n up to about
370) and kept in global memory beyond. Mutation events per second on an RTX
PRO 6000 Blackwell, 65,536 particles (32,768 at n = 128):

| n | 128 | 256 | 512 |
|---|---|---|---|
| shared memory | 3.6e9 | 3.3e9 | 2.1e9 (16 per block) |
| global memory | 2.6e9 | 3.4e9 | 4.1e9 |

## 8. Scaling

One sweep is L mutations of every particle. On a device that holds W
particles at once its wall-clock does not depend on how many of the W
slots are used, so:

- at a fixed number of sweeps, the relative variance falls as 1/(particles
  in use): a device, or a rank, added to a run that already fills its
  devices buys accuracy in proportion;
- at a fixed accuracy, the wall-clock falls in proportion to the devices
  until one sweep of P R particles fits them, and from there in proportion
  to the segments, down to the L/15 of section 5.

Across ranks nothing is communicated in Phase 2 but one number per island.
In Phase 1 each rank holds a share of the population and the ranks add
their visit counts; a rank's particles are resampled among themselves.

The FAR learner is the part that gains least from a GPU. Its chains have
very uneven lengths (most cycles are one step, a few are thousands), and a
launch lasts as long as its longest thread. The GPU engine gives each
launch a step budget and finishes the chains that overrun it on the host,
which keeps every cycle complete. On an RTX PRO 6000 Blackwell against 48
host threads, Phase 1 is 1.8 times faster at n = 48 and 1.2 times at
n = 64 and n = 128.

## 9. Numerical checks

`misc/verify_variance.py` builds the generator of section 1 on all (n+1)^n
states for each temperature of a saved schedule and computes (3.1) exactly.
Run on a 4 × 4 matrix with two zeros and a 5 × 5 lognormal one (6
temperatures each, t = 1.5):

| | 4 × 4 | 5 × 5 |
|---|---|---|
| gap of X at a = J, over the temperatures | 1.0001 to 1.0013 | 1.0002 to 1.0013 |
| gap of the trace Y | 1.28 to 1.74 | 1.68 to 2.07 |
| ν(perfect) | 0.958 to 0.966 | 0.958 to 0.967 |
| Σ v_i (perfect mixing) | 1.189 | 1.069 |
| σ² of (3.1), exact | 1.401 | 1.190 |
| bound of Proposition 4 | 2.737 | 2.856 |
| measured P · relvar, P = 64, 256, 1024 (20000 islands) | 1.42, 1.38, 1.43 | 1.19, 1.20, 1.23 |

The paper's gap bound is attained almost exactly by the full process, the
trace has a larger gap as Lemma 1 says, and the measured variance agrees
with (3.1) at every P tried.

### Weighted matrices against exact permanents

Thirteen matrices with exact permanents (subset DP), 40 seeds each on the
CPU engine and on the GPU engine, at the default ε = 0.1, δ = 0.05,
generated and run by `misc/bench_weighted.py`. z is
(ln estimate − ln exact) / reported standard error; "in CI" is the share of
runs whose 97.5% interval contains the exact value.

| matrix | n | mean z (cpu / gpu) | sd z (cpu / gpu) | in CI (cpu / gpu) |
|---|---|---|---|---|
| dense, uniform(0,1) | 20 | 0.16 / 0.27 | 0.97 / 1.11 | 1.00 / 0.95 |
| dense, lognormal σ = 2 | 20 | 0.02 / 0.16 | 1.05 / 0.90 | 0.97 / 1.00 |
| dense, lognormal σ = 4 | 18 | 0.14 / −0.26 | 1.04 / 0.75 | 0.93 / 1.00 |
| 25% dense, lognormal σ = 1.5 | 22 | 0.12 / 0.07 | 0.85 / 0.92 | 1.00 / 1.00 |
| 60% dense, entries 10^U(−6, 6) | 18 | 0.01 / 0.13 | 0.84 / 0.96 | 1.00 / 0.97 |
| near a permutation, rest 10^U(−4, −1) | 20 | 0.22 / −0.04 | 0.82 / 1.09 | 1.00 / 0.93 |
| two dense blocks coupled at 10⁻³ | 24 | −0.16 / 0.07 | 0.77 / 1.33 | 0.97 / 0.95 |
| Poisson(1.2) counts plus identity | 20 | −0.14 / −0.03 | 0.86 / 0.72 | 1.00 / 1.00 |
| rank one plus 0.1 · uniform | 20 | −0.11 / −0.11 | 1.12 / 1.12 | 0.97 / 0.97 |
| `data/weighted-12.mtx` | 12 | 0.21 / 0.30 | 1.03 / 1.02 | 0.95 / 0.95 |
| `data/weighted-16.mtx` | 16 | 0.18 / −0.22 | 1.09 / 1.14 | 0.95 / 0.97 |
| `data/weighted-20.mtx` | 20 | 0.05 / 0.16 | 0.90 / 0.91 | 0.97 / 1.00 |
| 0/1, 30% dense (for comparison) | 20 | −0.06 / 0.05 | 1.02 / 1.05 | 0.97 / 0.97 |

Over the 1040 runs the interval contained the exact value in 1015 (97.6%)
and the relative error was within 10% in 1032 (99.2%). No run failed, and
the regularization factor E[g] was 1.000 throughout. The largest |z| was
3.25.

Two larger weighted matrices without an exact value, 6 seeds per engine:

| matrix | ln per, cpu (sd over seeds) | ln per, gpu (sd over seeds) | reported se | seconds, cpu / gpu |
|---|---|---|---|---|
| 64 × 64 dense, lognormal σ = 2 | 315.813 (0.030) | 315.783 (0.031) | 0.038 | 2.5 / 3.8 |
| 96 × 96, 15% dense, lognormal σ = 1.5 | 253.743 (0.028) | 253.755 (0.040) | 0.037 | 20 / 13 |

The engines agree, and the spread over seeds matches the reported error.

The boundary check of section 4, over 300 seeds on three of the weighted
matrices at ε = 0.05 (2700 differences): mean within 0.06 of zero, standard
deviation 1.01 to 1.03, and 2 beyond 3.5 where a normal expects 1.3. So a
run with three boundaries raises a false alarm about once in 500.

### Weighted lattices against Kasteleyn's formula

The dimer model on a square lattice with bond weights is a weighted
permanent with an exact value at any size (`misc/dimers.py`), which checks
the estimator where the subset DP cannot. GPU engine, one RTX PRO 6000
Blackwell:

| lattice | n | bond weights | ln Z, estimate | ln Z, exact | z | time |
|---|---|---|---|---|---|---|
| 16 × 16 | 128 | 1, 1 | 69.9598 ± 0.0191 | 69.9716 | −0.61 | 26 s |
| 16 × 16 | 128 | 1, 0.5 | 34.9926 ± 0.0186 | 35.0184 | −1.39 | 27 s |
| 16 × 16 | 128 | 1, 0.1 | 4.1953 ± 0.0190 | 4.1993 | −0.21 | 32 s |
| 12 × 20 | 120 | 2, 0.3 | 89.7604 ± 0.0189 | 89.7489 | +0.61 | 29 s |
| 32 × 32 | 512 | 1, 0.5 | 149.2797 ± 0.0378 | 149.2338 | +1.21 | 11 min |

(ε = 0.05 for n ≤ 128, 0.1 for n = 512.) The 16 × 32 lattice (n = 256,
weights 1 and 0.5, exact ln Z = 71.4440) on six seeds at ε = 0.1 gave
z = −1.11, +0.47, +0.17, +1.15, +0.60, −0.05, two minutes each. Its
measured island variance was 0.136 against a planned 0.25, and the n = 512
run's 0.131: the perfect-mixing sizing of section 5 holds at these sizes.

### The cubic lattice

No closed form exists in three dimensions. The open 4 × 4 × 4 cube has
5,051,532,105 coverings (cell-by-cell DP, ln = 22.3430); the estimate is
22.3176 ± 0.0190, z = −1.34. Entropy per site, ln Z / sites, on periodic
boxes:

| box | 4³ | 4·4·6 | 4·6·6 | 6³ | 6·6·8 | 8³ |
|---|---|---|---|---|---|---|
| sites | 64 | 96 | 144 | 216 | 288 | 512 |
| entropy per site | 0.47502 | 0.46876 | 0.46292 | 0.45794 | 0.45629 | 0.45360 |
| ± | 0.00030 | 0.00020 | 0.00013 | 0.00008 | 0.00006 | 0.00007 |

The 8³ box (n = 256) takes two minutes. The three cubes are fitted exactly
by s∞ + a/L² + c/L³ with s∞ = 0.450; three sizes do not give that
extrapolation an error. Every finite box must exceed Schrijver's bound
½ ln(5⁵/6⁴) = 0.4401, and they do. Published estimates of the limit are
slightly lower, near 0.447.

## 10. What is not proven

- The error bar assumes the mean of the island values is close to normal.
  There is no finite-P bound here on the tail of one island's value; (3.1)
  and Proposition 4 are asymptotic in P.
- In the default mode the segment starts are checked, not guaranteed.
  `--rigorous-sampler` gives the guarantee at the sampler's explicit cost.
- ν(perfect) ≥ 1/2, which bounds the cost of a mutation, needs anchor
  weights within the paper's factor of the ideal ones; Phase 1 delivers
  them with high probability, not always. Bad weights make the run slow,
  and are reported (`weight check`, over-budget mutations).
- The statements quoted from the paper (Claim 14, the event bound, the
  sampler's stage time) are used as stated there.
