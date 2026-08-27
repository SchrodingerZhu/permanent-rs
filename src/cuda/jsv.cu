// Native CUDA implementation of the JSV/BSVV walker.
//
// This mirrors the CubeCL kernels in src/gpu_markov_chain.rs one-for-one;
// the only intentional difference is the random source, which here is
// cuRAND's device API rather than a hand-rolled Philox. Both backends run
// acceptance in log space - the weight tables arrive as ln(w),
// exp(beta*delta) is replaced by beta*delta, and the ratio pass accumulates
// with a streaming log-sum-exp. Linear f32
// degrades at large beta: intermediate products like e^(-beta) * w graze the
// subnormal range from beta ~ 18 on (mostly masked by the weight cap and the
// 2^-24 acceptance quantization), and past beta ~ 88 e^(-beta) itself decays
// to subnormal and then exact zero, at which point slides whose large weight
// ratio genuinely compensates the penalty are silently deleted from the
// chain. In log space every quantity stays within a few hundred, where f32
// is comfortable at any beta, and the weight cap stops being a numerical
// necessity. The two backends are statistically equivalent, not draw-for-draw
// identical.
//
// Compiled device-only (nvcc --ptx) and loaded through the CUDA driver API;
// no host-side CUDA code is compiled or linked, which keeps the build free of
// libcudart and of any host-toolchain coupling.

#include <curand_kernel.h>

#define NO_HOLE 0xFFFFFFFFu

// Threads per block, fixed so the shared-memory stride below is a compile-time
// constant. One thread per chain; a block therefore carries JSV_BLOCK chains.
#define JSV_BLOCK 32

extern __shared__ unsigned int jsv_smem[];

// One chain's matching arrays.
//
// The natural layout, row_match[chain * n + i], puts adjacent threads n words
// apart: for n=32 that is 128 bytes, so every warp access pulls 32 separate
// sectors and uses 4 bytes of each. transit() touches these arrays several
// times per proposal and runs thousands of proposals per launch, so that waste
// dominates. Staging them in shared memory turns it into one strided global
// pass at entry and exit. Inside shared memory the entries are chain-minor
// (index i at offset i*JSV_BLOCK + lane) so a warp's lanes land in 32 distinct
// banks - the chain-major arrangement would put every lane in one bank.
//
// Shared entries are u16, not u32: the values are indices < n, and the shared
// budget caps a staged n well below 2^16, so nothing is lost - while the
// halved footprint doubles how many warps fit an SM (the staged kernels'
// occupancy is limited purely by shared memory). NO_HOLE truncates to 0xFFFF
// on store; flush_match widens it back. Two adjacent lanes' u16s share one
// 4-byte bank word, which the shared memory hardware serves without conflict.
template <bool SHARED> struct MatchWord { typedef unsigned int type; };
template <> struct MatchWord<true> { typedef unsigned short type; };

template <bool SHARED>
struct MatchRef {
    typedef typename MatchWord<SHARED>::type word;
    word *row;
    word *col;

    static __device__ __forceinline__ unsigned int stride() { return SHARED ? JSV_BLOCK : 1u; }
    __device__ __forceinline__ unsigned int row_at(unsigned int i) const { return row[i * stride()]; }
    __device__ __forceinline__ unsigned int col_at(unsigned int i) const { return col[i * stride()]; }
    __device__ __forceinline__ void set_row(unsigned int i, unsigned int v) { row[i * stride()] = (word)v; }
    __device__ __forceinline__ void set_col(unsigned int i, unsigned int v) { col[i * stride()] = (word)v; }
};

// Bind a chain's arrays, copying into shared memory when SHARED.
template <bool SHARED>
__device__ __forceinline__ MatchRef<SHARED> bind_match(unsigned int *__restrict__ row_match,
                                                       unsigned int *__restrict__ col_match,
                                                       unsigned int base, unsigned int n) {
    typedef typename MatchRef<SHARED>::word word;
    MatchRef<SHARED> m;
    if (SHARED) {
        word *smem = (word *)jsv_smem;
        m.row = &smem[threadIdx.x];
        m.col = &smem[JSV_BLOCK * n + threadIdx.x];
        for (unsigned int i = 0; i < n; ++i) {
            m.set_row(i, row_match[base + i]);
            m.set_col(i, col_match[base + i]);
        }
    } else {
        m.row = (word *)&row_match[base];
        m.col = (word *)&col_match[base];
    }
    return m;
}

template <bool SHARED>
__device__ __forceinline__ void flush_match(const MatchRef<SHARED> &m,
                                            unsigned int *__restrict__ row_match,
                                            unsigned int *__restrict__ col_match,
                                            unsigned int base, unsigned int n) {
    if (SHARED) {
        // The narrow store truncated NO_HOLE; widen the sentinel back out.
        const unsigned int hole = (typename MatchRef<SHARED>::word)NO_HOLE;
        for (unsigned int i = 0; i < n; ++i) {
            unsigned int r = m.row_at(i);
            unsigned int c = m.col_at(i);
            row_match[base + i] = (r == hole) ? NO_HOLE : r;
            col_match[base + i] = (c == hole) ? NO_HOLE : c;
        }
    }
}

// Adjacency accessor. The staged kernels read a bit-packed copy in shared
// memory - n*n/8 bytes, built cooperatively at launch entry - replacing one
// of the two random global loads per proposal with a shared load and a bit
// test. The global fallback reads the u32 table directly.
template <bool SHARED>
struct AdjRef {
    const unsigned int *bits;
    __device__ __forceinline__ unsigned int at(unsigned int index) const {
        return SHARED ? ((bits[index >> 5] >> (index & 31u)) & 1u) : bits[index];
    }
};

template <bool SHARED>
__device__ __forceinline__ AdjRef<SHARED> bind_adjacency(
    const unsigned int *__restrict__ adjacency, unsigned int n) {
    AdjRef<SHARED> a;
    if (SHARED) {
        // The packed adjacency lives right after the two matching arrays,
        // whose extent in u32 words depends on the staged entry width.
        unsigned int match_words =
            (2u * JSV_BLOCK * n * (unsigned int)sizeof(typename MatchWord<SHARED>::type) + 3u) / 4u;
        unsigned int *words = &jsv_smem[match_words];
        unsigned int total = n * n;
        unsigned int nwords = (total + 31u) >> 5;
        for (unsigned int w = threadIdx.x; w < nwords; w += JSV_BLOCK) {
            unsigned int word = 0;
            unsigned int base = w << 5;
            unsigned int limit = min(32u, total - base);
            for (unsigned int b = 0; b < limit; ++b)
                word |= (adjacency[base + b] & 1u) << b;
            words[w] = word;
        }
        __syncthreads();
        a.bits = words;
    } else {
        a.bits = adjacency;
    }
    return a;
}

// Uniform in [0, 1). Deliberately not curand_uniform, which returns (0, 1]:
// this is the same 24-bit construction the CubeCL kernels use, so a given
// draw maps to the same acceptance decision on both backends.
__device__ __forceinline__ float uniform_f32(unsigned int value) {
    return (float)(value >> 8) * (1.0f / 16777216.0f);
}

// ln of a uniform draw on (0, 1]: the +1 shifts the 24-bit lattice off zero,
// so the log is always finite and the acceptance test `log u < log p` accepts
// with probability p up to the same 2^-24 quantization as the linear test.
__device__ __forceinline__ float log_uniform_f32(unsigned int value) {
    return logf(((float)(value >> 8) + 1.0f) * (1.0f / 16777216.0f));
}

// Streaming log-sum-exp: accumulates terms given in log space as
// (max m, scaled sum s) with sum = s * e^m. Every exp argument is <= 0, so
// nothing can overflow, and a term far below the running max contributes a
// clean underflow-to-zero instead of poisoning the sum. Strictly, today's
// terms are already range-bounded by the bootstrap's step-factor cap (w'/w
// in [1/4, 4]) and the ratio factor <= 1, so this is insurance rather than a
// fix: it keeps the estimator's numerics correct even if those caps are ever
// relaxed. It replaces a Kahan-compensated linear accumulator.
struct LogSumExp {
    float max;
    float scaled;

    __device__ __forceinline__ void add(float log_value) {
        if (log_value > max) {
            scaled = scaled * expf(max - log_value) + 1.0f;
            max = log_value;
        } else {
            scaled += expf(log_value - max);
        }
    }
};

// Per-chain random source.
//
// The generic form just forwards to curand(). Philox is specialised because
// cuRAND's scalar curand() picks from a four-word buffer with
// `switch(state->STATE++)`: chains here follow data-dependent paths and so
// consume different numbers of draws, their STATE values desynchronise, and
// that switch then diverges across the warp on *every* draw. curand4() leaves
// STATE pinned at 0 and takes a single uniform path, so we pull four words at
// a time and shift them down ourselves - the same structure the CubeCL kernel
// uses, and worth ~2x here.
template <typename S>
struct Rng {
    S state;
    __device__ __forceinline__ unsigned int next() { return curand(&state); }
};

template <>
struct Rng<curandStatePhilox4_32_10_t> {
    curandStatePhilox4_32_10_t state;
    unsigned int lane_0;
    unsigned int lane_1;
    unsigned int lane_2;
    unsigned int lane_3;
    unsigned int remaining;

    __device__ __forceinline__ unsigned int next() {
        if (remaining == 0) {
            uint4 block = curand4(&state);
            lane_0 = block.x;
            lane_1 = block.y;
            lane_2 = block.z;
            lane_3 = block.w;
            remaining = 4;
        }
        unsigned int out = lane_0;
        lane_0 = lane_1;
        lane_1 = lane_2;
        lane_2 = lane_3;
        remaining -= 1;
        return out;
    }
};

// Per-chain registers carried through transit, mirroring ChainRegs.
template <typename S>
struct ChainRegs {
    unsigned int hole_u;
    unsigned int hole_v;
    unsigned int active;
    Rng<S> rng;

    __device__ __forceinline__ unsigned int next_u32() { return rng.next(); }

    // One menu-based Metropolis-Hastings proposal. A perfect matching draws
    // one of its n removals; a near-perfect one draws from its 2n-1 menu (add
    // across the holes, or slide either hole). The asymmetric menu sizes enter
    // as the Hastings factor n/(2n-1) on removals and its inverse on adds.
    // `__restrict__` throughout: without it nvcc must assume the matching
    // arrays it writes may alias the weight/adjacency tables it reads, and so
    // reloads those tables after every store. Marking them distinct also lets
    // the read-only ones go through the constant/texture path.
    // `log_weights` holds ln(w); `beta_delta[delta + 2]` holds beta * delta.
    // `log_hastings` = ln(n / (2n - 1)), hoisted by the caller.
    template <bool SHARED>
    __device__ void transit(MatchRef<SHARED> &m,
                            const float *__restrict__ log_weights,
                            const AdjRef<SHARED> &adj,
                            const float *__restrict__ beta_delta, float log_hastings,
                            unsigned int n) {
        unsigned int nn = n;
        if (hole_u == NO_HOLE) {
            // remove one of the n matched edges; Hastings factor n/(2n-1)
            unsigned int u = next_u32() % n;
            unsigned int v = m.row_at(u);
            unsigned int index = (unsigned int)u * nn + (unsigned int)v;
            unsigned int activity = adj.at(index);
            float log_probability = log_weights[index] + beta_delta[3 - activity] + log_hastings;
            if (log_probability >= 0.0f || log_uniform_f32(next_u32()) < log_probability) {
                m.set_row(u, NO_HOLE);
                m.set_col(v, NO_HOLE);
                hole_u = u;
                hole_v = v;
                active -= activity;
            }
        } else {
            unsigned int slot = next_u32() % (2 * n - 1);
            if (slot == 0) {
                // add across the holes; Hastings factor (2n-1)/n
                unsigned int index = (unsigned int)hole_u * nn + (unsigned int)hole_v;
                unsigned int activity = adj.at(index);
                float log_probability =
                    beta_delta[activity + 1] - log_weights[index] - log_hastings;
                if (log_probability >= 0.0f || log_uniform_f32(next_u32()) < log_probability) {
                    m.set_row(hole_u, hole_v);
                    m.set_col(hole_v, hole_u);
                    active += activity;
                    hole_u = NO_HOLE;
                    hole_v = NO_HOLE;
                }
            } else {
                // Slide one of the holes. A column slide is a row slide with
                // (row, col, hole_u, hole_v) swapped, so both take one code
                // path: within a warp the ~50/50 orientation split costs a
                // few predicated selects instead of serializing two branch
                // bodies through the loads, the draw and the writes.
                typedef typename MatchRef<SHARED>::word word;
                const unsigned int st = MatchRef<SHARED>::stride();
                bool row_slide = slot < n;
                word *prim = row_slide ? m.row : m.col;
                word *sec = row_slide ? m.col : m.row;
                unsigned int hp = row_slide ? hole_u : hole_v;
                unsigned int hq = row_slide ? hole_v : hole_u;
                unsigned int pick = row_slide ? slot - 1 : slot - n;
                // the picked column (row slide) / row (column slide), != hq
                unsigned int s = (pick >= hq) ? pick + 1 : pick;
                unsigned int z = sec[s * st];
                unsigned int gained = adj.at(row_slide ? hp * nn + s : s * nn + hp);
                unsigned int lost = adj.at(row_slide ? z * nn + s : s * nn + z);
                float log_probability =
                    beta_delta[gained + 2 - lost] +
                    log_weights[row_slide ? z * nn + hq : hq * nn + z] -
                    log_weights[(unsigned int)hole_u * nn + (unsigned int)hole_v];
                if (log_probability >= 0.0f || log_uniform_f32(next_u32()) < log_probability) {
                    prim[hp * st] = (word)s;
                    sec[s * st] = (word)hp;
                    prim[z * st] = (word)NO_HOLE;
                    if (row_slide) hole_u = z; else hole_v = z;
                    active = active + gained - lost;
                }
            }
        }
    }
};

// cuRAND states are opaque and differently sized per generator, so they live
// in a raw byte buffer the host sizes from jsv_state_size_* below.
template <typename S>
__device__ __forceinline__ ChainRegs<S> load_regs(const unsigned int *holes_u,
                                                  const unsigned int *holes_v,
                                                  const unsigned int *active_counts,
                                                  const void *states, unsigned int chain) {
    ChainRegs<S> regs;
    regs.hole_u = holes_u[chain];
    regs.hole_v = holes_v[chain];
    regs.active = active_counts[chain];
    regs.rng = ((const Rng<S> *)states)[chain];
    return regs;
}

template <typename S>
__device__ __forceinline__ void store_regs(const ChainRegs<S> &regs, unsigned int *holes_u,
                                           unsigned int *holes_v, unsigned int *active_counts,
                                           void *states, unsigned int chain) {
    holes_u[chain] = regs.hole_u;
    holes_v[chain] = regs.hole_v;
    active_counts[chain] = regs.active;
    // The full generator state round-trips, so a launch boundary is invisible
    // to the stream - unlike the CubeCL path, which persists only a counter.
    ((Rng<S> *)states)[chain] = regs.rng;
}

// The three passes, written once and instantiated per generator and per
// matching-storage choice. SHARED=false is the fallback for graphs too large
// to stage a block's matchings in shared memory.

template <typename S, bool SHARED>
__device__ __forceinline__ void warmup_body(
    unsigned int *__restrict__ row_match, unsigned int *__restrict__ col_match,
    const float *__restrict__ weights, const unsigned int *__restrict__ adjacency,
    const float *__restrict__ exp_beta, unsigned int *__restrict__ holes_u,
    unsigned int *__restrict__ holes_v, unsigned int *__restrict__ active_counts,
    void *__restrict__ states, unsigned int n, unsigned int num_chains,
    unsigned long long iterations) {
    unsigned int chain = blockIdx.x * blockDim.x + threadIdx.x;
    // Block-wide: builds the packed adjacency and syncs, so it must run
    // before the range guard peels off out-of-range threads.
    AdjRef<SHARED> adj = bind_adjacency<SHARED>(adjacency, n);
    if (chain >= num_chains) return;
    unsigned int base = chain * n;
    ChainRegs<S> regs = load_regs<S>(holes_u, holes_v, active_counts, states, chain);
    MatchRef<SHARED> m = bind_match<SHARED>(row_match, col_match, base, n);
    const float log_hastings = logf((float)n / (float)(2 * n - 1));
    for (unsigned long long i = 0; i < iterations; ++i)
        regs.transit(m, weights, adj, exp_beta, log_hastings, n);
    flush_match<SHARED>(m, row_match, col_match, base, n);
    store_regs<S>(regs, holes_u, holes_v, active_counts, states, chain);
}

template <typename S, bool SHARED>
__device__ __forceinline__ void occupancy_body(
    unsigned int *__restrict__ row_match, unsigned int *__restrict__ col_match,
    const float *__restrict__ weights, const unsigned int *__restrict__ adjacency,
    const float *__restrict__ exp_beta, unsigned int *__restrict__ holes_u,
    unsigned int *__restrict__ holes_v, unsigned int *__restrict__ active_counts,
    void *__restrict__ states, unsigned int *__restrict__ histogram, unsigned int n,
    unsigned int num_chains, unsigned long long samples, unsigned long long interval) {
    unsigned int chain = blockIdx.x * blockDim.x + threadIdx.x;
    // Block-wide: builds the packed adjacency and syncs, so it must run
    // before the range guard peels off out-of-range threads.
    AdjRef<SHARED> adj = bind_adjacency<SHARED>(adjacency, n);
    if (chain >= num_chains) return;
    unsigned int base = chain * n;
    ChainRegs<S> regs = load_regs<S>(holes_u, holes_v, active_counts, states, chain);
    MatchRef<SHARED> m = bind_match<SHARED>(row_match, col_match, base, n);
    const float log_hastings = logf((float)n / (float)(2 * n - 1));
    for (unsigned long long s = 0; s < samples; ++s) {
        for (unsigned long long i = 0; i < interval; ++i)
            regs.transit(m, weights, adj, exp_beta, log_hastings, n);
        unsigned int slot = (regs.hole_u == NO_HOLE) ? n * n : regs.hole_u * n + regs.hole_v;
        atomicAdd(&histogram[slot], 1u);
    }
    flush_match<SHARED>(m, row_match, col_match, base, n);
    store_regs<S>(regs, holes_u, holes_v, active_counts, states, chain);
}

template <typename S, bool SHARED>
__device__ __forceinline__ void ratio_body(
    unsigned int *__restrict__ row_match, unsigned int *__restrict__ col_match,
    const float *__restrict__ weights, const float *__restrict__ next_weights,
    const unsigned int *__restrict__ adjacency, const float *__restrict__ exp_beta,
    const float *__restrict__ ratio_terms, unsigned int *__restrict__ holes_u,
    unsigned int *__restrict__ holes_v, unsigned int *__restrict__ active_counts,
    void *__restrict__ states, float *__restrict__ sums, float *__restrict__ corrections,
    unsigned int *__restrict__ perfect_active, unsigned int n, unsigned int num_chains,
    unsigned long long samples, unsigned long long interval) {
    unsigned int chain = blockIdx.x * blockDim.x + threadIdx.x;
    // Block-wide: builds the packed adjacency and syncs, so it must run
    // before the range guard peels off out-of-range threads.
    AdjRef<SHARED> adj = bind_adjacency<SHARED>(adjacency, n);
    if (chain >= num_chains) return;
    unsigned int base = chain * n;
    ChainRegs<S> regs = load_regs<S>(holes_u, holes_v, active_counts, states, chain);
    MatchRef<SHARED> m = bind_match<SHARED>(row_match, col_match, base, n);
    const float log_hastings = logf((float)n / (float)(2 * n - 1));
    // `ratio_terms[k]` arrives as (beta - beta') * k, i.e. already in log
    // space, and the weight tables are ln(w); each sample's term is a sum of
    // logs, accumulated by streaming log-sum-exp. The chain hands back
    // (max, scaled) with term-sum = scaled * e^max; the host recombines the
    // per-chain pairs in f64.
    LogSumExp acc;
    acc.max = -INFINITY;
    acc.scaled = 0.0f;
    unsigned int hits = 0;
    for (unsigned long long s = 0; s < samples; ++s) {
        for (unsigned long long i = 0; i < interval; ++i)
            regs.transit(m, weights, adj, exp_beta, log_hastings, n);
        if (regs.hole_u == NO_HOLE) {
            unsigned int inactive = n - regs.active;
            acc.add(ratio_terms[inactive]);
            if (inactive == 0) hits += 1;
        } else {
            unsigned int index = regs.hole_u * n + regs.hole_v;
            unsigned int inactive = n - 1 - regs.active;
            acc.add(ratio_terms[inactive] + next_weights[index] - weights[index]);
        }
    }
    flush_match<SHARED>(m, row_match, col_match, base, n);
    store_regs<S>(regs, holes_u, holes_v, active_counts, states, chain);
    sums[chain] = acc.max;
    corrections[chain] = acc.scaled;
    perfect_active[chain] = hits;
}

// Thin extern "C" entry points; the host selects by name.
#define JSV_PASS_ARGS_WARMUP                                                                  \
    unsigned int *row_match, unsigned int *col_match, const float *weights,                   \
        const unsigned int *adjacency, const float *exp_beta, unsigned int *holes_u,          \
        unsigned int *holes_v, unsigned int *active_counts, void *states, unsigned int n,     \
        unsigned int num_chains, unsigned long long iterations
#define JSV_PASS_FWD_WARMUP                                                                   \
    row_match, col_match, weights, adjacency, exp_beta, holes_u, holes_v, active_counts,      \
        states, n, num_chains, iterations

#define JSV_PASS_ARGS_OCCUPANCY                                                               \
    unsigned int *row_match, unsigned int *col_match, const float *weights,                   \
        const unsigned int *adjacency, const float *exp_beta, unsigned int *holes_u,          \
        unsigned int *holes_v, unsigned int *active_counts, void *states,                     \
        unsigned int *histogram, unsigned int n, unsigned int num_chains,                     \
        unsigned long long samples, unsigned long long interval
#define JSV_PASS_FWD_OCCUPANCY                                                                \
    row_match, col_match, weights, adjacency, exp_beta, holes_u, holes_v, active_counts,      \
        states, histogram, n, num_chains, samples, interval

#define JSV_PASS_ARGS_RATIO                                                                   \
    unsigned int *row_match, unsigned int *col_match, const float *weights,                   \
        const float *next_weights, const unsigned int *adjacency, const float *exp_beta,      \
        const float *ratio_terms, unsigned int *holes_u, unsigned int *holes_v,               \
        unsigned int *active_counts, void *states, float *sums, float *corrections,           \
        unsigned int *perfect_active, unsigned int n, unsigned int num_chains,                \
        unsigned long long samples, unsigned long long interval
#define JSV_PASS_FWD_RATIO                                                                    \
    row_match, col_match, weights, next_weights, adjacency, exp_beta, ratio_terms, holes_u,   \
        holes_v, active_counts, states, sums, corrections, perfect_active, n, num_chains,     \
        samples, interval

#define JSV_ENTRY(PASS, UPASS, SUFFIX, STATE, TAG, SHARED)                                    \
    extern "C" __global__ void __launch_bounds__(JSV_BLOCK)                                   \
        jsv_##PASS##TAG##_##SUFFIX(JSV_PASS_ARGS_##UPASS) {                                   \
        PASS##_body<STATE, SHARED>(JSV_PASS_FWD_##UPASS);                                     \
    }

#define JSV_KERNELS(SUFFIX, STATE)                                                            \
    extern "C" __global__ void jsv_state_size_##SUFFIX(unsigned int *out) {                   \
        if (threadIdx.x == 0 && blockIdx.x == 0) out[0] = (unsigned int)sizeof(Rng<STATE>);   \
    }                                                                                         \
                                                                                              \
    extern "C" __global__ void jsv_init_##SUFFIX(void *states, unsigned long long seed,       \
                                                 unsigned int num_chains) {                   \
        unsigned int chain = blockIdx.x * blockDim.x + threadIdx.x;                           \
        if (chain >= num_chains) return;                                                      \
        Rng<STATE> local;                                                                     \
        /* subsequence = chain gives every chain a disjoint stream. */                        \
        curand_init(seed, chain, 0, &local.state);                                            \
        local = Rng<STATE>{local.state};                                                      \
        ((Rng<STATE> *)states)[chain] = local;                                                \
    }                                                                                         \
                                                                                              \
    JSV_ENTRY(warmup, WARMUP, SUFFIX, STATE, , true)                                          \
    JSV_ENTRY(warmup, WARMUP, SUFFIX, STATE, _global, false)                                  \
    JSV_ENTRY(occupancy, OCCUPANCY, SUFFIX, STATE, , true)                                    \
    JSV_ENTRY(occupancy, OCCUPANCY, SUFFIX, STATE, _global, false)                            \
    JSV_ENTRY(ratio, RATIO, SUFFIX, STATE, , true)                                            \
    JSV_ENTRY(ratio, RATIO, SUFFIX, STATE, _global, false)

JSV_KERNELS(philox, curandStatePhilox4_32_10_t)
JSV_KERNELS(xorwow, curandStateXORWOW_t)
JSV_KERNELS(mrg32k3a, curandStateMRG32k3a_t)
