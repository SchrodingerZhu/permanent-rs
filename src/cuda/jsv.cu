// Native CUDA implementation of the JSV/BSVV walker.
//
// This mirrors the CubeCL kernels in src/gpu_markov_chain.rs one-for-one; the
// only intentional difference is the random source, which here is cuRAND's
// device API rather than a hand-rolled Philox. Everything else - the menu
// structure, the Hastings factors, the modulo range reduction and the [0,1)
// mapping - is kept bit-for-bit identical so the two backends stay
// comparable.
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
template <bool SHARED>
struct MatchRef {
    unsigned int *row;
    unsigned int *col;

    static __device__ __forceinline__ unsigned int stride() { return SHARED ? JSV_BLOCK : 1u; }
    __device__ __forceinline__ unsigned int row_at(unsigned int i) const { return row[i * stride()]; }
    __device__ __forceinline__ unsigned int col_at(unsigned int i) const { return col[i * stride()]; }
    __device__ __forceinline__ void set_row(unsigned int i, unsigned int v) { row[i * stride()] = v; }
    __device__ __forceinline__ void set_col(unsigned int i, unsigned int v) { col[i * stride()] = v; }
};

// Bind a chain's arrays, copying into shared memory when SHARED.
template <bool SHARED>
__device__ __forceinline__ MatchRef<SHARED> bind_match(unsigned int *__restrict__ row_match,
                                                       unsigned int *__restrict__ col_match,
                                                       unsigned int base, unsigned int n) {
    MatchRef<SHARED> m;
    if (SHARED) {
        m.row = &jsv_smem[threadIdx.x];
        m.col = &jsv_smem[JSV_BLOCK * n + threadIdx.x];
        for (unsigned int i = 0; i < n; ++i) {
            m.set_row(i, row_match[base + i]);
            m.set_col(i, col_match[base + i]);
        }
    } else {
        m.row = &row_match[base];
        m.col = &col_match[base];
    }
    return m;
}

template <bool SHARED>
__device__ __forceinline__ void flush_match(const MatchRef<SHARED> &m,
                                            unsigned int *__restrict__ row_match,
                                            unsigned int *__restrict__ col_match,
                                            unsigned int base, unsigned int n) {
    if (SHARED) {
        for (unsigned int i = 0; i < n; ++i) {
            row_match[base + i] = m.row_at(i);
            col_match[base + i] = m.col_at(i);
        }
    }
}

// Uniform in [0, 1). Deliberately not curand_uniform, which returns (0, 1]:
// this is the same 24-bit construction the CubeCL kernels use, so a given
// draw maps to the same acceptance decision on both backends.
__device__ __forceinline__ float uniform_f32(unsigned int value) {
    return (float)(value >> 8) * (1.0f / 16777216.0f);
}

// Kahan/Neumaier compensated accumulation, matching CompensatedF32::add.
struct Compensated {
    float sum;
    float correction;

    __device__ __forceinline__ void add(float value) {
        float next = sum + value;
        correction += (fabsf(sum) >= fabsf(value)) ? ((sum - next) + value)
                                                   : ((value - next) + sum);
        sum = next;
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
    template <bool SHARED>
    __device__ void transit(MatchRef<SHARED> &m,
                            const float *__restrict__ weights,
                            const unsigned int *__restrict__ adjacency,
                            const float *__restrict__ exp_beta, unsigned int n) {
        unsigned int nn = n;
        if (hole_u == NO_HOLE) {
            // remove one of the n matched edges; Hastings factor n/(2n-1)
            unsigned int u = next_u32() % n;
            unsigned int v = m.row_at(u);
            unsigned int index = (unsigned int)u * nn + (unsigned int)v;
            unsigned int activity = adjacency[index];
            float probability = weights[index] * exp_beta[3 - activity] *
                                ((float)n / (float)(2 * n - 1));
            if (probability >= 1.0f || uniform_f32(next_u32()) < probability) {
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
                unsigned int activity = adjacency[index];
                float probability = exp_beta[activity + 1] / weights[index] *
                                    ((float)(2 * n - 1) / (float)n);
                if (probability >= 1.0f || uniform_f32(next_u32()) < probability) {
                    m.set_row(hole_u, hole_v);
                    m.set_col(hole_v, hole_u);
                    active += activity;
                    hole_u = NO_HOLE;
                    hole_v = NO_HOLE;
                }
            } else if (slot < n) {
                // slide onto the row hole: column v != hole_v, matched to row z
                unsigned int pick = slot - 1;
                unsigned int v = (pick >= hole_v) ? pick + 1 : pick;
                unsigned int z = m.col_at(v);
                unsigned int gained = adjacency[(unsigned int)hole_u * nn + (unsigned int)v];
                unsigned int lost = adjacency[(unsigned int)z * nn + (unsigned int)v];
                float probability = exp_beta[gained + 2 - lost] *
                                    weights[(unsigned int)z * nn + (unsigned int)hole_v] /
                                    weights[(unsigned int)hole_u * nn + (unsigned int)hole_v];
                if (probability >= 1.0f || uniform_f32(next_u32()) < probability) {
                    m.set_row(hole_u, v);
                    m.set_col(v, hole_u);
                    m.set_row(z, NO_HOLE);
                    hole_u = z;
                    active = active + gained - lost;
                }
            } else {
                // slide onto the column hole: row u != hole_u, matched to col z
                unsigned int pick = slot - n;
                unsigned int u = (pick >= hole_u) ? pick + 1 : pick;
                unsigned int z = m.row_at(u);
                unsigned int gained = adjacency[(unsigned int)u * nn + (unsigned int)hole_v];
                unsigned int lost = adjacency[(unsigned int)u * nn + (unsigned int)z];
                float probability = exp_beta[gained + 2 - lost] *
                                    weights[(unsigned int)hole_u * nn + (unsigned int)z] /
                                    weights[(unsigned int)hole_u * nn + (unsigned int)hole_v];
                if (probability >= 1.0f || uniform_f32(next_u32()) < probability) {
                    m.set_row(u, hole_v);
                    m.set_col(hole_v, u);
                    m.set_col(z, NO_HOLE);
                    hole_v = z;
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
    if (chain >= num_chains) return;
    unsigned int base = chain * n;
    ChainRegs<S> regs = load_regs<S>(holes_u, holes_v, active_counts, states, chain);
    MatchRef<SHARED> m = bind_match<SHARED>(row_match, col_match, base, n);
    for (unsigned long long i = 0; i < iterations; ++i)
        regs.transit(m, weights, adjacency, exp_beta, n);
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
    if (chain >= num_chains) return;
    unsigned int base = chain * n;
    ChainRegs<S> regs = load_regs<S>(holes_u, holes_v, active_counts, states, chain);
    MatchRef<SHARED> m = bind_match<SHARED>(row_match, col_match, base, n);
    for (unsigned long long s = 0; s < samples; ++s) {
        for (unsigned long long i = 0; i < interval; ++i)
            regs.transit(m, weights, adjacency, exp_beta, n);
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
    if (chain >= num_chains) return;
    unsigned int base = chain * n;
    ChainRegs<S> regs = load_regs<S>(holes_u, holes_v, active_counts, states, chain);
    MatchRef<SHARED> m = bind_match<SHARED>(row_match, col_match, base, n);
    Compensated acc;
    acc.sum = 0.0f;
    acc.correction = 0.0f;
    unsigned int hits = 0;
    for (unsigned long long s = 0; s < samples; ++s) {
        for (unsigned long long i = 0; i < interval; ++i)
            regs.transit(m, weights, adjacency, exp_beta, n);
        if (regs.hole_u == NO_HOLE) {
            unsigned int inactive = n - regs.active;
            acc.add(ratio_terms[inactive]);
            if (inactive == 0) hits += 1;
        } else {
            unsigned int index = regs.hole_u * n + regs.hole_v;
            unsigned int inactive = n - 1 - regs.active;
            acc.add(ratio_terms[inactive] * next_weights[index] / weights[index]);
        }
    }
    flush_match<SHARED>(m, row_match, col_match, base, n);
    store_regs<S>(regs, holes_u, holes_v, active_counts, states, chain);
    sums[chain] = acc.sum;
    corrections[chain] = acc.correction;
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
