//! The chains of `permanent_core::chain` in CubeCL's kernel language,
//! restricted to what WGSL has: 32-bit integers and `f32`.
//!
//! They follow the same steps and the same random draws as the generic
//! chains. A 64-bit quantity is two words, low then high, and a dispatch
//! runs a bounded number of steps and leaves each chain's state in a
//! buffer, so the host can dispatch again until every chain is done.

use cubecl::prelude::*;

/// "No vertex", as in `permanent_core::chain::lane`.
pub const NONE: u32 = 0xFFFF;

/// Words of a chain's state: its stream, then what the chain itself keeps.
pub const STATE: usize = 24;
// words 0..9 are the stream: key, blocks generated, the last block, and
// how many of its outputs are left
/// Set once a chain has started, when it is over, and if it ran out of steps.
pub const FLAGS: usize = 9;
pub const STARTED: u32 = 1;
pub const OVER: u32 = 2;
pub const CAPPED: u32 = 4;
/// An annealing trial ended on a perfect matching.
pub const PERFECT: u32 = 8;
// the FAR chain
pub const HOLE_U: usize = 10;
pub const HOLE_V: usize = 11;
pub const CYCLES: usize = 12;
pub const STEPS: usize = 13;
pub const ENERGY: usize = 14;
pub const PERFECT_ENERGY: usize = 16;
// the relaxed-assignment chains
pub const CONFLICTED: usize = 10;
pub const UNASSIGNED: usize = 11;
pub const EVENTS: usize = 12;
/// The annealing trial's stage and its 64-bit tick clock.
pub const STAGE: usize = 14;
pub const TICKS: usize = 15;

/// In the column array: two or more rows are on this column.
pub const MULTI: u32 = NONE - 1;
/// Rows in a collision that the list can hold.
pub const CONFLICTS: usize = 32;

/// Words of working memory of a relaxed assignment: the matching, its
/// inverse and the collision list.
pub const fn lane_words(n: usize) -> usize {
    2 * n + CONFLICTS
}

const M0: u32 = 0xD251_1F53;
const M1: u32 = 0xCD9E_8D57;
const W0: u32 = 0x9E37_79B9;
const W1: u32 = 0xBB67_AE85;

/// The next block of Philox4x32-10 into `rng`, laid out as a chain's state.
#[cube]
fn refill(rng: &mut Array<u32>) {
    let mut c0 = rng[2];
    let mut c1 = rng[3];
    let mut c2 = 0u32;
    let mut c3 = 0u32;
    let mut k0 = rng[0];
    let mut k1 = rng[1];
    let mut round = 0u32;
    while round < 10 {
        let h0 = u32::mul_hi(M0, c0);
        let l0 = M0 * c0;
        let h1 = u32::mul_hi(M1, c2);
        let l1 = M1 * c2;
        c0 = h1 ^ c1 ^ k0;
        c2 = h0 ^ c3 ^ k1;
        c1 = l1;
        c3 = l0;
        k0 += W0;
        k1 += W1;
        round += 1;
    }
    rng[4] = c0;
    rng[5] = c1;
    rng[6] = c2;
    rng[7] = c3;
    rng[8] = 4;
    let low = rng[2] + 1;
    rng[2] = low;
    if low == 0 {
        rng[3] += 1;
    }
}

#[cube]
fn next_u32(rng: &mut Array<u32>) -> u32 {
    if rng[8] == 0 {
        refill(rng);
    }
    let left = rng[8] - 1;
    rng[8] = left;
    // fixed indices only: a compiler keeps such an array in registers
    let mut out = rng[7];
    if left == 3 {
        out = rng[4];
    } else if left == 2 {
        out = rng[5];
    } else if left == 1 {
        out = rng[6];
    }
    out
}

/// Read a chain's stream from its state.
#[cube]
fn load_stream(state: &mut [u32], at: usize, rng: &mut Array<u32>) {
    rng[0] = state[at];
    rng[1] = state[at + 1];
    rng[2] = state[at + 2];
    rng[3] = state[at + 3];
    rng[4] = state[at + 4];
    rng[5] = state[at + 5];
    rng[6] = state[at + 6];
    rng[7] = state[at + 7];
    rng[8] = state[at + 8];
}

/// Write a chain's stream to its state.
#[cube]
fn store_stream(state: &mut [u32], at: usize, rng: &mut Array<u32>) {
    state[at] = rng[0];
    state[at + 1] = rng[1];
    state[at + 2] = rng[2];
    state[at + 3] = rng[3];
    state[at + 4] = rng[4];
    state[at + 5] = rng[5];
    state[at + 6] = rng[6];
    state[at + 7] = rng[7];
    state[at + 8] = rng[8];
}

// The FAR tally: chains still running, chains that ran out of steps, the
// 64-bit totals of steps and of perfect energy over the chains that ended,
// then for each column the 64-bit energy sum of the visits to its hole
// class, the visits of each group to each class, the cycles of each
// group's chains that ended, and the particles still running.
pub const LATE: usize = 0;
pub const CAPPED_CHAINS: usize = 1;
pub const STEPS_TOTAL: usize = 2;
pub const ENERGY_TOTAL: usize = 4;
pub const CLASSES: usize = 6;

/// Add the 64-bit (low, high) to the two words of `tally` at `at`. The sum
/// is exact once every thread is done: the previous low word tells a
/// thread whether its own addition carried.
#[cube]
fn add64(tally: &[Atomic<u32>], at: usize, low: u32, high: u32) {
    let before = tally[at].fetch_add(low);
    let after = before + low;
    let mut carry = 0u32;
    if after < before {
        carry = 1;
    }
    tally[at + 1].fetch_add(high + carry);
}

/// Whether the 64-bit (ah, al) is below (bh, bl).
#[cube]
fn below64(ah: u32, al: u32, bh: u32, bl: u32) -> bool {
    ah < bh || (ah == bh && al < bl)
}

/// Draw from line `line` of the packed alias table at `table` in `tables`.
#[cube]
fn alias_draw(tables: &[u32], table: usize, line: u32, n: u32, rng: &mut Array<u32>) -> u32 {
    let i = u32::mul_hi(next_u32(rng), n);
    let entry = table + 2 * (line as usize * n as usize + i as usize);
    let low = tables[entry];
    let high = tables[entry + 1];
    // 48 random bits against the 48-bit threshold above the alias
    let first = next_u32(rng);
    let second = next_u32(rng);
    let mut out = low & 0xFFFF;
    if first < high || (first == high && (second >> 16) < (low >> 16)) {
        out = i;
    }
    out
}

/// The FAR chain (`permanent_core::chain::far`) on every particle that is
/// not over, for at most `budget` steps of this dispatch. With `fresh`,
/// every particle starts a chain on the stream keyed by its two `seeds`.
///
/// `tables`: the packed row alias table, the column one, the move
/// thresholds and the entry energies, each entry 64 bits. `scratch`: the
/// matching and its inverse, 2 n words a particle. `tally`: see `LATE`;
/// its list holds at most `listed` of the particles still running.
#[cube(launch_unchecked)]
pub fn far(
    tables: &[u32],
    sigma: &mut [u32],
    scratch: &mut [u32],
    state: &mut [u32],
    tally: &[Atomic<u32>],
    seeds: &[u32],
    n: u32,
    count: u32,
    anchor: u32,
    groups: u32,
    cycles: u32,
    budget: u32,
    max_steps: u32,
    fresh: u32,
    listed: u32,
) {
    let particle = ABSOLUTE_POS;
    let size = n as usize;
    let at = particle * STATE;
    let mut active = false;
    if particle < count as usize {
        active = fresh != 0 || (state[at + FLAGS] & OVER) == 0;
    }
    if active {
        let rows = particle * 2 * size;
        let cols = rows + size;
        let col_alias = 2 * size * size;
        let moves = 4 * size * size;
        let energies = 8 * size * size;
        let mut rng = Array::<u32>::new(9usize);
        if fresh != 0 {
            rng[0] = seeds[2 * particle];
            rng[1] = seeds[2 * particle + 1];
            rng[2] = 0;
            rng[3] = 0;
            rng[8] = 0;
        } else {
            load_stream(state, at, &mut rng);
        }
        let mut hole_u = NONE.runtime();
        let mut hole_v = 0u32;
        let mut done = 0u32;
        let mut steps = 0u32;
        let mut h_low = 0u32;
        let mut h_high = 0u32;
        let mut sum_low = 0u32;
        let mut sum_high = 0u32;
        if fresh != 0 {
            let mut v = 0usize;
            while v < size {
                scratch[cols + v] = NONE;
                v += 1;
            }
            let mut u = 0usize;
            while u < size {
                let v = sigma[particle * size + u];
                scratch[rows + u] = v;
                scratch[cols + v as usize] = u as u32;
                let cell = energies + 2 * (u * size + v as usize);
                let low = h_low + tables[cell];
                let mut carry = 0u32;
                if low < h_low {
                    carry = 1;
                }
                h_low = low;
                h_high = h_high + tables[cell + 1] + carry;
                u += 1;
            }
        } else {
            hole_u = state[at + HOLE_U];
            hole_v = state[at + HOLE_V];
            done = state[at + CYCLES];
            steps = state[at + STEPS];
            h_low = state[at + ENERGY];
            h_high = state[at + ENERGY + 1];
            sum_low = state[at + PERFECT_ENERGY];
            sum_high = state[at + PERFECT_ENERGY + 1];
        }
        let group = (particle as u32) % groups;
        let visits = CLASSES + 2 * size + group as usize * size;
        let ended = CLASSES + 2 * size + groups as usize * size;
        let mut taken = 0u32;
        let mut running = true;
        while running {
            if taken >= budget || steps >= max_steps {
                running = false;
            } else if hole_u == NONE {
                if done == cycles {
                    running = false;
                } else {
                    done += 1;
                    let low = sum_low + h_low;
                    let mut carry = 0u32;
                    if low < sum_low {
                        carry = 1;
                    }
                    sum_low = low;
                    sum_high = sum_high + h_high + carry;
                    steps += 1;
                    taken += 1;
                    if (next_u32(&mut rng) & 1) == 0 {
                        let u = u32::mul_hi(next_u32(&mut rng), n);
                        let v = scratch[rows + u as usize];
                        scratch[rows + u as usize] = NONE;
                        scratch[cols + v as usize] = NONE;
                        let cell = energies + 2 * (u as usize * size + v as usize);
                        let e_low = tables[cell];
                        let mut borrow = 0u32;
                        if h_low < e_low {
                            borrow = 1;
                        }
                        h_low -= e_low;
                        h_high = h_high - tables[cell + 1] - borrow;
                        hole_u = u;
                        hole_v = v;
                    }
                }
            } else {
                if hole_u == anchor {
                    tally[visits + hole_v as usize].fetch_add(1);
                    add64(tally, CLASSES + 2 * hole_v as usize, h_low, h_high);
                }
                steps += 1;
                taken += 1;
                let cell = hole_u as usize * size + hole_v as usize;
                let add_low = tables[moves + 4 * cell];
                let add_high = tables[moves + 4 * cell + 1];
                let row_low = tables[moves + 4 * cell + 2];
                let row_high = tables[moves + 4 * cell + 3];
                let x_high = next_u32(&mut rng);
                let x_low = next_u32(&mut rng);
                if below64(x_high, x_low, add_high, add_low) {
                    scratch[rows + hole_u as usize] = hole_v;
                    scratch[cols + hole_v as usize] = hole_u;
                    let e = energies + 2 * cell;
                    let low = h_low + tables[e];
                    let mut carry = 0u32;
                    if low < h_low {
                        carry = 1;
                    }
                    h_low = low;
                    h_high = h_high + tables[e + 1] + carry;
                    hole_u = NONE;
                } else if below64(x_high, x_low, row_high, row_low) {
                    // the row hole takes over a matched column
                    let z = alias_draw(tables, 0usize, hole_u, n, &mut rng);
                    if z != hole_v {
                        let y = scratch[cols + z as usize];
                        scratch[rows + y as usize] = NONE;
                        scratch[rows + hole_u as usize] = z;
                        scratch[cols + z as usize] = hole_u;
                        let plus = energies + 2 * (hole_u as usize * size + z as usize);
                        let minus = energies + 2 * (y as usize * size + z as usize);
                        let low = h_low + tables[plus];
                        let mut carry = 0u32;
                        if low < h_low {
                            carry = 1;
                        }
                        h_high = h_high + tables[plus + 1] + carry;
                        let e_low = tables[minus];
                        let mut borrow = 0u32;
                        if low < e_low {
                            borrow = 1;
                        }
                        h_low = low - e_low;
                        h_high = h_high - tables[minus + 1] - borrow;
                        hole_u = y;
                    }
                } else {
                    // x - row < row - add: the column hole takes over a matched row
                    let mut borrow = 0u32;
                    if x_low < row_low {
                        borrow = 1;
                    }
                    let d_low = x_low - row_low;
                    let d_high = x_high - row_high - borrow;
                    let mut borrow = 0u32;
                    if row_low < add_low {
                        borrow = 1;
                    }
                    let w_low = row_low - add_low;
                    let w_high = row_high - add_high - borrow;
                    if below64(d_high, d_low, w_high, w_low) {
                        let y = alias_draw(tables, col_alias, hole_v, n, &mut rng);
                        if y != hole_u {
                            let z = scratch[rows + y as usize];
                            scratch[rows + y as usize] = hole_v;
                            scratch[cols + hole_v as usize] = y;
                            scratch[cols + z as usize] = NONE;
                            let plus = energies + 2 * (y as usize * size + hole_v as usize);
                            let minus = energies + 2 * (y as usize * size + z as usize);
                            let low = h_low + tables[plus];
                            let mut carry = 0u32;
                            if low < h_low {
                                carry = 1;
                            }
                            h_high = h_high + tables[plus + 1] + carry;
                            let e_low = tables[minus];
                            let mut borrow = 0u32;
                            if low < e_low {
                                borrow = 1;
                            }
                            h_low = low - e_low;
                            h_high = h_high - tables[minus + 1] - borrow;
                            hole_v = z;
                        }
                    }
                }
            }
        }
        let mut flags = STARTED.runtime();
        if hole_u == NONE && done == cycles {
            // complete: the particle takes the matching the chain stopped on
            let mut u = 0usize;
            while u < size {
                sigma[particle * size + u] = scratch[rows + u];
                u += 1;
            }
            flags = STARTED | OVER;
        } else if steps >= max_steps {
            // out of steps: the particle stays where it was
            flags = STARTED | OVER | CAPPED;
            tally[CAPPED_CHAINS].fetch_add(1);
        }
        if (flags & OVER) != 0 {
            tally[ended + group as usize].fetch_add(done);
            add64(tally, STEPS_TOTAL, steps, 0u32);
            add64(tally, ENERGY_TOTAL, sum_low, sum_high);
        } else {
            let slot = tally[LATE].fetch_add(1);
            if slot < listed {
                tally[ended + groups as usize + slot as usize].store(particle as u32);
            }
        }
        store_stream(state, at, &mut rng);
        state[at + FLAGS] = flags;
        state[at + HOLE_U] = hole_u;
        state[at + HOLE_V] = hole_v;
        state[at + CYCLES] = done;
        state[at + STEPS] = steps;
        state[at + ENERGY] = h_low;
        state[at + ENERGY + 1] = h_high;
        state[at + PERFECT_ENERGY] = sum_low;
        state[at + PERFECT_ENERGY + 1] = sum_high;
    }
}

/// Whether row u is in the collision list. `counts`: rows in a collision,
/// unmatched rows, and whether the list overflowed.
#[cube]
fn is_conflicted(scratch: &mut [u32], list: usize, counts: &mut Array<u32>, u: u32) -> bool {
    let mut found = false;
    let mut i = 0usize;
    while i < counts[0] as usize {
        if scratch[list + i] == u {
            found = true;
        }
        i += 1;
    }
    found
}

#[cube]
fn add_conflict(scratch: &mut [u32], list: usize, counts: &mut Array<u32>, u: u32) {
    if counts[0] == CONFLICTS as u32 {
        counts[2] = 1;
    } else {
        scratch[list + counts[0] as usize] = u;
        counts[0] += 1;
    }
}

#[cube]
fn drop_conflict(scratch: &mut [u32], list: usize, counts: &mut Array<u32>, u: u32) {
    let size = counts[0] as usize;
    let mut at = size;
    let mut i = 0usize;
    while i < size {
        if at == size && scratch[list + i] == u {
            at = i;
        }
        i += 1;
    }
    if at < size {
        counts[0] -= 1;
        scratch[list + at] = scratch[list + counts[0] as usize];
    }
}

/// Unmatch row u.
#[cube]
fn remove(scratch: &mut [u32], rows: usize, n: usize, counts: &mut Array<u32>, u: u32) {
    let cols = rows + n;
    let list = cols + n;
    let v = scratch[rows + u as usize];
    if v != NONE {
        scratch[rows + u as usize] = NONE;
        counts[1] += 1;
        if scratch[cols + v as usize] != MULTI {
            // u was alone on v
            scratch[cols + v as usize] = NONE;
        } else {
            drop_conflict(scratch, list, counts, u);
            // if one row is left on v, it is no longer in a collision
            let mut left = 0u32;
            let mut last = NONE.runtime();
            let mut i = 0usize;
            while i < counts[0] as usize {
                let w = scratch[list + i];
                if scratch[rows + w as usize] == v {
                    left += 1;
                    last = w;
                }
                i += 1;
            }
            if left == 1 {
                drop_conflict(scratch, list, counts, last);
                scratch[cols + v as usize] = last;
            }
        }
    }
}

/// Put the unmatched row u on column v.
#[cube]
fn place(scratch: &mut [u32], rows: usize, n: usize, counts: &mut Array<u32>, u: u32, v: u32) {
    let cols = rows + n;
    let list = cols + n;
    scratch[rows + u as usize] = v;
    counts[1] -= 1;
    let occupant = scratch[cols + v as usize];
    if occupant == NONE {
        scratch[cols + v as usize] = u;
    } else if occupant == MULTI {
        add_conflict(scratch, list, counts, u);
    } else {
        scratch[cols + v as usize] = MULTI;
        add_conflict(scratch, list, counts, occupant);
        add_conflict(scratch, list, counts, u);
    }
}

/// One ring of the dynamics (`permanent_core::chain::dynamics`): pick the
/// row whose clock rang, unmatch it, and redraw it from its row of B unless
/// it stays unmatched. `fast` is 1 + J and `unassign` the probability of
/// staying unmatched as a fraction of 2^64.
#[cube]
fn ring(
    tables: &[u32],
    scratch: &mut [u32],
    rows: usize,
    n: u32,
    counts: &mut Array<u32>,
    rng: &mut Array<u32>,
    fast_low: u32,
    fast_high: u32,
    unassign_low: u32,
    unassign_high: u32,
) {
    let size = n as usize;
    let list = rows + 2 * size;
    // the 64-bit rates: conflicted (1 + J), and that plus the other rows
    let conflicted = counts[0];
    let c_low = conflicted * fast_low;
    let c_high = u32::mul_hi(conflicted, fast_low) + conflicted * fast_high;
    let others = n - conflicted;
    let r_low = c_low + others;
    let mut r_high = c_high;
    if r_low < c_low {
        r_high += 1;
    }
    // the high 64 bits of (64 random bits) x rate, by 32-bit limbs
    let x_high = next_u32(rng);
    let x_low = next_u32(rng);
    let middle_a = u32::mul_hi(x_low, r_low) + x_low * r_high;
    let mut carries = 0u32;
    if middle_a < x_low * r_high {
        carries += 1;
    }
    let middle = middle_a + x_high * r_low;
    if middle < middle_a {
        carries += 1;
    }
    let low_a = x_high * r_high + u32::mul_hi(x_low, r_high);
    let mut t_high = u32::mul_hi(x_high, r_high);
    if low_a < x_high * r_high {
        t_high += 1;
    }
    let low_b = low_a + u32::mul_hi(x_high, r_low);
    if low_b < low_a {
        t_high += 1;
    }
    let t_low = low_b + carries;
    if t_low < low_b {
        t_high += 1;
    }
    let mut u = 0u32;
    if below64(t_high, t_low, c_high, c_low) {
        u = scratch[list + u32::mul_hi(next_u32(rng), conflicted) as usize];
    } else {
        // uniform over the rows outside the list, by rejection
        let mut searching = true;
        while searching {
            u = u32::mul_hi(next_u32(rng), n);
            if !is_conflicted(scratch, list, counts, u) {
                searching = false;
            }
        }
    }
    remove(scratch, rows, size, counts, u);
    let d_high = next_u32(rng);
    let d_low = next_u32(rng);
    if !below64(d_high, d_low, unassign_high, unassign_low) {
        let v = alias_draw(tables, 0usize, u, n, rng);
        place(scratch, rows, size, counts, u, v);
    }
}

/// An exponential waiting time of rate 1 from 32 random bits.
#[cube]
fn exponential(bits: u32) -> f32 {
    0.0 - f32::ln((f32::cast_from(bits) + 0.5) * 2.328_306_4e-10)
}

/// The rate at which some clock rings, as a real number.
#[cube]
fn rate(n: u32, conflicted: u32, fast_low: u32, fast_high: u32) -> f32 {
    let low = conflicted * fast_low;
    let high = u32::mul_hi(conflicted, fast_low) + conflicted * fast_high;
    let total = low + (n - conflicted);
    let mut carry = 0u32;
    if total < low {
        carry = 1;
    }
    f32::cast_from(high + carry) * 4_294_967_296.0 + f32::cast_from(total)
}

/// The mutation kernel K_t (`permanent_core::chain::dynamics::mutate`) on
/// every particle that is not over, for at most `budget` events of this
/// dispatch. `times`: per particle, the time still to spend in perfect
/// matchings and the chain time elapsed. `tally[0]`: chains still running.
#[cube(launch_unchecked)]
pub fn mutate(
    tables: &[u32],
    sigma: &mut [u32],
    scratch: &mut [u32],
    state: &mut [u32],
    times: &mut [f32],
    tally: &[Atomic<u32>],
    n: u32,
    count: u32,
    perfect_time: f32,
    fast_low: u32,
    fast_high: u32,
    unassign_low: u32,
    unassign_high: u32,
    cap_low: u32,
    cap_high: u32,
    budget: u32,
) {
    let particle = ABSOLUTE_POS;
    let size = n as usize;
    let at = particle * STATE;
    let mut active = false;
    if particle < count as usize {
        active = (state[at + FLAGS] & OVER) == 0;
    }
    if active {
        let rows = particle * (2 * size + CONFLICTS);
        let mut rng = Array::<u32>::new(9usize);
        load_stream(state, at, &mut rng);
        let mut counts = Array::<u32>::new(3usize);
        counts[0] = 0;
        counts[1] = 0;
        counts[2] = 0;
        let mut remaining = perfect_time;
        let mut time = 0.0f32;
        let mut events_low = 0u32;
        let mut events_high = 0u32;
        if (state[at + FLAGS] & STARTED) == 0 {
            let mut u = 0usize;
            while u < size {
                let v = sigma[particle * size + u];
                scratch[rows + u] = v;
                scratch[rows + size + v as usize] = u as u32;
                u += 1;
            }
        } else {
            counts[0] = state[at + CONFLICTED];
            counts[1] = state[at + UNASSIGNED];
            events_low = state[at + EVENTS];
            events_high = state[at + EVENTS + 1];
            remaining = times[2 * particle];
            time = times[2 * particle + 1];
        }
        let mut flags = STARTED.runtime();
        let mut taken = 0u32;
        let mut running = true;
        while running {
            if taken >= budget {
                running = false;
            } else {
                let dt = exponential(next_u32(&mut rng)) / rate(n, counts[0], fast_low, fast_high);
                let mut ringing = true;
                if counts[0] == 0 && counts[1] == 0 {
                    if dt >= remaining {
                        time += remaining;
                        flags = STARTED | OVER;
                        ringing = false;
                        running = false;
                    } else {
                        remaining -= dt;
                    }
                }
                if ringing {
                    time += dt;
                    taken += 1;
                    events_low += 1;
                    if events_low == 0 {
                        events_high += 1;
                    }
                    if below64(cap_high, cap_low, events_high, events_low) {
                        flags = STARTED | OVER | CAPPED;
                        running = false;
                    } else {
                        ring(
                            tables,
                            scratch,
                            rows,
                            n,
                            &mut counts,
                            &mut rng,
                            fast_low,
                            fast_high,
                            unassign_low,
                            unassign_high,
                        );
                        if counts[2] != 0 {
                            flags = STARTED | OVER | CAPPED;
                            running = false;
                        }
                    }
                }
            }
        }
        if flags == (STARTED | OVER) {
            let mut u = 0usize;
            while u < size {
                sigma[particle * size + u] = scratch[rows + u];
                u += 1;
            }
        }
        if (flags & OVER) == 0 {
            tally[0].fetch_add(1);
        }
        store_stream(state, at, &mut rng);
        state[at + FLAGS] = flags;
        state[at + CONFLICTED] = counts[0];
        state[at + UNASSIGNED] = counts[1];
        state[at + EVENTS] = events_low;
        state[at + EVENTS + 1] = events_high;
        times[2 * particle] = remaining;
        times[2 * particle + 1] = time;
    }
}

/// One trial of the annealing sampler
/// (`permanent_core::chain::dynamics::trial`) on every particle that is not
/// over: from the all-unmatched state through every activity stage. A trial
/// that ends on a perfect matching writes it to the particle and sets
/// `PERFECT`. `tables`: the packed row alias table, then for each stage the
/// probability of staying unmatched as a fraction of 2^64.
/// `ticks_per_time`: 2^40 ticks over the stage time.
#[cube(launch_unchecked)]
pub fn anneal(
    tables: &[u32],
    sigma: &mut [u32],
    scratch: &mut [u32],
    state: &mut [u32],
    tally: &[Atomic<u32>],
    n: u32,
    count: u32,
    stages: u32,
    ticks_per_time: f32,
    fast_low: u32,
    fast_high: u32,
    cap_low: u32,
    cap_high: u32,
    budget: u32,
) {
    let particle = ABSOLUTE_POS;
    let size = n as usize;
    let at = particle * STATE;
    let mut active = false;
    if particle < count as usize {
        active = (state[at + FLAGS] & OVER) == 0;
    }
    if active {
        let rows = particle * (2 * size + CONFLICTS);
        let thresholds = 2 * size * size;
        let mut rng = Array::<u32>::new(9usize);
        load_stream(state, at, &mut rng);
        let mut counts = Array::<u32>::new(3usize);
        counts[0] = 0;
        counts[1] = n;
        counts[2] = 0;
        let mut stage = 0u32;
        let mut ticks_low = 0u32;
        let mut ticks_high = 0u32;
        let mut events_low = 0u32;
        let mut events_high = 0u32;
        if (state[at + FLAGS] & STARTED) == 0 {
            let mut u = 0usize;
            while u < size {
                scratch[rows + u] = NONE;
                scratch[rows + size + u] = NONE;
                u += 1;
            }
        } else {
            counts[0] = state[at + CONFLICTED];
            counts[1] = state[at + UNASSIGNED];
            events_low = state[at + EVENTS];
            events_high = state[at + EVENTS + 1];
            stage = state[at + STAGE];
            ticks_low = state[at + TICKS];
            ticks_high = state[at + TICKS + 1];
        }
        let mut flags = STARTED.runtime();
        let mut taken = 0u32;
        let mut running = true;
        while running {
            if taken >= budget {
                running = false;
            } else {
                // the waiting time in ticks, as a 64-bit integer
                let wait = exponential(next_u32(&mut rng)) * (ticks_per_time / rate(n, counts[0], fast_low, fast_high));
                let wait_high = u32::cast_from(wait / 4_294_967_296.0);
                let wait_low = u32::cast_from(wait - f32::cast_from(wait_high) * 4_294_967_296.0);
                let low = ticks_low + wait_low;
                let mut carry = 0u32;
                if low < ticks_low {
                    carry = 1;
                }
                ticks_low = low;
                ticks_high = ticks_high + wait_high + carry;
                // past 2^40 ticks the stage is over
                if ticks_high > 256 || (ticks_high == 256 && ticks_low > 0) {
                    stage += 1;
                    ticks_low = 0;
                    ticks_high = 0;
                    if stage == stages {
                        flags = STARTED | OVER;
                        running = false;
                    }
                } else {
                    taken += 1;
                    events_low += 1;
                    if events_low == 0 {
                        events_high += 1;
                    }
                    if below64(cap_high, cap_low, events_high, events_low) {
                        flags = STARTED | OVER | CAPPED;
                        running = false;
                    } else {
                        let threshold = thresholds + 2 * stage as usize;
                        ring(
                            tables,
                            scratch,
                            rows,
                            n,
                            &mut counts,
                            &mut rng,
                            fast_low,
                            fast_high,
                            tables[threshold],
                            tables[threshold + 1],
                        );
                        if counts[2] != 0 {
                            flags = STARTED | OVER | CAPPED;
                            running = false;
                        }
                    }
                }
            }
        }
        if flags == (STARTED | OVER) && counts[0] == 0 && counts[1] == 0 {
            flags = STARTED | OVER | PERFECT;
            let mut u = 0usize;
            while u < size {
                sigma[particle * size + u] = scratch[rows + u];
                u += 1;
            }
        }
        if (flags & OVER) == 0 {
            tally[0].fetch_add(1);
        }
        store_stream(state, at, &mut rng);
        state[at + FLAGS] = flags;
        state[at + CONFLICTED] = counts[0];
        state[at + UNASSIGNED] = counts[1];
        state[at + EVENTS] = events_low;
        state[at + EVENTS + 1] = events_high;
        state[at + STAGE] = stage;
        state[at + TICKS] = ticks_low;
        state[at + TICKS + 1] = ticks_high;
    }
}

/// Words `measure` writes for a particle: its energy and its
/// regularization log-gap as 64-bit sums, then how many of its entries
/// have a log-gap of minus infinity.
pub const MEASURED: usize = 5;

/// Sums over every particle's matching of the entries of `tables`: for each
/// cell its energy and its log-gap, 64-bit fixed point with 32 fraction
/// bits. A cell whose log-gap is minus infinity holds the smallest value
/// and is counted, not added.
#[cube(launch_unchecked)]
pub fn measure(tables: &[u32], sigma: &[u32], out: &mut [u32], n: u32, count: u32) {
    let particle = ABSOLUTE_POS;
    let size = n as usize;
    if particle < count as usize {
        let gaps = 2 * size * size;
        let mut e_low = 0u32;
        let mut e_high = 0u32;
        let mut g_low = 0u32;
        let mut g_high = 0u32;
        let mut infinite = 0u32;
        let mut u = 0usize;
        while u < size {
            let cell = 2 * (u * size + sigma[particle * size + u] as usize);
            let low = e_low + tables[cell];
            let mut carry = 0u32;
            if low < e_low {
                carry = 1;
            }
            e_low = low;
            e_high = e_high + tables[cell + 1] + carry;
            let gap_low = tables[gaps + cell];
            let gap_high = tables[gaps + cell + 1];
            if gap_high == 0x8000_0000 && gap_low == 0 {
                infinite += 1;
            } else {
                let low = g_low + gap_low;
                let mut carry = 0u32;
                if low < g_low {
                    carry = 1;
                }
                g_low = low;
                g_high = g_high + gap_high + carry;
            }
            u += 1;
        }
        let at = particle * MEASURED;
        out[at] = e_low;
        out[at + 1] = e_high;
        out[at + 2] = g_low;
        out[at + 3] = g_high;
        out[at + 4] = infinite;
    }
}

/// Resampling: particle p of `next` becomes a copy of particle
/// `ancestors[p]` of `sigma`.
#[cube(launch_unchecked)]
pub fn gather(sigma: &[u32], ancestors: &[u32], next: &mut [u32], n: u32, count: u32) {
    let particle = ABSOLUTE_POS;
    let size = n as usize;
    if particle < count as usize {
        let from = ancestors[particle] as usize * size;
        let mut u = 0usize;
        while u < size {
            next[particle * size + u] = sigma[from + u];
            u += 1;
        }
    }
}

/// Words `collect` writes for a chain: its state, then its matching.
pub const fn collected(n: usize) -> usize {
    STATE + n
}

/// Copy the state and the matching of the FAR chains of `targets` to `out`,
/// so the host can continue them.
#[cube(launch_unchecked)]
pub fn collect(targets: &[u32], scratch: &[u32], state: &[u32], out: &mut [u32], n: u32, count: u32) {
    let slot = ABSOLUTE_POS;
    let size = n as usize;
    if slot < count as usize {
        let particle = targets[slot] as usize;
        let to = slot * (STATE + size);
        let mut i = 0usize;
        while i < STATE {
            out[to + i] = state[particle * STATE + i];
            i += 1;
        }
        let mut u = 0usize;
        while u < size {
            out[to + STATE + u] = scratch[particle * 2 * size + u];
            u += 1;
        }
    }
}

/// Row p of `rows` replaces particle `targets[p]` of `sigma`.
#[cube(launch_unchecked)]
pub fn scatter(rows: &[u32], targets: &[u32], sigma: &mut [u32], n: u32, count: u32) {
    let slot = ABSOLUTE_POS;
    let size = n as usize;
    if slot < count as usize {
        let to = targets[slot] as usize * size;
        let mut u = 0usize;
        while u < size {
            sigma[to + u] = rows[slot * size + u];
            u += 1;
        }
    }
}
