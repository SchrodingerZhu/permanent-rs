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
// the stream: key, blocks generated, the last block, outputs of it left
const KEY: usize = 0;
const BLOCK: usize = 2;
const LANES: usize = 4;
const LEFT: usize = 8;
/// Set once a chain has started, when it is over, and if it ran out of steps.
pub const FLAGS: usize = 9;
pub const STARTED: u32 = 1;
pub const OVER: u32 = 2;
pub const CAPPED: u32 = 4;
// the FAR chain
pub const HOLE_U: usize = 10;
pub const HOLE_V: usize = 11;
pub const CYCLES: usize = 12;
pub const STEPS: usize = 13;
pub const ENERGY: usize = 14;
pub const PERFECT_ENERGY: usize = 16;

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
    rng[(7 - left) as usize]
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
/// not over, for at most `budget` steps of this dispatch.
///
/// `tables`: the packed row alias table, the column one, the move
/// thresholds and the entry energies, each entry 64 bits. `scratch`: the
/// matching and its inverse, 2 n words a particle. `tally`: chains still
/// running, then for each column the 64-bit energy sum of the visits to
/// its hole class, then the visits of each group to each class.
#[cube(launch_unchecked)]
pub fn far(
    tables: &[u32],
    sigma: &mut [u32],
    scratch: &mut [u32],
    state: &mut [u32],
    tally: &[Atomic<u32>],
    n: u32,
    count: u32,
    anchor: u32,
    groups: u32,
    cycles: u32,
    budget: u32,
    max_steps: u32,
) {
    let particle = ABSOLUTE_POS;
    let size = n as usize;
    let at = particle * STATE;
    let mut active = false;
    if particle < count as usize {
        active = (state[at + FLAGS] & OVER) == 0;
    }
    if active {
        let rows = particle * 2 * size;
        let cols = rows + size;
        let col_alias = 2 * size * size;
        let moves = 4 * size * size;
        let energies = 8 * size * size;
        let mut rng = Array::<u32>::new(9usize);
        let mut i = 0usize;
        while i < 9 {
            rng[i] = state[at + i];
            i += 1;
        }
        let mut hole_u = NONE.runtime();
        let mut hole_v = 0u32;
        let mut done = 0u32;
        let mut steps = 0u32;
        let mut h_low = 0u32;
        let mut h_high = 0u32;
        let mut sum_low = 0u32;
        let mut sum_high = 0u32;
        if (state[at + FLAGS] & STARTED) == 0 {
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
        let visits = 1 + 2 * size + group as usize * size;
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
                    // 64-bit sum from two words: the previous low word tells
                    // this thread whether its addition carried
                    let before = tally[1 + 2 * hole_v as usize].fetch_add(h_low);
                    let mut carry = 0u32;
                    if before + h_low < before {
                        carry = 1;
                    }
                    tally[2 + 2 * hole_v as usize].fetch_add(h_high + carry);
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
        } else {
            tally[0].fetch_add(1);
        }
        let mut i = 0usize;
        while i < 9 {
            state[at + i] = rng[i];
            i += 1;
        }
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
