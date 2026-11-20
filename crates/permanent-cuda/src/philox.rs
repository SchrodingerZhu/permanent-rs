//! Philox4x32-10 (Salmon et al., "Parallel Random Numbers: As Easy as 1, 2,
//! 3", SC'11), the counter-based generator of the GPU kernels. The key is
//! the particle's stream seed and the counter numbers its blocks of four
//! outputs, so streams of different particles are disjoint by construction
//! and a kernel needs no generator state from the host beyond the seed.

use cuda_device::device;

const PHILOX_M0: u32 = 0xD251_1F53;
const PHILOX_M1: u32 = 0xCD9E_8D57;
const PHILOX_W0: u32 = 0x9E37_79B9;
const PHILOX_W1: u32 = 0xBB67_AE85;

#[derive(Clone, Copy)]
pub struct Philox {
    key_0: u32,
    key_1: u32,
    block_lo: u32,
    block_hi: u32,
    lane_0: u32,
    lane_1: u32,
    lane_2: u32,
    lane_3: u32,
    /// Outputs of the current block not yet handed out.
    left: u32,
}

#[device]
pub fn philox_new(seed: u64) -> Philox {
    Philox {
        key_0: seed as u32,
        key_1: (seed >> 32) as u32,
        block_lo: 0,
        block_hi: 0,
        lane_0: 0,
        lane_1: 0,
        lane_2: 0,
        lane_3: 0,
        left: 0,
    }
}

/// The ten rounds on counter (c0, c1, c2, c3) under key (k0, k1).
#[device]
pub fn philox_block(mut c0: u32, mut c1: u32, mut c2: u32, mut c3: u32, mut k0: u32, mut k1: u32) -> (u32, u32, u32, u32) {
    let mut round = 0;
    while round < 10 {
        let p0 = PHILOX_M0 as u64 * c0 as u64;
        let p1 = PHILOX_M1 as u64 * c2 as u64;
        let next_0 = (p1 >> 32) as u32 ^ c1 ^ k0;
        let next_2 = (p0 >> 32) as u32 ^ c3 ^ k1;
        c1 = p1 as u32;
        c3 = p0 as u32;
        c0 = next_0;
        c2 = next_2;
        k0 = k0.wrapping_add(PHILOX_W0);
        k1 = k1.wrapping_add(PHILOX_W1);
        round += 1;
    }
    (c0, c1, c2, c3)
}

/// The next 32 uniform bits.
#[device]
pub fn philox_next(state: &mut Philox) -> u32 {
    if state.left == 0 {
        let (a, b, c, d) = philox_block(state.block_lo, state.block_hi, 0, 0, state.key_0, state.key_1);
        state.lane_0 = a;
        state.lane_1 = b;
        state.lane_2 = c;
        state.lane_3 = d;
        state.left = 4;
        state.block_lo = state.block_lo.wrapping_add(1);
        if state.block_lo == 0 {
            state.block_hi = state.block_hi.wrapping_add(1);
        }
    }
    state.left -= 1;
    if state.left == 3 {
        state.lane_0
    } else if state.left == 2 {
        state.lane_1
    } else if state.left == 1 {
        state.lane_2
    } else {
        state.lane_3
    }
}

/// The next 64 uniform bits.
#[device]
pub fn philox_next_u64(state: &mut Philox) -> u64 {
    let hi = philox_next(state) as u64;
    (hi << 32) | philox_next(state) as u64
}

#[cfg(test)]
mod test {
    use super::*;

    /// Known answers from the Random123 distribution (kat_vectors).
    #[test]
    fn matches_the_reference_vectors() {
        assert_eq!(philox_block(0, 0, 0, 0, 0, 0), (0x6627_e8d5, 0xe169_c58d, 0xbc57_ac4c, 0x9b00_dbd8));
        let f = u32::MAX;
        assert_eq!(philox_block(f, f, f, f, f, f), (0x408f_276d, 0x41c8_3b0e, 0xa20b_c7c6, 0x6d54_51fd));
        assert_eq!(
            philox_block(0x243f_6a88, 0x85a3_08d3, 0x1319_8a2e, 0x0370_7344, 0xa409_3822, 0x299f_31d0),
            (0xd16c_fe09, 0x94fd_cceb, 0x5001_e420, 0x2412_6ea1)
        );
    }

    #[test]
    fn hands_out_each_block_once() {
        let mut state = philox_new(0);
        let first: Vec<u32> = (0..4).map(|_| philox_next(&mut state)).collect();
        assert_eq!(first, [0x6627_e8d5, 0xe169_c58d, 0xbc57_ac4c, 0x9b00_dbd8]);
        let (a, _, _, _) = philox_block(1, 0, 0, 0, 0, 0);
        assert_eq!(philox_next(&mut state), a);
    }
}
