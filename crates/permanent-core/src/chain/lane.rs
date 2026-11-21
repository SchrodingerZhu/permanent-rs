//! One particle's working memory: arrays of n 16-bit words, laid out with a
//! stride so that a block of GPU threads can interleave theirs in shared
//! memory. Element i of array a is at `base + (a n + i) stride`.

/// "No vertex": an unmatched row, a free column, or no hole.
pub const NONE: u32 = u16::MAX as u32;
/// Largest matrix whose vertices fit a word below the sentinels.
pub const MAX_N: usize = (NONE - 2) as usize;

/// The matching as row -> column.
pub const ROW: usize = 0;
/// The matching as column -> row.
pub const COL: usize = 1;

#[derive(Clone, Copy)]
pub struct Lane {
    base: *mut u16,
    stride: usize,
    pub n: usize,
}

impl Lane {
    /// # Safety
    /// `base` must be valid at this stride, for as long as the lane is used,
    /// for the words the chain addresses: two arrays of n for FAR,
    /// `matching::words(n)` for a relaxed assignment.
    #[inline(always)]
    pub unsafe fn new(base: *mut u16, stride: usize, n: usize) -> Self {
        Lane { base, stride, n }
    }

    /// A lane on host memory. `words` must outlive its use and hold what
    /// the chain addresses, as for [`Lane::new`].
    pub fn on(words: &mut [u16], n: usize) -> Self {
        assert!(n <= MAX_N && words.len() >= 2 * n);
        Lane { base: words.as_mut_ptr(), stride: 1, n }
    }

    /// # Safety
    /// `array` and `index` must be inside the lane.
    #[inline(always)]
    pub unsafe fn get(&self, array: usize, index: u32) -> u32 {
        unsafe { *self.base.add((array * self.n + index as usize) * self.stride) as u32 }
    }

    /// # Safety
    /// `array` and `index` must be inside the lane.
    #[inline(always)]
    pub unsafe fn set(&self, array: usize, index: u32, value: u32) {
        unsafe { *self.base.add((array * self.n + index as usize) * self.stride) = value as u16 }
    }
}
