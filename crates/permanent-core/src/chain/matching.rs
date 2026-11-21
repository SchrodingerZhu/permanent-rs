//! A relaxed assignment as the dynamics keeps it: every row is unmatched or
//! on one column, and columns may be shared.
//!
//! Besides the matching and its inverse, a particle keeps a short list of
//! the rows that share a column. The penalty J keeps collisions rare at
//! every activity, which is what makes a short list enough; a particle
//! that overflows it is treated like one that hit its event cap.

use super::lane::{COL, Lane, NONE, ROW};

/// In the column array: two or more rows are on this column.
pub const MULTI: u32 = NONE - 1;
/// Rows in a collision that the list can hold.
pub const CONFLICTS: usize = 32;
/// The collision list, after the two arrays of n.
const LIST: usize = 2;

/// Words of a lane that holds a relaxed assignment.
pub const fn words(n: usize) -> usize {
    2 * n + CONFLICTS
}

#[derive(Clone, Copy)]
pub struct Matching {
    lane: Lane,
    /// Rows in a collision.
    pub conflicted: u32,
    pub unassigned: u32,
    /// The collision list was full when a row had to be added.
    pub overflow: bool,
}

impl Matching {
    /// An assignment on `lane`, which must hold `words(n)` words. Fill it
    /// with [`Matching::load`] or [`Matching::clear`].
    #[inline(always)]
    pub fn new(lane: Lane) -> Self {
        Matching { lane, conflicted: 0, unassigned: 0, overflow: false }
    }

    #[inline(always)]
    pub fn is_perfect(&self) -> bool {
        self.conflicted == 0 && self.unassigned == 0
    }

    /// Start from the perfect matching stored at `stored` (n words).
    ///
    /// # Safety
    /// `stored` must be valid for n words and hold a permutation.
    #[inline(always)]
    pub unsafe fn load(&mut self, stored: *const u16) {
        let mut u = 0;
        while u < self.lane.n as u32 {
            unsafe {
                let v = *stored.add(u as usize) as u32;
                self.lane.set(ROW, u, v);
                self.lane.set(COL, v, u);
            }
            u += 1;
        }
        self.conflicted = 0;
        self.unassigned = 0;
        self.overflow = false;
    }

    /// Start with every row unmatched.
    ///
    /// # Safety
    /// The lane must hold `words(n)` words.
    #[inline(always)]
    pub unsafe fn clear(&mut self) {
        let n = self.lane.n as u32;
        let mut u = 0;
        while u < n {
            unsafe {
                self.lane.set(ROW, u, NONE);
                self.lane.set(COL, u, NONE);
            }
            u += 1;
        }
        self.conflicted = 0;
        self.unassigned = n;
        self.overflow = false;
    }

    /// Write the row -> column map to `stored`.
    ///
    /// # Safety
    /// `stored` must be valid for n words.
    #[inline(always)]
    pub unsafe fn store(&self, stored: *mut u16) {
        let mut u = 0;
        while u < self.lane.n as u32 {
            unsafe { *stored.add(u as usize) = self.lane.get(ROW, u) as u16 };
            u += 1;
        }
    }

    /// The `index`-th row of the collision list.
    ///
    /// # Safety
    /// `index` must be below `conflicted`.
    #[inline(always)]
    pub unsafe fn conflict(&self, index: u32) -> u32 {
        unsafe { self.lane.get(LIST, index) }
    }

    /// # Safety
    /// The lane must hold `words(n)` words.
    #[inline(always)]
    pub unsafe fn is_conflicted(&self, u: u32) -> bool {
        let mut i = 0;
        while i < self.conflicted {
            if unsafe { self.conflict(i) } == u {
                return true;
            }
            i += 1;
        }
        false
    }

    #[inline(always)]
    unsafe fn add_conflict(&mut self, u: u32) {
        if self.conflicted as usize == CONFLICTS {
            self.overflow = true;
            return;
        }
        unsafe { self.lane.set(LIST, self.conflicted, u) };
        self.conflicted += 1;
    }

    #[inline(always)]
    unsafe fn drop_conflict(&mut self, u: u32) {
        let mut i = 0;
        while i < self.conflicted {
            unsafe {
                if self.conflict(i) == u {
                    self.conflicted -= 1;
                    self.lane.set(LIST, i, self.conflict(self.conflicted));
                    return;
                }
            }
            i += 1;
        }
    }

    /// Unmatch row u.
    ///
    /// # Safety
    /// `u` must be below n and the lane hold `words(n)` words.
    #[inline(always)]
    pub unsafe fn remove(&mut self, u: u32) {
        unsafe {
            let v = self.lane.get(ROW, u);
            if v == NONE {
                return;
            }
            self.lane.set(ROW, u, NONE);
            self.unassigned += 1;
            if self.lane.get(COL, v) != MULTI {
                // u was alone on v
                self.lane.set(COL, v, NONE);
                return;
            }
            self.drop_conflict(u);
            // if one row is left on v, it is no longer in a collision
            let mut left = 0u32;
            let mut last = NONE;
            let mut i = 0;
            while i < self.conflicted {
                let w = self.conflict(i);
                if self.lane.get(ROW, w) == v {
                    left += 1;
                    last = w;
                }
                i += 1;
            }
            if left == 1 {
                self.drop_conflict(last);
                self.lane.set(COL, v, last);
            }
        }
    }

    /// Put the unmatched row u on column v.
    ///
    /// # Safety
    /// `u` and `v` must be below n and the lane hold `words(n)` words.
    #[inline(always)]
    pub unsafe fn place(&mut self, u: u32, v: u32) {
        unsafe {
            self.lane.set(ROW, u, v);
            self.unassigned -= 1;
            let occupant = self.lane.get(COL, v);
            if occupant == NONE {
                self.lane.set(COL, v, u);
            } else if occupant == MULTI {
                self.add_conflict(u);
            } else {
                self.lane.set(COL, v, MULTI);
                self.add_conflict(occupant);
                self.add_conflict(u);
            }
        }
    }
}
