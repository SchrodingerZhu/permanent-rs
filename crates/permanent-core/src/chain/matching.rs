//! A relaxed assignment as the dynamics keeps it: every row is unmatched or
//! on one column, and columns may be shared.
//!
//! An event asks who is on a column, never which column a given row is on,
//! so the state is one array over the columns and a short list, with no
//! row -> column array. Every row has a seat, a column of its own: a row
//! alone on a column sits there, an unmatched row sits on a free column, and
//! a row that shares a column is on the list with that column and its seat.
//! The seats are a bijection between rows and columns, so the row seated on
//! a uniform column is a uniform row, which is how the dynamics draws a row
//! outside the list.
//!
//! The penalty J keeps collisions rare at every activity, which is what
//! makes a short list enough; a particle that overflows it is treated like
//! one that hit its event cap.

use super::lane::{Lane, MAX_N, NONE};

/// Rows in a collision that the list can hold.
pub const CONFLICTS: usize = 32;

/// The arrays of the lane: what sits on each column, then the list.
///
/// The word of a column is the row alone on it; or `PARKED` and the
/// unmatched row seated there, when the column is free; or `NONE` when it is
/// free and the seat of a row of the list; or `MULTI` when two or more rows
/// are on it, one of which has its seat there.
const SEAT: usize = 0;
const LIST: usize = 1;
const MULTI: u32 = NONE - 1;
const PARKED: u32 = 1 << 15;

/// The words of a list entry: the row, the column it shares, and its seat.
const ENTRY: u32 = 3;
const ENTRY_ROW: u32 = 0;
const ENTRY_COLUMN: u32 = 1;
const ENTRY_SEAT: u32 = 2;

/// Words of a lane that holds a relaxed assignment.
pub const fn words(n: usize) -> usize {
    n + ENTRY as usize * CONFLICTS
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

    /// An assignment on host memory, which must outlive its use.
    pub fn on(memory: &mut [u16], n: usize) -> Self {
        assert!(n <= MAX_N && memory.len() >= words(n));
        // SAFETY: the memory holds the words of a relaxed assignment
        Matching::new(unsafe { Lane::new(memory.as_mut_ptr(), 1, n) })
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
            unsafe { self.lane.set(SEAT, *stored.add(u as usize) as u32, u) };
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
            unsafe { self.lane.set(SEAT, u, PARKED | u) };
            u += 1;
        }
        self.conflicted = 0;
        self.unassigned = n;
        self.overflow = false;
    }

    /// Write the row -> column map of a perfect matching to `stored`.
    ///
    /// # Safety
    /// The matching must be perfect and `stored` valid for n words.
    #[inline(always)]
    pub unsafe fn store(&self, stored: *mut u16) {
        let mut v = 0;
        while v < self.lane.n as u32 {
            unsafe { *stored.add(self.lane.get(SEAT, v) as usize) = v as u16 };
            v += 1;
        }
    }

    #[inline(always)]
    unsafe fn entry(&self, index: u32, word: u32) -> u32 {
        unsafe { self.lane.get(LIST, ENTRY * index + word) }
    }

    #[inline(always)]
    unsafe fn set_entry(&self, index: u32, word: u32, value: u32) {
        unsafe { self.lane.set(LIST, ENTRY * index + word, value) }
    }

    #[inline(always)]
    unsafe fn add_conflict(&mut self, u: u32, v: u32, seat: u32) {
        if self.conflicted as usize == CONFLICTS {
            self.overflow = true;
            return;
        }
        unsafe {
            self.set_entry(self.conflicted, ENTRY_ROW, u);
            self.set_entry(self.conflicted, ENTRY_COLUMN, v);
            self.set_entry(self.conflicted, ENTRY_SEAT, seat);
        }
        self.conflicted += 1;
    }

    /// Take the `index`-th row off the collision list.
    #[inline(always)]
    unsafe fn drop_conflict(&mut self, index: u32) {
        self.conflicted -= 1;
        if index != self.conflicted {
            let mut word = 0;
            while word < ENTRY {
                unsafe { self.set_entry(index, word, self.entry(self.conflicted, word)) };
                word += 1;
            }
        }
    }

    /// Unmatch the row seated on column `seat` and return it, or `NONE`
    /// when no row outside the list sits there. The caller goes on to
    /// [`Matching::place`] or [`Matching::park`] the row from that seat.
    ///
    /// # Safety
    /// `seat` must be below n and the lane hold `words(n)` words.
    #[inline(always)]
    pub unsafe fn lift(&mut self, seat: u32) -> u32 {
        let word = unsafe { self.lane.get(SEAT, seat) };
        if word >= MULTI {
            return NONE;
        }
        if word & PARKED == 0 {
            self.unassigned += 1;
        }
        word & !PARKED
    }

    /// Unmatch the `index`-th row of the collision list and return it with
    /// its seat, for the caller to [`Matching::place`] or
    /// [`Matching::park`] it from there.
    ///
    /// # Safety
    /// `index` must be below `conflicted` and the lane hold `words(n)` words.
    #[inline(always)]
    pub unsafe fn lift_conflict(&mut self, index: u32) -> (u32, u32) {
        unsafe {
            let u = self.entry(index, ENTRY_ROW);
            let v = self.entry(index, ENTRY_COLUMN);
            let mut seat = self.entry(index, ENTRY_SEAT);
            self.unassigned += 1;
            // the other rows on v: how many, and one of them
            let mut left = 0u32;
            let mut other = 0;
            let mut i = 0;
            while i < self.conflicted {
                if i != index && self.entry(i, ENTRY_COLUMN) == v {
                    left += 1;
                    other = i;
                }
                i += 1;
            }
            if seat == v {
                // the seat on v stays with a row that is on v
                seat = self.entry(other, ENTRY_SEAT);
                self.set_entry(other, ENTRY_SEAT, v);
            }
            if left == 1 {
                // the other row is alone on v, where it sits by now. The
                // later entry goes first, so that the earlier one stays put.
                self.lane.set(SEAT, v, self.entry(other, ENTRY_ROW));
                self.drop_conflict(index.max(other));
                self.drop_conflict(index.min(other));
            } else {
                self.drop_conflict(index);
            }
            (u, seat)
        }
    }

    /// Leave the unmatched row u on its seat.
    ///
    /// # Safety
    /// u must have been lifted from `seat`.
    #[inline(always)]
    pub unsafe fn park(&mut self, u: u32, seat: u32) {
        unsafe { self.lane.set(SEAT, seat, PARKED | u) };
    }

    /// Put the unmatched row u, lifted from `seat`, on column v.
    ///
    /// # Safety
    /// u must have been lifted from `seat`, and v be below n.
    #[inline(always)]
    pub unsafe fn place(&mut self, u: u32, seat: u32, v: u32) {
        self.unassigned -= 1;
        unsafe {
            if v == seat {
                self.lane.set(SEAT, v, u);
                return;
            }
            let word = self.lane.get(SEAT, v);
            if word == MULTI {
                self.lane.set(SEAT, seat, NONE);
                self.add_conflict(u, v, seat);
            } else if word == NONE {
                // v is free, and u takes it for its seat from a row of the
                // list, which gets the seat u leaves
                let mut i = 0;
                while i < self.conflicted {
                    if self.entry(i, ENTRY_SEAT) == v {
                        self.set_entry(i, ENTRY_SEAT, seat);
                    }
                    i += 1;
                }
                self.lane.set(SEAT, seat, NONE);
                self.lane.set(SEAT, v, u);
            } else if word & PARKED != 0 {
                // v is free, and the unmatched row seated there swaps seats with u
                self.lane.set(SEAT, seat, word);
                self.lane.set(SEAT, v, u);
            } else {
                // the row alone on v keeps its seat there
                self.lane.set(SEAT, v, MULTI);
                self.lane.set(SEAT, seat, NONE);
                self.add_conflict(word, v, v);
                self.add_conflict(u, v, seat);
            }
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    /// The row -> column map the state stands for, after checking that the
    /// seats are a bijection and that the counters agree with it.
    fn assignment(matching: &Matching, n: usize) -> Vec<u32> {
        let mut column = vec![NONE; n];
        let mut seats = vec![0; n];
        let mut unassigned = 0;
        for v in 0..n as u32 {
            let word = unsafe { matching.lane.get(SEAT, v) };
            if word < MULTI {
                let u = (word & !PARKED) as usize;
                seats[u] += 1;
                if word & PARKED == 0 {
                    column[u] = v;
                } else {
                    unassigned += 1;
                }
            }
        }
        for i in 0..matching.conflicted {
            let (u, v, seat) = unsafe {
                (matching.entry(i, ENTRY_ROW), matching.entry(i, ENTRY_COLUMN), matching.entry(i, ENTRY_SEAT))
            };
            seats[u as usize] += 1;
            column[u as usize] = v;
            let word = unsafe { matching.lane.get(SEAT, seat) };
            assert!(word == if seat == v { MULTI } else { NONE });
            assert_eq!(unsafe { matching.lane.get(SEAT, v) }, MULTI);
        }
        assert!(seats.iter().all(|&count| count == 1), "seats {seats:?}");
        assert_eq!(matching.unassigned, unassigned);
        column
    }

    /// Random events on a small matrix, against the row -> column map
    /// kept directly.
    #[test]
    fn lifting_and_placing_rows_keep_the_assignment() {
        let n = 6;
        let mut rng = SmallRng::seed_from_u64(3);
        let mut memory = vec![0u16; words(n)];
        let mut matching = Matching::on(&mut memory, n);
        let mut expected = vec![NONE; n];
        unsafe { matching.clear() };
        for step in 0..200_000 {
            if step % 1000 == 500 {
                let start: Vec<u16> = vec![3, 0, 5, 1, 2, 4];
                unsafe { matching.load(start.as_ptr()) };
                expected = start.iter().map(|&v| v as u32).collect();
            }
            let (u, seat) = unsafe {
                if matching.conflicted > 0 && rng.random_bool(0.6) {
                    matching.lift_conflict(rng.random_range(0..matching.conflicted))
                } else {
                    let seat = rng.random_range(0..n as u32);
                    (matching.lift(seat), seat)
                }
            };
            if u == NONE {
                continue;
            }
            if rng.random_bool(0.2) {
                unsafe { matching.park(u, seat) };
                expected[u as usize] = NONE;
            } else {
                let v = rng.random_range(0..n as u32);
                unsafe { matching.place(u, seat, v) };
                expected[u as usize] = v;
            }
            assert!(!matching.overflow);
            assert_eq!(assignment(&matching, n), expected, "step {step}");
            let shared =
                (0..n).filter(|&u| expected[u] != NONE && expected.iter().filter(|&&v| v == expected[u]).count() > 1);
            assert_eq!(matching.conflicted as usize, shared.count());
            if matching.is_perfect() {
                let mut stored = vec![0u16; n];
                unsafe { matching.store(stored.as_mut_ptr()) };
                assert!(stored.iter().zip(&expected).all(|(&stored, &v)| stored as u32 == v));
            }
        }
    }
}
