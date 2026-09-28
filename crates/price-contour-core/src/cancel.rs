//! Cooperative cancellation for long single-pass calls.
//!
//! A [`CancelFlag`] is shared between the thread running a computation and
//! any thread that may want to stop it. Kernels poll it once per block of
//! quotes and return [`PriceContourError::Cancelled`] once it is set. Polling
//! never changes block boundaries or reduction order, so an uncancelled run is
//! bit-identical to a run without a flag. See `docs/DESIGN_DECISIONS.md` §14.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use crate::error::{PriceContourError, Result};

#[derive(Debug)]
struct FlagState {
    cancelled: AtomicBool,
    /// Polls so far: one per block a kernel checks.
    polls: AtomicU64,
    /// The poll count at which the flag sets itself; `u64::MAX` is never.
    /// A test hook ([`CancelFlag::cancel_after_polls`]) that cancels
    /// deterministically in the middle of a phase.
    trip_at: AtomicU64,
}

impl Default for FlagState {
    fn default() -> Self {
        Self {
            cancelled: AtomicBool::new(false),
            polls: AtomicU64::new(0),
            trip_at: AtomicU64::new(u64::MAX),
        }
    }
}

/// A shareable, one-way cancellation flag. Clones share the same flag.
#[derive(Clone, Debug, Default)]
pub struct CancelFlag(Arc<FlagState>);

impl CancelFlag {
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the flag. Idempotent; a flag cannot be reset.
    pub fn cancel(&self) {
        self.0.cancelled.store(true, Ordering::Release);
    }

    /// Read the flag without counting a poll (for callers, not kernels).
    pub fn peek(&self) -> bool {
        self.0.cancelled.load(Ordering::Acquire)
    }

    /// A kernel's poll: counts it, and reports whether the flag is set.
    pub fn is_cancelled(&self) -> bool {
        let polls = self.0.polls.fetch_add(1, Ordering::Relaxed) + 1;
        if polls >= self.0.trip_at.load(Ordering::Relaxed) {
            self.cancel();
        }
        self.peek()
    }

    /// `Err(Cancelled)` once the flag is set (a poll).
    pub fn check(&self) -> Result<()> {
        if self.is_cancelled() {
            Err(PriceContourError::Cancelled)
        } else {
            Ok(())
        }
    }

    /// Polls so far.
    pub fn polls(&self) -> u64 {
        self.0.polls.load(Ordering::Relaxed)
    }

    /// Test hook: the flag sets itself on the `n`-th poll from now (`n >= 1`),
    /// so a test can cancel at a precise point inside a phase.
    pub fn cancel_after_polls(&self, n: u64) {
        let at = self.polls().saturating_add(n.max(1));
        self.0.trip_at.store(at, Ordering::Relaxed);
    }
}

/// Whether an optional flag is set; `None` is never cancelled.
#[inline]
pub(crate) fn is_cancelled(cancel: Option<&CancelFlag>) -> bool {
    cancel.is_some_and(CancelFlag::is_cancelled)
}

/// `Err(Cancelled)` when an optional flag is set.
#[inline]
pub fn check_cancelled(cancel: Option<&CancelFlag>) -> Result<()> {
    cancel.map_or(Ok(()), CancelFlag::check)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clones_share_the_flag() {
        let flag = CancelFlag::new();
        let other = flag.clone();
        assert!(!other.is_cancelled());
        assert!(flag.check().is_ok());
        flag.cancel();
        assert!(other.is_cancelled());
        assert!(matches!(other.check(), Err(PriceContourError::Cancelled)));
        flag.cancel();
        assert!(flag.is_cancelled());
    }

    #[test]
    fn cancel_after_polls_trips_on_the_nth_poll() {
        let flag = CancelFlag::new();
        assert!(!flag.is_cancelled());
        flag.cancel_after_polls(3);
        assert!(!flag.is_cancelled());
        assert!(!flag.is_cancelled());
        assert!(!flag.peek());
        assert!(flag.is_cancelled());
        assert_eq!(flag.polls(), 4);
    }

    #[test]
    fn none_is_never_cancelled() {
        assert!(!is_cancelled(None));
        assert!(check_cancelled(None).is_ok());
    }
}
