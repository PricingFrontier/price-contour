//! Cooperative cancellation for long single-pass calls.
//!
//! A [`CancelFlag`] is shared between the thread running a computation and
//! any thread that may want to stop it. Kernels poll it once per block of
//! quotes and return [`PriceContourError::Cancelled`] once it is set. Polling
//! never changes block boundaries or reduction order, so an uncancelled run is
//! bit-identical to a run without a flag. See `docs/DESIGN_DECISIONS.md` §14.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use crate::error::{PriceContourError, Result};

/// A shareable, one-way cancellation flag. Clones share the same flag.
#[derive(Clone, Debug, Default)]
pub struct CancelFlag(Arc<AtomicBool>);

impl CancelFlag {
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the flag. Idempotent; a flag cannot be reset.
    pub fn cancel(&self) {
        self.0.store(true, Ordering::Release);
    }

    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }

    /// `Err(Cancelled)` once the flag is set.
    pub fn check(&self) -> Result<()> {
        if self.is_cancelled() {
            Err(PriceContourError::Cancelled)
        } else {
            Ok(())
        }
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
    fn none_is_never_cancelled() {
        assert!(!is_cancelled(None));
        assert!(check_cancelled(None).is_ok());
    }
}
