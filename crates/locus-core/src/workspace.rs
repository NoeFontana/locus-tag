//! Thread-local workspace arena for per-candidate ephemeral allocations.
//!
//! Shared across sequential pipeline stages (quad extraction, decoding).
//! Each Rayon worker thread owns an independent instance, reset at the
//! start of every candidate iteration.

use bumpalo::Bump;
use std::cell::RefCell;

thread_local! {
    /// Reusable per-thread arena shared across pipeline stages.
    pub(crate) static WORKSPACE_ARENA: RefCell<Bump> =
        RefCell::new(Bump::with_capacity(8 * 1024));
}
