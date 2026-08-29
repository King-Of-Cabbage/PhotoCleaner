//! The vocabulary of per-file scan outcomes, and the rule that keeps it honest.
//!
//! The pipeline used to swallow errors. `exact_fingerprint` was called as
//! `if let Ok(..)`, so a hashing failure left `quick_hash` and `sha256` as
//! `None` while `scan_state` stayed `SUCCESS`; the file looked scanned and its
//! row said so. Elsewhere the states themselves disagreed with each other -
//! `FAILED` for a video probe, `DECODE_FAILED` for an image, `AI_FAILED` for
//! inference - with no single list to check against.
//!
//! [`StateAccumulator`] fixes the direction of travel: the first real failure
//! sticks, and no later stage can quietly report success over it.

/// Everything worked.
pub const SUCCESS: &str = "SUCCESS";
/// Nothing needed doing; the stored analysis was still valid.
pub const REUSED: &str = "REUSED";
/// The extension is outside the configured media set. Not a failure.
pub const UNSUPPORTED: &str = "UNSUPPORTED";

/// The file could not be opened or read at all.
pub const IO_FAILED: &str = "IO_FAILED";
/// Dimensions or container metadata could not be established.
pub const METADATA_FAILED: &str = "METADATA_FAILED";
/// An image could not be decoded.
pub const DECODE_FAILED: &str = "DECODE_FAILED";
/// Quick hash or SHA-256 failed.
pub const HASH_FAILED: &str = "HASH_FAILED";
/// A perceptual hash was required and could not be produced.
pub const PHASH_FAILED: &str = "PHASH_FAILED";
/// An embedding was required and inference failed.
pub const AI_FAILED: &str = "AI_FAILED";
/// Deep analysis was requested but the model or runtime is not usable.
pub const AI_UNAVAILABLE: &str = "AI_UNAVAILABLE";
/// ffprobe could not describe a video.
pub const VIDEO_PROBE_FAILED: &str = "VIDEO_PROBE_FAILED";

// There is deliberately no `LIVE_PHOTO_METADATA_FAILED`. Missing Apple pairing
// metadata is not a failure of the file's analysis - the file scanned fine, and
// plenty of perfectly good movies carry no content identifier. How sure the
// pairing is belongs on the pairing, and `scanner::apply_live_photo_pairing`
// says it there: `LIVE_PHOTO` when the identifier confirmed it,
// `PROBABLE_LIVE_PHOTO` when only the filenames agreed, `UNPAIRED_LIVE_PHOTO`
// when the container claims a partner that was never scanned. A state here
// would have counted ordinary videos as failures, which is the same false alarm
// that `UNSUPPORTED` used to raise.

/// Every state this build can write, in the order a report should list them.
pub const ALL: &[&str] = &[
    SUCCESS,
    REUSED,
    UNSUPPORTED,
    IO_FAILED,
    METADATA_FAILED,
    DECODE_FAILED,
    HASH_FAILED,
    PHASH_FAILED,
    AI_FAILED,
    AI_UNAVAILABLE,
    VIDEO_PROBE_FAILED,
];

/// True when the state means the file was not fully analysed.
///
/// `UNSUPPORTED` is deliberately not a failure: the file is simply not media
/// this tool handles, and mixing it into the failure counts made every scan of
/// a folder containing a `.txt` look broken.
pub fn is_failure(state: &str) -> bool {
    !matches!(state, SUCCESS | REUSED | UNSUPPORTED)
}

/// Ranks states so a later stage cannot report success over an earlier failure.
pub fn severity(state: &str) -> u8 {
    match state {
        SUCCESS | REUSED => 0,
        UNSUPPORTED => 1,
        _ => 2,
    }
}

/// Accumulates the outcome of one file across the pipeline stages.
///
/// Records the *first* failure. The earliest stage to fail is the most
/// informative one: if a file cannot be read, that it also has no pHash is a
/// consequence, not a second problem.
#[derive(Clone, Debug)]
pub struct StateAccumulator {
    state: String,
    stage: Option<String>,
    message: Option<String>,
}

impl Default for StateAccumulator {
    fn default() -> Self {
        Self::new()
    }
}

impl StateAccumulator {
    pub fn new() -> Self {
        Self {
            state: SUCCESS.to_string(),
            stage: None,
            message: None,
        }
    }

    /// Starts from an already-known state, such as the one a probe returned.
    pub fn from_state(state: &str, stage: Option<&str>, message: Option<&str>) -> Self {
        Self {
            state: state.to_string(),
            stage: stage.map(str::to_string),
            message: message.map(str::to_string),
        }
    }

    pub fn state(&self) -> &str {
        &self.state
    }

    pub fn is_failed(&self) -> bool {
        is_failure(&self.state)
    }

    /// Records a failure, unless one is already recorded.
    pub fn fail(&mut self, state: &str, stage: &str, message: impl Into<String>) {
        if self.is_failed() {
            return;
        }
        self.state = state.to_string();
        self.stage = Some(stage.to_string());
        self.message = Some(message.into());
    }

    /// Records a non-failure state, but never downgrades an existing failure.
    pub fn note(&mut self, state: &str) {
        if severity(state) >= severity(&self.state) && !self.is_failed() {
            self.state = state.to_string();
        }
    }

    /// `(state, failure_stage, failure_message)` as stored on the row.
    pub fn finish(self) -> (String, Option<String>, Option<String>) {
        (self.state, self.stage, self.message)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unsupported_is_not_counted_as_a_failure() {
        assert!(!is_failure(SUCCESS));
        assert!(!is_failure(REUSED));
        assert!(!is_failure(UNSUPPORTED));
        assert!(is_failure(HASH_FAILED));
        assert!(is_failure(VIDEO_PROBE_FAILED));
    }

    #[test]
    fn a_later_success_cannot_erase_an_earlier_failure() {
        let mut state = StateAccumulator::new();
        state.fail(HASH_FAILED, "EXACT_HASH", "permission denied");
        state.note(SUCCESS);
        let (final_state, stage, message) = state.finish();
        assert_eq!(final_state, HASH_FAILED);
        assert_eq!(stage.as_deref(), Some("EXACT_HASH"));
        assert_eq!(message.as_deref(), Some("permission denied"));
    }

    #[test]
    fn the_first_failure_is_the_one_that_is_reported() {
        let mut state = StateAccumulator::new();
        state.fail(IO_FAILED, "OPEN", "the file is locked");
        state.fail(PHASH_FAILED, "PHASH", "decode failed");
        let (final_state, stage, message) = state.finish();
        assert_eq!(
            final_state, IO_FAILED,
            "the root cause must win over its consequence"
        );
        assert_eq!(stage.as_deref(), Some("OPEN"));
        assert_eq!(message.as_deref(), Some("the file is locked"));
    }

    #[test]
    fn a_clean_run_reports_success_with_no_detail() {
        let mut state = StateAccumulator::new();
        state.note(SUCCESS);
        let (final_state, stage, message) = state.finish();
        assert_eq!(final_state, SUCCESS);
        assert!(stage.is_none());
        assert!(message.is_none());
    }

    #[test]
    fn a_probe_failure_can_seed_the_accumulator() {
        let mut state = StateAccumulator::from_state(
            VIDEO_PROBE_FAILED,
            Some("VIDEO_PROBE"),
            Some("ffprobe exited with 1"),
        );
        assert!(state.is_failed());
        state.fail(AI_FAILED, "AI", "inference failed");
        let (final_state, stage, _) = state.finish();
        assert_eq!(final_state, VIDEO_PROBE_FAILED);
        assert_eq!(stage.as_deref(), Some("VIDEO_PROBE"));
    }

    #[test]
    fn unsupported_survives_a_note_of_success() {
        let mut state = StateAccumulator::from_state(UNSUPPORTED, None, None);
        state.note(SUCCESS);
        assert_eq!(state.state(), UNSUPPORTED);
    }

    #[test]
    fn every_listed_state_is_classified() {
        for state in ALL {
            // Each state is either a failure or one of the three benign ones;
            // this catches a constant being added without being classified.
            let benign = matches!(*state, SUCCESS | REUSED | UNSUPPORTED);
            assert_eq!(!benign, is_failure(state), "unclassified state {state}");
        }
    }
}
