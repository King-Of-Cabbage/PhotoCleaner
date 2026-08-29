//! The single owner of scan progress.
//!
//! Progress used to be an `Arc<Mutex<ScanProgress>>` that five different places
//! wrote to directly, and the results contradicted each other:
//!
//! * every worker set `stage = MediaProbe` on every candidate, while the
//!   database writer thread - which runs *concurrently* with the workers - set
//!   `stage = DatabaseFinalize` and `stage_completed = 0` every time it flushed
//!   a batch. The stage therefore flipped back and forth for the whole scan and
//!   the stage bar restarted from zero over and over;
//! * `completed_assets` was assigned from three places, one of which reset it
//!   to `0` after discovery and another of which forced it to `total`, so it
//!   could go backwards on screen;
//! * each of the three loops carried its own copy of the 100 ms emit throttle.
//!
//! Here there is exactly one mutable owner. Callers describe *what happened*
//! ([`ProgressEvent`]); [`ProgressCoordinator`] decides what that means for the
//! numbers, and is the only code that may write them. Two rules are enforced
//! rather than hoped for:
//!
//! 1. **`completed_assets` is monotonic.** It never decreases and never passes
//!    `total_assets`.
//! 2. **The stage has one owner.** Only [`ProgressEvent::EnterStage`] changes
//!    it, and only the pipeline sends that. Workers and the database writer can
//!    describe their activity and their queue depths, but cannot claim a stage.

use std::sync::Mutex;
use std::time::{Duration, Instant};

use super::{ScanProgress, ScanStage};

/// How often progress is pushed to the UI while a stage is running.
const EMIT_INTERVAL: Duration = Duration::from_millis(100);

/// What one finished file did, as counted for the summary.
///
/// The worker decides these from the scanned file and the snapshot map; the
/// coordinator only adds them up. Keeping the two apart is what stops a new
/// scan state from being silently counted as nothing.
#[derive(Clone, Copy, Debug, Default)]
pub struct FileCounters {
    pub reused: bool,
    pub unsupported: bool,
    pub new: bool,
    pub updated: bool,
    pub failed: bool,
    pub standard_computed: bool,
    pub ai_computed: bool,
    pub ai_reused: bool,
    pub ai_stale: bool,
}

/// The report a worker files when one candidate is done.
#[derive(Clone, Debug)]
pub struct FileFinished {
    /// True when this file was the last component of its asset.
    pub asset_completed: bool,
    /// Wall time for this file, used for the smoothed throughput and the ETA.
    pub elapsed: Duration,
    pub counters: FileCounters,
    pub decode_queue_len: usize,
    pub db_queue_len: usize,
}

/// Everything that is allowed to change progress.
#[derive(Clone, Debug)]
pub enum ProgressEvent {
    /// One more file found during discovery. The total is not yet known.
    FileDiscovered,
    /// One discovered file was not media. Counted, never treated as a failure.
    UnsupportedDiscovered,
    /// A directory entry could not be read during the walk. This one *is* a
    /// failure: the file exists and could not be looked at.
    DiscoveryFailed,
    /// Discovery is over and the totals are final.
    DiscoveryFinished {
        asset_total: usize,
        file_total: usize,
        /// Assets already complete before any worker ran, i.e. the unsupported
        /// files, which have no work to do.
        already_complete: usize,
    },
    /// The pipeline moves to a new stage. The only event that sets the stage.
    EnterStage {
        stage: ScanStage,
        activity: String,
        completed: usize,
        total: usize,
    },
    /// A worker picked up a file. Describes activity only - not a stage.
    FileStarted {
        activity: String,
        metadata_queue_len: usize,
    },
    /// A worker finished a file.
    FileFinished(FileFinished),
    /// The database writer reports how far behind it is.
    DatabaseQueue { db_queue_len: usize },
    /// The scan is over.
    Finished { completed: usize },
}

pub struct ProgressCoordinator {
    state: ScanProgress,
}

impl ProgressCoordinator {
    pub fn new(worker_count: usize) -> Self {
        Self {
            state: ScanProgress {
                worker_count,
                ..ScanProgress::default()
            },
        }
    }

    pub fn snapshot(&self) -> ScanProgress {
        self.state.clone()
    }

    pub fn apply(&mut self, event: ProgressEvent) {
        match event {
            ProgressEvent::FileDiscovered => {
                self.state.discovered += 1;
                self.state.discovered_files = self.state.discovered;
                // Before discovery ends the total is only a lower bound, which
                // is what `total_known` tells the UI.
                self.state.total = self.state.discovered;
            }
            ProgressEvent::UnsupportedDiscovered => {
                self.state.unsupported_files += 1;
            }
            ProgressEvent::DiscoveryFailed => {
                self.state.failed_files += 1;
            }
            ProgressEvent::DiscoveryFinished {
                asset_total,
                file_total,
                already_complete,
            } => {
                self.state.total_known = true;
                self.state.total_assets = asset_total;
                self.state.total = asset_total;
                self.state.processed_files = 0;
                self.set_completed(already_complete);
                self.state.stage_completed = 0;
                self.state.stage_total = file_total;
            }
            ProgressEvent::EnterStage {
                stage,
                activity,
                completed,
                total,
            } => {
                self.state.stage = stage;
                self.state.activity = activity;
                self.state.stage_completed = completed;
                self.state.stage_total = total;
            }
            ProgressEvent::FileStarted {
                activity,
                metadata_queue_len,
            } => {
                self.state.activity = activity;
                self.state.metadata_queue_len = metadata_queue_len;
                self.state.active_workers += 1;
                self.state.processing += 1;
            }
            ProgressEvent::FileFinished(finished) => self.apply_file_finished(finished),
            ProgressEvent::DatabaseQueue { db_queue_len } => {
                self.state.db_queue_len = db_queue_len;
            }
            ProgressEvent::Finished { completed } => {
                self.state.stage = ScanStage::Done;
                self.state.activity = "扫描完成".to_string();
                self.state.total = completed;
                self.state.total_assets = completed;
                self.set_completed(completed);
                self.state.processing = 0;
                self.state.active_workers = 0;
                self.state.stage_total = completed;
                self.state.stage_completed = completed;
                self.state.eta = Some(Duration::ZERO);
            }
        }
    }

    fn apply_file_finished(&mut self, finished: FileFinished) {
        self.state.active_workers = self.state.active_workers.saturating_sub(1);
        self.state.processing = self.state.processing.saturating_sub(1);
        self.state.processed_files += 1;
        self.state.decode_queue_len = finished.decode_queue_len;
        self.state.db_queue_len = finished.db_queue_len;
        if self.state.stage_total == 0 || self.state.stage_completed < self.state.stage_total {
            self.state.stage_completed += 1;
        }
        if finished.asset_completed {
            self.set_completed(self.state.completed + 1);
        }

        let counters = finished.counters;
        if counters.reused {
            self.state.reused_files += 1;
        }
        if counters.unsupported {
            self.state.unsupported_files += 1;
        }
        if counters.new {
            self.state.new_files += 1;
        }
        if counters.updated {
            self.state.updated_files += 1;
        }
        if counters.failed {
            self.state.failed_files += 1;
        }
        if counters.standard_computed {
            self.state.standard_computed += 1;
        } else {
            self.state.standard_reused += 1;
        }
        if counters.ai_computed {
            self.state.ai_computed += 1;
        }
        if counters.ai_reused {
            self.state.ai_reused += 1;
        }
        if counters.ai_stale {
            self.state.ai_stale += 1;
        }

        self.update_throughput(finished.elapsed);
    }

    /// The one place `completed` and `completed_assets` are written.
    ///
    /// They are the same number and always have been; keeping two fields in
    /// sync from three call sites is exactly how they came apart.
    fn set_completed(&mut self, value: usize) {
        let capped = if self.state.total_known {
            value.min(self.state.total_assets)
        } else {
            value
        };
        // Monotonic: a progress bar that goes backwards is read as a bug in the
        // scan, and it hid the real double-counting that caused it.
        let next = capped.max(self.state.completed);
        if capped < self.state.completed {
            crate::logging::error(format!(
                "Progress went backwards: {} -> {}; holding at {}",
                self.state.completed, capped, self.state.completed
            ));
        }
        if self.state.total_known && value > self.state.total_assets {
            crate::logging::error(format!(
                "Progress invariant violated: completed_assets={} total_assets={}",
                value, self.state.total_assets
            ));
        }
        self.state.completed = next;
        self.state.completed_assets = next;
    }

    fn update_throughput(&mut self, elapsed: Duration) {
        let instant = if elapsed.is_zero() {
            0.0
        } else {
            1.0 / elapsed.as_secs_f64()
        };
        self.state.throughput = if self.state.throughput <= 0.0 {
            instant
        } else {
            self.state.throughput * 0.8 + instant * 0.2
        };
        let remaining = self.state.total.saturating_sub(self.state.completed);
        self.state.eta = if self.state.completed < 10 || self.state.throughput <= 0.0 {
            None
        } else {
            Some(Duration::from_secs_f64(
                remaining as f64 / self.state.throughput,
            ))
        };
    }
}

/// The coordinator plus the emit throttle, shared by every thread in the scan.
///
/// The callback is boxed rather than generic so that the worker, the discovery
/// walk and the database writer can all take a plain `Arc<ProgressHub>` instead
/// of each growing its own type parameter.
pub struct ProgressHub {
    inner: Mutex<HubInner>,
    emit: Box<dyn Fn(ScanProgress) + Send + Sync>,
}

struct HubInner {
    coordinator: ProgressCoordinator,
    last_emit: Instant,
}

impl ProgressHub {
    pub fn new(worker_count: usize, emit: impl Fn(ScanProgress) + Send + Sync + 'static) -> Self {
        Self {
            inner: Mutex::new(HubInner {
                coordinator: ProgressCoordinator::new(worker_count),
                // Subtracting can underflow the platform clock's epoch, so the
                // first emit is simply allowed through instead.
                last_emit: Instant::now()
                    .checked_sub(EMIT_INTERVAL)
                    .unwrap_or_else(Instant::now),
            }),
            emit: Box::new(emit),
        }
    }

    /// Applies an event, emitting at most once per [`EMIT_INTERVAL`].
    pub fn apply(&self, event: ProgressEvent) {
        self.dispatch(event, false);
    }

    /// Applies an event and emits immediately. For stage changes and the end of
    /// the scan, where a delayed frame is a visibly stuck UI.
    pub fn apply_now(&self, event: ProgressEvent) {
        self.dispatch(event, true);
    }

    pub fn snapshot(&self) -> ScanProgress {
        self.inner
            .lock()
            .map(|inner| inner.coordinator.snapshot())
            .unwrap_or_default()
    }

    fn dispatch(&self, event: ProgressEvent, force: bool) {
        let snapshot = {
            let Ok(mut inner) = self.inner.lock() else {
                return;
            };
            inner.coordinator.apply(event);
            if force || inner.last_emit.elapsed() >= EMIT_INTERVAL {
                inner.last_emit = Instant::now();
                Some(inner.coordinator.snapshot())
            } else {
                None
            }
        };
        // Emitted outside the lock: the callback repaints the UI, and holding
        // the progress lock across it serialised every worker behind the paint.
        if let Some(snapshot) = snapshot {
            (self.emit)(snapshot);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn finished(asset_completed: bool) -> FileFinished {
        FileFinished {
            asset_completed,
            elapsed: Duration::from_millis(10),
            counters: FileCounters::default(),
            decode_queue_len: 0,
            db_queue_len: 0,
        }
    }

    #[test]
    fn completed_never_exceeds_the_total() {
        let mut coordinator = ProgressCoordinator::new(1);
        coordinator.apply(ProgressEvent::DiscoveryFinished {
            asset_total: 2,
            file_total: 2,
            already_complete: 0,
        });
        for _ in 0..5 {
            coordinator.apply(ProgressEvent::FileFinished(finished(true)));
        }
        let state = coordinator.snapshot();
        assert_eq!(state.completed, 2);
        assert_eq!(state.completed_assets, state.completed);
    }

    #[test]
    fn completed_never_goes_backwards() {
        let mut coordinator = ProgressCoordinator::new(1);
        coordinator.apply(ProgressEvent::DiscoveryFinished {
            asset_total: 10,
            file_total: 10,
            already_complete: 4,
        });
        assert_eq!(coordinator.snapshot().completed, 4);
        // A later, smaller value must not rewind the bar.
        coordinator.set_completed(1);
        assert_eq!(coordinator.snapshot().completed, 4);
    }

    #[test]
    fn a_worker_cannot_change_the_stage() {
        let mut coordinator = ProgressCoordinator::new(1);
        coordinator.apply(ProgressEvent::EnterStage {
            stage: ScanStage::MediaProbe,
            activity: "probe".to_string(),
            completed: 0,
            total: 3,
        });
        coordinator.apply(ProgressEvent::FileStarted {
            activity: "解析 JPG".to_string(),
            metadata_queue_len: 0,
        });
        coordinator.apply(ProgressEvent::DatabaseQueue { db_queue_len: 7 });
        let state = coordinator.snapshot();
        assert_eq!(state.stage, ScanStage::MediaProbe);
        assert_eq!(state.activity, "解析 JPG");
        assert_eq!(state.db_queue_len, 7);
    }

    #[test]
    fn the_database_writer_cannot_reset_the_stage_bar() {
        let mut coordinator = ProgressCoordinator::new(1);
        coordinator.apply(ProgressEvent::EnterStage {
            stage: ScanStage::MediaProbe,
            activity: "probe".to_string(),
            completed: 0,
            total: 4,
        });
        coordinator.apply(ProgressEvent::FileFinished(finished(true)));
        coordinator.apply(ProgressEvent::FileFinished(finished(true)));
        coordinator.apply(ProgressEvent::DatabaseQueue { db_queue_len: 2 });
        assert_eq!(coordinator.snapshot().stage_completed, 2);
    }

    #[test]
    fn stage_completed_stops_at_the_stage_total() {
        let mut coordinator = ProgressCoordinator::new(1);
        coordinator.apply(ProgressEvent::EnterStage {
            stage: ScanStage::MediaProbe,
            activity: "probe".to_string(),
            completed: 0,
            total: 1,
        });
        coordinator.apply(ProgressEvent::FileFinished(finished(false)));
        coordinator.apply(ProgressEvent::FileFinished(finished(false)));
        assert_eq!(coordinator.snapshot().stage_completed, 1);
    }

    #[test]
    fn counters_are_added_once_each() {
        let mut coordinator = ProgressCoordinator::new(1);
        coordinator.apply(ProgressEvent::DiscoveryFinished {
            asset_total: 1,
            file_total: 1,
            already_complete: 0,
        });
        let mut report = finished(true);
        report.counters = FileCounters {
            failed: true,
            standard_computed: true,
            ai_stale: true,
            ai_computed: true,
            ..FileCounters::default()
        };
        coordinator.apply(ProgressEvent::FileFinished(report));
        let state = coordinator.snapshot();
        assert_eq!(state.failed_files, 1);
        assert_eq!(state.standard_computed, 1);
        assert_eq!(state.standard_reused, 0);
        assert_eq!(state.ai_stale, 1);
        assert_eq!(state.ai_computed, 1);
    }

    #[test]
    fn the_final_event_squares_the_numbers_off() {
        let mut coordinator = ProgressCoordinator::new(2);
        coordinator.apply(ProgressEvent::DiscoveryFinished {
            asset_total: 3,
            file_total: 3,
            already_complete: 0,
        });
        coordinator.apply(ProgressEvent::FileFinished(finished(true)));
        coordinator.apply(ProgressEvent::Finished { completed: 3 });
        let state = coordinator.snapshot();
        assert_eq!(state.stage, ScanStage::Done);
        assert_eq!(state.completed, 3);
        assert_eq!(state.completed_assets, 3);
        assert_eq!(state.stage_completed, state.stage_total);
        assert_eq!(state.eta, Some(Duration::ZERO));
        assert_eq!(state.processing, 0);
    }
}
