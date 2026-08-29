use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::Result;
use crossbeam_channel::{bounded, Receiver, Sender};
use walkdir::WalkDir;

use crate::config::{RecognitionSettings, Settings};
use crate::database::{Database, FileSnapshot, RecognitionSummary};
use crate::embedding::AiInferenceEngine;
use crate::media_probe::{self, MediaRole, MediaType};
use crate::metadata;
use crate::paths::PortablePaths;
use crate::perf::{self, StagePerf, StageTimer};
use crate::scan_planner::{
    add_to_summary, plan_for_file, AnalysisPlan, ArtifactState, PlannerContext, ScanModeKind,
    ScanPlanSummary, WorkDecision,
};
use crate::scan_state::{self, StateAccumulator};

mod progress;

pub use progress::{FileCounters, ProgressEvent};
use progress::{FileFinished, ProgressHub};

const DELETE_STAGING_DIR: &str = "PhotoCleaner_待删除";
const METADATA_QUEUE: usize = 512;
const DB_QUEUE: usize = 512;
const DB_BATCH_SIZE: usize = 250;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScanMode {
    Standard,
    Deep,
}

impl ScanMode {
    pub fn label(self) -> &'static str {
        match self {
            Self::Standard => "标准扫描",
            Self::Deep => "深度扫描",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScanStage {
    Discovering,
    MediaProbe,
    ExactHash,
    AiEmbedding,
    LivePhotoPairing,
    Grouping,
    DatabaseFinalize,
    Done,
}

impl ScanStage {
    pub fn label(self) -> &'static str {
        match self {
            Self::Discovering => "查找媒体",
            Self::MediaProbe => "解析媒体信息",
            Self::ExactHash => "计算精确指纹",
            Self::AiEmbedding => "AI Embedding",
            Self::LivePhotoPairing => "Live Photo 配对",
            Self::Grouping => "整理分组",
            Self::DatabaseFinalize => "保存数据库",
            Self::Done => "扫描完成",
        }
    }

    pub fn perf_name(self) -> &'static str {
        match self {
            Self::Discovering => "DISCOVERING",
            Self::MediaProbe => "MEDIA_PROBE",
            Self::ExactHash => "EXACT_HASH",
            Self::AiEmbedding => "AI_EMBEDDING",
            Self::LivePhotoPairing => "LIVE_PHOTO_PAIRING",
            Self::Grouping => "GROUPING",
            Self::DatabaseFinalize => "DATABASE_FINALIZE",
            Self::Done => "DONE",
        }
    }
}

#[derive(Clone, Debug)]
pub struct ScanProgress {
    pub stage: ScanStage,
    pub activity: String,
    pub discovered: usize,
    pub discovered_files: usize,
    pub processed_files: usize,
    pub total_assets: usize,
    pub completed_assets: usize,
    pub total_known: bool,
    pub total: usize,
    pub completed: usize,
    pub new_files: usize,
    pub updated_files: usize,
    pub reused_files: usize,
    pub standard_computed: usize,
    pub standard_reused: usize,
    pub ai_computed: usize,
    pub ai_reused: usize,
    pub ai_stale: usize,
    pub ai_pending: usize,
    pub unsupported_files: usize,
    pub failed_files: usize,
    pub processing: usize,
    pub stage_total: usize,
    pub stage_completed: usize,
    pub throughput: f64,
    pub throughput_unit: &'static str,
    pub eta: Option<Duration>,
    pub metadata_queue_len: usize,
    pub decode_queue_len: usize,
    pub db_queue_len: usize,
    pub active_workers: usize,
    pub worker_count: usize,
}

impl Default for ScanProgress {
    fn default() -> Self {
        Self {
            stage: ScanStage::Discovering,
            activity: "正在查找媒体...".to_string(),
            discovered: 0,
            discovered_files: 0,
            processed_files: 0,
            total_assets: 0,
            completed_assets: 0,
            total_known: false,
            total: 0,
            completed: 0,
            new_files: 0,
            updated_files: 0,
            reused_files: 0,
            standard_computed: 0,
            standard_reused: 0,
            ai_computed: 0,
            ai_reused: 0,
            ai_stale: 0,
            ai_pending: 0,
            unsupported_files: 0,
            failed_files: 0,
            processing: 0,
            stage_total: 0,
            stage_completed: 0,
            throughput: 0.0,
            throughput_unit: "文件/秒",
            eta: None,
            metadata_queue_len: 0,
            decode_queue_len: 0,
            db_queue_len: 0,
            active_workers: 0,
            worker_count: 0,
        }
    }
}

#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct ScanSummary {
    pub discovered: usize,
    pub completed: usize,
    pub new_files: usize,
    pub updated_files: usize,
    pub reused_files: usize,
    pub standard_computed: usize,
    pub standard_reused: usize,
    pub ai_computed: usize,
    pub ai_reused: usize,
    pub ai_stale: usize,
    pub unsupported_files: usize,
    pub failed_files: usize,
    pub images: usize,
    pub videos: usize,
    pub live_photos: usize,
    pub duplicate_groups: usize,
    pub similarity_groups: usize,
    pub elapsed_ms: u128,
    /// The execution provider the deep scan actually finished on, or `None` in
    /// standard mode. Reported so a "deep scan" that silently ran on the CPU is
    /// visible instead of being assumed to have used the GPU.
    pub ai_device: Option<String>,
}

#[derive(Clone, Debug)]
pub struct ScanOutcome {
    pub summary: ScanSummary,
    pub stage_perf: Vec<StagePerf>,
    pub written: usize,
}

#[derive(Clone, Debug)]
pub struct ScannedMediaFile {
    pub plan: AnalysisPlan,
    pub asset_key: String,
    pub relative_path: String,
    pub file_name: String,
    pub extension: String,
    pub media_type: MediaType,
    pub media_role: MediaRole,
    pub file_size: u64,
    pub created_time: Option<String>,
    pub modified_time: String,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub duration_ms: Option<i64>,
    pub container: Option<String>,
    pub video_codec: Option<String>,
    pub audio_codec: Option<String>,
    pub frame_rate: Option<f64>,
    pub content_identifier: Option<String>,
    pub quick_hash: Option<String>,
    pub sha256: Option<String>,
    pub phash: Option<u64>,
    pub embedding: Option<Vec<u8>>,
    pub ai_model_id: Option<String>,
    pub ai_model_hash: Option<String>,
    pub ai_preprocess_version: Option<i64>,
    pub embedding_dimension: Option<i64>,
    pub embedding_dtype: Option<String>,
    pub grouping_signature: Option<String>,
    pub scan_state: String,
    /// Which pipeline stage failed, when `scan_state` is a failure.
    pub failure_stage: Option<String>,
    /// The underlying error text. Previously the failure report printed a fixed
    /// sentence per state, so the "native error" column was fiction.
    pub failure_message: Option<String>,
    /// Set when the container's Apple metadata confirms a Live Photo
    /// component, as opposed to a filename that merely looks like one.
    pub apple_live_photo: bool,
    pub live_photo_pairing: Option<String>,
}

impl ScannedMediaFile {
    fn apply_outcome(&mut self, outcome: crate::scan_state::StateAccumulator) {
        let (state, stage, message) = outcome.finish();
        self.scan_state = state;
        self.failure_stage = stage;
        self.failure_message = message;
    }
}

#[derive(Clone, Debug)]
struct FileCandidate {
    absolute_path: PathBuf,
    asset_key: String,
    relative_path: String,
    file_name: String,
    extension: String,
    file_size: u64,
    created_time: Option<String>,
    modified_time: String,
}

enum DbMessage {
    File(ScannedMediaFile),
}

#[derive(Clone, Debug)]
struct DiscoveryResult {
    unsupported: Vec<ScannedMediaFile>,
    candidates: Vec<FileCandidate>,
    asset_component_total: HashMap<String, usize>,
}

#[derive(Clone, Debug)]
struct AssetProgressTracker {
    component_total: HashMap<String, usize>,
    component_done: HashMap<String, usize>,
    completed_assets: HashSet<String>,
}

impl AssetProgressTracker {
    fn new(component_total: HashMap<String, usize>) -> Self {
        Self {
            component_total,
            component_done: HashMap::new(),
            completed_assets: HashSet::new(),
        }
    }

    fn mark_component_finished(&mut self, asset_key: &str) -> bool {
        let done = self
            .component_done
            .entry(asset_key.to_string())
            .and_modify(|value| *value += 1)
            .or_insert(1);
        let total = self.component_total.get(asset_key).copied().unwrap_or(1);
        if *done >= total {
            self.completed_assets.insert(asset_key.to_string())
        } else {
            false
        }
    }
}

pub fn run_pipeline(
    paths: &PortablePaths,
    root: PathBuf,
    mode: ScanMode,
    progress: impl Fn(ScanProgress) + Send + Sync + 'static,
) -> Result<ScanOutcome> {
    let started = Instant::now();
    // Recognition thresholds are read once per scan, so a scan is classified
    // entirely with the values that were in effect when it started.
    let recognition = Settings::load_or_create(paths)
        .unwrap_or_default()
        .recognition;
    let db = Database::open(paths)?;
    let library = db.upsert_library(&root)?;
    let scan_run_id = db.create_scan_run(&library.id, mode.label())?;
    // Embedding BLOBs are only pulled into memory when DEEP might reuse them.
    let snapshots = Arc::new(db.load_file_snapshots(&library.id, mode == ScanMode::Deep)?);
    let model_hash = crate::embedding::model_hash(paths);
    drop(db);

    let worker_count = worker_count_for(mode);
    // One owner for every progress number in the scan; see `scanner::progress`.
    let hub = Arc::new(ProgressHub::new(worker_count, progress));
    let stage_perf = Arc::new(Mutex::new(Vec::new()));
    let plan_summary = Arc::new(Mutex::new(ScanPlanSummary::default()));
    let media_probe_timer = Arc::new(Mutex::new(StageTimer::new(
        ScanStage::MediaProbe.perf_name(),
    )));
    let exact_hash_timer = Arc::new(Mutex::new(StageTimer::new(
        ScanStage::ExactHash.perf_name(),
    )));
    let root = root.canonicalize().unwrap_or(root);

    let discovery = discover_files(&root, paths, hub.clone())?;
    let total_assets = discovery.asset_component_total.len();
    let asset_progress = Arc::new(Mutex::new(AssetProgressTracker::new(
        discovery.asset_component_total.clone(),
    )));
    {
        // Unsupported files have no work to do, so their assets are already
        // complete before the first worker starts.
        let unsupported_completed = {
            let mut tracker = asset_progress.lock().unwrap();
            discovery
                .unsupported
                .iter()
                .filter(|file| tracker.mark_component_finished(&file.asset_key))
                .count()
        };
        hub.apply(ProgressEvent::DiscoveryFinished {
            asset_total: total_assets,
            file_total: discovery.candidates.len(),
            already_complete: unsupported_completed,
        });
        hub.apply_now(ProgressEvent::EnterStage {
            stage: ScanStage::MediaProbe,
            activity: "正在处理媒体资产".to_string(),
            completed: 0,
            total: discovery.candidates.len(),
        });
    }

    let (candidate_tx, candidate_rx) = bounded::<FileCandidate>(METADATA_QUEUE);
    let (db_tx, db_rx) = bounded::<DbMessage>(DB_QUEUE);

    let signature = grouping_signature(mode, &recognition);
    let planner_context = PlannerContext {
        requested_mode: mode_kind(mode),
        is_video: false,
        is_image: false,
        model_hash: model_hash.clone(),
        grouping_signature: signature.clone(),
    };
    let ai_engine = if mode == ScanMode::Deep {
        Some(Arc::new(AiInferenceEngine::start(paths.clone())?))
    } else {
        None
    };

    let mut workers = Vec::new();
    for _ in 0..worker_count {
        workers.push({
            let rx = candidate_rx.clone();
            let tx = db_tx.clone();
            let snapshots = snapshots.clone();
            let hub = hub.clone();
            let media_probe_timer = media_probe_timer.clone();
            let exact_hash_timer = exact_hash_timer.clone();
            let plan_summary = plan_summary.clone();
            let planner_context = planner_context.clone();
            let paths = paths.clone();
            let asset_progress = asset_progress.clone();
            let ai_engine = ai_engine.clone();
            thread::spawn(move || {
                worker_loop(
                    paths,
                    rx,
                    tx,
                    snapshots,
                    hub,
                    media_probe_timer,
                    exact_hash_timer,
                    plan_summary,
                    planner_context,
                    asset_progress,
                    ai_engine,
                )
            })
        });
    }
    drop(db_tx);

    let db_writer = {
        let paths = paths.clone();
        let library_id = library.id.clone();
        let hub = hub.clone();
        let stage_perf = stage_perf.clone();
        thread::spawn(move || db_writer_loop(&paths, &library_id, db_rx, hub, stage_perf))
    };

    for candidate in discovery.candidates {
        candidate_tx.send(candidate)?;
    }
    drop(candidate_tx);

    let mut all_files = discovery.unsupported;
    for worker in workers {
        all_files.append(&mut worker.join().unwrap_or_default());
    }
    if let Ok(timer) = Arc::try_unwrap(media_probe_timer).map(|m| m.into_inner().unwrap()) {
        push_perf(&stage_perf, timer.finish());
    }
    if let Ok(timer) = Arc::try_unwrap(exact_hash_timer).map(|m| m.into_inner().unwrap()) {
        push_perf(&stage_perf, timer.finish());
    }
    // The writer thread ran alongside the workers, but only now is the database
    // the only thing left to wait for - which is when the UI should say so.
    enter_stage(&hub, ScanStage::DatabaseFinalize, "保存数据库", 0, 0);
    let written = db_writer.join().unwrap_or(Ok(0))?;

    let plan_summary_value = plan_summary.lock().unwrap().clone();
    log_scan_plan(mode, &plan_summary_value);
    if mode == ScanMode::Deep && plan_summary_value.ai_compute > 0 {
        crate::embedding::ensure_deep_available(paths).map_err(|err| {
            anyhow::anyhow!(
                "{}\n基础分析缓存已保留，但不能把 STANDARD 结果标记为 DEEP 完成。",
                err
            )
        })?;
    }

    enter_stage(&hub, ScanStage::LivePhotoPairing, "Live Photo 配对", 0, 0);
    apply_live_photo_pairing(&mut all_files);
    let mut pairing_timer = StageTimer::new(ScanStage::LivePhotoPairing.perf_name());
    for file in &all_files {
        pairing_timer.record_file(file.file_size, Duration::ZERO, &file.file_name);
    }
    push_perf(&stage_perf, pairing_timer.finish());

    // The same rule as the worker loop, and for the same reason: rows produced
    // by `reused_like` carry no analysis at all, so writing one would replace a
    // complete stored row with nulls. Only `REUSED` was excluded here, which
    // left `AI_UNAVAILABLE` free to do exactly that as soon as a deep scan ran
    // without a model and the file happened to be part of a Live Photo.
    let paired_files: Vec<_> = all_files
        .iter()
        .filter(|file| {
            file.live_photo_pairing.is_some()
                && !matches!(
                    file.scan_state.as_str(),
                    scan_state::REUSED | scan_state::AI_UNAVAILABLE
                )
        })
        .cloned()
        .collect();
    if !paired_files.is_empty() {
        let mut db = Database::open(paths)?;
        db.insert_media_batch(&library.id, &paired_files)?;
    }

    enter_stage(&hub, ScanStage::Grouping, "整理分组", all_files.len(), 0);
    let mut grouping_timer = StageTimer::new(ScanStage::Grouping.perf_name());
    let recognition_summary = {
        let mut db = Database::open(paths)?;
        db.rebuild_recognition_groups(&library.id, &recognition, Some(&signature))?
    };
    grouping_timer.record_file(
        recognition_summary.candidate_pairs as u64,
        Duration::ZERO,
        "candidate_pairs",
    );
    write_similarity_diagnostic(paths, &recognition_summary)?;
    write_recognition_report(paths, &recognition_summary)?;
    crate::logging::info(format!(
        "[GROUPING]\ncandidate_pairs={}\nverified_pairs={}\nrejected_pairs={}\nduplicate_groups={}\nnear_duplicate_groups={}\nburst_groups={}\nvisual_similarity_groups={}\ngroup_members={}\nlargest_group_size={}",
        recognition_summary.candidate_pairs,
        recognition_summary.exact_pairs + recognition_summary.near_duplicate_pairs + recognition_summary.burst_pairs + recognition_summary.visually_similar_pairs,
        recognition_summary.rejected_pairs,
        recognition_summary.exact_pairs,
        recognition_summary.near_duplicate_pairs,
        recognition_summary.burst_pairs,
        recognition_summary.visually_similar_pairs,
        recognition_summary.group_members,
        recognition_summary.largest_group_size
    ));
    push_perf(&stage_perf, grouping_timer.finish());

    let mut summary = summarize(&all_files);
    summary.elapsed_ms = started.elapsed().as_millis();
    summary.standard_computed = plan_summary_value.standard_compute;
    summary.standard_reused = plan_summary_value.standard_reuse;
    summary.ai_reused = plan_summary_value.ai_reuse;
    summary.ai_stale = plan_summary_value.ai_stale;
    summary.duplicate_groups = recognition_summary.duplicate_groups;
    summary.similarity_groups = recognition_summary.similarity_groups;
    // Asked after the workers are done, not at start-up: if CUDA gave out
    // partway through, this reports where the scan actually ended.
    summary.ai_device = ai_engine.as_ref().map(|engine| engine.active_device());
    write_failure_report(paths, &all_files)?;

    hub.apply_now(ProgressEvent::Finished {
        completed: summary.completed,
    });

    let perf = stage_perf.lock().unwrap().clone();
    let db = Database::open(paths)?;
    db.finish_scan_run(
        scan_run_id,
        &summary,
        &perf,
        &plan_summary_value,
        mode_code(mode),
    )?;
    Ok(ScanOutcome {
        summary,
        stage_perf: perf,
        written,
    })
}

pub fn rebuild_recognition_only(
    paths: &PortablePaths,
    root: PathBuf,
) -> Result<RecognitionSummary> {
    let recognition = Settings::load_or_create(paths)
        .unwrap_or_default()
        .recognition;
    let db = Database::open(paths)?;
    let library = db.upsert_library(&root)?;
    drop(db);
    let mut db = Database::open(paths)?;
    let summary = db.rebuild_recognition_groups(&library.id, &recognition, None)?;
    write_similarity_diagnostic(paths, &summary)?;
    write_recognition_report(paths, &summary)?;
    Ok(summary)
}

fn discover_files(
    root: &Path,
    paths: &PortablePaths,
    hub: Arc<ProgressHub>,
) -> Result<DiscoveryResult> {
    let mut timer = StageTimer::new(ScanStage::Discovering.perf_name());
    let mut unsupported = Vec::new();
    let mut candidates = Vec::new();
    let no_plan = AnalysisPlan {
        metadata: WorkDecision::NotRequired,
        quick_hash: WorkDecision::NotRequired,
        sha256: WorkDecision::NotRequired,
        phash: WorkDecision::NotRequired,
        video_fingerprint: WorkDecision::NotRequired,
        ai_embedding: WorkDecision::NotRequired,
        ann_index: WorkDecision::NotRequired,
        grouping_rebuild: false,
    };

    for entry in WalkDir::new(root).follow_links(false).into_iter() {
        let item_start = Instant::now();
        let Ok(entry) = entry else {
            hub.apply(ProgressEvent::DiscoveryFailed);
            continue;
        };
        let path = entry.path();
        if should_skip(path, paths) || !entry.file_type().is_file() {
            continue;
        }
        let Ok(meta) = fs::metadata(path) else {
            hub.apply(ProgressEvent::DiscoveryFailed);
            continue;
        };
        let extension = extension(path);
        let file_name = file_name(path);
        let relative_path = relative(root, path);
        let (created_time, modified_time) = metadata::file_times(&meta);
        let media_type = media_probe::classify_extension(&extension);

        timer.record_file(meta.len(), item_start.elapsed(), &file_name);
        hub.apply(ProgressEvent::FileDiscovered);

        if media_type == MediaType::Unsupported {
            unsupported.push(ScannedMediaFile {
                plan: no_plan.clone(),
                asset_key: relative_path.clone(),
                relative_path,
                file_name,
                extension,
                media_type,
                media_role: MediaRole::Unsupported,
                file_size: meta.len(),
                created_time,
                modified_time,
                width: None,
                height: None,
                duration_ms: None,
                container: None,
                video_codec: None,
                audio_codec: None,
                frame_rate: None,
                content_identifier: None,
                quick_hash: None,
                sha256: None,
                phash: None,
                embedding: None,
                ai_model_id: None,
                ai_model_hash: None,
                ai_preprocess_version: None,
                embedding_dimension: None,
                embedding_dtype: None,
                grouping_signature: None,
                scan_state: scan_state::UNSUPPORTED.to_string(),
                failure_stage: None,
                failure_message: None,
                apple_live_photo: false,
                live_photo_pairing: None,
            });
            hub.apply(ProgressEvent::UnsupportedDiscovered);
            continue;
        }

        candidates.push(FileCandidate {
            absolute_path: path.to_path_buf(),
            asset_key: String::new(),
            relative_path,
            file_name,
            extension,
            file_size: meta.len(),
            created_time,
            modified_time,
        });
    }
    assign_discovery_asset_keys(&mut candidates);
    let mut asset_component_total = HashMap::new();
    for file in &unsupported {
        *asset_component_total
            .entry(file.asset_key.clone())
            .or_insert(0) += 1;
    }
    for candidate in &candidates {
        *asset_component_total
            .entry(candidate.asset_key.clone())
            .or_insert(0) += 1;
    }
    // The totals are published by the caller with `DiscoveryFinished`, once it
    // knows how many assets were already complete. Publishing them here as well
    // was the second of the three places that assigned `completed_assets`.
    perf::log_stage(&timer.finish());
    Ok(DiscoveryResult {
        unsupported,
        candidates,
        asset_component_total,
    })
}

#[allow(clippy::too_many_arguments)]
fn worker_loop(
    paths: PortablePaths,
    rx: Receiver<FileCandidate>,
    tx: Sender<DbMessage>,
    snapshots: Arc<HashMap<String, FileSnapshot>>,
    hub: Arc<ProgressHub>,
    media_probe_timer: Arc<Mutex<StageTimer>>,
    exact_hash_timer: Arc<Mutex<StageTimer>>,
    plan_summary: Arc<Mutex<ScanPlanSummary>>,
    planner_context: PlannerContext,
    asset_progress: Arc<Mutex<AssetProgressTracker>>,
    ai_engine: Option<Arc<AiInferenceEngine>>,
) -> Vec<ScannedMediaFile> {
    let mut files = Vec::new();
    while let Ok(candidate) = rx.recv() {
        // A worker describes what it is doing; it no longer claims the stage.
        // Every worker used to set `stage = MediaProbe` here on every file,
        // which fought with the database writer thread for the same field.
        hub.apply(ProgressEvent::FileStarted {
            activity: format!("解析 {}", display_kind(&candidate.extension)),
            metadata_queue_len: rx.len(),
        });
        let item_start = Instant::now();
        let scanned = scan_candidate(
            candidate,
            &snapshots,
            &media_probe_timer,
            &exact_hash_timer,
            &plan_summary,
            &planner_context,
            &paths,
            ai_engine.as_deref(),
        );
        let elapsed = item_start.elapsed();

        let asset_completed = asset_progress
            .lock()
            .unwrap()
            .mark_component_finished(&scanned.asset_key);
        hub.apply(ProgressEvent::FileFinished(FileFinished {
            asset_completed,
            elapsed,
            counters: counters_for(&scanned, &snapshots),
            decode_queue_len: rx.len(),
            db_queue_len: tx.len(),
        }));

        // `REUSED` and `AI_UNAVAILABLE` both come out of `reused_like`, whose
        // analysis fields are all `None`. Writing such a row would replace a
        // complete stored analysis with nulls, so these two states - and only
        // these two - are deliberately not sent to the writer.
        if !matches!(
            scanned.scan_state.as_str(),
            scan_state::REUSED | scan_state::AI_UNAVAILABLE
        ) {
            let _ = tx.send(DbMessage::File(scanned.clone()));
        }
        files.push(scanned);
    }
    files
}

fn db_writer_loop(
    paths: &PortablePaths,
    library_id: &str,
    rx: Receiver<DbMessage>,
    hub: Arc<ProgressHub>,
    stage_perf: Arc<Mutex<Vec<StagePerf>>>,
) -> Result<usize> {
    let mut db = Database::open(paths)?;
    let mut timer = StageTimer::new(ScanStage::DatabaseFinalize.perf_name());
    let mut batch = Vec::with_capacity(DB_BATCH_SIZE);
    let mut written = 0;

    // This thread runs *alongside* the workers. It used to announce
    // `DatabaseFinalize` and reset `stage_completed` to zero on every flush,
    // which restarted the media-probe bar several times per scan. It now
    // reports only the one number it owns: how far behind it is.
    while let Ok(DbMessage::File(file)) = rx.recv() {
        batch.push(file);
        if batch.len() >= DB_BATCH_SIZE {
            written += flush_batch(&mut db, library_id, &mut batch, &mut timer)?;
            hub.apply(ProgressEvent::DatabaseQueue {
                db_queue_len: rx.len(),
            });
        }
    }
    if !batch.is_empty() {
        written += flush_batch(&mut db, library_id, &mut batch, &mut timer)?;
        hub.apply(ProgressEvent::DatabaseQueue { db_queue_len: 0 });
    }
    push_perf(&stage_perf, timer.finish());
    Ok(written)
}

fn scan_candidate(
    candidate: FileCandidate,
    snapshots: &HashMap<String, FileSnapshot>,
    media_probe_timer: &Arc<Mutex<StageTimer>>,
    exact_hash_timer: &Arc<Mutex<StageTimer>>,
    plan_summary: &Arc<Mutex<ScanPlanSummary>>,
    planner_context: &PlannerContext,
    paths: &PortablePaths,
    ai_engine: Option<&AiInferenceEngine>,
) -> ScannedMediaFile {
    let (state, snapshot) = artifact_state_for(&candidate, snapshots);
    let mut ctx = planner_context.clone();
    let candidate_media = media_probe::classify_extension(&candidate.extension);
    ctx.is_video = candidate_media == MediaType::Video;
    ctx.is_image = candidate_media == MediaType::Image;
    let plan = plan_for_file(&state, &ctx);
    add_to_summary(&mut plan_summary.lock().unwrap(), &plan);

    if ctx.requested_mode == ScanModeKind::Deep
        && matches!(
            plan.ai_embedding,
            WorkDecision::Compute | WorkDecision::Stale
        )
        && ctx.model_hash.is_none()
    {
        // Nothing is recomputed in this case, so the row keeps whatever the
        // database already holds; only the state and the reason are new.
        let mut file = reused_like(candidate, plan, scan_state::AI_UNAVAILABLE, snapshot);
        file.failure_stage = Some("AI_EMBEDDING".to_string());
        file.failure_message = Some(
            "deep mode was requested but no model hash is available, so no embedding could be computed"
                .to_string(),
        );
        return file;
    }

    let standard_all_reuse = plan.metadata == WorkDecision::Reuse
        && plan.quick_hash == WorkDecision::Reuse
        && plan.sha256 == WorkDecision::Reuse
        && matches!(plan.phash, WorkDecision::Reuse | WorkDecision::NotRequired)
        && matches!(
            plan.video_fingerprint,
            WorkDecision::Reuse | WorkDecision::NotRequired
        );
    let ai_ok = matches!(
        plan.ai_embedding,
        WorkDecision::Reuse | WorkDecision::NotRequired
    );
    if standard_all_reuse && ai_ok {
        return reused_like(candidate, plan, scan_state::REUSED, snapshot);
    }

    // Every stage reports into one accumulator, so a failure early in the file
    // cannot be papered over by a later stage that happened to succeed.
    let mut outcome = StateAccumulator::new();

    // Metadata is only re-probed when the planner says so. Re-probing a file we
    // already understand is the most common wasted read in a rescan.
    let reuse_metadata = matches!(plan.metadata, WorkDecision::Reuse) && snapshot.is_some();
    let mut probe = match (reuse_metadata, snapshot) {
        (true, Some(snapshot)) => probe_from_snapshot(&candidate, snapshot),
        _ => {
            let started = Instant::now();
            let mut value =
                media_probe::probe(paths, &candidate.absolute_path, &candidate.extension);
            media_probe_timer.lock().unwrap().record_file(
                candidate.file_size,
                started.elapsed(),
                &candidate.file_name,
            );
            if matches!(candidate.extension.as_str(), "heic" | "heif")
                && value.media_type == MediaType::Image
            {
                // The header gives the stored dimensions; decoding gives the
                // dimensions after orientation. Failing here is not fatal, the
                // header answer stands.
                if let Ok((width, height)) =
                    crate::embedding::decoded_image_dimensions(paths, &candidate.absolute_path)
                {
                    value.width = Some(width);
                    value.height = Some(height);
                }
            }
            value
        }
    };
    if reuse_metadata {
        // Pairing runs later and rewrites the role for Live Photo components.
        probe.media_role = media_probe::default_role_for(probe.media_type);
    }
    if scan_state::is_failure(&probe.scan_state) {
        outcome.fail(
            &probe.scan_state,
            probe.failure_stage.as_deref().unwrap_or("MEDIA_PROBE"),
            probe
                .failure_message
                .clone()
                .unwrap_or_else(|| "media probe failed".to_string()),
        );
    }

    // Quick hash and SHA-256 both come out of a single pass over the file, so
    // they are reused or recomputed together.
    let reuse_exact = matches!(plan.quick_hash, WorkDecision::Reuse)
        && matches!(plan.sha256, WorkDecision::Reuse)
        && snapshot.is_some();
    let mut quick_hash = None;
    let mut sha256 = None;
    if reuse_exact {
        if let Some(snapshot) = snapshot {
            quick_hash = snapshot.reusable.quick_hash.clone();
            sha256 = snapshot.reusable.sha256.clone();
        }
    } else {
        let started = Instant::now();
        // This was `if let Ok(..)` with no else: a hashing failure left both
        // fields None while the row still claimed SUCCESS.
        match exact_fingerprint(&candidate.absolute_path, candidate.file_size) {
            Ok((quick, sha)) => {
                quick_hash = Some(format!("{quick:016x}"));
                sha256 = Some(sha);
            }
            Err(error) => outcome.fail(scan_state::HASH_FAILED, "EXACT_HASH", format!("{error:#}")),
        }
        exact_hash_timer.lock().unwrap().record_file(
            candidate.file_size,
            started.elapsed(),
            &candidate.file_name,
        );
    }

    let phash = if matches!(plan.phash, WorkDecision::Reuse) {
        snapshot.and_then(|snapshot| snapshot.reusable.phash)
    } else if probe.media_type == MediaType::Image
        && !matches!(plan.phash, WorkDecision::NotRequired)
    {
        match crate::phash::compute_for_file(paths, &candidate.absolute_path) {
            Ok(value) => Some(value),
            Err(error) => {
                outcome.fail(scan_state::PHASH_FAILED, "PHASH", format!("{error:#}"));
                None
            }
        }
    } else {
        None
    };
    let (
        embedding,
        ai_model_id,
        ai_model_hash,
        ai_preprocess_version,
        embedding_dimension,
        embedding_dtype,
    ) = if matches!(plan.ai_embedding, WorkDecision::Reuse) {
        // Carry the stored embedding and the model identity that produced it
        // straight back out, so the row is rewritten with what it already had
        // instead of being nulled.
        match snapshot {
            Some(snapshot) => (
                snapshot.reusable.embedding.clone(),
                state.embedding_model_id.clone(),
                state.embedding_model_hash.clone(),
                state.embedding_preprocess_version,
                state.embedding_dimension,
                state.embedding_dtype.clone(),
            ),
            None => (None, None, None, None, None, None),
        }
    } else if probe.media_type == MediaType::Image
        && matches!(
            plan.ai_embedding,
            WorkDecision::Compute | WorkDecision::Stale
        )
    {
        let result = if let Some(engine) = ai_engine {
            engine.embed_image_file(paths, &candidate.absolute_path)
        } else {
            crate::embedding::embed_image_file(paths, &candidate.absolute_path)
        };
        match result {
            Ok(bytes) => {
                let (model_hash, preprocess, dtype, dimension, model_id) =
                    crate::embedding::embedding_cache_metadata(paths);
                (
                    Some(bytes),
                    Some(model_id.to_string()),
                    model_hash,
                    Some(preprocess),
                    Some(dimension),
                    Some(dtype.to_string()),
                )
            }
            Err(error) => {
                outcome.fail(scan_state::AI_FAILED, "AI_EMBEDDING", format!("{error:#}"));
                (None, None, None, None, None, None)
            }
        }
    } else {
        (None, None, None, None, None, None)
    };

    let mut scanned = ScannedMediaFile {
        plan,
        asset_key: asset_key_for(&candidate),
        relative_path: candidate.relative_path,
        file_name: candidate.file_name,
        extension: candidate.extension,
        media_type: probe.media_type,
        media_role: probe.media_role,
        file_size: candidate.file_size,
        created_time: candidate.created_time,
        modified_time: candidate.modified_time,
        width: probe.width,
        height: probe.height,
        duration_ms: probe.duration_ms,
        container: probe.container,
        video_codec: probe.video_codec,
        audio_codec: probe.audio_codec,
        frame_rate: probe.frame_rate,
        content_identifier: probe.content_identifier,
        quick_hash,
        sha256,
        phash,
        embedding,
        ai_model_id,
        ai_model_hash,
        ai_preprocess_version,
        embedding_dimension,
        embedding_dtype,
        grouping_signature: Some(planner_context.grouping_signature.clone()),
        scan_state: scan_state::SUCCESS.to_string(),
        failure_stage: None,
        failure_message: None,
        apple_live_photo: probe.apple_live_photo,
        live_photo_pairing: None,
    };
    // The container's own creation time is a better capture time than the
    // filesystem's, which changes when a file is copied.
    if scanned.created_time.is_none() {
        scanned.created_time = probe.creation_time;
    }
    scanned.apply_outcome(outcome);
    scanned
}

/// Decides what is still valid for this file by comparing it with the row the
/// database already holds.
///
/// This used to be hardcoded to `file_unchanged: false`, which forced every
/// artifact of every file to be recomputed on every scan and made the entire
/// snapshot cache dead weight.
fn artifact_state_for<'a>(
    candidate: &FileCandidate,
    snapshots: &'a HashMap<String, FileSnapshot>,
) -> (ArtifactState, Option<&'a FileSnapshot>) {
    let Some(snapshot) = snapshots.get(&candidate.relative_path) else {
        return (ArtifactState::default(), None);
    };
    let unchanged = snapshot.file_size == candidate.file_size
        && snapshot.modified_time == candidate.modified_time;
    if !unchanged {
        // Size or mtime moved: nothing derived from the old bytes can be
        // trusted, the embedding included.
        return (ArtifactState::default(), None);
    }
    let mut state = snapshot.artifact_state.clone();
    state.file_unchanged = true;
    (state, Some(snapshot))
}

/// Rebuilds a [`media_probe::MediaProbe`] from stored columns, so rescanning an
/// unchanged file never reopens it just to learn its dimensions again.
fn probe_from_snapshot(
    candidate: &FileCandidate,
    snapshot: &FileSnapshot,
) -> media_probe::MediaProbe {
    let media_type = media_probe::classify_extension(&candidate.extension);
    let content_identifier = snapshot.reusable.content_identifier.clone();
    media_probe::MediaProbe {
        media_type,
        media_role: media_probe::default_role_for(media_type),
        width: snapshot.reusable.width.map(|value| value.max(0) as u32),
        height: snapshot.reusable.height.map(|value| value.max(0) as u32),
        duration_ms: snapshot.reusable.duration_ms,
        container: snapshot.reusable.container.clone(),
        video_codec: snapshot.reusable.video_codec.clone(),
        audio_codec: snapshot.reusable.audio_codec.clone(),
        frame_rate: snapshot.reusable.frame_rate,
        apple_live_photo: content_identifier.is_some(),
        content_identifier,
        creation_time: None,
        scan_state: scan_state::SUCCESS.to_string(),
        failure_stage: None,
        failure_message: None,
    }
}

/// A row for a file nothing was recomputed for.
///
/// The analysis fields stay `None` on purpose - this row must never be written,
/// or it would replace a complete stored analysis with nulls - but the Apple
/// content identifier is carried over from the snapshot. Live Photo pairing
/// runs over every scanned file, reused ones included, and without the
/// identifier a rescan would demote a pair the previous scan had confirmed to a
/// filename guess.
fn reused_like(
    candidate: FileCandidate,
    plan: AnalysisPlan,
    state: &str,
    snapshot: Option<&FileSnapshot>,
) -> ScannedMediaFile {
    let media_type = media_probe::classify_extension(&candidate.extension);
    let content_identifier =
        snapshot.and_then(|snapshot| snapshot.reusable.content_identifier.clone());
    ScannedMediaFile {
        plan,
        asset_key: asset_key_for(&candidate),
        relative_path: candidate.relative_path,
        file_name: candidate.file_name,
        extension: candidate.extension,
        media_type,
        media_role: media_probe::default_role_for(media_type),
        file_size: candidate.file_size,
        created_time: candidate.created_time,
        modified_time: candidate.modified_time,
        width: None,
        height: None,
        duration_ms: None,
        container: None,
        video_codec: None,
        audio_codec: None,
        frame_rate: None,
        apple_live_photo: content_identifier.is_some(),
        content_identifier,
        quick_hash: None,
        sha256: None,
        phash: None,
        embedding: None,
        ai_model_id: None,
        ai_model_hash: None,
        ai_preprocess_version: None,
        embedding_dimension: None,
        embedding_dtype: None,
        grouping_signature: None,
        scan_state: state.to_string(),
        failure_stage: None,
        failure_message: None,
        live_photo_pairing: None,
    }
}

/// Both fingerprints for one file, streamed.
///
/// This used to `fs::read` the whole file. A single multi-gigabyte MOV was
/// therefore pulled into memory in one allocation purely to be hashed, on every
/// scan that could not reuse the stored value.
///
/// Note that a zero-byte file now reports the real SHA-256 of no bytes rather
/// than an empty string. Empty files still all agree with each other, which is
/// correct: they are identical.
fn exact_fingerprint(path: &Path, file_size: u64) -> Result<(u64, String)> {
    let fingerprint = crate::hashing::fingerprint_file(path, file_size)?;
    Ok((fingerprint.quick_hash, fingerprint.sha256))
}

fn flush_batch(
    db: &mut Database,
    library_id: &str,
    batch: &mut Vec<ScannedMediaFile>,
    timer: &mut StageTimer,
) -> Result<usize> {
    let started = Instant::now();
    let written = db.insert_media_batch(library_id, batch)?;
    timer.add_db_write(started.elapsed());
    for file in batch.iter() {
        timer.record_file(file.file_size, Duration::ZERO, &file.file_name);
    }
    batch.clear();
    Ok(written)
}

/// The only way a stage is entered, and the only caller is the pipeline.
fn enter_stage(
    hub: &Arc<ProgressHub>,
    stage: ScanStage,
    activity: &str,
    completed: usize,
    total: usize,
) {
    hub.apply_now(ProgressEvent::EnterStage {
        stage,
        activity: activity.to_string(),
        completed,
        total,
    });
}

/// Reads one finished file as a set of counters.
///
/// Pure, and separate from the coordinator that adds them up: what a file *was*
/// is a question about the scan pipeline, and how it lands on a progress bar is
/// not. Matching on a hand-written list of state strings meant a newly added
/// state was silently counted as nothing at all, which is why the catch-all
/// arm now asks `scan_state::is_failure` rather than naming states.
fn counters_for(
    scanned: &ScannedMediaFile,
    snapshots: &HashMap<String, FileSnapshot>,
) -> FileCounters {
    let mut counters = FileCounters::default();
    match scanned.scan_state.as_str() {
        scan_state::REUSED => counters.reused = true,
        scan_state::UNSUPPORTED => counters.unsupported = true,
        scan_state::SUCCESS => {
            if snapshots.contains_key(&scanned.relative_path) {
                counters.updated = true;
            } else {
                counters.new = true;
            }
        }
        other if scan_state::is_failure(other) => counters.failed = true,
        _ => {}
    }
    counters.standard_computed = [
        scanned.plan.metadata,
        scanned.plan.quick_hash,
        scanned.plan.sha256,
        scanned.plan.phash,
        scanned.plan.video_fingerprint,
    ]
    .iter()
    .any(|decision| *decision == WorkDecision::Compute || *decision == WorkDecision::Stale);
    match scanned.plan.ai_embedding {
        WorkDecision::Compute if scanned.embedding.is_some() => counters.ai_computed = true,
        WorkDecision::Reuse => counters.ai_reused = true,
        WorkDecision::Stale => {
            counters.ai_stale = true;
            counters.ai_computed = scanned.embedding.is_some();
        }
        _ => {}
    }
    counters
}

/// The still and the movie carry the same Apple content identifier. This is a
/// Live Photo, not a guess.
pub const LIVE_PHOTO: &str = "LIVE_PHOTO";
/// A still and a movie share a filename stem and nothing contradicts the
/// pairing, but no container metadata confirms it.
pub const PROBABLE_LIVE_PHOTO: &str = "PROBABLE_LIVE_PHOTO";
/// The container says this file is a Live Photo component, but its partner is
/// not in the scanned folder. Recorded rather than dropped: deleting one half
/// of a Live Photo because the other half was never seen is exactly the kind of
/// data loss this tool has to avoid.
pub const UNPAIRED_LIVE_PHOTO: &str = "UNPAIRED_LIVE_PHOTO";

/// Groups the components of each asset, and says how sure it is.
///
/// The previous version paired purely on filename stems and labelled every
/// result `PROBABLE_LIVE_PHOTO`, so a genuine Live Photo confirmed by Apple
/// metadata and a holiday video that happened to be named like the photo beside
/// it were indistinguishable - and both were "probable". Since staged deletion
/// moves whole assets, that difference decides whether a user's video is moved
/// along with a photo they chose to delete.
///
/// Three passes, in decreasing order of confidence:
///
/// 1. `com.apple.quicktime.content.identifier` - identical in both files.
/// 2. Filename stem, for anything the metadata could not settle.
/// 3. Sidecars (`.aae`) follow the asset that shares their stem. A sidecar is
///    never a Live Photo component on its own, but it must move with the file
///    it describes or it is left behind as an orphan.
fn apply_live_photo_pairing(files: &mut [ScannedMediaFile]) {
    let mut assigned: HashSet<usize> = HashSet::new();

    let mut by_identifier: HashMap<String, Vec<usize>> = HashMap::new();
    for (idx, file) in files.iter().enumerate() {
        let Some(identifier) = file
            .content_identifier
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
        else {
            continue;
        };
        by_identifier
            .entry(identifier.to_string())
            .or_default()
            .push(idx);
    }
    let mut identifier_groups: Vec<&Vec<usize>> = by_identifier.values().collect();
    // Sorted so the asset keys a scan produces do not depend on hash order.
    identifier_groups.sort_by_key(|indexes| indexes.first().copied().unwrap_or(usize::MAX));
    for indexes in identifier_groups {
        if !has_image_and_video(files, indexes) {
            continue;
        }
        let asset_key = anchor_asset_key(files, indexes);
        mark_group(files, indexes, &asset_key, LIVE_PHOTO, &mut assigned);
    }

    let mut by_stem: HashMap<String, Vec<usize>> = HashMap::new();
    for (idx, file) in files.iter().enumerate() {
        by_stem
            .entry(stem_key(&file.relative_path))
            .or_default()
            .push(idx);
    }
    let mut stem_groups: Vec<&Vec<usize>> = by_stem.values().collect();
    stem_groups.sort_by_key(|indexes| indexes.first().copied().unwrap_or(usize::MAX));
    for indexes in &stem_groups {
        let pending: Vec<usize> = indexes
            .iter()
            .copied()
            .filter(|idx| !assigned.contains(idx) && is_asset_component(&files[*idx]))
            .collect();
        if pending.is_empty() {
            continue;
        }
        match indexes.iter().copied().find(|idx| assigned.contains(idx)) {
            // Part of this stem was already confirmed by metadata. The rest
            // joins that asset, but only as a probable member: the identifier
            // did not vouch for it.
            Some(anchor) => {
                let asset_key = files[anchor].asset_key.clone();
                let anchor_identifier = files[anchor].content_identifier.clone();
                // A file that carries its own, different identifier is a
                // different asset that merely shares a name. Merging it would
                // stage someone else's video for deletion.
                let joinable: Vec<usize> = pending
                    .into_iter()
                    .filter(
                        |idx| match (&files[*idx].content_identifier, &anchor_identifier) {
                            (Some(own), Some(anchor_id)) => own == anchor_id,
                            (Some(_), None) => false,
                            _ => true,
                        },
                    )
                    .collect();
                if joinable.is_empty() {
                    continue;
                }
                mark_group(
                    files,
                    &joinable,
                    &asset_key,
                    PROBABLE_LIVE_PHOTO,
                    &mut assigned,
                );
            }
            None => {
                if !has_image_and_video(files, &pending) {
                    continue;
                }
                let asset_key = anchor_asset_key(files, &pending);
                mark_group(
                    files,
                    &pending,
                    &asset_key,
                    PROBABLE_LIVE_PHOTO,
                    &mut assigned,
                );
            }
        }
    }

    // A component whose own container says "Live Photo" but that found no
    // partner. It stays its own asset; the label is what stops the UI from
    // treating it as an ordinary standalone file.
    for (idx, file) in files.iter_mut().enumerate() {
        if !assigned.contains(&idx) && file.apple_live_photo && file.live_photo_pairing.is_none() {
            file.live_photo_pairing = Some(UNPAIRED_LIVE_PHOTO.to_string());
        }
    }

    for indexes in &stem_groups {
        let Some(anchor) = indexes
            .iter()
            .copied()
            .find(|idx| is_asset_component(&files[*idx]))
        else {
            continue;
        };
        let asset_key = files[anchor].asset_key.clone();
        for idx in indexes.iter().copied() {
            if files[idx].media_type == MediaType::Sidecar {
                files[idx].asset_key = asset_key.clone();
                files[idx].media_role = MediaRole::Sidecar;
            }
        }
    }
}

/// Images and videos are asset components; sidecars and unsupported files ride
/// along with one but never define it.
fn is_asset_component(file: &ScannedMediaFile) -> bool {
    matches!(file.media_type, MediaType::Image | MediaType::Video)
}

fn has_image_and_video(files: &[ScannedMediaFile], indexes: &[usize]) -> bool {
    indexes
        .iter()
        .any(|idx| files[*idx].media_type == MediaType::Image)
        && indexes
            .iter()
            .any(|idx| files[*idx].media_type == MediaType::Video)
}

/// The asset is keyed on its still image, which is what the user sees and what
/// the grouping tabs show.
fn anchor_asset_key(files: &[ScannedMediaFile], indexes: &[usize]) -> String {
    indexes
        .iter()
        .find(|idx| files[**idx].media_type == MediaType::Image)
        .or_else(|| indexes.first())
        .map(|idx| files[*idx].asset_key.clone())
        .unwrap_or_default()
}

fn mark_group(
    files: &mut [ScannedMediaFile],
    indexes: &[usize],
    asset_key: &str,
    pairing: &str,
    assigned: &mut HashSet<usize>,
) {
    for idx in indexes.iter().copied() {
        if !assigned.insert(idx) {
            continue;
        }
        files[idx].asset_key = asset_key.to_string();
        files[idx].live_photo_pairing = Some(pairing.to_string());
        files[idx].media_role = match files[idx].media_type {
            MediaType::Video => MediaRole::PairedVideo,
            MediaType::Image => MediaRole::PrimaryImage,
            other => media_probe::default_role_for(other),
        };
    }
}

fn summarize(files: &[ScannedMediaFile]) -> ScanSummary {
    let mut summary = ScanSummary {
        discovered: files.len(),
        ..Default::default()
    };
    let mut assets = HashMap::<String, MediaType>::new();
    let mut live_keys = HashSet::new();
    let mut failed_assets = HashSet::new();
    let mut unsupported_assets = HashSet::new();
    let mut reused_assets = HashSet::new();
    let mut ai_computed_assets = HashSet::new();
    for file in files {
        match file.scan_state.as_str() {
            scan_state::REUSED => {
                reused_assets.insert(file.asset_key.clone());
            }
            scan_state::UNSUPPORTED => {
                unsupported_assets.insert(file.asset_key.clone());
            }
            scan_state::SUCCESS => summary.new_files += 1,
            other if scan_state::is_failure(other) => {
                failed_assets.insert(file.asset_key.clone());
            }
            _ => {}
        }
        if file.embedding.is_some() {
            ai_computed_assets.insert(file.asset_key.clone());
        }
        assets
            .entry(file.asset_key.clone())
            .or_insert(file.media_type);
        // An unpaired component is counted as the plain image or video it is.
        // Calling it a Live Photo would put a number on screen that the folder
        // cannot show, because the other half is not there.
        if matches!(
            file.live_photo_pairing.as_deref(),
            Some(LIVE_PHOTO) | Some(PROBABLE_LIVE_PHOTO)
        ) {
            live_keys.insert(file.asset_key.clone());
        }
    }
    summary.completed = assets.len();
    summary.failed_files = failed_assets.len();
    summary.unsupported_files = unsupported_assets.len();
    summary.reused_files = reused_assets.len();
    summary.ai_computed = ai_computed_assets.len();
    for (key, media_type) in assets {
        if live_keys.contains(&key) {
            summary.live_photos += 1;
        } else if media_type == MediaType::Video {
            summary.videos += 1;
        } else {
            summary.images += 1;
        }
    }
    summary
}

#[derive(Clone, Debug)]
/// One row of FAILURE_REPORT.md.
///
/// All three fields are owned because the interesting values now come from the
/// file itself - the stage the pipeline actually stopped at and the error the
/// operating system or a decoder actually produced - rather than from a fixed
/// set of sentences chosen by matching on the state name.
struct FailureInfo {
    stage: String,
    category: String,
    native_error: String,
}

fn write_similarity_diagnostic(paths: &PortablePaths, summary: &RecognitionSummary) -> Result<()> {
    let text = format!(
        "# SIMILARITY_DIAGNOSTIC\n\n\
Embedding count: {}\n\n\
Candidate search mode: {}\n\n\
ANN queries: {}\n\n\
ANN raw neighbors: {}\n\n\
ANN unique pairs: {}\n\n\
ANN pairs past the cosine gate: {}\n\n\
Candidate pairs: {}\n\n\
Cosine distribution over candidate pairs only:\n\n\
- >=0.90: {}\n\
- >=0.92: {}\n\
- >=0.94: {}\n\
- >=0.96: {}\n\
- >=0.98: {}\n\n\
These buckets cover the pairs the candidate search proposed, not every pair in the library. Computing them over every pair would keep the O(n^2) sweep alive purely for a diagnostic.\n\n\
Important: DINO cosine is only used as a candidate signal. It is not formatted as a percent and is not treated as final duplicate confidence.\n",
        summary.embedding_count,
        summary.candidate_search_mode,
        summary.ann_queries,
        summary.ann_raw_neighbors,
        summary.ann_unique_pairs,
        summary.ann_filtered_pairs,
        summary.candidate_pairs,
        summary.candidate_cosine_ge_090,
        summary.candidate_cosine_ge_092,
        summary.candidate_cosine_ge_094,
        summary.candidate_cosine_ge_096,
        summary.candidate_cosine_ge_098
    );
    fs::write(paths.root.join("SIMILARITY_DIAGNOSTIC.md"), text)?;
    Ok(())
}

fn write_recognition_report(paths: &PortablePaths, summary: &RecognitionSummary) -> Result<()> {
    let text = format!(
        "# RECOGNITION_REPORT\n\n\
## Counts\n\n\
- Embeddings: {}\n\
- Raw candidate pairs: {}\n\
- Exact pairs: {}\n\
- Near Duplicate pairs: {}\n\
- Burst pairs: {}\n\
- Similar Only pairs: {}\n\
- Rejected pairs: {}\n\
- Duplicate groups: {}\n\
- Similarity groups: {}\n\
- Group members: {}\n\
- Largest group size: {}\n\n\
## Classification Rules\n\n\
EXACT_DUPLICATE: SHA256 is identical.\n\n\
NEAR_DUPLICATE: multiple signals agree, currently pHash distance is very small with DINO support, or DINO is extremely high and pHash is still close. DINO alone is not enough.\n\n\
BURST_SIMILAR: high visual candidate plus close pHash and short capture-time distance. Stored separately from duplicate cleanup.\n\n\
VISUALLY_SIMILAR: DINO candidate remains high but stronger duplicate evidence is missing. Stored separately from duplicate cleanup.\n\n\
## Current Limitations\n\n\
SSIM/local-geometry verification is not yet a full ORB+RANSAC implementation in this pass. The current conservative gate uses SHA256, pHash distance, DINO cosine, and capture-time separation to avoid treating DINO-only pairs as duplicates.\n",
        summary.embedding_count,
        summary.candidate_pairs,
        summary.exact_pairs,
        summary.near_duplicate_pairs,
        summary.burst_pairs,
        summary.visually_similar_pairs,
        summary.rejected_pairs,
        summary.duplicate_groups,
        summary.similarity_groups,
        summary.group_members,
        summary.largest_group_size
    );
    fs::write(paths.root.join("RECOGNITION_REPORT.md"), text)?;
    Ok(())
}

/// The stage a state implies when the row predates the stage being recorded.
///
/// Rows written by an older build, or reused from an older database, have no
/// `failure_stage`. Rather than printing an empty cell, name the stage the
/// state could only have come from.
fn fallback_stage_for(state: &str) -> &'static str {
    match state {
        scan_state::IO_FAILED => "OPEN",
        scan_state::METADATA_FAILED => "MEDIA_PROBE",
        scan_state::DECODE_FAILED => "IMAGE_DECODE",
        scan_state::HASH_FAILED => "EXACT_HASH",
        scan_state::PHASH_FAILED => "PHASH",
        scan_state::AI_FAILED | scan_state::AI_UNAVAILABLE => "AI_EMBEDDING",
        scan_state::VIDEO_PROBE_FAILED => "VIDEO_PROBE",
        _ => "UNKNOWN",
    }
}

/// Used only when a row carries no message of its own; see [`fallback_stage_for`].
fn fallback_message_for(state: &str) -> &'static str {
    match state {
        scan_state::IO_FAILED => "the file could not be opened or read",
        scan_state::METADATA_FAILED => "dimensions or container metadata could not be established",
        scan_state::DECODE_FAILED => "the image payload could not be decoded",
        scan_state::HASH_FAILED => "the file could not be read for hashing",
        scan_state::PHASH_FAILED => "a perceptual hash could not be produced",
        scan_state::AI_FAILED => "preprocessing or DINOv2 inference failed",
        scan_state::AI_UNAVAILABLE => "the AI model or ONNX Runtime provider is unavailable",
        scan_state::VIDEO_PROBE_FAILED => "ffprobe could not describe this video",
        _ => "no error detail was recorded",
    }
}

/// Turns a failed row into a report row.
///
/// Two things changed here. The state is now the error category directly, so
/// the report can never claim a category the pipeline does not actually write;
/// and the stage and message are the ones the failing stage recorded, so a
/// permission error reads as a permission error instead of as a generic
/// sentence about decoding. `UNSUPPORTED` no longer appears at all - a `.txt`
/// sitting in a photo folder is not a failure, and counting it as one made
/// every scan look worse than it was.
fn failure_info(file: &ScannedMediaFile) -> Option<FailureInfo> {
    if !scan_state::is_failure(&file.scan_state) {
        return None;
    }
    Some(FailureInfo {
        stage: file
            .failure_stage
            .clone()
            .unwrap_or_else(|| fallback_stage_for(&file.scan_state).to_string()),
        category: file.scan_state.clone(),
        native_error: file
            .failure_message
            .clone()
            .filter(|message| !message.trim().is_empty())
            .unwrap_or_else(|| fallback_message_for(&file.scan_state).to_string()),
    })
}

fn write_failure_report(paths: &PortablePaths, files: &[ScannedMediaFile]) -> Result<()> {
    let failures: Vec<_> = files
        .iter()
        .filter_map(|file| failure_info(file).map(|info| (file, info)))
        .collect();
    let report_path = paths.root.join("FAILURE_REPORT.md");
    let unsupported = files
        .iter()
        .filter(|file| file.scan_state == scan_state::UNSUPPORTED)
        .count();
    if failures.is_empty() {
        fs::write(
            report_path,
            format!(
                "# FAILURE_REPORT\n\n\
本次扫描没有失败项。\n\n\
跳过的非媒体文件（UNSUPPORTED，不算失败）：{unsupported}\n"
            ),
        )?;
        return Ok(());
    }

    let mut by_extension: HashMap<String, (usize, usize)> = HashMap::new();
    let mut by_stage: HashMap<String, usize> = HashMap::new();
    let mut by_category: HashMap<String, usize> = HashMap::new();
    let mut total_by_extension: HashMap<String, usize> = HashMap::new();
    for file in files {
        *total_by_extension
            .entry(file.extension.to_ascii_uppercase())
            .or_insert(0) += 1;
    }
    for (file, info) in &failures {
        let ext = file.extension.to_ascii_uppercase();
        let total = total_by_extension.get(&ext).copied().unwrap_or(0);
        let entry = by_extension.entry(ext).or_insert((0, total));
        entry.0 += 1;
        *by_stage.entry(info.stage.clone()).or_insert(0) += 1;
        *by_category.entry(info.category.clone()).or_insert(0) += 1;
    }

    let mut text = String::new();
    text.push_str("# FAILURE_REPORT\n\n");
    text.push_str(&format!("失败项：{}\n\n", failures.len()));
    text.push_str(&format!(
        "跳过的非媒体文件（UNSUPPORTED，不算失败）：{unsupported}\n\n"
    ));
    text.push_str("## Extension\n\n");
    for (ext, (failed, total)) in sorted_counts_with_total(by_extension) {
        text.push_str(&format!("- {ext}: {failed} / {total} failed\n"));
    }
    text.push_str("\n## Stage\n\n");
    for (stage, count) in sorted_counts(by_stage) {
        text.push_str(&format!("- {stage}: {count}\n"));
    }
    text.push_str("\n## Error\n\n");
    for (category, count) in sorted_counts(by_category) {
        text.push_str(&format!("- {category}: {count}\n"));
    }
    text.push_str("\n## Details\n\n");
    text.push_str("| asset_id | file_id | extension | media_type | pipeline_stage | error_category | native_error | file_path |\n");
    text.push_str("|---|---|---:|---|---|---|---|---|\n");
    for (file, info) in failures {
        text.push_str(&format!(
            "| {} | {} | {} | {:?} | {} | {} | {} | {} |\n",
            markdown_escape(&file.asset_key),
            markdown_escape(&file.relative_path),
            file.extension.to_ascii_uppercase(),
            file.media_type,
            info.stage,
            info.category,
            markdown_escape(&info.native_error),
            markdown_escape(&file.relative_path)
        ));
    }
    fs::write(report_path, text)?;
    Ok(())
}

fn sorted_counts<K: Ord>(counts: HashMap<K, usize>) -> Vec<(K, usize)> {
    let mut rows: Vec<_> = counts.into_iter().collect();
    rows.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    rows
}

fn sorted_counts_with_total(
    counts: HashMap<String, (usize, usize)>,
) -> Vec<(String, (usize, usize))> {
    let mut rows: Vec<_> = counts.into_iter().collect();
    rows.sort_by(|a, b| b.1 .0.cmp(&a.1 .0).then_with(|| a.0.cmp(&b.0)));
    rows
}

fn markdown_escape(text: &str) -> String {
    text.replace('|', "\\|").replace('\n', " ")
}

fn push_perf(stage_perf: &Arc<Mutex<Vec<StagePerf>>>, perf: StagePerf) {
    perf::log_stage(&perf);
    stage_perf.lock().unwrap().push(perf);
}

fn should_skip(path: &Path, portable_paths: &PortablePaths) -> bool {
    if portable_paths.is_inside_app_root(path) {
        return true;
    }
    path.components().any(|part| {
        let text = part.as_os_str().to_string_lossy();
        text.eq_ignore_ascii_case(DELETE_STAGING_DIR)
            || text.eq_ignore_ascii_case("System Volume Information")
            || text.eq_ignore_ascii_case("$RECYCLE.BIN")
    })
}

fn display_kind(extension: &str) -> &'static str {
    match media_probe::classify_extension(extension) {
        MediaType::Image if matches!(extension, "heic" | "heif") => "HEIC",
        MediaType::Image => "图片",
        MediaType::Video => "视频",
        MediaType::Sidecar => "AAE",
        MediaType::Unsupported => "文件",
    }
}

fn assign_discovery_asset_keys(candidates: &mut [FileCandidate]) {
    let mut by_stem: HashMap<String, Vec<usize>> = HashMap::new();
    for (idx, candidate) in candidates.iter().enumerate() {
        by_stem
            .entry(stem_key(&candidate.relative_path))
            .or_default()
            .push(idx);
    }

    let mut live_asset_keys = HashMap::new();
    for indexes in by_stem.values() {
        let has_image = indexes.iter().any(|idx| {
            media_probe::classify_extension(&candidates[*idx].extension) == MediaType::Image
        });
        let has_video = indexes.iter().any(|idx| {
            media_probe::classify_extension(&candidates[*idx].extension) == MediaType::Video
        });
        if has_image && has_video {
            let asset_key = indexes
                .iter()
                .find(|idx| {
                    media_probe::classify_extension(&candidates[**idx].extension)
                        == MediaType::Image
                })
                .map(|idx| candidates[*idx].relative_path.clone())
                .unwrap_or_else(|| candidates[indexes[0]].relative_path.clone());
            for idx in indexes {
                live_asset_keys.insert(*idx, asset_key.clone());
            }
        }
    }

    for (idx, candidate) in candidates.iter_mut().enumerate() {
        candidate.asset_key = live_asset_keys.get(&idx).cloned().unwrap_or_else(|| {
            match media_probe::classify_extension(&candidate.extension) {
                MediaType::Video | MediaType::Sidecar => stem_key(&candidate.relative_path),
                _ => candidate.relative_path.clone(),
            }
        });
    }
}

fn asset_key_for(candidate: &FileCandidate) -> String {
    candidate.asset_key.clone()
}

fn stem_key(relative_path: &str) -> String {
    let path = Path::new(relative_path);
    let parent = path
        .parent()
        .map(|p| p.to_string_lossy())
        .unwrap_or_default();
    let stem = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or(relative_path);
    if parent.is_empty() {
        stem.to_string()
    } else {
        format!("{}\\{}", parent, stem)
    }
}

fn relative(root: &Path, path: &Path) -> String {
    path.strip_prefix(root)
        .unwrap_or(path)
        .to_string_lossy()
        .replace('/', "\\")
}

fn file_name(path: &Path) -> String {
    path.file_name()
        .and_then(|n| n.to_str())
        .unwrap_or_default()
        .to_string()
}

fn extension(path: &Path) -> String {
    path.extension()
        .and_then(|e| e.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase()
}

fn standard_worker_count() -> usize {
    num_cpus::get().saturating_sub(1).clamp(1, 16)
}

fn deep_decode_worker_count() -> usize {
    num_cpus::get_physical().max(1).min(4)
}

fn worker_count_for(mode: ScanMode) -> usize {
    match mode {
        ScanMode::Standard => standard_worker_count(),
        ScanMode::Deep => deep_decode_worker_count(),
    }
}

fn mode_kind(mode: ScanMode) -> ScanModeKind {
    match mode {
        ScanMode::Standard => ScanModeKind::Standard,
        ScanMode::Deep => ScanModeKind::Deep,
    }
}

/// The signature stored per file, so grouping is rebuilt when, and only when,
/// something that affects grouping has actually moved.
///
/// It used to be the literal string `deep:ai_threshold_0.92:v1`, which never
/// changed no matter what the user configured.
fn grouping_signature(mode: ScanMode, recognition: &RecognitionSettings) -> String {
    format!(
        "{}|{}",
        mode_code(mode).to_ascii_lowercase(),
        recognition.signature()
    )
}

fn mode_code(mode: ScanMode) -> &'static str {
    match mode {
        ScanMode::Standard => "STANDARD",
        ScanMode::Deep => "DEEP",
    }
}

fn log_scan_plan(mode: ScanMode, summary: &ScanPlanSummary) {
    crate::logging::info(format!(
        "[SCAN PLAN]\nmode={}\nassets={}\nstandard_compute={}\nstandard_reuse={}\nai_compute={}\nai_reuse={}\nai_stale={}\ngrouping_rebuild={}",
        mode.label(),
        summary.assets,
        summary.standard_compute,
        summary.standard_reuse,
        summary.ai_compute,
        summary.ai_reuse,
        summary.ai_stale,
        summary.grouping_rebuild
    ));
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{ImageBuffer, Rgb};

    #[test]
    fn recognizes_supported_media_extensions_case_insensitively() {
        assert_eq!(media_probe::classify_extension("JPG"), MediaType::Image);
        assert_eq!(media_probe::classify_extension("heic"), MediaType::Image);
        assert_eq!(media_probe::classify_extension("mov"), MediaType::Video);
        assert_eq!(media_probe::classify_extension("aae"), MediaType::Sidecar);
        assert_eq!(
            media_probe::classify_extension("txt"),
            MediaType::Unsupported
        );
    }

    fn pairing_file(
        relative_path: &str,
        media_type: MediaType,
        content_identifier: Option<&str>,
        apple_live_photo: bool,
    ) -> ScannedMediaFile {
        let extension = relative_path
            .rsplit('.')
            .next()
            .unwrap_or_default()
            .to_ascii_lowercase();
        ScannedMediaFile {
            plan: AnalysisPlan {
                metadata: WorkDecision::NotRequired,
                quick_hash: WorkDecision::NotRequired,
                sha256: WorkDecision::NotRequired,
                phash: WorkDecision::NotRequired,
                video_fingerprint: WorkDecision::NotRequired,
                ai_embedding: WorkDecision::NotRequired,
                ann_index: WorkDecision::NotRequired,
                grouping_rebuild: false,
            },
            asset_key: relative_path.to_string(),
            relative_path: relative_path.to_string(),
            file_name: relative_path.to_string(),
            extension,
            media_type,
            media_role: media_probe::default_role_for(media_type),
            file_size: 1,
            created_time: None,
            modified_time: "2026-01-01T00:00:00+00:00".to_string(),
            width: None,
            height: None,
            duration_ms: None,
            container: None,
            video_codec: None,
            audio_codec: None,
            frame_rate: None,
            content_identifier: content_identifier.map(str::to_string),
            quick_hash: None,
            sha256: None,
            phash: None,
            embedding: None,
            ai_model_id: None,
            ai_model_hash: None,
            ai_preprocess_version: None,
            embedding_dimension: None,
            embedding_dtype: None,
            grouping_signature: None,
            scan_state: scan_state::SUCCESS.to_string(),
            failure_stage: None,
            failure_message: None,
            apple_live_photo,
            live_photo_pairing: None,
        }
    }

    /// A pair the container itself vouches for must not be reported with the
    /// same confidence as a pair guessed from two filenames.
    #[test]
    fn a_matching_apple_identifier_is_a_confirmed_live_photo() {
        let mut files = vec![
            pairing_file("IMG_0001.HEIC", MediaType::Image, Some("apple-id-1"), true),
            pairing_file("IMG_0001.MOV", MediaType::Video, Some("apple-id-1"), true),
        ];
        apply_live_photo_pairing(&mut files);
        assert_eq!(files[0].live_photo_pairing.as_deref(), Some(LIVE_PHOTO));
        assert_eq!(files[1].live_photo_pairing.as_deref(), Some(LIVE_PHOTO));
        assert_eq!(files[0].asset_key, files[1].asset_key);
        assert_eq!(files[1].media_role, MediaRole::PairedVideo);
    }

    /// The identifier is what makes a pair certain, so files that only share a
    /// stem stay marked as a guess.
    #[test]
    fn a_shared_stem_alone_is_only_probable() {
        let mut files = vec![
            pairing_file("holiday.JPG", MediaType::Image, None, false),
            pairing_file("holiday.MOV", MediaType::Video, None, false),
        ];
        apply_live_photo_pairing(&mut files);
        assert_eq!(
            files[0].live_photo_pairing.as_deref(),
            Some(PROBABLE_LIVE_PHOTO)
        );
        assert_eq!(files[0].asset_key, files[1].asset_key);
    }

    /// Two files can share a name and still be different assets. Merging them
    /// would stage one user's video for deletion with someone else's photo.
    #[test]
    fn a_conflicting_identifier_prevents_a_stem_merge() {
        let mut files = vec![
            pairing_file("IMG_0002.HEIC", MediaType::Image, Some("apple-a"), true),
            pairing_file("IMG_0002.MOV", MediaType::Video, Some("apple-a"), true),
            pairing_file("IMG_0002.mp4", MediaType::Video, Some("apple-b"), true),
        ];
        apply_live_photo_pairing(&mut files);
        assert_eq!(files[0].asset_key, files[1].asset_key);
        assert_ne!(files[2].asset_key, files[0].asset_key);
        assert_eq!(
            files[2].live_photo_pairing.as_deref(),
            Some(UNPAIRED_LIVE_PHOTO)
        );
    }

    /// A sidecar has to travel with the file it describes, or a staged move
    /// leaves an orphan `.aae` behind.
    #[test]
    fn a_sidecar_follows_its_asset() {
        let mut files = vec![
            pairing_file("IMG_0003.HEIC", MediaType::Image, Some("apple-c"), true),
            pairing_file("IMG_0003.MOV", MediaType::Video, Some("apple-c"), true),
            pairing_file("IMG_0003.AAE", MediaType::Sidecar, None, false),
        ];
        apply_live_photo_pairing(&mut files);
        assert_eq!(files[2].asset_key, files[0].asset_key);
        assert_eq!(files[2].media_role, MediaRole::Sidecar);
        // The sidecar is not itself a Live Photo component.
        assert!(files[2].live_photo_pairing.is_none());
    }

    /// Half a Live Photo is still worth naming: silently treating it as an
    /// ordinary video is how the other half gets deleted.
    #[test]
    fn a_live_photo_component_with_no_partner_is_labelled() {
        let mut files = vec![pairing_file(
            "IMG_0004.MOV",
            MediaType::Video,
            Some("apple-d"),
            true,
        )];
        apply_live_photo_pairing(&mut files);
        assert_eq!(
            files[0].live_photo_pairing.as_deref(),
            Some(UNPAIRED_LIVE_PHOTO)
        );
    }

    /// An ordinary photo folder must not acquire pairings out of nowhere.
    #[test]
    fn unrelated_files_are_left_alone() {
        let mut files = vec![
            pairing_file("a.jpg", MediaType::Image, None, false),
            pairing_file("b.mov", MediaType::Video, None, false),
        ];
        apply_live_photo_pairing(&mut files);
        assert!(files[0].live_photo_pairing.is_none());
        assert!(files[1].live_photo_pairing.is_none());
        assert_ne!(files[0].asset_key, files[1].asset_key);
    }

    fn write_test_image(path: &std::path::Path, side: u32, seed: u32) {
        let img = ImageBuffer::<Rgb<u8>, Vec<u8>>::from_fn(side, side, |x, y| {
            Rgb([(x + seed) as u8, (y + seed) as u8, seed as u8])
        });
        img.save(path).unwrap();
    }

    #[test]
    fn second_standard_scan_reuses_unchanged_files() {
        let portable = tempfile::tempdir().unwrap();
        let media = tempfile::tempdir().unwrap();
        let paths = PortablePaths::from_root(portable.path().join("PhotoCleaner"));
        paths.ensure_layout().unwrap();

        for idx in 0..12u32 {
            write_test_image(&media.path().join(format!("image_{idx}.png")), 8, idx);
        }

        let first = run_pipeline(
            &paths,
            media.path().to_path_buf(),
            ScanMode::Standard,
            |_| {},
        )
        .unwrap();
        assert_eq!(first.summary.completed, 12);
        assert_eq!(first.summary.reused_files, 0);
        assert_eq!(first.summary.standard_computed, 12);
        assert_eq!(first.written, 12);

        let second = run_pipeline(
            &paths,
            media.path().to_path_buf(),
            ScanMode::Standard,
            |_| {},
        )
        .unwrap();
        assert_eq!(second.summary.completed, 12);
        assert_eq!(second.summary.reused_files, 12);
        assert_eq!(second.summary.standard_reused, 12);
        assert_eq!(second.summary.standard_computed, 0);
        // Nothing was rewritten, so nothing could have been nulled out.
        assert_eq!(second.written, 0);
    }

    #[test]
    fn only_the_changed_file_is_recomputed() {
        let portable = tempfile::tempdir().unwrap();
        let media = tempfile::tempdir().unwrap();
        let paths = PortablePaths::from_root(portable.path().join("PhotoCleaner"));
        paths.ensure_layout().unwrap();

        for idx in 0..12u32 {
            write_test_image(&media.path().join(format!("image_{idx}.png")), 8, idx);
        }
        run_pipeline(
            &paths,
            media.path().to_path_buf(),
            ScanMode::Standard,
            |_| {},
        )
        .unwrap();

        // A different pixel size gives a different file size, so the change is
        // detected even if the clock has not ticked since the first scan.
        write_test_image(&media.path().join("image_3.png"), 24, 99);

        let second = run_pipeline(
            &paths,
            media.path().to_path_buf(),
            ScanMode::Standard,
            |_| {},
        )
        .unwrap();
        assert_eq!(second.summary.completed, 12);
        assert_eq!(second.summary.standard_computed, 1);
        assert_eq!(second.summary.standard_reused, 11);
        assert_eq!(second.summary.reused_files, 11);
        assert_eq!(second.written, 1);
    }

    #[test]
    fn reused_files_keep_their_stored_fingerprints() {
        let portable = tempfile::tempdir().unwrap();
        let media = tempfile::tempdir().unwrap();
        let paths = PortablePaths::from_root(portable.path().join("PhotoCleaner"));
        paths.ensure_layout().unwrap();
        write_test_image(&media.path().join("image_0.png"), 8, 1);

        run_pipeline(
            &paths,
            media.path().to_path_buf(),
            ScanMode::Standard,
            |_| {},
        )
        .unwrap();
        let library_id = Database::open(&paths).unwrap().list_libraries().unwrap()[0]
            .id
            .clone();
        let stored = Database::open(&paths)
            .unwrap()
            .load_file_snapshots(&library_id, false)
            .unwrap();
        let first = stored.values().next().unwrap().clone();
        assert!(
            first.reusable.sha256.is_some(),
            "first scan stored no sha256"
        );
        assert!(first.reusable.phash.is_some(), "first scan stored no phash");

        run_pipeline(
            &paths,
            media.path().to_path_buf(),
            ScanMode::Standard,
            |_| {},
        )
        .unwrap();
        let after = Database::open(&paths)
            .unwrap()
            .load_file_snapshots(&library_id, false)
            .unwrap();
        let second = after.values().next().unwrap();
        assert_eq!(first.reusable.sha256, second.reusable.sha256);
        assert_eq!(first.reusable.quick_hash, second.reusable.quick_hash);
        assert_eq!(first.reusable.phash, second.reusable.phash);
        assert_eq!(first.reusable.width, second.reusable.width);
    }

    #[test]
    fn artifact_state_detects_size_and_mtime_changes() {
        use crate::database::ReusableArtifacts;

        let candidate = FileCandidate {
            absolute_path: PathBuf::from(r"C:\photos\a.jpg"),
            asset_key: "a.jpg".to_string(),
            relative_path: "a.jpg".to_string(),
            file_name: "a.jpg".to_string(),
            extension: "jpg".to_string(),
            file_size: 100,
            created_time: None,
            modified_time: "2026-01-01T00:00:00Z".to_string(),
        };
        let snapshot = FileSnapshot {
            file_size: 100,
            modified_time: "2026-01-01T00:00:00Z".to_string(),
            artifact_state: ArtifactState {
                metadata_valid: true,
                quick_hash_valid: true,
                sha256_valid: true,
                phash_valid: true,
                ..Default::default()
            },
            reusable: ReusableArtifacts::default(),
        };

        let mut snapshots = HashMap::new();
        snapshots.insert("a.jpg".to_string(), snapshot.clone());
        let (state, found) = artifact_state_for(&candidate, &snapshots);
        assert!(state.file_unchanged);
        assert!(state.sha256_valid);
        assert!(found.is_some());

        let mut resized = snapshots.clone();
        resized.get_mut("a.jpg").unwrap().file_size = 101;
        let (state, found) = artifact_state_for(&candidate, &resized);
        assert!(!state.file_unchanged);
        assert!(!state.sha256_valid);
        assert!(found.is_none());

        let mut touched = snapshots.clone();
        touched.get_mut("a.jpg").unwrap().modified_time = "2026-02-02T00:00:00Z".to_string();
        let (state, found) = artifact_state_for(&candidate, &touched);
        assert!(!state.file_unchanged);
        assert!(found.is_none());

        let empty = HashMap::new();
        let (state, found) = artifact_state_for(&candidate, &empty);
        assert!(!state.file_unchanged);
        assert!(found.is_none());
    }

    #[test]
    fn standard_then_deep_does_not_mark_assets_fully_reused_without_ai() {
        let portable = tempfile::tempdir().unwrap();
        let media = tempfile::tempdir().unwrap();
        let paths = PortablePaths::from_root(portable.path().join("PhotoCleaner"));
        paths.ensure_layout().unwrap();

        let file = media.path().join("image.png");
        let img = ImageBuffer::<Rgb<u8>, Vec<u8>>::from_pixel(8, 8, Rgb([1, 2, 3]));
        img.save(file).unwrap();

        run_pipeline(
            &paths,
            media.path().to_path_buf(),
            ScanMode::Standard,
            |_| {},
        )
        .unwrap();
        let deep = run_pipeline(&paths, media.path().to_path_buf(), ScanMode::Deep, |_| {});
        assert!(deep.is_err());
    }

    #[test]
    fn live_photo_counts_as_one_completed_asset() {
        let portable = tempfile::tempdir().unwrap();
        let media = tempfile::tempdir().unwrap();
        let paths = PortablePaths::from_root(portable.path().join("PhotoCleaner"));
        paths.ensure_layout().unwrap();

        let image_path = media.path().join("IMG_001.png");
        let img = ImageBuffer::<Rgb<u8>, Vec<u8>>::from_pixel(8, 8, Rgb([1, 2, 3]));
        img.save(image_path).unwrap();
        fs::write(media.path().join("IMG_001.mov"), b"fake mov payload").unwrap();

        let events = Arc::new(Mutex::new(Vec::<ScanProgress>::new()));
        let captured = events.clone();
        let outcome = run_pipeline(
            &paths,
            media.path().to_path_buf(),
            ScanMode::Standard,
            move |progress| captured.lock().unwrap().push(progress),
        )
        .unwrap();

        assert_eq!(outcome.summary.discovered, 2);
        assert_eq!(outcome.summary.completed, 1);
        assert_eq!(outcome.summary.live_photos, 1);
        let mut highest_completed = 0;
        for progress in events.lock().unwrap().iter() {
            if progress.total_known && progress.total > 0 {
                assert!(progress.completed <= progress.total);
            }
            if progress.stage_total > 0 {
                assert!(progress.stage_completed <= progress.stage_total);
            }
            // The bar the user watches must never rewind.
            assert!(
                progress.completed >= highest_completed,
                "completed went backwards: {} after {}",
                progress.completed,
                highest_completed
            );
            assert_eq!(progress.completed, progress.completed_assets);
            highest_completed = progress.completed;
        }
    }

    #[test]
    fn asset_progress_tracker_counts_duplicate_events_once() {
        let mut totals = HashMap::new();
        totals.insert("asset-a".to_string(), 2);
        let mut tracker = AssetProgressTracker::new(totals);

        assert!(!tracker.mark_component_finished("asset-a"));
        assert!(tracker.mark_component_finished("asset-a"));
        assert!(!tracker.mark_component_finished("asset-a"));
        assert_eq!(tracker.completed_assets.len(), 1);
    }
}
