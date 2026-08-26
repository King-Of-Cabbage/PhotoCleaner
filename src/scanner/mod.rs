use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::Result;
use crossbeam_channel::{bounded, Receiver, Sender};
use sha2::{Digest, Sha256};
use walkdir::WalkDir;
use xxhash_rust::xxh3::xxh3_64;

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
    pub live_photo_pairing: Option<String>,
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
    let progress = Arc::new(progress);
    let db = Database::open(paths)?;
    let library = db.upsert_library(&root)?;
    let scan_run_id = db.create_scan_run(&library.id, mode.label())?;
    let snapshots = Arc::new(db.load_file_snapshots(&library.id)?);
    let model_hash = crate::embedding::model_hash(paths);
    drop(db);

    let worker_count = worker_count_for(mode);
    let shared = Arc::new(Mutex::new(ScanProgress {
        worker_count,
        ..Default::default()
    }));
    let stage_perf = Arc::new(Mutex::new(Vec::new()));
    let plan_summary = Arc::new(Mutex::new(ScanPlanSummary::default()));
    let media_probe_timer = Arc::new(Mutex::new(StageTimer::new(
        ScanStage::MediaProbe.perf_name(),
    )));
    let exact_hash_timer = Arc::new(Mutex::new(StageTimer::new(
        ScanStage::ExactHash.perf_name(),
    )));
    let root = root.canonicalize().unwrap_or(root);

    let discovery = discover_files(&root, paths, shared.clone(), progress.clone())?;
    let total_assets = discovery.asset_component_total.len();
    let asset_progress = Arc::new(Mutex::new(AssetProgressTracker::new(
        discovery.asset_component_total.clone(),
    )));
    {
        let unsupported_completed = {
            let mut tracker = asset_progress.lock().unwrap();
            discovery
                .unsupported
                .iter()
                .filter(|file| tracker.mark_component_finished(&file.asset_key))
                .count()
        };
        let mut state = shared.lock().unwrap();
        state.total_known = true;
        state.total_assets = total_assets;
        state.total = total_assets;
        state.completed = unsupported_completed;
        state.completed_assets = unsupported_completed;
        state.processed_files = 0;
        state.stage = ScanStage::MediaProbe;
        state.activity = "正在处理媒体资产".to_string();
        state.stage_completed = 0;
        state.stage_total = discovery.candidates.len();
        progress(state.clone());
    }

    let (candidate_tx, candidate_rx) = bounded::<FileCandidate>(METADATA_QUEUE);
    let (db_tx, db_rx) = bounded::<DbMessage>(DB_QUEUE);

    let planner_context = PlannerContext {
        requested_mode: mode_kind(mode),
        is_video: false,
        model_hash: model_hash.clone(),
        grouping_signature: grouping_signature(mode),
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
            let shared = shared.clone();
            let progress = progress.clone();
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
                    shared,
                    progress,
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
        let shared = shared.clone();
        let progress = progress.clone();
        let stage_perf = stage_perf.clone();
        thread::spawn(move || {
            db_writer_loop(&paths, &library_id, db_rx, shared, progress, stage_perf)
        })
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

    update_stage(
        &shared,
        &progress,
        ScanStage::LivePhotoPairing,
        "Live Photo 配对",
        0,
        0,
    );
    apply_live_photo_pairing(&mut all_files);
    let mut pairing_timer = StageTimer::new(ScanStage::LivePhotoPairing.perf_name());
    for file in &all_files {
        pairing_timer.record_file(file.file_size, Duration::ZERO, &file.file_name);
    }
    push_perf(&stage_perf, pairing_timer.finish());

    let paired_files: Vec<_> = all_files
        .iter()
        .filter(|file| file.live_photo_pairing.is_some() && file.scan_state != "REUSED")
        .cloned()
        .collect();
    if !paired_files.is_empty() {
        let mut db = Database::open(paths)?;
        db.insert_media_batch(&library.id, &paired_files)?;
    }

    update_stage(
        &shared,
        &progress,
        ScanStage::Grouping,
        "整理分组",
        all_files.len(),
        0,
    );
    let mut grouping_timer = StageTimer::new(ScanStage::Grouping.perf_name());
    let recognition = {
        let mut db = Database::open(paths)?;
        db.rebuild_recognition_groups(&library.id)?
    };
    grouping_timer.record_file(
        recognition.candidate_pairs as u64,
        Duration::ZERO,
        "candidate_pairs",
    );
    write_similarity_diagnostic(paths, &recognition)?;
    write_recognition_report(paths, &recognition)?;
    crate::logging::info(format!(
        "[GROUPING]\ncandidate_pairs={}\nverified_pairs={}\nrejected_pairs={}\nduplicate_groups={}\nnear_duplicate_groups={}\nburst_groups={}\nvisual_similarity_groups={}\ngroup_members={}\nlargest_group_size={}",
        recognition.candidate_pairs,
        recognition.exact_pairs + recognition.near_duplicate_pairs + recognition.burst_pairs + recognition.visually_similar_pairs,
        recognition.rejected_pairs,
        recognition.exact_pairs,
        recognition.near_duplicate_pairs,
        recognition.burst_pairs,
        recognition.visually_similar_pairs,
        recognition.group_members,
        recognition.largest_group_size
    ));
    push_perf(&stage_perf, grouping_timer.finish());

    let mut summary = summarize(&all_files);
    summary.elapsed_ms = started.elapsed().as_millis();
    summary.standard_computed = plan_summary_value.standard_compute;
    summary.standard_reused = plan_summary_value.standard_reuse;
    summary.ai_reused = plan_summary_value.ai_reuse;
    summary.ai_stale = plan_summary_value.ai_stale;
    summary.duplicate_groups = recognition.duplicate_groups;
    summary.similarity_groups = recognition.similarity_groups;
    write_failure_report(paths, &all_files)?;

    {
        let mut state = shared.lock().unwrap();
        state.stage = ScanStage::Done;
        state.total = summary.completed;
        state.total_assets = summary.completed;
        state.activity = "扫描完成".to_string();
        state.completed = summary.completed;
        state.completed_assets = summary.completed;
        state.processing = 0;
        state.stage_total = summary.completed;
        state.stage_completed = summary.completed;
        state.eta = Some(Duration::ZERO);
        progress(state.clone());
    }

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
    let db = Database::open(paths)?;
    let library = db.upsert_library(&root)?;
    drop(db);
    let mut db = Database::open(paths)?;
    let recognition = db.rebuild_recognition_groups(&library.id)?;
    write_similarity_diagnostic(paths, &recognition)?;
    write_recognition_report(paths, &recognition)?;
    Ok(recognition)
}

fn discover_files(
    root: &Path,
    paths: &PortablePaths,
    shared: Arc<Mutex<ScanProgress>>,
    progress: Arc<impl Fn(ScanProgress) + Send + Sync + 'static>,
) -> Result<DiscoveryResult> {
    let mut timer = StageTimer::new(ScanStage::Discovering.perf_name());
    let mut unsupported = Vec::new();
    let mut candidates = Vec::new();
    let mut last_emit = Instant::now();
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
            shared.lock().unwrap().failed_files += 1;
            continue;
        };
        let path = entry.path();
        if should_skip(path, paths) || !entry.file_type().is_file() {
            continue;
        }
        let Ok(meta) = fs::metadata(path) else {
            shared.lock().unwrap().failed_files += 1;
            continue;
        };
        let extension = extension(path);
        let file_name = file_name(path);
        let relative_path = relative(root, path);
        let (created_time, modified_time) = metadata::file_times(&meta);
        let media_type = media_probe::classify_extension(&extension);

        {
            let mut state = shared.lock().unwrap();
            state.discovered += 1;
            state.discovered_files = state.discovered;
            state.total = state.discovered;
            timer.record_file(meta.len(), item_start.elapsed(), &file_name);
            if last_emit.elapsed() >= Duration::from_millis(100) {
                progress(state.clone());
                last_emit = Instant::now();
            }
        }

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
                scan_state: "UNSUPPORTED".to_string(),
                live_photo_pairing: None,
            });
            let mut state = shared.lock().unwrap();
            state.unsupported_files += 1;
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
    {
        let mut state = shared.lock().unwrap();
        state.total_known = true;
        state.total_assets = asset_component_total.len();
        state.total = asset_component_total.len();
        state.completed = 0;
        state.completed_assets = 0;
        state.stage_completed = state.discovered;
        state.stage_total = state.discovered;
        progress(state.clone());
    }
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
    shared: Arc<Mutex<ScanProgress>>,
    progress: Arc<impl Fn(ScanProgress) + Send + Sync + 'static>,
    media_probe_timer: Arc<Mutex<StageTimer>>,
    exact_hash_timer: Arc<Mutex<StageTimer>>,
    plan_summary: Arc<Mutex<ScanPlanSummary>>,
    planner_context: PlannerContext,
    asset_progress: Arc<Mutex<AssetProgressTracker>>,
    ai_engine: Option<Arc<AiInferenceEngine>>,
) -> Vec<ScannedMediaFile> {
    let mut files = Vec::new();
    let mut last_emit = Instant::now();
    while let Ok(candidate) = rx.recv() {
        {
            let mut state = shared.lock().unwrap();
            state.active_workers += 1;
            state.processing += 1;
            state.metadata_queue_len = rx.len();
            state.stage = ScanStage::MediaProbe;
            state.activity = format!("解析 {}", display_kind(&candidate.extension));
        }
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

        {
            let mut state = shared.lock().unwrap();
            state.active_workers = state.active_workers.saturating_sub(1);
            state.processing = state.processing.saturating_sub(1);
            state.processed_files += 1;
            state.stage_completed += 1;
            state.decode_queue_len = rx.len();
            state.db_queue_len = tx.len();
            if asset_progress
                .lock()
                .unwrap()
                .mark_component_finished(&scanned.asset_key)
            {
                state.completed += 1;
                state.completed_assets = state.completed;
            }
            debug_assert!(state.completed <= state.total);
            debug_assert!(state.stage_completed <= state.stage_total || state.stage_total == 0);
            if state.completed > state.total {
                crate::logging::error(format!(
                    "Progress invariant violated: completed_assets={} total_assets={}",
                    state.completed, state.total
                ));
                state.completed = state.total;
                state.completed_assets = state.total;
            }
            if state.stage_total > 0 && state.stage_completed > state.stage_total {
                crate::logging::error(format!(
                    "Stage progress invariant violated: stage_completed={} stage_total={}",
                    state.stage_completed, state.stage_total
                ));
                state.stage_completed = state.stage_total;
            }
            update_progress_from_plan(&mut state, &scanned, &snapshots);
            update_throughput(&mut state, elapsed);
            if last_emit.elapsed() >= Duration::from_millis(100) {
                progress(state.clone());
                last_emit = Instant::now();
            }
        }

        if !matches!(scanned.scan_state.as_str(), "REUSED" | "AI_UNAVAILABLE") {
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
    shared: Arc<Mutex<ScanProgress>>,
    progress: Arc<impl Fn(ScanProgress) + Send + Sync + 'static>,
    stage_perf: Arc<Mutex<Vec<StagePerf>>>,
) -> Result<usize> {
    let mut db = Database::open(paths)?;
    let mut timer = StageTimer::new(ScanStage::DatabaseFinalize.perf_name());
    let mut batch = Vec::with_capacity(DB_BATCH_SIZE);
    let mut written = 0;

    while let Ok(DbMessage::File(file)) = rx.recv() {
        batch.push(file);
        if batch.len() >= DB_BATCH_SIZE {
            written += flush_batch(&mut db, library_id, &mut batch, &mut timer)?;
            emit_db_progress(&shared, &progress, rx.len());
        }
    }
    if !batch.is_empty() {
        written += flush_batch(&mut db, library_id, &mut batch, &mut timer)?;
        emit_db_progress(&shared, &progress, 0);
    }
    push_perf(&stage_perf, timer.finish());
    Ok(written)
}

fn scan_candidate(
    candidate: FileCandidate,
    _snapshots: &HashMap<String, FileSnapshot>,
    media_probe_timer: &Arc<Mutex<StageTimer>>,
    exact_hash_timer: &Arc<Mutex<StageTimer>>,
    plan_summary: &Arc<Mutex<ScanPlanSummary>>,
    planner_context: &PlannerContext,
    paths: &PortablePaths,
    ai_engine: Option<&AiInferenceEngine>,
) -> ScannedMediaFile {
    let state = ArtifactState {
        file_unchanged: false,
        ..Default::default()
    };
    let mut ctx = planner_context.clone();
    ctx.is_video = media_probe::classify_extension(&candidate.extension) == MediaType::Video;
    let plan = plan_for_file(&state, &ctx);
    add_to_summary(&mut plan_summary.lock().unwrap(), &plan);

    if ctx.requested_mode == ScanModeKind::Deep
        && matches!(
            plan.ai_embedding,
            WorkDecision::Compute | WorkDecision::Stale
        )
        && ctx.model_hash.is_none()
    {
        return reused_like(candidate, plan, "AI_UNAVAILABLE");
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
        return reused_like(candidate, plan, "REUSED");
    }

    let mut probe = None;
    if !matches!(plan.metadata, WorkDecision::Reuse)
        || !matches!(plan.phash, WorkDecision::Reuse | WorkDecision::NotRequired)
    {
        let started = Instant::now();
        let value = media_probe::probe(&candidate.absolute_path, &candidate.extension);
        media_probe_timer.lock().unwrap().record_file(
            candidate.file_size,
            started.elapsed(),
            &candidate.file_name,
        );
        probe = Some(value);
    }
    let mut probe =
        probe.unwrap_or_else(|| media_probe::probe(&candidate.absolute_path, &candidate.extension));
    if matches!(candidate.extension.as_str(), "heic" | "heif")
        && probe.media_type == MediaType::Image
    {
        if let Ok((width, height)) =
            crate::embedding::decoded_image_dimensions(paths, &candidate.absolute_path)
        {
            probe.width = Some(width);
            probe.height = Some(height);
        }
    }

    let mut quick_hash = None;
    let mut sha256 = None;
    if !matches!(plan.quick_hash, WorkDecision::Reuse)
        || !matches!(plan.sha256, WorkDecision::Reuse)
    {
        let started = Instant::now();
        if let Ok((quick, sha)) = exact_fingerprint(&candidate.absolute_path, candidate.file_size) {
            quick_hash = Some(format!("{quick:016x}"));
            sha256 = Some(sha);
        }
        exact_hash_timer.lock().unwrap().record_file(
            candidate.file_size,
            started.elapsed(),
            &candidate.file_name,
        );
    }
    let phash = if probe.media_type == MediaType::Image
        && !matches!(plan.phash, WorkDecision::Reuse | WorkDecision::NotRequired)
    {
        simple_image_phash(paths, &candidate.absolute_path)
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
    ) = if probe.media_type == MediaType::Image
        && matches!(
            plan.ai_embedding,
            WorkDecision::Compute | WorkDecision::Stale
        ) {
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
            Err(_) => (None, None, None, None, None, None),
        }
    } else {
        (None, None, None, None, None, None)
    };
    let scan_state = if probe.media_type == MediaType::Image
        && matches!(
            plan.ai_embedding,
            WorkDecision::Compute | WorkDecision::Stale
        )
        && embedding.is_none()
    {
        "AI_FAILED".to_string()
    } else {
        probe.scan_state
    };

    ScannedMediaFile {
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
        scan_state,
        live_photo_pairing: None,
    }
}

fn simple_image_phash(paths: &PortablePaths, path: &Path) -> Option<u64> {
    let rgb = crate::embedding::decode_image_rgb(paths, path, 8).ok()?;
    let values: Vec<u8> = rgb
        .chunks_exact(3)
        .map(|pixel| {
            (0.299 * pixel[0] as f32 + 0.587 * pixel[1] as f32 + 0.114 * pixel[2] as f32) as u8
        })
        .collect();
    let avg = values.iter().map(|v| *v as u64).sum::<u64>() / values.len() as u64;
    let mut hash = 0u64;
    for (idx, value) in values.iter().enumerate() {
        if *value as u64 >= avg {
            hash |= 1u64 << idx;
        }
    }
    Some(hash)
}

fn reused_like(candidate: FileCandidate, plan: AnalysisPlan, state: &str) -> ScannedMediaFile {
    let media_type = media_probe::classify_extension(&candidate.extension);
    ScannedMediaFile {
        plan,
        asset_key: asset_key_for(&candidate),
        relative_path: candidate.relative_path,
        file_name: candidate.file_name,
        extension: candidate.extension,
        media_type,
        media_role: default_role_for(media_type),
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
        scan_state: state.to_string(),
        live_photo_pairing: None,
    }
}

fn exact_fingerprint(path: &Path, file_size: u64) -> Result<(u64, String)> {
    let bytes = fs::read(path)?;
    let quick = if bytes.len() > 128 * 1024 {
        let mut sample = Vec::with_capacity(128 * 1024);
        sample.extend_from_slice(&bytes[..64 * 1024]);
        sample.extend_from_slice(&bytes[bytes.len() - 64 * 1024..]);
        xxh3_64(&sample)
    } else {
        xxh3_64(&bytes)
    };
    let sha256 = if file_size > 0 {
        format!("{:x}", Sha256::digest(&bytes))
    } else {
        String::new()
    };
    Ok((quick, sha256))
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

fn emit_db_progress(
    shared: &Arc<Mutex<ScanProgress>>,
    progress: &Arc<impl Fn(ScanProgress) + Send + Sync + 'static>,
    db_queue_len: usize,
) {
    let mut state = shared.lock().unwrap();
    state.stage = ScanStage::DatabaseFinalize;
    state.activity = "保存数据库".to_string();
    state.stage_completed = 0;
    state.stage_total = 0;
    state.db_queue_len = db_queue_len;
    progress(state.clone());
}

fn update_stage(
    shared: &Arc<Mutex<ScanProgress>>,
    progress: &Arc<impl Fn(ScanProgress) + Send + Sync + 'static>,
    stage: ScanStage,
    activity: &str,
    stage_completed: usize,
    stage_total: usize,
) {
    let mut state = shared.lock().unwrap();
    state.stage = stage;
    state.activity = activity.to_string();
    state.stage_completed = stage_completed;
    state.stage_total = stage_total;
    progress(state.clone());
}

fn update_progress_from_plan(
    state: &mut ScanProgress,
    scanned: &ScannedMediaFile,
    snapshots: &HashMap<String, FileSnapshot>,
) {
    match scanned.scan_state.as_str() {
        "REUSED" => state.reused_files += 1,
        "UNSUPPORTED" => state.unsupported_files += 1,
        "FAILED" | "DECODE_FAILED" | "AI_UNAVAILABLE" | "AI_FAILED" => state.failed_files += 1,
        "SUCCESS" => {
            if snapshots.contains_key(&scanned.relative_path) {
                state.updated_files += 1;
            } else {
                state.new_files += 1;
            }
        }
        _ => {}
    }
    let standard_compute = [
        scanned.plan.metadata,
        scanned.plan.quick_hash,
        scanned.plan.sha256,
        scanned.plan.phash,
        scanned.plan.video_fingerprint,
    ]
    .iter()
    .any(|decision| *decision == WorkDecision::Compute || *decision == WorkDecision::Stale);
    if standard_compute {
        state.standard_computed += 1;
    } else {
        state.standard_reused += 1;
    }
    match scanned.plan.ai_embedding {
        WorkDecision::Compute if scanned.embedding.is_some() => state.ai_computed += 1,
        WorkDecision::Reuse => state.ai_reused += 1,
        WorkDecision::Stale => {
            state.ai_stale += 1;
            if scanned.embedding.is_some() {
                state.ai_computed += 1;
            }
        }
        _ => {}
    }
}

fn update_throughput(state: &mut ScanProgress, elapsed: Duration) {
    let instant = if elapsed.is_zero() {
        0.0
    } else {
        1.0 / elapsed.as_secs_f64()
    };
    state.throughput = if state.throughput <= 0.0 {
        instant
    } else {
        state.throughput * 0.8 + instant * 0.2
    };
    let remaining = state.total.saturating_sub(state.completed);
    state.eta = if state.completed < 10 || state.throughput <= 0.0 {
        None
    } else {
        Some(Duration::from_secs_f64(remaining as f64 / state.throughput))
    };
}

fn apply_live_photo_pairing(files: &mut [ScannedMediaFile]) {
    let mut by_stem: HashMap<String, Vec<usize>> = HashMap::new();
    for (idx, file) in files.iter().enumerate() {
        by_stem
            .entry(stem_key(&file.relative_path))
            .or_default()
            .push(idx);
    }
    let mut paired = HashSet::new();
    for indexes in by_stem.values() {
        let has_image = indexes
            .iter()
            .any(|idx| files[*idx].media_type == MediaType::Image);
        let has_video = indexes
            .iter()
            .any(|idx| files[*idx].media_type == MediaType::Video);
        if !has_image || !has_video {
            continue;
        }
        let asset_key = indexes
            .iter()
            .find(|idx| files[**idx].media_type == MediaType::Image)
            .map(|idx| files[*idx].asset_key.clone())
            .unwrap_or_else(|| files[indexes[0]].asset_key.clone());
        for idx in indexes {
            if paired.insert(*idx) {
                files[*idx].asset_key = asset_key.clone();
                files[*idx].live_photo_pairing = Some("PROBABLE_LIVE_PHOTO".to_string());
                if files[*idx].media_type == MediaType::Video {
                    files[*idx].media_role = MediaRole::PairedVideo;
                }
            }
        }
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
            "REUSED" => {
                reused_assets.insert(file.asset_key.clone());
            }
            "UNSUPPORTED" => {
                unsupported_assets.insert(file.asset_key.clone());
            }
            "FAILED" | "DECODE_FAILED" | "AI_UNAVAILABLE" | "AI_FAILED" => {
                failed_assets.insert(file.asset_key.clone());
            }
            "SUCCESS" => summary.new_files += 1,
            _ => {}
        }
        if file.embedding.is_some() {
            ai_computed_assets.insert(file.asset_key.clone());
        }
        assets
            .entry(file.asset_key.clone())
            .or_insert(file.media_type);
        if file.live_photo_pairing.is_some() {
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
struct FailureInfo {
    stage: &'static str,
    category: &'static str,
    native_error: String,
}

fn write_similarity_diagnostic(paths: &PortablePaths, summary: &RecognitionSummary) -> Result<()> {
    let text = format!(
        "# SIMILARITY_DIAGNOSTIC\n\n\
Embedding count: {}\n\n\
Candidate pairs: {}\n\n\
Nearest/candidate cosine distribution:\n\n\
- >=0.90: {}\n\
- >=0.92: {}\n\
- >=0.94: {}\n\
- >=0.96: {}\n\
- >=0.98: {}\n\n\
Important: DINO cosine is only used as a candidate signal. It is not formatted as a percent and is not treated as final duplicate confidence.\n",
        summary.embedding_count,
        summary.candidate_pairs,
        summary.cosine_ge_090,
        summary.cosine_ge_092,
        summary.cosine_ge_094,
        summary.cosine_ge_096,
        summary.cosine_ge_098
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

fn failure_info(file: &ScannedMediaFile) -> Option<FailureInfo> {
    match file.scan_state.as_str() {
        "DECODE_FAILED" if matches!(file.extension.as_str(), "heic" | "heif") => {
            Some(FailureInfo {
                stage: "HEIC_DECODE",
                category: "IMAGE_DECODE_FAILED",
                native_error: "HEIF dimensions or image payload could not be decoded".to_string(),
            })
        }
        "DECODE_FAILED" if file.media_type == MediaType::Image => Some(FailureInfo {
            stage: "IMAGE_DECODE",
            category: "IMAGE_DECODE_FAILED",
            native_error: "image::image_dimensions failed".to_string(),
        }),
        "FAILED" if file.media_type == MediaType::Video => Some(FailureInfo {
            stage: "VIDEO_PROBE",
            category: "VIDEO_PROBE_FAILED",
            native_error: "video file could not be opened or probe data was empty".to_string(),
        }),
        "AI_UNAVAILABLE" => Some(FailureInfo {
            stage: "AI",
            category: "AI_RUNTIME_UNAVAILABLE",
            native_error: "AI model or ONNX Runtime CPU provider unavailable".to_string(),
        }),
        "AI_FAILED" => Some(FailureInfo {
            stage: "AI",
            category: "AI_PREPROCESS_FAILED",
            native_error: "image preprocessing or DINOv2 inference failed".to_string(),
        }),
        "UNSUPPORTED" => Some(FailureInfo {
            stage: "DISCOVERY",
            category: "UNSUPPORTED_FORMAT",
            native_error: "extension is outside the configured media set".to_string(),
        }),
        "FAILED" => Some(FailureInfo {
            stage: "MEDIA_PROBE",
            category: "UNKNOWN_ERROR",
            native_error: "media probe failed".to_string(),
        }),
        _ => None,
    }
}

fn write_failure_report(paths: &PortablePaths, files: &[ScannedMediaFile]) -> Result<()> {
    let failures: Vec<_> = files
        .iter()
        .filter_map(|file| failure_info(file).map(|info| (file, info)))
        .collect();
    let report_path = paths.root.join("FAILURE_REPORT.md");
    if failures.is_empty() {
        fs::write(
            report_path,
            "# FAILURE_REPORT\n\n本次扫描没有读取失败或不支持项。\n",
        )?;
        return Ok(());
    }

    let mut by_extension: HashMap<String, (usize, usize)> = HashMap::new();
    let mut by_stage: HashMap<&'static str, usize> = HashMap::new();
    let mut by_category: HashMap<&'static str, usize> = HashMap::new();
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
        *by_stage.entry(info.stage).or_insert(0) += 1;
        *by_category.entry(info.category).or_insert(0) += 1;
    }

    let mut text = String::new();
    text.push_str("# FAILURE_REPORT\n\n");
    text.push_str(&format!("失败项：{}\n\n", failures.len()));
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

fn default_role_for(media_type: MediaType) -> MediaRole {
    match media_type {
        MediaType::Image => MediaRole::PrimaryImage,
        MediaType::Video => MediaRole::SingleVideo,
        MediaType::Sidecar => MediaRole::Sidecar,
        MediaType::Unsupported => MediaRole::Unsupported,
    }
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

fn grouping_signature(mode: ScanMode) -> String {
    match mode {
        ScanMode::Standard => "standard:phash:v1".to_string(),
        ScanMode::Deep => "deep:ai_threshold_0.92:v1".to_string(),
    }
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

    #[test]
    fn second_standard_scan_recomputes_unchanged_files() {
        let portable = tempfile::tempdir().unwrap();
        let media = tempfile::tempdir().unwrap();
        let paths = PortablePaths::from_root(portable.path().join("PhotoCleaner"));
        paths.ensure_layout().unwrap();

        for idx in 0..12 {
            let file = media.path().join(format!("image_{idx}.png"));
            let img = ImageBuffer::<Rgb<u8>, Vec<u8>>::from_fn(8, 8, |x, y| {
                Rgb([(x + idx) as u8, (y + idx) as u8, idx as u8])
            });
            img.save(file).unwrap();
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

        let second = run_pipeline(
            &paths,
            media.path().to_path_buf(),
            ScanMode::Standard,
            |_| {},
        )
        .unwrap();
        assert_eq!(second.summary.completed, 12);
        assert_eq!(second.summary.reused_files, 0);
        assert_eq!(second.summary.standard_computed, 12);
        assert_eq!(second.summary.standard_reused, 0);
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
        for progress in events.lock().unwrap().iter() {
            if progress.total_known && progress.total > 0 {
                assert!(progress.completed <= progress.total);
            }
            if progress.stage_total > 0 {
                assert!(progress.stage_completed <= progress.stage_total);
            }
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
