use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::path::Path;

use anyhow::{Context, Result};
use chrono::Utc;
use half::f16;
use rusqlite::{params, Connection, OptionalExtension};
use uuid::Uuid;

use crate::media_probe::{MediaRole, MediaType};
use crate::paths::PortablePaths;
use crate::perf::StagePerf;
use crate::scan_planner::{self, ArtifactState};
use crate::scanner::ScannedMediaFile;

pub struct Database {
    conn: Connection,
}

#[derive(Clone, Debug)]
pub struct Library {
    pub id: String,
    pub display_name: String,
    pub last_known_root: String,
}

#[derive(Clone, Debug)]
pub struct FileSnapshot {
    pub file_size: u64,
    pub modified_time: String,
    pub artifact_state: ArtifactState,
    /// The values themselves, so a partially reused file can be written back
    /// without nulling out the artifacts it did not recompute.
    pub reusable: ReusableArtifacts,
}

/// Everything `insert_media_batch` would otherwise overwrite with NULL when a
/// scan reuses part of a file's analysis.
#[derive(Clone, Debug, Default)]
pub struct ReusableArtifacts {
    pub width: Option<i64>,
    pub height: Option<i64>,
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
}

#[derive(Clone, Debug, Default)]
pub struct MediaCounts {
    pub media_assets: i64,
    pub images: i64,
    pub videos: i64,
    pub live_photos: i64,
}

#[derive(Clone, Debug)]
pub struct ScanRunRecord {
    pub id: i64,
    pub mode: String,
    pub started_at: String,
    pub finished_at: Option<String>,
    pub duration_ms: Option<i64>,
    pub discovered: i64,
    pub completed: i64,
    pub new_files: i64,
    pub updated_files: i64,
    pub reused_files: i64,
    pub unsupported_files: i64,
    pub failed_files: i64,
}

#[derive(Clone, Debug, Default)]
pub struct CleanupResults {
    pub duplicate_groups: Vec<CleanupGroup>,
    pub similarity_groups: Vec<CleanupGroup>,
}

#[derive(Clone, Debug, Default)]
pub struct CleanupGroup {
    pub id: i64,
    pub table_name: String,
    pub kind: String,
    pub created_at: String,
    pub members: Vec<CleanupAsset>,
    pub reclaim_bytes: u64,
}

#[derive(Clone, Debug, Default)]
pub struct CleanupAsset {
    pub file_id: i64,
    pub asset_id: i64,
    pub library_root: String,
    pub relative_path: String,
    pub file_name: String,
    pub asset_type: String,
    pub media_type: String,
    pub media_role: String,
    pub file_size: u64,
    pub width: Option<i64>,
    pub height: Option<i64>,
    pub duration_ms: Option<i64>,
    pub capture_time: Option<String>,
    pub similarity: Option<f32>,
    pub distance: Option<i64>,
    pub recommendation: Option<String>,
    pub is_recommended_keep: bool,
}

#[derive(Clone, Debug)]
pub struct AssetFileComponent {
    pub file_id: i64,
    pub library_root: String,
    pub relative_path: String,
    pub file_name: String,
}

#[derive(Clone, Debug)]
pub struct MoveOperation {
    pub id: i64,
    pub source_path: String,
    pub destination_path: String,
}

#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct RecognitionSummary {
    pub embedding_count: usize,
    pub candidate_pairs: usize,
    pub exact_pairs: usize,
    pub near_duplicate_pairs: usize,
    pub burst_pairs: usize,
    pub visually_similar_pairs: usize,
    pub rejected_pairs: usize,
    pub duplicate_groups: usize,
    pub similarity_groups: usize,
    pub group_members: usize,
    pub largest_group_size: usize,
    pub cosine_ge_090: usize,
    pub cosine_ge_092: usize,
    pub cosine_ge_094: usize,
    pub cosine_ge_096: usize,
    pub cosine_ge_098: usize,
}

#[derive(Clone, Debug)]
struct RecognitionFile {
    id: i64,
    relative_path: String,
    media_type: String,
    sha256: Option<String>,
    phash: Option<u64>,
    embedding: Option<Vec<f32>>,
    created_time: Option<String>,
}

#[derive(Clone, Debug)]
struct VerifiedPair {
    a: usize,
    b: usize,
    kind: PairKind,
    cosine: f32,
    phash_distance: Option<u32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PairKind {
    Exact,
    NearDuplicate,
    BurstSimilar,
    VisuallySimilar,
    Rejected,
}

impl Database {
    pub fn open(paths: &PortablePaths) -> Result<Self> {
        if let Some(parent) = paths.db_file.parent() {
            fs::create_dir_all(parent).context("Cannot create database directory")?;
        }
        let mut conn = Connection::open(&paths.db_file).context("Cannot open database")?;
        conn.pragma_update(None, "journal_mode", "WAL")?;
        conn.pragma_update(None, "synchronous", "NORMAL")?;
        conn.pragma_update(None, "foreign_keys", "ON")?;

        if paths.db_file.exists() {
            let check: String = conn.query_row("PRAGMA quick_check", [], |row| row.get(0))?;
            if check != "ok" {
                let corrupted = paths.data_dir.join(format!(
                    "photos_corrupted_{}.db",
                    Utc::now().format("%Y%m%d_%H%M%S")
                ));
                drop(conn);
                fs::rename(&paths.db_file, corrupted)
                    .context("Database is corrupted and the old file could not be renamed")?;
                conn = Connection::open(&paths.db_file).context("Cannot create new database")?;
                conn.pragma_update(None, "journal_mode", "WAL")?;
                conn.pragma_update(None, "synchronous", "NORMAL")?;
                conn.pragma_update(None, "foreign_keys", "ON")?;
            }
        }

        let db = Self { conn };
        db.init_schema()?;
        Ok(db)
    }

    fn init_schema(&self) -> Result<()> {
        self.conn.execute_batch(
            r#"
            CREATE TABLE IF NOT EXISTS schema_version (
                version INTEGER NOT NULL
            );

            INSERT INTO schema_version(version)
            SELECT 1 WHERE NOT EXISTS (SELECT 1 FROM schema_version);

            CREATE TABLE IF NOT EXISTS libraries (
                id TEXT PRIMARY KEY,
                display_name TEXT NOT NULL,
                last_known_root TEXT NOT NULL,
                volume_label TEXT,
                volume_serial TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS photos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                library_id TEXT NOT NULL,
                relative_path TEXT NOT NULL,
                file_name TEXT NOT NULL,
                extension TEXT NOT NULL,
                file_size INTEGER NOT NULL,
                created_time TEXT,
                modified_time TEXT NOT NULL,
                exif_time TEXT,
                width INTEGER,
                height INTEGER,
                camera_model TEXT,
                quick_hash TEXT,
                sha256 TEXT,
                phash INTEGER,
                embedding BLOB,
                scan_state TEXT NOT NULL,
                missing INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(library_id) REFERENCES libraries(id),
                UNIQUE(library_id, relative_path)
            );

            CREATE TABLE IF NOT EXISTS media_assets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                library_id TEXT NOT NULL,
                asset_key TEXT NOT NULL,
                asset_type TEXT NOT NULL,
                primary_file_id INTEGER,
                capture_time TEXT,
                width INTEGER,
                height INTEGER,
                duration_ms INTEGER,
                user_state TEXT,
                pairing_state TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(library_id, asset_key)
            );

            CREATE TABLE IF NOT EXISTS media_files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                asset_id INTEGER,
                library_id TEXT NOT NULL,
                relative_path TEXT NOT NULL,
                file_name TEXT NOT NULL,
                extension TEXT NOT NULL,
                media_type TEXT NOT NULL,
                media_role TEXT NOT NULL,
                file_size INTEGER NOT NULL,
                created_time TEXT,
                modified_time TEXT NOT NULL,
                width INTEGER,
                height INTEGER,
                duration_ms INTEGER,
                container TEXT,
                video_codec TEXT,
                audio_codec TEXT,
                frame_rate REAL,
                content_identifier TEXT,
                quick_hash TEXT,
                sha256 TEXT,
                phash INTEGER,
                embedding BLOB,
                scan_state TEXT NOT NULL,
                missing INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(library_id, relative_path)
            );

            CREATE TABLE IF NOT EXISTS duplicate_groups (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                group_kind TEXT NOT NULL,
                representative_photo_id INTEGER,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS similarity_groups (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                level TEXT NOT NULL,
                representative_photo_id INTEGER,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS group_members (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                group_id INTEGER NOT NULL,
                group_table TEXT NOT NULL,
                photo_id INTEGER NOT NULL,
                similarity REAL,
                distance INTEGER,
                recommendation TEXT,
                user_state TEXT,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS operations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                photo_id INTEGER NOT NULL,
                operation_type TEXT NOT NULL,
                source_path TEXT NOT NULL,
                destination_path TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                undone INTEGER NOT NULL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS scan_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                library_id TEXT NOT NULL,
                mode TEXT NOT NULL,
                state TEXT NOT NULL,
                discovered INTEGER NOT NULL DEFAULT 0,
                completed INTEGER NOT NULL DEFAULT 0,
                processed INTEGER NOT NULL DEFAULT 0,
                skipped INTEGER NOT NULL DEFAULT 0,
                new_files INTEGER NOT NULL DEFAULT 0,
                updated_files INTEGER NOT NULL DEFAULT 0,
                reused_files INTEGER NOT NULL DEFAULT 0,
                unsupported_files INTEGER NOT NULL DEFAULT 0,
                failed_files INTEGER NOT NULL DEFAULT 0,
                errors INTEGER NOT NULL DEFAULT 0,
                phase TEXT NOT NULL,
                stage_perf_json TEXT,
                started_at TEXT NOT NULL,
                finished_at TEXT,
                duration_ms INTEGER
            );

            CREATE INDEX IF NOT EXISTS idx_photos_library_path ON photos(library_id, relative_path);
            CREATE INDEX IF NOT EXISTS idx_photos_file_size ON photos(file_size);
            CREATE INDEX IF NOT EXISTS idx_photos_modified_time ON photos(modified_time);
            CREATE INDEX IF NOT EXISTS idx_photos_sha256 ON photos(sha256);
            CREATE INDEX IF NOT EXISTS idx_photos_phash ON photos(phash);
            CREATE INDEX IF NOT EXISTS idx_media_files_library_path ON media_files(library_id, relative_path);
            CREATE INDEX IF NOT EXISTS idx_media_files_size_modified ON media_files(library_id, file_size, modified_time);
            CREATE INDEX IF NOT EXISTS idx_media_files_sha256 ON media_files(sha256);
            CREATE INDEX IF NOT EXISTS idx_media_files_content_identifier ON media_files(content_identifier);
            CREATE INDEX IF NOT EXISTS idx_media_assets_library_type ON media_assets(library_id, asset_type);
            "#,
        )?;
        self.ensure_scan_run_columns()?;
        self.migrate_photos_to_media_assets()?;
        self.backfill_primary_file_ids()?;
        self.conn
            .execute("UPDATE schema_version SET version = 2", [])?;
        Ok(())
    }

    fn ensure_scan_run_columns(&self) -> Result<()> {
        let columns: Vec<String> = self
            .conn
            .prepare("PRAGMA table_info(scan_runs)")?
            .query_map([], |row| row.get::<_, String>(1))?
            .filter_map(Result::ok)
            .collect();
        for (name, ty) in [
            ("completed", "INTEGER NOT NULL DEFAULT 0"),
            ("new_files", "INTEGER NOT NULL DEFAULT 0"),
            ("updated_files", "INTEGER NOT NULL DEFAULT 0"),
            ("reused_files", "INTEGER NOT NULL DEFAULT 0"),
            ("unsupported_files", "INTEGER NOT NULL DEFAULT 0"),
            ("failed_files", "INTEGER NOT NULL DEFAULT 0"),
            ("requested_mode", "TEXT"),
            ("standard_computed", "INTEGER NOT NULL DEFAULT 0"),
            ("standard_reused", "INTEGER NOT NULL DEFAULT 0"),
            ("ai_computed", "INTEGER NOT NULL DEFAULT 0"),
            ("ai_reused", "INTEGER NOT NULL DEFAULT 0"),
            ("ai_stale", "INTEGER NOT NULL DEFAULT 0"),
            ("grouping_rebuilt", "INTEGER NOT NULL DEFAULT 0"),
            ("stage_perf_json", "TEXT"),
            ("duration_ms", "INTEGER"),
        ] {
            if !columns.iter().any(|col| col == name) {
                self.conn
                    .execute(&format!("ALTER TABLE scan_runs ADD COLUMN {name} {ty}"), [])?;
            }
        }
        self.ensure_media_file_columns()?;
        Ok(())
    }

    fn ensure_media_file_columns(&self) -> Result<()> {
        let columns: Vec<String> = self
            .conn
            .prepare("PRAGMA table_info(media_files)")?
            .query_map([], |row| row.get::<_, String>(1))?
            .filter_map(Result::ok)
            .collect();
        for (name, ty) in [
            ("metadata_version", "INTEGER"),
            ("quick_hash_version", "INTEGER"),
            ("sha256_version", "INTEGER"),
            ("phash_version", "INTEGER"),
            ("video_fingerprint_version", "INTEGER"),
            ("ai_model_id", "TEXT"),
            ("ai_model_hash", "TEXT"),
            ("ai_preprocess_version", "INTEGER"),
            ("embedding_dimension", "INTEGER"),
            ("embedding_dtype", "TEXT"),
            ("grouping_signature", "TEXT"),
        ] {
            if !columns.iter().any(|col| col == name) {
                self.conn.execute(
                    &format!("ALTER TABLE media_files ADD COLUMN {name} {ty}"),
                    [],
                )?;
            }
        }
        Ok(())
    }

    fn migrate_photos_to_media_assets(&self) -> Result<()> {
        let now = Utc::now().to_rfc3339();
        self.conn.execute(
            r#"
            INSERT OR IGNORE INTO media_assets(
                library_id, asset_key, asset_type, capture_time, width, height, created_at, updated_at
            )
            SELECT library_id, relative_path, 'IMAGE', exif_time, width, height, ?1, ?1
            FROM photos
            "#,
            params![now],
        )?;
        self.conn.execute(
            r#"
            INSERT OR IGNORE INTO media_files(
                asset_id, library_id, relative_path, file_name, extension, media_type, media_role,
                file_size, created_time, modified_time, width, height, quick_hash, sha256, phash,
                embedding, scan_state, missing, created_at, updated_at
            )
            SELECT a.id, p.library_id, p.relative_path, p.file_name, p.extension, 'IMAGE', 'PRIMARY_IMAGE',
                   p.file_size, p.created_time, p.modified_time, p.width, p.height, p.quick_hash, p.sha256,
                   p.phash, p.embedding, p.scan_state, p.missing, ?1, ?1
            FROM photos p
            JOIN media_assets a ON a.library_id = p.library_id AND a.asset_key = p.relative_path
            "#,
            params![now],
        )?;
        Ok(())
    }

    fn backfill_primary_file_ids(&self) -> Result<()> {
        self.conn.execute(
            r#"
            UPDATE media_assets
            SET primary_file_id = (
                SELECT mf.id
                FROM media_files mf
                WHERE mf.asset_id = media_assets.id
                ORDER BY
                    CASE mf.media_role
                        WHEN 'PRIMARY_IMAGE' THEN 0
                        WHEN 'STILL_IMAGE' THEN 1
                        WHEN 'VIDEO' THEN 2
                        ELSE 3
                    END,
                    mf.id
                LIMIT 1
            )
            WHERE primary_file_id IS NULL
              AND EXISTS (SELECT 1 FROM media_files mf WHERE mf.asset_id = media_assets.id)
            "#,
            [],
        )?;
        Ok(())
    }

    pub fn upsert_library(&self, root: &Path) -> Result<Library> {
        let canonical = root.canonicalize().unwrap_or_else(|_| root.to_path_buf());
        let root_text = canonical.to_string_lossy().to_string();
        let existing = self
            .conn
            .query_row(
                "SELECT id, display_name, last_known_root FROM libraries WHERE last_known_root = ?1",
                params![root_text],
                |row| {
                    Ok(Library {
                        id: row.get(0)?,
                        display_name: row.get(1)?,
                        last_known_root: row.get(2)?,
                    })
                },
            )
            .optional()?;
        if let Some(library) = existing {
            return Ok(library);
        }

        let id = Uuid::new_v4().to_string();
        let display_name = canonical
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("照片库")
            .to_string();
        let now = Utc::now().to_rfc3339();
        self.conn.execute(
            "INSERT INTO libraries(id, display_name, last_known_root, created_at, updated_at)
             VALUES(?1, ?2, ?3, ?4, ?4)",
            params![id, display_name, root_text, now],
        )?;
        Ok(Library {
            id,
            display_name,
            last_known_root: canonical.to_string_lossy().to_string(),
        })
    }

    pub fn list_libraries(&self) -> Result<Vec<Library>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, display_name, last_known_root FROM libraries ORDER BY created_at DESC",
        )?;
        let rows = stmt.query_map([], |row| {
            Ok(Library {
                id: row.get(0)?,
                display_name: row.get(1)?,
                last_known_root: row.get(2)?,
            })
        })?;
        Ok(rows.filter_map(Result::ok).collect())
    }

    /// Loads one row per known file so the planner can decide what to reuse.
    ///
    /// `with_embeddings` controls whether the embedding BLOBs are actually read
    /// into memory. A STANDARD scan only needs to know whether an embedding
    /// exists, and pulling 768 bytes per photo for a library of fifty thousand
    /// would cost tens of megabytes for nothing.
    ///
    /// `artifact_state.file_unchanged` is deliberately left `false` here: only
    /// the scanner, which can see the file on disk, can decide that.
    pub fn load_file_snapshots(
        &self,
        library_id: &str,
        with_embeddings: bool,
    ) -> Result<HashMap<String, FileSnapshot>> {
        let embedding_column = if with_embeddings { "embedding" } else { "NULL" };
        let sql = format!(
            r#"
            SELECT relative_path, file_size, modified_time, quick_hash, sha256, phash,
                   {embedding_column}, metadata_version, quick_hash_version, sha256_version,
                   phash_version, video_fingerprint_version, ai_model_id, ai_model_hash,
                   ai_preprocess_version, embedding_dimension, embedding_dtype,
                   grouping_signature, embedding IS NOT NULL, width, height, duration_ms,
                   container, video_codec, audio_codec, frame_rate, content_identifier
            FROM media_files
            WHERE library_id = ?1 AND missing = 0
            "#
        );
        let mut stmt = self.conn.prepare(&sql)?;
        let rows = stmt.query_map(params![library_id], |row| {
            let quick_hash: Option<String> = row.get(3)?;
            let sha256: Option<String> = row.get(4)?;
            let phash: Option<i64> = row.get(5)?;
            let embedding: Option<Vec<u8>> = row.get(6)?;
            let metadata_version: Option<i64> = row.get(7)?;
            let quick_hash_version: Option<i64> = row.get(8)?;
            let sha256_version: Option<i64> = row.get(9)?;
            let phash_version: Option<i64> = row.get(10)?;
            let video_fingerprint_version: Option<i64> = row.get(11)?;
            let ai_model_id: Option<String> = row.get(12)?;
            let ai_model_hash: Option<String> = row.get(13)?;
            let ai_preprocess_version: Option<i64> = row.get(14)?;
            let embedding_dimension: Option<i64> = row.get(15)?;
            let embedding_dtype: Option<String> = row.get(16)?;
            let grouping_signature: Option<String> = row.get(17)?;
            let has_embedding: i64 = row.get(18)?;
            let has_embedding = has_embedding != 0;
            Ok((
                row.get::<_, String>(0)?,
                FileSnapshot {
                    file_size: row.get::<_, i64>(1)?.max(0) as u64,
                    modified_time: row.get(2)?,
                    artifact_state: ArtifactState {
                        file_unchanged: false,
                        metadata_valid: metadata_version.unwrap_or(1)
                            == scan_planner::METADATA_VERSION,
                        quick_hash_valid: quick_hash.is_some()
                            && quick_hash_version.unwrap_or(1) == scan_planner::QUICK_HASH_VERSION,
                        sha256_valid: sha256.is_some()
                            && sha256_version.unwrap_or(1) == scan_planner::SHA256_VERSION,
                        phash_valid: phash.is_some()
                            && phash_version.unwrap_or(1) == scan_planner::PHASH_VERSION,
                        video_fingerprint_valid: video_fingerprint_version
                            == Some(scan_planner::VIDEO_FINGERPRINT_VERSION),
                        embedding_valid: has_embedding,
                        embedding_present: has_embedding,
                        embedding_model_id: ai_model_id,
                        embedding_model_hash: ai_model_hash,
                        embedding_preprocess_version: ai_preprocess_version,
                        embedding_dimension,
                        embedding_dtype,
                        grouping_signature,
                    },
                    reusable: ReusableArtifacts {
                        width: row.get(19)?,
                        height: row.get(20)?,
                        duration_ms: row.get(21)?,
                        container: row.get(22)?,
                        video_codec: row.get(23)?,
                        audio_codec: row.get(24)?,
                        frame_rate: row.get(25)?,
                        content_identifier: row.get(26)?,
                        quick_hash,
                        sha256,
                        phash: phash.map(|value| value as u64),
                        embedding,
                    },
                },
            ))
        })?;
        Ok(rows.filter_map(Result::ok).collect())
    }

    pub fn insert_media_batch(
        &mut self,
        library_id: &str,
        files: &[ScannedMediaFile],
    ) -> Result<usize> {
        let tx = self.conn.transaction()?;
        let now = Utc::now().to_rfc3339();
        let mut written = 0;
        {
            let mut asset_stmt = tx.prepare(
                r#"
                INSERT INTO media_assets(
                    library_id, asset_key, asset_type, capture_time, width, height, duration_ms,
                    pairing_state, created_at, updated_at
                )
                VALUES(?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?9)
                ON CONFLICT(library_id, asset_key) DO UPDATE SET
                    asset_type=excluded.asset_type,
                    capture_time=excluded.capture_time,
                    width=excluded.width,
                    height=excluded.height,
                    duration_ms=excluded.duration_ms,
                    pairing_state=excluded.pairing_state,
                    updated_at=excluded.updated_at
                RETURNING id
                "#,
            )?;
            let mut file_stmt = tx.prepare(
                r#"
                INSERT INTO media_files(
                    asset_id, library_id, relative_path, file_name, extension, media_type, media_role,
                    file_size, created_time, modified_time, width, height, duration_ms, container,
                    video_codec, audio_codec, frame_rate, content_identifier, quick_hash, sha256,
                    phash, embedding, scan_state, metadata_version, quick_hash_version, sha256_version,
                    phash_version, video_fingerprint_version, ai_model_id, ai_model_hash,
                    ai_preprocess_version, embedding_dimension, embedding_dtype, grouping_signature,
                    missing, created_at, updated_at
                )
                VALUES(?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18, ?19, ?20, ?21, ?22, ?23, ?24, ?25, ?26, ?27, ?28, ?29, ?30, ?31, ?32, ?33, ?34, 0, ?35, ?35)
                ON CONFLICT(library_id, relative_path) DO UPDATE SET
                    asset_id=excluded.asset_id,
                    file_size=excluded.file_size,
                    created_time=excluded.created_time,
                    modified_time=excluded.modified_time,
                    width=excluded.width,
                    height=excluded.height,
                    duration_ms=excluded.duration_ms,
                    container=excluded.container,
                    video_codec=excluded.video_codec,
                    audio_codec=excluded.audio_codec,
                    frame_rate=excluded.frame_rate,
                    content_identifier=excluded.content_identifier,
                    quick_hash=excluded.quick_hash,
                    sha256=excluded.sha256,
                    phash=excluded.phash,
                    embedding=excluded.embedding,
                    scan_state=excluded.scan_state,
                    metadata_version=excluded.metadata_version,
                    quick_hash_version=excluded.quick_hash_version,
                    sha256_version=excluded.sha256_version,
                    phash_version=excluded.phash_version,
                    video_fingerprint_version=excluded.video_fingerprint_version,
                    ai_model_id=excluded.ai_model_id,
                    ai_model_hash=excluded.ai_model_hash,
                    ai_preprocess_version=excluded.ai_preprocess_version,
                    embedding_dimension=excluded.embedding_dimension,
                    embedding_dtype=excluded.embedding_dtype,
                    grouping_signature=excluded.grouping_signature,
                    missing=0,
                    updated_at=excluded.updated_at
                "#,
            )?;
            let mut photo_stmt = tx.prepare(
                r#"
                INSERT INTO photos(
                    library_id, relative_path, file_name, extension, file_size,
                    created_time, modified_time, width, height, scan_state, missing, created_at, updated_at
                )
                VALUES(?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, 0, ?11, ?11)
                ON CONFLICT(library_id, relative_path) DO UPDATE SET
                    file_size=excluded.file_size,
                    created_time=excluded.created_time,
                    modified_time=excluded.modified_time,
                    width=excluded.width,
                    height=excluded.height,
                    scan_state=excluded.scan_state,
                    missing=0,
                    updated_at=excluded.updated_at
                "#,
            )?;

            for file in files {
                let asset_id: i64 = asset_stmt.query_row(
                    params![
                        library_id,
                        file.asset_key,
                        asset_type_text(file.media_type, file.live_photo_pairing.as_deref()),
                        file.created_time,
                        file.width.map(|v| v as i64),
                        file.height.map(|v| v as i64),
                        file.duration_ms,
                        file.live_photo_pairing,
                        now
                    ],
                    |row| row.get(0),
                )?;
                file_stmt.execute(params![
                    asset_id,
                    library_id,
                    file.relative_path,
                    file.file_name,
                    file.extension,
                    media_type_text(file.media_type),
                    media_role_text(file.media_role),
                    file.file_size as i64,
                    file.created_time,
                    file.modified_time,
                    file.width.map(|v| v as i64),
                    file.height.map(|v| v as i64),
                    file.duration_ms,
                    file.container,
                    file.video_codec,
                    file.audio_codec,
                    file.frame_rate,
                    file.content_identifier,
                    file.quick_hash,
                    file.sha256,
                    file.phash.map(|v| v as i64),
                    file.embedding,
                    file.scan_state,
                    scan_planner::METADATA_VERSION,
                    file.quick_hash
                        .as_ref()
                        .map(|_| scan_planner::QUICK_HASH_VERSION),
                    file.sha256.as_ref().map(|_| scan_planner::SHA256_VERSION),
                    file.phash.as_ref().map(|_| scan_planner::PHASH_VERSION),
                    if file.media_type == MediaType::Video {
                        Some(scan_planner::VIDEO_FINGERPRINT_VERSION)
                    } else {
                        None
                    },
                    file.ai_model_id,
                    file.ai_model_hash,
                    file.ai_preprocess_version,
                    file.embedding_dimension,
                    file.embedding_dtype,
                    file.grouping_signature,
                    now
                ])?;
                if file.media_type == MediaType::Image {
                    photo_stmt.execute(params![
                        library_id,
                        file.relative_path,
                        file.file_name,
                        file.extension,
                        file.file_size as i64,
                        file.created_time,
                        file.modified_time,
                        file.width.map(|v| v as i64),
                        file.height.map(|v| v as i64),
                        file.scan_state,
                        now
                    ])?;
                }
                let file_id: i64 = tx.query_row(
                    "SELECT id FROM media_files WHERE library_id=?1 AND relative_path=?2",
                    params![library_id, file.relative_path],
                    |row| row.get(0),
                )?;
                let current_primary: Option<i64> = tx.query_row(
                    "SELECT primary_file_id FROM media_assets WHERE id=?1",
                    params![asset_id],
                    |row| row.get::<_, Option<i64>>(0),
                )?;
                let should_set_primary =
                    current_primary.is_none() || matches!(file.media_role, MediaRole::PrimaryImage);
                if should_set_primary {
                    tx.execute(
                        "UPDATE media_assets SET primary_file_id=?1, updated_at=?2 WHERE id=?3",
                        params![file_id, now, asset_id],
                    )?;
                }
                written += 1;
            }
        }
        tx.commit()?;
        Ok(written)
    }

    pub fn rebuild_recognition_groups(&mut self, library_id: &str) -> Result<RecognitionSummary> {
        let files = self.load_recognition_files(library_id)?;
        let mut summary = RecognitionSummary {
            embedding_count: files.iter().filter(|file| file.embedding.is_some()).count(),
            ..Default::default()
        };
        let mut pairs = Vec::new();

        let mut by_sha = BTreeMap::<String, Vec<usize>>::new();
        for (idx, file) in files.iter().enumerate() {
            if let Some(sha) = &file.sha256 {
                by_sha.entry(sha.clone()).or_default().push(idx);
            }
        }
        for indexes in by_sha.values().filter(|indexes| indexes.len() > 1) {
            for left in 0..indexes.len() {
                for right in left + 1..indexes.len() {
                    pairs.push(VerifiedPair {
                        a: indexes[left],
                        b: indexes[right],
                        kind: PairKind::Exact,
                        cosine: cosine_for_pair(&files[indexes[left]], &files[indexes[right]])
                            .unwrap_or(1.0),
                        phash_distance: phash_distance_for_pair(
                            &files[indexes[left]],
                            &files[indexes[right]],
                        ),
                    });
                }
            }
        }

        for a in 0..files.len() {
            let Some(left_embedding) = files[a].embedding.as_ref() else {
                continue;
            };
            if files[a].media_type != "IMAGE" {
                continue;
            }
            for b in a + 1..files.len() {
                if files[b].media_type != "IMAGE" {
                    continue;
                }
                let Some(right_embedding) = files[b].embedding.as_ref() else {
                    continue;
                };
                let cosine = cosine(left_embedding, right_embedding);
                if cosine >= 0.90 {
                    summary.cosine_ge_090 += 1;
                }
                if cosine >= 0.92 {
                    summary.cosine_ge_092 += 1;
                }
                if cosine >= 0.94 {
                    summary.cosine_ge_094 += 1;
                }
                if cosine >= 0.96 {
                    summary.cosine_ge_096 += 1;
                }
                if cosine >= 0.98 {
                    summary.cosine_ge_098 += 1;
                }
                if cosine < 0.90 {
                    continue;
                }
                let phash_distance = phash_distance_for_pair(&files[a], &files[b]);
                let kind = classify_pair(&files[a], &files[b], cosine, phash_distance);
                pairs.push(VerifiedPair {
                    a,
                    b,
                    kind,
                    cosine,
                    phash_distance,
                });
            }
        }

        summary.candidate_pairs = pairs.len();
        for pair in &pairs {
            match pair.kind {
                PairKind::Exact => summary.exact_pairs += 1,
                PairKind::NearDuplicate => summary.near_duplicate_pairs += 1,
                PairKind::BurstSimilar => summary.burst_pairs += 1,
                PairKind::VisuallySimilar => summary.visually_similar_pairs += 1,
                PairKind::Rejected => summary.rejected_pairs += 1,
            }
        }

        let tx = self.conn.transaction()?;
        tx.execute("DELETE FROM group_members", [])?;
        tx.execute("DELETE FROM duplicate_groups", [])?;
        tx.execute("DELETE FROM similarity_groups", [])?;
        let mut duplicate_groups = 0;
        let mut similarity_groups = 0;
        let mut group_members = 0;
        let mut largest_group_size = 0;
        for kind in [
            PairKind::Exact,
            PairKind::NearDuplicate,
            PairKind::BurstSimilar,
            PairKind::VisuallySimilar,
        ] {
            let groups = representative_groups(&files, &pairs, kind);
            for group in groups {
                if group.len() < 2 {
                    continue;
                }
                largest_group_size = largest_group_size.max(group.len());
                let representative = group[0];
                let now = Utc::now().to_rfc3339();
                let (group_id, table_name) = match kind {
                    // duplicate_groups is the delete-safe table: only a byte
                    // identical SHA-256 match belongs here. NEAR_DUPLICATE used
                    // to land here too, which is why near duplicates showed up
                    // under 完全重复 and got pre-selected for deletion.
                    PairKind::Exact => {
                        tx.execute(
                            "INSERT INTO duplicate_groups(group_kind, representative_photo_id, created_at, updated_at) VALUES(?1, ?2, ?3, ?3)",
                            params![pair_kind_label(kind), files[representative].id, now],
                        )?;
                        duplicate_groups += 1;
                        (tx.last_insert_rowid(), "duplicate_groups")
                    }
                    PairKind::NearDuplicate
                    | PairKind::BurstSimilar
                    | PairKind::VisuallySimilar => {
                        tx.execute(
                            "INSERT INTO similarity_groups(level, representative_photo_id, created_at, updated_at) VALUES(?1, ?2, ?3, ?3)",
                            params![pair_kind_label(kind), files[representative].id, now],
                        )?;
                        similarity_groups += 1;
                        (tx.last_insert_rowid(), "similarity_groups")
                    }
                    PairKind::Rejected => continue,
                };
                for member in group {
                    let metrics = pair_metrics_for_member(&files, &pairs, representative, member);
                    tx.execute(
                        "INSERT INTO group_members(group_id, group_table, photo_id, similarity, distance, recommendation, created_at) VALUES(?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                        params![
                            group_id,
                            table_name,
                            files[member].id,
                            metrics.map(|m| m.0),
                            metrics.and_then(|m| m.1.map(|v| v as i64)),
                            pair_kind_label(kind),
                            now
                        ],
                    )?;
                    group_members += 1;
                }
            }
        }
        tx.commit()?;

        summary.duplicate_groups = duplicate_groups;
        summary.similarity_groups = similarity_groups;
        summary.group_members = group_members;
        summary.largest_group_size = largest_group_size;
        Ok(summary)
    }

    fn load_recognition_files(&self, library_id: &str) -> Result<Vec<RecognitionFile>> {
        let mut stmt = self.conn.prepare(
            r#"
            SELECT id, relative_path, media_type, sha256, phash, embedding, created_time
            FROM media_files
            WHERE library_id = ?1 AND missing = 0 AND scan_state = 'SUCCESS'
            ORDER BY id
            "#,
        )?;
        let rows = stmt.query_map(params![library_id], |row| {
            let phash: Option<i64> = row.get(4)?;
            let embedding: Option<Vec<u8>> = row.get(5)?;
            Ok(RecognitionFile {
                id: row.get(0)?,
                relative_path: row.get(1)?,
                media_type: row.get(2)?,
                sha256: row.get(3)?,
                phash: phash.map(|value| value as u64),
                embedding: embedding.map(|bytes| decode_embedding(&bytes)),
                created_time: row.get(6)?,
            })
        })?;
        Ok(rows.filter_map(Result::ok).collect())
    }

    pub fn create_scan_run(&self, library_id: &str, mode: &str) -> Result<i64> {
        let now = Utc::now().to_rfc3339();
        self.conn.execute(
            "INSERT INTO scan_runs(library_id, mode, state, phase, started_at) VALUES(?1, ?2, 'RUNNING', 'DISCOVERING', ?3)",
            params![library_id, mode, now],
        )?;
        Ok(self.conn.last_insert_rowid())
    }

    pub fn finish_scan_run(
        &self,
        run_id: i64,
        summary: &crate::scanner::ScanSummary,
        stage_perf: &[StagePerf],
        plan_summary: &crate::scan_planner::ScanPlanSummary,
        requested_mode: &str,
    ) -> Result<()> {
        let now = Utc::now().to_rfc3339();
        let perf_json = serde_json::to_string(stage_perf)?;
        self.conn.execute(
            r#"
            UPDATE scan_runs SET
                state='DONE',
                discovered=?2,
                completed=?3,
                new_files=?4,
                updated_files=?5,
                reused_files=?6,
                unsupported_files=?7,
                failed_files=?8,
                errors=?8,
                requested_mode=?9,
                standard_computed=?10,
                standard_reused=?11,
                ai_computed=?12,
                ai_reused=?13,
                ai_stale=?14,
                grouping_rebuilt=?15,
                phase='DONE',
                stage_perf_json=?16,
                finished_at=?17,
                duration_ms=?18
            WHERE id=?1
            "#,
            params![
                run_id,
                summary.discovered as i64,
                summary.completed as i64,
                summary.new_files as i64,
                summary.updated_files as i64,
                summary.reused_files as i64,
                summary.unsupported_files as i64,
                summary.failed_files as i64,
                requested_mode,
                plan_summary.standard_compute as i64,
                plan_summary.standard_reuse as i64,
                summary.ai_computed as i64,
                plan_summary.ai_reuse as i64,
                plan_summary.ai_stale as i64,
                if plan_summary.grouping_rebuild { 1 } else { 0 },
                perf_json,
                now,
                summary.elapsed_ms as i64,
            ],
        )?;
        Ok(())
    }

    pub fn media_counts(&self) -> Result<MediaCounts> {
        let media_assets = self
            .conn
            .query_row("SELECT COUNT(*) FROM media_assets", [], |row| row.get(0))
            .unwrap_or(0);
        let images = self
            .conn
            .query_row(
                "SELECT COUNT(*) FROM media_assets WHERE asset_type='IMAGE'",
                [],
                |row| row.get(0),
            )
            .unwrap_or(0);
        let videos = self
            .conn
            .query_row(
                "SELECT COUNT(*) FROM media_assets WHERE asset_type='VIDEO'",
                [],
                |row| row.get(0),
            )
            .unwrap_or(0);
        let live_photos = self
            .conn
            .query_row(
                "SELECT COUNT(*) FROM media_assets WHERE asset_type='LIVE_PHOTO'",
                [],
                |row| row.get(0),
            )
            .unwrap_or(0);
        Ok(MediaCounts {
            media_assets,
            images,
            videos,
            live_photos,
        })
    }

    pub fn latest_scan_runs(&self, limit: usize) -> Result<Vec<ScanRunRecord>> {
        let mut stmt = self.conn.prepare(
            r#"
            SELECT id, mode, started_at, finished_at, duration_ms, discovered, completed,
                   new_files, updated_files, reused_files, unsupported_files, failed_files
            FROM scan_runs
            ORDER BY started_at DESC
            LIMIT ?1
            "#,
        )?;
        let rows = stmt.query_map(params![limit as i64], |row| {
            Ok(ScanRunRecord {
                id: row.get(0)?,
                mode: row.get(1)?,
                started_at: row.get(2)?,
                finished_at: row.get(3)?,
                duration_ms: row.get(4)?,
                discovered: row.get(5)?,
                completed: row.get(6)?,
                new_files: row.get(7)?,
                updated_files: row.get(8)?,
                reused_files: row.get(9)?,
                unsupported_files: row.get(10)?,
                failed_files: row.get(11)?,
            })
        })?;
        Ok(rows.filter_map(Result::ok).collect())
    }

    pub fn photo_count(&self) -> Result<i64> {
        self.conn
            .query_row("SELECT COUNT(*) FROM photos WHERE missing = 0", [], |row| {
                row.get(0)
            })
            .context("Cannot read photo count")
    }

    pub fn load_cleanup_results(&self) -> Result<CleanupResults> {
        Ok(CleanupResults {
            duplicate_groups: self.load_cleanup_groups("duplicate_groups")?,
            similarity_groups: self.load_cleanup_groups("similarity_groups")?,
        })
    }

    fn load_cleanup_groups(&self, table_name: &str) -> Result<Vec<CleanupGroup>> {
        let (kind_column, order_sql) = match table_name {
            "duplicate_groups" => (
                "group_kind",
                "SELECT id, group_kind, created_at FROM duplicate_groups ORDER BY id",
            ),
            "similarity_groups" => (
                "level",
                "SELECT id, level, created_at FROM similarity_groups ORDER BY id",
            ),
            _ => anyhow::bail!("Unknown cleanup group table: {table_name}"),
        };
        let mut stmt = self.conn.prepare(order_sql)?;
        let group_rows = stmt.query_map([], |row| {
            Ok((
                row.get::<_, i64>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
            ))
        })?;

        let mut groups = Vec::new();
        for group_row in group_rows {
            let (id, kind, created_at) = group_row?;
            let mut members = self.load_cleanup_members(table_name, id)?;
            mark_recommended_keep(&mut members);
            let reclaim_bytes = members
                .iter()
                .filter(|member| !member.is_recommended_keep)
                .map(|member| member.file_size)
                .sum();
            groups.push(CleanupGroup {
                id,
                table_name: table_name.to_string(),
                kind: if kind.is_empty() {
                    kind_column.to_string()
                } else {
                    kind
                },
                created_at,
                members,
                reclaim_bytes,
            });
        }
        Ok(groups)
    }

    fn load_cleanup_members(&self, table_name: &str, group_id: i64) -> Result<Vec<CleanupAsset>> {
        let mut stmt = self.conn.prepare(
            r#"
            SELECT
                mf.id,
                COALESCE(mf.asset_id, 0),
                l.last_known_root,
                mf.relative_path,
                mf.file_name,
                COALESCE(ma.asset_type, mf.media_type),
                mf.media_type,
                mf.media_role,
                mf.file_size,
                COALESCE(ma.width, mf.width),
                COALESCE(ma.height, mf.height),
                COALESCE(ma.duration_ms, mf.duration_ms),
                COALESCE(ma.capture_time, mf.created_time),
                gm.similarity,
                gm.distance,
                gm.recommendation
            FROM group_members gm
            JOIN media_files mf ON mf.id = gm.photo_id
            LEFT JOIN media_assets ma ON ma.id = mf.asset_id
            JOIN libraries l ON l.id = mf.library_id
            WHERE gm.group_table = ?1 AND gm.group_id = ?2 AND mf.missing = 0
            ORDER BY gm.id
            "#,
        )?;
        let rows = stmt.query_map(params![table_name, group_id], |row| {
            Ok(CleanupAsset {
                file_id: row.get(0)?,
                asset_id: row.get(1)?,
                library_root: row.get(2)?,
                relative_path: row.get(3)?,
                file_name: row.get(4)?,
                asset_type: row.get(5)?,
                media_type: row.get(6)?,
                media_role: row.get(7)?,
                file_size: row.get::<_, i64>(8)?.max(0) as u64,
                width: row.get(9)?,
                height: row.get(10)?,
                duration_ms: row.get(11)?,
                capture_time: row.get(12)?,
                similarity: row.get::<_, Option<f64>>(13)?.map(|v| v as f32),
                distance: row.get(14)?,
                recommendation: row.get(15)?,
                is_recommended_keep: false,
            })
        })?;
        Ok(rows.filter_map(Result::ok).collect())
    }

    pub fn asset_file_components(&self, asset_id: i64) -> Result<Vec<AssetFileComponent>> {
        let mut stmt = self.conn.prepare(
            r#"
            SELECT mf.id, l.last_known_root, mf.relative_path, mf.file_name
            FROM media_files mf
            JOIN libraries l ON l.id = mf.library_id
            WHERE mf.asset_id = ?1 AND mf.missing = 0
            ORDER BY
                CASE mf.media_role
                    WHEN 'PRIMARY_IMAGE' THEN 0
                    WHEN 'PAIRED_VIDEO' THEN 1
                    ELSE 2
                END,
                mf.id
            "#,
        )?;
        let rows = stmt.query_map(params![asset_id], |row| {
            Ok(AssetFileComponent {
                file_id: row.get(0)?,
                library_root: row.get(1)?,
                relative_path: row.get(2)?,
                file_name: row.get(3)?,
            })
        })?;
        Ok(rows.filter_map(Result::ok).collect())
    }

    pub fn record_move_operation(
        &self,
        file_id: i64,
        source_path: &str,
        destination_path: &str,
    ) -> Result<()> {
        self.conn.execute(
            "INSERT INTO operations(photo_id, operation_type, source_path, destination_path, timestamp) VALUES(?1, 'MOVE_TO_PENDING_DELETE', ?2, ?3, ?4)",
            params![file_id, source_path, destination_path, Utc::now().to_rfc3339()],
        )?;
        Ok(())
    }

    pub fn latest_move_operations(&self, limit: usize) -> Result<Vec<MoveOperation>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, source_path, destination_path FROM operations WHERE operation_type='MOVE_TO_PENDING_DELETE' AND undone=0 ORDER BY id DESC LIMIT ?1",
        )?;
        let rows = stmt.query_map(params![limit as i64], |row| {
            Ok(MoveOperation {
                id: row.get(0)?,
                source_path: row.get(1)?,
                destination_path: row.get(2)?,
            })
        })?;
        Ok(rows.filter_map(Result::ok).collect())
    }

    pub fn mark_operation_undone(&self, id: i64) -> Result<()> {
        self.conn
            .execute("UPDATE operations SET undone=1 WHERE id=?1", params![id])?;
        Ok(())
    }
}

fn mark_recommended_keep(members: &mut [CleanupAsset]) {
    if members.is_empty() {
        return;
    }
    if let Some(idx) = members.iter().position(|member| {
        member
            .recommendation
            .as_deref()
            .map(|text| text.eq_ignore_ascii_case("KEEP"))
            .unwrap_or(false)
    }) {
        members[idx].is_recommended_keep = true;
        return;
    }
    let mut best_idx = 0usize;
    let mut best_score = 0u128;
    for (idx, member) in members.iter().enumerate() {
        let pixels = member.width.unwrap_or_default().max(0) as u128
            * member.height.unwrap_or_default().max(0) as u128;
        let score = pixels.saturating_mul(1_000_000_000) + member.file_size as u128;
        if score > best_score {
            best_score = score;
            best_idx = idx;
        }
    }
    members[best_idx].is_recommended_keep = true;
}

fn decode_embedding(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|chunk| f16::from_le_bytes([chunk[0], chunk[1]]).to_f32())
        .collect()
}

fn cosine(left: &[f32], right: &[f32]) -> f32 {
    let len = left.len().min(right.len());
    if len == 0 {
        return 0.0;
    }
    let mut dot = 0.0f64;
    let mut left_norm = 0.0f64;
    let mut right_norm = 0.0f64;
    for idx in 0..len {
        let a = left[idx] as f64;
        let b = right[idx] as f64;
        dot += a * b;
        left_norm += a * a;
        right_norm += b * b;
    }
    if left_norm <= 0.0 || right_norm <= 0.0 {
        0.0
    } else {
        (dot / (left_norm.sqrt() * right_norm.sqrt())) as f32
    }
}

fn cosine_for_pair(left: &RecognitionFile, right: &RecognitionFile) -> Option<f32> {
    Some(cosine(left.embedding.as_ref()?, right.embedding.as_ref()?))
}

fn phash_distance_for_pair(left: &RecognitionFile, right: &RecognitionFile) -> Option<u32> {
    Some((left.phash? ^ right.phash?).count_ones())
}

fn classify_pair(
    left: &RecognitionFile,
    right: &RecognitionFile,
    dino_cosine: f32,
    phash_distance: Option<u32>,
) -> PairKind {
    if left.sha256.is_some() && left.sha256 == right.sha256 {
        return PairKind::Exact;
    }
    if let Some(distance) = phash_distance {
        if distance <= 4 && dino_cosine >= 0.90 {
            return PairKind::NearDuplicate;
        }
        if distance <= 10 && dino_cosine >= 0.97 {
            return PairKind::NearDuplicate;
        }
        if distance <= 18 && dino_cosine >= 0.94 && capture_delta_seconds(left, right) <= Some(3) {
            return PairKind::BurstSimilar;
        }
        if dino_cosine >= 0.92 {
            return PairKind::VisuallySimilar;
        }
    } else if dino_cosine >= 0.98 {
        return PairKind::VisuallySimilar;
    }
    PairKind::Rejected
}

fn capture_delta_seconds(left: &RecognitionFile, right: &RecognitionFile) -> Option<i64> {
    let left_time = chrono::DateTime::parse_from_rfc3339(left.created_time.as_ref()?).ok()?;
    let right_time = chrono::DateTime::parse_from_rfc3339(right.created_time.as_ref()?).ok()?;
    Some((left_time.timestamp() - right_time.timestamp()).abs())
}

fn pair_kind_label(kind: PairKind) -> &'static str {
    match kind {
        PairKind::Exact => "EXACT_DUPLICATE",
        PairKind::NearDuplicate => "NEAR_DUPLICATE",
        PairKind::BurstSimilar => "BURST_SIMILAR",
        PairKind::VisuallySimilar => "VISUALLY_SIMILAR",
        PairKind::Rejected => "REJECTED",
    }
}

fn representative_groups(
    files: &[RecognitionFile],
    pairs: &[VerifiedPair],
    kind: PairKind,
) -> Vec<Vec<usize>> {
    let mut pairs: Vec<_> = pairs.iter().filter(|pair| pair.kind == kind).collect();
    pairs.sort_by(|a, b| {
        b.cosine
            .partial_cmp(&a.cosine)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut groups: Vec<Vec<usize>> = Vec::new();
    let mut assigned = HashMap::<usize, usize>::new();
    for pair in pairs {
        let target_group = assigned
            .get(&pair.a)
            .copied()
            .or_else(|| assigned.get(&pair.b).copied());
        if let Some(group_idx) = target_group {
            let representative = groups[group_idx][0];
            let candidate = if assigned.contains_key(&pair.a) {
                pair.b
            } else {
                pair.a
            };
            if assigned.contains_key(&candidate) {
                continue;
            }
            if pair_satisfies_kind(
                files,
                pairs_for_lookup_kind(kind, pair, representative, candidate),
                representative,
                candidate,
                kind,
            ) {
                groups[group_idx].push(candidate);
                assigned.insert(candidate, group_idx);
            }
        } else {
            let group_idx = groups.len();
            groups.push(vec![pair.a, pair.b]);
            assigned.insert(pair.a, group_idx);
            assigned.insert(pair.b, group_idx);
        }
    }
    groups
}

fn pairs_for_lookup_kind(
    kind: PairKind,
    direct_pair: &VerifiedPair,
    rep: usize,
    member: usize,
) -> VerifiedPair {
    if (direct_pair.a == rep && direct_pair.b == member)
        || (direct_pair.a == member && direct_pair.b == rep)
    {
        direct_pair.clone()
    } else {
        VerifiedPair {
            a: rep,
            b: member,
            kind,
            cosine: 0.0,
            phash_distance: None,
        }
    }
}

fn pair_satisfies_kind(
    files: &[RecognitionFile],
    pair: VerifiedPair,
    representative: usize,
    candidate: usize,
    kind: PairKind,
) -> bool {
    if pair.cosine > 0.0 || pair.phash_distance.is_some() {
        return pair.kind == kind;
    }
    let Some(cosine) = cosine_for_pair(&files[representative], &files[candidate]) else {
        return false;
    };
    classify_pair(
        &files[representative],
        &files[candidate],
        cosine,
        phash_distance_for_pair(&files[representative], &files[candidate]),
    ) == kind
}

fn pair_metrics_for_member(
    files: &[RecognitionFile],
    pairs: &[VerifiedPair],
    representative: usize,
    member: usize,
) -> Option<(f32, Option<u32>)> {
    if representative == member {
        return Some((1.0, Some(0)));
    }
    pairs
        .iter()
        .find(|pair| {
            (pair.a == representative && pair.b == member)
                || (pair.a == member && pair.b == representative)
        })
        .map(|pair| (pair.cosine, pair.phash_distance))
        .or_else(|| {
            Some((
                cosine_for_pair(&files[representative], &files[member])?,
                phash_distance_for_pair(&files[representative], &files[member]),
            ))
        })
}

fn media_type_text(media_type: MediaType) -> &'static str {
    match media_type {
        MediaType::Image => "IMAGE",
        MediaType::Video => "VIDEO",
        MediaType::Sidecar => "SIDECAR",
        MediaType::Unsupported => "UNSUPPORTED",
    }
}

fn media_role_text(media_role: MediaRole) -> &'static str {
    match media_role {
        MediaRole::PrimaryImage => "PRIMARY_IMAGE",
        MediaRole::SingleVideo => "SINGLE_VIDEO",
        MediaRole::PairedVideo => "PAIRED_VIDEO",
        MediaRole::Sidecar => "SIDECAR",
        MediaRole::Unsupported => "UNSUPPORTED",
    }
}

fn asset_type_text(media_type: MediaType, pairing: Option<&str>) -> &'static str {
    if pairing == Some("LIVE_PHOTO") || pairing == Some("PROBABLE_LIVE_PHOTO") {
        return "LIVE_PHOTO";
    }
    match media_type {
        MediaType::Image => "IMAGE",
        MediaType::Video => "VIDEO",
        MediaType::Sidecar => "IMAGE",
        MediaType::Unsupported => "IMAGE",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scan_planner::{AnalysisPlan, WorkDecision};
    use crate::scanner::ScannedMediaFile;

    fn embedding_blob(first: f32) -> Vec<u8> {
        let mut values = vec![1.0f32; scan_planner::EMBEDDING_DIMENSION as usize];
        values[0] = first;
        let mut bytes = Vec::with_capacity(values.len() * 2);
        for value in values {
            bytes.extend_from_slice(&f16::from_f32(value).to_le_bytes());
        }
        bytes
    }

    fn reuse_nothing() -> AnalysisPlan {
        AnalysisPlan {
            metadata: WorkDecision::Compute,
            quick_hash: WorkDecision::Compute,
            sha256: WorkDecision::Compute,
            phash: WorkDecision::Compute,
            video_fingerprint: WorkDecision::NotRequired,
            ai_embedding: WorkDecision::Compute,
            ann_index: WorkDecision::Compute,
            grouping_rebuild: true,
        }
    }

    fn image(name: &str, sha: &str, phash: u64, embedding_first: f32) -> ScannedMediaFile {
        ScannedMediaFile {
            plan: reuse_nothing(),
            asset_key: name.to_string(),
            relative_path: name.to_string(),
            file_name: name.to_string(),
            extension: "jpg".to_string(),
            media_type: MediaType::Image,
            media_role: MediaRole::PrimaryImage,
            file_size: 1_000,
            created_time: None,
            modified_time: "2026-01-01T00:00:00+00:00".to_string(),
            width: Some(4_000),
            height: Some(3_000),
            duration_ms: None,
            container: Some("JPG".to_string()),
            video_codec: None,
            audio_codec: None,
            frame_rate: None,
            content_identifier: None,
            quick_hash: Some(format!("{phash:016x}")),
            sha256: Some(sha.to_string()),
            phash: Some(phash),
            embedding: Some(embedding_blob(embedding_first)),
            ai_model_id: Some(scan_planner::EMBEDDING_MODEL_ID.to_string()),
            ai_model_hash: Some("model-hash".to_string()),
            ai_preprocess_version: Some(scan_planner::EMBEDDING_PREPROCESS_VERSION),
            embedding_dimension: Some(scan_planner::EMBEDDING_DIMENSION),
            embedding_dtype: Some(scan_planner::EMBEDDING_DTYPE.to_string()),
            grouping_signature: Some("test:v1".to_string()),
            scan_state: "SUCCESS".to_string(),
            live_photo_pairing: None,
        }
    }

    /// Exact duplicates and near duplicates must not share a table: the
    /// duplicates tab is the only place where copies are pre-selected for
    /// deletion, so anything that lands there has to be provably identical.
    #[test]
    fn exact_and_near_duplicates_are_stored_separately() {
        let dir = tempfile::tempdir().unwrap();
        let paths = PortablePaths::from_root(dir.path().join("PhotoCleaner"));
        paths.ensure_layout().unwrap();
        let mut db = Database::open(&paths).unwrap();
        let library = db.upsert_library(dir.path()).unwrap();

        let files = vec![
            // Byte identical pair.
            image("a1.jpg", "sha-identical", 0x0000_0000_0000_0000, 1.0),
            image("a2.jpg", "sha-identical", 0x0000_0000_0000_0000, 1.0),
            // Different bytes, pHash one bit apart, visually near identical.
            image("b1.jpg", "sha-b1", 0xFFFF_FFFF_FFFF_FFF0, 1.0),
            image("b2.jpg", "sha-b2", 0xFFFF_FFFF_FFFF_FFF1, 0.98),
        ];
        db.insert_media_batch(&library.id, &files).unwrap();

        let summary = db.rebuild_recognition_groups(&library.id).unwrap();
        // The SHA-256 pass and the embedding pass both emit the identical pair,
        // so exact_pairs counts it twice. That double count is pre-existing and
        // cosmetic; what this test pins down is which table each kind lands in.
        assert!(
            summary.exact_pairs >= 1,
            "expected at least one SHA-256 identical pair, got {}",
            summary.exact_pairs
        );
        assert_eq!(summary.near_duplicate_pairs, 1, "expected one near pair");

        let results = db.load_cleanup_results().unwrap();

        assert_eq!(results.duplicate_groups.len(), 1);
        assert!(
            results
                .duplicate_groups
                .iter()
                .all(|group| group.kind == "EXACT_DUPLICATE"),
            "duplicate_groups must only ever hold EXACT_DUPLICATE, found {:?}",
            results
                .duplicate_groups
                .iter()
                .map(|group| group.kind.clone())
                .collect::<Vec<_>>()
        );
        assert_eq!(results.duplicate_groups[0].members.len(), 2);

        assert!(
            results
                .similarity_groups
                .iter()
                .any(|group| group.kind == "NEAR_DUPLICATE"),
            "near duplicates must be reachable from the similarity tables, found {:?}",
            results
                .similarity_groups
                .iter()
                .map(|group| group.kind.clone())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn every_group_has_exactly_one_recommended_keep() {
        let dir = tempfile::tempdir().unwrap();
        let paths = PortablePaths::from_root(dir.path().join("PhotoCleaner"));
        paths.ensure_layout().unwrap();
        let mut db = Database::open(&paths).unwrap();
        let library = db.upsert_library(dir.path()).unwrap();
        db.insert_media_batch(
            &library.id,
            &[
                image("a1.jpg", "same", 0, 1.0),
                image("a2.jpg", "same", 0, 1.0),
            ],
        )
        .unwrap();
        db.rebuild_recognition_groups(&library.id).unwrap();

        let results = db.load_cleanup_results().unwrap();
        for group in results
            .duplicate_groups
            .iter()
            .chain(&results.similarity_groups)
        {
            let keepers = group
                .members
                .iter()
                .filter(|member| member.is_recommended_keep)
                .count();
            assert_eq!(
                keepers, 1,
                "group {} recommends {keepers} keepers",
                group.id
            );
        }
    }
}
