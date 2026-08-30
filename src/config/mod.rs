use std::fs;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::paths::PortablePaths;

/// Bumped whenever the meaning of a recognition threshold changes, so a stored
/// grouping signature from an older build is never mistaken for a match.
const RECOGNITION_RULES_VERSION: u32 = 2;

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default)]
pub struct Settings {
    pub thumbnail_cache_limit_mb: u64,
    pub cpu_threads: CpuThreadSetting,
    pub recognition: RecognitionSettings,
    pub retention: RetentionSettings,
}

/// Which copy of a duplicate the user would rather keep.
///
/// Deliberately separate from [`RecognitionSettings`], and read at a different
/// stage: recognition decides what belongs together, retention decides which
/// member of the group to recommend keeping. Nothing here appears in
/// [`RecognitionSettings::signature`], because changing a folder preference must
/// not invalidate a single embedding or rebuild a single group.
///
/// Defaults are "off, with no folders", so upgrading an existing installation
/// leaves every recommendation exactly as it was.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(default)]
pub struct RetentionSettings {
    /// When false, the preferred-folder list is ignored entirely and the
    /// historical resolution-then-size rule decides.
    pub enabled: bool,
    /// Folders the user would rather keep, best first.
    pub preferred_folders: Vec<PreferredFolder>,
    /// Prefer a Live Photo that still has both halves over the same Live Photo
    /// that has lost one.
    pub prefer_complete_live_photo: bool,
    pub prefer_higher_resolution: bool,
    /// Prefer camera formats over the ones that usually mean an export.
    pub prefer_original_format: bool,
    /// Prefer the copy that still carries capture time, dimensions and content
    /// identifier.
    pub prefer_metadata_complete: bool,
    /// The weakest signal, and the last one consulted: a bigger file is only
    /// sometimes a better one.
    pub prefer_larger_original: bool,
}

impl Default for RetentionSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            preferred_folders: Vec::new(),
            // These only take effect once `enabled` is true, so switching the
            // policy on gives the user a sensible whole rather than a set of
            // switches they must find and turn on one at a time.
            prefer_complete_live_photo: true,
            prefer_higher_resolution: true,
            prefer_original_format: true,
            prefer_metadata_complete: true,
            prefer_larger_original: true,
        }
    }
}

/// One entry in the preferred-folder list.
///
/// `path` is library-relative when it is not an absolute path - `精选` means
/// "the 精选 folder inside whichever library this file belongs to", so a
/// portable library keeps working after it is moved to another drive. An
/// absolute path such as `D:\Photos\精选` is matched against the file's full
/// location instead.
///
/// `priority` is 1-based and lower is stronger: 1 beats 2. Values below 1 are
/// clamped rather than rejected, because this file is hand-editable and a `0`
/// should not quietly turn a rule off.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(default)]
pub struct PreferredFolder {
    pub path: String,
    pub priority: i32,
}

impl Default for PreferredFolder {
    fn default() -> Self {
        Self {
            path: String::new(),
            priority: 1,
        }
    }
}

/// Every threshold the duplicate/similarity classifier uses.
///
/// These values used to be written twice: once here, where the user could edit
/// them, and once as literals inside `classify_pair` and the candidate sweep,
/// which is what the algorithm actually read. Editing the config changed
/// nothing. This struct is now the single source of truth.
///
/// The defaults reproduce exactly what the hardcoded literals used to do, so
/// upgrading changes nothing until the user edits `config/settings.json`.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(default)]
pub struct RecognitionSettings {
    /// Minimum DINO cosine for a pair to even be considered. Below this the
    /// pair is never looked at again.
    pub candidate_cosine: f32,

    /// NEAR_DUPLICATE, strict arm: a very small pHash distance with ordinary
    /// visual agreement.
    pub near_duplicate_phash_strict: u32,
    pub near_duplicate_cosine_strict: f32,

    /// NEAR_DUPLICATE, relaxed arm: a larger pHash distance is only accepted
    /// when the visual agreement is much stronger.
    pub near_duplicate_phash_standard: u32,
    pub near_duplicate_cosine_standard: f32,

    /// BURST_SIMILAR: loose pHash distance, high cosine, and shots taken within
    /// a few seconds of each other.
    pub burst_phash_loose: u32,
    pub burst_cosine: f32,
    pub burst_max_capture_gap_seconds: i64,

    /// VISUALLY_SIMILAR: high visual agreement without duplicate evidence.
    pub visually_similar_cosine: f32,
    /// Same, for pairs where at least one side has no pHash at all. Without
    /// that corroborating signal the bar is deliberately higher.
    pub visually_similar_cosine_without_phash: f32,
}

impl Default for RecognitionSettings {
    fn default() -> Self {
        Self {
            candidate_cosine: 0.90,
            near_duplicate_phash_strict: 4,
            near_duplicate_cosine_strict: 0.90,
            near_duplicate_phash_standard: 10,
            near_duplicate_cosine_standard: 0.97,
            burst_phash_loose: 18,
            burst_cosine: 0.94,
            burst_max_capture_gap_seconds: 3,
            visually_similar_cosine: 0.92,
            visually_similar_cosine_without_phash: 0.98,
        }
    }
}

impl RecognitionSettings {
    /// A stable fingerprint of every value that can change grouping.
    ///
    /// Stored per file as `grouping_signature`. When it moves, grouping is
    /// rebuilt from the embeddings and pHashes already on disk; no DINO
    /// inference is rerun, because none of these values affect the embedding.
    pub fn signature(&self) -> String {
        format!(
            "v{}|cand{:.4}|nd{}@{:.4}|nd{}@{:.4}|burst{}@{:.4}/{}s|vs{:.4}|vsnp{:.4}",
            RECOGNITION_RULES_VERSION,
            self.candidate_cosine,
            self.near_duplicate_phash_strict,
            self.near_duplicate_cosine_strict,
            self.near_duplicate_phash_standard,
            self.near_duplicate_cosine_standard,
            self.burst_phash_loose,
            self.burst_cosine,
            self.burst_max_capture_gap_seconds,
            self.visually_similar_cosine,
            self.visually_similar_cosine_without_phash
        )
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "mode", content = "threads")]
pub enum CpuThreadSetting {
    Auto,
    Fixed(usize),
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            thumbnail_cache_limit_mb: 1024,
            cpu_threads: CpuThreadSetting::Auto,
            recognition: RecognitionSettings::default(),
            retention: RetentionSettings::default(),
        }
    }
}

impl Default for CpuThreadSetting {
    fn default() -> Self {
        Self::Auto
    }
}

impl Settings {
    /// Reads `config/settings.json`, creating it with defaults when absent.
    ///
    /// The struct is `#[serde(default)]`, so a file written by an older build
    /// keeps the keys it does have and picks up defaults for the rest instead
    /// of failing to parse and silently reverting everything.
    pub fn load_or_create(paths: &PortablePaths) -> Result<Self> {
        if paths.settings_file.exists() {
            let text = fs::read_to_string(&paths.settings_file).context("无法读取配置文件")?;
            return serde_json::from_str(&text).context("配置文件格式错误");
        }
        let settings = Self::default();
        settings.save(paths)?;
        Ok(settings)
    }

    pub fn save(&self, paths: &PortablePaths) -> Result<()> {
        let text = serde_json::to_string_pretty(self).context("无法生成配置文件")?;
        fs::write(&paths.settings_file, text).context("无法写入配置文件")
    }

    pub fn resolved_cpu_threads(&self) -> usize {
        match self.cpu_threads {
            CpuThreadSetting::Auto => num_cpus::get().saturating_sub(1).max(1),
            CpuThreadSetting::Fixed(n) => n.max(1),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The defaults must reproduce the literals that used to live inside
    /// classify_pair, so upgrading does not silently reclassify a library.
    #[test]
    fn defaults_match_the_previously_hardcoded_thresholds() {
        let recognition = RecognitionSettings::default();
        assert_eq!(recognition.candidate_cosine, 0.90);
        assert_eq!(recognition.near_duplicate_phash_strict, 4);
        assert_eq!(recognition.near_duplicate_cosine_strict, 0.90);
        assert_eq!(recognition.near_duplicate_phash_standard, 10);
        assert_eq!(recognition.near_duplicate_cosine_standard, 0.97);
        assert_eq!(recognition.burst_phash_loose, 18);
        assert_eq!(recognition.burst_cosine, 0.94);
        assert_eq!(recognition.burst_max_capture_gap_seconds, 3);
        assert_eq!(recognition.visually_similar_cosine, 0.92);
        assert_eq!(recognition.visually_similar_cosine_without_phash, 0.98);
    }

    #[test]
    fn signature_changes_when_any_threshold_changes() {
        let base = RecognitionSettings::default();
        let baseline = base.signature();

        let mut tightened = base.clone();
        tightened.near_duplicate_phash_strict = 3;
        assert_ne!(tightened.signature(), baseline);

        let mut looser = base.clone();
        looser.visually_similar_cosine = 0.90;
        assert_ne!(looser.signature(), baseline);

        let mut slower_burst = base.clone();
        slower_burst.burst_max_capture_gap_seconds = 5;
        assert_ne!(slower_burst.signature(), baseline);

        assert_eq!(RecognitionSettings::default().signature(), baseline);
    }

    #[test]
    fn a_config_file_from_an_older_build_still_loads() {
        // No `recognition` block, and the legacy flat threshold keys that this
        // build no longer understands.
        let legacy = r#"{
            "thumbnail_cache_limit_mb": 256,
            "phash_threshold_strict": 4,
            "phash_threshold_standard": 8,
            "phash_threshold_loose": 12,
            "ai_high_similarity": 0.92,
            "ai_possible_similarity": 0.84,
            "cpu_threads": { "mode": "Fixed", "threads": 6 }
        }"#;
        let settings: Settings = serde_json::from_str(legacy).unwrap();
        assert_eq!(settings.thumbnail_cache_limit_mb, 256);
        assert_eq!(settings.resolved_cpu_threads(), 6);
        assert_eq!(settings.recognition, RecognitionSettings::default());
        // The retention block did not exist when this file was written, and an
        // upgrade must not turn a new policy on behind the user's back.
        assert_eq!(settings.retention, RetentionSettings::default());
        assert!(!settings.retention.enabled);
        assert!(settings.retention.preferred_folders.is_empty());
    }

    /// A folder list written by hand, with keys left out.
    #[test]
    fn a_partial_retention_block_fills_in_the_rest() {
        let partial = r#"{
            "retention": {
                "enabled": true,
                "preferred_folders": [
                    { "path": "精选", "priority": 1 },
                    { "path": "原片" }
                ]
            }
        }"#;
        let settings: Settings = serde_json::from_str(partial).unwrap();
        assert!(settings.retention.enabled);
        assert_eq!(settings.retention.preferred_folders.len(), 2);
        assert_eq!(settings.retention.preferred_folders[0].path, "精选");
        // `priority` was omitted, so it defaults to the strongest rank rather
        // than to zero, which would have read as "no opinion".
        assert_eq!(settings.retention.preferred_folders[1].priority, 1);
        assert!(settings.retention.prefer_complete_live_photo);
    }

    /// Retention must never move the grouping signature: reordering a folder
    /// list would otherwise rebuild every group in the library.
    #[test]
    fn retention_settings_do_not_affect_the_grouping_signature() {
        let baseline = Settings::default();
        let mut with_policy = Settings::default();
        with_policy.retention.enabled = true;
        with_policy.retention.preferred_folders = vec![PreferredFolder {
            path: "精选".to_string(),
            priority: 1,
        }];
        assert_eq!(
            with_policy.recognition.signature(),
            baseline.recognition.signature()
        );
    }

    #[test]
    fn a_partial_recognition_block_keeps_defaults_for_the_rest() {
        let partial = r#"{ "recognition": { "burst_cosine": 0.99 } }"#;
        let settings: Settings = serde_json::from_str(partial).unwrap();
        assert_eq!(settings.recognition.burst_cosine, 0.99);
        assert_eq!(settings.recognition.candidate_cosine, 0.90);
        assert_eq!(settings.thumbnail_cache_limit_mb, 1024);
    }

    #[test]
    fn round_trips_through_json() {
        let mut settings = Settings::default();
        settings.recognition.burst_phash_loose = 21;
        settings.retention.enabled = true;
        settings.retention.preferred_folders = vec![PreferredFolder {
            path: "D:\\Photos\\精选".to_string(),
            priority: 1,
        }];
        let text = serde_json::to_string(&settings).unwrap();
        let parsed: Settings = serde_json::from_str(&text).unwrap();
        assert_eq!(parsed.recognition, settings.recognition);
        assert_eq!(parsed.retention, settings.retention);
        assert_eq!(parsed.thumbnail_cache_limit_mb, 1024);
    }
}
