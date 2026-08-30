//! Which copy to keep, and why.
//!
//! Recognition decides *what belongs together*. Retention decides *which member
//! of a group the user probably wants to keep*. They are deliberately two
//! stages, and this module is the whole of the second one:
//!
//! ```text
//! scan -> recognition / grouping -> retention -> recommended_keep
//! ```
//!
//! Nothing here may flow backwards. A folder preference must never raise a
//! similarity, pull a file into a group, or change a classification - if it
//! could, the user would be tuning the detector every time they reordered a
//! list of folders, and the validation numbers would stop measuring the
//! detector at all. This module therefore reads only what recognition already
//! decided and never writes to it.
//!
//! The previous rule was a single expression in `database::mark_recommended_keep`:
//! pixels first, file size as the tiebreak. It had no idea that the user keeps
//! their edited selects in one folder and phone dumps in another, and it had no
//! way to say why it chose what it chose.
//!
//! # Precedence
//!
//! Candidates are ordered on a tuple, not a weighted sum, because the layers are
//! not interchangeable: a stated folder preference has to beat a slightly larger
//! pixel count outright, and no weighting is safe from a large enough image.
//!
//! 1. **Readable.** A copy the scanner could not read is never the recommended
//!    keep while a readable one exists. This is the one thing that outranks the
//!    user's folder order, because "keep the file in 精选" cannot mean "keep the
//!    corrupt one".
//! 2. **Preferred folder.** The user's explicit list, best priority first.
//! 3. **Asset completeness.** A Live Photo missing its partner loses to the same
//!    Live Photo that still has both halves.
//! 4. **Resolution.** More pixels.
//! 5. **Format.** Camera originals over formats that usually mean an export.
//! 6. **Metadata completeness.** Capture time, dimensions, content identifier.
//! 7. **File size.** Weak evidence only, and last: bigger is not better, it is
//!    just occasionally correlated with better.
//!
//! Ties below all of that are broken on `(relative path, file id)` so the same
//! library always produces the same recommendation.

pub mod export;

use std::cmp::Ordering;
use std::path::Path;

use crate::config::{PreferredFolder, RetentionSettings};

/// What the asset this candidate belongs to is made of.
///
/// Retention scores whole assets, never loose files: a Live Photo is one
/// candidate whose components happen to be a HEIC, a MOV and possibly an AAE.
/// Scoring the HEIC and the MOV separately would let a group recommend keeping
/// half of one asset and half of another.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AssetComponents {
    pub has_image: bool,
    pub has_video: bool,
    pub has_sidecar: bool,
    pub file_count: usize,
}

/// One member of a group, as retention sees it.
#[derive(Clone, Debug, Default)]
pub struct RetentionCandidate {
    pub file_id: i64,
    pub asset_id: i64,
    /// Absolute root of the library this file belongs to.
    pub library_root: String,
    /// Path within that library, as stored by the scanner.
    pub relative_path: String,
    /// `LIVE_PHOTO`, `IMAGE`, `VIDEO`, as recorded on the asset.
    pub asset_type: String,
    pub width: Option<i64>,
    pub height: Option<i64>,
    pub file_size: u64,
    pub capture_time: Option<String>,
    pub content_identifier: Option<String>,
    pub components: AssetComponents,
    /// False when the scanner's last look at this file ended in a failure
    /// state. Used instead of touching the disk: this runs while the results
    /// list is being drawn, and stat-ing every candidate would stall the UI.
    pub readable: bool,
}

impl RetentionCandidate {
    fn extension(&self) -> String {
        Path::new(&self.relative_path.replace('\\', "/"))
            .extension()
            .and_then(|value| value.to_str())
            .unwrap_or_default()
            .to_ascii_lowercase()
    }

    fn pixels(&self) -> u128 {
        let width = self.width.unwrap_or_default().max(0) as u128;
        let height = self.height.unwrap_or_default().max(0) as u128;
        width * height
    }

    fn is_live_photo(&self) -> bool {
        self.asset_type.eq_ignore_ascii_case("LIVE_PHOTO")
    }

    /// 1 when nothing is missing, 0 when a Live Photo has lost a half.
    ///
    /// A plain photo scores 1 rather than 0: it is not an incomplete anything.
    /// Deliberately *not* scored above a plain photo either - "prefer the
    /// complete Live Photo" is a rule about two versions of the same Live
    /// Photo, and stretching it into "a Live Photo beats a photo" would be the
    /// kind of arbitrary judgement this scoring is supposed to avoid.
    fn completeness(&self) -> u8 {
        if !self.is_live_photo() {
            return 1;
        }
        if self.components.file_count == 0 {
            // Nothing is known about this asset's components. Absence of
            // evidence is not evidence of a missing half, and scoring it as
            // incomplete would push a perfectly good copy down the list.
            return 1;
        }
        if self.components.has_image && self.components.has_video {
            1
        } else {
            0
        }
    }

    /// How many of the metadata fields worth having are actually present.
    fn metadata_fields(&self) -> u8 {
        let mut fields = 0;
        if self
            .capture_time
            .as_deref()
            .map(|value| !value.trim().is_empty())
            .unwrap_or(false)
        {
            fields += 1;
        }
        if self.width.unwrap_or_default() > 0 && self.height.unwrap_or_default() > 0 {
            fields += 1;
        }
        if self
            .content_identifier
            .as_deref()
            .map(|value| !value.trim().is_empty())
            .unwrap_or(false)
        {
            fields += 1;
        }
        fields
    }
}

/// Ranks formats by how likely they are to be the original rather than an export.
///
/// Kept coarse on purpose. A JPEG can be a camera original or a re-save, and
/// nothing in the file reliably says which, so the ranking only separates the
/// cases where the format itself is evidence.
fn format_rank(extension: &str) -> u8 {
    match extension {
        // Camera capture formats.
        "heic" | "heif" | "dng" | "raw" | "arw" | "cr2" | "cr3" | "nef" | "orf" | "rw2" => 3,
        // Could be an original; usually is.
        "jpg" | "jpeg" | "tif" | "tiff" => 2,
        // Usually the result of an export, a screenshot, or a web save.
        "png" | "bmp" | "webp" => 1,
        _ => 0,
    }
}

/// A preferred-folder rule with its path already split for comparison.
#[derive(Clone, Debug)]
struct CompiledRule {
    /// The path exactly as the user wrote it, for reporting.
    display: String,
    /// Lower-cased path components.
    components: Vec<String>,
    /// True when the rule names an absolute location rather than a path inside
    /// a library.
    absolute: bool,
    priority: i32,
    /// Position in the configured list, so equally specific and equally ranked
    /// rules still resolve in a fixed order.
    order: usize,
}

/// Which rule a file matched.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FolderMatch {
    pub folder: String,
    pub priority: i32,
    /// Number of path components the rule matched. A deeper rule is a more
    /// deliberate statement about that folder, so it wins.
    pub depth: usize,
}

/// The preferred-folder list, compiled once per load rather than per candidate.
#[derive(Clone, Debug, Default)]
pub struct FolderMatcher {
    rules: Vec<CompiledRule>,
}

impl FolderMatcher {
    pub fn compile(folders: &[PreferredFolder]) -> Self {
        let rules = folders
            .iter()
            .enumerate()
            .filter_map(|(order, folder)| {
                let raw = folder.path.trim();
                if raw.is_empty() {
                    return None;
                }
                let components = path_components(raw);
                if components.is_empty() {
                    return None;
                }
                Some(CompiledRule {
                    display: folder.path.clone(),
                    components,
                    absolute: is_absolute_path(raw),
                    priority: folder.priority,
                    order,
                })
            })
            .collect();
        Self { rules }
    }

    /// Finds the rule that governs this file.
    ///
    /// An absolute rule is matched against the file's full path; a relative one
    /// against its path inside the library, so a portable library still obeys
    /// "精选 first" after the whole folder is moved to another drive.
    ///
    /// When several rules match, the deepest one wins, then the best priority,
    /// then the earlier position in the list. Every step is total, so the answer
    /// never depends on iteration order.
    pub fn match_file(&self, library_root: &str, relative_path: &str) -> Option<FolderMatch> {
        if self.rules.is_empty() {
            return None;
        }
        let relative = path_components(relative_path);
        let root = path_components(library_root);
        let mut absolute = root.clone();
        absolute.extend(relative.iter().cloned());

        let mut best: Option<(usize, &CompiledRule)> = None;
        for rule in &self.rules {
            let subject = if rule.absolute { &absolute } else { &relative };
            if !starts_with_components(subject, &rule.components) {
                continue;
            }
            // Depth is measured on the file's full path for every rule, not on
            // whichever subject it happened to be matched against. A relative
            // rule of one component inside a library three folders deep is a
            // narrower statement than an absolute rule naming the drive, and
            // comparing the raw component counts would have said the opposite.
            let depth = if rule.absolute {
                rule.components.len()
            } else {
                root.len() + rule.components.len()
            };
            best = Some(match best {
                None => (depth, rule),
                Some(current) => {
                    if better_rule((depth, rule), current) {
                        (depth, rule)
                    } else {
                        current
                    }
                }
            });
        }
        best.map(|(depth, rule)| FolderMatch {
            folder: rule.display.clone(),
            priority: rule.priority,
            depth,
        })
    }
}

/// Total order over matching rules: deeper first, then better priority, then
/// earlier in the list.
fn better_rule(candidate: (usize, &CompiledRule), current: (usize, &CompiledRule)) -> bool {
    match candidate.0.cmp(&current.0) {
        std::cmp::Ordering::Greater => true,
        std::cmp::Ordering::Less => false,
        std::cmp::Ordering::Equal => match candidate.1.priority.cmp(&current.1.priority) {
            std::cmp::Ordering::Less => true,
            std::cmp::Ordering::Greater => false,
            std::cmp::Ordering::Equal => candidate.1.order < current.1.order,
        },
    }
}

/// Splits a path into lower-cased comparison components.
///
/// Written by hand rather than with `Path::components` because the scanner
/// stores Windows-style relative paths with `\`, and on any other platform
/// `Path` reads `a\b` as one component - which would make every test of this
/// function pass for the wrong reason. Both separators are accepted, `.`
/// segments are dropped, and the `\\?\` long-path prefix is removed so that the
/// same folder matches whether or not it arrived through a long path.
fn path_components(path: &str) -> Vec<String> {
    let trimmed = path
        .trim()
        .trim_start_matches("\\\\?\\UNC\\")
        .trim_start_matches("\\\\?\\");
    trimmed
        .split(['\\', '/'])
        .filter(|part| !part.is_empty() && *part != ".")
        .map(|part| {
            // Windows ignores trailing dots and spaces on a path component, so
            // `精选.` and `精选` are the same folder - but `..` is all dots and
            // must survive as itself rather than being trimmed to nothing.
            let trimmed = if part.chars().all(|character| character == '.') {
                part
            } else {
                part.trim_end_matches(['.', ' '])
            };
            // Windows compares paths case-insensitively; `to_lowercase` is the
            // Unicode-aware form, so a folder named 精选 or Ärger behaves the
            // same as one named in ASCII.
            trimmed.to_lowercase()
        })
        .filter(|part| !part.is_empty())
        .collect()
}

fn is_absolute_path(path: &str) -> bool {
    let trimmed = path.trim().trim_start_matches("\\\\?\\");
    if trimmed.starts_with("\\\\") || trimmed.starts_with('/') || trimmed.starts_with('\\') {
        return true;
    }
    let bytes = trimmed.as_bytes();
    bytes.len() >= 2 && bytes[0].is_ascii_alphabetic() && bytes[1] == b':'
}

/// Component-wise prefix test.
///
/// This is the whole reason paths are split at all: `starts_with` on the raw
/// string reports that `D:\Photos\精选备份\IMG_0001.JPG` is inside
/// `D:\Photos\精选`, and the user would find a photo they meant to keep
/// recommended for deletion because two folder names share a prefix.
fn starts_with_components(subject: &[String], prefix: &[String]) -> bool {
    if prefix.len() > subject.len() {
        return false;
    }
    subject
        .iter()
        .zip(prefix.iter())
        .all(|(left, right)| left == right)
}

/// Why a candidate was recommended. Reported so a report or a UI can say
/// something better than a number.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RetentionReason {
    PreferredFolder {
        folder: String,
        priority: i32,
    },
    NoPreferredFolderMatch,
    CompleteLivePhoto,
    IncompleteLivePhoto,
    HigherResolution {
        pixels: u128,
    },
    OriginalFormat {
        extension: String,
    },
    MetadataComplete {
        fields: u8,
    },
    LargerFile {
        bytes: u64,
    },
    /// Every other candidate was unreadable at the last scan.
    OnlyReadableCopy,
    /// Nothing above separated the candidates; the stable tiebreak chose.
    DeterministicTieBreak,
}

impl RetentionReason {
    pub fn label(&self) -> String {
        match self {
            Self::PreferredFolder { folder, priority } => {
                format!("preferred folder {folder} (priority {priority})")
            }
            Self::NoPreferredFolderMatch => "no preferred folder matched".to_string(),
            Self::CompleteLivePhoto => "complete Live Photo".to_string(),
            Self::IncompleteLivePhoto => "incomplete Live Photo".to_string(),
            Self::HigherResolution { pixels } => format!("higher resolution ({pixels} px)"),
            Self::OriginalFormat { extension } => format!("original format .{extension}"),
            Self::MetadataComplete { fields } => format!("metadata fields present: {fields}"),
            Self::LargerFile { bytes } => format!("larger file ({bytes} bytes)"),
            Self::OnlyReadableCopy => "the only readable copy".to_string(),
            Self::DeterministicTieBreak => "stable tiebreak on path and id".to_string(),
        }
    }
}

/// What settled the group. Reported separately from the reasons so a report can
/// count "decided by the user's folder order" against "fell through to quality".
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecisionBasis {
    /// The policy is off; the historical resolution-then-size rule chose.
    PolicyDisabled,
    /// The winner sits in a better preferred folder than the runner-up.
    PreferredFolder,
    /// Folders were equal or absent; a quality layer chose.
    QualityFallback,
    /// Everything was equal; the stable tiebreak chose.
    DeterministicTieBreak,
}

impl DecisionBasis {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::PolicyDisabled => "POLICY_DISABLED",
            Self::PreferredFolder => "PREFERRED_FOLDER",
            Self::QualityFallback => "QUALITY_FALLBACK",
            Self::DeterministicTieBreak => "DETERMINISTIC_TIE_BREAK",
        }
    }
}

/// The recommendation for one group, with its evidence.
#[derive(Clone, Debug)]
pub struct RetentionDecision {
    pub recommended_file_id: i64,
    pub recommended_asset_id: i64,
    /// A flat number for display only. **Never sort by this**: the real order is
    /// the tuple in [`ScoreKey`], and collapsing it into one integer is lossy by
    /// construction.
    pub score: u64,
    pub basis: DecisionBasis,
    pub matched_preferred_folder: Option<String>,
    pub matched_folder_priority: Option<i32>,
    pub reasons: Vec<RetentionReason>,
}

impl RetentionDecision {
    /// One line, for `groups.csv` and the report.
    pub fn reason_text(&self) -> String {
        if self.reasons.is_empty() {
            return "no distinguishing signal".to_string();
        }
        self.reasons
            .iter()
            .map(RetentionReason::label)
            .collect::<Vec<_>>()
            .join("; ")
    }
}

/// The ordering key. Compared field by field, in this order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScoreKey {
    /// 1 readable, 0 not. First, so a corrupt copy never wins on folder order.
    readable: u8,
    /// Higher is better, so a missing match sorts below every real one.
    folder_score: u64,
    completeness: u8,
    pixels: u128,
    format_rank: u8,
    metadata_fields: u8,
    file_size: u64,
}

impl ScoreKey {
    fn comparison_tuple(&self) -> (u8, u64, u8, u128, u8, u8, u64) {
        (
            self.readable,
            self.folder_score,
            self.completeness,
            self.pixels,
            self.format_rank,
            self.metadata_fields,
            self.file_size,
        )
    }
}

impl Ord for ScoreKey {
    fn cmp(&self, other: &Self) -> Ordering {
        self.comparison_tuple().cmp(&other.comparison_tuple())
    }
}

impl PartialOrd for ScoreKey {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Priorities are turned into a descending score so the whole key can be
/// compared with a single `Ord`. Priority 1 is the strongest; anything below 1
/// is clamped rather than rejected, because a settings file is hand-editable and
/// a `0` should not silently disable the rule the user wrote.
const MAX_FOLDER_PRIORITY: i32 = 1_000_000;

fn folder_score(matched: Option<&FolderMatch>) -> u64 {
    match matched {
        None => 0,
        Some(matched) => {
            let priority = matched.priority.clamp(1, MAX_FOLDER_PRIORITY);
            (MAX_FOLDER_PRIORITY as u64 + 1).saturating_sub(priority as u64)
        }
    }
}

/// One candidate's score, and the match that produced it.
#[derive(Clone, Debug)]
pub struct RetentionScore {
    pub key: ScoreKey,
    pub matched: Option<FolderMatch>,
    pub display_score: u64,
}

/// Scores one candidate under one policy.
///
/// Every layer the policy turns off is flattened to a constant rather than
/// removed, so switching a preference off cannot reorder the layers below it.
pub fn score_asset_for_retention(
    policy: &RetentionSettings,
    matcher: &FolderMatcher,
    candidate: &RetentionCandidate,
) -> RetentionScore {
    let matched = if policy.enabled {
        matcher.match_file(&candidate.library_root, &candidate.relative_path)
    } else {
        None
    };
    let extension = candidate.extension();
    let key = ScoreKey {
        readable: u8::from(candidate.readable),
        folder_score: folder_score(matched.as_ref()),
        completeness: if policy.enabled && policy.prefer_complete_live_photo {
            candidate.completeness()
        } else {
            0
        },
        pixels: if !policy.enabled || policy.prefer_higher_resolution {
            candidate.pixels()
        } else {
            0
        },
        format_rank: if policy.enabled && policy.prefer_original_format {
            format_rank(&extension)
        } else {
            0
        },
        metadata_fields: if policy.enabled && policy.prefer_metadata_complete {
            candidate.metadata_fields()
        } else {
            0
        },
        file_size: if !policy.enabled || policy.prefer_larger_original {
            candidate.file_size
        } else {
            0
        },
    };
    let summary = display_score(&key);
    RetentionScore {
        key,
        matched,
        display_score: summary,
    }
}

/// A coarse, human-readable summary of the key. Lossy on purpose.
fn display_score(key: &ScoreKey) -> u64 {
    let folder = if key.folder_score > 0 { 1_000_000 } else { 0 };
    let megapixels = (key.pixels / 1_000_000).min(999) as u64;
    folder
        + u64::from(key.completeness) * 100_000
        + megapixels * 100
        + u64::from(key.format_rank) * 10
        + u64::from(key.metadata_fields)
}

/// Chooses the member of a group to recommend keeping.
///
/// Returns `None` only for an empty group. Every non-empty group gets exactly
/// one recommendation, which the UI relies on.
pub fn choose_recommended_keep(
    policy: &RetentionSettings,
    matcher: &FolderMatcher,
    candidates: &[RetentionCandidate],
) -> Option<RetentionDecision> {
    if candidates.is_empty() {
        return None;
    }
    let mut scored: Vec<(usize, RetentionScore)> = candidates
        .iter()
        .enumerate()
        .map(|(index, candidate)| (index, score_asset_for_retention(policy, matcher, candidate)))
        .collect();

    // Sorted rather than max-by so the runner-up is available: what separated
    // first from second is exactly what the report needs to explain.
    scored.sort_by(|left, right| {
        right
            .1
            .key
            .cmp(&left.1.key)
            .then_with(|| tie_break(&candidates[left.0]).cmp(&tie_break(&candidates[right.0])))
    });

    let (winner_index, winner) = scored[0].clone();
    let runner_up = scored.get(1).map(|(_, score)| score.clone());
    let candidate = &candidates[winner_index];

    let basis = decision_basis(policy, &winner, runner_up.as_ref());
    let reasons = reasons_for(policy, candidate, &winner, runner_up.as_ref());

    Some(RetentionDecision {
        recommended_file_id: candidate.file_id,
        recommended_asset_id: candidate.asset_id,
        score: winner.display_score,
        basis,
        matched_preferred_folder: winner.matched.as_ref().map(|m| m.folder.clone()),
        matched_folder_priority: winner.matched.as_ref().map(|m| m.priority),
        reasons,
    })
}

/// The stable tiebreak: lower-cased relative path, then file id.
///
/// Without this the winner among identical candidates depended on the order rows
/// came back from SQLite, which is stable in practice and guaranteed by nothing.
fn tie_break(candidate: &RetentionCandidate) -> (String, i64) {
    (candidate.relative_path.to_lowercase(), candidate.file_id)
}

fn decision_basis(
    policy: &RetentionSettings,
    winner: &RetentionScore,
    runner_up: Option<&RetentionScore>,
) -> DecisionBasis {
    if !policy.enabled {
        return DecisionBasis::PolicyDisabled;
    }
    let Some(runner_up) = runner_up else {
        // A group of one. Nothing was compared, so nothing chose.
        return DecisionBasis::DeterministicTieBreak;
    };
    if winner.key.folder_score != runner_up.key.folder_score {
        return DecisionBasis::PreferredFolder;
    }
    if winner.key == runner_up.key {
        return DecisionBasis::DeterministicTieBreak;
    }
    DecisionBasis::QualityFallback
}

/// The specific signals that put the winner ahead of the runner-up.
///
/// Only differences are listed. Reporting every property of the winner would
/// bury the one fact the user wants: what actually decided it.
fn reasons_for(
    policy: &RetentionSettings,
    candidate: &RetentionCandidate,
    winner: &RetentionScore,
    runner_up: Option<&RetentionScore>,
) -> Vec<RetentionReason> {
    let mut reasons = Vec::new();
    let Some(runner_up) = runner_up else {
        if let Some(matched) = &winner.matched {
            reasons.push(RetentionReason::PreferredFolder {
                folder: matched.folder.clone(),
                priority: matched.priority,
            });
        }
        return reasons;
    };

    if winner.key.readable > runner_up.key.readable {
        reasons.push(RetentionReason::OnlyReadableCopy);
    }
    if winner.key.folder_score > runner_up.key.folder_score {
        if let Some(matched) = &winner.matched {
            reasons.push(RetentionReason::PreferredFolder {
                folder: matched.folder.clone(),
                priority: matched.priority,
            });
        }
    } else if policy.enabled && winner.matched.is_none() && runner_up.matched.is_none() {
        reasons.push(RetentionReason::NoPreferredFolderMatch);
    }
    if winner.key.completeness > runner_up.key.completeness {
        reasons.push(RetentionReason::CompleteLivePhoto);
    }
    if winner.key.pixels > runner_up.key.pixels {
        reasons.push(RetentionReason::HigherResolution {
            pixels: winner.key.pixels,
        });
    }
    if winner.key.format_rank > runner_up.key.format_rank {
        reasons.push(RetentionReason::OriginalFormat {
            extension: candidate.extension(),
        });
    }
    if winner.key.metadata_fields > runner_up.key.metadata_fields {
        reasons.push(RetentionReason::MetadataComplete {
            fields: winner.key.metadata_fields,
        });
    }
    if winner.key.file_size > runner_up.key.file_size
        && winner.key.pixels == runner_up.key.pixels
        && winner.key.format_rank == runner_up.key.format_rank
    {
        reasons.push(RetentionReason::LargerFile {
            bytes: winner.key.file_size,
        });
    }
    if reasons.is_empty() {
        reasons.push(RetentionReason::DeterministicTieBreak);
    }
    reasons
}

#[cfg(test)]
mod tests {
    use super::*;

    fn policy(folders: &[(&str, i32)]) -> RetentionSettings {
        RetentionSettings {
            enabled: true,
            preferred_folders: folders
                .iter()
                .map(|(path, priority)| PreferredFolder {
                    path: (*path).to_string(),
                    priority: *priority,
                })
                .collect(),
            ..RetentionSettings::default()
        }
    }

    fn candidate(file_id: i64, relative_path: &str, width: i64, height: i64) -> RetentionCandidate {
        RetentionCandidate {
            file_id,
            asset_id: file_id,
            library_root: "D:\\Photos".to_string(),
            relative_path: relative_path.to_string(),
            asset_type: "IMAGE".to_string(),
            width: Some(width),
            height: Some(height),
            file_size: 1_000,
            capture_time: Some("2026-01-01T00:00:00+00:00".to_string()),
            content_identifier: None,
            components: AssetComponents {
                has_image: true,
                file_count: 1,
                ..AssetComponents::default()
            },
            readable: true,
        }
    }

    fn decide(
        settings: &RetentionSettings,
        candidates: &[RetentionCandidate],
    ) -> RetentionDecision {
        let matcher = FolderMatcher::compile(&settings.preferred_folders);
        choose_recommended_keep(settings, &matcher, candidates).unwrap()
    }

    /// The whole point of the feature: what the user said beats what the
    /// pixels say.
    #[test]
    fn preferred_folder_beats_quality_fallback() {
        let settings = policy(&[("精选", 1), ("原片", 2)]);
        let candidates = vec![
            candidate(1, "精选\\2026\\IMG_0001.HEIC", 3_000, 2_000),
            // Higher resolution, but the user ranked this folder second.
            candidate(2, "原片\\2026\\IMG_0001.HEIC", 6_000, 4_000),
        ];
        let decision = decide(&settings, &candidates);
        assert_eq!(decision.recommended_file_id, 1);
        assert_eq!(decision.basis, DecisionBasis::PreferredFolder);
        assert_eq!(decision.matched_folder_priority, Some(1));
    }

    #[test]
    fn nested_paths_below_a_preferred_folder_still_match() {
        let settings = policy(&[("精选", 1)]);
        let matcher = FolderMatcher::compile(&settings.preferred_folders);
        let matched = matcher.match_file("D:\\Photos", "精选\\2026\\03\\IMG_0001.HEIC");
        assert_eq!(matched.map(|m| m.priority), Some(1));
    }

    /// A string prefix test would call this a match and recommend deleting a
    /// photo the user meant to keep.
    #[test]
    fn a_similar_folder_name_does_not_match() {
        let settings = policy(&[("精选", 1)]);
        let matcher = FolderMatcher::compile(&settings.preferred_folders);
        assert!(matcher
            .match_file("D:\\Photos", "精选备份\\IMG_0001.JPG")
            .is_none());
        assert!(matcher
            .match_file("D:\\Photos", "精选\\IMG_0001.JPG")
            .is_some());
    }

    #[test]
    fn the_more_specific_preferred_path_wins() {
        let settings = policy(&[("D:\\Photos", 1), ("D:\\Photos\\精选", 3)]);
        let matcher = FolderMatcher::compile(&settings.preferred_folders);
        let matched = matcher
            .match_file("D:\\Photos", "精选\\2026\\IMG_0001.HEIC")
            .unwrap();
        assert_eq!(matched.folder, "D:\\Photos\\精选");
        assert_eq!(matched.priority, 3);
        assert_eq!(matched.depth, 3);
    }

    /// A relative rule and an absolute one have to be compared on the same
    /// ruler. `精选` inside `D:\Photos` names the same folder as
    /// `D:\Photos\精选`, and is narrower than `D:\Photos`.
    #[test]
    fn relative_and_absolute_rules_are_ranked_on_the_same_depth() {
        let settings = policy(&[("D:\\Photos", 1), ("精选", 3)]);
        let matcher = FolderMatcher::compile(&settings.preferred_folders);
        let matched = matcher
            .match_file("D:\\Photos", "精选\\2026\\IMG_0001.HEIC")
            .unwrap();
        assert_eq!(
            matched.folder, "精选",
            "the narrower rule must win even though it is written relatively"
        );
        assert_eq!(matched.depth, 3, "d: / photos / 精选");
    }

    /// Upgrading must not silently change anyone's recommendations.
    ///
    /// The one deliberate difference from the old rule is the readability
    /// layer, which applies whether or not the policy is on: recommending a
    /// copy the scanner could not read is never right.
    #[test]
    fn a_disabled_policy_reproduces_the_old_resolution_then_size_rule() {
        let settings = RetentionSettings::default();
        assert!(!settings.enabled);
        let candidates = vec![
            candidate(1, "精选\\IMG_0001.HEIC", 1_000, 1_000),
            candidate(2, "微信\\IMG_0001.JPG", 4_000, 3_000),
        ];
        let decision = decide(&settings, &candidates);
        assert_eq!(decision.recommended_file_id, 2, "more pixels still wins");
        assert_eq!(decision.basis, DecisionBasis::PolicyDisabled);
        assert!(decision.matched_preferred_folder.is_none());
    }

    #[test]
    fn higher_resolution_wins_when_no_folder_preference_applies() {
        let settings = policy(&[("精选", 1)]);
        let candidates = vec![
            candidate(1, "微信\\IMG_0001.JPG", 1_000, 1_000),
            candidate(2, "手机备份\\IMG_0001.JPG", 4_000, 3_000),
        ];
        let decision = decide(&settings, &candidates);
        assert_eq!(decision.recommended_file_id, 2);
        assert_eq!(decision.basis, DecisionBasis::QualityFallback);
        assert!(decision
            .reasons
            .iter()
            .any(|reason| matches!(reason, RetentionReason::HigherResolution { .. })));
    }

    #[test]
    fn a_complete_live_photo_beats_a_half_of_the_same_one() {
        let settings = policy(&[]);
        let mut complete = candidate(1, "备份A\\IMG_1001.HEIC", 4_000, 3_000);
        complete.asset_type = "LIVE_PHOTO".to_string();
        complete.components = AssetComponents {
            has_image: true,
            has_video: true,
            has_sidecar: true,
            file_count: 3,
        };
        let mut half = candidate(2, "备份B\\IMG_1001.HEIC", 4_000, 3_000);
        half.asset_type = "LIVE_PHOTO".to_string();
        half.components = AssetComponents {
            has_image: true,
            file_count: 1,
            ..AssetComponents::default()
        };
        let decision = decide(&settings, &[complete, half]);
        assert_eq!(decision.recommended_file_id, 1);
        assert!(decision
            .reasons
            .contains(&RetentionReason::CompleteLivePhoto));
    }

    /// The Live Photo in the preferred folder is kept as one asset, not
    /// assembled from the best file in each folder.
    #[test]
    fn a_live_photo_is_judged_as_one_asset() {
        let settings = policy(&[("精选", 1), ("备份", 2)]);
        let mut selected = candidate(1, "精选\\IMG_1001.HEIC", 4_000, 3_000);
        selected.asset_type = "LIVE_PHOTO".to_string();
        selected.asset_id = 100;
        selected.components = AssetComponents {
            has_image: true,
            has_video: true,
            has_sidecar: true,
            file_count: 3,
        };
        let mut backup = candidate(2, "备份\\IMG_1001.HEIC", 4_000, 3_000);
        backup.asset_type = "LIVE_PHOTO".to_string();
        backup.asset_id = 200;
        backup.components = AssetComponents {
            has_image: true,
            has_video: true,
            file_count: 2,
            ..AssetComponents::default()
        };
        let decision = decide(&settings, &[selected, backup]);
        assert_eq!(decision.recommended_asset_id, 100);
        assert_eq!(decision.basis, DecisionBasis::PreferredFolder);
    }

    /// "Keep what is in 精选" cannot mean "keep the corrupt one".
    #[test]
    fn an_unreadable_copy_never_wins_on_folder_priority_alone() {
        let settings = policy(&[("精选", 1), ("备份", 2)]);
        let mut broken = candidate(1, "精选\\IMG_0001.HEIC", 4_000, 3_000);
        broken.readable = false;
        let good = candidate(2, "备份\\IMG_0001.HEIC", 4_000, 3_000);
        let decision = decide(&settings, &[broken, good]);
        assert_eq!(decision.recommended_file_id, 2);
        assert!(decision
            .reasons
            .contains(&RetentionReason::OnlyReadableCopy));
    }

    #[test]
    fn identical_candidates_resolve_the_same_way_every_time() {
        let settings = policy(&[("精选", 1)]);
        let first = candidate(7, "精选\\b.JPG", 100, 100);
        let second = candidate(3, "精选\\a.JPG", 100, 100);
        let forward = decide(&settings, &[first.clone(), second.clone()]);
        let reversed = decide(&settings, &[second, first]);
        assert_eq!(forward.recommended_file_id, reversed.recommended_file_id);
        assert_eq!(
            forward.recommended_file_id, 3,
            "the lexically first path is the stable choice"
        );
        assert_eq!(forward.basis, DecisionBasis::DeterministicTieBreak);
    }

    #[test]
    fn unicode_and_case_are_handled_the_windows_way() {
        let settings = policy(&[("Photos\\精选", 1)]);
        let matcher = FolderMatcher::compile(&settings.preferred_folders);
        assert!(matcher
            .match_file("D:\\Library", "photos\\精选\\IMG_0001.HEIC")
            .is_some());
        assert!(matcher
            .match_file("D:\\Library", "PHOTOS\\精选\\IMG_0001.HEIC")
            .is_some());
    }

    #[test]
    fn an_absolute_rule_does_not_match_another_drive() {
        let settings = policy(&[("D:\\Photos\\精选", 1)]);
        let matcher = FolderMatcher::compile(&settings.preferred_folders);
        assert!(matcher
            .match_file("D:\\Photos", "精选\\IMG_0001.HEIC")
            .is_some());
        assert!(matcher
            .match_file("E:\\Photos", "精选\\IMG_0001.HEIC")
            .is_none());
    }

    #[test]
    fn a_long_path_prefix_matches_the_same_folder() {
        let settings = policy(&[("D:\\Photos\\精选", 1)]);
        let matcher = FolderMatcher::compile(&settings.preferred_folders);
        assert!(matcher
            .match_file("\\\\?\\D:\\Photos", "精选\\IMG_0001.HEIC")
            .is_some());
    }

    #[test]
    fn forward_and_backward_separators_are_the_same_path() {
        let settings = policy(&[("Photos/精选", 1)]);
        let matcher = FolderMatcher::compile(&settings.preferred_folders);
        assert!(matcher
            .match_file("D:\\Library", "Photos\\精选\\IMG_0001.HEIC")
            .is_some());
    }

    #[test]
    fn the_original_format_only_decides_when_everything_above_is_equal() {
        let settings = policy(&[]);
        let heic = candidate(1, "a\\IMG_0001.HEIC", 4_000, 3_000);
        let png = candidate(2, "a\\IMG_0001.PNG", 4_000, 3_000);
        let decision = decide(&settings, &[png, heic]);
        assert_eq!(decision.recommended_file_id, 1);
        assert!(decision
            .reasons
            .iter()
            .any(|reason| matches!(reason, RetentionReason::OriginalFormat { .. })));
    }

    /// File size is the weakest signal and must never outrank pixels.
    #[test]
    fn a_bigger_file_does_not_beat_a_larger_image() {
        let settings = policy(&[]);
        let mut bloated = candidate(1, "a\\IMG_0001.JPG", 1_000, 1_000);
        bloated.file_size = 50_000_000;
        let sharp = candidate(2, "a\\IMG_0002.JPG", 4_000, 3_000);
        let decision = decide(&settings, &[bloated, sharp]);
        assert_eq!(decision.recommended_file_id, 2);
    }

    #[test]
    fn an_empty_group_has_no_recommendation() {
        let settings = policy(&[("精选", 1)]);
        let matcher = FolderMatcher::compile(&settings.preferred_folders);
        assert!(choose_recommended_keep(&settings, &matcher, &[]).is_none());
    }

    #[test]
    fn the_reason_text_names_the_folder() {
        let settings = policy(&[("精选", 1), ("微信", 4)]);
        let candidates = vec![
            candidate(1, "微信\\IMG_0001.JPG", 4_000, 3_000),
            candidate(2, "精选\\IMG_0001.HEIC", 1_000, 1_000),
        ];
        let decision = decide(&settings, &candidates);
        assert_eq!(decision.recommended_file_id, 2);
        assert!(decision.reason_text().contains("精选"));
        assert!(decision.reason_text().contains("priority 1"));
    }
}
