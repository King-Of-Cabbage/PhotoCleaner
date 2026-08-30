//! CSV and Markdown exports that explain every retention decision.
//!
//! The point of these files is falsifiability. A recommendation the user
//! disagrees with is only useful if they can see *why* it was made - which
//! folder rule matched, at what priority, and what decided the group when no
//! rule matched at all.
//!
//! # Scope
//!
//! These exports describe retention and nothing else. They deliberately do not
//! contain a precision or recall number, a candidate count, or an ANN result:
//! retention runs strictly after recognition and cannot change any of them, and
//! a file that mixed the two would invite exactly the mistake this design is
//! built to prevent - tuning the detector by reordering a folder list.
//!
//! `candidate_pairs.csv` therefore is not written here. It belongs to a
//! recognition validation mode, which this build does not have yet; when it
//! lands, it can join on `file_id` and add the retention columns from
//! `assets.csv`, or call [`retention_report_section`] to append its own
//! Retention Policy section to a shared report.

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

use super::{
    score_asset_for_retention, AssetComponents, DecisionBasis, FolderMatcher, RetentionCandidate,
};
use crate::config::RetentionSettings;
use crate::database::{CleanupAsset, CleanupGroup, CleanupResults};
use crate::scan_state;

/// Group kinds whose members may be pre-selected for deletion.
///
/// Only byte-identical copies. A folder preference explains which copy to keep;
/// it never promotes a similarity judgement into an automatic deletion, and the
/// export says so per group so that the guarantee is visible rather than
/// implied.
const AUTO_PRESELECT_KINDS: &[&str] = &["EXACT_DUPLICATE"];

/// What was written, so the caller can tell the user where to look.
#[derive(Clone, Debug)]
pub struct ExportPaths {
    pub groups_csv: PathBuf,
    pub assets_csv: PathBuf,
    pub report_md: PathBuf,
}

/// Counts of how each group was settled. The two numbers section 15 of the
/// brief asks for are `preferred_folder` and `quality_fallback`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RetentionOutcomeCounts {
    pub preferred_folder: usize,
    pub quality_fallback: usize,
    pub deterministic_tie_break: usize,
    pub policy_disabled: usize,
    /// Groups whose keeper was fixed by an explicit `KEEP` on the group row, so
    /// retention was never consulted.
    pub not_decided_by_retention: usize,
}

impl RetentionOutcomeCounts {
    pub fn total(&self) -> usize {
        self.preferred_folder
            + self.quality_fallback
            + self.deterministic_tie_break
            + self.policy_disabled
            + self.not_decided_by_retention
    }

    fn record(&mut self, group: &CleanupGroup) {
        match group.retention.as_ref().map(|decision| decision.basis) {
            Some(DecisionBasis::PreferredFolder) => self.preferred_folder += 1,
            Some(DecisionBasis::QualityFallback) => self.quality_fallback += 1,
            Some(DecisionBasis::DeterministicTieBreak) => self.deterministic_tie_break += 1,
            Some(DecisionBasis::PolicyDisabled) => self.policy_disabled += 1,
            None => self.not_decided_by_retention += 1,
        }
    }
}

pub fn count_outcomes(results: &CleanupResults) -> RetentionOutcomeCounts {
    let mut counts = RetentionOutcomeCounts::default();
    for group in results
        .duplicate_groups
        .iter()
        .chain(&results.similarity_groups)
    {
        counts.record(group);
    }
    counts
}

/// Writes `groups.csv`, `assets.csv` and `validation_report.md` into `folder`.
pub fn write_retention_export(
    folder: &Path,
    policy: &RetentionSettings,
    results: &CleanupResults,
    components: &std::collections::HashMap<i64, AssetComponents>,
) -> Result<ExportPaths> {
    fs::create_dir_all(folder)
        .with_context(|| format!("cannot create export folder {}", folder.display()))?;
    let matcher = FolderMatcher::compile(&policy.preferred_folders);

    let groups_csv = folder.join("groups.csv");
    fs::write(&groups_csv, groups_csv_text(results))
        .with_context(|| format!("cannot write {}", groups_csv.display()))?;

    let assets_csv = folder.join("assets.csv");
    fs::write(
        &assets_csv,
        assets_csv_text(policy, &matcher, results, components),
    )
    .with_context(|| format!("cannot write {}", assets_csv.display()))?;

    let report_md = folder.join("validation_report.md");
    fs::write(&report_md, validation_report_text(policy, results))
        .with_context(|| format!("cannot write {}", report_md.display()))?;

    Ok(ExportPaths {
        groups_csv,
        assets_csv,
        report_md,
    })
}

pub fn groups_csv_text(results: &CleanupResults) -> String {
    let mut text = String::from(
        "group_table,group_id,group_kind,member_count,auto_preselect_allowed,\
recommended_file_id,recommended_asset_id,retention_score,decided_by,\
matched_preferred_folder,matched_folder_priority,recommended_keep_reason\n",
    );
    for group in results
        .duplicate_groups
        .iter()
        .chain(&results.similarity_groups)
    {
        let keeper = group
            .members
            .iter()
            .find(|member| member.is_recommended_keep);
        let decision = group.retention.as_ref();
        text.push_str(&format!(
            "{},{},{},{},{},{},{},{},{},{},{},{}\n",
            csv_field(&group.table_name),
            group.id,
            csv_field(&group.kind),
            group.members.len(),
            auto_preselect_allowed(&group.kind),
            decision
                .map(|d| d.recommended_file_id)
                .or_else(|| keeper.map(|member| member.file_id))
                .unwrap_or_default(),
            decision
                .map(|d| d.recommended_asset_id)
                .or_else(|| keeper.map(|member| member.asset_id))
                .unwrap_or_default(),
            decision.map(|d| d.score).unwrap_or_default(),
            csv_field(
                decision
                    .map(|d| d.basis.as_str())
                    .unwrap_or("EXPLICIT_KEEP")
            ),
            csv_field(
                decision
                    .and_then(|d| d.matched_preferred_folder.as_deref())
                    .unwrap_or("")
            ),
            decision
                .and_then(|d| d.matched_folder_priority)
                .map(|priority| priority.to_string())
                .unwrap_or_default(),
            csv_field(
                &decision
                    .map(|d| d.reason_text())
                    .unwrap_or_else(|| "explicit KEEP on the group row".to_string())
            ),
        ));
    }
    text
}

pub fn assets_csv_text(
    policy: &RetentionSettings,
    matcher: &FolderMatcher,
    results: &CleanupResults,
    components: &std::collections::HashMap<i64, AssetComponents>,
) -> String {
    let mut text = String::from(
        "group_table,group_id,group_kind,file_id,asset_id,relative_path,library_root,\
asset_type,media_type,media_role,width,height,file_size,capture_time,\
content_identifier,scan_state,readable,asset_file_count,asset_has_video,\
asset_has_sidecar,similarity,distance,is_recommended_keep,\
retention_score,matched_preferred_folder,matched_folder_priority\n",
    );
    for group in results
        .duplicate_groups
        .iter()
        .chain(&results.similarity_groups)
    {
        for member in &group.members {
            let candidate = candidate_from(member, components);
            let score = score_asset_for_retention(policy, matcher, &candidate);
            text.push_str(&format!(
                "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n",
                csv_field(&group.table_name),
                group.id,
                csv_field(&group.kind),
                member.file_id,
                member.asset_id,
                csv_field(&member.relative_path),
                csv_field(&member.library_root),
                csv_field(&member.asset_type),
                csv_field(&member.media_type),
                csv_field(&member.media_role),
                optional_number(member.width),
                optional_number(member.height),
                member.file_size,
                csv_field(member.capture_time.as_deref().unwrap_or("")),
                csv_field(member.content_identifier.as_deref().unwrap_or("")),
                csv_field(&member.scan_state),
                candidate.readable,
                // The shape of the asset, so a "complete Live Photo" claim in
                // groups.csv can be checked rather than taken on faith.
                candidate.components.file_count,
                candidate.components.has_video,
                candidate.components.has_sidecar,
                member
                    .similarity
                    .map(|value| format!("{value:.6}"))
                    .unwrap_or_default(),
                optional_number(member.distance),
                member.is_recommended_keep,
                score.display_score,
                csv_field(
                    score
                        .matched
                        .as_ref()
                        .map(|matched| matched.folder.as_str())
                        .unwrap_or("")
                ),
                score
                    .matched
                    .as_ref()
                    .map(|matched| matched.priority.to_string())
                    .unwrap_or_default(),
            ));
        }
    }
    text
}

/// Rebuilds the scoring input from a stored row.
///
/// Kept next to the exporter and identical in shape to the one the database
/// builds, so the numbers in `assets.csv` are the numbers that produced the
/// recommendation rather than a second, drifting implementation.
fn candidate_from(
    member: &CleanupAsset,
    components: &std::collections::HashMap<i64, AssetComponents>,
) -> RetentionCandidate {
    RetentionCandidate {
        file_id: member.file_id,
        asset_id: member.asset_id,
        library_root: member.library_root.clone(),
        relative_path: member.relative_path.clone(),
        asset_type: member.asset_type.clone(),
        width: member.width,
        height: member.height,
        file_size: member.file_size,
        capture_time: member.capture_time.clone(),
        content_identifier: member.content_identifier.clone(),
        components: components
            .get(&member.asset_id)
            .copied()
            .unwrap_or_default(),
        readable: !scan_state::is_failure(&member.scan_state),
    }
}

pub fn validation_report_text(policy: &RetentionSettings, results: &CleanupResults) -> String {
    let mut text = String::from("# VALIDATION_REPORT\n\n");
    text.push_str(&retention_report_section(policy, results));
    text.push_str(
        "\n## Not included here\n\n\
`candidate_pairs.csv`, precision and recall belong to the recognition validation \
mode, which this build does not have. They are deliberately absent rather than \
approximated: retention runs after recognition and cannot change a candidate, a \
distance, a cosine or a classification, so any number of that kind produced from \
this data would be measuring the detector through the wrong lens.\n",
    );
    text
}

/// The Retention Policy sections, ready to be appended to a shared report.
pub fn retention_report_section(policy: &RetentionSettings, results: &CleanupResults) -> String {
    let counts = count_outcomes(results);
    let mut text = String::new();

    text.push_str("## Retention Policy\n\n");
    text.push_str(&format!(
        "- Enabled: {}\n",
        if policy.enabled { "yes" } else { "no" }
    ));
    text.push_str(&format!(
        "- Prefer complete Live Photo: {}\n- Prefer higher resolution: {}\n\
- Prefer original format: {}\n- Prefer metadata complete: {}\n\
- Prefer larger original: {}\n\n",
        policy.prefer_complete_live_photo,
        policy.prefer_higher_resolution,
        policy.prefer_original_format,
        policy.prefer_metadata_complete,
        policy.prefer_larger_original
    ));

    text.push_str("### Preferred folders\n\n");
    if policy.preferred_folders.is_empty() {
        text.push_str("None configured.\n\n");
    } else {
        let mut folders = policy.preferred_folders.clone();
        // Shown in the order they actually take effect, not the order they
        // happen to sit in the file.
        folders.sort_by_key(|folder| folder.priority);
        for folder in folders {
            text.push_str(&format!(
                "- priority {}: {}\n",
                folder.priority, folder.path
            ));
        }
        text.push('\n');
    }

    text.push_str("## Retention outcomes\n\n");
    text.push_str(&format!(
        "- Groups decided by preferred folder: {}\n\
- Groups decided by quality fallback: {}\n\
- Groups decided by deterministic tiebreak: {}\n\
- Groups scored with the policy disabled: {}\n\
- Groups fixed by an explicit KEEP, retention not consulted: {}\n\
- Groups total: {}\n\n",
        counts.preferred_folder,
        counts.quality_fallback,
        counts.deterministic_tie_break,
        counts.policy_disabled,
        counts.not_decided_by_retention,
        counts.total()
    ));

    text.push_str("## Safety\n\n");
    text.push_str(
        "Retention only chooses which copy to recommend keeping. It never selects a \
file for deletion.\n\n\
- `EXACT_DUPLICATE`: byte-identical copies, so the non-recommended members stay \
eligible for the existing pre-selection.\n\
- `NEAR_DUPLICATE`, `BURST_SIMILAR`, `VISUALLY_SIMILAR`: similarity judgements, \
not proof that two files hold the same picture. Nothing in these groups is ever \
ticked on the user's behalf, whatever the folder priorities say.\n\n\
Live Photo pairing is decided during the scan from container metadata, with a \
filename fallback. Retention reads that pairing and never changes it, so a still \
in one folder is never joined to a movie in another because the first folder \
ranks higher.\n",
    );
    text
}

fn auto_preselect_allowed(kind: &str) -> bool {
    AUTO_PRESELECT_KINDS
        .iter()
        .any(|allowed| allowed.eq_ignore_ascii_case(kind))
}

fn optional_number(value: Option<i64>) -> String {
    value.map(|value| value.to_string()).unwrap_or_default()
}

/// Quotes a CSV field when it contains anything that would break a row.
///
/// Windows paths routinely contain commas and occasionally quotes, and a
/// spreadsheet that silently shifts every column after one such path is worse
/// than no export at all.
fn csv_field(value: &str) -> String {
    let needs_quotes = value.contains([',', '"', '\n', '\r']);
    if !needs_quotes {
        return value.to_string();
    }
    format!("\"{}\"", value.replace('"', "\"\""))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::PreferredFolder;
    use crate::retention::RetentionDecision;
    use std::collections::HashMap;

    fn member(file_id: i64, relative_path: &str, keep: bool) -> CleanupAsset {
        CleanupAsset {
            file_id,
            asset_id: file_id,
            library_root: "D:\\Photos".to_string(),
            relative_path: relative_path.to_string(),
            file_name: relative_path.to_string(),
            asset_type: "IMAGE".to_string(),
            media_type: "IMAGE".to_string(),
            media_role: "PRIMARY_IMAGE".to_string(),
            file_size: 2_048,
            width: Some(4_000),
            height: Some(3_000),
            duration_ms: None,
            capture_time: Some("2026-01-01T00:00:00+00:00".to_string()),
            similarity: Some(0.99),
            distance: Some(2),
            recommendation: None,
            is_recommended_keep: keep,
            content_identifier: None,
            scan_state: "SUCCESS".to_string(),
        }
    }

    fn policy() -> RetentionSettings {
        RetentionSettings {
            enabled: true,
            preferred_folders: vec![
                PreferredFolder {
                    path: "精选".to_string(),
                    priority: 1,
                },
                PreferredFolder {
                    path: "微信保存".to_string(),
                    priority: 4,
                },
            ],
            ..RetentionSettings::default()
        }
    }

    fn results_with(kind: &str) -> CleanupResults {
        let members = vec![
            member(1, "精选\\IMG_0001.HEIC", true),
            member(2, "微信保存\\IMG_0001.JPG", false),
        ];
        let settings = policy();
        let matcher = FolderMatcher::compile(&settings.preferred_folders);
        let components: HashMap<i64, AssetComponents> = HashMap::new();
        let candidates: Vec<RetentionCandidate> = members
            .iter()
            .map(|member| candidate_from(member, &components))
            .collect();
        let decision: RetentionDecision =
            super::super::choose_recommended_keep(&settings, &matcher, &candidates).unwrap();
        let group = CleanupGroup {
            id: 7,
            table_name: "duplicate_groups".to_string(),
            kind: kind.to_string(),
            created_at: "2026-01-01T00:00:00+00:00".to_string(),
            members,
            reclaim_bytes: 2_048,
            retention: Some(decision),
        };
        if kind == "EXACT_DUPLICATE" {
            CleanupResults {
                duplicate_groups: vec![group],
                similarity_groups: Vec::new(),
            }
        } else {
            CleanupResults {
                duplicate_groups: Vec::new(),
                similarity_groups: vec![group],
            }
        }
    }

    /// The export exists so a user can argue with a recommendation. If the
    /// reason column is empty it has failed at its only job.
    #[test]
    fn groups_csv_explains_why_the_keeper_was_chosen() {
        let text = groups_csv_text(&results_with("EXACT_DUPLICATE"));
        assert!(text.starts_with("group_table,group_id,group_kind"));
        assert!(text.contains("PREFERRED_FOLDER"));
        assert!(text.contains("精选"));
        assert!(text.contains("priority 1"));
    }

    #[test]
    fn assets_csv_carries_the_retention_columns_for_every_member() {
        let settings = policy();
        let matcher = FolderMatcher::compile(&settings.preferred_folders);
        let components = HashMap::new();
        let text = assets_csv_text(
            &settings,
            &matcher,
            &results_with("EXACT_DUPLICATE"),
            &components,
        );
        let lines: Vec<&str> = text.trim_end().lines().collect();
        assert_eq!(lines.len(), 3, "header plus both members");
        assert!(lines[0].contains("matched_preferred_folder"));
        assert!(lines[0].contains("retention_score"));
        assert!(lines[1].contains("精选"));
        assert!(lines[2].contains("微信保存"));
    }

    /// Only byte-identical groups may be pre-selected, and the export has to
    /// say so per row rather than leave it to be inferred.
    #[test]
    fn only_exact_duplicates_are_marked_auto_preselectable() {
        let exact = groups_csv_text(&results_with("EXACT_DUPLICATE"));
        assert!(exact.contains(",true,"), "EXACT should allow preselect");
        for kind in ["NEAR_DUPLICATE", "BURST_SIMILAR", "VISUALLY_SIMILAR"] {
            let text = groups_csv_text(&results_with(kind));
            assert!(
                text.contains(",false,"),
                "{kind} must not be auto-preselectable"
            );
        }
    }

    #[test]
    fn the_report_lists_the_folders_and_how_groups_were_decided() {
        let text = validation_report_text(&policy(), &results_with("EXACT_DUPLICATE"));
        assert!(text.contains("## Retention Policy"));
        assert!(text.contains("- Enabled: yes"));
        assert!(text.contains("priority 1: 精选"));
        assert!(text.contains("Groups decided by preferred folder: 1"));
        assert!(text.contains("Groups decided by quality fallback: 0"));
    }

    /// The report must not imply that a folder list changed the detector.
    #[test]
    fn the_report_says_retention_cannot_change_classification() {
        let text = validation_report_text(&policy(), &results_with("NEAR_DUPLICATE"));
        assert!(text.contains("never selects a file for deletion"));
        assert!(text.contains("cannot change a candidate"));
    }

    #[test]
    fn a_path_with_a_comma_does_not_shift_the_columns() {
        let mut results = results_with("EXACT_DUPLICATE");
        results.duplicate_groups[0].members[0].relative_path =
            "精选\\2026, 春节\\IMG_0001.HEIC".to_string();
        let settings = policy();
        let matcher = FolderMatcher::compile(&settings.preferred_folders);
        let components = HashMap::new();
        let text = assets_csv_text(&settings, &matcher, &results, &components);
        assert!(text.contains("\"精选\\2026, 春节\\IMG_0001.HEIC\""));
    }

    #[test]
    fn a_disabled_policy_reports_itself_as_disabled() {
        let empty = CleanupResults::default();
        let text = validation_report_text(&RetentionSettings::default(), &empty);
        assert!(text.contains("- Enabled: no"));
        assert!(text.contains("None configured."));
        assert!(text.contains("Groups total: 0"));
    }
}
