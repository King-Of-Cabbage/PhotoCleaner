use serde::{Deserialize, Serialize};

pub const METADATA_VERSION: i64 = 1;
pub const QUICK_HASH_VERSION: i64 = 1;
pub const SHA256_VERSION: i64 = 1;
pub const PHASH_VERSION: i64 = 1;
pub const VIDEO_FINGERPRINT_VERSION: i64 = 1;
pub const EMBEDDING_MODEL_ID: &str = "dinov2_vits14";
pub const EMBEDDING_PREPROCESS_VERSION: i64 = 1;
pub const EMBEDDING_DIMENSION: i64 = 384;
pub const EMBEDDING_DTYPE: &str = "float16";

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ScanModeKind {
    Standard,
    Deep,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum WorkDecision {
    Compute,
    Reuse,
    NotRequired,
    Stale,
    Failed,
}

#[derive(Clone, Debug, Default)]
pub struct ArtifactState {
    pub file_unchanged: bool,
    pub metadata_valid: bool,
    pub quick_hash_valid: bool,
    pub sha256_valid: bool,
    pub phash_valid: bool,
    pub video_fingerprint_valid: bool,
    pub embedding_valid: bool,
    pub embedding_present: bool,
    pub embedding_model_id: Option<String>,
    pub embedding_model_hash: Option<String>,
    pub embedding_preprocess_version: Option<i64>,
    pub embedding_dimension: Option<i64>,
    pub embedding_dtype: Option<String>,
    pub grouping_signature: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AnalysisPlan {
    pub metadata: WorkDecision,
    pub quick_hash: WorkDecision,
    pub sha256: WorkDecision,
    pub phash: WorkDecision,
    pub video_fingerprint: WorkDecision,
    pub ai_embedding: WorkDecision,
    pub ann_index: WorkDecision,
    pub grouping_rebuild: bool,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ScanPlanSummary {
    pub assets: usize,
    pub standard_compute: usize,
    pub standard_reuse: usize,
    pub ai_compute: usize,
    pub ai_reuse: usize,
    pub ai_stale: usize,
    pub grouping_rebuild: bool,
}

#[derive(Clone, Debug)]
pub struct PlannerContext {
    pub requested_mode: ScanModeKind,
    pub is_video: bool,
    pub model_hash: Option<String>,
    pub grouping_signature: String,
}

pub fn plan_for_file(state: &ArtifactState, ctx: &PlannerContext) -> AnalysisPlan {
    let file_changed = !state.file_unchanged;
    let metadata = basic_decision(file_changed, state.metadata_valid);
    let quick_hash = basic_decision(file_changed, state.quick_hash_valid);
    let sha256 = basic_decision(file_changed, state.sha256_valid);
    let phash = if ctx.is_video {
        WorkDecision::NotRequired
    } else {
        basic_decision(file_changed, state.phash_valid)
    };
    let video_fingerprint = if ctx.is_video {
        basic_decision(file_changed, state.video_fingerprint_valid)
    } else {
        WorkDecision::NotRequired
    };

    let ai_embedding = if ctx.requested_mode == ScanModeKind::Standard {
        WorkDecision::NotRequired
    } else if file_changed {
        WorkDecision::Compute
    } else if embedding_matches(state, ctx.model_hash.as_deref()) {
        WorkDecision::Reuse
    } else if state.embedding_present {
        WorkDecision::Stale
    } else {
        WorkDecision::Compute
    };

    let ann_index = if ctx.requested_mode == ScanModeKind::Deep {
        match ai_embedding {
            WorkDecision::Reuse | WorkDecision::Compute | WorkDecision::Stale => {
                WorkDecision::Compute
            }
            _ => WorkDecision::NotRequired,
        }
    } else {
        WorkDecision::NotRequired
    };

    let grouping_rebuild = if ctx.requested_mode == ScanModeKind::Deep {
        state.grouping_signature.as_deref() != Some(ctx.grouping_signature.as_str())
            || matches!(ai_embedding, WorkDecision::Compute | WorkDecision::Stale)
    } else {
        state.grouping_signature.as_deref() != Some(ctx.grouping_signature.as_str())
    };

    AnalysisPlan {
        metadata,
        quick_hash,
        sha256,
        phash,
        video_fingerprint,
        ai_embedding,
        ann_index,
        grouping_rebuild,
    }
}

pub fn add_to_summary(summary: &mut ScanPlanSummary, plan: &AnalysisPlan) {
    summary.assets += 1;
    let standard_compute = [
        plan.metadata,
        plan.quick_hash,
        plan.sha256,
        plan.phash,
        plan.video_fingerprint,
    ]
    .iter()
    .any(|decision| *decision == WorkDecision::Compute || *decision == WorkDecision::Stale);
    if standard_compute {
        summary.standard_compute += 1;
    } else {
        summary.standard_reuse += 1;
    }
    match plan.ai_embedding {
        WorkDecision::Compute => summary.ai_compute += 1,
        WorkDecision::Reuse => summary.ai_reuse += 1,
        WorkDecision::Stale => {
            summary.ai_stale += 1;
            summary.ai_compute += 1;
        }
        _ => {}
    }
    summary.grouping_rebuild |= plan.grouping_rebuild;
}

fn basic_decision(file_changed: bool, valid: bool) -> WorkDecision {
    if file_changed {
        WorkDecision::Compute
    } else if valid {
        WorkDecision::Reuse
    } else {
        WorkDecision::Compute
    }
}

fn embedding_matches(state: &ArtifactState, model_hash: Option<&str>) -> bool {
    state.embedding_valid
        && state.embedding_present
        && state.embedding_model_id.as_deref() == Some(EMBEDDING_MODEL_ID)
        && state.embedding_model_hash.as_deref() == model_hash
        && state.embedding_preprocess_version == Some(EMBEDDING_PREPROCESS_VERSION)
        && state.embedding_dimension == Some(EMBEDDING_DIMENSION)
        && state.embedding_dtype.as_deref() == Some(EMBEDDING_DTYPE)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx(mode: ScanModeKind, hash: &str) -> PlannerContext {
        PlannerContext {
            requested_mode: mode,
            is_video: false,
            model_hash: Some(hash.to_string()),
            grouping_signature: "deep:0.92:v1".to_string(),
        }
    }

    fn standard_state() -> ArtifactState {
        ArtifactState {
            file_unchanged: true,
            metadata_valid: true,
            quick_hash_valid: true,
            sha256_valid: true,
            phash_valid: true,
            ..Default::default()
        }
    }

    #[test]
    fn standard_then_deep_reuses_standard_but_computes_ai() {
        let plan = plan_for_file(&standard_state(), &ctx(ScanModeKind::Deep, "abc"));
        assert_eq!(plan.metadata, WorkDecision::Reuse);
        assert_eq!(plan.quick_hash, WorkDecision::Reuse);
        assert_eq!(plan.phash, WorkDecision::Reuse);
        assert_eq!(plan.ai_embedding, WorkDecision::Compute);
        assert!(plan.grouping_rebuild);
    }

    #[test]
    fn deep_then_deep_reuses_matching_embedding() {
        let mut state = standard_state();
        state.embedding_present = true;
        state.embedding_valid = true;
        state.embedding_model_id = Some(EMBEDDING_MODEL_ID.to_string());
        state.embedding_model_hash = Some("abc".to_string());
        state.embedding_preprocess_version = Some(EMBEDDING_PREPROCESS_VERSION);
        state.embedding_dimension = Some(EMBEDDING_DIMENSION);
        state.embedding_dtype = Some(EMBEDDING_DTYPE.to_string());
        state.grouping_signature = Some("deep:0.92:v1".to_string());
        let plan = plan_for_file(&state, &ctx(ScanModeKind::Deep, "abc"));
        assert_eq!(plan.ai_embedding, WorkDecision::Reuse);
        assert!(!plan.grouping_rebuild);
    }

    #[test]
    fn model_hash_change_marks_embedding_stale() {
        let mut state = standard_state();
        state.embedding_present = true;
        state.embedding_valid = true;
        state.embedding_model_id = Some(EMBEDDING_MODEL_ID.to_string());
        state.embedding_model_hash = Some("old".to_string());
        state.embedding_preprocess_version = Some(EMBEDDING_PREPROCESS_VERSION);
        state.embedding_dimension = Some(EMBEDDING_DIMENSION);
        state.embedding_dtype = Some(EMBEDDING_DTYPE.to_string());
        let plan = plan_for_file(&state, &ctx(ScanModeKind::Deep, "new"));
        assert_eq!(plan.ai_embedding, WorkDecision::Stale);
        assert!(plan.grouping_rebuild);
    }

    #[test]
    fn threshold_change_rebuilds_grouping_without_embedding_compute() {
        let mut state = standard_state();
        state.embedding_present = true;
        state.embedding_valid = true;
        state.embedding_model_id = Some(EMBEDDING_MODEL_ID.to_string());
        state.embedding_model_hash = Some("abc".to_string());
        state.embedding_preprocess_version = Some(EMBEDDING_PREPROCESS_VERSION);
        state.embedding_dimension = Some(EMBEDDING_DIMENSION);
        state.embedding_dtype = Some(EMBEDDING_DTYPE.to_string());
        state.grouping_signature = Some("deep:0.88:v1".to_string());
        let plan = plan_for_file(&state, &ctx(ScanModeKind::Deep, "abc"));
        assert_eq!(plan.ai_embedding, WorkDecision::Reuse);
        assert!(plan.grouping_rebuild);
    }

    #[test]
    fn changed_file_invalidates_all_artifacts() {
        let mut state = standard_state();
        state.file_unchanged = false;
        state.embedding_present = true;
        state.embedding_valid = true;
        let plan = plan_for_file(&state, &ctx(ScanModeKind::Deep, "abc"));
        assert_eq!(plan.metadata, WorkDecision::Compute);
        assert_eq!(plan.quick_hash, WorkDecision::Compute);
        assert_eq!(plan.ai_embedding, WorkDecision::Compute);
    }
}
