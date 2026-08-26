# PhotoCleaner Cache Behavior

## STANDARD -> STANDARD

Unchanged files reuse valid metadata, quick hash, SHA-256, pHash, video fingerprint, and Live Photo pairing artifacts. Standard grouping can rebuild if its grouping signature changes.

## STANDARD -> DEEP

Standard artifacts are reused when valid. AI embedding is planned separately. If no valid embedding exists for the current model id, model hash, preprocess version, dimension, and dtype, AI is planned as `COMPUTE`.

Current build note: real AI runtime/model is not packaged yet. When DEEP requires AI, the scan reports deep analysis unavailable instead of marking STANDARD cache as DEEP complete.

## DEEP -> STANDARD

Standard artifacts are reused. Existing AI embeddings are retained and are not deleted or recomputed.

## DEEP -> DEEP

If the file is unchanged and the embedding cache key matches, AI is `REUSE`. If the grouping threshold/signature changed, grouping rebuilds without rerunning AI inference.

## File Added

New files have no valid artifacts. STANDARD computes base artifacts. DEEP computes base artifacts and AI embedding when AI is available.

## File Modified

If size or modified time changed, metadata, quick hash, SHA-256, pHash, video fingerprint, AI embedding, and group membership are invalidated as required by the scan mode.

## Model Changed

If model id, model hash, preprocess version, embedding dimension, or dtype differs, existing embedding is `STALE` and DEEP plans AI recomputation.

## Threshold Changed

pHash and AI embeddings remain reusable. Candidate search/grouping is rebuilt using the new threshold.

## Interrupted Deep Scan

Committed embeddings remain valid per-file artifacts. A later DEEP scan reuses completed embeddings and computes only the missing or stale subset.
