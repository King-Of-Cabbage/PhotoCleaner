# PhotoCleaner Cache Behavior

## 实现状态

**Implemented.** 增量复用在 `scanner::artifact_state_for` 中生效：按 `relative_path` 找到数据库快照，比对 `file_size` 与 `modified_time` 得出 `file_unchanged`，其余有效位（metadata / quick hash / SHA-256 / pHash / video fingerprint / embedding + 模型身份）直接来自 `media_files` 的版本列。

复用的产物会被回填进写库记录，因此部分复用不会把未重算的列覆盖成 NULL。全部复用的文件根本不进数据库写入队列。

STANDARD 扫描不读取 embedding BLOB，只读 `embedding IS NOT NULL`；只有 DEEP 才会把 embedding 载入内存。

在此之前 `scan_candidate` 把 `file_unchanged` 硬编码为 `false` 并丢弃快照参数，下面描述的所有复用路径实际上一条都没有生效过。

## STANDARD -> STANDARD

Unchanged files reuse valid metadata, quick hash, SHA-256, pHash, video fingerprint, and Live Photo pairing artifacts. Standard grouping can rebuild if its grouping signature changes.

## STANDARD -> DEEP

Standard artifacts are reused when valid. AI embedding is planned separately. If no valid embedding exists for the current model id, model hash, preprocess version, dimension, and dtype, AI is planned as `COMPUTE`.

Current build note: in a plain `cargo build` the model and ONNX Runtime live beside the packaged exe, not beside `target/release/PhotoCleaner.exe`, so DEEP reports deep analysis unavailable rather than marking a STANDARD cache as DEEP complete. `scripts/package.ps1` produces the layout where DEEP can actually run.

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
