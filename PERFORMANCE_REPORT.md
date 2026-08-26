# PhotoCleaner Performance Report

Date: 2026-07-26

## Baseline

The previous Phase A scanner was inspected before changes. It used a single logical scan path that discovered files, read file metadata, probed image dimensions, accumulated all records in memory, and then wrote them to SQLite. The UI only received start/finish events, so per-stage timing and the reported late-scan slowdown could not be measured directly from the old executable without manual GUI interaction.

Observed old-code risks:

- No per-stage instrumentation.
- No bounded pipeline queues.
- No incremental reuse before image probing.
- `processed` and `skipped` overlapped semantically.
- HEIC/HEIF files were accepted by extension but normal image probing returned no width/height.
- No media asset model for video or Live Photo.

## New Scanner

The scanner now uses:

- Discovery thread.
- Bounded metadata queue.
- Multiple media worker threads.
- Bounded database queue.
- Single SQLite writer using prepared statements and batch transactions of 250 rows.
- WAL mode with `synchronous=NORMAL`.
- Incremental reuse by `library_id + relative_path + file_size + modified_time`.
- Per-stage `[PERF]` logs.

## Synthetic Test A: 500 JPG

Fixture: 500 generated 64x64 JPEG files.

First scan:

- Total: 171 ms
- Discovered: 500
- Completed: 500
- New: 500
- Reused: 0
- Failed: 0

Second scan of unchanged directory:

- Total: 90 ms
- Discovered: 500
- Completed: 500
- New: 0
- Reused: 500
- Failed: 0

Representative stage timing from logs:

- DISCOVERING: 500 files, 0.14 s, 3587 files/s
- MEDIA_PROBE: 500 files, 0.14 s, 3558 files/s
- EXACT_HASH: 500 files, 0.14 s, 3554 files/s
- DATABASE_FINALIZE: 500 files, 0.14 s, DB write 17 ms

Second scan stage behavior:

- DISCOVERING: 500 files, 0.07 s
- MEDIA_PROBE: 0 files
- EXACT_HASH: 0 files
- DATABASE_FINALIZE: 0 files
- REUSED: 500 files

## Bottleneck Found

The original source made the main bottleneck invisible: there was no stage instrumentation. The biggest structural issue was not a specific slow function in the small fixture; it was the lack of incremental reuse and the all-record accumulation before database writing. The new run confirms unchanged files avoid media probe, full file reads, hashing, and DB rewrites.

## HEIC / HEIF

Implemented in this build:

- `.heic` and `.heif` are scanned as image media.
- HEIF container `ispe` dimensions are parsed when present.
- Decode failure is recorded as `DECODE_FAILED`, not as skipped.

Not completed in this build:

- Portable libheif runtime is not packaged yet.
- HEIC thumbnail generation and pHash are therefore not claimed as complete.

## Video

Implemented in this build:

- `.mov`, `.mp4`, `.m4v`, `.avi`, `.mkv`, `.webm` are scanned as video media.
- MP4/MOV-like metadata probe reads duration, dimensions, codec markers, and Apple content identifier where available.
- Exact duplicate fingerprint uses quick hash plus SHA-256.

Not completed in this build:

- Frame sampling pHash sequence for visual video similarity.

## Live Photo

Implemented in this build:

- Media asset schema supports `LIVE_PHOTO`.
- Pairing uses Apple content identifier when readable.
- Fallback pairs same-directory same-basename image/video as `PROBABLE_LIVE_PHOTO`.
- AAE is scanned as sidecar and not displayed as standalone media.

Not completed in this build:

- Confirm/unpair UI and whole-asset file operations are still pending because file operation pages are not implemented in the existing Phase A app.

## Tests

- Unit tests: 10 passed.
- Incremental reuse test: passed.
- Release build: passed.

## Cache Planner Validation

STANDARD -> DEEP was tested with 20 generated JPEG files:

- STANDARD: standard_compute=20, standard_reuse=0, ai_compute=0
- DEEP after STANDARD: standard_compute=0, standard_reuse=20, ai_compute=20, ai_reuse=0
- Because no real AI model/runtime is packaged, DEEP exits with "深度分析不可用" instead of falsely completing.
