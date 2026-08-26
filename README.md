# PhotoCleaner

PhotoCleaner is a Windows local photo cleanup tool written in Rust. It scans a selected photo folder, stores local metadata in SQLite, and helps review duplicate or visually similar media without uploading user photos.

## Features

- Standard scan for media discovery, metadata, hashes, perceptual hashes, and local grouping inputs.
- Deep scan with offline DINOv2 ViT-S/14 ONNX inference through ONNX Runtime.
- 384-dimensional embeddings stored locally for similarity grouping.
- Duplicate and similar group browsing in the desktop UI.
- Large image preview, side-by-side comparison, manual selection, pending-delete staging, and undo for staged moves.
- Live Photo handling as one asset with all paired files kept together.
- Video discovery and filtering; videos can be opened from the local folder.
- Portable runtime layout using paths relative to `PhotoCleaner.exe`.

## Technology

- Rust 2021
- `eframe` / `egui` desktop UI
- SQLite through `rusqlite`
- ONNX Runtime loaded dynamically from `runtime/onnx/`
- DINOv2 ViT-S/14 model loaded from `models/dinov2_vits14.onnx`
- `ffmpeg` / `ffprobe` from `runtime/media/` for media probing and HEIC support

## Repository Layout

```text
src/                 Rust application source
models/              Offline AI model files
runtime/onnx/        ONNX Runtime DLLs
runtime/media/       ffmpeg and ffprobe runtime tools
packaging/           Portable package notes
scripts/             Build/package helper scripts
Cargo.toml           Rust package configuration
Cargo.lock           Locked Rust dependency graph
```

Generated folders such as `target/`, `dist/`, `outputs/`, `work/`, `data/`, `cache/`, and `logs/` are ignored.

## Running From Source

Install a Windows Rust toolchain, then run:

```powershell
cargo run --release
```

The app expects these local assets to exist when deep scan or media runtime features are used:

```text
models/dinov2_vits14.onnx
runtime/onnx/onnxruntime.dll
runtime/onnx/onnxruntime_providers_shared.dll
runtime/media/ffmpeg.exe
runtime/media/ffprobe.exe
```

## Building

```powershell
cargo test
cargo build --release
```

The current package script is:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/package.ps1
```

It creates a portable `dist/PhotoCleaner` folder from the release executable and runtime assets.

## Portable Layout

The portable package is expected to look like this:

```text
PhotoCleaner/
  PhotoCleaner.exe
  models/dinov2_vits14.onnx
  runtime/onnx/*.dll
  runtime/media/ffmpeg.exe
  runtime/media/ffprobe.exe
```

The program loads models and runtimes by relative path from the executable directory. It must not depend on Python, pip, conda, a system ONNX Runtime install, CUDA Toolkit, or development-machine paths.

## Scan Modes

STANDARD mode refreshes the selected folder and computes local metadata/hash features needed for duplicate review.

DEEP mode uses the same local scan data plus offline DINOv2 ONNX inference. CPU execution is sufficient for deep scan. CUDA is optional and should fall back to CPU when unavailable.

## Local Data

PhotoCleaner stores scan data in a local SQLite database under the portable data directory. User photos are not uploaded. Cleanup actions first move selected assets into a local pending-delete folder and record operations so the move can be undone.

## Large Files

The DINOv2 model, ONNX Runtime DLLs, ffmpeg tools, and packaged executables are large binary files. This repository tracks required source-side runtime assets with Git LFS.

## Known Limits

- CUDA availability depends on bundled runtime components and the local GPU/driver environment.
- HEIC preview depends on the bundled media runtime.
- The UI favors manual review. Recommendations are visual hints and do not delete files automatically.
