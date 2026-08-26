# PhotoCleaner Architecture

PhotoCleaner 是 Windows x64 portable 本地照片查重与相似照片整理工具。发布目录以 `PhotoCleaner.exe` 所在目录作为唯一程序根目录，所有配置、数据库、缓存、模型、运行库和日志都从该目录动态解析。

## 模块关系

- `app`: eframe/egui 应用状态、后台任务消息、页面切换。
- `config`: `config/settings.json` 的读取、默认值、保存。
- `database`: SQLite 初始化、WAL、`PRAGMA quick_check`、schema 与数据访问。
- `scanner`: 照片库递归发现、跳过规则、增量扫描入口。
- `metadata`: 文件元数据、基础图片尺寸与 EXIF 读取。
- `hashing`: size 分组、XXH3 quick fingerprint、SHA-256。
- `phash`: 64-bit pHash、16-bit block 倒排候选索引。
- `embedding`: DINOv2 ONNX embedding、Float16 BLOB 存储。
- `ann`: HNSW cosine ANN 索引。
- `grouping`: **Planned**（空壳）。分组目前实现在 `database::rebuild_recognition_groups` 里。分类落表规则：`EXACT_DUPLICATE` → `duplicate_groups`（唯一允许默认预选删除的表）；`NEAR_DUPLICATE` / `BURST_SIMILAR` / `VISUALLY_SIMILAR` → `similarity_groups`，一律不预选。
- `thumbnails`: **Implemented** — 后台缩略图解码。有界请求队列（256）、2~4 个通用解码线程 + 1 个 HEIC/ffmpeg 线程、`cache/thumbnails/` JPEG 磁盘缓存、按字节计量的 `LruBudget`。UI 线程只做「查缓存 / 投递请求 / 画占位符」，不做任何解码。磁盘缓存与内存 texture 预算都受 `thumbnail_cache_limit_mb` 约束。缩略图保持原始宽高比并补黑边，不再拉伸。
- `quality`: 清晰度、曝光、压缩近似指标与推荐保留标记。
- `file_ops`: 移动、加入待删除、撤销、永久删除二次确认。
- `hardware`: CPU/GPU 检测、ONNX Runtime provider 选择。
- `tasks`: 后台扫描、暂停、继续、取消、恢复。
- `logging`: `logs/` 滚动日志。
- `ui`: 中文界面组件、结果页、比较页、待删除页、设置页。

## SQLite Schema

数据库位于 `data/photos.db`，启用 WAL。Phase A 创建稳定基础 schema，后续阶段只做迁移追加。

- `libraries(id, display_name, last_known_root, volume_label, volume_serial, created_at, updated_at)`
- `photos(id, library_id, relative_path, file_name, extension, file_size, created_time, modified_time, exif_time, width, height, camera_model, quick_hash, sha256, phash, embedding, scan_state, missing, created_at, updated_at)`
- `duplicate_groups(id, group_kind, representative_photo_id, created_at, updated_at)`
- `similarity_groups(id, level, representative_photo_id, created_at, updated_at)`
- `group_members(id, group_id, group_table, photo_id, similarity, distance, recommendation, user_state, created_at)`
- `operations(id, photo_id, operation_type, source_path, destination_path, timestamp, undone)`
- `settings(key, value, updated_at)`
- `scan_runs(id, library_id, mode, state, discovered, processed, skipped, errors, phase, started_at, finished_at)`

核心索引覆盖 `library_id + relative_path`、`file_size`、`modified_time`、`sha256`、`phash`。

## 扫描状态机

固定阶段：

`DISCOVERING -> METADATA -> EXACT_HASH -> PHASH -> AI_EMBEDDING -> INDEXING -> GROUPING -> DONE`

标准扫描在 Phase A-C 使用 `DISCOVERING/METADATA/EXACT_HASH/PHASH/GROUPING/DONE`，不运行 AI。深度扫描在 Phase G 后加入 `AI_EMBEDDING/INDEXING`。任务状态持久化到 `scan_runs`，中断后从已提交记录继续。

## 线程模型

UI 线程只负责绘制和接收消息。目录扫描、hash、图片解码、AI inference、缩略图生成和数据库批量写入全部在后台线程执行。

- UI 与后台任务用 `crossbeam-channel` 通信。
- CPU 密集工作使用 rayon 线程池，默认 `logical_cpu_count - 1`，至少为 1。
- 数据库写入使用批量事务，避免每张照片独立提交。
- 每个文件错误记录到日志和扫描统计，不终止整次扫描。

## Portable 路径模型

启动时通过当前 exe 位置解析：

- `config/settings.json`
- `data/photos.db`
- `data/indexes/`
- `data/operations/`
- `cache/thumbnails/`
- `models/dinov2_vits14.onnx`
- `runtime/onnx/`
- `codecs/`
- `logs/`

程序不会默认写入 `C:\Program Files`、Registry 或 AppData。如果程序目录不可写，UI 显示：`当前程序目录不可写，Portable 模式无法保存数据库和配置，请将 PhotoCleaner 文件夹移动到可写目录。`

照片库以 `library_id + relative_path` 定位。换电脑或盘符变化时，只更新 `libraries.last_known_root`，不重建照片记录。

## ONNX Runtime 加载方式

ONNX Runtime 动态库从 `runtime/onnx/` 加载。模型固定为 `models/dinov2_vits14.onnx`，输入 224x224 RGB，按 DINOv2 normalization 预处理，取 CLS token 384 维 embedding，L2 normalize 后以 Float16 BLOB 存入 SQLite。

## CUDA Fallback 流程

启动深度扫描时：

1. 加载 `runtime/onnx/` 中的 GPU-enabled ONNX Runtime。
2. 检测 NVIDIA GPU。
3. 尝试初始化 CUDA Execution Provider。
4. 成功则显示 CUDA / GPU 名称。
5. 失败则写日志并自动使用 CPU Execution Provider。
6. CUDA batch 从 32 开始，OOM 时降为 16、8、4、1。
7. batch=1 仍失败时切换 CPU，继续任务。

CUDA 初始化失败、CUDA OOM、模型缺失或 runtime 缺失都不能导致程序无法启动。
