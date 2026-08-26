//! Background thumbnail pipeline.
//!
//! The UI thread must never decode an image. It calls
//! [`ThumbnailService::request`], which returns immediately: either the work is
//! queued on a background decoder, or the queue is saturated and the caller
//! draws a placeholder for this frame. Finished thumbnails come back through
//! [`ThumbnailService::poll`], and only then does the UI turn the RGB buffer
//! into a texture.
//!
//! Decoded thumbnails are also written to `cache/thumbnails/` as JPEG, so a
//! later session never decodes the same original twice. Both the on-disk cache
//! and the in-memory texture budget are bounded by
//! `Settings::thumbnail_cache_limit_mb`.

use std::collections::{HashMap, HashSet, VecDeque};
use std::fs;
use std::hash::Hash;
use std::path::{Path, PathBuf};
use std::thread::{self, JoinHandle};

use crossbeam_channel::{bounded, unbounded, Receiver, Sender, TrySendError};
use image::imageops::FilterType;
use image::DynamicImage;
use xxhash_rust::xxh3::xxh3_64;

use crate::paths::PortablePaths;

/// How many pending decode requests are allowed to pile up. When the queue is
/// full new requests are dropped rather than queued, so a user scrolling fast
/// through a large library cannot build an unbounded backlog.
pub const REQUEST_QUEUE_LIMIT: usize = 256;

/// ffmpeg is a separate process and far more expensive than an in-process JPEG
/// decode, so HEIC work gets its own, deliberately narrower lane.
const HEIC_WORKERS: usize = 1;

/// Colour used for the letterbox bars around a non-square thumbnail.
const LETTERBOX_FILL: u8 = 24;

#[derive(Clone, Debug)]
struct ThumbnailJob {
    key: String,
    path: PathBuf,
    size: u32,
    cache_file: PathBuf,
}

/// A finished decode handed back to the UI thread.
#[derive(Debug)]
pub struct ThumbnailReady {
    pub key: String,
    pub size: u32,
    pub result: Result<Vec<u8>, String>,
}

/// What happened to a [`ThumbnailService::request`] call.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RequestOutcome {
    /// Handed to a decoder thread.
    Queued,
    /// Already being decoded; nothing further to do.
    AlreadyPending,
    /// Queue saturated. The caller should draw a placeholder and try again on a
    /// later frame.
    QueueFull,
    /// The decoder threads are gone.
    Stopped,
}

pub struct ThumbnailService {
    general_tx: Option<Sender<ThumbnailJob>>,
    heic_tx: Option<Sender<ThumbnailJob>>,
    ready_rx: Receiver<ThumbnailReady>,
    pending: HashSet<String>,
    cache_dir: PathBuf,
    workers: Vec<JoinHandle<()>>,
}

impl ThumbnailService {
    /// Starts the decoder threads and schedules a one-off prune of the on-disk
    /// cache. `disk_limit_bytes` comes from `thumbnail_cache_limit_mb`.
    pub fn start(paths: &PortablePaths, disk_limit_bytes: u64) -> Self {
        let cache_dir = paths.thumbnails_dir.clone();
        let _ = fs::create_dir_all(&cache_dir);

        let (general_tx, general_rx) = bounded::<ThumbnailJob>(REQUEST_QUEUE_LIMIT);
        let (heic_tx, heic_rx) = bounded::<ThumbnailJob>(REQUEST_QUEUE_LIMIT);
        let (ready_tx, ready_rx) = unbounded::<ThumbnailReady>();

        let mut workers = Vec::new();
        for _ in 0..general_worker_count() {
            let rx = general_rx.clone();
            let tx = ready_tx.clone();
            let paths = paths.clone();
            workers.push(thread::spawn(move || worker_loop(paths, rx, tx, false)));
        }
        for _ in 0..HEIC_WORKERS {
            let rx = heic_rx.clone();
            let tx = ready_tx.clone();
            let paths = paths.clone();
            workers.push(thread::spawn(move || worker_loop(paths, rx, tx, true)));
        }

        {
            let dir = cache_dir.clone();
            thread::spawn(move || {
                if let Err(err) = prune_disk_cache(&dir, disk_limit_bytes) {
                    crate::logging::error(format!("thumbnail cache prune failed: {err}"));
                }
            });
        }

        Self {
            general_tx: Some(general_tx),
            heic_tx: Some(heic_tx),
            ready_rx,
            pending: HashSet::new(),
            cache_dir,
            workers,
        }
    }

    /// Queues a decode. Never blocks and never decodes on the calling thread.
    pub fn request(&mut self, key: &str, path: &Path, size: u32) -> RequestOutcome {
        if self.pending.contains(key) {
            return RequestOutcome::AlreadyPending;
        }
        let heic = is_heic_path(path);
        let job = ThumbnailJob {
            key: key.to_string(),
            path: path.to_path_buf(),
            size,
            cache_file: self.cache_file_for(path, size),
        };
        let sender = if heic {
            self.heic_tx.as_ref()
        } else {
            self.general_tx.as_ref()
        };
        let Some(sender) = sender else {
            return RequestOutcome::Stopped;
        };
        match sender.try_send(job) {
            Ok(()) => {
                self.pending.insert(key.to_string());
                RequestOutcome::Queued
            }
            Err(TrySendError::Full(_)) => RequestOutcome::QueueFull,
            Err(TrySendError::Disconnected(_)) => RequestOutcome::Stopped,
        }
    }

    /// Collects whatever finished since the last frame.
    pub fn poll(&mut self) -> Vec<ThumbnailReady> {
        let ready: Vec<_> = self.ready_rx.try_iter().collect();
        for item in &ready {
            self.pending.remove(&item.key);
        }
        ready
    }

    pub fn pending_len(&self) -> usize {
        self.pending.len()
    }

    fn cache_file_for(&self, path: &Path, size: u32) -> PathBuf {
        let stamp = fs::metadata(path)
            .ok()
            .map(|meta| {
                let modified = meta
                    .modified()
                    .ok()
                    .and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok())
                    .map(|d| d.as_secs())
                    .unwrap_or(0);
                format!("{}:{}", meta.len(), modified)
            })
            .unwrap_or_default();
        let digest = xxh3_64(format!("{}|{}|{}", path.display(), stamp, size).as_bytes());
        self.cache_dir.join(format!("{digest:016x}_{size}.jpg"))
    }
}

impl Drop for ThumbnailService {
    fn drop(&mut self) {
        self.general_tx.take();
        self.heic_tx.take();
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }
}

fn general_worker_count() -> usize {
    num_cpus::get_physical().clamp(2, 4)
}

fn is_heic_path(path: &Path) -> bool {
    matches!(
        path.extension()
            .and_then(|value| value.to_str())
            .unwrap_or_default()
            .to_ascii_lowercase()
            .as_str(),
        "heic" | "heif"
    )
}

fn worker_loop(
    paths: PortablePaths,
    rx: Receiver<ThumbnailJob>,
    tx: Sender<ThumbnailReady>,
    heic: bool,
) {
    while let Ok(job) = rx.recv() {
        let result = decode_job(&paths, &job, heic);
        let ready = ThumbnailReady {
            key: job.key,
            size: job.size,
            result,
        };
        if tx.send(ready).is_err() {
            return;
        }
    }
}

fn decode_job(paths: &PortablePaths, job: &ThumbnailJob, heic: bool) -> Result<Vec<u8>, String> {
    if let Some(cached) = read_cached(&job.cache_file, job.size) {
        return Ok(cached);
    }
    let image = if heic {
        let png = crate::embedding::decode_heif_png_bytes(paths, &job.path)
            .map_err(|err| format!("{err:#}"))?;
        image::load_from_memory(&png).map_err(|err| err.to_string())?
    } else {
        image::open(&job.path).map_err(|err| err.to_string())?
    };
    let rgb = letterbox_square(&image, job.size);
    write_cached(&job.cache_file, &rgb, job.size);
    Ok(rgb)
}

fn read_cached(cache_file: &Path, size: u32) -> Option<Vec<u8>> {
    let bytes = fs::read(cache_file).ok()?;
    let image = image::load_from_memory(&bytes).ok()?;
    if image.width() != size || image.height() != size {
        return None;
    }
    Some(image.to_rgb8().into_raw())
}

fn write_cached(cache_file: &Path, rgb: &[u8], size: u32) {
    if rgb.len() != (size as usize) * (size as usize) * 3 {
        return;
    }
    let Some(buffer) = image::RgbImage::from_raw(size, size, rgb.to_vec()) else {
        return;
    };
    if let Some(parent) = cache_file.parent() {
        let _ = fs::create_dir_all(parent);
    }
    let _ = buffer.save(cache_file);
}

/// Scales `image` to fit inside `size` x `size` without distorting it, then
/// centres it on a square canvas. The old UI used `resize_exact`, which
/// stretched every non-square photo.
pub fn letterbox_square(image: &DynamicImage, size: u32) -> Vec<u8> {
    let size = size.max(1);
    let mut canvas = vec![LETTERBOX_FILL; (size as usize) * (size as usize) * 3];
    let scaled = image.resize(size, size, FilterType::Triangle).to_rgb8();
    let (width, height) = scaled.dimensions();
    if width == 0 || height == 0 {
        return canvas;
    }
    let offset_x = (size.saturating_sub(width) / 2) as usize;
    let offset_y = (size.saturating_sub(height) / 2) as usize;
    let row_bytes = (width as usize) * 3;
    let source = scaled.as_raw();
    for row in 0..height as usize {
        let destination = ((row + offset_y) * size as usize + offset_x) * 3;
        let start = row * row_bytes;
        if destination + row_bytes > canvas.len() || start + row_bytes > source.len() {
            break;
        }
        canvas[destination..destination + row_bytes]
            .copy_from_slice(&source[start..start + row_bytes]);
    }
    canvas
}

/// Deletes the oldest cached thumbnails until the directory fits in
/// `limit_bytes`. Returns the number of bytes kept.
pub fn prune_disk_cache(dir: &Path, limit_bytes: u64) -> std::io::Result<u64> {
    if !dir.exists() {
        return Ok(0);
    }
    let mut entries: Vec<(PathBuf, u64, std::time::SystemTime)> = Vec::new();
    let mut total = 0u64;
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let Ok(meta) = entry.metadata() else {
            continue;
        };
        if !meta.is_file() {
            continue;
        }
        let modified = meta.modified().unwrap_or(std::time::UNIX_EPOCH);
        total += meta.len();
        entries.push((entry.path(), meta.len(), modified));
    }
    if total <= limit_bytes {
        return Ok(total);
    }
    entries.sort_by_key(|(_, _, modified)| *modified);
    for (path, len, _) in entries {
        if total <= limit_bytes {
            break;
        }
        if fs::remove_file(&path).is_ok() {
            total = total.saturating_sub(len);
        }
    }
    Ok(total)
}

/// A least-recently-used budget measured in bytes.
///
/// This holds only keys and sizes, never the values themselves, so the caller
/// (which owns GPU texture handles) stays in control of the actual memory. It
/// is what finally makes `thumbnail_cache_limit_mb` mean something.
#[derive(Debug)]
pub struct LruBudget<K: Eq + Hash + Clone> {
    limit_bytes: u64,
    used_bytes: u64,
    order: VecDeque<K>,
    sizes: HashMap<K, u64>,
}

impl<K: Eq + Hash + Clone> LruBudget<K> {
    pub fn new(limit_bytes: u64) -> Self {
        Self {
            limit_bytes: limit_bytes.max(1),
            used_bytes: 0,
            order: VecDeque::new(),
            sizes: HashMap::new(),
        }
    }

    pub fn used_bytes(&self) -> u64 {
        self.used_bytes
    }

    pub fn len(&self) -> usize {
        self.sizes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.sizes.is_empty()
    }

    pub fn contains(&self, key: &K) -> bool {
        self.sizes.contains_key(key)
    }

    /// Marks `key` as most recently used.
    pub fn touch(&mut self, key: &K) {
        if !self.sizes.contains_key(key) {
            return;
        }
        if let Some(position) = self.order.iter().position(|item| item == key) {
            let entry = self.order.remove(position);
            if let Some(entry) = entry {
                self.order.push_back(entry);
            }
        }
    }

    /// Records `key` at `bytes` and returns whatever had to be evicted to stay
    /// under the limit. The key just inserted is never evicted.
    pub fn insert(&mut self, key: K, bytes: u64) -> Vec<K> {
        self.remove(&key);
        self.sizes.insert(key.clone(), bytes);
        self.order.push_back(key);
        self.used_bytes = self.used_bytes.saturating_add(bytes);

        let mut evicted = Vec::new();
        while self.used_bytes > self.limit_bytes && self.order.len() > 1 {
            let Some(oldest) = self.order.pop_front() else {
                break;
            };
            if let Some(size) = self.sizes.remove(&oldest) {
                self.used_bytes = self.used_bytes.saturating_sub(size);
            }
            evicted.push(oldest);
        }
        evicted
    }

    pub fn remove(&mut self, key: &K) {
        if let Some(size) = self.sizes.remove(key) {
            self.used_bytes = self.used_bytes.saturating_sub(size);
        }
        if let Some(position) = self.order.iter().position(|item| item == key) {
            let _ = self.order.remove(position);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{ImageBuffer, Rgb};

    #[test]
    fn lru_budget_evicts_oldest_first() {
        let mut budget = LruBudget::new(300);
        assert!(budget.insert("a", 100).is_empty());
        assert!(budget.insert("b", 100).is_empty());
        assert!(budget.insert("c", 100).is_empty());
        assert_eq!(budget.used_bytes(), 300);

        let evicted = budget.insert("d", 100);
        assert_eq!(evicted, vec!["a"]);
        assert_eq!(budget.used_bytes(), 300);
        assert!(!budget.contains(&"a"));
        assert!(budget.contains(&"d"));
    }

    #[test]
    fn lru_budget_touch_protects_recently_used() {
        let mut budget = LruBudget::new(200);
        budget.insert("a", 100);
        budget.insert("b", 100);
        budget.touch(&"a");

        let evicted = budget.insert("c", 100);
        assert_eq!(evicted, vec!["b"]);
        assert!(budget.contains(&"a"));
    }

    #[test]
    fn lru_budget_never_evicts_the_entry_just_inserted() {
        let mut budget = LruBudget::new(10);
        budget.insert("a", 5);
        let evicted = budget.insert("huge", 1_000);
        assert_eq!(evicted, vec!["a"]);
        assert!(budget.contains(&"huge"));
        assert_eq!(budget.len(), 1);
    }

    #[test]
    fn lru_budget_reinsert_does_not_double_count() {
        let mut budget = LruBudget::new(1_000);
        budget.insert("a", 100);
        budget.insert("a", 250);
        assert_eq!(budget.used_bytes(), 250);
        assert_eq!(budget.len(), 1);
    }

    #[test]
    fn letterbox_keeps_aspect_ratio_and_pads() {
        // 40x20 image: at size 20 it becomes 20x10 with 5 rows of padding above
        // and below, instead of being stretched to a 20x20 square.
        let wide = ImageBuffer::<Rgb<u8>, Vec<u8>>::from_pixel(40, 20, Rgb([200, 100, 50]));
        let rgb = letterbox_square(&DynamicImage::ImageRgb8(wide), 20);
        assert_eq!(rgb.len(), 20 * 20 * 3);

        // The top rows are letterbox bars, not stretched image data.
        assert_eq!(
            &rgb[0..3],
            [LETTERBOX_FILL, LETTERBOX_FILL, LETTERBOX_FILL].as_slice()
        );
        // The middle of the canvas carries the image itself.
        let centre = (10 * 20 + 10) * 3;
        for (actual, expected) in rgb[centre..centre + 3].iter().zip([200u8, 100, 50]) {
            let delta = (*actual as i32 - expected as i32).abs();
            assert!(delta <= 2, "expected ~{expected}, got {actual}");
        }
    }

    #[test]
    fn prune_disk_cache_keeps_newest_within_limit() {
        let dir = tempfile::tempdir().unwrap();
        for idx in 0..5 {
            let path = dir.path().join(format!("thumb_{idx}.jpg"));
            fs::write(&path, vec![0u8; 100]).unwrap();
            // Make modification times strictly ordered.
            std::thread::sleep(std::time::Duration::from_millis(15));
        }
        let kept = prune_disk_cache(dir.path(), 250).unwrap();
        assert!(kept <= 250, "kept {kept} bytes, limit was 250");
        let remaining = fs::read_dir(dir.path()).unwrap().count();
        assert_eq!(remaining, 2);
        assert!(dir.path().join("thumb_4.jpg").exists());
        assert!(!dir.path().join("thumb_0.jpg").exists());
    }

    #[test]
    fn prune_disk_cache_is_a_noop_under_the_limit() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("a.jpg"), vec![0u8; 10]).unwrap();
        let kept = prune_disk_cache(dir.path(), 1_000).unwrap();
        assert_eq!(kept, 10);
        assert_eq!(fs::read_dir(dir.path()).unwrap().count(), 1);
    }
}
