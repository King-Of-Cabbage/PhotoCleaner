use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct StagePerf {
    pub stage: String,
    pub files: usize,
    pub bytes: u64,
    pub elapsed_ms: u128,
    pub files_per_sec: f64,
    pub mb_per_sec: f64,
    pub p50_ms: u128,
    pub p95_ms: u128,
    pub slowest_file: Option<String>,
    pub slowest_ms: u128,
    pub db_write_ms: u128,
    pub queue_wait_ms: u128,
}

pub struct StageTimer {
    stage: String,
    started: Instant,
    files: usize,
    bytes: u64,
    samples: Vec<(u128, String)>,
    db_write: Duration,
    queue_wait: Duration,
}

impl StageTimer {
    pub fn new(stage: impl Into<String>) -> Self {
        Self {
            stage: stage.into(),
            started: Instant::now(),
            files: 0,
            bytes: 0,
            samples: Vec::new(),
            db_write: Duration::ZERO,
            queue_wait: Duration::ZERO,
        }
    }

    pub fn record_file(&mut self, bytes: u64, elapsed: Duration, file_name: impl Into<String>) {
        self.files += 1;
        self.bytes += bytes;
        self.samples.push((elapsed.as_millis(), file_name.into()));
    }

    pub fn add_db_write(&mut self, elapsed: Duration) {
        self.db_write += elapsed;
    }

    pub fn add_queue_wait(&mut self, elapsed: Duration) {
        self.queue_wait += elapsed;
    }

    pub fn finish(mut self) -> StagePerf {
        self.samples.sort_by_key(|sample| sample.0);
        let elapsed = self.started.elapsed();
        let elapsed_secs = elapsed.as_secs_f64().max(0.001);
        let p50 = percentile(&self.samples, 0.50);
        let p95 = percentile(&self.samples, 0.95);
        let slowest = self.samples.last().cloned();

        StagePerf {
            stage: self.stage,
            files: self.files,
            bytes: self.bytes,
            elapsed_ms: elapsed.as_millis(),
            files_per_sec: self.files as f64 / elapsed_secs,
            mb_per_sec: (self.bytes as f64 / 1_048_576.0) / elapsed_secs,
            p50_ms: p50,
            p95_ms: p95,
            slowest_file: slowest.as_ref().map(|(_, name)| name.clone()),
            slowest_ms: slowest.map(|(ms, _)| ms).unwrap_or(0),
            db_write_ms: self.db_write.as_millis(),
            queue_wait_ms: self.queue_wait.as_millis(),
        }
    }
}

pub fn log_stage(perf: &StagePerf) {
    crate::logging::info(format!(
        "[PERF] {}\nfiles={}\nbytes={}\nelapsed={:.2}s\nfiles_per_sec={:.1}\nmb_per_sec={:.2}\np50_ms={}\np95_ms={}\nslowest_file={}\nslowest_ms={}\ndb_write_ms={}\nqueue_wait_ms={}",
        perf.stage,
        perf.files,
        perf.bytes,
        perf.elapsed_ms as f64 / 1000.0,
        perf.files_per_sec,
        perf.mb_per_sec,
        perf.p50_ms,
        perf.p95_ms,
        perf.slowest_file.as_deref().unwrap_or("-"),
        perf.slowest_ms,
        perf.db_write_ms,
        perf.queue_wait_ms
    ));
}

fn percentile(samples: &[(u128, String)], pct: f64) -> u128 {
    if samples.is_empty() {
        return 0;
    }
    let idx = ((samples.len() - 1) as f64 * pct).round() as usize;
    samples[idx].0
}
