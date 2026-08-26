use std::fs;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::paths::PortablePaths;

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Settings {
    pub thumbnail_cache_limit_mb: u64,
    pub phash_threshold_strict: u32,
    pub phash_threshold_standard: u32,
    pub phash_threshold_loose: u32,
    pub ai_high_similarity: f32,
    pub ai_possible_similarity: f32,
    pub cpu_threads: CpuThreadSetting,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "mode", content = "threads")]
pub enum CpuThreadSetting {
    Auto,
    Fixed(usize),
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            thumbnail_cache_limit_mb: 1024,
            phash_threshold_strict: 4,
            phash_threshold_standard: 8,
            phash_threshold_loose: 12,
            ai_high_similarity: 0.92,
            ai_possible_similarity: 0.84,
            cpu_threads: CpuThreadSetting::Auto,
        }
    }
}

impl Settings {
    pub fn load_or_create(paths: &PortablePaths) -> Result<Self> {
        if paths.settings_file.exists() {
            let text = fs::read_to_string(&paths.settings_file).context("无法读取配置文件")?;
            return serde_json::from_str(&text).context("配置文件格式错误");
        }
        let settings = Self::default();
        settings.save(paths)?;
        Ok(settings)
    }

    pub fn save(&self, paths: &PortablePaths) -> Result<()> {
        let text = serde_json::to_string_pretty(self).context("无法生成配置文件")?;
        fs::write(&paths.settings_file, text).context("无法写入配置文件")
    }

    pub fn resolved_cpu_threads(&self) -> usize {
        match self.cpu_threads {
            CpuThreadSetting::Auto => num_cpus::get().saturating_sub(1).max(1),
            CpuThreadSetting::Fixed(n) => n.max(1),
        }
    }
}
