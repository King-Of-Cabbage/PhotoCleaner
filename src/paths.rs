use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

pub const PORTABLE_WRITE_ERROR: &str =
    "当前程序目录不可写，Portable 模式无法保存数据库和配置，请将 PhotoCleaner 文件夹移动到可写目录。";

#[derive(Clone, Debug)]
pub struct PortablePaths {
    pub root: PathBuf,
    pub config_dir: PathBuf,
    pub settings_file: PathBuf,
    pub data_dir: PathBuf,
    pub db_file: PathBuf,
    pub indexes_dir: PathBuf,
    pub operations_dir: PathBuf,
    pub cache_dir: PathBuf,
    pub thumbnails_dir: PathBuf,
    pub models_dir: PathBuf,
    pub runtime_onnx_dir: PathBuf,
    pub runtime_media_dir: PathBuf,
    pub codecs_dir: PathBuf,
    pub logs_dir: PathBuf,
}

impl PortablePaths {
    pub fn from_current_exe() -> Result<Self> {
        let exe = env::current_exe().context("无法定位 PhotoCleaner.exe")?;
        let root = exe
            .parent()
            .map(Path::to_path_buf)
            .context("无法定位程序根目录")?;
        Ok(Self::from_root(root))
    }

    pub fn from_root(root: PathBuf) -> Self {
        let config_dir = root.join("config");
        let data_dir = root.join("data");
        let cache_dir = root.join("cache");
        Self {
            settings_file: config_dir.join("settings.json"),
            indexes_dir: data_dir.join("indexes"),
            operations_dir: data_dir.join("operations"),
            db_file: data_dir.join("photos.db"),
            thumbnails_dir: cache_dir.join("thumbnails"),
            models_dir: root.join("models"),
            runtime_onnx_dir: root.join("runtime").join("onnx"),
            runtime_media_dir: root.join("runtime").join("media"),
            codecs_dir: root.join("codecs"),
            logs_dir: root.join("logs"),
            config_dir,
            data_dir,
            cache_dir,
            root,
        }
    }

    pub fn ensure_layout(&self) -> Result<()> {
        for dir in [
            &self.config_dir,
            &self.data_dir,
            &self.indexes_dir,
            &self.operations_dir,
            &self.cache_dir,
            &self.thumbnails_dir,
            &self.models_dir,
            &self.runtime_onnx_dir,
            &self.runtime_media_dir,
            &self.codecs_dir,
            &self.logs_dir,
        ] {
            fs::create_dir_all(dir).with_context(|| PORTABLE_WRITE_ERROR)?;
        }
        self.assert_writable()
    }

    pub fn assert_writable(&self) -> Result<()> {
        let probe = self.root.join(".photocleaner_write_test");
        fs::write(&probe, b"ok").with_context(|| PORTABLE_WRITE_ERROR)?;
        let _ = fs::remove_file(probe);
        Ok(())
    }

    pub fn is_inside_app_root(&self, path: &Path) -> bool {
        let Ok(path) = path.canonicalize() else {
            return false;
        };
        let Ok(root) = self.root.canonicalize() else {
            return false;
        };
        path.starts_with(root)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_expected_portable_paths() {
        let root = PathBuf::from(r"D:\Tools\PhotoCleaner");
        let paths = PortablePaths::from_root(root.clone());
        assert_eq!(paths.db_file, root.join("data").join("photos.db"));
        assert_eq!(
            paths.settings_file,
            root.join("config").join("settings.json")
        );
        assert_eq!(paths.thumbnails_dir, root.join("cache").join("thumbnails"));
        assert_eq!(paths.runtime_onnx_dir, root.join("runtime").join("onnx"));
        assert_eq!(paths.runtime_media_dir, root.join("runtime").join("media"));
    }
}
