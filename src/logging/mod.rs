use std::fs::{self, OpenOptions};
use std::io::Write;
use std::sync::{Mutex, OnceLock};

use chrono::Local;

use crate::paths::PortablePaths;

static LOG_FILE: OnceLock<Mutex<std::fs::File>> = OnceLock::new();

pub fn init(paths: &PortablePaths) -> std::io::Result<()> {
    fs::create_dir_all(&paths.logs_dir)?;
    rotate_logs(paths);
    let file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(paths.logs_dir.join("photocleaner.log"))?;
    let _ = LOG_FILE.set(Mutex::new(file));
    info("PhotoCleaner started");
    Ok(())
}

pub fn info(message: impl AsRef<str>) {
    write_line("INFO", message.as_ref());
}

pub fn error(message: impl AsRef<str>) {
    write_line("ERROR", message.as_ref());
}

fn write_line(level: &str, message: &str) {
    if let Some(file) = LOG_FILE.get() {
        if let Ok(mut file) = file.lock() {
            let _ = writeln!(
                file,
                "{} [{}] {}",
                Local::now().format("%Y-%m-%d %H:%M:%S"),
                level,
                message
            );
        }
    }
}

fn rotate_logs(paths: &PortablePaths) {
    let current = paths.logs_dir.join("photocleaner.log");
    let Ok(meta) = fs::metadata(&current) else {
        return;
    };
    if meta.len() <= 10 * 1024 * 1024 {
        return;
    }
    let rotated = paths.logs_dir.join(format!(
        "photocleaner_{}.log",
        Local::now().format("%Y%m%d_%H%M%S")
    ));
    let _ = fs::rename(current, rotated);

    if let Ok(entries) = fs::read_dir(&paths.logs_dir) {
        let mut logs: Vec<_> = entries
            .flatten()
            .filter(|e| e.file_name().to_string_lossy().starts_with("photocleaner_"))
            .collect();
        logs.sort_by_key(|e| e.metadata().and_then(|m| m.modified()).ok());
        while logs.len() > 5 {
            if let Some(entry) = logs.first() {
                let _ = fs::remove_file(entry.path());
            }
            logs.remove(0);
        }
    }
}
