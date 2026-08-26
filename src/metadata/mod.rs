use std::fs;
use std::path::Path;
use std::time::SystemTime;

use chrono::{DateTime, Utc};

pub fn system_time_to_rfc3339(time: Option<SystemTime>) -> Option<String> {
    time.map(|t| DateTime::<Utc>::from(t).to_rfc3339())
}

pub fn image_dimensions(path: &Path) -> Option<(u32, u32)> {
    image::image_dimensions(path).ok()
}

pub fn file_times(meta: &fs::Metadata) -> (Option<String>, String) {
    let created = system_time_to_rfc3339(meta.created().ok());
    let modified =
        system_time_to_rfc3339(meta.modified().ok()).unwrap_or_else(|| Utc::now().to_rfc3339());
    (created, modified)
}
