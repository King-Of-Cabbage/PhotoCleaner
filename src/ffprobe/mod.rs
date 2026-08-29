//! Video metadata via the bundled `ffprobe.exe`.
//!
//! `runtime/media/ffprobe.exe` ships with every portable build and was, until
//! now, never invoked. Video metadata came from a hand-rolled MP4 box scanner
//! that first did `fs::read` on the whole clip - a gigabyte of RAM to learn a
//! duration - and then walked the buffer byte by byte when a box header did not
//! line up.
//!
//! ffprobe reads only what it needs, understands containers this project never
//! will, and surfaces the Apple QuickTime tags that make Live Photo pairing
//! reliable instead of guesswork over filenames.
//!
//! The process boundary is deliberately thin: [`probe_path`] runs the
//! executable and hands the bytes to [`parse_ffprobe_json`], which is a pure
//! function. Everything interesting is therefore testable without ffprobe being
//! present.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{bail, Context, Result};
use serde::Deserialize;
use serde_json::Value;

use crate::paths::PortablePaths;

/// Apple writes the identifier shared by the two halves of a Live Photo here.
pub const APPLE_CONTENT_IDENTIFIER: &str = "com.apple.quicktime.content.identifier";
/// Present on the still frame of a Live Photo.
pub const APPLE_STILL_IMAGE_TIME: &str = "com.apple.quicktime.still-image-time";
/// Present when the capture was an automatic Live Photo.
pub const APPLE_LIVE_PHOTO_AUTO: &str = "com.apple.quicktime.live-photo.auto";

/// Everything the scanner wants to know about a video.
///
/// Every field is optional: ffprobe reports what the container happens to
/// carry, and a missing tag is normal rather than a failure.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct VideoMetadata {
    pub duration_ms: Option<i64>,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub video_codec: Option<String>,
    pub audio_codec: Option<String>,
    pub frame_rate: Option<f64>,
    /// `format_name` from ffprobe, e.g. `mov,mp4,m4a,3gp,3g2,mj2`.
    pub container: Option<String>,
    pub creation_time: Option<String>,
    /// `com.apple.quicktime.content.identifier`, the key to exact Live Photo
    /// pairing.
    pub content_identifier: Option<String>,
    pub still_image_time: Option<String>,
    pub live_photo_auto: Option<String>,
}

impl VideoMetadata {
    /// True when the container carries Apple's Live Photo marker, i.e. this is
    /// a confirmed component rather than a filename guess.
    pub fn is_apple_live_photo_component(&self) -> bool {
        self.content_identifier.is_some()
            || self.still_image_time.is_some()
            || self.live_photo_auto.is_some()
    }
}

/// Where a probe went wrong, so the scanner can record something truthful
/// instead of a hardcoded sentence.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ProbeFailure {
    /// `runtime/media/ffprobe.exe` is not in the portable layout.
    MissingExecutable(String),
    /// The process could not be started at all.
    LaunchFailed(String),
    /// ffprobe ran and refused the file.
    ExitStatus { code: Option<i32>, stderr: String },
    /// ffprobe produced output this build cannot read.
    MalformedJson(String),
}

impl ProbeFailure {
    pub fn message(&self) -> String {
        match self {
            Self::MissingExecutable(path) => format!("ffprobe not found at {path}"),
            Self::LaunchFailed(error) => format!("ffprobe could not be started: {error}"),
            Self::ExitStatus { code, stderr } => {
                let code = code
                    .map(|value| value.to_string())
                    .unwrap_or_else(|| "signal".to_string());
                if stderr.is_empty() {
                    format!("ffprobe exited with {code}")
                } else {
                    format!("ffprobe exited with {code}: {stderr}")
                }
            }
            Self::MalformedJson(error) => format!("ffprobe output could not be parsed: {error}"),
        }
    }
}

impl std::fmt::Display for ProbeFailure {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.message())
    }
}

impl std::error::Error for ProbeFailure {}

pub fn executable_path(paths: &PortablePaths) -> PathBuf {
    paths.runtime_media_dir.join("ffprobe.exe")
}

pub fn is_available(paths: &PortablePaths) -> bool {
    executable_path(paths).exists()
}

/// Runs the bundled ffprobe against `path` and parses its report.
pub fn probe_path(
    paths: &PortablePaths,
    path: &Path,
) -> std::result::Result<VideoMetadata, ProbeFailure> {
    let executable = executable_path(paths);
    if !executable.exists() {
        return Err(ProbeFailure::MissingExecutable(
            executable.display().to_string(),
        ));
    }

    let mut command = Command::new(&executable);
    command
        .arg("-hide_banner")
        .arg("-loglevel")
        .arg("error")
        .arg("-print_format")
        .arg("json")
        .arg("-show_format")
        .arg("-show_streams")
        .arg(input_path_argument(path));
    hide_console_window(&mut command);

    let output = command
        .output()
        .map_err(|error| ProbeFailure::LaunchFailed(error.to_string()))?;
    if !output.status.success() {
        return Err(ProbeFailure::ExitStatus {
            code: output.status.code(),
            stderr: String::from_utf8_lossy(&output.stderr).trim().to_string(),
        });
    }

    let text = String::from_utf8_lossy(&output.stdout);
    parse_ffprobe_json(&text).map_err(|error| ProbeFailure::MalformedJson(format!("{error:#}")))
}

/// Windows applies the `\\?\` prefix when a path is canonicalised; ffprobe does
/// not understand it, so it is stripped exactly as the ffmpeg decoder does.
fn input_path_argument(path: &Path) -> String {
    let text = path.to_string_lossy();
    text.strip_prefix(r"\\?\").unwrap_or(&text).to_string()
}

#[cfg(windows)]
fn hide_console_window(command: &mut Command) {
    use std::os::windows::process::CommandExt;
    // CREATE_NO_WINDOW. Without it every probe flashes a console window in
    // front of the GUI.
    command.creation_flags(0x0800_0000);
}

#[cfg(not(windows))]
fn hide_console_window(_command: &mut Command) {}

// ---------------------------------------------------------------------------
// The typed shape of `ffprobe -print_format json -show_format -show_streams`.
// Every field is optional on purpose: ffprobe omits whatever a container does
// not carry, and a build that insists on a field will fail on a perfectly good
// file.
// ---------------------------------------------------------------------------

#[derive(Debug, Default, Deserialize)]
struct FfprobeReport {
    #[serde(default)]
    streams: Vec<FfprobeStream>,
    #[serde(default)]
    format: Option<FfprobeFormat>,
}

#[derive(Debug, Default, Deserialize)]
struct FfprobeStream {
    #[serde(default)]
    codec_name: Option<String>,
    #[serde(default)]
    codec_type: Option<String>,
    #[serde(default)]
    width: Option<i64>,
    #[serde(default)]
    height: Option<i64>,
    #[serde(default)]
    coded_width: Option<i64>,
    #[serde(default)]
    coded_height: Option<i64>,
    #[serde(default)]
    r_frame_rate: Option<String>,
    #[serde(default)]
    avg_frame_rate: Option<String>,
    #[serde(default)]
    duration: Option<String>,
    #[serde(default)]
    tags: HashMap<String, Value>,
}

#[derive(Debug, Default, Deserialize)]
struct FfprobeFormat {
    #[serde(default)]
    format_name: Option<String>,
    #[serde(default)]
    duration: Option<String>,
    #[serde(default)]
    tags: HashMap<String, Value>,
}

/// Turns an ffprobe JSON report into [`VideoMetadata`].
///
/// Pure, so the whole extraction is testable without the executable.
pub fn parse_ffprobe_json(text: &str) -> Result<VideoMetadata> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        bail!("ffprobe produced no output");
    }
    let report: FfprobeReport =
        serde_json::from_str(trimmed).context("ffprobe JSON did not match the expected shape")?;

    let video = report
        .streams
        .iter()
        .find(|stream| stream.codec_type.as_deref() == Some("video"));
    let audio = report
        .streams
        .iter()
        .find(|stream| stream.codec_type.as_deref() == Some("audio"));

    let width = video
        .and_then(|stream| stream.width.or(stream.coded_width))
        .and_then(positive_dimension);
    let height = video
        .and_then(|stream| stream.height.or(stream.coded_height))
        .and_then(positive_dimension);

    // format.duration covers the container; a stream duration is the fallback
    // for formats that only report per-stream timing.
    let duration_ms = report
        .format
        .as_ref()
        .and_then(|format| format.duration.as_deref())
        .and_then(parse_seconds_to_millis)
        .or_else(|| {
            video
                .and_then(|stream| stream.duration.as_deref())
                .and_then(parse_seconds_to_millis)
        })
        .or_else(|| {
            audio
                .and_then(|stream| stream.duration.as_deref())
                .and_then(parse_seconds_to_millis)
        });

    let frame_rate = video.and_then(|stream| {
        stream
            .avg_frame_rate
            .as_deref()
            .and_then(parse_rational)
            .or_else(|| stream.r_frame_rate.as_deref().and_then(parse_rational))
    });

    let mut tags: HashMap<String, String> = HashMap::new();
    if let Some(format) = report.format.as_ref() {
        collect_tags(&format.tags, &mut tags);
    }
    for stream in &report.streams {
        collect_tags(&stream.tags, &mut tags);
    }

    Ok(VideoMetadata {
        duration_ms,
        width,
        height,
        video_codec: video.and_then(|stream| stream.codec_name.clone()),
        audio_codec: audio.and_then(|stream| stream.codec_name.clone()),
        frame_rate,
        container: report
            .format
            .as_ref()
            .and_then(|format| format.format_name.clone()),
        creation_time: lookup_tag(&tags, "creation_time"),
        content_identifier: lookup_tag(&tags, APPLE_CONTENT_IDENTIFIER),
        still_image_time: lookup_tag(&tags, APPLE_STILL_IMAGE_TIME),
        live_photo_auto: lookup_tag(&tags, APPLE_LIVE_PHOTO_AUTO),
    })
}

fn positive_dimension(value: i64) -> Option<u32> {
    if value > 0 && value <= u32::MAX as i64 {
        Some(value as u32)
    } else {
        None
    }
}

/// Tag keys are case-insensitive in practice; different muxers disagree about
/// capitalisation of the same tag.
fn collect_tags(source: &HashMap<String, Value>, target: &mut HashMap<String, String>) {
    for (key, value) in source {
        let Some(text) = value_as_string(value) else {
            continue;
        };
        if text.is_empty() {
            continue;
        }
        target.entry(key.to_ascii_lowercase()).or_insert(text);
    }
}

fn value_as_string(value: &Value) -> Option<String> {
    match value {
        Value::String(text) => Some(text.trim().to_string()),
        Value::Number(number) => Some(number.to_string()),
        Value::Bool(flag) => Some(flag.to_string()),
        _ => None,
    }
}

fn lookup_tag(tags: &HashMap<String, String>, key: &str) -> Option<String> {
    let value = tags.get(&key.to_ascii_lowercase())?;
    if is_unset(value) {
        None
    } else {
        Some(value.clone())
    }
}

fn is_unset(value: &str) -> bool {
    let trimmed = value.trim();
    trimmed.is_empty()
        || trimmed.eq_ignore_ascii_case("N/A")
        || trimmed.eq_ignore_ascii_case("unknown")
}

/// `"5.005"` -> `5005`. Rejects `N/A`, empty strings, negatives and non-finite
/// values rather than propagating nonsense into the database.
pub fn parse_seconds_to_millis(value: &str) -> Option<i64> {
    if is_unset(value) {
        return None;
    }
    let seconds: f64 = value.trim().parse().ok()?;
    if !seconds.is_finite() || seconds < 0.0 {
        return None;
    }
    let millis = (seconds * 1000.0).round();
    if millis > i64::MAX as f64 {
        return None;
    }
    Some(millis as i64)
}

/// `"30000/1001"` -> `29.97`. ffprobe writes `0/0` for streams with no
/// meaningful rate, and `N/A` when it has nothing at all.
pub fn parse_rational(value: &str) -> Option<f64> {
    if is_unset(value) {
        return None;
    }
    let trimmed = value.trim();
    let (numerator, denominator) = match trimmed.split_once('/') {
        Some((left, right)) => (left.trim(), right.trim()),
        None => (trimmed, "1"),
    };
    let numerator: f64 = numerator.parse().ok()?;
    let denominator: f64 = denominator.parse().ok()?;
    if denominator == 0.0 || !numerator.is_finite() || !denominator.is_finite() {
        return None;
    }
    let rate = numerator / denominator;
    if !rate.is_finite() || rate <= 0.0 {
        return None;
    }
    Some(rate)
}

#[cfg(test)]
mod tests {
    use super::*;

    const H264_AAC: &str = r#"
    {
      "streams": [
        {
          "index": 0,
          "codec_name": "h264",
          "codec_type": "video",
          "width": 1920,
          "height": 1080,
          "r_frame_rate": "30000/1001",
          "avg_frame_rate": "30000/1001",
          "duration": "5.005000",
          "tags": { "creation_time": "2026-01-02T03:04:05.000000Z", "rotate": "0" }
        },
        {
          "index": 1,
          "codec_name": "aac",
          "codec_type": "audio",
          "duration": "5.013000",
          "tags": {}
        }
      ],
      "format": {
        "filename": "IMG_1234.MOV",
        "nb_streams": 2,
        "format_name": "mov,mp4,m4a,3gp,3g2,mj2",
        "duration": "5.005000",
        "size": "4194304",
        "tags": {
          "creation_time": "2026-01-02T03:04:05.000000Z",
          "com.apple.quicktime.content.identifier": "8B7C1E2A-0F4D-4A11-9E33-1122334455AA",
          "com.apple.quicktime.live-photo.auto": "1"
        }
      }
    }"#;

    #[test]
    fn reads_an_h264_plus_aac_quicktime_file() {
        let metadata = parse_ffprobe_json(H264_AAC).unwrap();
        assert_eq!(metadata.video_codec.as_deref(), Some("h264"));
        assert_eq!(metadata.audio_codec.as_deref(), Some("aac"));
        assert_eq!(metadata.width, Some(1920));
        assert_eq!(metadata.height, Some(1080));
        assert_eq!(metadata.duration_ms, Some(5005));
        assert_eq!(
            metadata.container.as_deref(),
            Some("mov,mp4,m4a,3gp,3g2,mj2")
        );
        assert_eq!(
            metadata.creation_time.as_deref(),
            Some("2026-01-02T03:04:05.000000Z")
        );
    }

    #[test]
    fn ntsc_frame_rates_are_not_rounded_to_thirty() {
        let metadata = parse_ffprobe_json(H264_AAC).unwrap();
        let rate = metadata.frame_rate.unwrap();
        assert!(
            (rate - 29.97).abs() < 0.005,
            "30000/1001 should be ~29.97, got {rate}"
        );
    }

    #[test]
    fn apple_live_photo_tags_are_extracted() {
        let metadata = parse_ffprobe_json(H264_AAC).unwrap();
        assert_eq!(
            metadata.content_identifier.as_deref(),
            Some("8B7C1E2A-0F4D-4A11-9E33-1122334455AA")
        );
        assert_eq!(metadata.live_photo_auto.as_deref(), Some("1"));
        assert!(metadata.is_apple_live_photo_component());
    }

    #[test]
    fn hevc_is_read_the_same_way() {
        let json = r#"
        {
          "streams": [
            {
              "codec_name": "hevc",
              "codec_type": "video",
              "width": 3840,
              "height": 2160,
              "avg_frame_rate": "60/1"
            }
          ],
          "format": { "format_name": "mov,mp4", "duration": "12.5" }
        }"#;
        let metadata = parse_ffprobe_json(json).unwrap();
        assert_eq!(metadata.video_codec.as_deref(), Some("hevc"));
        assert_eq!(metadata.width, Some(3840));
        assert_eq!(metadata.height, Some(2160));
        assert_eq!(metadata.frame_rate, Some(60.0));
        assert_eq!(metadata.duration_ms, Some(12_500));
        assert_eq!(metadata.audio_codec, None);
        assert!(!metadata.is_apple_live_photo_component());
    }

    #[test]
    fn a_report_with_nothing_useful_still_parses() {
        let metadata = parse_ffprobe_json(r#"{"streams":[],"format":{}}"#).unwrap();
        assert_eq!(metadata, VideoMetadata::default());
    }

    #[test]
    fn missing_top_level_keys_are_tolerated() {
        let metadata = parse_ffprobe_json("{}").unwrap();
        assert_eq!(metadata, VideoMetadata::default());
    }

    #[test]
    fn not_available_placeholders_become_none() {
        let json = r#"
        {
          "streams": [
            {
              "codec_name": "h264",
              "codec_type": "video",
              "width": 0,
              "height": -1,
              "r_frame_rate": "0/0",
              "avg_frame_rate": "N/A",
              "duration": "N/A",
              "tags": { "creation_time": "N/A" }
            }
          ],
          "format": { "duration": "N/A", "format_name": "mov" }
        }"#;
        let metadata = parse_ffprobe_json(json).unwrap();
        assert_eq!(metadata.width, None);
        assert_eq!(metadata.height, None);
        assert_eq!(metadata.frame_rate, None);
        assert_eq!(metadata.duration_ms, None);
        assert_eq!(metadata.creation_time, None);
        assert_eq!(metadata.video_codec.as_deref(), Some("h264"));
    }

    #[test]
    fn a_stream_duration_covers_for_a_missing_format_duration() {
        let json = r#"
        {
          "streams": [
            { "codec_name": "h264", "codec_type": "video", "duration": "3.25" }
          ],
          "format": { "format_name": "matroska,webm" }
        }"#;
        let metadata = parse_ffprobe_json(json).unwrap();
        assert_eq!(metadata.duration_ms, Some(3250));
    }

    #[test]
    fn tag_lookup_ignores_capitalisation() {
        let json = r#"
        {
          "streams": [],
          "format": {
            "tags": { "Creation_Time": "2026-05-05T00:00:00Z" }
          }
        }"#;
        let metadata = parse_ffprobe_json(json).unwrap();
        assert_eq!(
            metadata.creation_time.as_deref(),
            Some("2026-05-05T00:00:00Z")
        );
    }

    #[test]
    fn numeric_tags_do_not_break_the_whole_report() {
        // Some muxers write numbers where a string is expected. Losing the
        // entire probe over one odd tag would be a poor trade.
        let json = r#"
        {
          "streams": [],
          "format": { "tags": { "com.apple.quicktime.live-photo.auto": 1 } }
        }"#;
        let metadata = parse_ffprobe_json(json).unwrap();
        assert_eq!(metadata.live_photo_auto.as_deref(), Some("1"));
    }

    #[test]
    fn empty_or_invalid_output_is_an_error_not_a_default() {
        assert!(parse_ffprobe_json("").is_err());
        assert!(parse_ffprobe_json("   ").is_err());
        assert!(parse_ffprobe_json("not json at all").is_err());
    }

    #[test]
    fn rational_parsing_covers_the_shapes_ffprobe_emits() {
        assert_eq!(parse_rational("60/1"), Some(60.0));
        assert_eq!(parse_rational("25"), Some(25.0));
        assert_eq!(parse_rational("0/0"), None);
        assert_eq!(parse_rational("N/A"), None);
        assert_eq!(parse_rational(""), None);
        assert_eq!(parse_rational("30/0"), None);
        assert_eq!(parse_rational("-30/1"), None);
        let ntsc = parse_rational("24000/1001").unwrap();
        assert!((ntsc - 23.976).abs() < 0.001);
    }

    #[test]
    fn duration_parsing_rejects_nonsense() {
        assert_eq!(parse_seconds_to_millis("1.5"), Some(1500));
        assert_eq!(parse_seconds_to_millis("0"), Some(0));
        assert_eq!(parse_seconds_to_millis("N/A"), None);
        assert_eq!(parse_seconds_to_millis(""), None);
        assert_eq!(parse_seconds_to_millis("-2"), None);
        assert_eq!(parse_seconds_to_millis("abc"), None);
    }

    #[test]
    fn failure_messages_name_what_went_wrong() {
        let missing = ProbeFailure::MissingExecutable("D:/x/ffprobe.exe".to_string());
        assert!(missing.message().contains("ffprobe.exe"));

        let status = ProbeFailure::ExitStatus {
            code: Some(1),
            stderr: "Invalid data found when processing input".to_string(),
        };
        assert!(status.message().contains("Invalid data"));
        assert!(status.message().contains('1'));
    }
}
