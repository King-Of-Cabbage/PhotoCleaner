//! What a file is, and what its container says about it.
//!
//! Video metadata now comes from the bundled ffprobe (see [`crate::ffprobe`]).
//! The previous implementation read the entire clip into memory and walked it
//! looking for `mvhd`/`tkhd` boxes, advancing one byte at a time whenever a
//! header did not line up - unbounded memory and a linear rescan, to learn a
//! duration.
//!
//! A bounded header scan survives only as a fallback for builds where
//! `runtime/media/ffprobe.exe` is absent, and it reads at most
//! [`HEADER_SCAN_LIMIT`] bytes.

use std::fs::File;
use std::io::Read;
use std::path::Path;

use crate::ffprobe::{self, ProbeFailure};
use crate::paths::PortablePaths;
use crate::scan_state;

/// Never read more than this from the front of a file for header parsing.
pub const HEADER_SCAN_LIMIT: usize = 4 * 1024 * 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MediaType {
    Image,
    Video,
    Sidecar,
    Unsupported,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MediaRole {
    PrimaryImage,
    SingleVideo,
    PairedVideo,
    Sidecar,
    Unsupported,
}

#[derive(Clone, Debug)]
pub struct MediaProbe {
    pub media_type: MediaType,
    pub media_role: MediaRole,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub duration_ms: Option<i64>,
    pub container: Option<String>,
    pub video_codec: Option<String>,
    pub audio_codec: Option<String>,
    pub frame_rate: Option<f64>,
    /// `com.apple.quicktime.content.identifier` when the container carries it.
    pub content_identifier: Option<String>,
    /// Container creation time, used as a capture-time fallback.
    pub creation_time: Option<String>,
    /// True only when Apple metadata confirms a Live Photo component. A
    /// filename that merely looks like one does not set this.
    pub apple_live_photo: bool,
    pub scan_state: String,
    pub failure_stage: Option<String>,
    pub failure_message: Option<String>,
}

impl MediaProbe {
    fn blank(media_type: MediaType, media_role: MediaRole, state: &str) -> Self {
        Self {
            media_type,
            media_role,
            width: None,
            height: None,
            duration_ms: None,
            container: None,
            video_codec: None,
            audio_codec: None,
            frame_rate: None,
            content_identifier: None,
            creation_time: None,
            apple_live_photo: false,
            scan_state: state.to_string(),
            failure_stage: None,
            failure_message: None,
        }
    }

    fn failed(
        media_type: MediaType,
        media_role: MediaRole,
        state: &str,
        stage: &str,
        message: String,
    ) -> Self {
        let mut probe = Self::blank(media_type, media_role, state);
        probe.failure_stage = Some(stage.to_string());
        probe.failure_message = Some(message);
        probe
    }
}

pub fn classify_extension(ext: &str) -> MediaType {
    match ext.to_ascii_lowercase().as_str() {
        "jpg" | "jpeg" | "png" | "webp" | "bmp" | "tif" | "tiff" | "heic" | "heif" => {
            MediaType::Image
        }
        "mov" | "mp4" | "m4v" | "avi" | "mkv" | "webm" => MediaType::Video,
        "aae" => MediaType::Sidecar,
        _ => MediaType::Unsupported,
    }
}

pub fn default_role_for(media_type: MediaType) -> MediaRole {
    match media_type {
        MediaType::Image => MediaRole::PrimaryImage,
        MediaType::Video => MediaRole::SingleVideo,
        MediaType::Sidecar => MediaRole::Sidecar,
        MediaType::Unsupported => MediaRole::Unsupported,
    }
}

pub fn probe(paths: &PortablePaths, path: &Path, extension: &str) -> MediaProbe {
    match classify_extension(extension) {
        MediaType::Image => probe_image(path, extension),
        MediaType::Video => probe_video(paths, path, extension),
        MediaType::Sidecar => {
            let mut probe =
                MediaProbe::blank(MediaType::Sidecar, MediaRole::Sidecar, scan_state::SUCCESS);
            probe.container = Some("AAE".to_string());
            probe
        }
        MediaType::Unsupported => MediaProbe::blank(
            MediaType::Unsupported,
            MediaRole::Unsupported,
            scan_state::UNSUPPORTED,
        ),
    }
}

fn probe_image(path: &Path, extension: &str) -> MediaProbe {
    let mut probe = MediaProbe::blank(
        MediaType::Image,
        MediaRole::PrimaryImage,
        scan_state::SUCCESS,
    );

    if matches!(extension, "heic" | "heif") {
        probe.container = Some("HEIF".to_string());
        match read_header(path, HEADER_SCAN_LIMIT) {
            Ok(bytes) => match parse_heif_dimensions(&bytes) {
                Some((width, height)) => {
                    probe.width = Some(width);
                    probe.height = Some(height);
                }
                None => {
                    return MediaProbe::failed(
                        MediaType::Image,
                        MediaRole::PrimaryImage,
                        scan_state::DECODE_FAILED,
                        "HEIC_HEADER",
                        format!(
                            "no ispe box in the first {} bytes of {}",
                            HEADER_SCAN_LIMIT,
                            path.display()
                        ),
                    );
                }
            },
            Err(error) => {
                return MediaProbe::failed(
                    MediaType::Image,
                    MediaRole::PrimaryImage,
                    scan_state::IO_FAILED,
                    "HEIC_HEADER",
                    error,
                );
            }
        }
        return probe;
    }

    match image::image_dimensions(path) {
        Ok((width, height)) => {
            probe.width = Some(width);
            probe.height = Some(height);
            probe.container = Some(extension.to_ascii_uppercase());
            probe
        }
        Err(error) => MediaProbe::failed(
            MediaType::Image,
            MediaRole::PrimaryImage,
            scan_state::DECODE_FAILED,
            "IMAGE_DIMENSIONS",
            error.to_string(),
        ),
    }
}

fn probe_video(paths: &PortablePaths, path: &Path, extension: &str) -> MediaProbe {
    let mut probe = MediaProbe::blank(
        MediaType::Video,
        MediaRole::SingleVideo,
        scan_state::SUCCESS,
    );
    probe.container = Some(extension.to_ascii_uppercase());

    match ffprobe::probe_path(paths, path) {
        Ok(metadata) => {
            let apple_live_photo = metadata.is_apple_live_photo_component();
            probe.width = metadata.width;
            probe.height = metadata.height;
            probe.duration_ms = metadata.duration_ms;
            probe.video_codec = metadata.video_codec;
            probe.audio_codec = metadata.audio_codec;
            probe.frame_rate = metadata.frame_rate;
            probe.creation_time = metadata.creation_time;
            probe.apple_live_photo = apple_live_photo;
            probe.content_identifier = metadata.content_identifier;
            if let Some(container) = metadata.container {
                probe.container = Some(container);
            }
            probe
        }
        // A build without the runtime folder still deserves to scan. Anything
        // else means ffprobe looked at the file and refused it, and guessing
        // after that would only produce worse data.
        Err(ProbeFailure::MissingExecutable(message)) => {
            match read_header(path, HEADER_SCAN_LIMIT) {
                Ok(bytes) => {
                    probe.duration_ms =
                        find_box_payload(&bytes, b"mvhd").and_then(parse_mvhd_duration);
                    if let Some((width, height)) =
                        find_box_payload(&bytes, b"tkhd").and_then(parse_tkhd_dimensions)
                    {
                        probe.width = Some(width);
                        probe.height = Some(height);
                    }
                    if probe.duration_ms.is_none() && probe.width.is_none() {
                        probe.scan_state = scan_state::VIDEO_PROBE_FAILED.to_string();
                        probe.failure_stage = Some("VIDEO_PROBE".to_string());
                        probe.failure_message = Some(format!(
                            "{message}; the bounded header fallback found no mvhd or tkhd box"
                        ));
                    }
                    probe
                }
                Err(error) => MediaProbe::failed(
                    MediaType::Video,
                    MediaRole::SingleVideo,
                    scan_state::IO_FAILED,
                    "VIDEO_HEADER",
                    format!("{message}; {error}"),
                ),
            }
        }
        Err(failure) => MediaProbe::failed(
            MediaType::Video,
            MediaRole::SingleVideo,
            scan_state::VIDEO_PROBE_FAILED,
            "VIDEO_PROBE",
            failure.message(),
        ),
    }
}

/// Reads at most `limit` bytes from the front of a file.
fn read_header(path: &Path, limit: usize) -> Result<Vec<u8>, String> {
    let file = File::open(path).map_err(|error| format!("cannot open: {error}"))?;
    let mut buffer = Vec::new();
    file.take(limit as u64)
        .read_to_end(&mut buffer)
        .map_err(|error| format!("cannot read: {error}"))?;
    Ok(buffer)
}

fn parse_heif_dimensions(bytes: &[u8]) -> Option<(u32, u32)> {
    find_box_payload(bytes, b"ispe").and_then(|payload| {
        if payload.len() < 12 {
            return None;
        }
        let width = u32::from_be_bytes(payload[4..8].try_into().ok()?);
        let height = u32::from_be_bytes(payload[8..12].try_into().ok()?);
        if width > 0 && height > 0 {
            Some((width, height))
        } else {
            None
        }
    })
}

fn parse_mvhd_duration(payload: &[u8]) -> Option<i64> {
    let version = *payload.first()?;
    if version == 1 {
        if payload.len() < 32 {
            return None;
        }
        let timescale = u32::from_be_bytes(payload[20..24].try_into().ok()?);
        let duration = u64::from_be_bytes(payload[24..32].try_into().ok()?);
        return scale_duration(duration, timescale);
    }
    if payload.len() < 20 {
        return None;
    }
    let timescale = u32::from_be_bytes(payload[12..16].try_into().ok()?);
    let duration = u32::from_be_bytes(payload[16..20].try_into().ok()?) as u64;
    scale_duration(duration, timescale)
}

fn parse_tkhd_dimensions(payload: &[u8]) -> Option<(u32, u32)> {
    if payload.len() < 8 {
        return None;
    }
    let width_fixed = read_u32_at(payload, payload.len().saturating_sub(8))?;
    let height_fixed = read_u32_at(payload, payload.len().saturating_sub(4))?;
    let width = width_fixed >> 16;
    let height = height_fixed >> 16;
    if width > 0 && height > 0 {
        Some((width, height))
    } else {
        None
    }
}

fn scale_duration(duration: u64, timescale: u32) -> Option<i64> {
    if timescale == 0 {
        return None;
    }
    Some(((duration as f64 / timescale as f64) * 1000.0).round() as i64)
}

fn read_u32_at(bytes: &[u8], start: usize) -> Option<u32> {
    Some(u32::from_be_bytes(
        bytes.get(start..start + 4)?.try_into().ok()?,
    ))
}

/// Walks ISO base media boxes looking for `box_type`.
///
/// A malformed size now ends the walk. The previous version advanced by a
/// single byte and tried again, which turned a corrupt header into a linear
/// scan of the whole buffer.
fn find_box_payload<'a>(bytes: &'a [u8], box_type: &[u8; 4]) -> Option<&'a [u8]> {
    let mut offset = 0usize;
    while offset + 8 <= bytes.len() {
        let size = u32::from_be_bytes(bytes[offset..offset + 4].try_into().ok()?) as usize;
        let name = &bytes[offset + 4..offset + 8];
        if size < 8 || offset + size > bytes.len() {
            return None;
        }
        let payload = &bytes[offset + 8..offset + size];
        if name == box_type {
            return Some(payload);
        }
        if matches!(
            name,
            b"meta" | b"moov" | b"trak" | b"mdia" | b"minf" | b"stbl" | b"iprp" | b"ipco"
        ) {
            // `meta` is a full box: version and flags precede its children.
            let nested = if name == b"meta" && payload.len() > 4 {
                &payload[4..]
            } else {
                payload
            };
            if let Some(found) = find_box_payload(nested, box_type) {
                return Some(found);
            }
        }
        offset += size;
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifies_media_extensions() {
        assert_eq!(classify_extension("JPG"), MediaType::Image);
        assert_eq!(classify_extension("heic"), MediaType::Image);
        assert_eq!(classify_extension("mov"), MediaType::Video);
        assert_eq!(classify_extension("AAE"), MediaType::Sidecar);
        assert_eq!(classify_extension("txt"), MediaType::Unsupported);
    }

    #[test]
    fn roles_follow_the_media_type() {
        assert_eq!(default_role_for(MediaType::Image), MediaRole::PrimaryImage);
        assert_eq!(default_role_for(MediaType::Video), MediaRole::SingleVideo);
        assert_eq!(default_role_for(MediaType::Sidecar), MediaRole::Sidecar);
    }

    fn iso_box(name: &[u8; 4], payload: &[u8]) -> Vec<u8> {
        let size = (payload.len() + 8) as u32;
        let mut out = Vec::new();
        out.extend_from_slice(&size.to_be_bytes());
        out.extend_from_slice(name);
        out.extend_from_slice(payload);
        out
    }

    #[test]
    fn finds_a_nested_box() {
        let mut ispe_payload = vec![0u8; 12];
        ispe_payload[4..8].copy_from_slice(&4032u32.to_be_bytes());
        ispe_payload[8..12].copy_from_slice(&3024u32.to_be_bytes());
        let ispe = iso_box(b"ispe", &ispe_payload);
        let ipco = iso_box(b"ipco", &ispe);
        let iprp = iso_box(b"iprp", &ipco);

        assert_eq!(parse_heif_dimensions(&iprp), Some((4032, 3024)));
    }

    /// A truncated or corrupt header must end the walk rather than degenerate
    /// into a byte-by-byte rescan.
    #[test]
    fn a_corrupt_box_size_stops_the_walk() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&3u32.to_be_bytes()); // size smaller than a header
        bytes.extend_from_slice(b"moov");
        bytes.extend_from_slice(&[0u8; 64]);
        assert_eq!(find_box_payload(&bytes, b"ispe"), None);

        let mut oversized = Vec::new();
        oversized.extend_from_slice(&9_000u32.to_be_bytes()); // longer than the buffer
        oversized.extend_from_slice(b"moov");
        oversized.extend_from_slice(&[0u8; 16]);
        assert_eq!(find_box_payload(&oversized, b"mvhd"), None);
    }

    #[test]
    fn an_empty_buffer_yields_nothing() {
        assert_eq!(find_box_payload(&[], b"mvhd"), None);
        assert_eq!(parse_heif_dimensions(&[]), None);
    }

    #[test]
    fn mvhd_version_zero_duration_is_scaled_to_millis() {
        let mut payload = vec![0u8; 20];
        payload[0] = 0; // version
        payload[12..16].copy_from_slice(&600u32.to_be_bytes()); // timescale
        payload[16..20].copy_from_slice(&3_000u32.to_be_bytes()); // duration
        assert_eq!(parse_mvhd_duration(&payload), Some(5_000));
    }

    #[test]
    fn a_zero_timescale_does_not_divide_by_zero() {
        let mut payload = vec![0u8; 20];
        payload[12..16].copy_from_slice(&0u32.to_be_bytes());
        payload[16..20].copy_from_slice(&3_000u32.to_be_bytes());
        assert_eq!(parse_mvhd_duration(&payload), None);
    }

    #[test]
    fn the_header_read_is_bounded() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("big.bin");
        std::fs::write(&path, vec![7u8; HEADER_SCAN_LIMIT * 2]).unwrap();
        let bytes = read_header(&path, HEADER_SCAN_LIMIT).unwrap();
        assert_eq!(
            bytes.len(),
            HEADER_SCAN_LIMIT,
            "a large file must not be read past the header limit"
        );
    }

    #[test]
    fn a_missing_file_reports_an_io_error() {
        let dir = tempfile::tempdir().unwrap();
        let missing = dir.path().join("nope.mov");
        assert!(read_header(&missing, HEADER_SCAN_LIMIT).is_err());
    }
}
