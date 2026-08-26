use std::fs;
use std::path::Path;

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
    pub content_identifier: Option<String>,
    pub scan_state: String,
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

pub fn probe(path: &Path, extension: &str) -> MediaProbe {
    match classify_extension(extension) {
        MediaType::Image => probe_image(path, extension),
        MediaType::Video => probe_video(path, extension),
        MediaType::Sidecar => MediaProbe {
            media_type: MediaType::Sidecar,
            media_role: MediaRole::Sidecar,
            width: None,
            height: None,
            duration_ms: None,
            container: Some("AAE".to_string()),
            video_codec: None,
            audio_codec: None,
            frame_rate: None,
            content_identifier: None,
            scan_state: "SUCCESS".to_string(),
        },
        MediaType::Unsupported => MediaProbe {
            media_type: MediaType::Unsupported,
            media_role: MediaRole::Unsupported,
            width: None,
            height: None,
            duration_ms: None,
            container: None,
            video_codec: None,
            audio_codec: None,
            frame_rate: None,
            content_identifier: None,
            scan_state: "UNSUPPORTED".to_string(),
        },
    }
}

fn probe_image(path: &Path, extension: &str) -> MediaProbe {
    let mut probe = MediaProbe {
        media_type: MediaType::Image,
        media_role: MediaRole::PrimaryImage,
        width: None,
        height: None,
        duration_ms: None,
        container: None,
        video_codec: None,
        audio_codec: None,
        frame_rate: None,
        content_identifier: None,
        scan_state: "SUCCESS".to_string(),
    };

    if matches!(extension, "heic" | "heif") {
        probe.container = Some("HEIF".to_string());
        match fs::read(path)
            .ok()
            .and_then(|bytes| parse_heif_dimensions(&bytes))
        {
            Some((width, height)) => {
                probe.width = Some(width);
                probe.height = Some(height);
            }
            None => probe.scan_state = "DECODE_FAILED".to_string(),
        }
        return probe;
    }

    match image::image_dimensions(path) {
        Ok((width, height)) => {
            probe.width = Some(width);
            probe.height = Some(height);
            probe.container = Some(extension.to_ascii_uppercase());
        }
        Err(_) => probe.scan_state = "DECODE_FAILED".to_string(),
    }
    probe
}

fn probe_video(path: &Path, extension: &str) -> MediaProbe {
    let bytes = fs::read(path).unwrap_or_default();
    let (duration_ms, width, height, content_identifier) = parse_mp4_like_metadata(&bytes);
    MediaProbe {
        media_type: MediaType::Video,
        media_role: MediaRole::SingleVideo,
        width,
        height,
        duration_ms,
        container: Some(extension.to_ascii_uppercase()),
        video_codec: find_codec_fourcc(&bytes),
        audio_codec: None,
        frame_rate: None,
        content_identifier,
        scan_state: if bytes.is_empty() {
            "FAILED"
        } else {
            "SUCCESS"
        }
        .to_string(),
    }
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

fn parse_mp4_like_metadata(
    bytes: &[u8],
) -> (Option<i64>, Option<u32>, Option<u32>, Option<String>) {
    let duration = find_box_payload(bytes, b"mvhd").and_then(parse_mvhd_duration);
    let dimensions = find_box_payload(bytes, b"tkhd").and_then(parse_tkhd_dimensions);
    let content_id = parse_live_photo_content_identifier(bytes);
    (
        duration,
        dimensions.map(|d| d.0),
        dimensions.map(|d| d.1),
        content_id,
    )
}

fn parse_live_photo_content_identifier(_bytes: &[u8]) -> Option<String> {
    None
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

fn find_codec_fourcc(bytes: &[u8]) -> Option<String> {
    for codec in [b"avc1", b"hvc1", b"hev1", b"mp4v", b"vp09"] {
        if bytes.windows(4).any(|w| w == codec) {
            return Some(String::from_utf8_lossy(codec).to_string());
        }
    }
    None
}

fn find_ascii_after(bytes: &[u8], needle: &[u8]) -> Option<String> {
    let pos = bytes.windows(needle.len()).position(|w| w == needle)?;
    let tail = &bytes[pos + needle.len()..bytes.len().min(pos + needle.len() + 256)];
    let text: String = tail
        .iter()
        .copied()
        .filter(|b| b.is_ascii_graphic() || *b == b' ')
        .map(char::from)
        .collect();
    let trimmed = text.trim_matches(|c: char| !c.is_ascii_alphanumeric() && c != '-');
    if trimmed.len() >= 8 {
        Some(trimmed.to_string())
    } else {
        None
    }
}

fn find_box_payload<'a>(bytes: &'a [u8], box_type: &[u8; 4]) -> Option<&'a [u8]> {
    let mut offset = 0usize;
    while offset + 8 <= bytes.len() {
        let size = u32::from_be_bytes(bytes[offset..offset + 4].try_into().ok()?) as usize;
        let name = &bytes[offset + 4..offset + 8];
        if size < 8 || offset + size > bytes.len() {
            offset += 1;
            continue;
        }
        let payload = &bytes[offset + 8..offset + size];
        if name == box_type {
            return Some(payload);
        }
        if matches!(
            name,
            b"meta" | b"moov" | b"trak" | b"mdia" | b"minf" | b"stbl" | b"iprp" | b"ipco"
        ) {
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
        assert_eq!(classify_extension("mov"), MediaType::Video);
        assert_eq!(classify_extension("AAE"), MediaType::Sidecar);
        assert_eq!(classify_extension("txt"), MediaType::Unsupported);
    }
}
