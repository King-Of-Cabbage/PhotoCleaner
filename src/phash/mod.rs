//! Perceptual hashing.
//!
//! The previous implementation decoded straight to 8x8, took the mean of the
//! grey values and set one bit per pixel above it. That is an *average* hash.
//! It was then compared with thresholds (4 / 10 / 18) that come from the pHash
//! literature and only make sense for a DCT hash, so the classification
//! thresholds were being applied to a metric they were never calibrated for.
//!
//! This is a real perceptual hash: EXIF orientation is honoured, the image is
//! reduced to 32x32 greyscale, a 2-D DCT-II is taken, and the top-left 8x8
//! block of low-frequency coefficients is compared against its own median.
//! Brightness shifts, re-compression and mild scaling barely move it; unrelated
//! pictures land far away.

use std::path::Path;

use anyhow::{Context, Result};
use image::imageops::FilterType;
use image::DynamicImage;

use crate::paths::PortablePaths;

/// Side of the greyscale matrix the DCT runs over.
const DCT_SIZE: usize = 32;
/// Side of the low-frequency block kept from that matrix.
const LOW_FREQUENCY_SIZE: usize = 8;

/// Hamming distance between two hashes, i.e. how many of the 64 bits differ.
pub fn distance(left: u64, right: u64) -> u32 {
    (left ^ right).count_ones()
}

/// Computes the 64-bit perceptual hash of an image file.
pub fn compute_for_file(paths: &PortablePaths, path: &Path) -> Result<u64> {
    let image = load_oriented(paths, path)?;
    Ok(compute_for_image(&image))
}

/// Same as [`compute_for_file`] but for an image already in memory. Kept public
/// so the hash can be tested without touching the filesystem.
pub fn compute_for_image(image: &DynamicImage) -> u64 {
    let grey = image
        .resize_exact(DCT_SIZE as u32, DCT_SIZE as u32, FilterType::Triangle)
        .to_luma8();
    let mut matrix = vec![0f64; DCT_SIZE * DCT_SIZE];
    for (index, pixel) in grey.pixels().enumerate() {
        matrix[index] = pixel.0[0] as f64;
    }
    let coefficients = dct_2d(&matrix, DCT_SIZE);

    let mut low = [0f64; LOW_FREQUENCY_SIZE * LOW_FREQUENCY_SIZE];
    for row in 0..LOW_FREQUENCY_SIZE {
        for column in 0..LOW_FREQUENCY_SIZE {
            low[row * LOW_FREQUENCY_SIZE + column] = coefficients[row * DCT_SIZE + column];
        }
    }

    // The DC term is the average brightness of the whole image. Leaving it in
    // would drag the median towards it and make the hash sensitive to exposure,
    // which is exactly what a perceptual hash must ignore. It is excluded from
    // the median and then pinned to it, so its own bit carries no signal
    // instead of carrying a constant one.
    let median = median_of(&low[1..]);
    low[0] = median;

    let mut hash = 0u64;
    for (index, value) in low.iter().enumerate() {
        if *value > median {
            hash |= 1u64 << index;
        }
    }
    hash
}

fn load_oriented(paths: &PortablePaths, path: &Path) -> Result<DynamicImage> {
    let extension = path
        .extension()
        .and_then(|value| value.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();
    let image = if matches!(extension.as_str(), "heic" | "heif") {
        let png = crate::embedding::decode_heif_png_bytes(paths, path)?;
        image::load_from_memory(&png)
            .with_context(|| format!("HEIC decode output unreadable: {}", path.display()))?
    } else {
        image::open(path).with_context(|| format!("image decode failed: {}", path.display()))?
    };
    Ok(apply_orientation(image, exif_orientation(path)))
}

/// Reads the EXIF orientation tag, defaulting to 1 (no transform) whenever the
/// file has no EXIF block or it cannot be parsed.
///
/// `kamadak-exif` was already a declared dependency with zero call sites; two
/// photos differing only by the orientation flag used to hash differently.
pub fn exif_orientation(path: &Path) -> u32 {
    let Ok(file) = std::fs::File::open(path) else {
        return 1;
    };
    let mut reader = std::io::BufReader::new(file);
    let Ok(exif) = exif::Reader::new().read_from_container(&mut reader) else {
        return 1;
    };
    exif.get_field(exif::Tag::Orientation, exif::In::PRIMARY)
        .and_then(|field| field.value.get_uint(0))
        .unwrap_or(1)
}

pub fn apply_orientation(image: DynamicImage, orientation: u32) -> DynamicImage {
    match orientation {
        2 => image.fliph(),
        3 => image.rotate180(),
        4 => image.flipv(),
        5 => image.rotate90().fliph(),
        6 => image.rotate90(),
        7 => image.rotate270().fliph(),
        8 => image.rotate270(),
        _ => image,
    }
}

fn median_of(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let middle = sorted.len() / 2;
    if sorted.len() % 2 == 0 {
        (sorted[middle - 1] + sorted[middle]) / 2.0
    } else {
        sorted[middle]
    }
}

/// Separable 2-D DCT-II: rows first, then columns. At 32x32 this is about
/// 65k multiply-adds, far cheaper than decoding the image in the first place.
fn dct_2d(input: &[f64], size: usize) -> Vec<f64> {
    let basis = dct_basis(size);
    let mut rows = vec![0f64; size * size];
    for row in 0..size {
        for k in 0..size {
            let mut sum = 0f64;
            for n in 0..size {
                sum += input[row * size + n] * basis[k * size + n];
            }
            rows[row * size + k] = sum;
        }
    }
    let mut output = vec![0f64; size * size];
    for column in 0..size {
        for k in 0..size {
            let mut sum = 0f64;
            for n in 0..size {
                sum += rows[n * size + column] * basis[k * size + n];
            }
            output[k * size + column] = sum;
        }
    }
    output
}

fn dct_basis(size: usize) -> Vec<f64> {
    let mut basis = vec![0f64; size * size];
    for k in 0..size {
        for n in 0..size {
            basis[k * size + n] =
                (std::f64::consts::PI * (n as f64 + 0.5) * k as f64 / size as f64).cos();
        }
    }
    basis
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{ImageBuffer, ImageFormat, Rgb};
    use std::io::Cursor;

    /// A picture with real low-frequency structure, so the DCT has something to
    /// describe. A flat or purely random image would make the test meaningless.
    /// Values top out around 200 so a brightness test can shift them without
    /// clipping, which would otherwise distort the AC coefficients.
    fn structured_image(side: u32, phase: u32) -> DynamicImage {
        let buffer = ImageBuffer::<Rgb<u8>, Vec<u8>>::from_fn(side, side, |x, y| {
            let fx = x as f32 / side as f32;
            let fy = y as f32 / side as f32;
            let blocks = (((x + phase) / (side / 8)) + (y / (side / 6))) % 2;
            let base = 35.0 + 120.0 * fx * (1.0 - fy) + if blocks == 0 { 0.0 } else { 45.0 };
            let value = base.clamp(0.0, 200.0) as u8;
            Rgb([value, value.saturating_sub(12), value.saturating_add(9)])
        });
        DynamicImage::ImageRgb8(buffer)
    }

    /// Structurally unrelated to `structured_image`: concentric rings instead of
    /// a diagonal block grid, with the brightness gradient running the other way.
    fn radial_image(side: u32) -> DynamicImage {
        let centre = side as f32 / 2.0;
        let buffer = ImageBuffer::<Rgb<u8>, Vec<u8>>::from_fn(side, side, |x, y| {
            let dx = x as f32 - centre;
            let dy = y as f32 - centre;
            let radius = (dx * dx + dy * dy).sqrt();
            let ring = ((radius / 11.0) as u32) % 2;
            let base =
                190.0 - 130.0 * (radius / centre).min(1.0) - if ring == 0 { 0.0 } else { 40.0 };
            let value = base.clamp(0.0, 200.0) as u8;
            Rgb([value, value, value])
        });
        DynamicImage::ImageRgb8(buffer)
    }

    fn brighten(image: &DynamicImage, delta: i16) -> DynamicImage {
        let rgb = image.to_rgb8();
        let buffer = ImageBuffer::<Rgb<u8>, Vec<u8>>::from_fn(rgb.width(), rgb.height(), |x, y| {
            let p = rgb.get_pixel(x, y).0;
            Rgb([
                (p[0] as i16 + delta).clamp(0, 255) as u8,
                (p[1] as i16 + delta).clamp(0, 255) as u8,
                (p[2] as i16 + delta).clamp(0, 255) as u8,
            ])
        });
        DynamicImage::ImageRgb8(buffer)
    }

    fn jpeg_roundtrip(image: &DynamicImage) -> DynamicImage {
        let mut bytes = Vec::new();
        image
            .write_to(&mut Cursor::new(&mut bytes), ImageFormat::Jpeg)
            .unwrap();
        image::load_from_memory(&bytes).unwrap()
    }

    #[test]
    fn hash_is_stable_for_the_same_image() {
        let image = structured_image(256, 0);
        assert_eq!(compute_for_image(&image), compute_for_image(&image));
    }

    #[test]
    fn dc_bit_carries_no_signal() {
        // Bit 0 is pinned to the median, so it can never be set, whatever the
        // overall exposure is.
        let image = structured_image(256, 0);
        assert_eq!(compute_for_image(&image) & 1, 0);
        assert_eq!(compute_for_image(&brighten(&image, 30)) & 1, 0);
    }

    #[test]
    fn near_variants_stay_close_and_unrelated_images_do_not() {
        let base = structured_image(256, 0);
        let base_hash = compute_for_image(&base);

        let brighter = distance(base_hash, compute_for_image(&brighten(&base, 24)));
        let recompressed = distance(base_hash, compute_for_image(&jpeg_roundtrip(&base)));
        let rescaled = distance(
            base_hash,
            compute_for_image(&base.resize_exact(230, 230, FilterType::Triangle)),
        );
        let cropped = distance(base_hash, compute_for_image(&base.crop_imm(6, 6, 244, 244)));
        let unrelated = distance(base_hash, compute_for_image(&radial_image(256)));

        // A uniform brightness shift only moves the DC term, which is pinned.
        assert!(brighter <= 2, "brightness shift moved {brighter} bits");
        assert!(
            recompressed <= 8,
            "JPEG round trip moved {recompressed} bits"
        );
        assert!(rescaled <= 10, "10% rescale moved {rescaled} bits");
        assert!(cropped <= 14, "small crop moved {cropped} bits");

        // The property that actually matters: every near variant is closer to
        // the original than an unrelated picture is.
        for (name, near) in [
            ("brighter", brighter),
            ("recompressed", recompressed),
            ("rescaled", rescaled),
            ("cropped", cropped),
        ] {
            assert!(
                near < unrelated,
                "{name} distance {near} was not below unrelated distance {unrelated}"
            );
        }
        assert!(
            unrelated >= 12,
            "unrelated images were only {unrelated} bits apart"
        );
    }

    #[test]
    fn orientation_is_normalised_before_hashing() {
        let base = structured_image(256, 0);
        let rotated = base.rotate90();
        // Rotating the pixels changes the hash...
        assert_ne!(compute_for_image(&base), compute_for_image(&rotated));
        // ...and undoing it through the EXIF tag brings it back exactly.
        let restored = apply_orientation(rotated, 8);
        assert_eq!(compute_for_image(&base), compute_for_image(&restored));
    }

    #[test]
    fn distance_counts_differing_bits() {
        assert_eq!(distance(0, 0), 0);
        assert_eq!(distance(0b1011, 0b1000), 2);
        assert_eq!(distance(u64::MAX, 0), 64);
    }

    #[test]
    fn missing_or_unreadable_files_report_orientation_one() {
        assert_eq!(exif_orientation(Path::new("does-not-exist.jpg")), 1);
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("not-an-image.txt");
        std::fs::write(&path, b"plain text").unwrap();
        assert_eq!(exif_orientation(&path), 1);
    }
}
