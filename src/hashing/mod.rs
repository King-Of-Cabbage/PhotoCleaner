//! Streaming file fingerprints.
//!
//! The quick hash intentionally keeps the same byte input as the previous
//! implementation: files up to 128 KiB are hashed whole, and larger files hash
//! the first 64 KiB followed by the last 64 KiB. Only the IO strategy changed,
//! so the stored quick-hash version remains compatible.

use std::collections::VecDeque;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

use anyhow::Result;
use sha2::{Digest, Sha256};
use xxhash_rust::xxh3::xxh3_64;

pub const STREAM_BUFFER_BYTES: usize = 1024 * 1024;
pub const QUICK_HASH_EDGE_BYTES: usize = 64 * 1024;
pub const QUICK_HASH_FULL_READ_LIMIT: usize = QUICK_HASH_EDGE_BYTES * 2;

pub struct FileFingerprint {
    pub quick_hash: u64,
    pub sha256: String,
    pub bytes_read: u64,
}

pub fn fingerprint_file(path: &Path, expected_size: u64) -> Result<FileFingerprint> {
    let file = File::open(path)?;
    let mut reader = BufReader::with_capacity(STREAM_BUFFER_BYTES, file);
    let mut sha = Sha256::new();
    let mut bytes_read = 0u64;
    let mut head = Vec::with_capacity(QUICK_HASH_EDGE_BYTES);
    let mut tail = VecDeque::with_capacity(QUICK_HASH_EDGE_BYTES);
    let mut full = Vec::with_capacity((expected_size as usize).min(QUICK_HASH_FULL_READ_LIMIT));
    let mut buffer = vec![0u8; STREAM_BUFFER_BYTES];

    loop {
        let read = reader.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        let chunk = &buffer[..read];
        sha.update(chunk);
        bytes_read += read as u64;

        if full.len() < QUICK_HASH_FULL_READ_LIMIT {
            let remaining = QUICK_HASH_FULL_READ_LIMIT - full.len();
            full.extend_from_slice(&chunk[..read.min(remaining)]);
        }

        if head.len() < QUICK_HASH_EDGE_BYTES {
            let remaining = QUICK_HASH_EDGE_BYTES - head.len();
            head.extend_from_slice(&chunk[..read.min(remaining)]);
        }

        for byte in chunk {
            if tail.len() == QUICK_HASH_EDGE_BYTES {
                tail.pop_front();
            }
            tail.push_back(*byte);
        }
    }

    let quick_hash = if bytes_read as usize > QUICK_HASH_FULL_READ_LIMIT {
        let mut sample = Vec::with_capacity(QUICK_HASH_FULL_READ_LIMIT);
        sample.extend_from_slice(&head);
        sample.extend(tail.iter().copied());
        xxh3_64(&sample)
    } else {
        xxh3_64(&full)
    };

    // Preserve the old persisted value for zero-byte files. For non-empty
    // files this is exactly the canonical SHA-256 digest, computed streaming.
    let sha256 = if bytes_read > 0 {
        format!("{:x}", sha.finalize())
    } else {
        String::new()
    };

    Ok(FileFingerprint {
        quick_hash,
        sha256,
        bytes_read,
    })
}

pub fn sha256_file(path: &Path) -> Result<String> {
    let file = File::open(path)?;
    let mut reader = BufReader::with_capacity(STREAM_BUFFER_BYTES, file);
    let mut sha = Sha256::new();
    let mut buffer = vec![0u8; STREAM_BUFFER_BYTES];
    loop {
        let read = reader.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        sha.update(&buffer[..read]);
    }
    Ok(format!("{:x}", sha.finalize()))
}

pub fn quick_hash_file(path: &Path, expected_size: u64) -> Result<u64> {
    Ok(fingerprint_file(path, expected_size)?.quick_hash)
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::Sha256;

    fn write_bytes(bytes: &[u8]) -> (tempfile::TempDir, std::path::PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("sample.bin");
        std::fs::write(&path, bytes).unwrap();
        (dir, path)
    }

    #[test]
    fn streaming_sha_matches_canonical_sha_for_non_empty_files() {
        let bytes = b"abc";
        let (_dir, path) = write_bytes(bytes);
        let fingerprint = fingerprint_file(&path, bytes.len() as u64).unwrap();
        assert_eq!(
            fingerprint.sha256,
            format!("{:x}", Sha256::digest(bytes.as_slice()))
        );
        assert_eq!(fingerprint.bytes_read, 3);
    }

    #[test]
    fn zero_byte_fingerprint_preserves_the_old_empty_sha_value() {
        let (_dir, path) = write_bytes(&[]);
        let fingerprint = fingerprint_file(&path, 0).unwrap();
        assert_eq!(fingerprint.sha256, "");
        assert_eq!(fingerprint.quick_hash, xxh3_64(&[]));
        assert_eq!(fingerprint.bytes_read, 0);
    }

    #[test]
    fn standalone_sha256_is_canonical_even_for_empty_files() {
        let (_dir, path) = write_bytes(&[]);
        assert_eq!(
            sha256_file(&path).unwrap(),
            format!("{:x}", Sha256::digest([]))
        );
    }

    #[test]
    fn quick_hash_uses_the_whole_small_file() {
        let bytes = vec![11u8; QUICK_HASH_EDGE_BYTES - 7];
        let (_dir, path) = write_bytes(&bytes);
        assert_eq!(
            quick_hash_file(&path, bytes.len() as u64).unwrap(),
            xxh3_64(&bytes)
        );
    }

    #[test]
    fn quick_hash_uses_head_and_tail_for_large_files() {
        let mut bytes: Vec<u8> = (0..(QUICK_HASH_FULL_READ_LIMIT + 33))
            .map(|idx| (idx % 251) as u8)
            .collect();
        bytes[QUICK_HASH_EDGE_BYTES + 10] ^= 0x7f;
        let (_dir, path) = write_bytes(&bytes);
        let mut sample = Vec::with_capacity(QUICK_HASH_FULL_READ_LIMIT);
        sample.extend_from_slice(&bytes[..QUICK_HASH_EDGE_BYTES]);
        sample.extend_from_slice(&bytes[bytes.len() - QUICK_HASH_EDGE_BYTES..]);
        assert_eq!(
            quick_hash_file(&path, bytes.len() as u64).unwrap(),
            xxh3_64(&sample)
        );
    }

    #[test]
    fn changing_the_middle_of_a_large_file_can_collide_by_design() {
        let first = vec![1u8; QUICK_HASH_FULL_READ_LIMIT + 4096];
        let mut second = first.clone();
        second[QUICK_HASH_EDGE_BYTES + 512] = 99;
        let (_a_dir, a) = write_bytes(&first);
        let (_b_dir, b) = write_bytes(&second);
        assert_eq!(
            quick_hash_file(&a, first.len() as u64).unwrap(),
            quick_hash_file(&b, second.len() as u64).unwrap()
        );
        assert_ne!(sha256_file(&a).unwrap(), sha256_file(&b).unwrap());
    }
}
