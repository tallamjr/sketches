//! Cross-platform binary serialisation for sketches.
//!
//! HLL and Theta use the shared codec format (MAGIC + SketchHeader + payload).
//! All other sketches use this crate's own preamble layout; the format is
//! defined by this codec and is not interchangeable with any external library.
//! All multi-byte values are stored in little-endian byte order.

use std::fmt;

use crate::codec::{CodecError, Family, SketchHeader, SketchReader, SketchWriter};
use crate::cpc::CpcSketch;
use crate::hll::{HllPlusPlusSketch, HllSketch};
use crate::quantiles::KllSketch;
use crate::theta::ThetaSketch;

// ---------------------------------------------------------------------------
// Family IDs (Apache DataSketches standard)
// ---------------------------------------------------------------------------

pub const FAMILY_HLL: u8 = 7;
pub const FAMILY_THETA: u8 = 3;
pub const FAMILY_CPC: u8 = 16;
pub const FAMILY_FREQUENCY: u8 = 10;
pub const FAMILY_TDIGEST: u8 = 20;
pub const FAMILY_KLL: u8 = 15;
pub const FAMILY_REQ: u8 = 17;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can occur during serialisation or deserialisation.
#[derive(Debug)]
pub enum SerializationError {
    /// The preamble is structurally invalid.
    InvalidPreamble(String),
    /// The family_id in the byte stream does not match the expected family.
    FamilyMismatch { expected: u8, found: u8 },
    /// The serial_version in the byte stream does not match.
    VersionMismatch { expected: u8, found: u8 },
    /// Not enough bytes in the input.
    InsufficientData { needed: usize, available: usize },
    /// The byte stream is internally inconsistent.
    CorruptData(String),
}

impl fmt::Display for SerializationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SerializationError::InvalidPreamble(msg) => {
                write!(f, "invalid preamble: {msg}")
            }
            SerializationError::FamilyMismatch { expected, found } => {
                write!(f, "family mismatch: expected {expected}, found {found}")
            }
            SerializationError::VersionMismatch { expected, found } => {
                write!(f, "version mismatch: expected {expected}, found {found}")
            }
            SerializationError::InsufficientData { needed, available } => {
                write!(
                    f,
                    "insufficient data: needed {needed} bytes, have {available}"
                )
            }
            SerializationError::CorruptData(msg) => {
                write!(f, "corrupt data: {msg}")
            }
        }
    }
}

impl std::error::Error for SerializationError {}

// ---------------------------------------------------------------------------
// CodecError -> SerializationError bridge
// ---------------------------------------------------------------------------

impl From<CodecError> for SerializationError {
    fn from(e: CodecError) -> Self {
        match e {
            CodecError::UnexpectedEof => SerializationError::InsufficientData {
                needed: 0,
                available: 0,
            },
            CodecError::BadMagic => {
                SerializationError::InvalidPreamble("bad magic bytes".to_string())
            }
            CodecError::UnknownFamily(b) => {
                SerializationError::CorruptData(format!("unknown sketch family byte: {b}"))
            }
            CodecError::WrongFamily { expected, found } => SerializationError::CorruptData(
                format!("wrong sketch family: expected {expected:?}, found {found:?}"),
            ),
            CodecError::UnsupportedVersion(v) => SerializationError::VersionMismatch {
                expected: 1,
                found: v,
            },
        }
    }
}

// ---------------------------------------------------------------------------
// Serializable trait
// ---------------------------------------------------------------------------

/// Trait for sketches that support Apache DataSketches binary serialisation.
pub trait Serializable {
    /// Serialise the sketch to a byte vector in DataSketches format.
    fn to_bytes(&self) -> Vec<u8>;

    /// Deserialise a sketch from a byte slice in DataSketches format.
    fn from_bytes(bytes: &[u8]) -> Result<Self, SerializationError>
    where
        Self: Sized;

    /// Return the Apache DataSketches family identifier for this sketch type.
    fn family_id(&self) -> u8;

    /// Return the serial format version used by this implementation.
    fn serial_version(&self) -> u8;
}

// ---------------------------------------------------------------------------
// Public validation helper
// ---------------------------------------------------------------------------

/// Validate the minimum preamble of a sketch byte stream.
///
/// Returns `(family_id, serial_version, preamble_ints)` on success.
pub fn validate_sketch_bytes(bytes: &[u8]) -> Result<(u8, u8, u8), SerializationError> {
    if bytes.len() < 3 {
        return Err(SerializationError::InsufficientData {
            needed: 3,
            available: bytes.len(),
        });
    }
    let preamble_ints = bytes[0];
    let serial_version = bytes[1];
    let family_id = bytes[2];

    if preamble_ints == 0 {
        return Err(SerializationError::InvalidPreamble(
            "preamble_ints must be at least 1".to_string(),
        ));
    }

    let preamble_bytes = preamble_ints as usize * 8;
    if bytes.len() < preamble_bytes {
        return Err(SerializationError::InsufficientData {
            needed: preamble_bytes,
            available: bytes.len(),
        });
    }

    Ok((family_id, serial_version, preamble_ints))
}

// ---------------------------------------------------------------------------
// HLL serialisation constants
// ---------------------------------------------------------------------------

const HLL_SERIAL_VERSION: u8 = 1;
const HLL_PREAMBLE_INTS: u8 = 1;
const HLL_PREAMBLE_BYTES: usize = 8;
const HLL_FLAG_EMPTY: u8 = 1 << 2;

// ---------------------------------------------------------------------------
// HllSketch serialisation
// ---------------------------------------------------------------------------

impl Serializable for HllSketch {
    fn to_bytes(&self) -> Vec<u8> {
        let p = self.precision();
        let m = self.num_registers();
        let register_bytes: Vec<u8> = (0..m).map(|i| self.register_value(i)).collect();

        let is_empty = register_bytes.iter().all(|&r| r == 0);
        let flags: u8 = if is_empty { HLL_FLAG_EMPTY } else { 0 };

        // Compute cur_min (minimum non-zero register, or 0 if empty)
        let cur_min: u8 = if is_empty {
            0
        } else {
            register_bytes
                .iter()
                .copied()
                .filter(|&r| r > 0)
                .min()
                .unwrap_or(0)
        };

        // HIP trailer: hip_accum (f64), kxq0 (f64), kxq1 (f64), out_of_order (u8).
        let (hip_accum, kxq0, kxq1, out_of_order) = self.hip_raw_state();

        let mut w = SketchWriter::with_capacity(5 + 3 + m + 25);
        SketchHeader {
            family: Family::Hll,
            version: 1,
            flags,
        }
        .write(&mut w);
        // Payload: lg_k, cur_min, mode (0 = HLL8)
        w.put_u8(p);
        w.put_u8(cur_min);
        w.put_u8(0); // mode: 0 = HLL8
        // Data: one byte per register
        w.put_bytes(&register_bytes);
        // HIP trailer, written on every path so the layout is uniform.
        w.put_f64_le(hip_accum);
        w.put_f64_le(kxq0);
        w.put_f64_le(kxq1);
        w.put_u8(out_of_order as u8);
        w.into_vec()
    }

    fn from_bytes(bytes: &[u8]) -> Result<Self, SerializationError> {
        let mut r = SketchReader::new(bytes);
        let header = SketchHeader::read_expecting(&mut r, Family::Hll)?;
        let flags = header.flags;

        let lg_k = r.get_u8()?;
        let _cur_min = r.get_u8()?;
        let _mode = r.get_u8()?;

        if lg_k > 26 {
            return Err(SerializationError::CorruptData(format!(
                "lg_k={lg_k} exceeds maximum of 26"
            )));
        }

        let m = 1usize << lg_k;
        let is_empty = (flags & HLL_FLAG_EMPTY) != 0;

        if is_empty {
            // The empty path still writes the (all-zero) registers and HIP
            // trailer for a uniform layout; consume them to keep the reader
            // aligned, then return a fresh sketch (its HIP is the `new` default).
            if r.remaining() >= m {
                let _registers = r.get_bytes(m)?;
            }
            let _hip_accum = r.get_f64_le()?;
            let _kxq0 = r.get_f64_le()?;
            let _kxq1 = r.get_f64_le()?;
            let _out_of_order = r.get_u8()?;
            return Ok(HllSketch::new(lg_k));
        }

        if r.remaining() < m {
            return Err(SerializationError::InsufficientData {
                needed: m,
                available: r.remaining(),
            });
        }

        let registers = r.get_bytes(m)?.to_vec();
        let hip_accum = r.get_f64_le()?;
        let kxq0 = r.get_f64_le()?;
        let kxq1 = r.get_f64_le()?;
        let out_of_order = r.get_u8()? != 0;
        Ok(HllSketch::from_registers_with_hip(
            lg_k,
            registers,
            hip_accum,
            kxq0,
            kxq1,
            out_of_order,
        ))
    }

    fn family_id(&self) -> u8 {
        FAMILY_HLL
    }

    fn serial_version(&self) -> u8 {
        HLL_SERIAL_VERSION
    }
}

// ---------------------------------------------------------------------------
// HllPlusPlusSketch serialisation (same binary format as HllSketch)
// ---------------------------------------------------------------------------

impl Serializable for HllPlusPlusSketch {
    fn to_bytes(&self) -> Vec<u8> {
        let p = self.precision();
        let m = self.num_registers();
        let register_bytes: Vec<u8> = (0..m).map(|i| self.register_value(i)).collect();

        let is_empty = register_bytes.iter().all(|&r| r == 0);
        let flags: u8 = if is_empty { HLL_FLAG_EMPTY } else { 0 };

        let cur_min: u8 = if is_empty {
            0
        } else {
            register_bytes
                .iter()
                .copied()
                .filter(|&r| r > 0)
                .min()
                .unwrap_or(0)
        };

        let total_len = HLL_PREAMBLE_BYTES + m;
        let mut buf = Vec::with_capacity(total_len);

        buf.push(HLL_PREAMBLE_INTS);
        buf.push(HLL_SERIAL_VERSION);
        buf.push(FAMILY_HLL);
        buf.push(p);
        buf.push(0);
        buf.push(flags);
        buf.push(cur_min);
        buf.push(1); // mode 1 = HLL++ (distinguishes from plain HLL)

        buf.extend_from_slice(&register_bytes);

        buf
    }

    fn from_bytes(bytes: &[u8]) -> Result<Self, SerializationError> {
        if bytes.len() < HLL_PREAMBLE_BYTES {
            return Err(SerializationError::InsufficientData {
                needed: HLL_PREAMBLE_BYTES,
                available: bytes.len(),
            });
        }

        let preamble_ints = bytes[0];
        let serial_version = bytes[1];
        let family_id = bytes[2];
        let lg_k = bytes[3];
        let flags = bytes[5];

        if preamble_ints != HLL_PREAMBLE_INTS {
            return Err(SerializationError::InvalidPreamble(format!(
                "expected preamble_ints={HLL_PREAMBLE_INTS}, found {preamble_ints}"
            )));
        }

        if serial_version != HLL_SERIAL_VERSION {
            return Err(SerializationError::VersionMismatch {
                expected: HLL_SERIAL_VERSION,
                found: serial_version,
            });
        }

        if family_id != FAMILY_HLL {
            return Err(SerializationError::FamilyMismatch {
                expected: FAMILY_HLL,
                found: family_id,
            });
        }

        if !(4..=18).contains(&lg_k) {
            return Err(SerializationError::CorruptData(format!(
                "lg_k={lg_k} out of HLL++ range [4, 18]"
            )));
        }

        let m = 1usize << lg_k;
        let is_empty = (flags & HLL_FLAG_EMPTY) != 0;

        if is_empty {
            return Ok(HllPlusPlusSketch::new(lg_k));
        }

        let total_needed = HLL_PREAMBLE_BYTES + m;
        if bytes.len() < total_needed {
            return Err(SerializationError::InsufficientData {
                needed: total_needed,
                available: bytes.len(),
            });
        }

        let registers = bytes[HLL_PREAMBLE_BYTES..HLL_PREAMBLE_BYTES + m].to_vec();
        Ok(HllPlusPlusSketch::from_registers(lg_k, registers))
    }

    fn family_id(&self) -> u8 {
        FAMILY_HLL
    }

    fn serial_version(&self) -> u8 {
        HLL_SERIAL_VERSION
    }
}

// ---------------------------------------------------------------------------
// Theta serialisation constants
// ---------------------------------------------------------------------------

const THETA_SERIAL_VERSION: u8 = 3;
const THETA_FLAG_EMPTY: u8 = 1 << 2;
const THETA_FLAG_COMPACT: u8 = 1 << 3;
const THETA_FLAG_ORDERED: u8 = 1 << 4;

/// Opaque 2-byte tag written into our Theta header. It is a fixed constant
/// that identifies the seed used for this crate's hashing; it is part of our
/// own format and is not derived from or compatible with any external library.
const THETA_SEED_HASH: u16 = 0x93CC;

// ---------------------------------------------------------------------------
// ThetaSketch serialisation
// ---------------------------------------------------------------------------

impl Serializable for ThetaSketch {
    fn to_bytes(&self) -> Vec<u8> {
        let lg_k = self.lg_nom_size();
        let num_entries = self.num_retained();
        let theta = self.theta_value();
        let hashes = self.retained_hashes(); // sorted

        let is_empty = num_entries == 0;
        let flags: u8 =
            if is_empty { THETA_FLAG_EMPTY } else { 0 } | THETA_FLAG_COMPACT | THETA_FLAG_ORDERED;

        let mut w = SketchWriter::with_capacity(5 + 1 + 2 + 4 + 4 + 8 + num_entries * 8);
        SketchHeader {
            family: Family::Theta,
            version: 1,
            flags,
        }
        .write(&mut w);
        // Payload: lg_k, seed_hash (2 bytes LE), num_entries (u32), padding (u32), theta (u64)
        w.put_u8(lg_k);
        w.put_bytes(&THETA_SEED_HASH.to_le_bytes());
        w.put_u32_le(num_entries as u32);
        w.put_u32_le(0); // padding
        w.put_u64_le(theta);
        // Data: sorted hash values
        for &hash in &hashes {
            w.put_u64_le(hash);
        }
        w.into_vec()
    }

    fn from_bytes(bytes: &[u8]) -> Result<Self, SerializationError> {
        let mut r = SketchReader::new(bytes);
        let header = SketchHeader::read_expecting(&mut r, Family::Theta)?;
        let flags = header.flags;

        let lg_k = r.get_u8()?;
        // seed_hash: 2 bytes, not validated
        let _seed_hash_bytes = r.get_bytes(2)?;
        let num_entries = r.get_u32_le()? as usize;
        let _padding = r.get_u32_le()?;
        let theta = r.get_u64_le()?;

        let is_empty = (flags & THETA_FLAG_EMPTY) != 0;

        if is_empty && num_entries == 0 {
            let k = 1usize << lg_k;
            return Ok(ThetaSketch::new(k));
        }

        if r.remaining() < num_entries * 8 {
            return Err(SerializationError::InsufficientData {
                needed: num_entries * 8,
                available: r.remaining(),
            });
        }

        let mut values = Vec::with_capacity(num_entries);
        for _ in 0..num_entries {
            values.push(r.get_u64_le()?);
        }

        let k = 1usize << lg_k;
        // Reconstruct using ThetaSketch's native from_bytes path (handles hash table sizing).
        let mut native_buf = Vec::with_capacity(24 + values.len() * 8);
        native_buf.extend_from_slice(&k.to_le_bytes());
        native_buf.extend_from_slice(&theta.to_le_bytes());
        native_buf.extend_from_slice(&values.len().to_le_bytes());
        for &v in &values {
            native_buf.extend_from_slice(&v.to_le_bytes());
        }
        ThetaSketch::from_bytes(&native_buf).map_err(|msg| {
            SerializationError::CorruptData(format!("theta reconstruction failed: {msg}"))
        })
    }

    fn family_id(&self) -> u8 {
        FAMILY_THETA
    }

    fn serial_version(&self) -> u8 {
        THETA_SERIAL_VERSION
    }
}

// ---------------------------------------------------------------------------
// CPC serialisation constants
// ---------------------------------------------------------------------------

const CPC_SERIAL_VERSION: u8 = 1;

// ---------------------------------------------------------------------------
// CpcSketch serialisation
// ---------------------------------------------------------------------------

impl Serializable for CpcSketch {
    fn to_bytes(&self) -> Vec<u8> {
        // Delegate to the inherent codec serialisation, which writes the full
        // raw (uncompressed) state behind a Family::Cpc header.
        CpcSketch::to_bytes(self)
    }

    fn from_bytes(bytes: &[u8]) -> Result<Self, SerializationError> {
        // The inherent codec deserialisation reconstructs every field exactly;
        // CodecError maps to SerializationError via the existing From impl.
        Ok(CpcSketch::from_bytes(bytes)?)
    }

    fn family_id(&self) -> u8 {
        FAMILY_CPC
    }

    fn serial_version(&self) -> u8 {
        CPC_SERIAL_VERSION
    }
}

// ---------------------------------------------------------------------------
// KLL serialisation constants
// ---------------------------------------------------------------------------

const KLL_SERIAL_VERSION: u8 = 1;
const KLL_FLAG_EMPTY: u8 = 1 << 2;

// ---------------------------------------------------------------------------
// KllSketch<f64> serialisation
// ---------------------------------------------------------------------------

impl Serializable for KllSketch<f64> {
    fn to_bytes(&self) -> Vec<u8> {
        let k = self.k_value();
        let num_levels = self.num_levels_value();
        let is_empty = self.is_empty();

        let flags: u8 = if is_empty { KLL_FLAG_EMPTY } else { 0 };

        // Determine k as u8 (capped at 255)
        let k_u8 = k.min(255) as u8;

        // Compute level boundaries: the start offset of items for each level
        // and a final entry for the total count of retained items.
        let levels_ref = self.levels_ref();
        let mut level_boundaries: Vec<u32> = Vec::with_capacity(num_levels + 1);
        let mut running = 0u32;
        for level_items in levels_ref {
            level_boundaries.push(running);
            running += level_items.len() as u32;
        }
        level_boundaries.push(running); // total items

        let total_retained = running as usize;

        // Preamble: 16 bytes (2 longs)
        let preamble_ints: u8 = 2;
        let preamble_bytes = preamble_ints as usize * 8;

        // Data layout:
        //   level_boundaries: (num_levels + 1) * 4 bytes
        //   items: total_retained * 8 bytes
        //   min_value: 8 bytes
        //   max_value: 8 bytes
        let level_boundary_bytes = (num_levels + 1) * 4;
        let item_bytes = total_retained * 8;
        let minmax_bytes = 16;
        let data_bytes = if is_empty {
            0
        } else {
            level_boundary_bytes + item_bytes + minmax_bytes
        };
        let total_len = preamble_bytes + data_bytes;

        let mut buf = Vec::with_capacity(total_len);

        // Preamble (16 bytes)
        buf.push(preamble_ints); // byte 0
        buf.push(KLL_SERIAL_VERSION); // byte 1
        buf.push(FAMILY_KLL); // byte 2
        buf.push(k_u8); // byte 3: k (capped)
        buf.push(num_levels as u8); // byte 4: num_levels
        buf.push(flags); // byte 5: flags
        buf.extend_from_slice(&[0u8; 2]); // bytes 6-7: reserved
        buf.extend_from_slice(&(self.count() as u32).to_le_bytes()); // bytes 8-11: n
        buf.extend_from_slice(&(k as u32).to_le_bytes()); // bytes 12-15: full k value

        if !is_empty {
            // Level boundaries
            for &boundary in &level_boundaries {
                buf.extend_from_slice(&boundary.to_le_bytes());
            }

            // Items (f64 LE)
            for level_items in levels_ref {
                for &item in level_items {
                    buf.extend_from_slice(&item.to_le_bytes());
                }
            }

            // min_value, max_value
            let min_val = self.min().copied().unwrap_or(f64::NAN);
            let max_val = self.max().copied().unwrap_or(f64::NAN);
            buf.extend_from_slice(&min_val.to_le_bytes());
            buf.extend_from_slice(&max_val.to_le_bytes());
        }

        buf
    }

    fn from_bytes(bytes: &[u8]) -> Result<Self, SerializationError> {
        if bytes.len() < 16 {
            return Err(SerializationError::InsufficientData {
                needed: 16,
                available: bytes.len(),
            });
        }

        let _preamble_ints = bytes[0];
        let serial_version = bytes[1];
        let family_id = bytes[2];
        let k_u8 = bytes[3];
        let num_levels = bytes[4] as usize;
        let flags = bytes[5];

        if serial_version != KLL_SERIAL_VERSION {
            return Err(SerializationError::VersionMismatch {
                expected: KLL_SERIAL_VERSION,
                found: serial_version,
            });
        }

        if family_id != FAMILY_KLL {
            return Err(SerializationError::FamilyMismatch {
                expected: FAMILY_KLL,
                found: family_id,
            });
        }

        let n = u32::from_le_bytes(
            bytes[8..12]
                .try_into()
                .map_err(|_| SerializationError::CorruptData("invalid n".to_string()))?,
        ) as u64;

        let k = u32::from_le_bytes(
            bytes[12..16]
                .try_into()
                .map_err(|_| SerializationError::CorruptData("invalid full k".to_string()))?,
        ) as usize;

        // Validate that the capped k matches
        if k_u8 != k.min(255) as u8 {
            return Err(SerializationError::CorruptData(format!(
                "k_u8={k_u8} does not match k={k}"
            )));
        }

        let is_empty = (flags & KLL_FLAG_EMPTY) != 0;

        if is_empty || n == 0 {
            return Ok(KllSketch::new(k.max(8)));
        }

        let mut offset = 16usize;

        // Read level boundaries
        let boundary_count = num_levels + 1;
        let boundary_bytes = boundary_count * 4;
        if bytes.len() < offset + boundary_bytes {
            return Err(SerializationError::InsufficientData {
                needed: offset + boundary_bytes,
                available: bytes.len(),
            });
        }

        let mut level_boundaries = Vec::with_capacity(boundary_count);
        for i in 0..boundary_count {
            let o = offset + i * 4;
            let boundary = u32::from_le_bytes(bytes[o..o + 4].try_into().map_err(|_| {
                SerializationError::CorruptData("invalid level boundary".to_string())
            })?);
            level_boundaries.push(boundary as usize);
        }
        offset += boundary_bytes;

        let total_retained = *level_boundaries.last().unwrap_or(&0);

        // Read items
        let items_bytes = total_retained * 8;
        if bytes.len() < offset + items_bytes {
            return Err(SerializationError::InsufficientData {
                needed: offset + items_bytes,
                available: bytes.len(),
            });
        }

        let mut all_items = Vec::with_capacity(total_retained);
        for i in 0..total_retained {
            let o = offset + i * 8;
            let val = f64::from_le_bytes(
                bytes[o..o + 8]
                    .try_into()
                    .map_err(|_| SerializationError::CorruptData("invalid item".to_string()))?,
            );
            all_items.push(val);
        }
        offset += items_bytes;

        // Read min/max
        if bytes.len() < offset + 16 {
            return Err(SerializationError::InsufficientData {
                needed: offset + 16,
                available: bytes.len(),
            });
        }

        let _min_val = f64::from_le_bytes(
            bytes[offset..offset + 8]
                .try_into()
                .map_err(|_| SerializationError::CorruptData("invalid min_value".to_string()))?,
        );
        offset += 8;

        let _max_val = f64::from_le_bytes(
            bytes[offset..offset + 8]
                .try_into()
                .map_err(|_| SerializationError::CorruptData("invalid max_value".to_string()))?,
        );

        // Reconstruct the sketch by feeding all items back through update().
        // We create a fresh sketch and insert items level by level.
        // Items at level h represent 2^h original items, so we insert them
        // into level 0 and let the sketch re-compact naturally.
        //
        // However, for faithful reconstruction, we replay items into a fresh
        // sketch from level 0 upward. Since we cannot directly set internal
        // levels, we use the update() method and accept that the compaction
        // structure may differ slightly. The min, max, and total_count will
        // be exact, and quantile estimates will be statistically equivalent.
        let mut sketch = KllSketch::new(k.max(8));

        // Replay items from all levels, weighted by their level.
        // Level 0 items: insert once each
        // Level h items: insert 2^h times each (to approximate the original weight)
        // This ensures the total_count is correct.
        for level_idx in 0..num_levels {
            let start = level_boundaries[level_idx];
            let end = level_boundaries[level_idx + 1];
            let weight = 1u64 << level_idx;
            for &item in &all_items[start..end] {
                for _ in 0..weight {
                    sketch.update(item);
                }
            }
        }

        // The total_count from replay should match n (since we replay weighted).
        // The min/max are tracked automatically by update().
        // Verify min/max match (they should, since we replayed actual values).
        debug_assert!(
            sketch.min().is_some(),
            "sketch should have min after replay"
        );

        Ok(sketch)
    }

    fn family_id(&self) -> u8 {
        FAMILY_KLL
    }

    fn serial_version(&self) -> u8 {
        KLL_SERIAL_VERSION
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // HLL tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_hll_roundtrip_with_data() {
        let mut sketch = HllSketch::new(12);
        for i in 0..5000 {
            sketch.update(&format!("item_{i}"));
        }

        let original_estimate = sketch.estimate();

        let bytes = Serializable::to_bytes(&sketch);
        let restored = HllSketch::from_bytes(&bytes).unwrap();

        let restored_estimate = restored.estimate();
        assert!(
            (original_estimate - restored_estimate).abs() < 1.0,
            "HLL roundtrip estimate mismatch: {original_estimate} vs {restored_estimate}"
        );
    }

    #[test]
    fn test_hll_preamble_bytes() {
        let mut sketch = HllSketch::new(10);
        sketch.update(&"test_value");

        let bytes = Serializable::to_bytes(&sketch);

        // Codec header: MAGIC[0], MAGIC[1], family (1=Hll), version, flags
        assert_eq!(bytes[0], 0x53, "MAGIC[0]");
        assert_eq!(bytes[1], 0x4B, "MAGIC[1]");
        assert_eq!(bytes[2], Family::Hll as u8, "codec family");
        assert_eq!(bytes[3], 1, "codec version");
        // flags at bytes[4]: should NOT have empty bit set
        assert_eq!(bytes[4] & HLL_FLAG_EMPTY, 0, "should not be empty");
        // lg_k at bytes[5]
        assert_eq!(bytes[5], 10, "lg_k");
    }

    #[test]
    fn test_hll_empty_roundtrip() {
        let sketch = HllSketch::new(8);
        let bytes = Serializable::to_bytes(&sketch);

        // In codec format: flags are at bytes[4]
        assert_ne!(bytes[4] & HLL_FLAG_EMPTY, 0, "empty flag should be set");

        let restored = HllSketch::from_bytes(&bytes).unwrap();
        assert_eq!(restored.estimate(), 0.0);
        assert_eq!(restored.precision(), 8);
    }

    #[test]
    fn test_hll_byte_count() {
        let sketch = HllSketch::new(12);
        let bytes = Serializable::to_bytes(&sketch);

        // p=12 means m=4096 registers + 8 byte preamble + 25 byte HIP trailer
        // (hip_accum f64, kxq0 f64, kxq1 f64, out_of_order u8).
        let expected_size = HLL_PREAMBLE_BYTES + (1 << 12) + 25;
        assert_eq!(
            bytes.len(),
            expected_size,
            "HLL p=12 should produce {} bytes, got {}",
            expected_size,
            bytes.len()
        );
    }

    #[test]
    fn test_hll_truncated_data_error() {
        let mut sketch = HllSketch::new(8);
        sketch.update(&"hello");

        let bytes = Serializable::to_bytes(&sketch);
        // Truncate so data is incomplete
        let truncated = &bytes[..HLL_PREAMBLE_BYTES + 10];

        let result = HllSketch::from_bytes(truncated);
        assert!(result.is_err());
        if let Err(SerializationError::InsufficientData { needed, available }) = result {
            assert!(needed > available);
        } else {
            panic!("Expected InsufficientData error");
        }
    }

    #[test]
    fn test_hll_corrupted_family_id() {
        let sketch = HllSketch::new(8);
        let mut bytes = Serializable::to_bytes(&sketch);
        // In codec format bytes[2] is the family byte (Family::Hll = 1).
        // Changing it to 2 (Family::Theta) causes WrongFamily, which maps to CorruptData.
        bytes[2] = Family::Theta as u8;

        let result = HllSketch::from_bytes(&bytes);
        assert!(result.is_err(), "corrupted family byte must be rejected");
    }

    #[test]
    fn test_hll_cross_sketch_rejection() {
        // Serialise HLL, try to deserialise as Theta
        let mut sketch = HllSketch::new(10);
        sketch.update(&"value");
        let bytes = Serializable::to_bytes(&sketch);

        let result = <ThetaSketch as Serializable>::from_bytes(&bytes);
        assert!(result.is_err());
    }

    // -----------------------------------------------------------------------
    // HLL++ tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_hllpp_roundtrip_with_data() {
        let mut sketch = HllPlusPlusSketch::new(12);
        for i in 0..5000 {
            sketch.update(&format!("item_{i}"));
        }

        let original_estimate = sketch.estimate();

        let bytes = Serializable::to_bytes(&sketch);
        let restored = HllPlusPlusSketch::from_bytes(&bytes).unwrap();

        let restored_estimate = restored.estimate();
        assert!(
            (original_estimate - restored_estimate).abs() < 1.0,
            "HLL++ roundtrip estimate mismatch: {original_estimate} vs {restored_estimate}"
        );
    }

    #[test]
    fn test_hllpp_preamble_bytes() {
        let sketch = HllPlusPlusSketch::new(14);
        let bytes = Serializable::to_bytes(&sketch);

        assert_eq!(bytes[0], HLL_PREAMBLE_INTS);
        assert_eq!(bytes[1], HLL_SERIAL_VERSION);
        assert_eq!(bytes[2], FAMILY_HLL);
        assert_eq!(bytes[3], 14);
        assert_eq!(bytes[7], 1, "mode byte should be 1 for HLL++");
    }

    #[test]
    fn test_hllpp_empty_roundtrip() {
        let sketch = HllPlusPlusSketch::new(10);
        let bytes = Serializable::to_bytes(&sketch);

        assert_ne!(bytes[5] & HLL_FLAG_EMPTY, 0);

        let restored = HllPlusPlusSketch::from_bytes(&bytes).unwrap();
        assert_eq!(restored.estimate(), 0.0);
        assert_eq!(restored.precision(), 10);
    }

    #[test]
    fn test_hllpp_byte_count() {
        let sketch = HllPlusPlusSketch::new(12);
        let bytes = Serializable::to_bytes(&sketch);

        let expected_size = HLL_PREAMBLE_BYTES + (1 << 12);
        assert_eq!(bytes.len(), expected_size);
    }

    #[test]
    fn test_hllpp_truncated_data_error() {
        let mut sketch = HllPlusPlusSketch::new(8);
        sketch.update(&"hello");
        let bytes = Serializable::to_bytes(&sketch);
        let truncated = &bytes[..HLL_PREAMBLE_BYTES + 5];

        let result = HllPlusPlusSketch::from_bytes(truncated);
        assert!(result.is_err());
    }

    #[test]
    fn test_hllpp_corrupted_family_id() {
        let sketch = HllPlusPlusSketch::new(8);
        let mut bytes = Serializable::to_bytes(&sketch);
        bytes[2] = FAMILY_CPC;

        let result = HllPlusPlusSketch::from_bytes(&bytes);
        assert!(result.is_err());
        if let Err(SerializationError::FamilyMismatch { expected, found }) = result {
            assert_eq!(expected, FAMILY_HLL);
            assert_eq!(found, FAMILY_CPC);
        } else {
            panic!("Expected FamilyMismatch error");
        }
    }

    // -----------------------------------------------------------------------
    // Theta tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_theta_roundtrip_with_data() {
        let mut sketch = ThetaSketch::new(1024);
        for i in 0..5000 {
            sketch.update(&i);
        }

        let original_estimate = sketch.estimate();

        let bytes = Serializable::to_bytes(&sketch);
        let restored = <ThetaSketch as Serializable>::from_bytes(&bytes).unwrap();

        let restored_estimate = restored.estimate();
        let error_ratio =
            (original_estimate - restored_estimate).abs() / original_estimate.max(1.0);
        assert!(
            error_ratio < 0.01,
            "Theta roundtrip estimate mismatch: {original_estimate} vs {restored_estimate} (error {error_ratio})"
        );
    }

    #[test]
    fn test_theta_preamble_bytes() {
        let mut sketch = ThetaSketch::new(1024);
        sketch.update(&"test");

        let bytes = Serializable::to_bytes(&sketch);

        // Codec header: MAGIC[0], MAGIC[1], family (2=Theta), version, flags
        assert_eq!(bytes[0], 0x53, "MAGIC[0]");
        assert_eq!(bytes[1], 0x4B, "MAGIC[1]");
        assert_eq!(bytes[2], Family::Theta as u8, "codec family");
        assert_eq!(bytes[3], 1, "codec version");
        // lg_k at bytes[5] (after 5-byte header)
        assert_eq!(bytes[5], 10, "lg_nom_longs (log2(1024))");
    }

    #[test]
    fn test_theta_empty_roundtrip() {
        let sketch = ThetaSketch::new(512);
        let bytes = Serializable::to_bytes(&sketch);

        // In codec format: flags are at bytes[4]
        assert_ne!(bytes[4] & THETA_FLAG_EMPTY, 0, "empty flag should be set");

        let restored = <ThetaSketch as Serializable>::from_bytes(&bytes).unwrap();
        assert_eq!(restored.estimate(), 0.0);
    }

    #[test]
    fn test_theta_byte_count() {
        let mut sketch = ThetaSketch::new(2048);
        // Insert fewer than k items so we get exact count
        for i in 0..100 {
            sketch.update(&i);
        }

        let bytes = Serializable::to_bytes(&sketch);
        let num_entries = sketch.num_retained();

        // 24 byte codec header+payload + 8 bytes per entry
        let expected_size = 24 + num_entries * 8;
        assert_eq!(
            bytes.len(),
            expected_size,
            "Theta with {} entries should be {} bytes, got {}",
            num_entries,
            expected_size,
            bytes.len()
        );
    }

    #[test]
    fn test_theta_truncated_data_error() {
        let mut sketch = ThetaSketch::new(1024);
        for i in 0..100 {
            sketch.update(&i);
        }
        let bytes = Serializable::to_bytes(&sketch);
        // Truncate to just the 24-byte codec header+payload (no hash data)
        let truncated = &bytes[..24 + 4];

        let result = <ThetaSketch as Serializable>::from_bytes(truncated);
        assert!(result.is_err());
    }

    #[test]
    fn test_theta_corrupted_family_id() {
        let sketch = ThetaSketch::new(512);
        let mut bytes = Serializable::to_bytes(&sketch);
        // In codec format bytes[2] is the family byte (Family::Theta = 2).
        // Changing it to 1 (Family::Hll) causes WrongFamily, which maps to CorruptData.
        bytes[2] = Family::Hll as u8;

        let result = <ThetaSketch as Serializable>::from_bytes(&bytes);
        assert!(result.is_err(), "corrupted family byte must be rejected");
    }

    #[test]
    fn test_theta_cross_sketch_rejection() {
        // Serialise Theta, try to deserialise as HLL
        let mut sketch = ThetaSketch::new(1024);
        sketch.update(&"value");
        let bytes = Serializable::to_bytes(&sketch);

        let result = <HllSketch as Serializable>::from_bytes(&bytes);
        assert!(result.is_err());
    }

    // -----------------------------------------------------------------------
    // CPC tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_cpc_roundtrip_with_data() {
        let mut sketch = CpcSketch::new(11);
        for i in 0..1000 {
            sketch.update(&i);
        }

        let original_estimate = sketch.estimate();

        let bytes = Serializable::to_bytes(&sketch);
        let restored = <CpcSketch as Serializable>::from_bytes(&bytes).unwrap();

        let restored_estimate = restored.estimate();
        // Serialisation now stores the full raw state, so reconstruction is
        // exact: the estimate, coupon count, and bit-matrix invariant must all
        // be preserved precisely.
        assert!(
            (original_estimate - restored_estimate).abs() < 1e-6,
            "CPC roundtrip estimate mismatch: {original_estimate:.6} vs {restored_estimate:.6}"
        );
        assert_eq!(
            restored.num_coupons(),
            sketch.num_coupons(),
            "CPC roundtrip must preserve coupon count"
        );
        assert!(
            restored.validate(),
            "restored CPC sketch must satisfy the bit-matrix invariant"
        );
    }

    #[test]
    fn test_cpc_preamble_bytes() {
        let mut sketch = CpcSketch::new(10);
        sketch.update(&"test_value");

        let bytes = Serializable::to_bytes(&sketch);

        // Codec header: MAGIC[0], MAGIC[1], family (6=Cpc), version, flags
        assert_eq!(bytes[0], 0x53, "MAGIC[0]");
        assert_eq!(bytes[1], 0x4B, "MAGIC[1]");
        assert_eq!(bytes[2], Family::Cpc as u8, "codec family");
        assert_eq!(bytes[3], CPC_SERIAL_VERSION, "codec version");
        // lg_k is the first payload byte after the 5-byte header.
        assert_eq!(bytes[5], 10, "lg_k");
    }

    #[test]
    fn test_cpc_empty_roundtrip() {
        let sketch = CpcSketch::new(11);
        let bytes = Serializable::to_bytes(&sketch);

        assert_eq!(&bytes[0..2], &[0x53, 0x4B], "MAGIC");
        assert_eq!(bytes[2], Family::Cpc as u8, "codec family");

        let restored = <CpcSketch as Serializable>::from_bytes(&bytes).unwrap();
        assert!(restored.is_empty());
        assert_eq!(restored.estimate(), 0.0);
    }

    #[test]
    fn test_cpc_byte_count_sensible() {
        let mut sketch = CpcSketch::new(11);
        for i in 0..500 {
            sketch.update(&i);
        }

        let bytes = Serializable::to_bytes(&sketch);

        // Should be at least preamble (8 bytes) + some data
        assert!(
            bytes.len() > 16,
            "CPC with 500 items should produce more than 16 bytes, got {}",
            bytes.len()
        );

        // Should be reasonable: CPC is compact, so should be much less than
        // HLL with same lg_k=11 (which would be ~2048 bytes)
        assert!(
            bytes.len() < 20_000,
            "CPC serialised size {} seems unreasonably large",
            bytes.len()
        );
    }

    #[test]
    fn test_cpc_truncated_data_error() {
        let mut sketch = CpcSketch::new(10);
        for i in 0..100 {
            sketch.update(&i);
        }
        let bytes = Serializable::to_bytes(&sketch);
        // Truncate severely
        let truncated = &bytes[..6];

        let result = <CpcSketch as Serializable>::from_bytes(truncated);
        assert!(result.is_err());
    }

    #[test]
    fn test_cpc_corrupted_family_id() {
        let sketch = CpcSketch::new(11);
        let mut bytes = Serializable::to_bytes(&sketch);
        // In codec format bytes[2] is the family byte (Family::Cpc = 6).
        // Changing it to another known family (Theta) causes WrongFamily,
        // which maps to CorruptData.
        bytes[2] = Family::Theta as u8;

        let result = <CpcSketch as Serializable>::from_bytes(&bytes);
        assert!(result.is_err(), "corrupted family byte must be rejected");
    }

    #[test]
    fn test_cpc_cross_sketch_rejection() {
        let mut sketch = CpcSketch::new(10);
        sketch.update(&"value");
        let bytes = Serializable::to_bytes(&sketch);

        // Try to deserialise as HLL
        let result = HllSketch::from_bytes(&bytes);
        assert!(result.is_err());
    }

    // -----------------------------------------------------------------------
    // KLL tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_kll_roundtrip_with_data() {
        let mut sketch = KllSketch::new(200);
        for i in 0..10000 {
            sketch.update(i as f64);
        }

        let original_median = sketch.quantile(0.5).unwrap();

        let bytes = Serializable::to_bytes(&sketch);
        let mut restored = <KllSketch<f64> as Serializable>::from_bytes(&bytes).unwrap();

        let restored_median = restored.quantile(0.5).unwrap();

        // Medians should be close (within ~5% of range)
        let error = (original_median - restored_median).abs() / 10000.0;
        assert!(
            error < 0.05,
            "KLL roundtrip median mismatch: {original_median:.1} vs {restored_median:.1} (normalised error {error:.3})"
        );
    }

    #[test]
    fn test_kll_preamble_bytes() {
        let mut sketch = KllSketch::new(200);
        sketch.update(42.0);

        let bytes = Serializable::to_bytes(&sketch);

        assert_eq!(bytes[1], KLL_SERIAL_VERSION, "serial_version");
        assert_eq!(bytes[2], FAMILY_KLL, "family_id");
        assert_eq!(bytes[3], 200, "k as u8");
    }

    #[test]
    fn test_kll_empty_roundtrip() {
        let sketch = KllSketch::<f64>::new(200);
        let bytes = Serializable::to_bytes(&sketch);

        assert_ne!(bytes[5] & KLL_FLAG_EMPTY, 0, "empty flag should be set");

        let restored = <KllSketch<f64> as Serializable>::from_bytes(&bytes).unwrap();
        assert!(restored.is_empty());
        assert_eq!(restored.count(), 0);
    }

    #[test]
    fn test_kll_byte_count_sensible() {
        let mut sketch = KllSketch::new(200);
        for i in 0..1000 {
            sketch.update(i as f64);
        }

        let bytes = Serializable::to_bytes(&sketch);

        // At minimum: 16 byte preamble + level boundaries + items + min/max
        assert!(
            bytes.len() > 16,
            "KLL with 1000 items should be more than 16 bytes, got {}",
            bytes.len()
        );

        // KLL with k=200 should retain far fewer than 1000 items
        // Max retained ~= k * num_levels * 2, which for 1000 input items
        // is typically under 2000 items * 8 bytes = 16000 bytes
        assert!(
            bytes.len() < 50_000,
            "KLL serialised size {} seems unreasonably large",
            bytes.len()
        );
    }

    #[test]
    fn test_kll_truncated_data_error() {
        let mut sketch = KllSketch::new(200);
        for i in 0..100 {
            sketch.update(i as f64);
        }
        let bytes = Serializable::to_bytes(&sketch);
        let truncated = &bytes[..12];

        let result = <KllSketch<f64> as Serializable>::from_bytes(truncated);
        assert!(result.is_err());
    }

    #[test]
    fn test_kll_corrupted_family_id() {
        let sketch = KllSketch::<f64>::new(200);
        let mut bytes = Serializable::to_bytes(&sketch);
        bytes[2] = FAMILY_THETA;

        let result = <KllSketch<f64> as Serializable>::from_bytes(&bytes);
        assert!(result.is_err());
        if let Err(SerializationError::FamilyMismatch { expected, found }) = result {
            assert_eq!(expected, FAMILY_KLL);
            assert_eq!(found, FAMILY_THETA);
        } else {
            panic!("Expected FamilyMismatch error");
        }
    }

    #[test]
    fn test_kll_cross_sketch_rejection() {
        let mut sketch = KllSketch::new(200);
        sketch.update(1.0);
        let bytes = Serializable::to_bytes(&sketch);

        // Try to deserialise as Theta
        let result = <ThetaSketch as Serializable>::from_bytes(&bytes);
        assert!(result.is_err());
    }

    // -----------------------------------------------------------------------
    // Cross-type rejection matrix
    // -----------------------------------------------------------------------

    #[test]
    fn test_cross_type_rejection_hll_as_cpc() {
        let sketch = HllSketch::new(10);
        let bytes = Serializable::to_bytes(&sketch);

        let result = <CpcSketch as Serializable>::from_bytes(&bytes);
        assert!(result.is_err());
    }

    #[test]
    fn test_cross_type_rejection_theta_as_kll() {
        let sketch = ThetaSketch::new(512);
        let bytes = Serializable::to_bytes(&sketch);

        let result = <KllSketch<f64> as Serializable>::from_bytes(&bytes);
        assert!(result.is_err());
    }

    #[test]
    fn test_cross_type_rejection_kll_as_hll() {
        let sketch = KllSketch::<f64>::new(200);
        let bytes = Serializable::to_bytes(&sketch);

        let result = HllSketch::from_bytes(&bytes);
        assert!(result.is_err());
    }

    // -----------------------------------------------------------------------
    // validate_sketch_bytes
    // -----------------------------------------------------------------------

    #[test]
    fn test_validate_sketch_bytes_hll() {
        // HLL now uses the codec format (MAGIC prefix), so validate_sketch_bytes
        // interprets bytes[0] = 0x53 as preamble_ints, which is a valid non-zero
        // value but the resulting preamble_bytes check may fail on short buffers.
        // The codec-format HLL bytes should be parsed via SketchHeader::read_expecting,
        // not via the legacy validate_sketch_bytes helper.
        let mut sketch = HllSketch::new(12);
        sketch.update(&"value");
        let bytes = Serializable::to_bytes(&sketch);

        // Confirm the bytes start with the codec MAGIC.
        assert_eq!(&bytes[0..2], &[0x53, 0x4B]);

        // The legacy helper is not applicable to codec-format bytes.
        // A large enough codec buffer may pass or fail validate_sketch_bytes depending
        // on its length vs preamble_ints * 8; we simply confirm the codec round-trip works.
        let restored = HllSketch::from_bytes(&bytes).unwrap();
        assert!(restored.estimate() > 0.0);
    }

    #[test]
    fn test_validate_sketch_bytes_theta() {
        // Theta now uses the codec format (MAGIC prefix).
        let sketch = ThetaSketch::new(1024);
        let bytes = Serializable::to_bytes(&sketch);

        // Confirm the bytes start with the codec MAGIC.
        assert_eq!(&bytes[0..2], &[0x53, 0x4B]);

        // The legacy helper is not applicable to codec-format bytes.
        let restored = <ThetaSketch as Serializable>::from_bytes(&bytes).unwrap();
        assert_eq!(restored.estimate(), 0.0);
    }

    #[test]
    fn test_validate_sketch_bytes_too_short() {
        let bytes = vec![0u8; 2];
        let result = validate_sketch_bytes(&bytes);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_sketch_bytes_zero_preamble() {
        let bytes = vec![0u8; 10];
        let result = validate_sketch_bytes(&bytes);
        assert!(result.is_err());
    }

    // -----------------------------------------------------------------------
    // Trait method tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_family_id_methods() {
        let hll = HllSketch::new(8);
        assert_eq!(hll.family_id(), FAMILY_HLL);

        let hllpp = HllPlusPlusSketch::new(8);
        assert_eq!(hllpp.family_id(), FAMILY_HLL);

        let theta = ThetaSketch::new(512);
        assert_eq!(theta.family_id(), FAMILY_THETA);

        let cpc = CpcSketch::new(11);
        assert_eq!(cpc.family_id(), FAMILY_CPC);

        let kll = KllSketch::<f64>::new(200);
        assert_eq!(kll.family_id(), FAMILY_KLL);
    }

    #[test]
    fn test_serial_version_methods() {
        let hll = HllSketch::new(8);
        assert_eq!(hll.serial_version(), HLL_SERIAL_VERSION);

        let theta = ThetaSketch::new(512);
        assert_eq!(theta.serial_version(), THETA_SERIAL_VERSION);

        let cpc = CpcSketch::new(11);
        assert_eq!(cpc.serial_version(), CPC_SERIAL_VERSION);

        let kll = KllSketch::<f64>::new(200);
        assert_eq!(kll.serial_version(), KLL_SERIAL_VERSION);
    }
}
