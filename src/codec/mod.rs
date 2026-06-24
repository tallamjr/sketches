//! Shared little-endian codec and self-describing header for sketch
//! serialization. This is our own format namespace, not Apache's.

use std::fmt;

pub const MAGIC: [u8; 2] = [0x53, 0x4B]; // "SK"

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Family {
    Hll = 1,
    Theta = 2,
    Kll = 3,
    Bloom = 4,
    CountMin = 5,
    Cpc = 6,
}

impl Family {
    fn from_u8(v: u8) -> Result<Family, CodecError> {
        match v {
            1 => Ok(Family::Hll),
            2 => Ok(Family::Theta),
            3 => Ok(Family::Kll),
            4 => Ok(Family::Bloom),
            5 => Ok(Family::CountMin),
            6 => Ok(Family::Cpc),
            other => Err(CodecError::UnknownFamily(other)),
        }
    }
}

#[derive(Debug)]
pub enum CodecError {
    UnexpectedEof,
    BadMagic,
    UnknownFamily(u8),
    WrongFamily { expected: Family, found: Family },
    UnsupportedVersion(u8),
}

impl fmt::Display for CodecError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CodecError::UnexpectedEof => {
                write!(f, "unexpected end of input while decoding sketch")
            }
            CodecError::BadMagic => write!(f, "bad magic bytes: not a sketch serialization"),
            CodecError::UnknownFamily(b) => write!(f, "unknown sketch family id: {b}"),
            CodecError::WrongFamily { expected, found } => write!(
                f,
                "wrong sketch family: expected {expected:?}, found {found:?}"
            ),
            CodecError::UnsupportedVersion(v) => {
                write!(f, "unsupported serialization version: {v}")
            }
        }
    }
}

impl std::error::Error for CodecError {}

pub struct SketchWriter(Vec<u8>);

impl SketchWriter {
    pub fn new() -> Self {
        SketchWriter(Vec::new())
    }

    pub fn with_capacity(n: usize) -> Self {
        SketchWriter(Vec::with_capacity(n))
    }

    pub fn put_u8(&mut self, v: u8) {
        self.0.push(v);
    }

    pub fn put_u32_le(&mut self, v: u32) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }

    pub fn put_u64_le(&mut self, v: u64) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }

    pub fn put_f64_le(&mut self, v: f64) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }

    pub fn put_bytes(&mut self, b: &[u8]) {
        self.0.extend_from_slice(b);
    }

    pub fn into_vec(self) -> Vec<u8> {
        self.0
    }
}

impl Default for SketchWriter {
    fn default() -> Self {
        Self::new()
    }
}

pub struct SketchReader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> SketchReader<'a> {
    pub fn new(buf: &'a [u8]) -> Self {
        SketchReader { buf, pos: 0 }
    }

    pub fn remaining(&self) -> usize {
        self.buf.len() - self.pos
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], CodecError> {
        if self.remaining() < n {
            return Err(CodecError::UnexpectedEof);
        }
        let s = &self.buf[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }

    pub fn get_u8(&mut self) -> Result<u8, CodecError> {
        Ok(self.take(1)?[0])
    }

    pub fn get_u32_le(&mut self) -> Result<u32, CodecError> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    pub fn get_u64_le(&mut self) -> Result<u64, CodecError> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    pub fn get_f64_le(&mut self) -> Result<f64, CodecError> {
        Ok(f64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    pub fn get_bytes(&mut self, n: usize) -> Result<&'a [u8], CodecError> {
        self.take(n)
    }
}

#[derive(Debug)]
pub struct SketchHeader {
    pub family: Family,
    pub version: u8,
    pub flags: u8,
}

impl SketchHeader {
    pub fn write(&self, w: &mut SketchWriter) {
        w.put_bytes(&MAGIC);
        w.put_u8(self.family as u8);
        w.put_u8(self.version);
        w.put_u8(self.flags);
    }

    pub fn read(r: &mut SketchReader) -> Result<SketchHeader, CodecError> {
        let magic = r.get_bytes(2)?;
        if magic != MAGIC {
            return Err(CodecError::BadMagic);
        }
        let family = Family::from_u8(r.get_u8()?)?;
        let version = r.get_u8()?;
        let flags = r.get_u8()?;
        Ok(SketchHeader {
            family,
            version,
            flags,
        })
    }

    pub fn read_expecting(
        r: &mut SketchReader,
        expected: Family,
    ) -> Result<SketchHeader, CodecError> {
        let h = SketchHeader::read(r)?;
        if h.family != expected {
            return Err(CodecError::WrongFamily {
                expected,
                found: h.family,
            });
        }
        Ok(h)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_and_scalars_roundtrip() {
        let mut w = SketchWriter::new();
        SketchHeader {
            family: Family::Hll,
            version: 1,
            flags: 0b0000_0001,
        }
        .write(&mut w);
        w.put_u32_le(0xDEAD_BEEF);
        w.put_f64_le(3.5);
        w.put_bytes(&[1, 2, 3]);
        let bytes = w.into_vec();

        let mut r = SketchReader::new(&bytes);
        let h = SketchHeader::read(&mut r).unwrap();
        assert_eq!(h.family, Family::Hll);
        assert_eq!(h.version, 1);
        assert_eq!(h.flags, 0b0000_0001);
        assert_eq!(r.get_u32_le().unwrap(), 0xDEAD_BEEF);
        assert_eq!(r.get_f64_le().unwrap(), 3.5);
        assert_eq!(r.get_bytes(3).unwrap(), &[1, 2, 3]);
        assert_eq!(r.remaining(), 0);
    }

    #[test]
    fn wrong_family_is_rejected() {
        let mut w = SketchWriter::new();
        SketchHeader {
            family: Family::Theta,
            version: 1,
            flags: 0,
        }
        .write(&mut w);
        let bytes = w.into_vec();
        let mut r = SketchReader::new(&bytes);
        let err = SketchHeader::read_expecting(&mut r, Family::Hll).unwrap_err();
        assert!(matches!(err, CodecError::WrongFamily { .. }));
    }

    #[test]
    fn truncated_input_is_unexpected_eof() {
        // A header needs 5 bytes (2 magic + family + version + flags);
        // 3 bytes is not enough.
        let buf = [0x53u8, 0x4B, 0x01];
        let mut r = SketchReader::new(&buf);
        assert!(matches!(
            SketchHeader::read(&mut r),
            Err(CodecError::UnexpectedEof)
        ));
    }

    #[test]
    fn bad_magic_is_rejected() {
        // First byte corrupted; valid magic is [0x53, 0x4B].
        let buf = [0xFFu8, 0xFF, 1, 1, 0];
        let mut r = SketchReader::new(&buf);
        assert!(matches!(
            SketchHeader::read(&mut r),
            Err(CodecError::BadMagic)
        ));
    }

    #[test]
    fn unknown_family_is_rejected() {
        // Valid magic, family id 99 is not defined.
        let buf = [0x53u8, 0x4B, 99, 1, 0];
        let mut r = SketchReader::new(&buf);
        assert!(matches!(
            SketchHeader::read(&mut r),
            Err(CodecError::UnknownFamily(99))
        ));
    }
}
