//! BC7 (BPTC) block decompression: all eight modes, partition sets, rotations and P-bits.
//!
//! Written from the normative tables and bit layout in Microsoft's "BC7 format" and the
//! Khronos data-format specification; `dds_bc.rs` holds the smaller block formats.

use super::dds_bc::Block;

/// An LSB-first reader over the 16 bytes of one BC7 block.
struct Bits {
    lo: u64,
    hi: u64,
    ofs: u32,
}

impl Bits {
    fn new(block: &[u8]) -> Self {
        let mut bytes = [0u8; 16];
        let n = block.len().min(16);
        if let (Some(d), Some(s)) = (bytes.get_mut(..n), block.get(..n)) {
            d.copy_from_slice(s);
        }
        let (lo, hi) = bytes.split_at(8);
        Self { lo: u64::from_le_bytes(lo.try_into().unwrap_or_default()), hi: u64::from_le_bytes(hi.try_into().unwrap_or_default()), ofs: 0 }
    }

    fn read(&mut self, n: u32) -> u64 {
        debug_assert!(n <= 64 && self.ofs + n <= 128);
        let v = if self.ofs >= 64 {
            self.hi >> (self.ofs - 64)
        } else if self.ofs == 0 {
            self.lo
        } else {
            (self.lo >> self.ofs) | (self.hi << (64 - self.ofs))
        };
        self.ofs += n;
        v & ((1u64 << n) - 1)
    }
}

// Per-mode field table (spec): subsets, partition bits, rotation bits, colour bits per
// component, alpha bits per component, per-endpoint P-bits (mode 1 uses shared P-bits
// instead), index bits, secondary index bits. Only mode 4 carries an index-select bit.
pub(crate) const N_SUBSETS: [u8; 8] = [3, 2, 3, 2, 1, 1, 1, 2];
pub(crate) const PARTITION_BITS: [u8; 8] = [4, 6, 6, 6, 0, 0, 0, 6];
const ROTATION_BITS: [u8; 8] = [0, 0, 0, 0, 2, 2, 0, 0];
const COLOR_BITS: [u8; 8] = [4, 6, 5, 7, 5, 7, 7, 5];
const ALPHA_BITS: [u8; 8] = [0, 0, 0, 0, 6, 8, 7, 5];
const EP_BITS: [u8; 8] = [1, 0, 0, 1, 0, 0, 1, 1];
const SP_BITS: [u8; 8] = [0, 1, 0, 0, 0, 0, 0, 0];
const INDEX_BITS: [u8; 8] = [3, 3, 2, 2, 2, 2, 4, 2];
const INDEX_BITS2: [u8; 8] = [0, 0, 0, 0, 3, 2, 0, 0];

const WEIGHTS: [[u8; 16]; 3] = [
    [0, 21, 43, 64, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 9, 18, 27, 37, 46, 55, 64, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64],
];

fn weights(index_bits: u8) -> &'static [u8; 16] {
    &WEIGHTS[usize::from(index_bits.saturating_sub(2).min(2))]
}

fn interpolate(e0: u8, e1: u8, index: u8, index_bits: u8) -> u8 {
    let w = u32::from(weights(index_bits)[index as usize]);
    ((u32::from(e0) * (64 - w) + u32::from(e1) * w + 32) >> 6) as u8
}

// Partition-set tables (normative spec data): 64 sets of 16 pixels mapping each pixel to
// its subset, for the 2-subset and 3-subset layouts.
pub(crate) const P2: [u8; 1024] = [
    0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1,
    0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1,
    0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0,
    1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1,
    0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
    1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0,
    0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0,
    1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1,
    0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0,
    1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1,
    0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1,
    0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0,
    1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0,
    0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0,
    0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1,
    1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0,
    1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1,
    0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1,
];

pub(crate) const P3: [u8; 1024] = [
    0, 0, 1, 1, 0, 0, 1, 1, 0, 2, 2, 1, 2, 2, 2, 2, 0, 0, 0, 1, 0, 0, 1, 1, 2, 2, 1, 1, 2, 2, 2, 1, 0, 0, 0, 0, 2, 0, 0, 1, 2, 2, 1, 1, 2, 2, 1, 1, 0, 2, 2, 2,
    0, 0, 2, 2, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 1, 1, 2, 2, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 2, 2, 0, 0, 2, 2, 0, 0, 2, 2, 0, 0, 2, 2,
    1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
    2, 2, 2, 2, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 1, 2, 0, 0, 1, 2, 0, 0, 1, 2, 0, 0, 1, 2, 0, 1, 1, 2, 0, 1, 1, 2, 0, 1, 1, 2, 0, 1, 1, 2,
    0, 1, 2, 2, 0, 1, 2, 2, 0, 1, 2, 2, 0, 1, 2, 2, 0, 0, 1, 1, 0, 1, 1, 2, 1, 1, 2, 2, 1, 2, 2, 2, 0, 0, 1, 1, 2, 0, 0, 1, 2, 2, 0, 0, 2, 2, 2, 0, 0, 0, 0, 1,
    0, 0, 1, 1, 0, 1, 1, 2, 1, 1, 2, 2, 0, 1, 1, 1, 0, 0, 1, 1, 2, 0, 0, 1, 2, 2, 0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 0, 0, 2, 2, 0, 0, 2, 2,
    0, 0, 2, 2, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 2, 2, 2, 0, 2, 2, 2, 0, 0, 0, 1, 0, 0, 0, 1, 2, 2, 2, 1, 2, 2, 2, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 2, 2,
    0, 1, 2, 2, 0, 0, 0, 0, 1, 1, 0, 0, 2, 2, 1, 0, 2, 2, 1, 0, 0, 1, 2, 2, 0, 1, 2, 2, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 1, 2, 1, 1, 2, 2, 2, 2, 2, 2,
    0, 1, 1, 0, 1, 2, 2, 1, 1, 2, 2, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 2, 2, 1, 1, 2, 2, 1, 0, 0, 2, 2, 1, 1, 0, 2, 1, 1, 0, 2, 0, 0, 2, 2, 0, 1, 1, 0,
    0, 1, 1, 0, 2, 0, 0, 2, 2, 2, 2, 2, 0, 0, 1, 1, 0, 1, 2, 2, 0, 1, 2, 2, 0, 0, 1, 1, 0, 0, 0, 0, 2, 0, 0, 0, 2, 2, 1, 1, 2, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 2,
    1, 1, 2, 2, 1, 2, 2, 2, 0, 2, 2, 2, 0, 0, 2, 2, 0, 0, 1, 2, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 2, 0, 0, 2, 2, 0, 2, 2, 2, 0, 1, 2, 0, 0, 1, 2, 0, 0, 1, 2, 0,
    0, 1, 2, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 0, 0, 0, 0, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 0, 1, 2, 0, 2, 0, 1, 2, 1, 2, 0, 1, 0, 1, 2, 0,
    0, 0, 1, 1, 2, 2, 0, 0, 1, 1, 2, 2, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0,
    0, 0, 0, 0, 2, 1, 2, 1, 2, 1, 2, 1, 0, 0, 2, 2, 1, 1, 2, 2, 0, 0, 2, 2, 1, 1, 2, 2, 0, 0, 2, 2, 0, 0, 1, 1, 0, 0, 2, 2, 0, 0, 1, 1, 0, 2, 2, 0, 1, 2, 2, 1,
    0, 2, 2, 0, 1, 2, 2, 1, 0, 1, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 0, 1, 0, 1, 0, 0, 0, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
    2, 2, 2, 2, 0, 2, 2, 2, 0, 1, 1, 1, 0, 2, 2, 2, 0, 1, 1, 1, 0, 0, 0, 2, 1, 1, 1, 2, 0, 0, 0, 2, 1, 1, 1, 2, 0, 0, 0, 0, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2,
    0, 2, 2, 2, 0, 1, 1, 1, 0, 1, 1, 1, 0, 2, 2, 2, 0, 0, 0, 2, 1, 1, 1, 2, 1, 1, 1, 2, 0, 0, 0, 2, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 2, 2, 2, 2, 0, 0, 0, 0,
    0, 0, 0, 0, 2, 1, 1, 2, 2, 1, 1, 2, 0, 1, 1, 0, 0, 1, 1, 0, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 2, 2, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 2, 2, 0, 0, 2, 2, 1, 1, 2, 2,
    1, 1, 2, 2, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 1, 2, 0, 0, 0, 2, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 1, 0, 2, 2, 2, 1, 2, 2, 2, 0, 2, 2, 2,
    1, 2, 2, 2, 0, 1, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 1, 1, 1, 2, 0, 1, 1, 2, 2, 0, 1, 2, 2, 2, 0,
];

// Anchor pixels carry one fewer index bit; subset 0's anchor is always pixel 0. The
// tables give the non-zero subsets' anchors per partition set.
pub(crate) const ANCHOR2: [u8; 64] = [
    15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 2, 8, 2, 2, 8, 8, 15, 2, 8, 2, 2, 8, 8, 2, 2, 15, 15, 6, 8, 2, 8, 15, 15, 2, 8, 2, 2,
    2, 15, 15, 6, 6, 2, 6, 8, 15, 15, 2, 2, 15, 15, 15, 15, 15, 2, 2, 15,
];

pub(crate) const ANCHOR3_SECOND: [u8; 64] = [
    3, 3, 15, 15, 8, 3, 15, 15, 8, 8, 6, 6, 6, 5, 3, 3, 3, 3, 8, 15, 3, 3, 6, 10, 5, 8, 8, 6, 8, 5, 15, 15, 8, 15, 3, 5, 6, 10, 8, 15, 15, 3, 15, 5, 15, 15,
    15, 15, 3, 15, 5, 5, 5, 8, 5, 10, 5, 10, 8, 13, 15, 12, 3, 3,
];

pub(crate) const ANCHOR3_THIRD: [u8; 64] = [
    15, 8, 8, 3, 15, 15, 3, 8, 15, 15, 15, 15, 15, 15, 15, 8, 15, 8, 15, 3, 15, 8, 15, 8, 3, 15, 6, 10, 15, 15, 10, 8, 15, 3, 15, 10, 10, 8, 9, 10, 6, 15, 8,
    15, 3, 6, 6, 8, 15, 3, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 3, 15, 15, 8,
];

/// One 16-byte BC7 block to sixteen RGBA texels. A block whose mode bits are all zero is
/// reserved by the spec and decodes as opaque black.
pub(crate) fn decode_bc7(block: &[u8]) -> Block {
    let mut rgba: Block = [[0, 0, 0, 255]; 16];
    if block.len() < 16 {
        return rgba;
    }
    let mut bits = Bits::new(&block[..16]);

    let mut mode: i32 = -1;
    for i in 0..8 {
        if bits.lo & (1 << i) != 0 {
            mode = i;
            break;
        }
    }
    if mode < 0 {
        return rgba;
    }
    let mode = mode as usize;
    bits.ofs = mode as u32 + 1;

    let ns = N_SUBSETS[mode] as usize;
    let partition = bits.read(u32::from(PARTITION_BITS[mode])) as usize;
    let rotation = bits.read(u32::from(ROTATION_BITS[mode])) as u8;
    let index_sel = if mode == 4 { bits.read(1) as u8 } else { 0 };

    // Endpoints: R, G, B in order, each subset × both endpoints, then alpha.
    let mut ep = [[[0u8; 4]; 2]; 3];
    for c in 0..3 {
        for e in ep.iter_mut().take(ns) {
            for v in e.iter_mut() {
                v[c] = bits.read(u32::from(COLOR_BITS[mode])) as u8;
            }
        }
    }
    if ALPHA_BITS[mode] > 0 {
        for e in ep.iter_mut().take(ns) {
            for v in e.iter_mut() {
                v[3] = bits.read(u32::from(ALPHA_BITS[mode])) as u8;
            }
        }
    } else {
        for e in ep.iter_mut().take(ns) {
            e[0][3] = 255;
            e[1][3] = 255;
        }
    }

    if EP_BITS[mode] != 0 {
        // A per-endpoint P-bit, LSB-appended to every component the mode carries (alpha
        // too when present — modes 0/3 leave alpha opaque).
        let comps = if ALPHA_BITS[mode] > 0 { 4 } else { 3 };
        for i in 0..ns * 2 {
            let p = bits.read(1) as u8;
            for v in ep[i / 2][i % 2].iter_mut().take(comps) {
                *v = (*v << 1) | p;
            }
        }
    } else if SP_BITS[mode] != 0 {
        // Mode 1: one shared P-bit per subset, applied to both endpoints.
        for e in ep.iter_mut().take(ns) {
            let p = bits.read(1) as u8;
            for v in e.iter_mut() {
                for vc in v.iter_mut().take(3) {
                    *vc = (*vc << 1) | p;
                }
            }
        }
    }

    // Scale each component to 8 bits by replicating the top bits; modes ≤ 3 get opaque
    // alpha.
    let color_prec = COLOR_BITS[mode] + u8::from(EP_BITS[mode] != 0 || SP_BITS[mode] != 0);
    let alpha_prec = if ALPHA_BITS[mode] > 0 { ALPHA_BITS[mode] + EP_BITS[mode] } else { 0 };
    let expand = |v: u8, prec: u8| -> u8 { ((u32::from(v) << (8 - prec)) | (u32::from(v) >> (2 * prec - 8))) as u8 };
    for e in ep.iter_mut().take(ns) {
        for v in e.iter_mut() {
            for vc in v.iter_mut().take(3) {
                *vc = expand(*vc, color_prec);
            }
            if alpha_prec > 0 {
                v[3] = expand(v[3], alpha_prec);
            }
        }
    }

    // Per-pixel subset assignment and anchors.
    let mut subset = [0u8; 16];
    for (i, s) in subset.iter_mut().enumerate() {
        *s = match ns {
            2 => P2[partition * 16 + i],
            3 => P3[partition * 16 + i],
            _ => 0,
        };
    }
    let mut anchor = [0u8; 3];
    if ns == 2 {
        anchor[1] = ANCHOR2[partition];
    } else if ns == 3 {
        anchor[1] = ANCHOR3_SECOND[partition];
        anchor[2] = ANCHOR3_THIRD[partition];
    }

    // Index streams. Primary has INDEX_BITS bits (anchors one fewer), secondary
    // INDEX_BITS2. Mode 4's index-select swaps which stream drives colour vs alpha.
    let mut primary = [0u8; 16];
    let mut secondary = [0u8; 16];
    for (i, p) in primary.iter_mut().enumerate() {
        let n = u32::from(INDEX_BITS[mode] - u8::from(i == anchor[subset[i] as usize] as usize));
        *p = bits.read(n) as u8;
    }
    if INDEX_BITS2[mode] > 0 {
        for (i, s) in secondary.iter_mut().enumerate() {
            let n = u32::from(INDEX_BITS2[mode] - u8::from(i == anchor[subset[i] as usize] as usize));
            *s = bits.read(n) as u8;
        }
    }
    let primary_is_alpha = mode == 4 && index_sel == 1;
    let mut color_index = [0u8; 16];
    let mut alpha_index = [0u8; 16];
    for i in 0..16 {
        color_index[i] = if primary_is_alpha { secondary[i] } else { primary[i] };
        alpha_index[i] = if INDEX_BITS2[mode] > 0 { if primary_is_alpha { primary[i] } else { secondary[i] } } else { primary[i] };
    }
    // The interpolation weights key off the width of the index stream feeding each
    // channel: swapped mode 4 puts colour on the secondary set.
    let color_bits = if primary_is_alpha { INDEX_BITS2[mode] } else { INDEX_BITS[mode] };
    let alpha_bits = if primary_is_alpha {
        INDEX_BITS[mode]
    } else if INDEX_BITS2[mode] > 0 {
        INDEX_BITS2[mode]
    } else {
        INDEX_BITS[mode]
    };

    for i in 0..16 {
        let s = subset[i] as usize;
        let mut c = [
            interpolate(ep[s][0][0], ep[s][1][0], color_index[i], color_bits),
            interpolate(ep[s][0][1], ep[s][1][1], color_index[i], color_bits),
            interpolate(ep[s][0][2], ep[s][1][2], color_index[i], color_bits),
            interpolate(ep[s][0][3], ep[s][1][3], alpha_index[i], alpha_bits),
        ];
        // Rotation swaps alpha into a colour channel (A→R/G/B per index).
        match rotation {
            1 => c.swap(0, 3),
            2 => c.swap(1, 3),
            3 => c.swap(2, 3),
            _ => {}
        }
        rgba[i] = c;
    }
    rgba
}
