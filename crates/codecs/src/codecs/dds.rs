//! DirectDraw Surface (DDS) textures, after Microsoft's public DDS documentation ("DDS",
//! "DDS_HEADER", "DDS_PIXELFORMAT", "DDS_HEADER_DXT10").
//!
//! Reading: a 2D texture, its top mip level or the whole chain, either uncompressed (8 to 32
//! bits per pixel described by channel masks, RGB or luminance, an alpha-only A8, or 16/32-bit
//! float RGBA) or block-compressed BC1–BC5 (DXT1/DXT3/DXT5, ATI1/BC4, ATI2/BC5, signed SNORM
//! forms included), with or without the DX10 header. The premultiplied DXT2/DXT4 and GIMP's
//! AEXP and YCoCg BC3 variants decode to straight RGB. Cubemaps, texture arrays, volume
//! textures and BC7 are refused rather than decoded wrongly.
//!
//! Writing: a 2D texture, uncompressed (BGRA or BGRX, luminance with or without alpha) or
//! BC1/BC2/BC3, optionally with its full mip chain. The legacy header describes all of these,
//! so no DX10 header is written; formats only DXGI can name (BC4–BC7, sRGB) aren't written.

use std::borrow::Cow;

use half::f16;

use super::dds_bc::{self, Block, Mask};
use crate::SampleType;
use crate::error::CodecError;
use crate::fidelity::Plan;
use crate::format::Format;
use crate::image::{ChannelLayout, Image};
use crate::options::{DdsCompression, EncodeOptions, Limits};

/// `DDS ` plus the 124-byte DDS_HEADER.
const HEADER: usize = 128;
/// DDS_HEADER_DXT10, present when the FourCC is `DX10`.
const DX10_HEADER: usize = 20;

const DDSD_DEPTH: u32 = 0x80_0000;
const DDPF_ALPHAPIXELS: u32 = 0x1;
const DDPF_ALPHA: u32 = 0x2;
const DDPF_FOURCC: u32 = 0x4;
const DDPF_RGB: u32 = 0x40;
const DDPF_LUMINANCE: u32 = 0x2_0000;
const DDSCAPS2_CUBEMAP: u32 = 0x200;
const DDSCAPS2_VOLUME: u32 = 0x20_0000;
const DIMENSION_TEXTURE2D: u32 = 3;
const MISC_TEXTURECUBE: u32 = 0x4;

fn bad(reason: impl std::fmt::Display) -> CodecError {
    CodecError::malformed(Format::Dds, reason)
}
fn unsupported(reason: impl Into<String>) -> CodecError {
    CodecError::unsupported(Format::Dds, reason)
}
fn le32(b: &[u8], at: usize) -> Result<u32, CodecError> {
    let bytes = b.get(at..at + 4).and_then(|s| <[u8; 4]>::try_from(s).ok()).ok_or_else(|| bad("truncated header"))?;
    Ok(u32::from_le_bytes(bytes))
}

/// A FourCC as text when it is printable, else as the number legacy writers store there.
fn fourcc_name(fourcc: u32) -> String {
    let bytes = fourcc.to_le_bytes();
    if bytes.iter().all(|c| c.is_ascii_graphic() || *c == b' ') { bytes.iter().map(|&c| char::from(c)).collect() } else { fourcc.to_string() }
}

/// One channel of an uncompressed pixel: where it sits and its largest value.
#[derive(Clone, Copy)]
struct Field {
    shift: u32,
    max: u32,
}

impl Field {
    /// `None` for an empty mask; an error for one with gaps or wider than the pixel.
    fn new(mask: u32, bits: u32) -> Result<Option<Field>, CodecError> {
        if mask == 0 {
            return Ok(None);
        }
        let shift = mask.trailing_zeros();
        let max = mask >> shift;
        if max & max.wrapping_add(1) != 0 || (bits < 32 && mask >> bits != 0) {
            return Err(bad(format!("channel mask {mask:#x} is not a contiguous field of a {bits}-bit pixel")));
        }
        Ok(Some(Field { shift, max }))
    }

    fn get(self, pixel: u32) -> u8 {
        let v = u64::from((pixel >> self.shift) & self.max);
        let max = u64::from(self.max);
        ((v * 255 + max / 2) / max) as u8
    }
}

#[derive(Clone, Copy)]
enum Encoding {
    /// Uncompressed: bytes per pixel, then red (or luminance), green, blue and alpha.
    Masks {
        bytes: usize,
        gray: bool,
        fields: [Option<Field>; 4],
    },
    /// RGBA, four half floats per pixel (D3DFMT_A16B16G16R16F, DXGI R16G16B16A16_FLOAT).
    Float16,
    /// RGBA, four floats per pixel (D3DFMT_A32B32G32R32F, DXGI R32G32B32A32_FLOAT).
    Float32,
    Bc1,
    Bc2,
    Bc3,
    /// One interpolated channel; `signed` is the SNORM form (BC4S).
    Bc4 {
        signed: bool,
    },
    /// Two interpolated channels; `signed` is the SNORM form (BC5S). Decodes as RGB with
    /// black blue, the normal-map convention.
    Bc5 {
        signed: bool,
    },
    /// BC2's blocks, but its RGB is premultiplied by alpha (DXT2).
    Dxt2,
    /// BC3's blocks, but its RGB is premultiplied by alpha (DXT4).
    Dxt4,
    /// A BC3 file GIMP's DDS plug-in writes for `AEXP`: RGB stores colour premultiplied by
    /// the interpolated "alpha" channel, which is the exponent's scale; output is opaque.
    Aexp,
    /// GIMP's `YCG1`: BC3 carrying YCoCg in its A, R and G channels.
    Ycg1,
    /// GIMP's `YCG2`: `YCG1` with the chroma channels scaled by B.
    Ycg2,
    /// `DDPF_ALPHA`: a single alpha channel, no colour at all (A8).
    Alpha {
        bytes: usize,
        field: Field,
    },
}

impl Encoding {
    /// An alpha-only pixel (`DDPF_ALPHA`): decodes as RGBA with black colour so the alpha
    /// data is intact — File › Open can route it to a document channel.
    fn alpha(bits: u32, mask: u32) -> Result<Encoding, CodecError> {
        if !matches!(bits, 8 | 16 | 24 | 32) {
            return Err(bad(format!("{bits} bits per pixel is not 8, 16, 24 or 32")));
        }
        let field = Field::new(mask, bits)?.ok_or_else(|| bad("the alpha pixel format has no channel mask"))?;
        Ok(Encoding::Alpha { bytes: bits as usize / 8, field })
    }

    fn masks(bits: u32, masks: [u32; 4], gray: bool) -> Result<Encoding, CodecError> {
        if !matches!(bits, 8 | 16 | 24 | 32) {
            return Err(bad(format!("{bits} bits per pixel is not 8, 16, 24 or 32")));
        }
        let mut fields = [None; 4];
        for (field, mask) in fields.iter_mut().zip(masks) {
            *field = Field::new(mask, bits)?;
        }
        let colour = if gray { fields[0].is_some() } else { fields[..3].iter().any(Option::is_some) };
        if !colour {
            return Err(bad("the pixel format has no colour channel masks"));
        }
        Ok(Encoding::Masks { bytes: bits as usize / 8, gray, fields })
    }

    /// The compression a save could write this encoding back as; `None` when the file's
    /// form isn't writable (premultiplied, GIMP's variants, float or signed). `alpha_pixels`
    /// is the header's `DDPF_ALPHAPIXELS` promise — a DXT1 carrying it reads back `bc1a`.
    fn compression(&self, alpha_pixels: bool) -> Option<DdsCompression> {
        Some(match self {
            Encoding::Masks { .. } => DdsCompression::None,
            Encoding::Alpha { .. } => DdsCompression::A8,
            Encoding::Bc1 if alpha_pixels => DdsCompression::Bc1a,
            Encoding::Bc1 => DdsCompression::Bc1,
            Encoding::Bc2 => DdsCompression::Bc2,
            Encoding::Bc3 => DdsCompression::Bc3,
            Encoding::Bc4 { signed: false } => DdsCompression::Bc4,
            Encoding::Bc5 { signed: false } => DdsCompression::Bc5,
            _ => return None,
        })
    }

    /// The image's sample type: float DDS encodings keep their depth; the rest is 8-bit.
    fn sample(&self) -> SampleType {
        match self {
            Encoding::Float16 => SampleType::F16,
            Encoding::Float32 => SampleType::F32,
            _ => SampleType::U8,
        }
    }

    /// Whether the encoding carries an alpha channel the decode reads. BC1 counts: it can
    /// hold 1-bit transparency. BC4/BC5's channels are data (normal maps), not alpha, and
    /// AEXP and YCG2 decode opaque.
    fn has_alpha(&self) -> bool {
        match self {
            Encoding::Masks { fields, .. } => fields[3].is_some(),
            Encoding::Bc4 { .. } | Encoding::Bc5 { .. } | Encoding::Aexp | Encoding::Ycg2 => false,
            _ => true,
        }
    }

    fn layout(&self) -> ChannelLayout {
        match self {
            Encoding::Masks { gray: true, fields, .. } if fields[3].is_some() => ChannelLayout::GrayA,
            Encoding::Masks { gray: true, .. } => ChannelLayout::Gray,
            Encoding::Masks { fields, .. } if fields[3].is_none() => ChannelLayout::Rgb,
            Encoding::Bc4 { .. } => ChannelLayout::Gray,
            Encoding::Bc5 { .. } | Encoding::Aexp | Encoding::Ycg2 => ChannelLayout::Rgb,
            Encoding::Ycg1 => ChannelLayout::Rgba,
            _ => ChannelLayout::Rgba,
        }
    }

    /// `DXGI_FORMAT_*_SRGB` names that mark the texture sRGB.
    fn dxgi_is_srgb(dxgi: u32) -> bool {
        matches!(dxgi, 29 | 72 | 75 | 78 | 91 | 93 | 99)
    }

    /// Stored row stride and byte size of the top mip level (block rows for BC).
    fn level_layout(&self, width: u32, height: u32, pitch: Option<u32>) -> Result<(usize, usize), CodecError> {
        let overflow = || bad("pixel data size overflow");
        let (w, h) = (usize::try_from(width).map_err(|_| overflow())?, usize::try_from(height).map_err(|_| overflow())?);
        let (row_bytes, rows) = match self {
            Encoding::Masks { bytes, .. } | Encoding::Alpha { bytes, .. } => {
                let packed = w.checked_mul(*bytes).ok_or_else(overflow)?;
                let row_bytes = match pitch {
                    Some(pitch) => usize::try_from(pitch).map_err(|_| overflow())?,
                    None => packed,
                };
                if row_bytes < packed {
                    return Err(bad(format!("row pitch {row_bytes} is smaller than the {packed}-byte pixel row")));
                }
                (row_bytes, h)
            }
            Encoding::Float16 => (w.checked_mul(8).ok_or_else(overflow)?, h),
            Encoding::Float32 => (w.checked_mul(16).ok_or_else(overflow)?, h),
            _ => {
                let block = if matches!(self, Encoding::Bc1 | Encoding::Bc4 { .. }) { 8 } else { 16 };
                (w.div_ceil(4).checked_mul(block).ok_or_else(overflow)?, h.div_ceil(4))
            }
        };
        let size = row_bytes.checked_mul(rows).ok_or_else(overflow)?;
        Ok((row_bytes, size))
    }
}

/// The legacy pixel format: a FourCC, or channel masks.
fn legacy_encoding(b: &[u8]) -> Result<Encoding, CodecError> {
    let (flags, fourcc, bits) = (le32(b, 80)?, le32(b, 84)?, le32(b, 88)?);
    let alpha = if flags & DDPF_ALPHAPIXELS != 0 { le32(b, 104)? } else { 0 };
    if flags & DDPF_FOURCC != 0 {
        // GIMP's DDS plug-in writes AEXP/YCG1/YCG2 files claiming "DXT5" and stashes the real
        // name in reserved header fields after a "GIMP-DDS" marker (file offsets 32–39, the
        // variant's FourCC at 44).
        if &fourcc.to_le_bytes() == b"DXT5" && b.get(32..40) == Some(b"GIMP-DDS") {
            match &le32(b, 44).unwrap_or(0).to_le_bytes() {
                b"AEXP" => return Ok(Encoding::Aexp),
                b"YCG1" => return Ok(Encoding::Ycg1),
                b"YCG2" => return Ok(Encoding::Ycg2),
                _ => {}
            }
        }
        return match &fourcc.to_le_bytes() {
            b"DXT1" => Ok(Encoding::Bc1),
            b"DXT3" => Ok(Encoding::Bc2),
            b"DXT5" | b"RXGB" => Ok(Encoding::Bc3),
            b"DXT2" => Ok(Encoding::Dxt2),
            b"DXT4" => Ok(Encoding::Dxt4),
            b"ATI1" | b"BC4U" => Ok(Encoding::Bc4 { signed: false }),
            b"BC4S" => Ok(Encoding::Bc4 { signed: true }),
            b"ATI2" | b"BC5U" | b"A2XY" => Ok(Encoding::Bc5 { signed: false }),
            b"BC5S" => Ok(Encoding::Bc5 { signed: true }),
            b"AEXP" => Ok(Encoding::Aexp),
            b"YCG1" => Ok(Encoding::Ycg1),
            b"YCG2" => Ok(Encoding::Ycg2),
            _ if fourcc == 113 => Ok(Encoding::Float16),
            _ if fourcc == 116 => Ok(Encoding::Float32),
            _ => Err(unsupported(format!("compression `{}` is not supported", fourcc_name(fourcc)))),
        };
    }
    if flags & DDPF_RGB != 0 {
        return Encoding::masks(bits, [le32(b, 92)?, le32(b, 96)?, le32(b, 100)?, alpha], false);
    }
    if flags & DDPF_LUMINANCE != 0 {
        return Encoding::masks(bits, [le32(b, 92)?, 0, 0, alpha], true);
    }
    if flags & DDPF_ALPHA != 0 {
        // Writers disagree on the slot: Microsoft's docs put the mask in R (offset 92),
        // others in A (104).
        let mask = match le32(b, 92)? {
            0 => le32(b, 104)?,
            mask => mask,
        };
        return Encoding::alpha(bits, mask);
    }
    Err(unsupported(format!("pixel format flags {flags:#x} are not RGB, luminance, alpha or a supported compression")))
}

/// The DX10 header's DXGI format, refusing anything but a single 2D texture.
fn dx10_encoding(b: &[u8]) -> Result<Encoding, CodecError> {
    let (dxgi, dimension, misc, array) = (le32(b, HEADER)?, le32(b, HEADER + 4)?, le32(b, HEADER + 8)?, le32(b, HEADER + 12)?);
    if misc & !MISC_TEXTURECUBE != 0 {
        return Err(unsupported(format!("unsupported resource misc flags {misc:#x}")));
    }
    if misc & MISC_TEXTURECUBE != 0 {
        return Err(unsupported("cubemaps are not supported"));
    }
    if dimension != DIMENSION_TEXTURE2D {
        return Err(unsupported(format!("only 2D textures are supported (resource dimension {dimension})")));
    }
    if array > 1 {
        return Err(unsupported(format!("texture arrays are not supported ({array} textures)")));
    }
    const RGBA: [u32; 4] = [0xff, 0xff00, 0xff_0000, 0xff00_0000];
    const BGRA: [u32; 4] = [0xff_0000, 0xff00, 0xff, 0xff00_0000];
    const BGRX: [u32; 4] = [0xff_0000, 0xff00, 0xff, 0];
    const R10A2: [u32; 4] = [0x3ff, 0xffc00, 0x3ff0_0000, 0xc000_0000];
    const RG8: [u32; 4] = [0xff, 0xff00, 0, 0];
    const B565: [u32; 4] = [0xf800, 0x7e0, 0x1f, 0];
    const B5A1: [u32; 4] = [0x7c00, 0x3e0, 0x1f, 0x8000];
    const B4A4: [u32; 4] = [0xf00, 0xf0, 0xf, 0xf000];
    match dxgi {
        28 | 29 => Encoding::masks(32, RGBA, false),
        87 | 90 | 91 => Encoding::masks(32, BGRA, false),
        88 | 92 | 93 => Encoding::masks(32, BGRX, false),
        24 => Encoding::masks(32, R10A2, false),
        49 => Encoding::masks(16, RG8, false),
        61 => Encoding::masks(8, [0xff, 0, 0, 0], true),
        85 => Encoding::masks(16, B565, false),
        86 => Encoding::masks(16, B5A1, false),
        115 => Encoding::masks(16, B4A4, false),
        10 => Ok(Encoding::Float16),
        2 => Ok(Encoding::Float32),
        70..=72 => Ok(Encoding::Bc1),
        73..=75 => Ok(Encoding::Bc2),
        76..=78 => Ok(Encoding::Bc3),
        79..=81 => Ok(Encoding::Bc4 { signed: dxgi == 81 }),
        82..=84 => Ok(Encoding::Bc5 { signed: dxgi == 84 }),
        _ => Err(unsupported(format!("DXGI format {dxgi} is not supported"))),
    }
}

/// The parsed header: the encoding, where its pixel data starts, the top level's dimensions,
/// the declared mip count, the stored row pitch (which is the top level's only), whether the
/// pixel format promises alpha pixels (a DXT1 reads back `bc1a`) and the DX10 header's sRGB
/// mark.
struct Parsed<'a> {
    encoding: Encoding,
    data: &'a [u8],
    width: u32,
    height: u32,
    levels: u32,
    pitch: Option<u32>,
    alpha_pixels: bool,
    srgb: bool,
}

impl Parsed<'_> {
    /// The mip count that can actually exist for this size — the declared field is hostile
    /// input, and the chain ends at 1×1.
    fn mip_count(&self) -> u32 {
        self.levels.min(self.width.max(self.height).max(1).ilog2() + 1)
    }
}

fn parse(b: &[u8]) -> Result<Parsed<'_>, CodecError> {
    if !b.starts_with(b"DDS ") {
        return Err(bad("missing the `DDS ` signature"));
    }
    if le32(b, 4)? != 124 {
        return Err(bad("header size is not 124"));
    }
    let (flags, height, width, depth, caps2) = (le32(b, 8)?, le32(b, 12)?, le32(b, 16)?, le32(b, 24)?, le32(b, 112)?);
    if caps2 & DDSCAPS2_CUBEMAP != 0 {
        return Err(unsupported("cubemaps are not supported"));
    }
    if caps2 & DDSCAPS2_VOLUME != 0 || (flags & DDSD_DEPTH != 0 && depth > 1) {
        return Err(unsupported("volume textures are not supported"));
    }
    let dx10 = le32(b, 80)? & DDPF_FOURCC != 0 && &le32(b, 84)?.to_le_bytes() == b"DX10";
    let (encoding, start) = if dx10 { (dx10_encoding(b)?, HEADER + DX10_HEADER) } else { (legacy_encoding(b)?, HEADER) };
    let levels = if flags & DDSD_MIPMAPCOUNT != 0 { le32(b, 28)?.max(1) } else { 1 };
    // The same header field is a total level size for BC, but a row stride when DDSD_PITCH is set.
    let pitch = if flags & DDSD_PITCH != 0 { Some(le32(b, 20)?) } else { None };
    let alpha_pixels = le32(b, 80)? & DDPF_ALPHAPIXELS != 0;
    let srgb = dx10 && Encoding::dxgi_is_srgb(le32(b, HEADER)?);
    Ok(Parsed { encoding, data: b.get(start..).unwrap_or_default(), width, height, levels, pitch, alpha_pixels, srgb })
}

pub(crate) fn decode(b: &[u8], limits: &Limits) -> Result<Image, CodecError> {
    level_image(&parse(b)?, 0, limits)
}

/// Every declared mip level as an image, largest first; [`decode`] reads the top one. The
/// chain ends at 1×1, so a header claiming more levels than reduce to it reads that many —
/// the field is hostile input, and each level re-walks the ones before it.
pub(crate) fn decode_levels(b: &[u8], limits: &Limits) -> Result<Vec<Image>, CodecError> {
    let p = parse(b)?;
    (0..p.mip_count()).map(|i| level_image(&p, i, limits)).collect()
}

/// The image of mip `index` (`0` is the top level). Levels are stored largest first and,
/// except the top level's stored pitch, tightly packed.
fn level_image(p: &Parsed, index: u32, limits: &Limits) -> Result<Image, CodecError> {
    let mut at = 0usize;
    let (mut w, mut h) = (p.width, p.height);
    for i in 0..index {
        let pitch = if i == 0 { p.pitch } else { None };
        let (_, size) = p.encoding.level_layout(w, h, pitch)?;
        at = at.checked_add(size).ok_or_else(|| bad("pixel data size overflow"))?;
        w = (w / 2).max(1);
        h = (h / 2).max(1);
    }
    let pitch = if index == 0 { p.pitch } else { None };
    let (row_pitch, size) = p.encoding.level_layout(w, h, pitch)?;
    limits.check(w, h, p.encoding.layout(), p.encoding.sample())?;
    let data = p.data.get(at..).and_then(|d| d.get(..size)).ok_or_else(|| bad(format!("truncated pixel data: {size} bytes needed")))?;
    decode_image(p.encoding, data, w, h, row_pitch)
}

/// What File › Open's DDS prompt asks about, read from the header without decoding — and the
/// encoding a save could write back, for the options dialog's match-source defaults.
pub(crate) fn info(b: &[u8]) -> Result<crate::options::DdsInfo, CodecError> {
    let p = parse(b)?;
    Ok(crate::options::DdsInfo {
        width: p.width,
        height: p.height,
        levels: p.levels,
        alpha: p.encoding.has_alpha(),
        compression: p.encoding.compression(p.alpha_pixels),
        srgb: p.srgb,
    })
}

fn decode_image(encoding: Encoding, data: &[u8], width: u32, height: u32, row_pitch: usize) -> Result<Image, CodecError> {
    let layout = encoding.layout();
    let (w, h) = (width as usize, height as usize);
    match encoding {
        Encoding::Float16 => {
            let pixels: Vec<f16> = data.as_chunks::<2>().0.iter().map(|c| f16::from_bits(u16::from_le_bytes(*c))).collect();
            return Image::from_f16(width, height, layout, &pixels);
        }
        Encoding::Float32 => {
            let pixels: Vec<f32> = data.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect();
            return Image::from_f32(width, height, layout, &pixels);
        }
        _ => {}
    }
    let channels = layout.channels();
    let row_size = w.checked_mul(channels).ok_or_else(|| bad("pixel size overflow"))?;
    let total = row_size.checked_mul(h).ok_or_else(|| bad("pixel size overflow"))?;
    let mut pixels = vec![0u8; total];
    match encoding {
        Encoding::Masks { bytes, gray, fields } => {
            for (src_row, dst_row) in data.chunks_exact(row_pitch).zip(pixels.chunks_exact_mut(row_size)) {
                // Stop at the output width, leaving any stored row padding unread.
                for (src, dst) in src_row.chunks_exact(bytes).zip(dst_row.chunks_exact_mut(channels)) {
                    let pixel = src.iter().rev().fold(0u32, |v, &byte| v << 8 | u32::from(byte));
                    let channel = |i: usize| fields[i].map_or(0, |f| f.get(pixel));
                    let values = if gray { [channel(0), channel(3), 0, 0] } else { [channel(0), channel(1), channel(2), channel(3)] };
                    dst.copy_from_slice(&values[..channels]);
                }
            }
        }
        Encoding::Alpha { bytes, field } => {
            for (src_row, dst_row) in data.chunks_exact(row_pitch).zip(pixels.chunks_exact_mut(row_size)) {
                for (src, dst) in src_row.chunks_exact(bytes).zip(dst_row.chunks_exact_mut(channels)) {
                    let pixel = src.iter().rev().fold(0u32, |v, &byte| v << 8 | u32::from(byte));
                    dst.copy_from_slice(&[0, 0, 0, field.get(pixel)][..channels]);
                }
            }
        }
        _ => {
            let block_size = if matches!(encoding, Encoding::Bc1 | Encoding::Bc4 { .. }) { 8 } else { 16 };
            let blocks_x = w.div_ceil(4);
            for (i, block) in data.chunks_exact(block_size).enumerate() {
                let mut texels = match encoding {
                    Encoding::Bc1 => bc1(block, true),
                    Encoding::Bc2 | Encoding::Dxt2 => with_alpha(bc1(block.get(8..).unwrap_or_default(), false), bc2_alpha(block)),
                    Encoding::Bc3 | Encoding::Dxt4 | Encoding::Aexp | Encoding::Ycg1 | Encoding::Ycg2 => {
                        with_alpha(bc1(block.get(8..).unwrap_or_default(), false), bc3_alpha(block))
                    }
                    Encoding::Bc4 { signed } => std::array::from_fn(|t| {
                        let g = bc4_texel(block, t, signed);
                        [g, g, g, 255]
                    }),
                    Encoding::Bc5 { signed } => {
                        std::array::from_fn(|t| [bc4_texel(block, t, signed), bc4_texel(block.get(8..).unwrap_or_default(), t, signed), 0, 255])
                    }
                    _ => continue,
                };
                // The premultiplied and GIMP channel orders become straight RGBA per texel.
                for p in texels.iter_mut() {
                    match encoding {
                        Encoding::Dxt2 | Encoding::Dxt4 => {
                            let a = u32::from(p[3]);
                            for c in p.iter_mut().take(3) {
                                *c = (u32::from(*c) * 255).checked_div(a).map_or(0, |v| v.min(255) as u8);
                            }
                        }
                        Encoding::Aexp => {
                            let a = u32::from(p[3]);
                            for c in p.iter_mut().take(3) {
                                *c = ((u32::from(*c) * a + 1) >> 8) as u8;
                            }
                            p[3] = 255;
                        }
                        Encoding::Ycg1 | Encoding::Ycg2 => *p = ycocg(p, matches!(encoding, Encoding::Ycg2)),
                        _ => {}
                    }
                }
                let (bx, by) = ((i % blocks_x) * 4, (i / blocks_x) * 4);
                for (t, texel) in texels.iter().enumerate() {
                    let (x, y) = (bx + t % 4, by + t / 4);
                    if x < w && y < h {
                        let at = (y * w + x) * channels;
                        if let Some(dst) = pixels.get_mut(at..at + channels) {
                            dst.copy_from_slice(&texel[..channels]);
                        }
                    }
                }
            }
        }
    }
    Image::from_u8(width, height, layout, pixels)
}

/// One texel of a BC4 block, or of either half of a BC5 one: the same interpolated block BC3
/// uses for alpha, signed (SNORM) when `signed`.
fn bc4_texel(block: &[u8], t: usize, signed: bool) -> u8 {
    let byte = |i: usize| block.get(i).copied().unwrap_or(0);
    let palette = if signed { dds_bc::alpha_palette_signed(byte(0) as i8, byte(1) as i8) } else { dds_bc::alpha_palette(byte(0), byte(1)) };
    let bits = u64::from_le_bytes([byte(2), byte(3), byte(4), byte(5), byte(6), byte(7), 0, 0]);
    palette[((bits >> (3 * t)) & 7) as usize]
}

/// One YCoCg texel of GIMP's DDS variants as RGBA: luma sits in the stored alpha channel,
/// orange and green chroma in red and green; `scaled` (YCG2) divides the chroma by a scale
/// in blue, otherwise blue is the output alpha.
fn ycocg(p: &[u8; 4], scaled: bool) -> [u8; 4] {
    const OFFSET: f32 = 0.5 * 256.0 / 255.0;
    let y = f32::from(p[3]) / 255.0;
    let (mut co, mut cg) = (f32::from(p[0]) / 255.0 - OFFSET, f32::from(p[1]) / 255.0 - OFFSET);
    let mut alpha = 255;
    if scaled {
        let s = 1.0 / ((255.0 / 8.0) * (f32::from(p[2]) / 255.0) + 1.0);
        co *= s;
        cg *= s;
    } else {
        alpha = p[2];
    }
    let r = (y + co - cg).clamp(0.0, 1.0);
    let g = (y + cg).clamp(0.0, 1.0);
    let b = (y - co - cg).clamp(0.0, 1.0);
    [(r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8, alpha]
}

/// A BC1 colour block. BC2 and BC3 colour blocks always have four colours; only BC1's
/// `c0 <= c1` order selects three colours and transparent black.
fn bc1(block: &[u8], one_bit_alpha: bool) -> [[u8; 4]; 16] {
    let byte = |i: usize| block.get(i).copied().unwrap_or(0);
    let (c0, c1) = (u16::from_le_bytes([byte(0), byte(1)]), u16::from_le_bytes([byte(2), byte(3)]));
    let palette = dds_bc::colour_palette(c0, c1, c0 > c1 || !one_bit_alpha);
    let indices = u32::from_le_bytes([byte(4), byte(5), byte(6), byte(7)]);
    std::array::from_fn(|t| palette[(indices >> (2 * t)) as usize & 3])
}

/// BC2: four explicit bits of alpha per texel.
fn bc2_alpha(block: &[u8]) -> [u8; 16] {
    let bits = u64::from_le_bytes(std::array::from_fn(|i| block.get(i).copied().unwrap_or(0)));
    std::array::from_fn(|t| ((bits >> (4 * t)) & 0xf) as u8 * 17)
}

/// BC3: two alpha endpoints and a 3-bit index per texel into eight values, or into six plus
/// 0 and 255 when `a0 <= a1`.
fn bc3_alpha(block: &[u8]) -> [u8; 16] {
    let byte = |i: usize| block.get(i).copied().unwrap_or(0);
    let palette = dds_bc::alpha_palette(byte(0), byte(1));
    let bits = u64::from_le_bytes([byte(2), byte(3), byte(4), byte(5), byte(6), byte(7), 0, 0]);
    std::array::from_fn(|t| palette[((bits >> (3 * t)) & 7) as usize])
}

fn with_alpha(mut texels: [[u8; 4]; 16], alpha: [u8; 16]) -> [[u8; 4]; 16] {
    for (texel, a) in texels.iter_mut().zip(alpha) {
        texel[3] = a;
    }
    texels
}

const DDSD_CAPS: u32 = 0x1;
const DDSD_HEIGHT: u32 = 0x2;
const DDSD_WIDTH: u32 = 0x4;
const DDSD_PITCH: u32 = 0x8;
const DDSD_PIXELFORMAT: u32 = 0x1000;
const DDSD_MIPMAPCOUNT: u32 = 0x2_0000;
const DDSD_LINEARSIZE: u32 = 0x8_0000;
const DDSCAPS_COMPLEX: u32 = 0x8;
const DDSCAPS_TEXTURE: u32 = 0x1000;
const DDSCAPS_MIPMAP: u32 = 0x40_0000;

fn encode_err(reason: impl std::fmt::Display) -> CodecError {
    CodecError::encode(Format::Dds, reason)
}

/// How a written texture stores its pixels.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Store {
    /// 8-bit luminance.
    Luminance,
    /// 8-bit luminance, then 8-bit alpha.
    LuminanceAlpha,
    /// Blue, green, red and an unused byte (written as 255, so a reader that takes it for
    /// alpha still sees an opaque texture).
    Bgrx,
    Bgra,
    Bc1,
    Bc2,
    Bc3,
}

impl Store {
    /// The storage for an image of `layout`, as [`crate::fidelity::plan`] chose it.
    fn new(compression: DdsCompression, layout: ChannelLayout) -> Result<Store, CodecError> {
        Ok(match (compression, layout) {
            (DdsCompression::None, ChannelLayout::Gray) => Store::Luminance,
            (DdsCompression::None, ChannelLayout::GrayA) => Store::LuminanceAlpha,
            (DdsCompression::None, ChannelLayout::Rgb) => Store::Bgrx,
            (DdsCompression::None, ChannelLayout::Rgba) => Store::Bgra,
            (DdsCompression::Bc1, ChannelLayout::Rgba) => Store::Bc1,
            (DdsCompression::Bc2, ChannelLayout::Rgba) => Store::Bc2,
            (DdsCompression::Bc3, ChannelLayout::Rgba) => Store::Bc3,
            (_, layout) => return Err(encode_err(format!("{layout:?} pixels can't be written as `{}`", compression.name()))),
        })
    }

    /// Bytes per pixel, or per 4×4 block for a block compression.
    fn unit(self) -> usize {
        match self {
            Store::Luminance => 1,
            Store::LuminanceAlpha => 2,
            Store::Bgrx | Store::Bgra => 4,
            Store::Bc1 => 8,
            Store::Bc2 | Store::Bc3 => 16,
        }
    }

    fn is_block(self) -> bool {
        matches!(self, Store::Bc1 | Store::Bc2 | Store::Bc3)
    }

    /// Bytes of one row of pixels (or of 4×4 blocks) of a level `width` wide.
    fn row_bytes(self, width: usize) -> Option<usize> {
        if self.is_block() { width.div_ceil(4).checked_mul(self.unit()) } else { width.checked_mul(self.unit()) }
    }

    fn level_bytes(self, width: usize, height: usize) -> Option<usize> {
        let rows = if self.is_block() { height.div_ceil(4) } else { height };
        self.row_bytes(width)?.checked_mul(rows)
    }

    /// DDS_PIXELFORMAT: flags, FourCC, bits per pixel and the red, green, blue and alpha masks.
    fn pixel_format(self) -> (u32, [u8; 4], u32, [u32; 4]) {
        match self {
            Store::Luminance => (DDPF_LUMINANCE, [0; 4], 8, [0xff, 0, 0, 0]),
            Store::LuminanceAlpha => (DDPF_LUMINANCE | DDPF_ALPHAPIXELS, [0; 4], 16, [0xff, 0, 0, 0xff00]),
            Store::Bgrx => (DDPF_RGB, [0; 4], 32, [0xff_0000, 0xff00, 0xff, 0]),
            Store::Bgra => (DDPF_RGB | DDPF_ALPHAPIXELS, [0; 4], 32, [0xff_0000, 0xff00, 0xff, 0xff00_0000]),
            Store::Bc1 => (DDPF_FOURCC, *b"DXT1", 0, [0; 4]),
            Store::Bc2 => (DDPF_FOURCC, *b"DXT3", 0, [0; 4]),
            Store::Bc3 => (DDPF_FOURCC, *b"DXT5", 0, [0; 4]),
        }
    }
}

/// Levels in a full mip chain: the larger side halves down to 1.
fn mip_levels(width: u32, height: u32) -> u32 {
    32 - width.max(height).max(1).leading_zeros()
}

fn half((w, h): (usize, usize)) -> (usize, usize) {
    ((w / 2).max(1), (h / 2).max(1))
}

pub(crate) fn encode(image: &Image, plan: Plan, opts: &EncodeOptions) -> Result<Vec<u8>, CodecError> {
    let img = image.converted(plan.layout, SampleType::U8);
    let pixels = img.as_u8().ok_or_else(|| encode_err("the image was not converted to 8 bits per channel"))?;
    let store = Store::new(opts.dds.compression, img.layout())?;
    let (width, height) = img.dimensions();
    let levels = if opts.dds.mipmaps { mip_levels(width, height) } else { 1 };
    let top = (width as usize, height as usize);
    // Every level's size, checked before anything is allocated.
    let mut sizes = Vec::new();
    let mut size = top;
    for _ in 0..levels {
        sizes.push(store.level_bytes(size.0, size.1).ok_or_else(|| encode_err("texture size overflow"))?);
        size = half(size);
    }
    let total = sizes.iter().try_fold(HEADER, |total, &s| total.checked_add(s)).ok_or_else(|| encode_err("texture size overflow"))?;
    let header = header(width, height, store, levels, sizes.first().copied().unwrap_or(0))?;
    let mut out = crate::options::alloc_zeroed::<u8>(total)?;
    out.get_mut(..HEADER).ok_or_else(|| encode_err("output buffer too small"))?.copy_from_slice(&header);

    let channels = img.layout().channels();
    let mut level = Cow::Borrowed(pixels);
    let (mut size, mut at) = (top, HEADER);
    for (i, &bytes) in sizes.iter().enumerate() {
        let dst = out.get_mut(at..at + bytes).ok_or_else(|| encode_err("output buffer too small"))?;
        write_level(store, &level, size, dst)?;
        at += bytes;
        if i + 1 < sizes.len() {
            let next = half(size);
            level = Cow::Owned(downsample(&level, size, next, channels)?);
            size = next;
        }
    }
    Ok(out)
}

/// The `DDS ` signature and DDS_HEADER. `top` is the top level's size in bytes.
fn header(width: u32, height: u32, store: Store, levels: u32, top: usize) -> Result<[u8; HEADER], CodecError> {
    let mut b = [0u8; HEADER];
    let mut put = |at: usize, v: u32| {
        if let Some(d) = b.get_mut(at..at + 4) {
            d.copy_from_slice(&v.to_le_bytes());
        }
    };
    // Uncompressed: the pitch of one row. Compressed: the whole top level (its "linear size").
    let (size_flag, pitch_or_size) = if store.is_block() { (DDSD_LINEARSIZE, Some(top)) } else { (DDSD_PITCH, store.row_bytes(width as usize)) };
    let pitch_or_size =
        pitch_or_size.and_then(|v| u32::try_from(v).ok()).ok_or_else(|| encode_err(format!("a {width}x{height} level is too large for a DDS header")))?;
    let mipmapped = levels > 1;
    put(4, 124);
    put(8, DDSD_CAPS | DDSD_HEIGHT | DDSD_WIDTH | DDSD_PIXELFORMAT | size_flag | if mipmapped { DDSD_MIPMAPCOUNT } else { 0 });
    put(12, height);
    put(16, width);
    put(20, pitch_or_size);
    put(28, levels);
    let (flags, fourcc, bits, masks) = store.pixel_format();
    put(76, 32);
    put(80, flags);
    put(84, u32::from_le_bytes(fourcc));
    put(88, bits);
    for (i, mask) in masks.into_iter().enumerate() {
        put(92 + 4 * i, mask);
    }
    put(108, DDSCAPS_TEXTURE | if mipmapped { DDSCAPS_COMPLEX | DDSCAPS_MIPMAP } else { 0 });
    b[..4].copy_from_slice(b"DDS ");
    Ok(b)
}

/// Runs `f` on each `len`-byte row of `out` with its index, in parallel on native targets.
fn rows(out: &mut [u8], len: usize, f: impl Fn(usize, &mut [u8]) + Send + Sync) {
    #[cfg(not(target_arch = "wasm32"))]
    {
        use rayon::prelude::*;
        out.par_chunks_mut(len.max(1)).enumerate().for_each(|(i, row)| f(i, row));
    }
    #[cfg(target_arch = "wasm32")]
    for (i, row) in out.chunks_mut(len.max(1)).enumerate() {
        f(i, row);
    }
}

/// Writes one level of `width` × `height` pixels (in the image's layout) into `dst`.
fn write_level(store: Store, pixels: &[u8], (width, height): (usize, usize), dst: &mut [u8]) -> Result<(), CodecError> {
    let row_bytes = store.row_bytes(width).ok_or_else(|| encode_err("texture size overflow"))?;
    match store {
        Store::Luminance | Store::LuminanceAlpha => {
            // The image's gray (and alpha) samples, in the same order.
            let src = pixels.get(..dst.len()).ok_or_else(|| encode_err("the level is smaller than its size"))?;
            dst.copy_from_slice(src);
        }
        Store::Bgrx | Store::Bgra => {
            let channels = if store == Store::Bgra { 4 } else { 3 };
            rows(dst, row_bytes, |y, row| {
                let src = pixels.get(y * width * channels..).unwrap_or_default();
                for (d, s) in row.as_chunks_mut::<4>().0.iter_mut().zip(src.chunks_exact(channels)) {
                    let byte = |i: usize| s.get(i).copied().unwrap_or(255);
                    *d = [byte(2), byte(1), byte(0), byte(3)];
                }
            });
        }
        Store::Bc1 | Store::Bc2 | Store::Bc3 => {
            let (texels, _) = pixels.as_chunks::<4>();
            rows(dst, row_bytes, |by, row| {
                for (bx, d) in row.chunks_exact_mut(store.unit()).enumerate() {
                    let (block, inside) = gather(texels, (width, height), bx, by);
                    let encoded: &[u8] = match store {
                        Store::Bc1 => &dds_bc::encode_bc1(&block, inside),
                        Store::Bc2 => &dds_bc::encode_bc2(&block, inside),
                        _ => &dds_bc::encode_bc3(&block, inside),
                    };
                    for (d, s) in d.iter_mut().zip(encoded) {
                        *d = *s;
                    }
                }
            });
        }
    }
    Ok(())
}

/// Block (`bx`, `by`)'s texels, and which of them lie inside the image.
fn gather(texels: &[[u8; 4]], (width, height): (usize, usize), bx: usize, by: usize) -> (Block, Mask) {
    let mut block = [[0u8; 4]; 16];
    let mut inside: Mask = 0;
    for (t, texel) in block.iter_mut().enumerate() {
        let (x, y) = (bx * 4 + t % 4, by * 4 + t / 4);
        if x < width
            && y < height
            && let Some(p) = texels.get(y * width + x)
        {
            *texel = *p;
            inside |= 1 << t;
        }
    }
    (block, inside)
}

/// The next mip level: each pixel the average of the 2×2 pixels it covers, or of 2×3, 3×2 or
/// 3×3 along the last column and row of an odd-sized level, so no pixel is dropped.
fn downsample(src: &[u8], (w, h): (usize, usize), (nw, nh): (usize, usize), channels: usize) -> Result<Vec<u8>, CodecError> {
    let len = nw.checked_mul(nh).and_then(|n| n.checked_mul(channels)).ok_or_else(|| encode_err("mip level size overflow"))?;
    let mut dst = crate::options::alloc_zeroed::<u8>(len)?;
    let span = |i: usize, n: usize, size: usize| 2 * i..=if i + 1 == n { size.saturating_sub(1) } else { 2 * i + 1 };
    rows(&mut dst, nw * channels, |y, row| {
        let ys = span(y, nh, h);
        for (x, px) in row.chunks_exact_mut(channels.max(1)).enumerate() {
            let xs = span(x, nw, w);
            for (c, v) in px.iter_mut().enumerate() {
                let (mut sum, mut n) = (0u32, 0u32);
                for sy in ys.clone() {
                    for sx in xs.clone() {
                        if let Some(&s) = src.get((sy * w + sx) * channels + c) {
                            sum += u32::from(s);
                            n += 1;
                        }
                    }
                }
                *v = ((sum + n / 2) / n.max(1)) as u8;
            }
        }
    });
    Ok(dst)
}
