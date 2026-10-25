//! DirectDraw Surface (DDS) textures, after Microsoft's public DDS documentation ("DDS",
//! "DDS_HEADER", "DDS_PIXELFORMAT", "DDS_HEADER_DXT10").
//!
//! Reading: the top mip level of a 2D texture, either uncompressed (8 to 32 bits per pixel
//! described by channel masks, RGB or luminance, or 16/32-bit float) or block-compressed
//! BC1–BC5 and BC7, with or without the DX10 header. The premultiplied DXT2/DXT4 and GIMP's
//! AEXP and YCoCg BC3 variants decode to straight RGB. Cubemaps read and write as the six
//! faces `+X -X +Y -Y +Z -Z`, each face's mip chain contiguous (DDS surface order is
//! face-major); a legacy header's `caps2` face bits can mark a partial cube, a DX10 header's
//! `MISC_TEXTURECUBE` means all six. Texture arrays and volume textures are still refused.
//!
//! Writing: a 2D texture or cubemap, uncompressed (BGRA or BGRX, luminance with or without
//! alpha) or BC1–BC5 and BC7, optionally with its full mip chain. BC4, BC5 and BC7 have no
//! legacy pixel format name, so they are written with the DX10 header and their DXGI format.
//!
//! Written from Microsoft's public specification and the conventions the tools' files show;
//! the GIMP variants' reserved-field layout is known from its plug-in's public source and
//! reproduced without its code. Decodes are verified against FFmpeg, ImageMagick and
//! DirectXTex output. The BC7 encoder is a permissively licensed port — see `dds_bc7_enc.rs`
//! and `ATTRIBUTION.md`.

use std::borrow::Cow;

use super::dds_bc::{self, Block, Mask};
use super::dds_bc7;
use super::dds_bc7_enc;
use crate::SampleType;
use crate::error::CodecError;
use crate::f16;
use crate::fidelity::Plan;
use crate::format::Format;
use crate::image::{ChannelLayout, Image};
use crate::options::{DdsCompression, DdsCubeMap, DdsMipFilter, DdsQuality, EncodeOptions, Limits};

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
const DDPF_NORMAL: u32 = 0x8000_0000;
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
    Bc7,
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
            Encoding::Bc7 => DdsCompression::Bc7,
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

    /// `DXGI_FORMAT_*_SRGB` names that mark the texture sRGB.
    fn dxgi_is_srgb(dxgi: u32) -> bool {
        matches!(dxgi, 29 | 35 | 72 | 75 | 78 | 91 | 93 | 96 | 99)
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

/// The DX10 header's DXGI format, and whether it declares a cubemap (`MISC_TEXTURECUBE` —
/// a cubemap's array size counts its six faces; arrays of cubes are refused). Anything but
/// a 2D resource is refused.
fn dx10_encoding(b: &[u8]) -> Result<(Encoding, bool), CodecError> {
    let (dxgi, dimension, misc, array) = (le32(b, HEADER)?, le32(b, HEADER + 4)?, le32(b, HEADER + 8)?, le32(b, HEADER + 12)?);
    if misc & !MISC_TEXTURECUBE != 0 {
        return Err(unsupported(format!("unsupported resource misc flags {misc:#x}")));
    }
    let cube = misc & MISC_TEXTURECUBE != 0;
    if dimension != DIMENSION_TEXTURE2D {
        return Err(unsupported(format!("only 2D textures are supported (resource dimension {dimension})")));
    }
    if cube {
        if array % 6 != 0 {
            return Err(bad(format!("a cubemap's array size is a multiple of 6, not {array}")));
        }
        if array != 6 {
            return Err(unsupported(format!("arrays of cubemaps are not supported ({array} cubes)")));
        }
    } else if array > 1 {
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
        97..=99 => Ok(Encoding::Bc7),
        _ => Err(unsupported(format!("DXGI format {dxgi} is not supported"))),
    }
    .map(|e| (e, cube))
}

/// The parsed header: the encoding, where its pixel data starts, the top level's dimensions,
/// the declared mip count, the stored row pitch (which is the top level's only), whether the
/// pixel format promises alpha pixels (a DXT1 reads back `bc1a`), the DX10 header's sRGB
/// mark and the cubemap faces present (bit 0 = +X … bit 5 = -Z; 0 = a plain 2D texture).
struct Parsed<'a> {
    encoding: Encoding,
    data: &'a [u8],
    width: u32,
    height: u32,
    levels: u32,
    pitch: Option<u32>,
    alpha_pixels: bool,
    srgb: bool,
    face_mask: u32,
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
    if caps2 & DDSCAPS2_VOLUME != 0 || (flags & DDSD_DEPTH != 0 && depth > 1) {
        return Err(unsupported("volume textures are not supported"));
    }
    let dx10 = le32(b, 80)? & DDPF_FOURCC != 0 && &le32(b, 84)?.to_le_bytes() == b"DX10";
    let (encoding, cube, start) = if dx10 {
        let (e, cube) = dx10_encoding(b)?;
        (e, cube, HEADER + DX10_HEADER)
    } else {
        (legacy_encoding(b)?, false, HEADER)
    };
    // A legacy cubemap marks its faces in caps2 bits 10–15; the flag alone means all six.
    let face_mask = if !dx10 && caps2 & DDSCAPS2_CUBEMAP != 0 {
        match (caps2 >> 10) & 0x3f {
            0 => 0x3f,
            mask => mask,
        }
    } else if cube {
        0x3f
    } else {
        0
    };
    let levels = if flags & DDSD_MIPMAPCOUNT != 0 { le32(b, 28)?.max(1) } else { 1 };
    let pitch = if flags & DDSD_PITCH != 0 { Some(le32(b, 20)?) } else { None };
    let alpha_pixels = le32(b, 80)? & DDPF_ALPHAPIXELS != 0;
    let srgb = dx10 && Encoding::dxgi_is_srgb(le32(b, HEADER)?);
    Ok(Parsed { encoding, data: b.get(start..).unwrap_or_default(), width, height, levels, pitch, alpha_pixels, srgb, face_mask })
}

pub(crate) fn decode(b: &[u8], limits: &Limits) -> Result<Image, CodecError> {
    level_image(&parse(b)?, 0, 0, limits)
}

/// Every declared mip level as an image, largest first; [`decode`] reads the top one. The
/// chain ends at 1×1, so a header claiming more levels than reduce to it reads that many —
/// the field is hostile input, and each level re-walks the ones before it. A cubemap's chain
/// is its +X face's.
pub(crate) fn decode_levels(b: &[u8], limits: &Limits) -> Result<Vec<Image>, CodecError> {
    let p = parse(b)?;
    (0..p.mip_count()).map(|i| level_image(&p, 0, i, limits)).collect()
}

/// Every stored cubemap face: `(canonical index, mip chain)` pairs in file order, the index
/// `0`–`5` for `+X -X +Y -Y +Z -Z` (a partial legacy cube skips its missing faces) and each
/// chain level-0 first. A plain 2D texture yields a single `0`-indexed face.
pub(crate) fn decode_faces(b: &[u8], limits: &Limits) -> Result<Vec<(u8, Vec<Image>)>, CodecError> {
    let p = parse(b)?;
    let mask = if p.face_mask == 0 { 1 } else { p.face_mask };
    let mut faces = Vec::with_capacity(mask.count_ones() as usize);
    for (ordinal, index) in (0..6u8).filter(|i| mask & (1 << i) != 0).enumerate() {
        let chain = (0..p.mip_count()).map(|i| level_image(&p, ordinal as u32, i, limits)).collect::<Result<_, _>>()?;
        faces.push((index, chain));
    }
    Ok(faces)
}

/// The image of mip `index` (`0` is the top level) of face `face` (`0` for a plain texture).
/// Levels are stored largest first and, except the top level's stored pitch, tightly packed;
/// a cubemap's faces are contiguous whole mip chains, face-major.
fn level_image(p: &Parsed, face: u32, index: u32, limits: &Limits) -> Result<Image, CodecError> {
    let (mut w, mut h, mut chain) = (p.width, p.height, 0usize);
    for i in 0..p.mip_count() {
        let pitch = if i == 0 { p.pitch } else { None };
        let (_, size) = p.encoding.level_layout(w, h, pitch)?;
        chain = chain.checked_add(size).ok_or_else(|| bad("pixel data size overflow"))?;
        w = (w / 2).max(1);
        h = (h / 2).max(1);
    }
    let mut at = (face as usize).checked_mul(chain).ok_or_else(|| bad("pixel data size overflow"))?;
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
        cube: p.face_mask != 0,
        faces: if p.face_mask == 0 { 1 } else { p.face_mask.count_ones() },
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
                    Encoding::Bc7 => dds_bc7::decode_bc7(block),
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
    Bc4,
    Bc5,
    Bc7,
    /// BC1 blocks under the DXT1 name, but the header promises `DDPF_ALPHAPIXELS` (DXT1a).
    Bc1a,
    /// One alpha byte per pixel (`DDPF_ALPHA`, DXGI A8_UNORM).
    Alpha8,
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
            (DdsCompression::Bc1a, ChannelLayout::Rgba) => Store::Bc1a,
            (DdsCompression::Bc2, ChannelLayout::Rgba) => Store::Bc2,
            (DdsCompression::Bc3, ChannelLayout::Rgba) => Store::Bc3,
            (DdsCompression::Bc4, ChannelLayout::Gray) => Store::Bc4,
            (DdsCompression::Bc5, ChannelLayout::GrayA) => Store::Bc5,
            (DdsCompression::Bc7, ChannelLayout::Rgba) => Store::Bc7,
            // A8 keeps only the alpha: a layout without one is an all-opaque mask.
            (DdsCompression::A8, _) => Store::Alpha8,
            (_, layout) => return Err(encode_err(format!("{layout:?} pixels can't be written as `{}`", compression.name()))),
        })
    }

    /// Bytes per pixel, or per 4×4 block for a block compression.
    fn unit(self) -> usize {
        match self {
            Store::Luminance | Store::Alpha8 => 1,
            Store::LuminanceAlpha => 2,
            Store::Bgrx | Store::Bgra => 4,
            Store::Bc1 | Store::Bc1a | Store::Bc4 => 8,
            _ => 16,
        }
    }

    fn is_block(self) -> bool {
        matches!(self, Store::Bc1 | Store::Bc1a | Store::Bc2 | Store::Bc3 | Store::Bc4 | Store::Bc5 | Store::Bc7)
    }

    /// The DXGI format the DX10 header names, when the store has no legacy pixel format or
    /// `srgb` asks for a name only DXGI has. Luminance, A8, BC4 and BC5 have no sRGB form.
    fn dxgi(self, srgb: bool) -> Option<u32> {
        let f = match (self, srgb) {
            (Store::Bc4, _) => 80,                          // BC4_UNORM
            (Store::Bc5, _) => 83,                          // BC5_UNORM
            (Store::Bc7, false) => 98,                      // BC7_UNORM
            (Store::Bc7, true) => 99,                       // BC7_UNORM_SRGB
            (Store::Bc1, true) | (Store::Bc1a, true) => 72, // BC1_UNORM_SRGB
            (Store::Bc2, true) => 75,                       // BC2_UNORM_SRGB
            (Store::Bc3, true) => 78,                       // BC3_UNORM_SRGB
            (Store::Bgra, true) => 91,                      // B8G8R8A8_UNORM_SRGB
            (Store::Bgrx, true) => 93,                      // B8G8R8X8_UNORM_SRGB
            _ => return None,
        };
        Some(f)
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
    /// `dx10` writes the DX10 FourCC for a store that also has a legacy name (an sRGB mark).
    fn pixel_format(self, dx10: bool) -> (u32, [u8; 4], u32, [u32; 4]) {
        if dx10 {
            return (DDPF_FOURCC, *b"DX10", 0, [0; 4]);
        }
        match self {
            Store::Luminance => (DDPF_LUMINANCE, [0; 4], 8, [0xff, 0, 0, 0]),
            Store::LuminanceAlpha => (DDPF_LUMINANCE | DDPF_ALPHAPIXELS, [0; 4], 16, [0xff, 0, 0, 0xff00]),
            Store::Alpha8 => (DDPF_ALPHA, [0; 4], 8, [0, 0, 0, 0xff]),
            Store::Bgrx => (DDPF_RGB, [0; 4], 32, [0xff_0000, 0xff00, 0xff, 0]),
            Store::Bgra => (DDPF_RGB | DDPF_ALPHAPIXELS, [0; 4], 32, [0xff_0000, 0xff00, 0xff, 0xff00_0000]),
            Store::Bc1 => (DDPF_FOURCC, *b"DXT1", 0, [0; 4]),
            Store::Bc1a => (DDPF_FOURCC | DDPF_ALPHAPIXELS, *b"DXT1", 0, [0; 4]),
            Store::Bc2 => (DDPF_FOURCC, *b"DXT3", 0, [0; 4]),
            Store::Bc3 => (DDPF_FOURCC, *b"DXT5", 0, [0; 4]),
            _ => (DDPF_FOURCC, *b"DX10", 0, [0; 4]),
        }
    }
}

fn half((w, h): (usize, usize)) -> (usize, usize) {
    ((w / 2).max(1), (h / 2).max(1))
}

pub(crate) fn encode(image: &Image, plan: Plan, opts: &EncodeOptions) -> Result<Vec<u8>, CodecError> {
    if opts.dds.cube != DdsCubeMap::Off {
        return Err(encode_err("a cubemap write needs six faces"));
    }
    encode_faces(std::slice::from_ref(image), plan, opts)
}

/// `faces` is a single image for a plain 2D texture or the six faces `+X -X +Y -Y +Z -Z`
/// for a cubemap. A cubemap's faces must be square and one size; each face gets its own
/// mip chain and the chains store face-major, contiguous.
pub(crate) fn encode_faces(faces: &[Image], plan: Plan, opts: &EncodeOptions) -> Result<Vec<u8>, CodecError> {
    if !matches!(faces.len(), 1 | 6) {
        return Err(encode_err("a cubemap write needs six faces"));
    }
    let cube = faces.len() == 6;
    let (width, height) = faces[0].dimensions();
    if cube {
        for img in faces {
            let (w, h) = img.dimensions();
            if w != h || (w, h) != (width, height) {
                return Err(encode_err("a cubemap's faces must be square and one size"));
            }
        }
    }
    let store = Store::new(opts.dds.compression, plan.layout)?;
    let mut chains = Vec::with_capacity(faces.len());
    for img in faces {
        chains.push(mip_chain(img, plan.layout, opts)?);
    }

    // Every level's size, checked before anything is allocated.
    let mut sizes = Vec::with_capacity(chains.len());
    for levels in &chains {
        let mut face = Vec::with_capacity(levels.len());
        for &(_, size) in levels {
            face.push(store.level_bytes(size.0, size.1).ok_or_else(|| encode_err("texture size overflow"))?);
        }
        sizes.push(face);
    }
    // BC4, BC5 and BC7 have no legacy pixel format name: the file gets the DX10 header, as does
    // any store the sRGB mark applies to (its name exists only in DXGI).
    let dxgi = store.dxgi(opts.dds.srgb);
    let base = if dxgi.is_some() { HEADER + DX10_HEADER } else { HEADER };
    let total = sizes.iter().flatten().try_fold(base, |total, &s| total.checked_add(s)).ok_or_else(|| encode_err("texture size overflow"))?;
    let header = header(
        width,
        height,
        store,
        chains[0].len() as u32,
        sizes.first().and_then(|f| f.first()).copied().unwrap_or(0),
        HeaderOpts { dx10: dxgi.is_some(), normal_map: opts.dds.normal_map, cube },
    )?;
    let mut out = crate::options::alloc_zeroed::<u8>(total)?;
    out.get_mut(..HEADER).ok_or_else(|| encode_err("output buffer too small"))?.copy_from_slice(&header);
    if let Some(dxgi) = dxgi {
        let d = out.get_mut(HEADER..base).ok_or_else(|| encode_err("output buffer too small"))?;
        let ext = [dxgi, DIMENSION_TEXTURE2D, if cube { MISC_TEXTURECUBE } else { 0 }, if cube { 6 } else { 1 }, 0];
        for (i, v) in ext.iter().enumerate() {
            d.get_mut(4 * i..4 * i + 4).ok_or_else(|| encode_err("output buffer too small"))?.copy_from_slice(&v.to_le_bytes());
        }
    }

    let mut at = base;
    for (levels, sizes) in chains.iter().zip(sizes.iter()) {
        for ((pixels, size), &bytes) in levels.iter().zip(sizes.iter()) {
            let dst = out.get_mut(at..at + bytes).ok_or_else(|| encode_err("output buffer too small"))?;
            write_level(store, pixels, *size, dst, opts.dds.quality)?;
            at += bytes;
        }
    }
    Ok(out)
}

/// A level's pixel buffer and its `(width, height)`.
type MipLevel = (Vec<u8>, (usize, usize));

/// One face's level pixel buffers, converted to `layout` and filtered down from the top
/// per the mip options.
fn mip_chain(image: &Image, layout: ChannelLayout, opts: &EncodeOptions) -> Result<Vec<MipLevel>, CodecError> {
    let img = image.converted(layout, SampleType::U8);
    let pixels = img.as_u8().ok_or_else(|| encode_err("the image was not converted to 8 bits per channel"))?;
    let top = img.dimensions();
    let top = (top.0 as usize, top.1 as usize);
    let channels = img.layout().channels();

    let mut levels: Vec<(Vec<u8>, (usize, usize))> = vec![(pixels.to_vec(), top)];
    if opts.dds.export_premult_alpha {
        premultiply(&mut levels[0].0, channels);
    }
    if opts.dds.mipmaps {
        let linear = opts.dds.srgb && opts.dds.mip_gamma_correct;
        let mut size = top;
        while size.0 > 1 || size.1 > 1 {
            let next = half(size);
            if next.0.min(next.1) < opts.dds.min_mip_size.max(1) as usize {
                break;
            }
            if opts.dds.max_mip_count > 0 && levels.len() >= opts.dds.max_mip_count as usize {
                break;
            }
            let last = &levels[levels.len() - 1].0;
            let level = if opts.dds.mip_premult_alpha && !opts.dds.export_premult_alpha {
                // Straight-alpha content filters in premult space so transparent texels do
                // not bleed their RGB, then returns to straight.
                let mut premult = last.clone();
                premultiply(&mut premult, channels);
                let mut level = downsample(&premult, size, next, channels, opts.dds.mip_filter, linear)?;
                unpremultiply(&mut level, channels);
                level
            } else {
                downsample(last, size, next, channels, opts.dds.mip_filter, linear)?
            };
            levels.push((level, next));
            size = next;
        }
    }
    if opts.dds.flip {
        for (pixels, size) in levels.iter_mut() {
            flip_rows(pixels, size.0.saturating_mul(channels));
        }
    }
    Ok(levels)
}

/// The marks `header` writes past a plain 2D texture's. `dx10` names the DX10 pixel format
/// (a DX10_HEADER follows). `normal_map` sets `DDPF_NORMAL`, the flag engines read for
/// tangent-space normal-map detection. `cube` marks `DDSCAPS2_CUBEMAP` with all six face
/// bits in `caps2` (and forces `DDSCAPS_COMPLEX`, which a cubemap has whether or not it is
/// mipmapped).
#[derive(Clone, Copy)]
struct HeaderOpts {
    dx10: bool,
    normal_map: bool,
    cube: bool,
}

/// The `DDS ` signature and DDS_HEADER. `top` is the top level's size in bytes.
fn header(width: u32, height: u32, store: Store, levels: u32, top: usize, opts: HeaderOpts) -> Result<[u8; HEADER], CodecError> {
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
    let (flags, fourcc, bits, masks) = store.pixel_format(opts.dx10);
    put(76, 32);
    put(80, if opts.normal_map { flags | DDPF_NORMAL } else { flags });
    put(84, u32::from_le_bytes(fourcc));
    put(88, bits);
    for (i, mask) in masks.into_iter().enumerate() {
        put(92 + 4 * i, mask);
    }
    put(108, DDSCAPS_TEXTURE | if mipmapped || opts.cube { DDSCAPS_COMPLEX } else { 0 } | if mipmapped { DDSCAPS_MIPMAP } else { 0 });
    // caps2 bits 10–15 mark the +X -X +Y -Y +Z -Z faces — all six for a written cube.
    put(112, if opts.cube { DDSCAPS2_CUBEMAP | 0xfc00 } else { 0 });
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
/// `quality` is the BC7 encoder's effort; the other stores ignore it.
fn write_level(store: Store, pixels: &[u8], (width, height): (usize, usize), dst: &mut [u8], quality: DdsQuality) -> Result<(), CodecError> {
    let row_bytes = store.row_bytes(width).ok_or_else(|| encode_err("texture size overflow"))?;
    match store {
        Store::Luminance | Store::LuminanceAlpha => {
            // The image's gray (and alpha) samples, in the same order.
            let src = pixels.get(..dst.len()).ok_or_else(|| encode_err("the level is smaller than its size"))?;
            dst.copy_from_slice(src);
        }
        Store::Alpha8 => {
            // A8 stores the alpha alone; a layout without one is all-opaque.
            let stride = pixels.len() / (width * height).max(1);
            let alpha = stride == 2 || stride == 4;
            for (d, s) in dst.iter_mut().zip(pixels.chunks_exact(stride.max(1))) {
                *d = if alpha { s[stride - 1] } else { 255 };
            }
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
        Store::Bc1 | Store::Bc1a | Store::Bc2 | Store::Bc3 | Store::Bc4 | Store::Bc5 | Store::Bc7 => {
            let channels = pixels.len() / (width * height).max(1);
            let texels: Cow<[[u8; 4]]> = if channels == 4 {
                Cow::Borrowed(pixels.as_chunks::<4>().0)
            } else {
                Cow::Owned(pixels.chunks_exact(channels.max(1)).map(|s| [s[0], s.get(1).copied().unwrap_or(0), 0, 255]).collect())
            };
            rows(dst, row_bytes, |by, row| {
                for (bx, d) in row.chunks_exact_mut(store.unit()).enumerate() {
                    if store == Store::Bc7 {
                        // The BC7 encoder fits the whole block: edge texels are replicated.
                        let encoded = dds_bc7_enc::encode_bc7(&gather_clamped(&texels, (width, height), bx, by), quality);
                        d.copy_from_slice(&encoded);
                        continue;
                    }
                    let (block, inside) = gather(&texels, (width, height), bx, by);
                    let encoded: &[u8] = match store {
                        Store::Bc1 | Store::Bc1a => &dds_bc::encode_bc1(&block, inside),
                        Store::Bc2 => &dds_bc::encode_bc2(&block, inside),
                        Store::Bc3 => &dds_bc::encode_bc3(&block, inside),
                        Store::Bc4 => &dds_bc::encode_bc4(&block, inside),
                        _ => &dds_bc::encode_bc5(&block, inside),
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

/// Block (`bx`, `by`)'s texels with the image's edge texels replicated where the block
/// runs off the image, for an encoder that weighs every texel (BC7).
fn gather_clamped(texels: &[[u8; 4]], (width, height): (usize, usize), bx: usize, by: usize) -> Block {
    let mut block = [[0u8; 4]; 16];
    for (t, texel) in block.iter_mut().enumerate() {
        let (x, y) = ((bx * 4 + t % 4).min(width.saturating_sub(1)), (by * 4 + t / 4).min(height.saturating_sub(1)));
        if let Some(p) = texels.get(y * width + x) {
            *texel = *p;
        }
    }
    block
}

/// The alpha channel index in a `channels`-strided pixel, when the layout has one (GrayA,
/// RGBA — the last channel). Colour channels are everything before it.
fn alpha_at(channels: usize) -> Option<usize> {
    match channels {
        2 | 4 => Some(channels - 1),
        _ => None,
    }
}

/// The next mip level under `filter`. `linear` filters the colour channels in linear light
/// (an sRGB texture's stored values are EOTF'd; averaging them directly darkens the chain).
/// Alpha always filters in raw space — it encodes coverage, not luminance.
fn downsample(src: &[u8], size: (usize, usize), next: (usize, usize), channels: usize, filter: DdsMipFilter, linear: bool) -> Result<Vec<u8>, CodecError> {
    match filter {
        DdsMipFilter::Box => box_downsample(src, size, next, channels, linear),
        DdsMipFilter::Triangle => triangle_downsample(src, size, next, channels, linear),
        DdsMipFilter::Min => extreme_downsample(src, size, next, channels, false),
        DdsMipFilter::Max => extreme_downsample(src, size, next, channels, true),
        DdsMipFilter::Kaiser => filtered_downsample(src, size, next, channels, kaiser_kernel, 3.0, linear),
        DdsMipFilter::MitchellNetravali => filtered_downsample(src, size, next, channels, mitchell_kernel, 2.0, linear),
    }
}

/// Multiply the colour channels by alpha, so transparent texels do not leak their RGB into
/// filtered mips. A no-op on a layout without alpha.
fn premultiply(pixels: &mut [u8], channels: usize) {
    let Some(a) = alpha_at(channels) else { return };
    for px in pixels.chunks_exact_mut(channels) {
        let alpha = u32::from(px[a]);
        for c in px.iter_mut().take(a) {
            *c = ((u32::from(*c) * alpha + 127) / 255) as u8;
        }
    }
}

/// The reverse of [`premultiply`]: a fully transparent texel's colour is lost (it carried
/// none anyway).
fn unpremultiply(pixels: &mut [u8], channels: usize) {
    let Some(a) = alpha_at(channels) else { return };
    for px in pixels.chunks_exact_mut(channels) {
        let alpha = u32::from(px[a]);
        for c in px.iter_mut().take(a) {
            *c = (u32::from(*c) * 255 + alpha / 2).checked_div(alpha).map_or(0, |v| v.min(255) as u8);
        }
    }
}

/// Reverse the row order of a `row_len`-byte-row image (the GL-vs-D3D origin flip).
fn flip_rows(pixels: &mut [u8], row_len: usize) {
    if row_len == 0 {
        return;
    }
    let h = pixels.len() / row_len;
    for y in 0..h / 2 {
        for x in 0..row_len {
            let (top, bot) = (y * row_len + x, (h - 1 - y) * row_len + x);
            pixels.swap(top, bot);
        }
    }
}

/// sRGB byte → linear-light value, the piecewise IEC EOTF.
fn srgb_to_linear_lut() -> [f32; 256] {
    std::array::from_fn(|i| {
        let v = i as f32 / 255.0;
        if v <= 0.04045 { v / 12.92 } else { ((v + 0.055) / 1.055).powf(2.4) }
    })
}

/// Linear-light value → sRGB byte.
fn linear_to_srgb_u8(v: f32) -> u8 {
    let v = v.clamp(0.0, 1.0);
    let s = if v <= 0.003_130_8 { v * 12.92 } else { 1.055 * v.powf(1.0 / 2.4) - 0.055 };
    (s.clamp(0.0, 1.0) * 255.0).round() as u8
}

/// Box: each pixel the average of the 2×2 it covers, or of 2×3, 3×2 or 3×3 along the last
/// column and row of an odd-sized level, so no pixel is dropped.
fn box_downsample(src: &[u8], (w, h): (usize, usize), (nw, nh): (usize, usize), channels: usize, linear: bool) -> Result<Vec<u8>, CodecError> {
    let len = nw.checked_mul(nh).and_then(|n| n.checked_mul(channels)).ok_or_else(|| encode_err("mip level size overflow"))?;
    let mut dst = crate::options::alloc_zeroed::<u8>(len)?;
    let span = |i: usize, n: usize, size: usize| 2 * i..=if i + 1 == n { size.saturating_sub(1) } else { 2 * i + 1 };
    let lut = srgb_to_linear_lut();
    let alpha = alpha_at(channels);
    rows(&mut dst, nw * channels, |y, row| {
        let ys = span(y, nh, h);
        for (x, px) in row.chunks_exact_mut(channels.max(1)).enumerate() {
            let xs = span(x, nw, w);
            for (c, v) in px.iter_mut().enumerate() {
                let is_colour = alpha != Some(c);
                let (mut sum, mut n) = (0.0f32, 0u32);
                for sy in ys.clone() {
                    for sx in xs.clone() {
                        if let Some(&s) = src.get((sy * w + sx) * channels + c) {
                            sum += if linear && is_colour { lut[s as usize] } else { f32::from(s) };
                            n += 1;
                        }
                    }
                }
                *v = if linear && is_colour { linear_to_srgb_u8(sum / n.max(1) as f32) } else { ((sum + (n / 2) as f32) / n.max(1) as f32) as u8 };
            }
        }
    });
    Ok(dst)
}

/// Tent `[1,3,3,1]` separable 2:1 (NVIDIA's "triangle" filter class): a wider footprint
/// than box, less aliasing in mips. Edges replicate.
fn triangle_downsample(src: &[u8], (w, h): (usize, usize), (nw, nh): (usize, usize), channels: usize, linear: bool) -> Result<Vec<u8>, CodecError> {
    let len = nw.checked_mul(nh).and_then(|n| n.checked_mul(channels)).ok_or_else(|| encode_err("mip level size overflow"))?;
    let mut dst = crate::options::alloc_zeroed::<u8>(len)?;
    const TAP: [u32; 4] = [1, 3, 3, 1]; // sums to 8; total weight 64
    let lut = srgb_to_linear_lut();
    let alpha = alpha_at(channels);
    rows(&mut dst, nw * channels, |y, row| {
        for (x, px) in row.chunks_exact_mut(channels.max(1)).enumerate() {
            for (c, v) in px.iter_mut().enumerate() {
                let is_colour = alpha != Some(c);
                let mut acc = 0.0f32;
                for (ty, &wy) in TAP.iter().enumerate() {
                    let sy = (y * 2 + ty).saturating_sub(1).min(h.saturating_sub(1));
                    for (tx, &wx) in TAP.iter().enumerate() {
                        let sx = (x * 2 + tx).saturating_sub(1).min(w.saturating_sub(1));
                        if let Some(&s) = src.get((sy * w + sx) * channels + c) {
                            acc += (wx * wy) as f32 * if linear && is_colour { lut[s as usize] } else { f32::from(s) };
                        }
                    }
                }
                *v = if linear && is_colour { linear_to_srgb_u8(acc / 64.0) } else { ((acc + 32.0) / 64.0) as u8 };
            }
        }
    });
    Ok(dst)
}

/// Per-channel extremum of the covered 2×2 — the filters height and parallax-occlusion
/// maps want (preserves min-clip / max-height range).
fn extreme_downsample(src: &[u8], (w, h): (usize, usize), (nw, nh): (usize, usize), channels: usize, take_max: bool) -> Result<Vec<u8>, CodecError> {
    let len = nw.checked_mul(nh).and_then(|n| n.checked_mul(channels)).ok_or_else(|| encode_err("mip level size overflow"))?;
    let mut dst = crate::options::alloc_zeroed::<u8>(len)?;
    rows(&mut dst, nw * channels, |y, row| {
        for (x, px) in row.chunks_exact_mut(channels.max(1)).enumerate() {
            for (c, v) in px.iter_mut().enumerate() {
                let mut best = 0u8;
                for ty in 0..2usize {
                    for tx in 0..2usize {
                        if let Some(&s) = src.get(((2 * y + ty).min(h.saturating_sub(1)) * w + (2 * x + tx).min(w.saturating_sub(1))) * channels + c) {
                            best = if take_max {
                                best.max(s)
                            } else if ty == 0 && tx == 0 {
                                s
                            } else {
                                best.min(s)
                            };
                        }
                    }
                }
                *v = best;
            }
        }
    });
    Ok(dst)
}

fn sinc_pi(t: f32) -> f32 {
    if t == 0.0 { 1.0 } else { (t * std::f32::consts::PI).sin() / (t * std::f32::consts::PI) }
}

/// The modified Bessel function I0 (Cephes' polynomial approximation), for the Kaiser
/// window — `std::cyl_bessel_i` has no Rust equivalent.
fn bessel_i0(x: f32) -> f32 {
    let ax = x.abs();
    if ax < 3.75 {
        let y = (x / 3.75) * (x / 3.75);
        1.0 + y * (3.515_622_9 + y * (3.089_942_4 + y * (1.206_749_2 + y * (0.265_973_2 + y * (0.036_076_8 + y * 0.004_581_3)))))
    } else {
        let y = 3.75 / ax;
        (ax.exp() / ax.sqrt())
            * (0.398_942_3
                + y * (0.013_285_92
                    + y * (0.002_253_19
                        + y * (-0.001_575_65 + y * (0.009_162_81 + y * (-0.020_577_06 + y * (0.026_355_37 + y * (-0.016_476_33 + y * 0.003_923_77))))))))
    }
}

/// Kaiser-Bessel windowed sinc (width 3, alpha 4 — the usual mip defaults).
fn kaiser_kernel(t: f32) -> f32 {
    const WIDTH: f32 = 3.0;
    const ALPHA: f32 = 4.0;
    if t.abs() >= WIDTH {
        return 0.0;
    }
    let r = (1.0 - (t / WIDTH) * (t / WIDTH)).max(0.0).sqrt();
    sinc_pi(t) * bessel_i0(ALPHA * r) / bessel_i0(ALPHA)
}

/// Mitchell-Netravali cubic, B = C = 1/3 (support |t| < 2).
fn mitchell_kernel(t: f32) -> f32 {
    let t = t.abs();
    if t < 1.0 {
        (7.0 * t * t * t - 12.0 * t * t + 16.0 / 3.0) / 6.0
    } else if t < 2.0 {
        (-7.0 / 3.0 * t * t * t + 12.0 * t * t - 20.0 * t + 32.0 / 3.0) / 6.0
    } else {
        0.0
    }
}

/// Generic separable weighted downsample: `kernel`'s argument is the source offset in
/// destination-pixel units, `support` its radius in those units. Edges replicate. The
/// `linear` flag weighs colour channels in linear light like the other filters.
fn filtered_downsample(
    src: &[u8],
    (w, h): (usize, usize),
    (nw, nh): (usize, usize),
    channels: usize,
    kernel: fn(f32) -> f32,
    support: f32,
    linear: bool,
) -> Result<Vec<u8>, CodecError> {
    let len = nw.checked_mul(nh).and_then(|n| n.checked_mul(channels)).ok_or_else(|| encode_err("mip level size overflow"))?;
    let mut dst = crate::options::alloc_zeroed::<u8>(len)?;
    let lut = srgb_to_linear_lut();
    let alpha = alpha_at(channels);
    let span = (2.0 * support).ceil() as i32;
    rows(&mut dst, nw * channels, |y, row| {
        for (x, px) in row.chunks_exact_mut(channels.max(1)).enumerate() {
            let (cx, cy) = (2 * x + 1, 2 * y + 1);
            for (c, v) in px.iter_mut().enumerate() {
                let is_colour = alpha != Some(c);
                let (mut acc, mut wsum) = (0.0f32, 0.0f32);
                for iy in cy as i32 - span..=cy as i32 + span {
                    let wy = kernel((iy as f32 + 0.5 - cy as f32) / 2.0);
                    if wy == 0.0 {
                        continue;
                    }
                    let sy = iy.clamp(0, h.saturating_sub(1) as i32) as usize;
                    for ix in cx as i32 - span..=cx as i32 + span {
                        let wgt = wy * kernel((ix as f32 + 0.5 - cx as f32) / 2.0);
                        if wgt == 0.0 {
                            continue;
                        }
                        let sx = ix.clamp(0, w.saturating_sub(1) as i32) as usize;
                        if let Some(&s) = src.get((sy * w + sx) * channels + c) {
                            wsum += wgt;
                            acc += wgt * if linear && is_colour { lut[s as usize] } else { f32::from(s) };
                        }
                    }
                }
                let mean = acc * if wsum != 0.0 { 1.0 / wsum } else { 0.0 };
                *v = if linear && is_colour { linear_to_srgb_u8(mean) } else { mean.clamp(0.0, 255.0).round() as u8 };
            }
        }
    });
    Ok(dst)
}
