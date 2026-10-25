//! Decode limits and encode options.

use crate::error::CodecError;
use crate::image::{ChannelLayout, SampleType};

/// Decompression-bomb guards. Dimension checks precede pixel allocation;
/// metadata decoding checks its budget before growing decoded output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Limits {
    pub max_width: u32,
    pub max_height: u32,
    /// Maximum `width * height`.
    pub max_pixels: u64,
    /// Maximum bytes for the decoded pixel buffer (and passed on as the
    /// allocation budget to underlying decoders).
    /// PNG also applies this to the aggregate decoded text/XMP UTF-8 bytes,
    /// including keywords, separately from pixels. Animated WebP charges
    /// every retained frame's full pixel copy plus its own copy of the
    /// file's ICC profile against this budget. This is not a total RSS cap.
    pub max_alloc: u64,
}

impl Default for Limits {
    fn default() -> Self {
        // 2^30 pixels (e.g. 32768²) and, on 64-bit targets, 8 GiB: enough for the large
        // documents Photoshop users open (a 20000² RGBA 16-bit image is 3.2 GB). On 32-bit
        // targets (wasm32 has a 4 GiB address space shared with the document, history and
        // the decoder's own buffers) 1 GiB is the most one buffer can safely ask for.
        let max_alloc = if cfg!(target_pointer_width = "64") { 8 << 30 } else { 1 << 30 };
        Limits { max_width: 1 << 18, max_height: 1 << 18, max_pixels: 1 << 30, max_alloc }
    }
}

impl Limits {
    /// Effectively unlimited (still bounded by `usize` overflow checks).
    pub fn none() -> Self {
        Limits { max_width: u32::MAX, max_height: u32::MAX, max_pixels: u64::MAX, max_alloc: u64::MAX }
    }

    /// Validate declared dimensions for an output buffer of the given layout
    /// and sample type.
    pub fn check(&self, width: u32, height: u32, layout: ChannelLayout, sample: SampleType) -> Result<(), CodecError> {
        self.check_bytes(width, height, (layout.channels() * sample.bytes()) as u64)
    }

    pub(crate) fn check_bytes(&self, width: u32, height: u32, bytes_per_pixel: u64) -> Result<(), CodecError> {
        if width == 0 || height == 0 {
            return Err(CodecError::InvalidImage(format!("zero-sized image {width}x{height}")));
        }
        if width > self.max_width || height > self.max_height {
            return Err(CodecError::LimitExceeded(format!("dimensions {width}x{height} exceed max {}x{}", self.max_width, self.max_height)));
        }
        let pixels = width as u64 * height as u64;
        if pixels > self.max_pixels {
            return Err(CodecError::LimitExceeded(format!("{pixels} pixels exceed max {}", self.max_pixels)));
        }
        let bytes = pixels.saturating_mul(bytes_per_pixel);
        if bytes > self.max_alloc {
            return Err(CodecError::LimitExceeded(format!("{bytes} bytes exceed max_alloc {}", self.max_alloc)));
        }
        if usize::try_from(bytes).is_err() {
            return Err(CodecError::LimitExceeded(format!("{bytes} bytes do not fit in memory")));
        }
        Ok(())
    }

    /// Decompression-bomb guard for formats that compress: refuse a file too small to hold
    /// `declared_bytes` of (stored, pre-expansion) image data, given that the compression
    /// never beats `max_ratio`:1. Called before the output buffer is allocated, so a
    /// header-only hostile file cannot make the decoder reserve gigabytes. A fixed 1 MiB of
    /// slack keeps tiny images and headers with large metadata unaffected.
    pub(crate) fn check_plausible(declared_bytes: u64, file_len: usize, max_ratio: u64) -> Result<(), CodecError> {
        let capacity = (file_len as u64).saturating_mul(max_ratio).saturating_add(1 << 20);
        if declared_bytes > capacity {
            return Err(CodecError::LimitExceeded(format!("a {file_len} byte file cannot hold {declared_bytes} bytes of image data")));
        }
        Ok(())
    }

    pub(crate) fn alloc_usize(&self) -> usize {
        usize::try_from(self.max_alloc).unwrap_or(usize::MAX)
    }
}

/// Options for [`crate::decode_with`].
#[derive(Debug, Clone, Default)]
pub struct DecodeOptions {
    pub limits: Limits,
    /// Keep the pixels as stored instead of applying the EXIF / TIFF
    /// orientation (by default they are turned upright and the metadata's
    /// orientation is rewritten to 1).
    pub keep_orientation: bool,
}

/// PNG zlib effort.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PngCompression {
    None,
    Fast,
    #[default]
    Default,
    Best,
}

/// How a PNG's `sTER` chunk pairs its two views (PNG Extensions 1.4.0, §3.6).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PngStereo {
    /// Side by side with the right-eye view first (cross-eyed viewing, mode 0).
    CrossFuse,
    /// Side by side with the left-eye view first (parallel viewing, mode 1).
    DivergingFuse,
}

impl PngStereo {
    /// The view for an `sTER` mode byte; reserved and unknown values are `None`.
    pub fn from_mode(mode: u8) -> Option<Self> {
        match mode {
            0 => Some(Self::CrossFuse),
            1 => Some(Self::DivergingFuse),
            _ => None,
        }
    }

    /// The `sTER` mode byte.
    pub fn mode(self) -> u8 {
        match self {
            Self::CrossFuse => 0,
            Self::DivergingFuse => 1,
        }
    }
}

/// TIFF strip compression. The default is none, as in Photoshop: uncompressed strips are the
/// only kind every TIFF reader opens (Deflate and the predictor are not baseline TIFF, #4484).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TiffCompression {
    #[default]
    None,
    Lzw,
    Deflate,
    PackBits,
}

/// JPEG scan layout: Photoshop's JPEG Options › Format Options.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum JpegFormat {
    /// Baseline ("Standard"): one sequential scan with the standard Huffman tables.
    #[default]
    Baseline,
    /// Baseline Optimized: one sequential scan with Huffman tables fitted to the image. Only at
    /// 4:4:4 for now: a 4:2:0 (subsampled RGB) baseline scan keeps the standard tables, because
    /// fitted tables make jpeg-encoder write one scan per component, which zune-jpeg 0.5.15 (our
    /// decoder) reads back wrongly when chroma is subsampled (see `web::encode_jpeg_rgb8`).
    Optimized,
    /// Progressive: the image refines over several scans as it loads.
    Progressive,
}

impl JpegFormat {
    /// `"baseline"`, `"optimized"` or `"progressive"`, as parameters and scripts spell it.
    pub fn name(self) -> &'static str {
        match self {
            JpegFormat::Baseline => "baseline",
            JpegFormat::Optimized => "optimized",
            JpegFormat::Progressive => "progressive",
        }
    }

    /// The inverse of [`Self::name`].
    pub fn parse(s: &str) -> Option<Self> {
        [JpegFormat::Baseline, JpegFormat::Optimized, JpegFormat::Progressive].into_iter().find(|f| f.name() == s)
    }
}

/// OpenEXR compression (lossless methods only).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ExrCompression {
    None,
    Rle,
    Zip1,
    #[default]
    Zip16,
    Piz,
}

/// Bits per sample of a DPX file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DpxBits {
    Bits8,
    Bits10,
    Bits12,
    Bits16,
}

impl DpxBits {
    /// `8`, `10`, `12` or `16` as [`DpxBits`]; any other number is `None`.
    pub fn from_bits(bits: u64) -> Option<Self> {
        match bits {
            8 => Some(DpxBits::Bits8),
            10 => Some(DpxBits::Bits10),
            12 => Some(DpxBits::Bits12),
            16 => Some(DpxBits::Bits16),
            _ => None,
        }
    }

    /// The bits per sample.
    pub fn bits(self) -> u8 {
        match self {
            DpxBits::Bits8 => 8,
            DpxBits::Bits10 => 10,
            DpxBits::Bits12 => 12,
            DpxBits::Bits16 => 16,
        }
    }
}

/// How a DDS texture stores its pixels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DdsCompression {
    /// Uncompressed, 8 bits per channel: BGRA (BGRX without alpha), or luminance with or
    /// without alpha for grayscale. Lossless.
    #[default]
    None,
    /// BC1 (DXT1): 4 bits per pixel. Alpha is on or off: texels under half opacity become
    /// transparent black.
    Bc1,
    /// BC2 (DXT3): 8 bits per pixel, with 4 bits of explicit alpha per texel.
    Bc2,
    /// BC3 (DXT5): 8 bits per pixel, with smooth interpolated alpha.
    Bc3,
    /// BC4 (ATI1): 4 bits per pixel, a single interpolated channel; stores the gray channel.
    Bc4,
    /// BC5 (ATI2): 8 bits per pixel, two interpolated channels; stores gray and alpha.
    Bc5,
    /// BC7: 8 bits per pixel, partitioned endpoints; the best quality RGBA block format.
    Bc7,
    /// BC1 block data under the DXT1 name, but the header declares `DDPF_ALPHAPIXELS`: the
    /// file promises 1-bit punch-through alpha (DXT1a). Plain `bc1` writes no such promise.
    Bc1a,
    /// Uncompressed 8-bit alpha only (`DDPF_ALPHA`): the file is a single-channel mask.
    A8,
}

impl DdsCompression {
    pub const ALL: [DdsCompression; 9] = [
        DdsCompression::None,
        DdsCompression::Bc1,
        DdsCompression::Bc1a,
        DdsCompression::Bc2,
        DdsCompression::Bc3,
        DdsCompression::Bc4,
        DdsCompression::Bc5,
        DdsCompression::Bc7,
        DdsCompression::A8,
    ];

    /// The name scripts and the CLI use: `none`, `bc1`, `bc1a`, `bc2`, `bc3`, `bc4`, `bc5`,
    /// `bc7` or `a8`.
    pub fn name(self) -> &'static str {
        match self {
            DdsCompression::None => "none",
            DdsCompression::Bc1 => "bc1",
            DdsCompression::Bc1a => "bc1a",
            DdsCompression::Bc2 => "bc2",
            DdsCompression::Bc3 => "bc3",
            DdsCompression::Bc4 => "bc4",
            DdsCompression::Bc5 => "bc5",
            DdsCompression::Bc7 => "bc7",
            DdsCompression::A8 => "a8",
        }
    }

    /// [`Self::name`] back to a compression, also taking the DXT and ATI names (`dxt1`,
    /// `dxt3`, `dxt5`, `ati1`, `ati2`), in any case. Anything else is `None`.
    pub fn from_name(name: &str) -> Option<Self> {
        match name.to_ascii_lowercase().as_str() {
            "none" => Some(DdsCompression::None),
            "bc1" | "dxt1" => Some(DdsCompression::Bc1),
            "bc1a" | "dxt1a" => Some(DdsCompression::Bc1a),
            "bc2" | "dxt3" => Some(DdsCompression::Bc2),
            "bc3" | "dxt5" => Some(DdsCompression::Bc3),
            "bc4" | "ati1" | "bc4u" => Some(DdsCompression::Bc4),
            "bc5" | "ati2" | "bc5u" => Some(DdsCompression::Bc5),
            "bc7" => Some(DdsCompression::Bc7),
            "a8" => Some(DdsCompression::A8),
            _ => None,
        }
    }

    /// A block compression (lossy), not uncompressed pixels.
    pub fn is_block(self) -> bool {
        !matches!(self, DdsCompression::None | DdsCompression::A8)
    }
}

/// Block-compression effort for DDS writing: the bc7enc search depth (`uber` passes and
/// refinements). Only BC7 reads it; BC1–BC5 have a single speed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DdsQuality {
    /// `uber_level` 0, no least-squares pass, the partition filterbank on.
    Fastest,
    /// `uber_level` 2, the partition filterbank on.
    Normal,
    /// `uber_level` 3.
    Production,
    /// The encoder's defaults: `uber_level` 4 with least squares.
    #[default]
    Highest,
}

impl DdsQuality {
    pub const ALL: [DdsQuality; 4] = [DdsQuality::Fastest, DdsQuality::Normal, DdsQuality::Production, DdsQuality::Highest];

    /// The name scripts and the CLI use: `fastest`, `normal`, `production` or `highest`.
    pub fn name(self) -> &'static str {
        match self {
            DdsQuality::Fastest => "fastest",
            DdsQuality::Normal => "normal",
            DdsQuality::Production => "production",
            DdsQuality::Highest => "highest",
        }
    }

    /// [`Self::name`] back to a quality, also taking the shorter `fast`/`fine`, in any case.
    /// Anything else is `None`.
    pub fn from_name(name: &str) -> Option<Self> {
        match name.to_ascii_lowercase().as_str() {
            "fastest" | "fast" => Some(DdsQuality::Fastest),
            "normal" => Some(DdsQuality::Normal),
            "production" => Some(DdsQuality::Production),
            "highest" | "fine" | "finest" => Some(DdsQuality::Highest),
            _ => None,
        }
    }
}

/// The 2:1 filter the mip chain is generated with, after the NVIDIA Texture Tools
/// Exporter's list.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DdsMipFilter {
    /// Each texel the average of the 2×2 it covers (the default).
    #[default]
    Box,
    /// A `[1,3,3,1]` tent: a wider footprint than box, less aliasing.
    Triangle,
    /// Per-channel minimum of the covered texels (height-map min-clips).
    Min,
    /// Per-channel maximum of the covered texels (parallax-occlusion ranges).
    Max,
    /// Kaiser-Bessel windowed sinc (the texture tools' usual best).
    Kaiser,
    /// Mitchell-Netravali cubic, B = C = 1/3.
    MitchellNetravali,
}

impl DdsMipFilter {
    pub const ALL: [DdsMipFilter; 6] =
        [DdsMipFilter::Box, DdsMipFilter::Triangle, DdsMipFilter::Min, DdsMipFilter::Max, DdsMipFilter::Kaiser, DdsMipFilter::MitchellNetravali];

    /// The name scripts and the CLI use.
    pub fn name(self) -> &'static str {
        match self {
            DdsMipFilter::Box => "box",
            DdsMipFilter::Triangle => "triangle",
            DdsMipFilter::Min => "min",
            DdsMipFilter::Max => "max",
            DdsMipFilter::Kaiser => "kaiser",
            DdsMipFilter::MitchellNetravali => "mitchell",
        }
    }

    /// [`Self::name`] back to a filter, in any case. Anything else is `None`.
    pub fn from_name(name: &str) -> Option<Self> {
        match name.to_ascii_lowercase().as_str() {
            "box" => Some(DdsMipFilter::Box),
            "triangle" => Some(DdsMipFilter::Triangle),
            "min" => Some(DdsMipFilter::Min),
            "max" => Some(DdsMipFilter::Max),
            "kaiser" => Some(DdsMipFilter::Kaiser),
            "mitchell" | "mitchell-netravali" | "mitchell_netravali" => Some(DdsMipFilter::MitchellNetravali),
            _ => None,
        }
    }
}

/// Where a DDS cubemap's six faces live, shared by [`DdsImport`] and [`DdsOptions`]. On
/// import `Off` reads only the +X face as a plain texture; on export `Off` writes one.
/// `Layers` moves one layer per face, `Cross` the horizontal-cross sheet — `-X +Z +X -Z`
/// across the middle row with `+Y` above and `-Y` below `+Z`, faces unrotated — as one 4:3
/// image.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DdsCubeMap {
    #[default]
    Off,
    Layers,
    Cross,
}

impl DdsCubeMap {
    /// The name scripts and the CLI use: `off`, `layers` or `cross`.
    pub fn name(self) -> &'static str {
        match self {
            DdsCubeMap::Off => "off",
            DdsCubeMap::Layers => "layers",
            DdsCubeMap::Cross => "cross",
        }
    }

    /// [`Self::name`] back to a choice, also taking `faces` for `layers` and `sheet`/`box`
    /// for `cross`, in any case. Anything else is `None`.
    pub fn from_name(name: &str) -> Option<Self> {
        match name.to_ascii_lowercase().as_str() {
            "off" | "none" | "2d" => Some(DdsCubeMap::Off),
            "layers" | "faces" => Some(DdsCubeMap::Layers),
            "cross" | "sheet" | "box" => Some(DdsCubeMap::Cross),
            _ => None,
        }
    }
}

/// What a DDS file is written with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DdsOptions {
    pub compression: DdsCompression,
    /// Write the mip chain (down to [`Self::min_mip_size`], at most [`Self::max_mip_count`]
    /// levels), each level a [`Self::mip_filter`] of the one above.
    pub mipmaps: bool,
    /// The encoder's search effort (BC7 only).
    pub quality: DdsQuality,
    /// Mark the texture as sRGB. The DX10 header's `_SRGB` DXGI formats express this, so a
    /// marked texture is written with one; luminance, A8, BC4 and BC5 have no sRGB name and
    /// are written unchanged.
    pub srgb: bool,
    /// Which 2:1 filter generates the mip chain.
    pub mip_filter: DdsMipFilter,
    /// Stop the chain before a level whose smaller side falls below this.
    pub min_mip_size: u32,
    /// Cap the chain at this many levels; 0 = no cap.
    pub max_mip_count: u32,
    /// Filter mip levels in linear light when [`Self::srgb`] marks the texture (stored
    /// values are EOTF'd, so averaging them directly darkens the chain). Alpha is never
    /// EOTF'd. Ignored without `srgb`.
    pub mip_gamma_correct: bool,
    /// Filter mip levels in premultiplied space so transparent texels do not bleed their
    /// RGB into the lower levels. A no-op on images with no alpha.
    pub mip_premult_alpha: bool,
    /// Store RGB premultiplied by alpha in the file — engines sampling premultiplied
    /// content expect it (and it is the premultiplied DXT2/DXT4 storage form).
    pub export_premult_alpha: bool,
    /// Set `DDPF_NORMAL` in the pixel-format flags; some engines read it for tangent-space
    /// normal-map detection.
    pub normal_map: bool,
    /// Write rows bottom-up (the GL-vs-D3D origin toggle the GIMP plug-in offers).
    pub flip: bool,
    /// Write a cubemap: `Layers` takes the document's six bottom-most full-canvas square
    /// raster layers as the faces `+X -X +Y -Y +Z -Z` in stack order, `Cross` splits a 4:3
    /// cross-layout sheet. `Off` writes a plain 2D texture.
    pub cube: DdsCubeMap,
}

impl Default for DdsOptions {
    fn default() -> Self {
        DdsOptions {
            compression: DdsCompression::None,
            mipmaps: false,
            quality: DdsQuality::Highest,
            srgb: false,
            mip_filter: DdsMipFilter::Box,
            min_mip_size: 1,
            max_mip_count: 0,
            mip_gamma_correct: true,
            mip_premult_alpha: true,
            export_premult_alpha: false,
            normal_map: false,
            flip: false,
            cube: DdsCubeMap::Off,
        }
    }
}

/// How a DDS texture opens.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DdsImport {
    /// Where the texture's alpha goes: the layer's transparency (the default), or an "Alpha 1"
    /// document channel preserving it as editable data — game textures often carry data such
    /// as specular maps in alpha rather than transparency (the NVIDIA/Intel plug-ins' "load
    /// transparency as alpha channel"). A cubemap's faces open with their own alpha; each
    /// face that has data in alpha also gains a "Face ±X Alpha" channel.
    pub alpha_as_channel: bool,
    /// A layer per mip level (the Intel Texture Works plugin's choice), not the top level only.
    /// A cubemap opens a layer per face per level.
    pub mips_as_layers: bool,
    /// Flip rows vertically on open (the GIMP plug-in's "load flipped" toggle for textures
    /// written for the opposite GL/D3D origin convention).
    pub flip: bool,
    /// Where a cubemap's faces go: a layer per face (the default), the cross-layout sheet,
    /// or `Off` for the +X face as a plain texture.
    pub cube: DdsCubeMap,
}

impl Default for DdsImport {
    fn default() -> Self {
        DdsImport { alpha_as_channel: false, mips_as_layers: false, flip: false, cube: DdsCubeMap::Layers }
    }
}

/// What [`crate::dds_info`] reports about a DDS texture, from its header alone.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DdsInfo {
    /// The top mip level's size.
    pub width: u32,
    pub height: u32,
    /// The declared mip count (1 = the top level only).
    pub levels: u32,
    /// The encoding carries an alpha channel.
    pub alpha: bool,
    /// The compression a save could write back (`bc1a` when the header promises
    /// punch-through alpha); `None` when the encoding has no writable form
    /// (premultiplied, RXGB/YCoCg, float or signed variants).
    pub compression: Option<DdsCompression>,
    /// The DX10 header marks the texture `_SRGB`.
    pub srgb: bool,
    /// The header declares a cubemap (DX10 `MISC_TEXTURECUBE` or the legacy
    /// `DDSCAPS2_CUBEMAP`); [`Self::faces`] counts the stored faces.
    pub cube: bool,
    /// Faces the file stores: 6 for a complete cubemap, fewer when the legacy face bits
    /// mark a partial one, 1 for a plain 2D texture.
    pub faces: u32,
}

/// Options for [`crate::encode`].
#[derive(Debug, Clone)]
pub struct EncodeOptions {
    /// JPEG quality 1..=100 (also AVIF quality when enabled).
    pub jpeg_quality: u8,
    /// JPEG 4:2:0 chroma subsampling (`false` = 4:4:4).
    pub jpeg_chroma_subsampling: bool,
    /// JPEG scan layout (baseline, baseline optimized or progressive).
    pub jpeg_format: JpegFormat,
    pub png_compression: PngCompression,
    /// Write Adam7-interlaced PNG.
    pub png_interlaced: bool,
    /// Write an `sTER` chunk marking the image as a stereo pair (PNG Extensions 1.4.0, §3.6),
    /// so a stereo PNG keeps its mark when saved again (#6063).
    pub png_stereo: Option<PngStereo>,
    /// WebP: lossless (VP8L, via `image-webp`) or lossy (our VP8 encoder, see
    /// `codecs::vp8`) with [`Self::webp_quality`].
    pub webp_lossless: bool,
    /// Lossy WebP quality 0..=100 (the source application's scale; 75–85 is typical for photos).
    pub webp_quality: u8,
    pub tiff_compression: TiffCompression,
    /// Always write TIFF as BigTIFF (8-byte offsets). Without it a TIFF is written as BigTIFF
    /// only when it could pass the 4 GiB a classic TIFF can address.
    pub tiff_bigtiff: bool,
    pub exr_compression: ExrCompression,
    /// DPX bits per sample. `None` follows the image: 8 for 8-bit, 16 otherwise.
    pub dpx_bits: Option<DpxBits>,
    /// DDS compression, mip chain, quality and flag choices.
    pub dds: DdsOptions,
    /// Embed the ICC profile when the format supports it.
    pub embed_icc: bool,
    /// Embed EXIF/XMP/DPI/text when the format supports it.
    pub embed_metadata: bool,
}

impl Default for EncodeOptions {
    fn default() -> Self {
        EncodeOptions {
            jpeg_quality: 90,
            jpeg_chroma_subsampling: true,
            jpeg_format: JpegFormat::Baseline,
            png_compression: PngCompression::Default,
            png_interlaced: false,
            png_stereo: None,
            webp_lossless: true,
            webp_quality: 80,
            tiff_compression: TiffCompression::None,
            tiff_bigtiff: false,
            exr_compression: ExrCompression::Zip16,
            dpx_bits: None,
            dds: DdsOptions::default(),
            embed_icc: true,
            embed_metadata: true,
        }
    }
}

/// A zero-filled buffer of `len` elements, allocated fallibly: an allocation failure becomes
/// [`CodecError::LimitExceeded`] instead of aborting the process (which `catch_unwind` cannot
/// stop).
pub(crate) fn alloc_zeroed<T: Copy + Default>(len: usize) -> Result<Vec<T>, CodecError> {
    let mut v = Vec::new();
    v.try_reserve_exact(len).map_err(|_| CodecError::LimitExceeded(format!("cannot allocate {len} elements for the image")))?;
    v.resize(len, T::default());
    Ok(v)
}
