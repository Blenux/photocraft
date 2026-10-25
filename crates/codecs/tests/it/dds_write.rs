//! Writing DDS textures: every storage read back through the reader, the header fields, sizes
//! that aren't a multiple of four, the mip chain, and the block-compression error on ramps.

use photocraft_codecs::{ChannelLayout, DdsCompression, DdsOptions, EncodeOptions, FidelityWarning, Format, Image, decode, encode, fidelity_warnings_with};

const DDSD_PITCH: u32 = 0x8;
const DDSD_MIPMAPCOUNT: u32 = 0x2_0000;
const DDSD_LINEARSIZE: u32 = 0x8_0000;
const REQUIRED: u32 = 0x1007; // CAPS | HEIGHT | WIDTH | PIXELFORMAT
const DDSCAPS_COMPLEX: u32 = 0x8;
const DDSCAPS_TEXTURE: u32 = 0x1000;
const DDSCAPS_MIPMAP: u32 = 0x40_0000;
const DDPF_ALPHAPIXELS: u32 = 0x1;
const DDPF_FOURCC: u32 = 0x4;
const DDPF_RGB: u32 = 0x40;
const DDPF_LUMINANCE: u32 = 0x2_0000;

const BLOCKS: [DdsCompression; 3] = [DdsCompression::Bc1, DdsCompression::Bc2, DdsCompression::Bc3];

fn opts(compression: DdsCompression, mipmaps: bool) -> EncodeOptions {
    EncodeOptions { dds: DdsOptions { compression, mipmaps, ..Default::default() }, ..Default::default() }
}

fn le32(b: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(b[at..at + 4].try_into().unwrap())
}

/// A ramp along one line through RGB (red up, green up half as fast, blue down) across the
/// texture, and alpha falling from opaque down it: every block's colours lie on a line, which
/// is what a colour block stores.
fn ramp(w: u32, h: u32, layout: ChannelLayout) -> Image {
    let ch = layout.channels();
    let at = |v: u32, n: u32| (v * 255 / (n - 1).max(1)) as u8;
    let mut data = Vec::new();
    for y in 0..h {
        for x in 0..w {
            let r = at(x, w);
            let rgba = [r, 64 + r / 2, 255 - r, 255 - at(y, h)];
            match ch {
                1 => data.push(rgba[0]),
                2 => data.extend([rgba[0], rgba[3]]),
                _ => data.extend(&rgba[..ch]),
            }
        }
    }
    Image::from_u8(w, h, layout, data).unwrap()
}

/// The same RGBA image, fully opaque.
fn opaque(img: &Image) -> Image {
    let data = img.data().chunks(4).flat_map(|p| [p[0], p[1], p[2], 255]).collect();
    Image::from_u8(img.width(), img.height(), ChannelLayout::Rgba, data).unwrap()
}

/// Largest difference in channel `c` (of four) between RGBA images.
fn max_diff_channel(a: &Image, b: &Image, c: usize) -> u8 {
    a.data().chunks(4).zip(b.data().chunks(4)).map(|(x, y)| x[c].abs_diff(y[c])).max().unwrap_or(0)
}

/// Largest colour difference between RGBA images where `b` is visible (BC1 turns texels under
/// half opacity into transparent black).
fn max_colour_diff(a: &Image, b: &Image) -> u8 {
    a.data().chunks(4).zip(b.data().chunks(4)).filter(|(_, y)| y[3] != 0).flat_map(|(x, y)| (0..3).map(|c| x[c].abs_diff(y[c]))).max().unwrap_or(0)
}

/// A 2D gradient: red across, green down and blue along the diagonal, opaque.
fn gradient(w: u32, h: u32) -> Image {
    let data = (0..h).flat_map(|y| (0..w).flat_map(move |x| [(x * 4).min(255) as u8, (y * 4).min(255) as u8, ((x + y) * 2).min(255) as u8, 255])).collect();
    Image::from_u8(w, h, ChannelLayout::Rgba, data).unwrap()
}

/// PSNR of the colour (not the alpha) of two RGBA images.
fn psnr(a: &Image, b: &Image) -> f64 {
    let rgb = |i: &Image| i.data().chunks(4).flat_map(|p| [p[0], p[1], p[2]]).map(f64::from).collect::<Vec<_>>();
    let (a, b) = (rgb(a), rgb(b));
    let mse = a.iter().zip(&b).map(|(x, y)| (x - y).powi(2)).sum::<f64>() / a.len() as f64;
    if mse == 0.0 { f64::INFINITY } else { 10.0 * (255.0f64.powi(2) / mse).log10() }
}

/// Bytes of one level as a reader computes them from the header's pixel format.
fn level_bytes(file: &[u8], w: usize, h: usize) -> usize {
    if le32(file, 80) & DDPF_FOURCC != 0 {
        // Eight-byte blocks for BC1 (DXT1) and BC4 (DXGI 80); sixteen for the rest.
        let block = if &file[84..88] == b"DXT1" || (&file[84..88] == b"DX10" && le32(file, 128) == 80) { 8 } else { 16 };
        w.div_ceil(4) * h.div_ceil(4) * block
    } else {
        w * h * le32(file, 88) as usize / 8
    }
}

/// Mip level `i` of a written texture, cut out as a single-level texture and decoded.
fn level(file: &[u8], i: u32) -> Image {
    let head = if &file[84..88] == b"DX10" { 148 } else { 128 };
    let (w, h) = (le32(file, 16) as usize, le32(file, 12) as usize);
    let size = |i: u32| ((w >> i).max(1), (h >> i).max(1));
    let start = head + (0..i).map(|j| level_bytes(file, size(j).0, size(j).1)).sum::<usize>();
    let (lw, lh) = size(i);
    let mut b = file[..head].to_vec();
    b[12..16].copy_from_slice(&(lh as u32).to_le_bytes());
    b[16..20].copy_from_slice(&(lw as u32).to_le_bytes());
    b[28..32].copy_from_slice(&1u32.to_le_bytes());
    // The pitch field is a row stride for uncompressed textures and the level size for BC.
    let pitch = if le32(file, 80) & DDPF_FOURCC != 0 { level_bytes(file, lw, lh) } else { lw * le32(file, 88) as usize / 8 };
    b[20..24].copy_from_slice(&(pitch as u32).to_le_bytes());
    b.extend(&file[start..start + level_bytes(file, lw, lh)]);
    decode(&b).unwrap()
}

#[test]
fn uncompressed_textures_round_trip_every_layout_exactly() {
    for (layout, pf_flags, bits, masks) in [
        (ChannelLayout::Gray, DDPF_LUMINANCE, 8, [0xff, 0, 0, 0]),
        (ChannelLayout::GrayA, DDPF_LUMINANCE | DDPF_ALPHAPIXELS, 16, [0xff, 0, 0, 0xff00]),
        (ChannelLayout::Rgb, DDPF_RGB, 32, [0xff_0000, 0xff00, 0xff, 0]),
        (ChannelLayout::Rgba, DDPF_RGB | DDPF_ALPHAPIXELS, 32, [0xff_0000, 0xff00, 0xff, 0xff00_0000]),
    ] {
        for (w, h) in [(1, 1), (5, 3), (17, 9)] {
            let img = ramp(w, h, layout);
            let b = encode(&img, Format::Dds, &EncodeOptions::default()).unwrap();
            let back = decode(&b).unwrap();
            assert_eq!(back.layout(), layout, "{layout:?} {w}x{h}");
            assert_eq!(back.data(), img.data(), "{layout:?} {w}x{h}");

            assert_eq!((&b[..4], le32(&b, 4), le32(&b, 76)), (&b"DDS "[..], 124, 32));
            assert_eq!(le32(&b, 8), REQUIRED | DDSD_PITCH, "pitch, not linear size, and no mip count");
            assert_eq!((le32(&b, 12), le32(&b, 16)), (h, w));
            assert_eq!(le32(&b, 20), w * bits / 8, "pitch of {layout:?} {w}x{h}");
            assert_eq!((le32(&b, 28), le32(&b, 108)), (1, DDSCAPS_TEXTURE));
            assert_eq!((le32(&b, 80), le32(&b, 88)), (pf_flags, bits));
            assert_eq!([le32(&b, 92), le32(&b, 96), le32(&b, 100), le32(&b, 104)], masks);
            assert_eq!(b.len(), 128 + (w * h * bits / 8) as usize);
        }
    }
}

#[test]
fn block_compressed_textures_round_trip_through_the_reader() {
    for (compression, fourcc, block) in [(DdsCompression::Bc1, b"DXT1", 8), (DdsCompression::Bc2, b"DXT3", 16), (DdsCompression::Bc3, b"DXT5", 16)] {
        // Not multiples of four, and the 1×1 and 2×2 textures smaller than one block.
        for (w, h) in [(1, 1), (2, 2), (4, 4), (5, 3), (7, 13), (33, 17)] {
            let img = ramp(w, h, ChannelLayout::Rgba);
            let b = encode(&img, Format::Dds, &opts(compression, false)).unwrap();
            let linear = w.div_ceil(4) * h.div_ceil(4) * block;
            assert_eq!(&b[84..88], fourcc);
            assert_eq!((le32(&b, 80), le32(&b, 88)), (DDPF_FOURCC, 0));
            assert_eq!(le32(&b, 8), REQUIRED | DDSD_LINEARSIZE, "linear size, not pitch");
            assert_eq!(le32(&b, 20), linear, "{compression:?} {w}x{h}: the top level's size");
            assert_eq!(b.len(), 128 + linear as usize);

            let back = decode(&b).unwrap();
            assert_eq!((back.dimensions(), back.layout()), ((w, h), ChannelLayout::Rgba));
            // Steep ramps across tiny textures put three colours where a block interpolates at
            // thirds, so this bound is loose; it still catches swapped channels or misplaced
            // blocks. `block_compression_error_on_gradients_is_bounded` is the tight one. A BC1
            // block with transparent texels has only three colours, so BC1's colours are
            // checked on the opaque ramp.
            let (src, back_colour) = if compression == DdsCompression::Bc1 {
                let o = opaque(&img);
                let back = decode(&encode(&o, Format::Dds, &opts(compression, false)).unwrap()).unwrap();
                (o, back)
            } else {
                (img.clone(), back.clone())
            };
            let colour = max_colour_diff(&src, &back_colour);
            assert!(colour <= 24, "{compression:?} {w}x{h}: colour off by {colour}");
            let alpha = max_diff_channel(&img, &back, 3);
            match compression {
                DdsCompression::Bc1 => assert!(back.data().chunks(4).zip(img.data().chunks(4)).all(|(b, i)| b[3] == if i[3] < 128 { 0 } else { 255 })),
                _ => assert!(alpha <= 8, "{compression:?} {w}x{h}: alpha off by {alpha}"),
            }
        }
    }
}

#[test]
fn bc4_bc5_and_bc7_write_dx10_files_the_reader_reads_back() {
    // BC4 stores the gray channel, BC5 gray and alpha, BC7 the full RGBA; none of them
    // has a legacy FourCC, so each file carries the DX10 header naming its DXGI format.
    for (compression, layout, dxgi) in [(DdsCompression::Bc4, ChannelLayout::Gray, 80), (DdsCompression::Bc5, ChannelLayout::GrayA, 83)] {
        for (w, h) in [(1, 1), (4, 4), (5, 3), (7, 13)] {
            let img = ramp(w, h, layout);
            let b = encode(&img, Format::Dds, &opts(compression, false)).unwrap();
            let block = if compression == DdsCompression::Bc4 { 8 } else { 16 };
            let linear = w.div_ceil(4) * h.div_ceil(4) * block;
            assert_eq!(&b[84..88], b"DX10", "{compression:?} {w}x{h}");
            assert_eq!([le32(&b, 128), le32(&b, 132), le32(&b, 136), le32(&b, 140)], [dxgi, 3, 0, 1]);
            assert_eq!(le32(&b, 20), linear, "{compression:?} {w}x{h}: the top level's size");
            assert_eq!(b.len(), 148 + linear as usize);

            let back = decode(&b).unwrap();
            if compression == DdsCompression::Bc4 {
                // One interpolated channel comes back as grayscale, within a few levels.
                assert_eq!(back.layout(), ChannelLayout::Gray);
                let err = back.data().iter().zip(img.data()).map(|(a, b)| a.abs_diff(*b)).max().unwrap_or(0);
                assert!(err <= 16, "{compression:?} {w}x{h}: gray off by {err}");
            } else {
                // BC5 reads back as RGB: gray in red, alpha in green, black blue.
                assert_eq!(back.layout(), ChannelLayout::Rgb);
                for (px, src) in back.data().chunks(3).zip(img.data().chunks(2)) {
                    assert!(px[0].abs_diff(src[0]) <= 16 && px[1].abs_diff(src[1]) <= 16 && px[2] == 0, "{compression:?} {w}x{h}: {px:?} for {src:?}");
                }
            }
        }
    }
}

#[test]
fn bc7_writes_a_dx10_file_with_the_best_block_quality() {
    for (w, h) in [(1, 1), (4, 4), (5, 3), (7, 13), (33, 17)] {
        let img = ramp(w, h, ChannelLayout::Rgba);
        let b = encode(&img, Format::Dds, &opts(DdsCompression::Bc7, false)).unwrap();
        assert_eq!(&b[84..88], b"DX10");
        assert_eq!([le32(&b, 128), le32(&b, 132), le32(&b, 136), le32(&b, 140)], [98, 3, 0, 1]);
        assert_eq!(b.len(), 148 + w.div_ceil(4) as usize * h.div_ceil(4) as usize * 16);

        let back = decode(&b).unwrap();
        assert_eq!((back.dimensions(), back.layout()), ((w, h), ChannelLayout::Rgba));
        let colour = max_colour_diff(&img, &back);
        assert!(colour <= 12, "{w}x{h}: colour off by {colour}");
        // BC7's separate alpha endpoints are five bits with a four-value selector ramp, so
        // a full-range alpha gradient inside one block can land a third of a ramp apart.
        let alpha = max_diff_channel(&img, &back, 3);
        assert!(alpha <= 40, "{w}x{h}: alpha off by {alpha}");
    }
    // BC7's partitioned endpoints beat BC3 on a 2D gradient.
    let (line, plane) = (ramp(64, 64, ChannelLayout::Rgba), gradient(64, 64));
    let roundtrip = |img: &Image| decode(&encode(img, Format::Dds, &opts(DdsCompression::Bc7, false)).unwrap()).unwrap();
    let (line_back, plane_back) = (roundtrip(&line), roundtrip(&plane));
    assert!(max_colour_diff(&line, &line_back) <= 4, "colour ramp off by {}", max_colour_diff(&line, &line_back));
    assert!(psnr(&plane, &plane_back) >= 42.0, "2D gradient PSNR {:.1} dB", psnr(&plane, &plane_back));
}

#[test]
fn dx10_block_stores_have_a_block_per_small_mip_level() {
    for (compression, layout, block) in
        [(DdsCompression::Bc4, ChannelLayout::Gray, 8), (DdsCompression::Bc5, ChannelLayout::GrayA, 16), (DdsCompression::Bc7, ChannelLayout::Rgba, 16)]
    {
        let img = ramp(12, 6, layout);
        let b = encode(&img, Format::Dds, &opts(compression, true)).unwrap();
        assert_eq!(&b[84..88], b"DX10");
        assert_eq!(le32(&b, 28), 4);
        assert_eq!(b.len(), 148 + (6 + 2 + 1 + 1) * block);
        for i in 0..4 {
            assert_eq!(level(&b, i).dimensions(), ((12 >> i).max(1), (6 >> i).max(1)), "{compression:?} level {i}");
        }
    }
}

#[test]
fn gray_and_rgb_sources_are_written_as_rgba_blocks() {
    for layout in [ChannelLayout::Gray, ChannelLayout::GrayA, ChannelLayout::Rgb] {
        for compression in BLOCKS {
            let img = ramp(8, 8, layout);
            let back = decode(&encode(&img, Format::Dds, &opts(compression, false)).unwrap()).unwrap();
            let want = img.convert(ChannelLayout::Rgba, img.sample_type());
            assert_eq!(back.layout(), ChannelLayout::Rgba);
            let colour = max_colour_diff(&want, &back);
            assert!(colour <= 8, "{layout:?} {compression:?}: colour off by {colour}");
        }
    }
}

#[test]
fn block_compression_error_on_gradients_is_bounded() {
    let (line, plane) = (ramp(64, 64, ChannelLayout::Rgba), gradient(64, 64));
    for compression in BLOCKS {
        let roundtrip = |img: &Image| decode(&encode(img, Format::Dds, &opts(compression, false)).unwrap()).unwrap();
        let (line_back, plane_back) = (roundtrip(&line), roundtrip(&plane));
        let (line_err, plane_err) = (max_colour_diff(&line, &line_back), max_colour_diff(&plane, &plane_back));
        let p = psnr(&plane, &plane_back);
        let alpha = max_diff_channel(&line, &line_back, 3);
        // Colours on a line only lose the RGB565 rounding of the endpoints (up to 4 levels in
        // red and blue). A gradient in two directions is not on a line: the spread off a
        // block's best line alone limits this one to about 40 dB, and RGB565 costs a little
        // more (the encoder measures 38.6 dB and 9 levels at most).
        assert!(line_err <= 5, "{compression:?}: colour ramp off by {line_err}");
        assert!(plane_err <= 10, "{compression:?}: 2D gradient off by {plane_err}");
        assert!(p >= 38.0, "{compression:?}: 2D gradient PSNR {p:.1} dB");
        match compression {
            DdsCompression::Bc1 => assert!(line_back.data().chunks(4).zip(line.data().chunks(4)).all(|(b, i)| b[3] == if i[3] < 128 { 0 } else { 255 })),
            // Four bits: within half a step of 17.
            DdsCompression::Bc2 => assert!(alpha <= 8, "BC2 alpha off by {alpha}"),
            // Eight values across a block's few levels of alpha.
            _ => assert!(alpha <= 2, "BC3 alpha off by {alpha}"),
        }
    }
}

#[test]
fn solid_and_cutout_textures_keep_their_values() {
    // RGB565-exact colours and BC2's 4-bit alpha steps come back exactly.
    let solid = Image::from_u8(6, 5, ChannelLayout::Rgba, [255, 0, 255, 170].repeat(30)).unwrap();
    for compression in BLOCKS {
        let back = decode(&encode(&solid, Format::Dds, &opts(compression, false)).unwrap()).unwrap();
        let want = if compression == DdsCompression::Bc1 { [255, 0, 255, 255] } else { [255, 0, 255, 170] };
        assert!(back.data().chunks(4).all(|p| p == want), "{compression:?}: {:?}", &back.data()[..4]);
    }
    // A cutout: BC1's transparent texels decode as transparent black, the rest stay opaque.
    let cutout: Vec<u8> = (0..36).flat_map(|i| if (i / 6 + i % 6) % 2 == 0 { [200, 40, 40, 255] } else { [10, 10, 10, 0] }).collect();
    let cutout = Image::from_u8(6, 6, ChannelLayout::Rgba, cutout).unwrap();
    for compression in BLOCKS {
        let back = decode(&encode(&cutout, Format::Dds, &opts(compression, false)).unwrap()).unwrap();
        assert_eq!(max_diff_channel(&cutout, &back, 3), 0, "{compression:?}");
        if compression == DdsCompression::Bc1 {
            assert!(back.data().chunks(4).filter(|p| p[3] == 0).all(|p| p == [0, 0, 0, 0]));
        }
    }
}

#[test]
fn mipmaps_write_the_full_chain_box_filtered() {
    // 16×8 halves to 8×4, 4×2, 2×1 and 1×1: five levels.
    let img = ramp(16, 8, ChannelLayout::Rgba);
    // Premultiplied filtering is the default; this test pins the straight box filter.
    let mut o = opts(DdsCompression::None, true);
    o.dds.mip_premult_alpha = false;
    let b = encode(&img, Format::Dds, &o).unwrap();
    assert_eq!(le32(&b, 8), REQUIRED | DDSD_PITCH | DDSD_MIPMAPCOUNT);
    assert_eq!(le32(&b, 28), 5);
    assert_eq!(le32(&b, 108), DDSCAPS_TEXTURE | DDSCAPS_COMPLEX | DDSCAPS_MIPMAP);
    assert_eq!(b.len(), 128 + 4 * (16 * 8 + 8 * 4 + 4 * 2 + 2 + 1));
    assert_eq!(decode(&b).unwrap().data(), img.data(), "the reader still reads the top level");
    let mut above = img.clone();
    for i in 1..5 {
        let got = level(&b, i);
        assert_eq!(got.dimensions(), ((16 >> i).max(1), (8 >> i).max(1)));
        let want = box_filter(&above);
        assert_eq!(got.data(), want.data(), "level {i}");
        above = got;
    }
    // 1×1 has a single level.
    let one = encode(&ramp(1, 1, ChannelLayout::Rgba), Format::Dds, &opts(DdsCompression::None, true)).unwrap();
    assert_eq!((le32(&one, 28), le32(&one, 8) & DDSD_MIPMAPCOUNT), (1, 0));
}

/// The documented filter, written independently: each pixel averages the 2×2 pixels it covers,
/// and the last column and row of an odd-sized level take in the leftover pixel too.
fn box_filter(src: &Image) -> Image {
    let (w, h) = (src.width() as usize, src.height() as usize);
    let (nw, nh) = ((w / 2).max(1), (h / 2).max(1));
    let ch = src.layout().channels();
    let span = |i: usize, n: usize, size: usize| (2 * i, if i + 1 == n { size } else { 2 * i + 2 });
    let mut out = Vec::new();
    for y in 0..nh {
        for x in 0..nw {
            let ((x0, x1), (y0, y1)) = (span(x, nw, w), span(y, nh, h));
            for c in 0..ch {
                let values: Vec<u32> =
                    (y0..y1).flat_map(|sy| (x0..x1).map(move |sx| (sy, sx))).map(|(sy, sx)| u32::from(src.data()[(sy * w + sx) * ch + c])).collect();
                let n = values.len() as u32;
                out.push(((values.iter().sum::<u32>() + n / 2) / n) as u8);
            }
        }
    }
    Image::from_u8(nw as u32, nh as u32, src.layout(), out).unwrap()
}

#[test]
fn odd_sized_mip_chains_keep_every_pixel() {
    // 5×3 → 2×1 → 1×1. The 2×1 level's right pixel averages columns 2–4 of all three rows.
    let img = ramp(5, 3, ChannelLayout::Gray);
    let b = encode(&img, Format::Dds, &opts(DdsCompression::None, true)).unwrap();
    assert_eq!(le32(&b, 28), 3);
    assert_eq!(b.len(), 128 + 5 * 3 + 2 + 1);
    let second = level(&b, 1);
    assert_eq!(second.data(), box_filter(&img).data());
    let px = |x: usize, y: usize| u32::from(img.data()[y * 5 + x]);
    let right: u32 = (0..3).flat_map(|y| (2..5).map(move |x| (x, y))).map(|(x, y)| px(x, y)).sum();
    assert_eq!(u32::from(second.data()[1]), (right + 4) / 9);
}

#[test]
fn block_compressed_mip_chains_have_a_block_per_small_level() {
    for compression in BLOCKS {
        let block = if compression == DdsCompression::Bc1 { 8 } else { 16 };
        let img = ramp(12, 6, ChannelLayout::Rgba);
        let b = encode(&img, Format::Dds, &opts(compression, true)).unwrap();
        // 12×6, 6×3, 3×1, 1×1: 3×2, 2×1, 1 and 1 blocks.
        assert_eq!(le32(&b, 28), 4);
        assert_eq!(le32(&b, 8), REQUIRED | DDSD_LINEARSIZE | DDSD_MIPMAPCOUNT);
        assert_eq!(le32(&b, 20), 6 * block);
        assert_eq!(b.len(), 128 + (6 + 2 + 1 + 1) * block as usize);
        for i in 0..4 {
            let got = level(&b, i);
            assert_eq!(got.dimensions(), ((12 >> i).max(1), (6 >> i).max(1)), "{compression:?} level {i}");
        }
    }
}

#[test]
fn strips_and_tiny_textures_encode_with_every_option() {
    for (w, h) in [(1, 1), (1, 9), (9, 1), (2, 3), (3, 2)] {
        for compression in DdsCompression::ALL {
            for mipmaps in [false, true] {
                let img = ramp(w, h, ChannelLayout::Rgba);
                let b = encode(&img, Format::Dds, &opts(compression, mipmaps)).unwrap_or_else(|e| panic!("{w}x{h} {compression:?} {mipmaps}: {e}"));
                assert_eq!(decode(&b).unwrap().dimensions(), (w, h));
            }
        }
    }
}

#[test]
fn fidelity_warnings_follow_the_compression() {
    let translucent = ramp(8, 8, ChannelLayout::Rgba);
    let warns = |c| fidelity_warnings_with(&translucent, Format::Dds, &opts(c, false));
    assert!(warns(DdsCompression::None).is_empty(), "uncompressed RGBA loses nothing");
    for c in DdsCompression::ALL {
        if c.is_block() {
            assert!(warns(c).contains(&FidelityWarning::LossyCompression), "{c:?}");
        } else {
            assert!(!warns(c).contains(&FidelityWarning::LossyCompression), "{c:?}");
        }
    }
    assert!(warns(DdsCompression::Bc1).contains(&FidelityWarning::AlphaBinarized));
    assert!(!warns(DdsCompression::Bc3).contains(&FidelityWarning::AlphaBinarized));
}

#[test]
fn compression_names_round_trip() {
    for c in DdsCompression::ALL {
        assert_eq!(DdsCompression::from_name(c.name()), Some(c));
    }
    assert_eq!(DdsCompression::from_name("DXT5"), Some(DdsCompression::Bc3));
    assert_eq!(DdsCompression::from_name("dxt1"), Some(DdsCompression::Bc1));
    for bad in ["", "bc6h", "dxt2", "uncompressed "] {
        assert_eq!(DdsCompression::from_name(bad), None, "{bad:?}");
    }
    for q in photocraft_codecs::DdsQuality::ALL {
        assert_eq!(photocraft_codecs::DdsQuality::from_name(q.name()), Some(q));
    }
    for bad in ["", "best", "Fast "] {
        assert_eq!(photocraft_codecs::DdsQuality::from_name(bad), None, "{bad:?}");
    }
}

#[test]
fn srgb_marks_the_dxgi_format_and_forces_the_dx10_header() {
    let img = ramp(8, 8, ChannelLayout::Rgba);
    for (compression, dxgi) in [
        (DdsCompression::Bc1, 72),  // BC1_UNORM_SRGB
        (DdsCompression::Bc2, 75),  // BC2_UNORM_SRGB
        (DdsCompression::Bc3, 78),  // BC3_UNORM_SRGB
        (DdsCompression::Bc7, 99),  // BC7_UNORM_SRGB
        (DdsCompression::None, 91), // B8G8R8A8_UNORM_SRGB
    ] {
        let mut o = opts(compression, false);
        o.dds.srgb = true;
        let b = encode(&img, Format::Dds, &o).unwrap();
        assert_eq!(&b[84..88], b"DX10", "{compression:?}: sRGB has no legacy name");
        assert_eq!(le32(&b, 128), dxgi, "{compression:?}");
        assert_eq!(decode(&b).unwrap().dimensions(), (8, 8), "{compression:?}");
    }
    // BC4 and BC5 have no sRGB DXGI name: the mark changes nothing for them.
    for compression in [DdsCompression::Bc4, DdsCompression::Bc5] {
        let img = ramp(8, 8, if compression == DdsCompression::Bc4 { ChannelLayout::Gray } else { ChannelLayout::GrayA });
        let mut o = opts(compression, false);
        o.dds.srgb = true;
        let b = encode(&img, Format::Dds, &o).unwrap();
        assert_eq!(le32(&b, 128), if compression == DdsCompression::Bc4 { 80 } else { 83 }, "{compression:?}");
    }
}

#[test]
fn bc7_fast_and_fine_both_decode() {
    let img = gradient(16, 16);
    for quality in photocraft_codecs::DdsQuality::ALL {
        let mut o = opts(DdsCompression::Bc7, false);
        o.dds.quality = quality;
        let b = encode(&img, Format::Dds, &o).unwrap();
        assert_eq!(&b[84..88], b"DX10");
        let back = decode(&b).unwrap();
        assert!(psnr(&img, &back) >= 30.0, "{quality:?}: PSNR {:.1} dB", psnr(&img, &back));
    }
}

#[test]
fn decode_dds_levels_reads_every_mip() {
    let img = ramp(16, 8, ChannelLayout::Rgba);
    let mut o = opts(DdsCompression::None, true);
    o.dds.mip_premult_alpha = false;
    let b = encode(&img, Format::Dds, &o).unwrap();
    let levels = photocraft_codecs::decode_dds_levels(&b, &photocraft_codecs::DecodeOptions::default()).unwrap();
    assert_eq!(levels.len(), 5);
    let mut above = img.clone();
    for (i, got) in levels.iter().enumerate() {
        assert_eq!(got.dimensions(), ((16 >> i).max(1), (8 >> i).max(1)), "level {i}");
        if i > 0 {
            assert_eq!(got.data(), box_filter(&above).data(), "level {i}");
        }
        above = got.clone();
    }
    // The plain reader still reads the top level only.
    assert_eq!(decode(&b).unwrap().data(), img.data());
}

#[test]
fn dds_info_reports_size_levels_and_alpha() {
    let b = encode(&ramp(16, 8, ChannelLayout::Rgba), Format::Dds, &opts(DdsCompression::Bc3, true)).unwrap();
    let info = photocraft_codecs::dds_info(&b).unwrap();
    assert_eq!((info.width, info.height, info.levels, info.alpha), (16, 8, 5, true));
    let flat = encode(&ramp(4, 4, ChannelLayout::Gray), Format::Dds, &EncodeOptions::default()).unwrap();
    let info = photocraft_codecs::dds_info(&flat).unwrap();
    assert_eq!((info.width, info.height, info.levels, info.alpha), (4, 4, 1, false));
    assert!(photocraft_codecs::dds_info(b"not a dds").is_err());
}

/// Encode timings on a synthetic ~30 MP texture (6336×4752). Ignored by default; run in release:
///
/// ```sh
/// cargo test --release -p photocraft-codecs --test it -- dds_write:: --ignored --nocapture
/// ```
#[test]
#[ignore]
fn dds_encode_timings_on_30_mp() {
    let (w, h) = (6336u32, 4752u32);
    let mut s = 0x1234_5678u32;
    let mut data = Vec::with_capacity(w as usize * h as usize * 4);
    for y in 0..h {
        for x in 0..w {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            data.extend([(x / 25 + y / 19) as u8, (x / 31) as u8 ^ (s as u8 & 3), (y / 23) as u8, (255 - x / 25) as u8]);
        }
    }
    let img = Image::from_u8(w, h, ChannelLayout::Rgba, data).unwrap();
    for compression in DdsCompression::ALL {
        for mipmaps in [false, true] {
            let t = std::time::Instant::now();
            let b = encode(&img, Format::Dds, &opts(compression, mipmaps)).unwrap();
            eprintln!("{compression:?} mipmaps={mipmaps}: {:.0} ms, {} MB", t.elapsed().as_secs_f64() * 1e3, b.len() >> 20);
        }
    }
}

const DDPF_ALPHA: u32 = 0x2;
const DDPF_NORMAL: u32 = 0x8000_0000;

fn dds_opts(compression: DdsCompression) -> DdsOptions {
    DdsOptions { compression, ..Default::default() }
}

fn with(o: DdsOptions) -> EncodeOptions {
    EncodeOptions { dds: o, ..Default::default() }
}

/// The source rows, bottom-up.
fn flipped(img: &Image) -> Vec<u8> {
    let w = img.width() as usize;
    let ch = img.layout().channels();
    img.data().chunks(w * ch).rev().flatten().copied().collect()
}

#[test]
fn flip_writes_the_rows_bottom_up() {
    let img = ramp(4, 4, ChannelLayout::Rgba);
    let b = encode(&img, Format::Dds, &with(DdsOptions { flip: true, ..dds_opts(DdsCompression::None) })).unwrap();
    assert_eq!(decode(&b).unwrap().data(), flipped(&img).as_slice());
    // Lossy storage too: a flipped BC3 chain decodes to the flipped ramp within its error.
    let b = encode(&img, Format::Dds, &with(DdsOptions { flip: true, compression: DdsCompression::Bc3, mipmaps: true, ..Default::default() })).unwrap();
    let flipped_img = Image::from_u8(4, 4, ChannelLayout::Rgba, flipped(&img)).unwrap();
    assert!(max_diff_channel(&decode(&b).unwrap(), &flipped_img, 3) <= 8, "flipped alpha");
    assert!(max_colour_diff(&flipped_img, &decode(&b).unwrap()) <= 8, "flipped colour");
}

#[test]
fn normal_map_marks_the_pixel_format_flags() {
    let img = ramp(4, 4, ChannelLayout::Rgba);
    let b = encode(&img, Format::Dds, &with(DdsOptions { normal_map: true, ..dds_opts(DdsCompression::Bc5) })).unwrap();
    assert_eq!(le32(&b, 80) & DDPF_NORMAL, DDPF_NORMAL);
    let b = encode(&img, Format::Dds, &opts(DdsCompression::Bc5, false)).unwrap();
    assert_eq!(le32(&b, 80) & DDPF_NORMAL, 0, "not asked, not set");
}

#[test]
fn bc1a_promises_punch_through_alpha() {
    let img = ramp(4, 4, ChannelLayout::Rgba);
    let b = encode(&img, Format::Dds, &opts(DdsCompression::Bc1a, false)).unwrap();
    assert_eq!(&b[84..88], b"DXT1");
    assert_eq!(le32(&b, 80) & DDPF_ALPHAPIXELS, DDPF_ALPHAPIXELS);
    let info = photocraft_codecs::dds_info(&b).unwrap();
    assert_eq!(info.compression, Some(DdsCompression::Bc1a), "reads back as bc1a");
    let b = encode(&img, Format::Dds, &opts(DdsCompression::Bc1, false)).unwrap();
    assert_eq!(le32(&b, 80) & DDPF_ALPHAPIXELS, 0, "bc1 makes no promise");
    assert_eq!(photocraft_codecs::dds_info(&b).unwrap().compression, Some(DdsCompression::Bc1));
}

#[test]
fn a8_writes_an_alpha_only_mask() {
    let img = ramp(4, 4, ChannelLayout::Rgba);
    let b = encode(&img, Format::Dds, &opts(DdsCompression::A8, false)).unwrap();
    assert_eq!(le32(&b, 80) & DDPF_ALPHA, DDPF_ALPHA, "DDPF_ALPHA pixel format");
    assert_eq!(b.len(), 128 + 16, "one byte per pixel");
    let back = decode(&b).unwrap();
    // The file is read as black RGBA whose alpha is the source's.
    assert_eq!(back.layout(), ChannelLayout::Rgba);
    assert!(back.data().chunks(4).all(|p| p[..3] == [0, 0, 0]));
    assert_eq!(back.data().chunks(4).map(|p| p[3]).collect::<Vec<_>>(), img.data().chunks(4).map(|p| p[3]).collect::<Vec<_>>());
    let info = photocraft_codecs::dds_info(&b).unwrap();
    assert_eq!(info.compression, Some(DdsCompression::A8));
}

#[test]
fn min_mip_size_and_max_mip_count_stop_the_chain() {
    let img = ramp(16, 8, ChannelLayout::Rgba);
    // Halving 16×8: 8×4, 4×2, 2×1, 1×1; a level whose smaller side would fall below 4 is
    // never written, leaving 16×8 and 8×4.
    let mut o = opts(DdsCompression::None, true);
    o.dds.min_mip_size = 4;
    let b = encode(&img, Format::Dds, &o).unwrap();
    assert_eq!(le32(&b, 28), 2, "16×8 and 8×4");
    let mut o = opts(DdsCompression::None, true);
    o.dds.min_mip_size = 2;
    let b = encode(&img, Format::Dds, &o).unwrap();
    assert_eq!(le32(&b, 28), 3, "16×8, 8×4 and 4×2");
    let mut o = opts(DdsCompression::None, true);
    o.dds.max_mip_count = 2;
    let b = encode(&img, Format::Dds, &o).unwrap();
    assert_eq!(le32(&b, 28), 2, "16×8 and 8×4");
    // A cap of 1 is a single level, as no chain at all.
    let mut o = opts(DdsCompression::None, true);
    o.dds.max_mip_count = 1;
    let b = encode(&img, Format::Dds, &o).unwrap();
    assert_eq!(le32(&b, 28), 1);
}

#[test]
fn min_max_filters_take_the_extremes() {
    // A 4×4 ramp; Min and Max take the per-channel extrema of each 2×2.
    let img = ramp(4, 4, ChannelLayout::Gray);
    for (filter, want) in [(photocraft_codecs::DdsMipFilter::Min, 0u8), (photocraft_codecs::DdsMipFilter::Max, 85u8)] {
        let mut o = opts(DdsCompression::None, true);
        o.dds.mip_filter = filter;
        let b = encode(&img, Format::Dds, &o).unwrap();
        let second = level(&b, 1);
        assert_eq!(second.dimensions(), (2, 2), "{filter:?}");
        // Top-left texel covers pixels (0..2, 0..2): the ramp's red channel is 0 at x=0.
        assert_eq!(second.data()[0], want, "{filter:?} takes the {want}-extreme of the 2×2");
    }
}

#[test]
fn triangle_and_windowed_filters_smooth_instead_of_averaging() {
    let img = gradient(8, 8);
    let mut box_o = opts(DdsCompression::None, true);
    box_o.dds.mip_premult_alpha = false;
    let boxed = level(&encode(&img, Format::Dds, &box_o).unwrap(), 1);
    for filter in [photocraft_codecs::DdsMipFilter::Triangle, photocraft_codecs::DdsMipFilter::Kaiser, photocraft_codecs::DdsMipFilter::MitchellNetravali] {
        let mut o = opts(DdsCompression::None, true);
        o.dds.mip_filter = filter;
        o.dds.mip_premult_alpha = false;
        let b = encode(&img, Format::Dds, &o).unwrap();
        let second = level(&b, 1);
        assert_eq!(second.dimensions(), (4, 4), "{filter:?}");
        assert!(max_colour_diff(&boxed, &second) < 32, "{filter:?} stays near the box average");
    }
}

#[test]
fn gamma_correct_mips_filter_srgb_in_linear_light() {
    // A black/white checkerboard: a straight average is 50% gray (128), a linear-light average
    // stored back as sRGB is ~188. The texture must be marked sRGB for the EOTF to apply.
    let data = (0..8).flat_map(|y| (0..8).map(move |x| if (x + y) % 2 == 0 { 0u8 } else { 255 }).flat_map(|v| [v, v, v, 255])).collect();
    let img = Image::from_u8(8, 8, ChannelLayout::Rgba, data).unwrap();
    let levels = |o: &EncodeOptions| photocraft_codecs::decode_dds_levels(&encode(&img, Format::Dds, o).unwrap(), &Default::default()).unwrap();
    let mut raw = opts(DdsCompression::None, true);
    raw.dds.srgb = true;
    raw.dds.mip_gamma_correct = false;
    let raw = levels(&raw)[1].clone();
    let mut lin = opts(DdsCompression::None, true);
    lin.dds.srgb = true;
    let lin = levels(&lin)[1].clone();
    assert!(raw.data()[0].abs_diff(128) <= 4, "raw average is half: {}", raw.data()[0]);
    assert!(lin.data()[0] >= 175, "linear-light average is ~186–188: {}", lin.data()[0]);
    // Without the sRGB mark the flag is inert.
    let mut plain = opts(DdsCompression::None, true);
    plain.dds.mip_gamma_correct = true;
    let plain = levels(&plain)[1].clone();
    assert!(plain.data()[0].abs_diff(128) <= 4, "no sRGB, no EOTF: {}", plain.data()[0]);
}

#[test]
fn premultiplied_filtering_keeps_transparent_rgb_out_of_the_mips() {
    // One transparent texel with a white RGB alongside opaque black: straight filtering averages
    // in the hidden white (~64 each), premultiplied filters it out.
    let mut data = vec![0u8; 4 * 4 * 4];
    data[0] = 255;
    data[1] = 255;
    data[2] = 255;
    // alpha stays 0 at index 3.
    let img = Image::from_u8(4, 4, ChannelLayout::Rgba, data).unwrap();
    let mut raw = opts(DdsCompression::None, true);
    raw.dds.mip_premult_alpha = false;
    let raw = level(&encode(&img, Format::Dds, &raw).unwrap(), 1);
    assert!(raw.data()[0] >= 32, "straight filtering bleeds the white in: {}", raw.data()[0]);
    let b = encode(&img, Format::Dds, &opts(DdsCompression::None, true)).unwrap();
    let pre = level(&b, 1);
    assert_eq!(pre.data()[0], 0, "the hidden white is filtered out");
    assert!(pre.data()[3] < 255, "the alpha still carries it: {}", pre.data()[3]);
}

#[test]
fn premultiplied_export_stores_rgb_times_alpha() {
    let data = vec![200, 100, 50, 128];
    let img = Image::from_u8(1, 1, ChannelLayout::Rgba, data).unwrap();
    let mut o = opts(DdsCompression::None, false);
    o.dds.export_premult_alpha = true;
    let b = encode(&img, Format::Dds, &o).unwrap();
    let back = decode(&b).unwrap();
    // RGB premultiplied by ×128/255, alpha unchanged.
    let px = back.data();
    assert!((99..=101).contains(&px[0]), "premultiplied red: {}", px[0]);
    assert!((49..=51).contains(&px[1]), "premultiplied green: {}", px[1]);
    assert!((24..=26).contains(&px[2]), "premultiplied blue: {}", px[2]);
    assert_eq!(px[3], 128);
}

#[test]
fn every_bc7_effort_decodes() {
    for quality in photocraft_codecs::DdsQuality::ALL {
        let img = gradient(8, 8);
        let b = encode(&img, Format::Dds, &with(DdsOptions { compression: DdsCompression::Bc7, quality, ..Default::default() })).unwrap();
        let back = decode(&b).unwrap();
        assert!(psnr(&img, &back) >= 28.0, "{quality:?}: PSNR {:.1} dB", psnr(&img, &back));
    }
}

/// Six 8×8 faces of a solid colour each, indexed `+X -X +Y -Y +Z -Z`.
fn six_faces() -> Vec<Image> {
    let colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 255], [0, 0, 0], [255, 255, 0]];
    colors
        .iter()
        .map(|&[r, g, b]| {
            let mut data = Vec::new();
            for _ in 0..64 {
                data.extend([r, g, b, 255]);
            }
            Image::from_u8(8, 8, ChannelLayout::Rgba, data).unwrap()
        })
        .collect()
}

fn faces_close(a: &Image, b: &Image, tol: u8) {
    assert_eq!(a.dimensions(), b.dimensions());
    for (x, y) in a.data().chunks(4).zip(b.data().chunks(4)) {
        for c in 0..3 {
            assert!(x[c].abs_diff(y[c]) <= tol, "channel {c}: {} vs {}", x[c], y[c]);
        }
    }
}

#[test]
fn a_cubemap_writes_six_faces_with_legacy_caps() {
    let faces = six_faces();
    let b = photocraft_codecs::encode_dds_cube(&faces, &opts(DdsCompression::Bc1, false)).unwrap();
    // caps: TEXTURE|COMPLEX, caps2: CUBEMAP|all six faces.
    assert_eq!(le32(&b, 112), 0xfe00, "caps2");
    assert!(le32(&b, 108) & DDSCAPS_COMPLEX != 0);
    let back = photocraft_codecs::decode_dds_faces(&b, &photocraft_codecs::DecodeOptions::default()).unwrap();
    assert_eq!(back.len(), 6);
    assert_eq!(back.iter().map(|(i, _)| *i).collect::<Vec<_>>(), [0, 1, 2, 3, 4, 5]);
    for ((_, chain), want) in back.iter().zip(&faces) {
        assert_eq!(chain.len(), 1);
        faces_close(&chain[0], want, 16);
    }
    // And it stays a valid plain decode: +X is the top image.
    assert_eq!(decode(&b).unwrap().dimensions(), (8, 8));
}

#[test]
fn a_dx10_cubemap_marks_misc_cube_and_array_six() {
    let mut o = opts(DdsCompression::Bc3, false);
    o.dds.srgb = true;
    let b = photocraft_codecs::encode_dds_cube(&six_faces(), &o).unwrap();
    assert_eq!(&b[84..88], b"DX10");
    assert_eq!(le32(&b, 136) & 0x4, 0x4, "MISC_TEXTURECUBE");
    assert_eq!(le32(&b, 140), 6, "array size counts faces");
}

#[test]
fn a_cubemap_with_mipmaps_stores_a_chain_per_face() {
    let b = photocraft_codecs::encode_dds_cube(&six_faces(), &opts(DdsCompression::Bc1, true)).unwrap();
    let back = photocraft_codecs::decode_dds_faces(&b, &photocraft_codecs::DecodeOptions::default()).unwrap();
    assert_eq!(back.len(), 6);
    for (_, chain) in &back {
        assert_eq!(chain.len(), 4, "8×8 → 8,4,2,1");
        assert_eq!(chain[3].dimensions(), (1, 1));
    }
}

#[test]
fn a_cubemap_requires_six_square_equal_faces() {
    let faces = six_faces();
    assert!(photocraft_codecs::encode_dds_cube(&faces[..5], &opts(DdsCompression::Bc1, false)).is_err());
    let mut non_square = six_faces();
    non_square[3] = Image::from_u8(8, 4, ChannelLayout::Rgba, vec![0; 8 * 4 * 4]).unwrap();
    assert!(photocraft_codecs::encode_dds_cube(&non_square, &opts(DdsCompression::Bc1, false)).is_err());
    let mut mixed = six_faces();
    mixed[1] = Image::from_u8(4, 4, ChannelLayout::Rgba, vec![0; 4 * 4 * 4]).unwrap();
    assert!(photocraft_codecs::encode_dds_cube(&mixed, &opts(DdsCompression::Bc1, false)).is_err());
}

#[test]
fn the_single_image_encoder_refuses_cube_options() {
    let mut o = opts(DdsCompression::Bc1, false);
    o.dds.cube = photocraft_codecs::DdsCubeMap::Layers;
    assert!(encode(&gradient(8, 8), Format::Dds, &o).is_err());
}
