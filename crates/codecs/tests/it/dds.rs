use photocraft_codecs::{
    ChannelLayout, CodecError, DecodeOptions, Format, Limits, SampleType, caps, decode, decode_as, decode_as_with, detect, encode, from_extension,
};

const DDPF_ALPHAPIXELS: u32 = 0x1;
const DDPF_FOURCC: u32 = 0x4;
const DDPF_RGB: u32 = 0x40;
const DDPF_LUMINANCE: u32 = 0x2_0000;
const DDSD_PITCH: u32 = 0x8;

/// A DDS_HEADER for a single-level 2D texture, after the `DDS ` signature.
fn header(width: u32, height: u32, pf_flags: u32, fourcc: &[u8; 4], bits: u32, masks: [u32; 4]) -> Vec<u8> {
    let mut b = vec![0u8; 128];
    b[0..4].copy_from_slice(b"DDS ");
    let mut put = |at: usize, v: u32| b[at..at + 4].copy_from_slice(&v.to_le_bytes());
    put(4, 124);
    put(8, 0x1007); // CAPS | HEIGHT | WIDTH | PIXELFORMAT
    put(12, height);
    put(16, width);
    put(76, 32);
    put(80, pf_flags);
    put(88, bits);
    for (i, m) in masks.into_iter().enumerate() {
        put(92 + 4 * i, m);
    }
    put(108, 0x1000); // DDSCAPS_TEXTURE
    b[84..88].copy_from_slice(fourcc);
    b
}

fn compressed(width: u32, height: u32, fourcc: &[u8; 4]) -> Vec<u8> {
    header(width, height, DDPF_FOURCC, fourcc, 0, [0; 4])
}

fn set_pitch(b: &mut [u8], pitch: u32) {
    b[8..12].copy_from_slice(&(0x1007 | DDSD_PITCH).to_le_bytes());
    b[20..24].copy_from_slice(&pitch.to_le_bytes());
}

/// The DX10 extension header after a `DX10` FourCC: format, dimension, misc flags, array size.
fn dx10(width: u32, height: u32, dxgi: u32, dimension: u32, misc: u32, array: u32) -> Vec<u8> {
    let mut b = compressed(width, height, b"DX10");
    for v in [dxgi, dimension, misc, array, 0] {
        b.extend(v.to_le_bytes());
    }
    b
}

/// BC1 colour endpoints as RGB565, then one 2-bit index per pixel, row by row.
fn bc1_block(c0: u16, c1: u16, rows: [u8; 4]) -> Vec<u8> {
    let mut b = Vec::new();
    b.extend(c0.to_le_bytes());
    b.extend(c1.to_le_bytes());
    b.extend(rows);
    b
}

const RED: u16 = 0xf800;
const GREEN: u16 = 0x07e0;
const BLUE: u16 = 0x001f;
const BLACK: u16 = 0x0000;
const WHITE: u16 = 0xffff;

/// Indices 0, 1 on the top row and 2, 3 below them, for the top-left 2x2 pixels.
const INDICES_0123: [u8; 4] = [0x04, 0x0e, 0, 0];

#[test]
fn dds_is_readable_and_writable() {
    assert_eq!(from_extension("texture.DDS"), Some(Format::Dds));
    assert!(caps(Format::Dds).read && caps(Format::Dds).write);
    let img = photocraft_codecs::Image::from_u8(1, 1, ChannelLayout::Rgba, vec![1, 2, 3, 0]).unwrap();
    let b = encode(&img, Format::Dds, &Default::default()).unwrap();
    assert_eq!(decode(&b).unwrap().data(), &[1, 2, 3, 0]);
}

#[test]
fn decodes_bc1_four_colour_blocks() {
    // c0 > c1: four opaque colours, the two in between at thirds.
    let mut b = compressed(2, 2, b"DXT1");
    b.extend(bc1_block(RED, BLUE, INDICES_0123));
    assert_eq!(detect(&b), Some(Format::Dds));
    let im = decode(&b).unwrap();
    assert_eq!((im.dimensions(), im.layout(), im.sample_type()), ((2, 2), ChannelLayout::Rgba, SampleType::U8));
    assert_eq!(im.data(), &[255, 0, 0, 255, 0, 0, 255, 255, 170, 0, 85, 255, 85, 0, 170, 255]);
}

#[test]
fn decodes_bc1_one_bit_alpha_blocks() {
    // c0 <= c1: three colours (the third halfway) and index 3 is transparent black.
    let mut b = compressed(2, 2, b"DXT1");
    b.extend(bc1_block(BLUE, RED, INDICES_0123));
    assert_eq!(decode(&b).unwrap().data(), &[0, 0, 255, 255, 255, 0, 0, 255, 128, 0, 128, 255, 0, 0, 0, 0]);
}

#[test]
fn decodes_bc2_explicit_alpha_with_four_colour_blocks() {
    let mut b = compressed(2, 2, b"DXT3");
    // Four bits of alpha per pixel: 15, 0 on the top row, 8, 1 below.
    b.extend([0x0f, 0x00, 0x18, 0x00, 0, 0, 0, 0]);
    // c0 < c1 would mean transparency in BC1; BC2 colour blocks always have four colours.
    b.extend(bc1_block(BLACK, WHITE, [0x04, 0x0b, 0, 0]));
    let im = decode(&b).unwrap();
    assert_eq!(im.layout(), ChannelLayout::Rgba);
    assert_eq!(im.data(), &[0, 0, 0, 255, 255, 255, 255, 0, 170, 170, 170, 136, 85, 85, 85, 17]);
}

#[test]
fn decodes_bc3_interpolated_alpha() {
    // a0 > a1: eight alpha values, six interpolated at sevenths. Indices 0, 1 / 2, 7.
    let mut b = compressed(2, 2, b"DXT5");
    b.extend([255, 0, 0x08, 0xa0, 0x03, 0, 0, 0]);
    b.extend(bc1_block(BLACK, WHITE, [0x04, 0x0b, 0, 0]));
    assert_eq!(decode(&b).unwrap().data(), &[0, 0, 0, 255, 255, 255, 255, 0, 170, 170, 170, 219, 85, 85, 85, 36]);

    // a0 <= a1: six values, four interpolated at fifths, then 0 and 255. Indices 2, 5 / 6, 7.
    let mut b = compressed(2, 2, b"DXT5");
    let bits: u64 = 2 | 5 << 3 | 6 << 12 | 7 << 15;
    b.extend([0, 255]);
    b.extend(&bits.to_le_bytes()[..6]);
    b.extend(bc1_block(GREEN, GREEN, [0; 4]));
    assert_eq!(decode(&b).unwrap().data(), &[0, 255, 0, 51, 0, 255, 0, 204, 0, 255, 0, 0, 0, 255, 0, 255]);
}

#[test]
fn bc_textures_need_not_be_a_multiple_of_four() {
    // 5x3 is two blocks wide and one high; the right block's first column is x = 4.
    let mut b = compressed(5, 3, b"DXT1");
    b.extend(bc1_block(RED, RED, [0; 4]));
    b.extend(bc1_block(BLUE, BLUE, [0; 4]));
    let im = decode(&b).unwrap();
    assert_eq!(im.dimensions(), (5, 3));
    let px = |x: usize, y: usize| &im.data()[(y * 5 + x) * 4..(y * 5 + x + 1) * 4];
    assert_eq!((px(3, 2), px(4, 0), px(4, 2)), (&[255, 0, 0, 255][..], &[0, 0, 255, 255][..], &[0, 0, 255, 255][..]));
}

#[test]
fn decodes_uncompressed_rgba_and_bgra_masks() {
    let rgba = [0xff, 0xff00, 0xff_0000, 0xff00_0000];
    let mut b = header(2, 1, DDPF_RGB | DDPF_ALPHAPIXELS, &[0; 4], 32, rgba);
    b.extend([1, 2, 3, 4, 250, 251, 252, 253]);
    let im = decode(&b).unwrap();
    assert_eq!(im.layout(), ChannelLayout::Rgba);
    assert_eq!(im.data(), &[1, 2, 3, 4, 250, 251, 252, 253]);

    let bgra = [0xff_0000, 0xff00, 0xff, 0xff00_0000];
    let mut b = header(2, 1, DDPF_RGB | DDPF_ALPHAPIXELS, &[0; 4], 32, bgra);
    b.extend([3, 2, 1, 4, 252, 251, 250, 253]);
    assert_eq!(decode(&b).unwrap().data(), &[1, 2, 3, 4, 250, 251, 252, 253]);

    // An alpha mask without DDPF_ALPHAPIXELS is not alpha: X8R8G8B8 opens as RGB.
    let mut b = header(1, 1, DDPF_RGB, &[0; 4], 32, bgra);
    b.extend([3, 2, 1, 0]);
    let im = decode(&b).unwrap();
    assert_eq!((im.layout(), im.data()), (ChannelLayout::Rgb, &[1u8, 2, 3][..]));
}

#[test]
fn decodes_packed_masks_and_luminance() {
    // R5G6B5: each field scaled to 0..255; 3 of 31 is 25.
    let mut b = header(4, 1, DDPF_RGB, &[0; 4], 16, [0xf800, 0x07e0, 0x001f, 0]);
    for v in [RED, GREEN, BLUE, 3 << 11] {
        b.extend(v.to_le_bytes());
    }
    let im = decode(&b).unwrap();
    assert_eq!(im.layout(), ChannelLayout::Rgb);
    assert_eq!(im.data(), &[255, 0, 0, 0, 255, 0, 0, 0, 255, 25, 0, 0]);

    // 24-bit rows are packed three bytes a pixel, without padding.
    let mut b = header(2, 2, DDPF_RGB, &[0; 4], 24, [0xff_0000, 0xff00, 0xff, 0]);
    b.extend([3, 2, 1, 6, 5, 4, 9, 8, 7, 12, 11, 10]);
    assert_eq!(decode(&b).unwrap().data(), &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);

    let mut b = header(2, 1, DDPF_LUMINANCE | DDPF_ALPHAPIXELS, &[0; 4], 16, [0xff, 0, 0, 0xff00]);
    b.extend([10, 20, 30, 40]);
    let im = decode(&b).unwrap();
    assert_eq!((im.layout(), im.data()), (ChannelLayout::GrayA, &[10u8, 20, 30, 40][..]));
}

#[test]
fn decodes_padded_rgb_rows() {
    // DWORD-aligned 24-bit rows: the fourth byte is padding, not the next pixel.
    let mut b = header(1, 2, DDPF_RGB, &[0; 4], 24, [0xff, 0xff00, 0xff_0000, 0]);
    set_pitch(&mut b, 4);
    b.extend([255, 0, 0, 99, 0, 255, 0, 88]);
    let im = decode(&b).unwrap();
    assert_eq!((im.dimensions(), im.layout()), ((1, 2), ChannelLayout::Rgb));
    assert_eq!(im.data(), &[255, 0, 0, 0, 255, 0]);

    // Padding can contain whole pixels; only the declared width is decoded.
    let mut b = header(2, 2, DDPF_RGB, &[0; 4], 16, [0xf800, 0x07e0, 0x001f, 0]);
    set_pitch(&mut b, 8);
    for v in [RED, GREEN, WHITE, WHITE, BLUE, BLACK, WHITE, WHITE] {
        b.extend(v.to_le_bytes());
    }
    assert_eq!(decode(&b).unwrap().data(), &[255, 0, 0, 0, 255, 0, 0, 0, 255, 0, 0, 0]);
}

#[test]
fn decodes_padded_luminance_rows() {
    let mut b = header(2, 2, DDPF_LUMINANCE, &[0; 4], 8, [0xff, 0, 0, 0]);
    set_pitch(&mut b, 4);
    b.extend([10, 20, 99, 98, 30, 40, 97, 96]);
    let im = decode(&b).unwrap();
    assert_eq!((im.layout(), im.data()), (ChannelLayout::Gray, &[10u8, 20, 30, 40][..]));

    let mut b = header(1, 2, DDPF_LUMINANCE | DDPF_ALPHAPIXELS, &[0; 4], 16, [0xff, 0, 0, 0xff00]);
    set_pitch(&mut b, 4);
    b.extend([10, 20, 99, 98, 30, 40, 97, 96]);
    let im = decode(&b).unwrap();
    assert_eq!((im.layout(), im.data()), (ChannelLayout::GrayA, &[10u8, 20, 30, 40][..]));
}

#[test]
fn decodes_padded_dx10_rows() {
    let mut b = dx10(1, 2, 28, 3, 0, 1); // R8G8B8A8_UNORM
    set_pitch(&mut b, 8);
    b.extend([1, 2, 3, 4, 99, 98, 97, 96, 5, 6, 7, 8, 95, 94, 93, 92]);
    assert_eq!(decode(&b).unwrap().data(), &[1, 2, 3, 4, 5, 6, 7, 8]);
}

#[test]
fn ignores_pitch_without_the_pitch_flag() {
    let mut b = header(1, 2, DDPF_RGB, &[0; 4], 24, [0xff, 0xff00, 0xff_0000, 0]);
    b[20..24].copy_from_slice(&u32::MAX.to_le_bytes());
    b.extend([255, 0, 0, 0, 255, 0]);
    assert_eq!(decode(&b).unwrap().data(), &[255, 0, 0, 0, 255, 0]);

    set_pitch(&mut b, 3); // An explicitly tightly packed row is also valid.
    assert_eq!(decode(&b).unwrap().data(), &[255, 0, 0, 0, 255, 0]);
}

#[test]
fn compressed_linear_size_is_not_a_row_pitch() {
    let mut b = compressed(4, 8, b"DXT1");
    b[8..12].copy_from_slice(&(0x1007u32 | 0x8_0000).to_le_bytes()); // DDSD_LINEARSIZE
    b[20..24].copy_from_slice(&16u32.to_le_bytes()); // Two 8-byte block rows.
    b.extend(bc1_block(RED, RED, [0; 4]));
    b.extend(bc1_block(BLUE, BLUE, [0; 4]));
    let im = decode(&b).unwrap();
    assert!(im.data()[..64].as_chunks::<4>().0.iter().all(|p| p == &[255, 0, 0, 255]));
    assert!(im.data()[64..].as_chunks::<4>().0.iter().all(|p| p == &[0, 0, 255, 255]));
}

#[test]
fn rejects_pitch_smaller_than_a_pixel_row() {
    for pitch in [0, 1, 2] {
        let mut b = header(1, 2, DDPF_RGB, &[0; 4], 24, [0xff, 0xff00, 0xff_0000, 0]);
        set_pitch(&mut b, pitch);
        b.extend([255, 0, 0, 0, 255, 0]);
        assert!(matches!(decode(&b), Err(CodecError::Malformed { .. })), "pitch {pitch}");
    }
}

#[test]
fn rejects_truncated_padded_rows() {
    let mut b = header(1, 2, DDPF_RGB, &[0; 4], 24, [0xff, 0xff00, 0xff_0000, 0]);
    set_pitch(&mut b, 4);
    b.extend([255, 0, 0, 99, 0, 255, 0, 88]);
    assert!(decode(&b).is_ok());
    for n in 128..b.len() {
        assert!(matches!(decode(&b[..n]), Err(CodecError::Malformed { .. })), "decoded {n} of {} bytes", b.len());
    }
}

#[test]
fn rejects_huge_pitch_without_allocating() {
    let mut b = header(1, 2, DDPF_LUMINANCE, &[0; 4], 8, [0xff, 0, 0, 0]);
    set_pitch(&mut b, u32::MAX);
    b.extend([10, 20]);
    // On 32-bit targets pitch * height overflows; on 64-bit the payload is too short.
    assert!(matches!(decode(&b), Err(CodecError::Malformed { .. })));
}

#[test]
fn decodes_dx10_headers() {
    let mut b = dx10(1, 1, 28, 3, 0, 1); // R8G8B8A8_UNORM
    b.extend([1, 2, 3, 4]);
    assert_eq!(decode(&b).unwrap().data(), &[1, 2, 3, 4]);

    let mut b = dx10(1, 1, 91, 3, 0, 1); // B8G8R8A8_UNORM_SRGB
    b.extend([3, 2, 1, 4]);
    assert_eq!(decode(&b).unwrap().data(), &[1, 2, 3, 4]);

    let mut b = dx10(2, 2, 71, 3, 0, 1); // BC1_UNORM
    b.extend(bc1_block(BLUE, RED, INDICES_0123));
    assert_eq!(decode(&b).unwrap().data(), &[0, 0, 255, 255, 255, 0, 0, 255, 128, 0, 128, 255, 0, 0, 0, 0]);
}

#[test]
fn only_the_top_mip_level_is_read() {
    let mut b = compressed(4, 4, b"DXT1");
    b[28..32].copy_from_slice(&3u32.to_le_bytes());
    b.extend(bc1_block(RED, RED, [0; 4]));
    b.extend(bc1_block(BLUE, BLUE, [0; 4]));
    b.extend(bc1_block(GREEN, GREEN, [0; 4]));
    let im = decode(&b).unwrap();
    assert_eq!(im.dimensions(), (4, 4));
    assert!(im.data().chunks(4).all(|p| p == [255, 0, 0, 255]));
}

/// A mip count the header declares is hostile: the chain ends at 1×1, so a file claiming
/// billions of levels (and padded to tempt the decode) reads the levels that reduce to it.
#[test]
fn a_declared_mip_count_is_capped_at_the_chain() {
    let mut b = compressed(4, 4, b"DXT1");
    b[8..12].copy_from_slice(&(0x1007u32 | 0x2_0000).to_le_bytes()); // DDSD_MIPMAPCOUNT
    b[28..32].copy_from_slice(&u32::MAX.to_le_bytes());
    b.extend(bc1_block(RED, RED, [0; 4]));
    b.extend(bc1_block(BLUE, BLUE, [0; 4]));
    b.extend(bc1_block(GREEN, GREEN, [0; 4]));
    b.extend([0x55; 8 * 100]); // trailing blocks a naive reader would decode as more levels
    let levels = photocraft_codecs::decode_dds_levels(&b, &DecodeOptions::default()).unwrap();
    assert_eq!(levels.len(), 3, "4×4 → 4×4, 2×2, 1×1");
    assert_eq!(levels[2].dimensions(), (1, 1));
}

#[test]
fn refuses_arrays_and_volumes() {
    let unsupported = |b: &[u8], what: &str| match decode_as(Format::Dds, b) {
        Err(CodecError::Unsupported { reason, .. }) => assert!(reason.contains(what), "{reason}"),
        other => panic!("{what}: {other:?}"),
    };
    let mut volume = compressed(4, 4, b"DXT1");
    volume[8..12].copy_from_slice(&(0x1007u32 | 0x80_0000).to_le_bytes()); // DDSD_DEPTH
    volume[24..28].copy_from_slice(&4u32.to_le_bytes());
    volume[112..116].copy_from_slice(&0x20_0000u32.to_le_bytes()); // DDSCAPS2_VOLUME
    volume.extend([0; 8 * 4]);
    unsupported(&volume, "volume");
    unsupported(&dx10(4, 4, 71, 3, 0x4, 12), "arrays of cubemaps"); // two cubes, six faces each
    unsupported(&dx10(4, 4, 71, 3, 0, 6), "array");
    unsupported(&dx10(4, 4, 71, 4, 0, 1), "2D");
    unsupported(&dx10(4, 4, 95, 3, 0, 1), "95"); // BC6H_UF16
    unsupported(&dx10(4, 4, 27, 3, 0, 1), "27"); // R8G8B8A8_SNORM
    match decode_as(Format::Dds, &dx10(4, 4, 71, 3, 0x4, 4)) {
        Err(CodecError::Malformed { message, .. }) => assert!(message.contains("multiple of 6"), "{message}"),
        other => panic!("a cubemap's array size counts its faces: {other:?}"),
    }
}

#[test]
fn rejects_truncated_files_at_every_length_without_panicking() {
    let mut b = dx10(5, 3, 77, 3, 0, 1); // BC3_UNORM: two 16-byte blocks
    b.extend([0x5a; 32]);
    assert!(decode(&b).is_ok());
    for n in 0..b.len() {
        assert!(decode_as(Format::Dds, &b[..n]).is_err(), "decoded {n} of {} bytes", b.len());
    }
    let mut b = header(3, 2, DDPF_RGB, &[0; 4], 24, [0xff_0000, 0xff00, 0xff, 0]);
    b.extend([7; 18]);
    assert!(decode(&b).is_ok());
    assert!(decode_as(Format::Dds, &b[..b.len() - 1]).is_err(), "one byte short");
}

#[test]
fn rejects_bogus_sizes_and_pixel_formats() {
    // Huge dimensions are refused before anything is allocated.
    let mut huge = compressed(u32::MAX, u32::MAX, b"DXT1");
    huge.extend([0; 8]);
    assert!(matches!(decode_as(Format::Dds, &huge), Err(CodecError::LimitExceeded(_))));
    let opts = DecodeOptions { limits: Limits { max_alloc: 1024, ..Default::default() }, ..Default::default() };
    let mut tight = header(64, 64, DDPF_RGB, &[0; 4], 32, [0xff, 0xff00, 0xff_0000, 0]);
    tight.extend(vec![0; 64 * 64 * 4]);
    assert!(matches!(decode_as_with(Format::Dds, &tight, &opts), Err(CodecError::LimitExceeded(_))));
    assert!(decode_as(Format::Dds, &compressed(0, 4, b"DXT1")).is_err(), "zero width");

    let mut bad_size = compressed(4, 4, b"DXT1");
    bad_size[4..8].copy_from_slice(&100u32.to_le_bytes());
    bad_size.extend([0; 8]);
    assert!(matches!(decode_as(Format::Dds, &bad_size), Err(CodecError::Malformed { .. })), "header size must be 124");

    for (bits, masks) in [
        (12, [0xf00, 0xf0, 0xf, 0]),        // not a whole number of bytes
        (16, [0xff_0000, 0xff00, 0xff, 0]), // a mask wider than the pixel
        (32, [0xf0f, 0xf0f0, 0xf_0000, 0]), // a mask with a gap
        (32, [0, 0, 0, 0]),                 // no colour at all
        (0, [0xff, 0xff00, 0xff_0000, 0]),  // zero bits per pixel
    ] {
        let mut b = header(1, 1, DDPF_RGB, &[0; 4], bits, masks);
        b.extend([0; 4]);
        assert!(decode_as(Format::Dds, &b).is_err(), "{bits} bits, masks {masks:x?}");
    }
    let mut no_format = header(1, 1, 0, &[0; 4], 32, [0xff, 0xff00, 0xff_0000, 0]);
    no_format.extend([0; 4]);
    assert!(matches!(decode_as(Format::Dds, &no_format), Err(CodecError::Unsupported { .. })));
}

/// A BC4 block, or one half of a BC5 one: two endpoints and a 3-bit index per texel —
/// the same interpolated block BC3 uses for alpha.
fn alpha_block(a0: u8, a1: u8, bits: u64) -> Vec<u8> {
    let mut b = vec![a0, a1];
    b.extend(&bits.to_le_bytes()[..6]);
    b
}

// Texel indices 0, 1 on the top row and 2, 7 below, for the 2x2 corner pixels.
const ALPHA_INDICES: u64 = 1 << 3 | 2 << 12 | 7 << 15;

#[test]
fn decodes_bc4_as_grayscale() {
    // a0 > a1: eight values, six at sevenths: palette 255, 0, 219, …, 36.
    for (fourcc, name) in [(b"ATI1", "ATI1"), (b"BC4U", "BC4U")] {
        let mut b = compressed(2, 2, fourcc);
        b.extend(alpha_block(255, 0, ALPHA_INDICES));
        let im = decode(&b).unwrap();
        assert_eq!((im.layout(), im.sample_type()), (ChannelLayout::Gray, SampleType::U8), "{name}");
        assert_eq!(im.data(), &[255, 0, 219, 36], "{name}");
    }
    // DXGI BC4_UNORM through the DX10 header.
    let mut b = dx10(2, 2, 80, 3, 0, 1);
    b.extend(alpha_block(255, 0, ALPHA_INDICES));
    assert_eq!(decode(&b).unwrap().data(), &[255, 0, 219, 36]);
}

#[test]
fn decodes_bc4_snorm_biased_to_unsigned() {
    // Signed endpoints 127, -128: the eight-value mode interpolates between them and the
    // output is biased by 128. Indices 0 and 1 give the endpoints exactly: 255 and 0.
    let mut b = compressed(2, 2, b"BC4S");
    b.extend(alpha_block(127, 128, ALPHA_INDICES));
    assert_eq!(decode(&b).unwrap().data(), &[255, 0, 219, 36]);
    // DXGI BC4_SNORM.
    let mut b = dx10(2, 2, 81, 3, 0, 1);
    b.extend(alpha_block(127, 128, ALPHA_INDICES));
    assert_eq!(decode(&b).unwrap().data(), &[255, 0, 219, 36]);
}

#[test]
fn decodes_bc5_as_two_channels_with_black_blue() {
    // Red: 255, 0 endpoints (eight values). Green: 0, 255 (six values, plus 0 and 255).
    for (fourcc, name) in [(b"ATI2", "ATI2"), (b"BC5U", "BC5U")] {
        let mut b = compressed(2, 2, fourcc);
        b.extend(alpha_block(255, 0, ALPHA_INDICES));
        b.extend(alpha_block(0, 255, ALPHA_INDICES));
        let im = decode(&b).unwrap();
        assert_eq!(im.layout(), ChannelLayout::Rgb, "{name}");
        assert_eq!(im.data(), &[255, 0, 0, 0, 255, 0, 219, 51, 0, 36, 255, 0], "{name}");
    }
    // DXGI BC5_UNORM, and BC5_SNORM with signed endpoints.
    let mut b = dx10(2, 2, 83, 3, 0, 1);
    b.extend(alpha_block(255, 0, ALPHA_INDICES));
    b.extend(alpha_block(0, 255, ALPHA_INDICES));
    assert_eq!(decode(&b).unwrap().data(), &[255, 0, 0, 0, 255, 0, 219, 51, 0, 36, 255, 0]);
    let mut b = dx10(2, 2, 84, 3, 0, 1);
    b.extend(alpha_block(127, 128, ALPHA_INDICES));
    b.extend(alpha_block(128, 127, ALPHA_INDICES));
    assert_eq!(decode(&b).unwrap().data(), &[255, 0, 0, 0, 255, 0, 219, 51, 0, 36, 255, 0]);
}

#[test]
fn decodes_bc7_blocks_through_the_dx10_header() {
    // A golden block produced by the reference encoder: the decoder's exact output.
    let mut b = dx10(4, 4, 98, 3, 0, 1);
    b.extend([0x20, 0x46, 0xee, 0x1a, 0x60, 0xbb, 0xae, 0x47, 0xf4, 0xd3, 0x8c, 0x81, 0x8e, 0x4d, 0xff, 0x4f]);
    let im = decode(&b).unwrap();
    assert_eq!((im.layout(), im.dimensions()), (ChannelLayout::Rgba, (4, 4)));
    let want: [u8; 64] = [
        0x9b, 0x90, 0x82, 0xa3, 0xab, 0x47, 0x99, 0x11, 0xb9, 0x00, 0xaf, 0xeb, 0xb9, 0x00, 0xaf, 0x59, 0x9b, 0x90, 0x82, 0xa3, 0xab, 0x47, 0x99, 0x11, 0xab,
        0x47, 0x99, 0xeb, 0x9b, 0x90, 0x82, 0xa3, 0xab, 0x47, 0x99, 0x11, 0x9b, 0x90, 0x82, 0x11, 0x8d, 0xd7, 0x6c, 0x11, 0xb9, 0x00, 0xaf, 0x11, 0x8d, 0xd7,
        0x6c, 0x11, 0x8d, 0xd7, 0x6c, 0x11, 0x8d, 0xd7, 0x6c, 0xeb, 0x9b, 0x90, 0x82, 0xa3,
    ];
    assert_eq!(im.data(), &want);
}

#[test]
fn decodes_premultiplied_dxt2_and_dxt4_as_straight_alpha() {
    // A solid rgb565 red of 123 (0x7800) stored premultiplied: colour * 255 / alpha.
    let mut b = compressed(2, 2, b"DXT2");
    b.extend([0x88; 8]); // 4-bit alpha of 8: 136 everywhere
    b.extend(bc1_block(0x7800, 0x7800, [0; 4]));
    assert_eq!(decode(&b).unwrap().data(), &[230, 0, 0, 136, 230, 0, 0, 136, 230, 0, 0, 136, 230, 0, 0, 136]);

    let mut b = compressed(2, 2, b"DXT4");
    b.extend(alpha_block(128, 128, 0)); // interpolated alpha of 128 everywhere
    b.extend(bc1_block(0x7800, 0x7800, [0; 4]));
    assert_eq!(decode(&b).unwrap().data(), &[245, 0, 0, 128, 245, 0, 0, 128, 245, 0, 0, 128, 245, 0, 0, 128]);
}

#[test]
fn decodes_rxgb_as_plain_bc3() {
    // RXGB is GIMP's name for an unswizzled DXT5 file: same bytes, same decode.
    let mut b = compressed(2, 2, b"RXGB");
    b.extend([255, 0, 0x08, 0xa0, 0x03, 0, 0, 0]);
    b.extend(bc1_block(BLACK, WHITE, [0x04, 0x0b, 0, 0]));
    assert_eq!(decode(&b).unwrap().data(), &[0, 0, 0, 255, 255, 255, 255, 0, 170, 170, 170, 219, 85, 85, 85, 36]);
}

/// A `DXT5` file as GIMP's DDS plug-in marks it: the real compression name sits in the
/// reserved header fields after its `GIMP-DDS` marker.
fn gimp_variant(variant: &[u8; 4], w: u32, h: u32) -> Vec<u8> {
    let mut b = compressed(w, h, b"DXT5");
    b[32..40].copy_from_slice(b"GIMP-DDS");
    b[44..48].copy_from_slice(variant);
    b
}

#[test]
fn decodes_gimps_aexp_ycg1_and_ycg2_variants() {
    // AEXP: colour scaled by the interpolated channel, opaque out. Red of 206 (0xc800)
    // times 128/256 is 103.
    let mut b = gimp_variant(b"AEXP", 2, 2);
    b.extend(alpha_block(128, 128, 0));
    b.extend(bc1_block(0xc800, 0xc800, [0; 4]));
    let im = decode(&b).unwrap();
    assert_eq!(im.layout(), ChannelLayout::Rgb);
    assert_eq!(im.data(), &[103, 0, 0, 103, 0, 0, 103, 0, 0, 103, 0, 0]);

    // YCG1: near-neutral chroma and full luma give near-white, and the blue channel is
    // the alpha. 0x843f is r 132, g 134, b 255: co ≈ 0.016, cg ≈ 0.024, so the decode is
    // r ≈ 253, g = 255, b ≈ 245, alpha 255.
    let mut b = gimp_variant(b"YCG1", 2, 2);
    b.extend(alpha_block(255, 255, 0)); // luma 1.0
    b.extend(bc1_block(0x843f, 0x843f, [0; 4]));
    let im = decode(&b).unwrap();
    assert_eq!(im.layout(), ChannelLayout::Rgba);
    for px in im.data().chunks(4) {
        assert!(px[0].abs_diff(253) <= 3 && px[1].abs_diff(255) <= 3 && px[2].abs_diff(245) <= 3 && px[3] == 255, "{px:?}");
    }

    // YCG2: the chroma is divided by the blue channel's scale — full blue flattens it,
    // so the same block decodes to plain opaque white.
    let mut b = gimp_variant(b"YCG2", 2, 2);
    b.extend(alpha_block(255, 255, 0));
    b.extend(bc1_block(0x843f, 0x843f, [0; 4]));
    let im = decode(&b).unwrap();
    assert_eq!(im.layout(), ChannelLayout::Rgb);
    for px in im.data().chunks(3) {
        assert!(px[0].abs_diff(255) <= 2 && px[1].abs_diff(255) <= 2 && px[2].abs_diff(255) <= 2, "{px:?}");
    }
}

#[test]
fn decodes_float_pixel_formats_keeping_their_depth() {
    // Legacy D3DFMT numbers in the FourCC field: 113 is A16B16G16R16F, 116 A32B32G32R32F.
    let mut b = compressed(1, 1, &113u32.to_le_bytes());
    for v in [0x3c00u16, 0x3800, 0x3400, 0x3c00] {
        b.extend(v.to_le_bytes());
    }
    let im = decode(&b).unwrap();
    assert_eq!((im.layout(), im.sample_type()), (ChannelLayout::Rgba, SampleType::F16));
    assert_eq!(im.to_f16_samples().unwrap(), [1.0, 0.5, 0.25, 1.0].map(photocraft_codecs::f16::from_f32));

    let mut b = compressed(1, 1, &116u32.to_le_bytes());
    for v in [1.0f32, 0.5, 0.25, 1.0] {
        b.extend(v.to_le_bytes());
    }
    let im = decode(&b).unwrap();
    assert_eq!(im.sample_type(), SampleType::F32);
    assert_eq!(im.to_f32_samples().unwrap(), [1.0, 0.5, 0.25, 1.0]);

    // The same through DXGI 10 and 2.
    let mut b = dx10(1, 1, 10, 3, 0, 1);
    for v in [0x3c00u16, 0x3800, 0x3400, 0x3c00] {
        b.extend(v.to_le_bytes());
    }
    assert_eq!(decode(&b).unwrap().to_f16_samples().unwrap(), [1.0, 0.5, 0.25, 1.0].map(photocraft_codecs::f16::from_f32));
    let mut b = dx10(1, 1, 2, 3, 0, 1);
    for v in [1.0f32, 0.5, 0.25, 1.0] {
        b.extend(v.to_le_bytes());
    }
    assert_eq!(decode(&b).unwrap().to_f32_samples().unwrap(), [1.0, 0.5, 0.25, 1.0]);
}

#[test]
fn decodes_dx10_uncompressed_variants() {
    // R8G8_UNORM.
    let mut b = dx10(2, 1, 49, 3, 0, 1);
    b.extend([10, 20, 30, 40]);
    assert_eq!(decode(&b).unwrap().data(), &[10, 20, 0, 30, 40, 0]);
    // R8_UNORM reads as grayscale.
    let mut b = dx10(2, 1, 61, 3, 0, 1);
    b.extend([10, 40]);
    let im = decode(&b).unwrap();
    assert_eq!((im.layout(), im.data()), (ChannelLayout::Gray, &[10u8, 40][..]));
    // R10G10B10A2_UNORM: top two bits are alpha, three levels (0, 85, 170, 255).
    let mut b = dx10(1, 1, 24, 3, 0, 1);
    b.extend((0xc000_0000u32 | 0x3ff << 20 | 0x3ff << 10 | 0x3ff).to_le_bytes());
    assert_eq!(decode(&b).unwrap().data(), &[255, 255, 255, 255]);
}

/// Six solid-colour BC1 faces, appended in `+X -X +Y -Y +Z -Z` order: one 8-byte block each.
fn six_faces() -> Vec<u8> {
    let mut data = Vec::new();
    for c in [RED, GREEN, BLUE, WHITE, BLACK, RED] {
        data.extend(bc1_block(c, c, [0; 4]));
    }
    data
}

/// The first pixel's colour of `face`, as RGB.
fn face_pixel(face: &photocraft_codecs::Image) -> [u8; 3] {
    let p = &face.data()[..4];
    [p[0], p[1], p[2]]
}

#[test]
fn decodes_a_legacy_cubemaps_faces_in_storage_order() {
    let mut b = compressed(4, 4, b"DXT1");
    b[112..116].copy_from_slice(&0xfe00u32.to_le_bytes()); // DDSCAPS2_CUBEMAP | all face bits
    b.extend(six_faces());
    let faces = photocraft_codecs::decode_dds_faces(&b, &DecodeOptions::default()).unwrap();
    assert_eq!(faces.len(), 6);
    assert_eq!(faces.iter().map(|(i, _)| *i).collect::<Vec<_>>(), [0, 1, 2, 3, 4, 5]);
    // +X is red, -X green, +Y blue.
    assert_eq!(face_pixel(&faces[0].1[0]), [255, 0, 0]);
    assert_eq!(face_pixel(&faces[1].1[0]), [0, 255, 0]);
    assert_eq!(face_pixel(&faces[2].1[0]), [0, 0, 255]);
    // `decode` reads the +X face's top level, and `dds_info` reports the cube.
    assert_eq!(face_pixel(&decode(&b).unwrap()), [255, 0, 0]);
    let info = photocraft_codecs::dds_info(&b).unwrap();
    assert!((info.cube, info.faces) == (true, 6));
}

#[test]
fn decodes_a_dx10_cubemap() {
    let mut b = dx10(4, 4, 71, 3, 4, 6); // BC1_UNORM, MISC_TEXTURECUBE, one cube
    b.extend(six_faces());
    let faces = photocraft_codecs::decode_dds_faces(&b, &DecodeOptions::default()).unwrap();
    assert_eq!(faces.len(), 6);
    assert_eq!(face_pixel(&faces[4].1[0]), [0, 0, 0], "+Z is the black face");
    let info = photocraft_codecs::dds_info(&b).unwrap();
    assert!((info.cube, info.faces) == (true, 6));
}

#[test]
fn decodes_a_partial_cubemap_by_its_face_bits() {
    let mut b = compressed(4, 4, b"DXT1");
    // caps2: CUBEMAP with only +X (bit 10) and +Y (bit 12).
    b[112..116].copy_from_slice(&(0x200u32 | (0b000101 << 10)).to_le_bytes());
    b.extend(bc1_block(RED, RED, [0; 4]));
    b.extend(bc1_block(BLUE, BLUE, [0; 4]));
    let faces = photocraft_codecs::decode_dds_faces(&b, &DecodeOptions::default()).unwrap();
    assert_eq!(faces.len(), 2);
    assert_eq!(faces.iter().map(|(i, _)| *i).collect::<Vec<_>>(), [0, 2]);
    assert_eq!(face_pixel(&faces[0].1[0]), [255, 0, 0]);
    assert_eq!(face_pixel(&faces[1].1[0]), [0, 0, 255]);
    assert_eq!(photocraft_codecs::dds_info(&b).unwrap().faces, 2);
}

#[test]
fn a_legacy_cubemap_flag_without_face_bits_means_all_six() {
    let mut b = compressed(4, 4, b"DXT1");
    b[112..116].copy_from_slice(&0x200u32.to_le_bytes()); // DDSCAPS2_CUBEMAP alone
    b.extend(six_faces());
    let faces = photocraft_codecs::decode_dds_faces(&b, &DecodeOptions::default()).unwrap();
    assert_eq!(faces.len(), 6);
}

#[test]
fn cubemap_faces_each_keep_their_mip_chain() {
    let mut b = compressed(4, 4, b"DXT1");
    b[8..12].copy_from_slice(&(0x1007u32 | 0x2_0000).to_le_bytes()); // DDSD_MIPMAPCOUNT
    b[28..32].copy_from_slice(&3u32.to_le_bytes());
    b[112..116].copy_from_slice(&0xfe00u32.to_le_bytes());
    for _ in 0..6 {
        b.extend(&six_faces()[..24]); // 3 blocks: 4x4, 2x2, 1x1
    }
    let faces = photocraft_codecs::decode_dds_faces(&b, &DecodeOptions::default()).unwrap();
    assert_eq!(faces.len(), 6);
    for (_, chain) in &faces {
        assert_eq!(chain.len(), 3);
        assert_eq!(chain[2].dimensions(), (1, 1));
    }
}

#[test]
fn a_truncated_cubemap_is_malformed_not_a_panic() {
    let mut b = compressed(4, 4, b"DXT1");
    b[112..116].copy_from_slice(&0xfe00u32.to_le_bytes());
    b.extend(bc1_block(RED, RED, [0; 4])); // one face of six
    assert!(photocraft_codecs::decode_dds_faces(&b, &DecodeOptions::default()).is_err());
}
