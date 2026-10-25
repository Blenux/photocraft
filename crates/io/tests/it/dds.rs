//! DDS import options (#8632): the alpha channel kept as the layer's transparency or routed to
//! an "Alpha 1" document channel (textures that carry data, such as specular maps, in alpha —
//! the NVIDIA/Intel plug-ins' "load transparency as alpha channel"), and the mip chain opened
//! as one layer per level or the top level only. Export writes a document's alpha channel back
//! as the texture's alpha (#8563).

use photocraft_io::*;

/// A 16×8 RGBA DDS with the full mip chain, uncompressed so every level is exact.
fn dds_with_mips() -> Vec<u8> {
    let ch = 4;
    let data: Vec<u8> = (0..16 * 8).flat_map(|i| [i as u8, 255 - i as u8, 40, 200 - i as u8]).collect();
    let img = photocraft_codecs::Image::from_u8(16, 8, photocraft_codecs::ChannelLayout::Rgba, data[..16 * 8 * ch].to_vec()).unwrap();
    photocraft_codecs::encode(
        &img,
        photocraft_codecs::Format::Dds,
        &photocraft_codecs::EncodeOptions {
            dds: photocraft_codecs::DdsOptions { compression: DdsCompression::None, mipmaps: true, ..Default::default() },
            ..Default::default()
        },
    )
    .unwrap()
}

#[test]
fn the_default_import_reads_the_top_level_with_its_alpha() {
    let r = import("t.dds", &dds_with_mips()).unwrap();
    assert_eq!(r.document.size, photocraft_geom::Size::new(16, 8));
    assert_eq!(r.document.layers.len(), 1);
    let l = &r.document.layers[0];
    let px = l.surface().unwrap().read_region(photocraft_geom::Rect::new(0, 0, 16, 8));
    assert!(px.chunks(4).any(|p| p[3] < 1.0), "alpha kept");
    assert_ne!(l.name, "Background", "a transparent image is not the background");
    assert!(r.document.channels.is_empty(), "no channel when the alpha is transparency");
}

#[test]
fn alpha_as_channel_opens_opaque_with_an_alpha_1_channel() {
    let opts = ImportOptions { dds: DdsImport { alpha_as_channel: true, mips_as_layers: false, ..Default::default() }, ..Default::default() };
    let r = import_with_options("t.dds", &dds_with_mips(), &opts).unwrap();
    assert_eq!(r.document.layers.len(), 1);
    let l = &r.document.layers[0];
    let px = l.surface().unwrap().read_region(photocraft_geom::Rect::new(0, 0, 16, 8));
    assert!(px.chunks(4).all(|p| p[3] == 1.0), "the layer is opaque");
    // Opaque images open as the locked Background, like a 32-bit Targa's.
    assert_eq!(l.name, "Background");
    let [channel] = &r.document.channels[..] else { panic!("the texture's alpha becomes Alpha 1") };
    assert_eq!(channel.name, "Alpha 1");
    let a = channel.surface.read_region(photocraft_geom::Rect::new(0, 0, 16, 8));
    assert_eq!(a.len(), 16 * 8);
    // The texture's alpha (200 - i over 255) is the channel, nothing discarded.
    for (i, &v) in a.iter().enumerate().take(8) {
        assert!((v - (200 - i) as f32 / 255.0).abs() < 0.01, "alpha {i} kept in the channel");
    }
    // Round trip: the channel writes back as the texture's alpha.
    let out = export(&r.document, "t.dds", &ExportOptions::default()).unwrap();
    let again = photocraft_codecs::decode(&out.bytes).unwrap();
    let px = again.to_normalized();
    for i in [0, 4, 8] {
        assert!((px[i * 4 + 3] - (200 - i) as f32 / 255.0).abs() < 0.01, "alpha {i} round-trips");
    }
}

#[test]
fn alpha_as_channel_on_an_opaque_texture_makes_no_channel() {
    let img = photocraft_codecs::Image::from_u8(4, 4, photocraft_codecs::ChannelLayout::Rgba, [10, 20, 30, 255].repeat(16)).unwrap();
    let b = photocraft_codecs::encode(&img, photocraft_codecs::Format::Dds, &photocraft_codecs::EncodeOptions::default()).unwrap();
    let opts = ImportOptions { dds: DdsImport { alpha_as_channel: true, mips_as_layers: false, ..Default::default() }, ..Default::default() };
    let r = import_with_options("t.dds", &b, &opts).unwrap();
    assert!(r.document.channels.is_empty(), "an all-opaque alpha holds no data to keep");
}

#[test]
fn alpha_as_channel_with_mips_keeps_the_top_levels_alpha() {
    let opts = ImportOptions { dds: DdsImport { alpha_as_channel: true, mips_as_layers: true, ..Default::default() }, ..Default::default() };
    let r = import_with_options("t.dds", &dds_with_mips(), &opts).unwrap();
    let names: Vec<&str> = r.document.layers.iter().map(|l| l.name.as_str()).collect();
    assert_eq!(names, ["Mip 0", "Mip 1", "Mip 2", "Mip 3", "Mip 4"]);
    for l in &r.document.layers {
        let b = l.surface().unwrap().content_bounds();
        let px = l.surface().unwrap().read_region(b);
        assert!(px.chunks(4).all(|p| p[3] == 1.0), "{} is opaque", l.name);
    }
    assert_eq!(r.document.channels.len(), 1);
}

#[test]
fn export_warns_when_a_bc4_texture_cannot_keep_the_alpha_channel() {
    let opts = ImportOptions { dds: DdsImport { alpha_as_channel: true, mips_as_layers: false, ..Default::default() }, ..Default::default() };
    let r = import_with_options("t.dds", &dds_with_mips(), &opts).unwrap();
    let out = export(
        &r.document,
        "t.dds",
        &ExportOptions {
            encode: photocraft_codecs::EncodeOptions {
                dds: photocraft_codecs::DdsOptions { compression: DdsCompression::Bc4, ..Default::default() },
                ..Default::default()
            },
            ..Default::default()
        },
    )
    .unwrap();
    assert!(out.warnings.iter().any(|w| w.contains("no alpha")), "{:?}", out.warnings);
}

#[test]
fn export_with_several_alpha_channels_writes_none() {
    let opts = ImportOptions { dds: DdsImport { alpha_as_channel: true, mips_as_layers: false, ..Default::default() }, ..Default::default() };
    let mut r = import_with_options("t.dds", &dds_with_mips(), &opts).unwrap();
    let extra = r.document.channels[0].clone();
    r.document.channels.push(extra);
    let out = export(
        &r.document,
        "t.dds",
        &ExportOptions {
            encode: photocraft_codecs::EncodeOptions {
                dds: photocraft_codecs::DdsOptions { compression: DdsCompression::Bc3, ..Default::default() },
                ..Default::default()
            },
            ..Default::default()
        },
    )
    .unwrap();
    assert!(out.warnings.iter().any(|w| w.contains("one alpha channel")), "{:?}", out.warnings);
}

#[test]
fn mipmaps_as_layers_opens_one_layer_per_level() {
    let opts = ImportOptions { dds: DdsImport { alpha_as_channel: false, mips_as_layers: true, ..Default::default() }, ..Default::default() };
    let r = import_with_options("t.dds", &dds_with_mips(), &opts).unwrap();
    // 16×8 → 8×4, 4×2, 2×1, 1×1.
    let names: Vec<&str> = r.document.layers.iter().map(|l| l.name.as_str()).collect();
    assert_eq!(names, ["Mip 0", "Mip 1", "Mip 2", "Mip 3", "Mip 4"]);
    for (i, l) in r.document.layers.iter().enumerate() {
        let b = l.surface().unwrap().content_bounds();
        assert_eq!((b.width(), b.height()), ((16 >> i).max(1), (8 >> i).max(1)), "Mip {i}");
    }
    // The document stays the top level's size.
    assert_eq!(r.document.size, photocraft_geom::Size::new(16, 8));
}

#[test]
fn mips_as_layers_on_a_single_level_texture_is_one_layer() {
    let img = photocraft_codecs::Image::from_u8(4, 4, photocraft_codecs::ChannelLayout::Rgba, [10, 20, 30, 200].repeat(16)).unwrap();
    let b = photocraft_codecs::encode(&img, photocraft_codecs::Format::Dds, &photocraft_codecs::EncodeOptions::default()).unwrap();
    let opts = ImportOptions { dds: DdsImport { alpha_as_channel: false, mips_as_layers: true, ..Default::default() }, ..Default::default() };
    let r = import_with_options("t.dds", &b, &opts).unwrap();
    assert_eq!(r.document.layers.len(), 1);
    assert_ne!(r.document.layers[0].name, "Mip 0", "no chain, no rename");
}

/// A 2×2 RGBA DDS whose rows differ: red over blue.
fn two_tone_dds() -> Vec<u8> {
    let data = [255, 0, 0, 255, 255, 0, 0, 255, 0, 0, 255, 255, 0, 0, 255, 255].to_vec();
    let img = photocraft_codecs::Image::from_u8(2, 2, photocraft_codecs::ChannelLayout::Rgba, data).unwrap();
    photocraft_codecs::encode(&img, photocraft_codecs::Format::Dds, &photocraft_codecs::EncodeOptions::default()).unwrap()
}

#[test]
fn flip_import_reads_the_rows_bottom_up() {
    let opts = ImportOptions { dds: DdsImport { flip: true, ..Default::default() }, ..Default::default() };
    let r = import_with_options("t.dds", &two_tone_dds(), &opts).unwrap();
    let px = r.document.layers[0].surface().unwrap().read_region(photocraft_geom::Rect::new(0, 0, 2, 2));
    // Top row is the file's bottom row: blue over red.
    assert!((px[0] - 0.0).abs() < 0.01 && (px[2] - 1.0).abs() < 0.01, "top-left is blue: {:?}", &px[..4]);
    assert!((px[8] - 1.0).abs() < 0.01 && (px[10] - 0.0).abs() < 0.01, "bottom-left is red: {:?}", &px[8..12]);
    // The default import keeps the file's order.
    let r = import("t.dds", &two_tone_dds()).unwrap();
    let px = r.document.layers[0].surface().unwrap().read_region(photocraft_geom::Rect::new(0, 0, 2, 2));
    assert!((px[0] - 1.0).abs() < 0.01, "top-left is red: {:?}", &px[..4]);
}

#[test]
fn a_save_with_no_dds_options_writes_what_the_file_was() {
    // A BC3 texture with a mip chain: importing then exporting without choosing DDS options
    // writes BC3 with the chain again, as the plug-ins' match-source default.
    let img = photocraft_codecs::Image::from_u8(8, 8, photocraft_codecs::ChannelLayout::Rgba, [10, 200, 30, 255].repeat(64)).unwrap();
    let bc3 = photocraft_codecs::encode(
        &img,
        photocraft_codecs::Format::Dds,
        &photocraft_codecs::EncodeOptions {
            dds: photocraft_codecs::DdsOptions { compression: DdsCompression::Bc3, mipmaps: true, ..Default::default() },
            ..Default::default()
        },
    )
    .unwrap();
    let r = import("t.dds", &bc3).unwrap();
    let out = export(&r.document, "t.dds", &ExportOptions::default()).unwrap();
    let info = photocraft_codecs::dds_info(&out.bytes).unwrap();
    assert_eq!(info.compression, Some(DdsCompression::Bc3), "the source's compression stands");
    assert!(info.levels > 1, "the source's mip chain stands");
    // An explicit choice overrides it.
    let out =
        export(&r.document, "t.dds", &ExportOptions { dds: Some(DdsOptions { compression: DdsCompression::Bc1, ..Default::default() }), ..Default::default() })
            .unwrap();
    assert_eq!(photocraft_codecs::dds_info(&out.bytes).unwrap().compression, Some(DdsCompression::Bc1));
}

#[test]
fn flip_export_writes_the_rows_bottom_up() {
    let r = import("t.dds", &two_tone_dds()).unwrap();
    let out = export(&r.document, "t.dds", &ExportOptions { dds: Some(DdsOptions { flip: true, ..Default::default() }), ..Default::default() }).unwrap();
    let back = photocraft_codecs::decode(&out.bytes).unwrap().to_normalized();
    assert!((back[0] - 0.0).abs() < 0.01 && (back[2] - 1.0).abs() < 0.01, "stored top-left is blue: {:?}", &back[..4]);
}

/// A 4×4 cubemap, each face a solid colour, uncompressed so round trips are exact.
fn cube_dds() -> (Vec<u8>, [[u8; 3]; 6]) {
    let colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [0, 255, 255], [255, 0, 255]];
    let faces: Vec<photocraft_codecs::Image> = colors
        .iter()
        .map(|&[r, g, b]| photocraft_codecs::Image::from_u8(4, 4, photocraft_codecs::ChannelLayout::Rgba, [r, g, b, 255].repeat(16)).unwrap())
        .collect();
    let b = photocraft_codecs::encode_dds_cube(
        &faces,
        &photocraft_codecs::EncodeOptions {
            dds: photocraft_codecs::DdsOptions { compression: DdsCompression::None, ..Default::default() },
            ..Default::default()
        },
    )
    .unwrap();
    (b, colors)
}

fn cube_import(mode: DdsCubeMap, bytes: &[u8]) -> ImportResult {
    import_with_options("c.dds", bytes, &ImportOptions { dds: DdsImport { cube: mode, ..Default::default() }, ..Default::default() }).unwrap()
}

#[test]
fn a_cubemap_imports_as_six_face_layers() {
    let (bytes, colors) = cube_dds();
    let r = cube_import(DdsCubeMap::Layers, &bytes);
    let names: Vec<&str> = r.document.layers.iter().map(|l| l.name.as_str()).collect();
    assert_eq!(names, ["Face +X", "Face -X", "Face +Y", "Face -Y", "Face +Z", "Face -Z"]);
    assert_eq!(r.document.size, photocraft_geom::Size::new(4, 4));
    // Each face's colour survives, in stack order (+X is bottom-most).
    for (l, &[r_, g, b_]) in r.document.layers.iter().zip(&colors) {
        let px = l.surface().unwrap().read_region(photocraft_geom::Rect::new(0, 0, 4, 4));
        assert!((px[0] - r_ as f32 / 255.0).abs() < 0.01, "{} red", l.name);
        assert!((px[1] - g as f32 / 255.0).abs() < 0.01, "{} green", l.name);
        assert!((px[2] - b_ as f32 / 255.0).abs() < 0.01, "{} blue", l.name);
    }
}

#[test]
fn a_cubemap_imports_as_the_cross_sheet() {
    let (bytes, colors) = cube_dds();
    let r = cube_import(DdsCubeMap::Cross, &bytes);
    assert_eq!(r.document.size, photocraft_geom::Size::new(16, 12));
    assert_eq!(r.document.layers.len(), 1);
    let px = r.document.layers[0].surface().unwrap().read_region(photocraft_geom::Rect::new(0, 0, 16, 12));
    // Cells: -X +Z +X -Z across the middle row, +Y above and -Y below +Z.
    for (face, &(cx, cy)) in [(2usize, 1usize), (0, 1), (1, 0), (1, 2), (1, 1), (3, 1)].iter().enumerate() {
        let at = ((cy * 4) * 16 + cx * 4 + 2) * 4; // middle pixel of the cell
        let [r_, g, b_] = colors[face];
        assert!(
            (px[at] - r_ as f32 / 255.0).abs() < 0.01 && (px[at + 1] - g as f32 / 255.0).abs() < 0.01 && (px[at + 2] - b_ as f32 / 255.0).abs() < 0.01,
            "face {face} at cell {cx},{cy}"
        );
    }
}

#[test]
fn a_cubemap_with_cube_off_opens_the_x_face_only() {
    let (bytes, _) = cube_dds();
    let r = cube_import(DdsCubeMap::Off, &bytes);
    assert_eq!(r.document.layers.len(), 1);
    let px = r.document.layers[0].surface().unwrap().read_region(photocraft_geom::Rect::new(0, 0, 4, 4));
    assert!((px[0] - 1.0).abs() < 0.01 && (px[1] - 0.0).abs() < 0.01, "+X is the red face");
}

#[test]
fn a_face_layers_document_saves_a_cubemap_back() {
    let (bytes, colors) = cube_dds();
    let r = cube_import(DdsCubeMap::Layers, &bytes);
    let out = export(&r.document, "c.dds", &ExportOptions { dds: Some(DdsOptions { cube: DdsCubeMap::Layers, ..Default::default() }), ..Default::default() })
        .unwrap();
    let info = photocraft_codecs::dds_info(&out.bytes).unwrap();
    assert!((info.cube, info.faces) == (true, 6));
    let faces = photocraft_codecs::decode_dds_faces(&out.bytes, &photocraft_codecs::DecodeOptions::default()).unwrap();
    assert_eq!(faces.len(), 6);
    for ((face, chain), &[r_, g, b_]) in faces.iter().zip(&colors) {
        let p = &chain[0].data()[..3];
        assert_eq!((*face, [p[0], p[1], p[2]]), (*face, [r_, g, b_]), "face {face} kept its colour");
    }
}

#[test]
fn a_cross_sheet_document_saves_a_cubemap_back() {
    let (bytes, colors) = cube_dds();
    let r = cube_import(DdsCubeMap::Cross, &bytes);
    let out = export(
        &r.document,
        "c.dds",
        &ExportOptions { dds: Some(DdsOptions { cube: DdsCubeMap::Cross, compression: DdsCompression::None, ..Default::default() }), ..Default::default() },
    )
    .unwrap();
    assert!(photocraft_codecs::dds_info(&out.bytes).unwrap().cube);
    let faces = photocraft_codecs::decode_dds_faces(&out.bytes, &photocraft_codecs::DecodeOptions::default()).unwrap();
    for (face, chain) in &faces {
        let p = &chain[0].data()[..3];
        let [r_, g, b_] = colors[usize::from(*face)];
        assert_eq!([p[0], p[1], p[2]], [r_, g, b_], "face {face} kept its colour");
    }
}

#[test]
fn a_cubemap_source_saves_as_a_cubemap_by_default() {
    let (bytes, _) = cube_dds();
    let r = cube_import(DdsCubeMap::Layers, &bytes);
    let out = export(&r.document, "c.dds", &ExportOptions::default()).unwrap();
    assert!(photocraft_codecs::dds_info(&out.bytes).unwrap().cube, "match-source keeps the cube");
    // So does a cross-sheet import.
    let r = cube_import(DdsCubeMap::Cross, &bytes);
    let out = export(&r.document, "c.dds", &ExportOptions::default()).unwrap();
    assert!(photocraft_codecs::dds_info(&out.bytes).unwrap().cube, "the cross sheet writes a cube too");
}

#[test]
fn a_cubemap_save_needs_six_pixel_layers() {
    let r = import("t.dds", &dds_with_mips()).unwrap(); // one layer, 16×8
    let out = export(&r.document, "c.dds", &ExportOptions { dds: Some(DdsOptions { cube: DdsCubeMap::Layers, ..Default::default() }), ..Default::default() });
    assert!(matches!(out, Err(IoError::Unsupported(_))), "{out:?}");
}

#[test]
fn a_cross_save_needs_a_four_three_sheet() {
    let r = import("t.dds", &dds_with_mips()).unwrap(); // 16×8, not 4:3 cells
    let out = export(&r.document, "c.dds", &ExportOptions { dds: Some(DdsOptions { cube: DdsCubeMap::Cross, ..Default::default() }), ..Default::default() });
    assert!(matches!(out, Err(IoError::Unsupported(_))), "{out:?}");
}
