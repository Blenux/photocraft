//! DDS Import on File › Open (#8632): the prompt a texture's alpha channel and mip chain
//! park behind, the answers it opens with, and the scripted `ddsAlphaChannel`/`ddsMipmaps` that
//! skip it.
use super::*;
use crate::dds_import_ui;
use egui_kittest::kittest::Queryable;
use photocraft_io::DdsImport;
use std::{cell::RefCell, rc::Rc};

/// A 16×8 RGBA DDS with a mip chain and an alpha channel: the conditional questions to ask.
fn dds_bytes() -> Vec<u8> {
    let img = photocraft_codecs::Image::from_u8(16, 8, photocraft_codecs::ChannelLayout::Rgba, [10, 20, 30, 200].repeat(16 * 8)).unwrap();
    photocraft_codecs::encode(
        &img,
        photocraft_codecs::Format::Dds,
        &photocraft_codecs::EncodeOptions {
            dds: photocraft_codecs::DdsOptions { compression: photocraft_codecs::DdsCompression::None, mipmaps: true },
            ..Default::default()
        },
    )
    .unwrap()
}

/// An app that opens synchronously, whose importer records the DDS answers it was asked with.
fn app_importing() -> (Prompted, Rc<RefCell<Option<DdsImport>>>) {
    let mut app = app_with_docs(0);
    app.background_jobs = false;
    let chosen = Rc::new(RefCell::new(None));
    let seen = chosen.clone();
    app.services.import = Some(Box::new(move |name: &str, _b: &[u8], opts: &photocraft_io::ImportOptions| {
        *seen.borrow_mut() = Some(opts.dds);
        Ok((
            photocraft_doc::Document::new(name, photocraft_geom::Size::new(16, 8), photocraft_color::ColorMode::Rgb, photocraft_color::SampleType::U8),
            Vec::new(),
        ))
    }));
    let h = egui_kittest::Harness::builder().with_size(egui::vec2(800.0, 600.0)).build_ui_state(
        |ui, app| {
            show(app, ui.ctx());
            dds_import_ui::show(app, ui.ctx());
        },
        app,
    );
    PhotocraftApp::setup_context(&h.ctx, crate::theme::ThemeKind::ALL[0]);
    (h, chosen)
}

/// Opening a mip-mapped DDS asks first; OK opens with what was answered, nothing before.
#[test]
fn opening_a_dds_texture_asks_about_alpha_and_mipmaps() {
    let (mut h, chosen) = app_importing();
    h.state_mut().open_file("textures/skin.dds", &dds_bytes()).unwrap();
    h.run_steps(2);
    let p = h.state().dds_imports.front().expect("parked");
    assert_eq!(p.name, "skin.dds");
    assert_eq!((p.info.width, p.info.height), (16, 8));
    assert!(p.info.levels > 1 && p.info.alpha);
    assert!(chosen.borrow().is_none(), "nothing is decoded before the answer");
    assert!(h.state_mut().open_file("textures/other.dds", &dds_bytes()).is_ok(), "later opens queue behind");

    // The first click on a just-opened modal only brings its layer to the front.
    h.get_by_label("DDS Import").click();
    h.run_steps(2);
    h.get_by_label("Mipmaps as separate layers").click();
    h.run_steps(2);
    h.get_by_label("Load transparency as alpha channel").click();
    h.run_steps(2);
    assert!(h.state().dds_imports.front().unwrap().options.alpha_as_channel, "alpha as a channel");
    h.get_by_label("OK").click();
    h.run_steps(3);
    assert_eq!(*chosen.borrow(), Some(DdsImport { alpha_as_channel: true, mips_as_layers: true, ..Default::default() }));
    assert!(h.state().dds_imports.front().is_some(), "the queued open asks next");
}

/// Cancel drops the open; the importer never runs.
#[test]
fn cancelling_the_dds_prompt_drops_the_open() {
    let (mut h, chosen) = app_importing();
    h.state_mut().open_file("t.dds", &dds_bytes()).unwrap();
    h.run_steps(2);
    h.get_by_label("Cancel").click();
    h.run_steps(3);
    assert!(h.state().dds_imports.is_empty());
    assert!(chosen.borrow().is_none());
    assert_eq!(h.state().ui.status, "Open cancelled");
}

/// `ddsAlphaChannel`/`ddsMipmaps` on `file.open`/`file.openAs` answer without asking; a texture with
/// no mip chain and no alpha never asks.
#[test]
fn scripted_dds_import_answers_and_plain_textures_skip_the_prompt() {
    let (mut h, chosen) = app_importing();
    h.state_mut().open_file_opts("t.dds", &dds_bytes(), Some(DdsImport { alpha_as_channel: true, mips_as_layers: true, ..Default::default() })).unwrap();
    assert!(h.state().dds_imports.is_empty(), "answered opens don't ask");
    assert_eq!(*chosen.borrow(), Some(DdsImport { alpha_as_channel: true, mips_as_layers: true, ..Default::default() }));

    // Params parse to the same answers, and bad values are refused.
    assert_eq!(
        dds_import_ui::params(&json!({"ddsAlphaChannel": true, "ddsMipmaps": true})).unwrap(),
        Some(DdsImport { alpha_as_channel: true, mips_as_layers: true, ..Default::default() })
    );
    assert!(dds_import_ui::params(&json!({})).unwrap().is_none());
    assert!(dds_import_ui::params(&json!({"ddsAlphaChannel": 1})).is_err());
    assert!(dds_import_ui::params(&json!({"ddsMipmaps": "yes"})).is_err());

    // A single-level opaque texture has no choices to make.
    let img = photocraft_codecs::Image::from_u8(4, 4, photocraft_codecs::ChannelLayout::Gray, [128].repeat(16)).unwrap();
    let flat = photocraft_codecs::encode(&img, photocraft_codecs::Format::Dds, &photocraft_codecs::EncodeOptions::default()).unwrap();
    h.state_mut().open_file("flat.dds", &flat).unwrap();
    assert!(h.state().dds_imports.is_empty(), "nothing to ask");
}
