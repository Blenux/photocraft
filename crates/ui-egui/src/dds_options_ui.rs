//! DDS Options: the compression and mip chain File › Save As and Save a Copy ask for before
//! writing a DirectDraw Surface texture (#1327). Photoshop has no DDS writer of its own (a plug-in
//! adds one), so this follows the texture tools' usual choices: uncompressed, BC1–BC3
//! (DXT1/DXT3/DXT5), BC4/BC5 (ATI1/ATI2), BC7 or an A8 mask, the mip chain's filter and limits,
//! the sRGB mark and the cubemap shape (see `photocraft_io::DdsOptions`).
//!
//! The save is parked in [`Prompt`] while the modal is up (egui is immediate-mode) and written
//! once answered. Cancel drops the save. Scripts skip the prompt by passing the `dds*` params
//! (`ddsCompression`, `ddsMipmaps`, … — see [`options_param`]).

use photocraft_io::{DdsCompression, DdsCubeMap, DdsOptions, DdsQuality};

use crate::{ExportSettings, PhotocraftApp};

/// A save waiting on the DDS Options answer.
pub struct Prompt {
    /// The document that requested the save, even if tabs change while this is open.
    doc: photocraft_doc::DocId,
    /// Where the texture goes.
    pub path: String,
    /// The options chosen, starting at what the document's own file was written with (the
    /// plug-ins' match-source default).
    pub options: DdsOptions,
    /// Whether the document's bottom layers can be cubemap faces (six pixel layers on a
    /// square canvas) or its canvas is a 4:3 cross-layout sheet — the dialog's Cubemap row.
    pub cube_layers: bool,
    pub cube_cross: bool,
    /// Save a Copy: the document keeps its path and unsaved changes.
    pub copy: bool,
    /// The step the answered save is recorded as (`file.saveAs` or `file.saveACopy`), so an
    /// action being recorded keeps it; `file.save` when File › Save confirmed the document's own
    /// texture (#3531), which plays back on the file the action is played on.
    pub record: &'static str,
}

/// `true` when `path` names a DDS texture.
pub fn wants_prompt(path: &str) -> bool {
    photocraft_engine::file_cmds::extension(path).and_then(|e| photocraft_codecs::from_extension(&e)) == Some(photocraft_codecs::Format::Dds)
}

/// The DDS options a command or control call asks for (`"ddsCompression"`, `"ddsMipmaps"`,
/// `"ddsQuality"`, `"ddsSrgb"`, `"ddsMipFilter"`, `"ddsMinMipSize"`, `"ddsMaxMipCount"`,
/// `"ddsMipGammaCorrect"`, `"ddsMipPremultAlpha"`, `"ddsPremultAlpha"`, `"ddsNormalMap"`,
/// `"ddsFlip"`, `"ddsCube"`), which skip the prompt.
pub fn options_param(params: &serde_json::Value, cmd: &str) -> Result<Option<DdsOptions>, String> {
    photocraft_engine::file_cmds::dds_param(params, cmd).map_err(|e| e.to_string())
}

/// The params a save with `options` is recorded with.
pub fn record(options: DdsOptions, step: &mut serde_json::Value) {
    step["ddsCompression"] = options.compression.name().into();
    step["ddsMipmaps"] = options.mipmaps.into();
    step["ddsQuality"] = options.quality.name().into();
    step["ddsSrgb"] = options.srgb.into();
    step["ddsMipFilter"] = options.mip_filter.name().into();
    step["ddsMinMipSize"] = options.min_mip_size.into();
    step["ddsMaxMipCount"] = options.max_mip_count.into();
    step["ddsMipGammaCorrect"] = options.mip_gamma_correct.into();
    step["ddsMipPremultAlpha"] = options.mip_premult_alpha.into();
    step["ddsPremultAlpha"] = options.export_premult_alpha.into();
    step["ddsNormalMap"] = options.normal_map.into();
    step["ddsFlip"] = options.flip.into();
    step["ddsCube"] = options.cube.name().into();
}

/// Parks the save behind the prompt. A prompt already waiting is never replaced.
pub fn park(app: &mut PhotocraftApp, path: String, copy: bool, record: &'static str) -> Result<(), String> {
    if app.save_options_open() {
        return Err(crate::SAVE_OPTIONS_OPEN.into());
    }
    let doc = app.active_doc_id()?;
    // The choices start at what the document's own file was written with, as the plug-ins'
    // dialogs do, falling back to uncompressed without mipmaps for anything else.
    let (options, cube_layers, cube_cross) = app
        .session
        .active()
        .map(|st| (photocraft_io::dds_source_options(&st.doc), photocraft_io::dds_face_layers_ok(&st.doc), photocraft_io::dds_cross_sheet(&st.doc)))
        .unwrap_or_default();
    app.dds_options = Some(Prompt { doc, path, options, cube_layers, cube_cross, copy, record });
    Ok(())
}

/// The block compressions go by their format names, which stay untranslated like other
/// technology names. Shared with the DDS Import prompt's facts line.
pub(crate) fn compression_label(c: DdsCompression) -> &'static str {
    match c {
        DdsCompression::None => tl!("None"),
        DdsCompression::Bc1 => "BC1 / DXT1",
        DdsCompression::Bc1a => "BC1a / DXT1a",
        DdsCompression::Bc2 => "BC2 / DXT3",
        DdsCompression::Bc3 => "BC3 / DXT5",
        DdsCompression::Bc4 => "BC4 / ATI1",
        DdsCompression::Bc5 => "BC5 / ATI2",
        DdsCompression::Bc7 => "BC7",
        DdsCompression::A8 => tl!("A8 / alpha only"),
    }
}

fn quality_label(q: DdsQuality) -> &'static str {
    match q {
        DdsQuality::Fastest => tl!("Fastest"),
        DdsQuality::Normal => tl!("Normal"),
        DdsQuality::Production => tl!("Production"),
        DdsQuality::Highest => tl!("Highest"),
    }
}

fn filter_label(f: photocraft_io::DdsMipFilter) -> &'static str {
    match f {
        photocraft_io::DdsMipFilter::Box => tl!("Box"),
        photocraft_io::DdsMipFilter::Triangle => tl!("Triangle"),
        photocraft_io::DdsMipFilter::Min => tl!("Min"),
        photocraft_io::DdsMipFilter::Max => tl!("Max"),
        photocraft_io::DdsMipFilter::Kaiser => tl!("Kaiser"),
        photocraft_io::DdsMipFilter::MitchellNetravali => tl!("Mitchell-Netravali"),
    }
}

fn cube_label(c: DdsCubeMap) -> &'static str {
    match c {
        DdsCubeMap::Off => tl!("Off"),
        DdsCubeMap::Layers => tl!("Faces from layers"),
        DdsCubeMap::Cross => tl!("Cross-layout sheet"),
    }
}

pub fn show(app: &mut PhotocraftApp, ctx: &egui::Context) {
    let Some(p) = &mut app.dds_options else { return };
    let (mut ok, mut cancel) = (false, false);
    let mut options = p.options;
    let modal = egui::Modal::new(egui::Id::unique("dds-options")).show(ctx, |ui| {
        ui.set_max_width(340.0);
        ui.label(egui::RichText::new(tl!("DDS Options")).font(crate::theme::semibold(15.0)));
        ui.add_space(4.0);
        crate::widgets::hairline(ui);
        ui.add_space(8.0);
        // The dialog sizes to the options that apply; the scroll cap only bites when a long
        // translation wraps a row.
        egui::ScrollArea::vertical().max_height(430.0).show(ui, |ui| {
            crate::widgets::section_label(ui, tl!("Compression"));
            egui::ComboBox::from_id_salt("dds-compression").selected_text(compression_label(options.compression)).show_ui(ui, |ui| {
                for c in DdsCompression::ALL {
                    ui.selectable_value(&mut options.compression, c, compression_label(c));
                }
            });
            // Only BC7's encoder has a speed/quality trade-off; the rest encode one way.
            if options.compression == DdsCompression::Bc7 {
                ui.horizontal(|ui| {
                    ui.label(tl!("Quality"));
                    for q in DdsQuality::ALL {
                        ui.radio_value(&mut options.quality, q, quality_label(q));
                    }
                });
            }
            // Options a codec can't honour stay out rather than sit disabled, so the dialog is
            // exactly as big as what it asks: A8, BC4 and BC5 have no sRGB DXGI name, and the
            // single/dual-channel formats have no colour alpha to premultiply.
            let color_alpha = !matches!(options.compression, DdsCompression::Bc4 | DdsCompression::Bc5 | DdsCompression::A8);
            if color_alpha {
                crate::widgets::checkbox(ui, &mut options.srgb, tl!("sRGB"));
            }
            ui.add_space(8.0);
            crate::widgets::checkbox(ui, &mut options.mipmaps, tl!("Generate Mipmaps"));
            if options.mipmaps {
                ui.horizontal(|ui| {
                    ui.label(tl!("Filter"));
                    egui::ComboBox::from_id_salt("dds-mip-filter").selected_text(filter_label(options.mip_filter)).show_ui(ui, |ui| {
                        for f in photocraft_io::DdsMipFilter::ALL {
                            ui.selectable_value(&mut options.mip_filter, f, filter_label(f));
                        }
                    });
                });
                ui.horizontal(|ui| {
                    ui.label(tl!("Stop at"));
                    ui.add(egui::DragValue::new(&mut options.min_mip_size).range(1..=16384));
                    ui.label(tl!("px; at most"));
                    let mut cap = options.max_mip_count as i32;
                    if ui.add(egui::DragValue::new(&mut cap).range(0..=14)).changed() {
                        options.max_mip_count = cap.max(0) as u32;
                    }
                    ui.label(tl!("levels"));
                });
                if options.srgb && color_alpha {
                    crate::widgets::checkbox(ui, &mut options.mip_gamma_correct, tl!("Filter in linear light"));
                }
                if color_alpha {
                    crate::widgets::checkbox(ui, &mut options.mip_premult_alpha, tl!("Filter premultiplied (no alpha bleed)"));
                }
            }
            ui.add_space(4.0);
            crate::widgets::checkbox(ui, &mut options.normal_map, tl!("Normal map"));
            if color_alpha {
                crate::widgets::checkbox(ui, &mut options.export_premult_alpha, tl!("Premultiplied alpha"));
            }
            crate::widgets::checkbox(ui, &mut options.flip, tl!("Flip vertically"));
            // A cubemap write when the document could hold one: six face layers, or a 4:3
            // cross-layout sheet. The choice needs one of those shapes to mean anything.
            if p.cube_layers || p.cube_cross {
                ui.add_space(4.0);
                ui.horizontal(|ui| {
                    ui.label(tl!("Cubemap"));
                    egui::ComboBox::from_id_salt("dds-cube").selected_text(cube_label(options.cube)).show_ui(ui, |ui| {
                        ui.selectable_value(&mut options.cube, DdsCubeMap::Off, tl!("Off"));
                        ui.add_enabled_ui(p.cube_layers, |ui| {
                            ui.selectable_value(&mut options.cube, DdsCubeMap::Layers, tl!("Faces from layers"));
                        });
                        ui.add_enabled_ui(p.cube_cross, |ui| {
                            ui.selectable_value(&mut options.cube, DdsCubeMap::Cross, tl!("Cross-layout sheet"));
                        });
                    });
                });
            }
        });
        ui.add_space(12.0);
        // The strip gets only the button height: `with_layout` would give it the area's whole
        // remembered rect, centring the buttons in leftover space and keeping the dialog the
        // size of its tallest contents so far.
        let strip = egui::vec2(ui.available_width(), crate::widgets::button_height(ui.ctx()));
        ui.allocate_ui_with_layout(strip, egui::Layout::right_to_left(egui::Align::Center), |ui| {
            ui.spacing_mut().item_spacing.x = 10.0;
            ok = crate::widgets::primary_button(ui, tl!("OK"), 84.0).clicked();
            cancel = crate::widgets::secondary_button(ui, tl!("Cancel"), 84.0).clicked();
        });
    });
    cancel |= modal.should_close();
    p.options = options;
    // A focused button takes Enter as its click without consuming the key: Enter on Cancel cancels.
    if ok || (!cancel && ctx.input_mut(|i| i.consume_key(egui::Modifiers::NONE, egui::Key::Enter))) {
        let Some(p) = app.dds_options.take() else { return };
        let settings = ExportSettings { dds: Some(p.options), ..app.save_settings() };
        let mut step = if p.record == "file.save" { serde_json::json!({}) } else { serde_json::json!({"path": p.path}) };
        record(p.options, &mut step);
        let written = app.with_document(p.doc, |app| app.write_document(p.path, &settings, p.copy));
        if written.is_ok() {
            // An action being recorded keeps the Save As or Save a Copy with this answer.
            app.session.journal.push((p.record.into(), step));
        }
        match written {
            Err(e) => {
                app.ui.status = crate::i18n::fmt(tl!("Save failed: {error}"), &[("error", &e)]);
                app.ui.status_error = true;
            }
            Ok(Some(_)) if !p.copy => crate::discard_ui::saved_document(app, ctx, p.doc),
            // A background save releases the close prompt when it finishes (`jobs_ui`).
            Ok(_) => {}
        }
    } else if cancel {
        app.dds_options = None;
        app.ui.status = tl!("Save cancelled").into();
        app.ui.status_error = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dds_names() {
        assert!(wants_prompt("a.dds"));
        assert!(wants_prompt("dir/sub/a.DDS"));
        assert!(wants_prompt("C:\\x\\y.dds"));
        assert!(!wants_prompt("a.dds.psd"));
        assert!(!wants_prompt(".dds"));
        assert!(!wants_prompt("a.png"));
    }

    #[test]
    fn options_params() {
        use serde_json::json;
        assert_eq!(options_param(&json!({}), "file.saveAs"), Ok(None));
        assert_eq!(
            options_param(&json!({"ddsCompression": "bc3"}), "file.saveAs"),
            Ok(Some(DdsOptions { compression: DdsCompression::Bc3, ..Default::default() }))
        );
        assert_eq!(
            options_param(&json!({"ddsMipmaps": true}), "file.saveAs"),
            Ok(Some(DdsOptions { compression: DdsCompression::None, mipmaps: true, ..Default::default() }))
        );
        assert_eq!(
            options_param(&json!({"ddsCompression": "bc7", "ddsQuality": "fast", "ddsSrgb": true}), "file.saveAs"),
            Ok(Some(DdsOptions { compression: DdsCompression::Bc7, quality: DdsQuality::Fastest, srgb: true, ..Default::default() }))
        );
        for bad in [
            json!({"ddsCompression": "bc6h"}),
            json!({"ddsCompression": 3}),
            json!({"ddsMipmaps": "yes"}),
            json!({"ddsQuality": "best"}),
            json!({"ddsSrgb": 1}),
        ] {
            assert!(options_param(&bad, "file.saveAs").is_err(), "{bad}");
        }
        let mut step = json!({"path": "a.dds"});
        record(DdsOptions { compression: DdsCompression::Bc1, mipmaps: true, ..Default::default() }, &mut step);
        assert_eq!(
            step,
            json!({
                "path": "a.dds", "ddsCompression": "bc1", "ddsMipmaps": true, "ddsQuality": "highest",
                "ddsSrgb": false, "ddsMipFilter": "box", "ddsMinMipSize": 1, "ddsMaxMipCount": 0,
                "ddsMipGammaCorrect": true, "ddsMipPremultAlpha": true, "ddsPremultAlpha": false,
                "ddsNormalMap": false, "ddsFlip": false, "ddsCube": "off"
            })
        );
        assert_eq!(options_param(&step, "file.saveAs"), Ok(Some(DdsOptions { compression: DdsCompression::Bc1, mipmaps: true, ..Default::default() })));
    }
}
