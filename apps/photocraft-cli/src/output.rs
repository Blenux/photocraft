use photocraft_io::{DdsImport, ExportOptions, ImportOptions};
use serde_json::Value;
use std::io::Write;

use crate::{
    Result,
    args::{ExportArgs, ImportArgs},
};

/// The `--format` family's options. A bad value never gets here: clap refused it.
impl From<&ExportArgs> for ExportOptions {
    fn from(a: &ExportArgs) -> Self {
        // TIFF output is flat unless asked: scripted conversions keep predictable sizes.
        let mut o = ExportOptions { tiff_layers: a.tiff_layers, tga_bits: a.tga_bits, ..Default::default() };
        if let Some(q) = a.quality {
            o.encode.jpeg_quality = q;
            // Asking for a quality asks for a lossy WebP; the default WebP stays lossless.
            o.encode.webp_quality = q;
            o.encode.webp_lossless = false;
        }
        o.encode.dpx_bits = a.dpx_bits;
        if let Some(c) = a.tiff_compression {
            o.encode.tiff_compression = c;
        }
        if let Some(f) = a.jpeg_format {
            o.encode.jpeg_format = f;
        }
        // Any DDS save flag builds `dds`; with none the export keeps what the input was
        // written with (the plug-ins' match-source default).
        fn dds(o: &mut ExportOptions) -> &mut photocraft_io::DdsOptions {
            o.dds.get_or_insert_with(photocraft_io::DdsOptions::default)
        }
        if let Some(c) = a.dds_compression {
            dds(&mut o).compression = c;
        }
        if a.dds_mipmaps {
            dds(&mut o).mipmaps = true;
        }
        if let Some(q) = a.dds_quality {
            dds(&mut o).quality = q;
        }
        if a.dds_srgb {
            dds(&mut o).srgb = true;
        }
        if let Some(f) = a.dds_mip_filter {
            dds(&mut o).mip_filter = f;
        }
        if let Some(n) = a.dds_min_mip_size {
            dds(&mut o).min_mip_size = n;
        }
        if let Some(n) = a.dds_max_mip_count {
            dds(&mut o).max_mip_count = n;
        }
        if a.dds_no_mip_gamma {
            dds(&mut o).mip_gamma_correct = false;
        }
        if a.dds_no_mip_premult {
            dds(&mut o).mip_premult_alpha = false;
        }
        if a.dds_premult {
            dds(&mut o).export_premult_alpha = true;
        }
        if a.dds_normal_map {
            dds(&mut o).normal_map = true;
        }
        if a.dds_flip {
            dds(&mut o).flip = true;
        }
        if let Some(c) = a.dds_cube {
            dds(&mut o).cube = c;
        }
        o
    }
}

/// The DDS open choices `--dds-alpha-channel`, `--dds-mips-as-layers`, `--dds-load-flip` and
/// `--dds-load-cube` make; every other import stays at its default.
impl From<&ImportArgs> for ImportOptions {
    fn from(a: &ImportArgs) -> Self {
        ImportOptions {
            dds: DdsImport {
                alpha_as_channel: a.dds_alpha_channel,
                mips_as_layers: a.dds_mips_as_layers,
                flip: a.dds_load_flip,
                cube: a.dds_load_cube.unwrap_or_default(),
            },
            ..Default::default()
        }
    }
}

pub(crate) fn warn_all(err: &mut dyn Write, ws: &[String]) {
    for w in ws {
        let _ = writeln!(err, "warning: {w}");
    }
}

/// How [`print_json`] lays the JSON out.
pub(crate) enum JsonStyle {
    Pretty,
    Compact,
}

pub(crate) fn print_json(out: &mut dyn Write, v: &Value, style: JsonStyle) -> Result {
    let s = match style {
        JsonStyle::Pretty => serde_json::to_string_pretty(v),
        JsonStyle::Compact => serde_json::to_string(v),
    }?;
    Ok(writeln!(out, "{s}")?)
}
