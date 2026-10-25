//! BC7 block compression: a safe-Rust port of `bc7enc` by Richard Geldreich, Jr.
//! (MIT licence or public domain; see `ATTRIBUTION.md`).
//!
//! Supports modes 1, 5, 6 and 7 like the upstream encoder: opaque blocks go to modes 1
//! or 6, blocks with alpha to 5, 6 or 7. Partition sets, anchors and the mode tables are
//! the shared spec data in `dds_bc7.rs`.

use std::sync::OnceLock;

use super::dds_bc::Block;
use super::dds_bc7::{ANCHOR2, ANCHOR3_SECOND, ANCHOR3_THIRD, N_SUBSETS, P2, P3, PARTITION_BITS};
use crate::options::DdsQuality;

type Color = [u8; 4];
type Vec4 = [f32; 4];

fn clampi(v: i32, lo: i32, hi: i32) -> i32 {
    v.clamp(lo, hi)
}

fn saturate(v: f32) -> f32 {
    v.clamp(0.0, 1.0)
}

fn vec4_saturate(v: Vec4) -> Vec4 {
    [saturate(v[0]), saturate(v[1]), saturate(v[2]), saturate(v[3])]
}

fn vec4_from_color(c: &Color) -> Vec4 {
    [f32::from(c[0]), f32::from(c[1]), f32::from(c[2]), f32::from(c[3])]
}

fn vec4_dot(a: &Vec4, b: &Vec4) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]
}

fn vec4_normalize(v: &mut Vec4) {
    let s = vec4_dot(v, v);
    if s != 0.0 {
        let s = 1.0 / s.sqrt();
        for c in v.iter_mut() {
            *c *= s;
        }
    }
}

const WEIGHTS2: [u32; 4] = [0, 21, 43, 64];
const WEIGHTS3: [u32; 8] = [0, 9, 18, 27, 37, 46, 55, 64];
const WEIGHTS4: [u32; 16] = [0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64];

// Precomputed weight constants used during least fit determination. For each weight
// entry: w*w, (1-w)*w, (1-w)*(1-w), w.
const WEIGHTS2X: [Vec4; 4] = [[0.0, 0.0, 1.0, 0.0], [0.107666, 0.220459, 0.451416, 0.328125], [0.451416, 0.220459, 0.107666, 0.671875], [1.0, 0.0, 0.0, 1.0]];
const WEIGHTS3X: [Vec4; 8] = [
    [0.0, 0.0, 1.0, 0.0],
    [0.019775, 0.120850, 0.738525, 0.140625],
    [0.079102, 0.202148, 0.516602, 0.281250],
    [0.177979, 0.243896, 0.334229, 0.421875],
    [0.334229, 0.243896, 0.177979, 0.578125],
    [0.516602, 0.202148, 0.079102, 0.718750],
    [0.738525, 0.120850, 0.019775, 0.859375],
    [1.0, 0.0, 0.0, 1.0],
];
#[rustfmt::skip]
const WEIGHTS4X: [Vec4; 16] = [
    [0.0, 0.0, 1.0, 0.0], [0.003906, 0.058594, 0.878906, 0.062500],
    [0.019775, 0.120850, 0.738525, 0.140625], [0.041260, 0.161865, 0.635010, 0.203125],
    [0.070557, 0.195068, 0.539307, 0.265625], [0.107666, 0.220459, 0.451416, 0.328125],
    [0.165039, 0.241211, 0.352539, 0.406250], [0.219727, 0.249023, 0.282227, 0.468750],
    [0.282227, 0.249023, 0.219727, 0.531250], [0.352539, 0.241211, 0.165039, 0.593750],
    [0.451416, 0.220459, 0.107666, 0.671875], [0.539307, 0.195068, 0.070557, 0.734375],
    [0.635010, 0.161865, 0.041260, 0.796875], [0.738525, 0.120850, 0.019775, 0.859375],
    [0.878906, 0.058594, 0.003906, 0.937500], [1.0, 0.0, 0.0, 1.0],
];

const PARTITION1: [u8; 16] = [0; 16];

const COLOR_INDEX_BITCOUNT: [u8; 8] = [3, 3, 2, 2, 2, 2, 4, 2];
const ALPHA_INDEX_BITCOUNT: [u8; 8] = [0, 0, 0, 0, 3, 2, 4, 2];
const MODE_HAS_P_BITS: [u8; 8] = [1, 1, 0, 1, 0, 0, 1, 1];
const MODE_HAS_SHARED_P_BITS: [u8; 8] = [0, 1, 0, 0, 0, 0, 0, 0];
const COLOR_PRECISION: [u8; 8] = [4, 6, 5, 7, 5, 7, 7, 5];
const ALPHA_PRECISION: [u8; 8] = [0, 0, 0, 0, 6, 8, 7, 5];

fn color_index_size(mode: usize, index_selection: u32) -> u32 {
    u32::from(COLOR_INDEX_BITCOUNT[mode]) + index_selection
}

fn alpha_index_size(mode: usize, index_selection: u32) -> u32 {
    u32::from(ALPHA_INDEX_BITCOUNT[mode]) - index_selection
}

fn mode_has_separate_alpha_selectors(mode: usize) -> bool {
    mode == 4 || mode == 5
}

#[derive(Clone, Copy)]
struct EndpointErr {
    error: u16,
    lo: u8,
    hi: u8,
}

// Tables derived at first use: endpoint midpoints per P-bit, the reduced mode-6
// quantization, and the optimal single-colour endpoint pairs for modes 1 and 7.
struct Tables {
    mode1_optimal: [[EndpointErr; 2]; 256],
    mode7_optimal: [[[EndpointErr; 2]; 2]; 256],
    mode1_midpoints: [[f32; 2]; 64],
    mode5_midpoints: [f32; 128],
    mode7_midpoints: [[f32; 2]; 32],
    mode6_reduced_quant: [[u8; 2]; 2048],
}

static TABLES: OnceLock<Tables> = OnceLock::new();

const MODE_1_OPTIMAL_INDEX: usize = 2;
const MODE_7_OPTIMAL_INDEX: usize = 1;

fn tables() -> &'static Tables {
    TABLES.get_or_init(|| {
        let mut t = Tables {
            mode1_optimal: [[EndpointErr { error: 0, lo: 0, hi: 0 }; 2]; 256],
            mode7_optimal: [[[EndpointErr { error: 0, lo: 0, hi: 0 }; 2]; 2]; 256],
            mode1_midpoints: [[0.0; 2]; 64],
            mode5_midpoints: [0.0; 128],
            mode7_midpoints: [[0.0; 2]; 32],
            mode6_reduced_quant: [[0; 2]; 2048],
        };

        for p in 0..2usize {
            for i in 0..32usize {
                let mut vl = ((i as u32) << 1 | p as u32) << 2;
                vl |= vl >> 6;
                let lo = vl as f32 / 255.0;
                let mut vh = ((i as u32 + 1).min(31) << 1 | p as u32) << 2;
                vh |= vh >> 6;
                let hi = vh as f32 / 255.0;
                t.mode7_midpoints[i][p] = if i == 31 { 1.0 } else { (lo + hi) / 2.0 };
            }
        }
        for p in 0..2usize {
            for i in 0..64usize {
                let mut vl = ((i as u32) << 1 | p as u32) << 1;
                vl |= vl >> 7;
                let lo = vl as f32 / 255.0;
                let mut vh = ((i as u32 + 1).min(63) << 1 | p as u32) << 1;
                vh |= vh >> 7;
                let hi = vh as f32 / 255.0;
                t.mode1_midpoints[i][p] = if i == 63 { 1.0 } else { (lo + hi) / 2.0 };
            }
        }
        for i in 0..128usize {
            let mut vl = (i as u32) << 1;
            vl |= vl >> 7;
            let lo = vl as f32 / 255.0;
            let mut vh = ((i as u32 + 1).min(127)) << 1;
            vh |= vh >> 7;
            let hi = vh as f32 / 255.0;
            t.mode5_midpoints[i] = if i == 127 { 1.0 } else { (lo + hi) / 2.0 };
        }
        for p in 0..2usize {
            for (i, q) in t.mode6_reduced_quant.iter_mut().enumerate() {
                let f = i as f32 / 2047.0;
                let (mut best_err, mut best_index) = (1e9f32, 0u8);
                for j in 0..64u32 {
                    let ik = (j * 127 + 31) / 63;
                    let k = ((ik << 1) + p as u32) as f32 / 255.0;
                    let e = (k - f).abs();
                    if e < best_err {
                        best_err = e;
                        best_index = ik as u8;
                    }
                }
                q[p] = best_index;
            }
        }
        for (c, row) in t.mode1_optimal.iter_mut().enumerate() {
            for (lp, e) in row.iter_mut().enumerate() {
                let mut best = EndpointErr { error: u16::MAX, lo: 0, hi: 0 };
                for l in 0..64u32 {
                    let mut low = ((l << 1) | lp as u32) << 1;
                    low |= low >> 7;
                    for h in 0..64u32 {
                        let mut high = ((h << 1) | lp as u32) << 1;
                        high |= high >> 7;
                        let k = (low * (64 - WEIGHTS3[MODE_1_OPTIMAL_INDEX]) + high * WEIGHTS3[MODE_1_OPTIMAL_INDEX] + 32) >> 6;
                        let err = (k as i32 - c as i32).pow(2);
                        if (err as u16) < best.error {
                            best = EndpointErr { error: err as u16, lo: l as u8, hi: h as u8 };
                        }
                    }
                }
                *e = best;
            }
        }
        for (c, row) in t.mode7_optimal.iter_mut().enumerate() {
            for (hp, row2) in row.iter_mut().enumerate() {
                for (lp, e) in row2.iter_mut().enumerate() {
                    let mut best = EndpointErr { error: u16::MAX, lo: 0, hi: 0 };
                    for l in 0..32u32 {
                        let mut low = ((l << 1) | lp as u32) << 2;
                        low |= low >> 6;
                        for h in 0..32u32 {
                            let mut high = ((h << 1) | hp as u32) << 2;
                            high |= high >> 6;
                            let k = (low * (64 - WEIGHTS2[MODE_7_OPTIMAL_INDEX]) + high * WEIGHTS2[MODE_7_OPTIMAL_INDEX] + 32) >> 6;
                            let err = (k as i32 - c as i32).pow(2);
                            if (err as u16) < best.error {
                                best = EndpointErr { error: err as u16, lo: l as u8, hi: h as u8 };
                            }
                        }
                    }
                    *e = best;
                }
            }
        }
        t
    })
}

/// Encoder knobs, a subset of `bc7enc_compress_block_params`: the upstream defaults are
/// the highest-quality settings PhotoCraft uses.
#[derive(Clone)]
struct CompParams {
    mode_mask: u32,
    max_partitions: u32,
    try_least_squares: bool,
    mode17_partition_estimation_filterbank: bool,
    uber_level: i32,
    force_selectors: bool,
    selectors: [u8; 16],
    force_alpha: bool,
    quant_mode6_endpoints: bool,
    bias_mode1_pbits: bool,
    pbit1_weight: f32,
    mode1_error_weight: f32,
    mode5_error_weight: f32,
    mode6_error_weight: f32,
    mode7_error_weight: f32,
    low_frequency_partition_weight: f32,
    perceptual: bool,
    weights: [u32; 4],
}

impl Default for CompParams {
    fn default() -> Self {
        Self {
            mode_mask: u32::MAX,
            max_partitions: 64,
            try_least_squares: true,
            mode17_partition_estimation_filterbank: false,
            uber_level: 4,
            force_selectors: false,
            selectors: [0; 16],
            force_alpha: false,
            quant_mode6_endpoints: true,
            bias_mode1_pbits: true,
            pbit1_weight: 1.0,
            mode1_error_weight: 1.0,
            mode5_error_weight: 1.0,
            mode6_error_weight: 1.0,
            mode7_error_weight: 1.0,
            low_frequency_partition_weight: 1.0,
            perceptual: true,
            weights: [128, 64, 16, 32],
        }
    }
}

struct CellParams<'a> {
    num_selector_weights: u32,
    selector_weights: &'a [u32],
    selector_weightsx: &'a [Vec4],
    comp_bits: u32,
    weights: [u32; 4],
    has_alpha: bool,
    has_pbits: bool,
    endpoints_share_pbit: bool,
    perceptual: bool,
}

#[derive(Clone)]
struct CellResults {
    best_overall_err: u64,
    low_endpoint: Color,
    high_endpoint: Color,
    pbits: [u32; 2],
    selectors: [u8; 16],
    temp: [u8; 16],
}

impl CellResults {
    fn new() -> Self {
        Self { best_overall_err: u64::MAX, low_endpoint: [0; 4], high_endpoint: [0; 4], pbits: [0; 2], selectors: [0; 16], temp: [0; 16] }
    }
}

#[derive(Clone)]
struct OptResults {
    mode: u32,
    partition: u32,
    selectors: [u8; 16],
    alpha_selectors: [u8; 16],
    low: [Color; 3],
    high: [Color; 3],
    pbits: [[u32; 2]; 3],
    rotation: u32,
    index_selector: u32,
}

impl OptResults {
    fn new() -> Self {
        Self {
            mode: 0,
            partition: 0,
            selectors: [0; 16],
            alpha_selectors: [0; 16],
            low: [[0; 4]; 3],
            high: [[0; 4]; 3],
            pbits: [[0; 2]; 3],
            rotation: 0,
            index_selector: 0,
        }
    }
}

// Least squares using normal equations:
// http://www.cs.cornell.edu/~bindel/class/cs3220-s12/notes/lec10.pdf
fn compute_least_squares_endpoints_rgba(pixels: &[Color], selectors: &[u8; 16], weightsx: &[Vec4], xl: &mut Vec4, xh: &mut Vec4) {
    let (mut z00, mut z10, mut z11) = (0.0f32, 0.0f32, 0.0f32);
    let (mut q00, mut t) = ([0.0f32; 4], [0.0f32; 4]);
    for (i, p) in pixels.iter().enumerate() {
        let w4 = weightsx[selectors[i] as usize];
        z00 += w4[0];
        z10 += w4[1];
        z11 += w4[2];
        let w = w4[3];
        for c in 0..4 {
            q00[c] += w * f32::from(p[c]);
            t[c] += f32::from(p[c]);
        }
    }
    let z01 = z10;
    let mut det = z00 * z11 - z01 * z10;
    if det != 0.0 {
        det = 1.0 / det;
    }
    let (iz00, iz01, iz10, iz11) = (z11 * det, -z01 * det, -z10 * det, z00 * det);
    for c in 0..4 {
        xl[c] = iz00 * q00[c] + iz01 * (t[c] - q00[c]);
        xh[c] = iz10 * q00[c] + iz11 * (t[c] - q00[c]);
    }
    for c in 0..4 {
        if xl[c] < 0.0 || xh[c] > 255.0 {
            let mut lo_v = u32::MAX;
            let mut hi_v = 0;
            for p in pixels {
                lo_v = lo_v.min(u32::from(p[c]));
                hi_v = hi_v.max(u32::from(p[c]));
            }
            if lo_v == hi_v {
                xl[c] = lo_v as f32;
                xh[c] = hi_v as f32;
            }
        }
    }
}

fn compute_least_squares_endpoints_rgb(pixels: &[Color], selectors: &[u8; 16], weightsx: &[Vec4], xl: &mut Vec4, xh: &mut Vec4) {
    let (mut z00, mut z10, mut z11) = (0.0f32, 0.0f32, 0.0f32);
    let (mut q00, mut t) = ([0.0f32; 3], [0.0f32; 3]);
    for (i, p) in pixels.iter().enumerate() {
        let w4 = weightsx[selectors[i] as usize];
        z00 += w4[0];
        z10 += w4[1];
        z11 += w4[2];
        let w = w4[3];
        for c in 0..3 {
            q00[c] += w * f32::from(p[c]);
            t[c] += f32::from(p[c]);
        }
    }
    let z01 = z10;
    let mut det = z00 * z11 - z01 * z10;
    if det != 0.0 {
        det = 1.0 / det;
    }
    let (iz00, iz01, iz10, iz11) = (z11 * det, -z01 * det, -z10 * det, z00 * det);
    for c in 0..3 {
        xl[c] = iz00 * q00[c] + iz01 * (t[c] - q00[c]);
        xh[c] = iz10 * q00[c] + iz11 * (t[c] - q00[c]);
    }
    xl[3] = 255.0;
    xh[3] = 255.0;
    for c in 0..3 {
        if xl[c] < 0.0 || xh[c] > 255.0 {
            let mut lo_v = u32::MAX;
            let mut hi_v = 0;
            for p in pixels {
                lo_v = lo_v.min(u32::from(p[c]));
                hi_v = hi_v.max(u32::from(p[c]));
            }
            if lo_v == hi_v {
                xl[c] = lo_v as f32;
                xh[c] = hi_v as f32;
            }
        }
    }
}

fn compute_least_squares_endpoints_a(pixels: &[Color], selectors: &[u8; 16], weightsx: &[Vec4]) -> (f32, f32) {
    let (mut z00, mut z10, mut z11) = (0.0f32, 0.0f32, 0.0f32);
    let (mut q00, mut t) = (0.0f32, 0.0f32);
    for (i, p) in pixels.iter().enumerate() {
        let w4 = weightsx[selectors[i] as usize];
        z00 += w4[0];
        z10 += w4[1];
        z11 += w4[2];
        let w = w4[3];
        q00 += w * f32::from(p[3]);
        t += f32::from(p[3]);
    }
    let z01 = z10;
    let mut det = z00 * z11 - z01 * z10;
    if det != 0.0 {
        det = 1.0 / det;
    }
    let (iz00, iz01, iz10, iz11) = (z11 * det, -z01 * det, -z10 * det, z00 * det);
    let q10 = t - q00;
    let (mut xl, mut xh) = (iz00 * q00 + iz01 * q10, iz10 * q00 + iz11 * q10);
    if xl < 0.0 || xh > 255.0 {
        let mut lo_v = u32::MAX;
        let mut hi_v = 0;
        for p in pixels {
            lo_v = lo_v.min(u32::from(p[3]));
            hi_v = hi_v.max(u32::from(p[3]));
        }
        if lo_v == hi_v {
            xl = lo_v as f32;
            xh = hi_v as f32;
        }
    }
    (xl, xh)
}

fn scale_color(c: &Color, params: &CellParams) -> Color {
    let n = params.comp_bits + u32::from(params.has_pbits);
    debug_assert!((4..=8).contains(&n));
    let mut r = [0u8; 4];
    for i in 0..4 {
        let mut v = u32::from(c[i]) << (8 - n);
        v |= v >> n;
        debug_assert!(v <= 255);
        r[i] = v as u8;
    }
    r
}

fn compute_color_distance_rgb(e1: &Color, e2: &Color, perceptual: bool, weights: &[u32; 4]) -> u64 {
    let (dr, dg, db) = if perceptual {
        let l1 = i32::from(e1[0]) * 109 + i32::from(e1[1]) * 366 + i32::from(e1[2]) * 37;
        let cr1 = (i32::from(e1[0]) << 9) - l1;
        let cb1 = (i32::from(e1[2]) << 9) - l1;
        let l2 = i32::from(e2[0]) * 109 + i32::from(e2[1]) * 366 + i32::from(e2[2]) * 37;
        let cr2 = (i32::from(e2[0]) << 9) - l2;
        let cb2 = (i32::from(e2[2]) << 9) - l2;
        ((l1 - l2) >> 8, (cr1 - cr2) >> 8, (cb1 - cb2) >> 8)
    } else {
        (i32::from(e1[0]) - i32::from(e2[0]), i32::from(e1[1]) - i32::from(e2[1]), i32::from(e1[2]) - i32::from(e2[2]))
    };
    u64::from(weights[0]) * (dr * dr) as u64 + u64::from(weights[1]) * (dg * dg) as u64 + u64::from(weights[2]) * (db * db) as u64
}

fn compute_color_distance_rgba(e1: &Color, e2: &Color, perceptual: bool, weights: &[u32; 4]) -> u64 {
    let da = i32::from(e1[3]) - i32::from(e2[3]);
    compute_color_distance_rgb(e1, e2, perceptual, weights) + u64::from(weights[3]) * (da * da) as u64
}

fn pack_mode1_to_one_color(pixels: &[Color], params: &CellParams, results: &mut CellResults, r: u32, g: u32, b: u32, selectors: &mut [u8; 16]) -> u64 {
    let t = tables();
    let (mut best_err, mut best_p) = (u32::MAX, 0usize);
    for p in 0..2 {
        let err =
            u32::from(t.mode1_optimal[r as usize][p].error) + u32::from(t.mode1_optimal[g as usize][p].error) + u32::from(t.mode1_optimal[b as usize][p].error);
        if err < best_err {
            best_err = err;
            best_p = p;
            if best_err == 0 {
                break;
            }
        }
    }
    let (er, eg, eb) = (t.mode1_optimal[r as usize][best_p], t.mode1_optimal[g as usize][best_p], t.mode1_optimal[b as usize][best_p]);
    results.low_endpoint = [er.lo, eg.lo, eb.lo, 0];
    results.high_endpoint = [er.hi, eg.hi, eb.hi, 0];
    results.pbits = [best_p as u32, 0];
    for s in selectors.iter_mut().take(pixels.len()) {
        *s = MODE_1_OPTIMAL_INDEX as u8;
    }
    let mut p = [0u8; 4];
    for (pc, (&lo_e, &hi_e)) in p.iter_mut().take(3).zip(results.low_endpoint.iter().zip(results.high_endpoint.iter())) {
        let mut low = ((u32::from(lo_e) << 1) | results.pbits[0]) << 1;
        low |= low >> 7;
        let mut high = ((u32::from(hi_e) << 1) | results.pbits[0]) << 1;
        high |= high >> 7;
        *pc = ((low * (64 - WEIGHTS3[MODE_1_OPTIMAL_INDEX]) + high * WEIGHTS3[MODE_1_OPTIMAL_INDEX] + 32) >> 6) as u8;
    }
    p[3] = 255;
    let total_err = pixels.iter().map(|px| compute_color_distance_rgb(&p, px, params.perceptual, &params.weights)).sum();
    results.best_overall_err = total_err;
    total_err
}

fn pack_mode7_to_one_color(pixels: &[Color], params: &CellParams, results: &mut CellResults, color: [u32; 4], selectors: &mut [u8; 16]) -> u64 {
    let t = tables();
    let (mut best_err, mut best_p) = (u32::MAX, 0usize);
    for p in 0..4 {
        let (hi_p, lo_p) = (p >> 1, p & 1);
        let err = color.iter().map(|&c| u32::from(t.mode7_optimal[c as usize][hi_p][lo_p].error)).sum();
        if err < best_err {
            best_err = err;
            best_p = p;
            if best_err == 0 {
                break;
            }
        }
    }
    let (hi_p, lo_p) = (best_p >> 1, best_p & 1);
    let (er, eg, eb, ea) = (
        t.mode7_optimal[color[0] as usize][hi_p][lo_p],
        t.mode7_optimal[color[1] as usize][hi_p][lo_p],
        t.mode7_optimal[color[2] as usize][hi_p][lo_p],
        t.mode7_optimal[color[3] as usize][hi_p][lo_p],
    );
    results.low_endpoint = [er.lo, eg.lo, eb.lo, ea.lo];
    results.high_endpoint = [er.hi, eg.hi, eb.hi, ea.hi];
    results.pbits = [lo_p as u32, hi_p as u32];
    for s in selectors.iter_mut().take(pixels.len()) {
        *s = MODE_7_OPTIMAL_INDEX as u8;
    }
    let mut p = [0u8; 4];
    for (pc, (&lo_e, &hi_e)) in p.iter_mut().zip(results.low_endpoint.iter().zip(results.high_endpoint.iter())) {
        let mut low = (u32::from(lo_e) << 1) | results.pbits[0];
        let mut high = (u32::from(hi_e) << 1) | results.pbits[1];
        low = (low << 2) | (low >> 6);
        high = (high << 2) | (high >> 6);
        *pc = ((low * (64 - WEIGHTS2[MODE_7_OPTIMAL_INDEX]) + high * WEIGHTS2[MODE_7_OPTIMAL_INDEX] + 32) >> 6) as u8;
    }
    let total_err = pixels.iter().map(|px| compute_color_distance_rgba(&p, px, params.perceptual, &params.weights)).sum();
    results.best_overall_err = total_err;
    total_err
}

fn evaluate_solution(low: &Color, high: &Color, pbits: [u32; 2], pixels: &[Color], params: &CellParams, results: &mut CellResults, comp: &CompParams) -> u64 {
    let mut quant_min = *low;
    let mut quant_max = *high;
    if params.has_pbits {
        let (min_p, max_p) = if params.endpoints_share_pbit { (pbits[0], pbits[0]) } else { (pbits[0], pbits[1]) };
        for c in 0..4 {
            quant_min[c] = (quant_min[c] << 1) | min_p as u8;
            quant_max[c] = (quant_max[c] << 1) | max_p as u8;
        }
    }
    let actual_min = scale_color(&quant_min, params);
    let actual_max = scale_color(&quant_max, params);

    let n = params.num_selector_weights as usize;
    let mut weighted = [[0u8; 4]; 16];
    weighted[0] = actual_min;
    weighted[n - 1] = actual_max;
    let nc = if params.has_alpha { 4 } else { 3 };
    for (wi, &w) in weighted.iter_mut().zip(params.selector_weights.iter()).take(n - 1).skip(1) {
        for (c, wc) in wi.iter_mut().enumerate().take(nc) {
            *wc = ((u32::from(actual_min[c]) * (64 - w) + u32::from(actual_max[c]) * w + 32) >> 6) as u8;
        }
    }

    let lr = i32::from(actual_min[0]);
    let lg = i32::from(actual_min[1]);
    let lb = i32::from(actual_min[2]);
    let dr = i32::from(actual_max[0]) - lr;
    let dg = i32::from(actual_max[1]) - lg;
    let db = i32::from(actual_max[2]) - lb;

    let mut total_err = 0u64;
    if comp.force_selectors {
        for (i, p) in pixels.iter().enumerate() {
            let sel = comp.selectors[i] as usize;
            let err = if params.has_alpha {
                compute_color_distance_rgba(&weighted[sel], p, params.perceptual, &params.weights)
            } else {
                compute_color_distance_rgb(&weighted[sel], p, params.perceptual, &params.weights)
            };
            total_err += err;
            results.temp[i] = sel as u8;
        }
    } else if !params.perceptual {
        if params.has_alpha {
            let la = i32::from(actual_min[3]);
            let da = i32::from(actual_max[3]) - la;
            let f = n as f32 / ((dr * dr + dg * dg + db * db + da * da) as f32 + 0.00000125);
            for (i, p) in pixels.iter().enumerate() {
                let dot = (i32::from(p[0]) - lr) * dr + (i32::from(p[1]) - lg) * dg + (i32::from(p[2]) - lb) * db + (i32::from(p[3]) - la) * da;
                let mut sel = clampi((dot as f32 * f + 0.5) as i32, 1, n as i32 - 1) as usize;
                let err0 = compute_color_distance_rgba(&weighted[sel - 1], p, false, &params.weights);
                let mut err1 = compute_color_distance_rgba(&weighted[sel], p, false, &params.weights);
                if err1 > err0 {
                    err1 = err0;
                    sel -= 1;
                }
                total_err += err1;
                results.temp[i] = sel as u8;
            }
        } else {
            let f = n as f32 / ((dr * dr + dg * dg + db * db) as f32 + 0.00000125);
            for (i, p) in pixels.iter().enumerate() {
                let dot = (i32::from(p[0]) - lr) * dr + (i32::from(p[1]) - lg) * dg + (i32::from(p[2]) - lb) * db;
                let sel = clampi((dot as f32 * f + 0.5) as i32, 1, n as i32 - 1) as usize;
                let err0 = compute_color_distance_rgb(&weighted[sel - 1], p, false, &params.weights);
                let err1 = compute_color_distance_rgb(&weighted[sel], p, false, &params.weights);
                let (best_err, best_sel) = if err0 < err1 { (err0, sel - 1) } else { (err1, sel) };
                total_err += best_err;
                results.temp[i] = best_sel as u8;
            }
        }
    } else {
        for (i, p) in pixels.iter().enumerate() {
            let (mut best_err, mut best_sel) = (u64::MAX, 0usize);
            for (j, w) in weighted.iter().enumerate().take(n) {
                let err = if params.has_alpha {
                    compute_color_distance_rgba(w, p, true, &params.weights)
                } else {
                    compute_color_distance_rgb(w, p, true, &params.weights)
                };
                if err < best_err {
                    best_err = err;
                    best_sel = j;
                }
            }
            total_err += best_err;
            results.temp[i] = best_sel as u8;
        }
    }

    if total_err < results.best_overall_err {
        results.best_overall_err = total_err;
        results.low_endpoint = *low;
        results.high_endpoint = *high;
        results.pbits = pbits;
        results.selectors[..pixels.len()].copy_from_slice(&results.temp[..pixels.len()]);
    }
    total_err
}

fn fix_degenerate_endpoints(mode: u32, trial_min: &mut Color, trial_max: &mut Color, xl: &Vec4, xh: &Vec4, iscale: i32, comp: &CompParams) {
    if mode != 1 && !(mode == 6 && comp.quant_mode6_endpoints) {
        return;
    }
    for i in 0..3 {
        if trial_min[i] == trial_max[i] && (xl[i] - xh[i]).abs() > 0.0 {
            if i32::from(trial_min[i]) > (iscale >> 1) {
                if trial_min[i] > 0 {
                    trial_min[i] -= 1;
                } else if i32::from(trial_max[i]) < iscale {
                    trial_max[i] += 1;
                }
            } else if i32::from(trial_max[i]) < iscale {
                trial_max[i] += 1;
            } else if trial_min[i] > 0 {
                trial_min[i] -= 1;
            }
        }
    }
}

fn find_optimal_solution(mode: u32, mut xl: Vec4, mut xh: Vec4, pixels: &[Color], params: &CellParams, results: &mut CellResults, comp: &CompParams) -> u64 {
    let t = tables();
    xl = vec4_saturate(xl);
    xh = vec4_saturate(xh);

    if params.has_pbits {
        let iscalep = (1i32 << (params.comp_bits + 1)) - 1;
        let scalep = iscalep as f32;
        let total_comps = if params.has_alpha { 4 } else { 3 };
        let mut best_pbits = [0u32; 2];
        let mut best_min = [0u8; 4];
        let mut best_max = [0u8; 4];

        if !params.endpoints_share_pbit {
            if params.comp_bits == 7 && comp.quant_mode6_endpoints {
                best_pbits[0] = 0;
                for c in 0..4 {
                    best_min[c] = t.mode6_reduced_quant[(xl[c] * 2047.0 + 0.5) as usize][0];
                }
                best_pbits[1] = 1;
                for c in 0..4 {
                    best_max[c] = t.mode6_reduced_quant[(xh[c] * 2047.0 + 0.5) as usize][1];
                }
            } else {
                let (mut best_err0, mut best_err1) = (1e9f32, 1e9f32);
                for p in 0..2u32 {
                    let (mut x_min, mut x_max) = ([0u8; 4], [0u8; 4]);
                    // The pbit selects which quantization interval a component falls in:
                    // pbit 0 quantizes even levels of 2^(comp_bits+1), pbit 1 odd ones.
                    if params.comp_bits == 5 {
                        for c in 0..4 {
                            let mut vl = (xl[c] * 31.0) as i32;
                            vl += i32::from(xl[c] > t.mode7_midpoints[vl as usize][p as usize]);
                            x_min[c] = clampi(vl * 2 + p as i32, p as i32, 62 + p as i32) as u8;
                            let mut vh = (xh[c] * 31.0) as i32;
                            vh += i32::from(xh[c] > t.mode7_midpoints[vh as usize][p as usize]);
                            x_max[c] = clampi(vh * 2 + p as i32, p as i32, 62 + p as i32) as u8;
                        }
                    } else {
                        for c in 0..4 {
                            x_min[c] = clampi(((xl[c] * scalep - p as f32) / 2.0 + 0.5) as i32 * 2 + p as i32, p as i32, iscalep - 1 + p as i32) as u8;
                            x_max[c] = clampi(((xh[c] * scalep - p as f32) / 2.0 + 0.5) as i32 * 2 + p as i32, p as i32, iscalep - 1 + p as i32) as u8;
                        }
                    }
                    let scaled_low = scale_color(&x_min, params);
                    let scaled_high = scale_color(&x_max, params);
                    let (mut err0, mut err1) = (0.0f32, 0.0f32);
                    for i in 0..total_comps {
                        err0 += (f32::from(scaled_low[i]) - xl[i] * 255.0).powi(2);
                        err1 += (f32::from(scaled_high[i]) - xh[i] * 255.0).powi(2);
                    }
                    if p == 1 {
                        err0 *= comp.pbit1_weight;
                        err1 *= comp.pbit1_weight;
                    }
                    if err0 < best_err0 {
                        best_err0 = err0;
                        best_pbits[0] = p;
                        for c in 0..4 {
                            best_min[c] = x_min[c] >> 1;
                        }
                    }
                    if err1 < best_err1 {
                        best_err1 = err1;
                        best_pbits[1] = p;
                        for c in 0..4 {
                            best_max[c] = x_max[c] >> 1;
                        }
                    }
                }
            }
        } else if mode == 1 && comp.bias_mode1_pbits {
            let mut x = 0.0f32;
            for c in 0..3 {
                x = x.max(xl[c]).max(xh[c]);
            }
            let p = u32::from(x > 253.0 / 255.0);
            let (mut x_min, mut x_max) = ([0u8; 4], [0u8; 4]);
            for c in 0..4 {
                let mut vl = (xl[c] * 63.0) as i32;
                vl += i32::from(xl[c] > t.mode1_midpoints[vl as usize][p as usize]);
                x_min[c] = clampi(vl * 2 + p as i32, p as i32, 126 + p as i32) as u8;
                let mut vh = (xh[c] * 63.0) as i32;
                vh += i32::from(xh[c] > t.mode1_midpoints[vh as usize][p as usize]);
                x_max[c] = clampi(vh * 2 + p as i32, p as i32, 126 + p as i32) as u8;
            }
            best_pbits = [p, p];
            for c in 0..4 {
                best_min[c] = x_min[c] >> 1;
                best_max[c] = x_max[c] >> 1;
            }
        } else {
            // Endpoints share pbits.
            let mut best_err = 1e9f32;
            for p in 0..2u32 {
                let (mut x_min, mut x_max) = ([0u8; 4], [0u8; 4]);
                if params.comp_bits == 6 {
                    for c in 0..4 {
                        let mut vl = (xl[c] * 63.0) as i32;
                        vl += i32::from(xl[c] > t.mode1_midpoints[vl as usize][p as usize]);
                        x_min[c] = clampi(vl * 2 + p as i32, p as i32, 126 + p as i32) as u8;
                        let mut vh = (xh[c] * 63.0) as i32;
                        vh += i32::from(xh[c] > t.mode1_midpoints[vh as usize][p as usize]);
                        x_max[c] = clampi(vh * 2 + p as i32, p as i32, 126 + p as i32) as u8;
                    }
                } else {
                    for c in 0..4 {
                        x_min[c] = clampi(((xl[c] * scalep - p as f32) / 2.0 + 0.5) as i32 * 2 + p as i32, p as i32, iscalep - 1 + p as i32) as u8;
                        x_max[c] = clampi(((xh[c] * scalep - p as f32) / 2.0 + 0.5) as i32 * 2 + p as i32, p as i32, iscalep - 1 + p as i32) as u8;
                    }
                }
                let scaled_low = scale_color(&x_min, params);
                let scaled_high = scale_color(&x_max, params);
                let mut err = 0.0f32;
                for i in 0..total_comps {
                    err += (f32::from(scaled_low[i]) / 255.0 - xl[i]).powi(2) + (f32::from(scaled_high[i]) / 255.0 - xh[i]).powi(2);
                }
                if p == 1 {
                    err *= comp.pbit1_weight;
                }
                if err < best_err {
                    best_err = err;
                    best_pbits = [p, p];
                    for c in 0..4 {
                        best_min[c] = x_min[c] >> 1;
                        best_max[c] = x_max[c] >> 1;
                    }
                }
            }
        }

        fix_degenerate_endpoints(mode, &mut best_min, &mut best_max, &xl, &xh, iscalep >> 1, comp);

        if results.best_overall_err == u64::MAX || best_min != results.low_endpoint || best_max != results.high_endpoint || best_pbits != results.pbits {
            evaluate_solution(&best_min, &best_max, best_pbits, pixels, params, results, comp);
        }
    } else {
        let iscale = (1i32 << params.comp_bits) - 1;
        let scale = iscale as f32;
        let (mut trial_min, mut trial_max) = ([0u8; 4], [0u8; 4]);
        if params.comp_bits == 7 {
            for c in 0..4 {
                let mut vl = (xl[c] * 127.0) as i32;
                vl += i32::from(xl[c] > t.mode5_midpoints[vl as usize]);
                trial_min[c] = clampi(vl, 0, 127) as u8;
                let mut vh = (xh[c] * 127.0) as i32;
                vh += i32::from(xh[c] > t.mode5_midpoints[vh as usize]);
                trial_max[c] = clampi(vh, 0, 127) as u8;
            }
        } else {
            for c in 0..4 {
                trial_min[c] = clampi((xl[c] * scale + 0.5) as i32, 0, 255) as u8;
                trial_max[c] = clampi((xh[c] * scale + 0.5) as i32, 0, 255) as u8;
            }
        }

        fix_degenerate_endpoints(mode, &mut trial_min, &mut trial_max, &xl, &xh, iscale, comp);

        if results.best_overall_err == u64::MAX || trial_min != results.low_endpoint || trial_max != results.high_endpoint {
            evaluate_solution(&trial_min, &trial_max, results.pbits, pixels, params, results, comp);
        }
    }
    results.best_overall_err
}

fn color_cell_compression(mode: u32, pixels: &[Color], params: &CellParams, results: &mut CellResults, comp: &CompParams) -> u64 {
    debug_assert!(mode == 6 || mode == 7 || !params.has_alpha);
    results.best_overall_err = u64::MAX;

    // A partition whose colours are all the same packs to a single colour.
    if mode == 1 {
        let cr = pixels[0][0];
        let cg = pixels[0][1];
        let cb = pixels[0][2];
        let all_same = pixels.iter().all(|p| p[0] == cr && p[1] == cg && p[2] == cb);
        if all_same {
            let mut selectors = results.selectors;
            let e = pack_mode1_to_one_color(pixels, params, results, u32::from(cr), u32::from(cg), u32::from(cb), &mut selectors);
            results.selectors = selectors;
            return e;
        }
    } else if mode == 7 {
        let c0 = pixels[0];
        let all_same = pixels.iter().all(|p| *p == c0);
        if all_same {
            let mut selectors = results.selectors;
            let e = pack_mode7_to_one_color(pixels, params, results, c0.map(u32::from), &mut selectors);
            results.selectors = selectors;
            return e;
        }
    }

    // The partition's mean colour and principal axis.
    let mut mean = [0.0f32; 4];
    for p in pixels {
        let c = vec4_from_color(p);
        for i in 0..4 {
            mean[i] += c[i];
        }
    }
    let n = pixels.len() as f32;
    let mean_scaled: Vec4 = [mean[0] / n, mean[1] / n, mean[2] / n, mean[3] / n];
    let mean: Vec4 = vec4_saturate([mean[0] / (n * 255.0), mean[1] / (n * 255.0), mean[2] / (n * 255.0), mean[3] / (n * 255.0)]);

    let mut axis: Vec4;
    if params.has_alpha {
        // Incremental PCA for RGBA.
        axis = [0.0; 4];
        for (i, p) in pixels.iter().enumerate() {
            let mut color = vec4_from_color(p);
            for c in 0..4 {
                color[c] -= mean_scaled[c];
            }
            let a: Vec4 = [color[0] * color[0], color[1] * color[0], color[2] * color[0], color[3] * color[0]];
            let b: Vec4 = [color[0] * color[1], color[1] * color[1], color[2] * color[1], color[3] * color[1]];
            let c2: Vec4 = [color[0] * color[2], color[1] * color[2], color[2] * color[2], color[3] * color[2]];
            let d: Vec4 = [color[0] * color[3], color[1] * color[3], color[2] * color[3], color[3] * color[3]];
            let mut nv = if i != 0 { axis } else { color };
            vec4_normalize(&mut nv);
            axis[0] += vec4_dot(&a, &nv);
            axis[1] += vec4_dot(&b, &nv);
            axis[2] += vec4_dot(&c2, &nv);
            axis[3] += vec4_dot(&d, &nv);
        }
        vec4_normalize(&mut axis);
    } else {
        // Covariance for RGB: no per-pixel normalization.
        let mut cov = [0.0f32; 6];
        let (mut vfr, mut vfg, mut vfb) = (0.9f32, 1.0f32, 0.7f32);
        for p in pixels {
            let r = f32::from(p[0]) - mean_scaled[0];
            let g = f32::from(p[1]) - mean_scaled[1];
            let b = f32::from(p[2]) - mean_scaled[2];
            cov[0] += r * r;
            cov[1] += r * g;
            cov[2] += r * b;
            cov[3] += g * g;
            cov[4] += g * b;
            cov[5] += b * b;
        }
        for _ in 0..3 {
            let r = vfr * cov[0] + vfg * cov[1] + vfb * cov[2];
            let g = vfr * cov[1] + vfg * cov[3] + vfb * cov[4];
            let b = vfr * cov[2] + vfg * cov[4] + vfb * cov[5];
            let m = r.abs().max(g.abs()).max(b.abs());
            if m > 1e-10 {
                let m = 1.0 / m;
                vfr = r * m;
                vfg = g * m;
                vfb = b * m;
            } else {
                vfr = r;
                vfg = g;
                vfb = b;
            }
        }
        let len = vfr * vfr + vfg * vfg + vfb * vfb;
        if len < 1e-10 {
            axis = [0.0; 4];
        } else {
            let len = 1.0 / len.sqrt();
            axis = [vfr * len, vfg * len, vfb * len, 0.0];
        }
    }

    if vec4_dot(&axis, &axis) < 0.5 {
        axis = if params.perceptual {
            [0.213, 0.715, 0.072, if params.has_alpha { 0.715 } else { 0.0 }]
        } else {
            [1.0, 1.0, 1.0, if params.has_alpha { 1.0 } else { 0.0 }]
        };
        vec4_normalize(&mut axis);
    }

    let (mut l, mut h) = (1e9f32, -1e9f32);
    for p in pixels {
        let mut q = vec4_from_color(p);
        for c in 0..4 {
            q[c] -= mean_scaled[c];
        }
        let d = vec4_dot(&q, &axis);
        l = l.min(d);
        h = h.max(d);
    }
    l *= 1.0 / 255.0;
    h *= 1.0 / 255.0;

    let b0: Vec4 = [axis[0] * l, axis[1] * l, axis[2] * l, axis[3] * l];
    let b1: Vec4 = [axis[0] * h, axis[1] * h, axis[2] * h, axis[3] * h];
    let c0: Vec4 = [mean[0] + b0[0], mean[1] + b0[1], mean[2] + b0[2], mean[3] + b0[3]];
    let c1: Vec4 = [mean[0] + b1[0], mean[1] + b1[1], mean[2] + b1[2], mean[3] + b1[3]];
    let mut min_color = vec4_saturate(c0);
    let mut max_color = vec4_saturate(c1);

    if min_color.iter().sum::<f32>() > max_color.iter().sum::<f32>() {
        std::mem::swap(&mut min_color, &mut max_color);
    }

    if find_optimal_solution(mode, min_color, max_color, pixels, params, results, comp) == 0 {
        return 0;
    }

    if comp.try_least_squares {
        let (mut xl, mut xh) = ([0.0f32; 4], [0.0f32; 4]);
        if params.has_alpha {
            compute_least_squares_endpoints_rgba(pixels, &results.selectors, params.selector_weightsx, &mut xl, &mut xh);
        } else {
            compute_least_squares_endpoints_rgb(pixels, &results.selectors, params.selector_weightsx, &mut xl, &mut xh);
        }
        for c in 0..4 {
            xl[c] *= 1.0 / 255.0;
            xh[c] *= 1.0 / 255.0;
        }
        if find_optimal_solution(mode, xl, xh, pixels, params, results, comp) == 0 {
            return 0;
        }
    }

    if comp.uber_level > 0 {
        let mut selectors_temp = [0u8; 16];
        let mut selectors_temp1 = [0u8; 16];
        selectors_temp[..pixels.len()].copy_from_slice(&results.selectors[..pixels.len()]);
        let max_selector = params.num_selector_weights - 1;
        let mut min_sel = 16u32;
        let mut max_sel = 0u32;
        for &s in selectors_temp.iter().take(pixels.len()) {
            min_sel = min_sel.min(u32::from(s));
            max_sel = max_sel.max(u32::from(s));
        }
        for i in 0..pixels.len() {
            let mut sel = selectors_temp[i];
            if u32::from(sel) == min_sel && u32::from(sel) < params.num_selector_weights - 1 {
                sel += 1;
            }
            selectors_temp1[i] = sel;
        }
        let (mut xl, mut xh) = ([0.0f32; 4], [0.0f32; 4]);
        if params.has_alpha {
            compute_least_squares_endpoints_rgba(pixels, &selectors_temp1, params.selector_weightsx, &mut xl, &mut xh);
        } else {
            compute_least_squares_endpoints_rgb(pixels, &selectors_temp1, params.selector_weightsx, &mut xl, &mut xh);
        }
        for c in 0..4 {
            xl[c] *= 1.0 / 255.0;
            xh[c] *= 1.0 / 255.0;
        }
        if find_optimal_solution(mode, xl, xh, pixels, params, results, comp) == 0 {
            return 0;
        }

        for i in 0..pixels.len() {
            let mut sel = selectors_temp[i];
            if u32::from(sel) == max_sel && sel > 0 {
                sel -= 1;
            }
            selectors_temp1[i] = sel;
        }
        if params.has_alpha {
            compute_least_squares_endpoints_rgba(pixels, &selectors_temp1, params.selector_weightsx, &mut xl, &mut xh);
        } else {
            compute_least_squares_endpoints_rgb(pixels, &selectors_temp1, params.selector_weightsx, &mut xl, &mut xh);
        }
        for c in 0..4 {
            xl[c] *= 1.0 / 255.0;
            xh[c] *= 1.0 / 255.0;
        }
        if find_optimal_solution(mode, xl, xh, pixels, params, results, comp) == 0 {
            return 0;
        }

        for i in 0..pixels.len() {
            let mut sel = selectors_temp[i];
            if u32::from(sel) == min_sel && u32::from(sel) < params.num_selector_weights - 1 {
                sel += 1;
            } else if u32::from(sel) == max_sel && sel > 0 {
                sel -= 1;
            }
            selectors_temp1[i] = sel;
        }
        if params.has_alpha {
            compute_least_squares_endpoints_rgba(pixels, &selectors_temp1, params.selector_weightsx, &mut xl, &mut xh);
        } else {
            compute_least_squares_endpoints_rgb(pixels, &selectors_temp1, params.selector_weightsx, &mut xl, &mut xh);
        }
        for c in 0..4 {
            xl[c] *= 1.0 / 255.0;
            xh[c] *= 1.0 / 255.0;
        }
        if find_optimal_solution(mode, xl, xh, pixels, params, results, comp) == 0 {
            return 0;
        }

        // Uber levels 2+ try more endpoint extrapolation by scaling the selectors.
        let uber_err_thresh = (pixels.len() as u32 * 56) >> 4;
        if comp.uber_level >= 2 && results.best_overall_err > u64::from(uber_err_thresh) {
            let q = if comp.uber_level >= 4 { comp.uber_level - 2 } else { 1 };
            let mut ly = -q;
            while ly <= 1 {
                let mut hy = max_selector as i32 - 1;
                while hy <= max_selector as i32 + q {
                    if ly != 0 || hy != max_selector as i32 {
                        for i in 0..pixels.len() {
                            let v = (max_selector as f32 * (f32::from(selectors_temp[i]) - ly as f32) / (hy - ly) as f32 + 0.5)
                                .floor()
                                .clamp(0.0, max_selector as f32);
                            selectors_temp1[i] = v as u8;
                        }
                        let (mut xl, mut xh) = ([0.0f32; 4], [0.0f32; 4]);
                        if params.has_alpha {
                            compute_least_squares_endpoints_rgba(pixels, &selectors_temp1, params.selector_weightsx, &mut xl, &mut xh);
                        } else {
                            compute_least_squares_endpoints_rgb(pixels, &selectors_temp1, params.selector_weightsx, &mut xl, &mut xh);
                        }
                        for c in 0..4 {
                            xl[c] *= 1.0 / 255.0;
                            xh[c] *= 1.0 / 255.0;
                        }
                        if find_optimal_solution(mode, xl, xh, pixels, params, results, comp) == 0 {
                            return 0;
                        }
                    }
                    hy += 1;
                }
                ly += 1;
            }
        }
    }

    if mode == 1 {
        let mut avg_results = results.clone();
        let r = (0.5 + mean[0] * 255.0) as u32;
        let g = (0.5 + mean[1] * 255.0) as u32;
        let b = (0.5 + mean[2] * 255.0) as u32;
        let mut temp = results.temp;
        let avg_err = pack_mode1_to_one_color(pixels, params, &mut avg_results, r, g, b, &mut temp);
        if avg_err < results.best_overall_err {
            *results = avg_results;
            results.selectors[..pixels.len()].copy_from_slice(&temp[..pixels.len()]);
            results.best_overall_err = avg_err;
        }
    } else if mode == 7 {
        let mut avg_results = results.clone();
        let color = mean.map(|v| (0.5 + v * 255.0) as u32);
        let mut temp = results.temp;
        let avg_err = pack_mode7_to_one_color(pixels, params, &mut avg_results, color, &mut temp);
        if avg_err < results.best_overall_err {
            *results = avg_results;
            results.selectors[..pixels.len()].copy_from_slice(&temp[..pixels.len()]);
            results.best_overall_err = avg_err;
        }
    }
    results.best_overall_err
}

fn color_cell_compression_est_mode1(pixels: &[Color], perceptual: bool, pweights: &[u32; 4], best_err_so_far: u64) -> u64 {
    // RGB bounds as an approximation of the block's principal axis.
    let (mut lo, mut hi) = ([255u32; 4], [0u32; 4]);
    for p in pixels {
        for c in 0..3 {
            lo[c] = lo[c].min(u32::from(p[c]));
            hi[c] = hi[c].max(u32::from(p[c]));
        }
    }
    let low_color: Color = [lo[0] as u8, lo[1] as u8, lo[2] as u8, 0];
    let high_color: Color = [hi[0] as u8, hi[1] as u8, hi[2] as u8, 0];

    let n = 8usize;
    let mut weighted = [[0u8; 4]; 8];
    weighted[0] = low_color;
    weighted[n - 1] = high_color;
    for i in 1..n - 1 {
        let w = WEIGHTS3[i];
        for c in 0..3 {
            weighted[i][c] = ((u32::from(low_color[c]) * (64 - w) + u32::from(high_color[c]) * w + 32) >> 6) as u8;
        }
    }

    let ar = i32::from(high_color[0]) - i32::from(low_color[0]);
    let ag = i32::from(high_color[1]) - i32::from(low_color[1]);
    let ab = i32::from(high_color[2]) - i32::from(low_color[2]);
    let dots: [i32; 8] = std::array::from_fn(|i| i32::from(weighted[i][0]) * ar + i32::from(weighted[i][1]) * ag + i32::from(weighted[i][2]) * ab);
    let thresh: [i32; 7] = std::array::from_fn(|i| (dots[i] + dots[i + 1] + 1) >> 1);

    let mut total_err = 0u64;
    if perceptual {
        let (mut l1, mut cr1, mut cb1) = ([0i32; 8], [0i32; 8], [0i32; 8]);
        for (j, e) in weighted.iter().enumerate() {
            l1[j] = i32::from(e[0]) * 109 + i32::from(e[1]) * 366 + i32::from(e[2]) * 37;
            cr1[j] = (i32::from(e[0]) << 9) - l1[j];
            cb1[j] = (i32::from(e[2]) << 9) - l1[j];
        }
        for p in pixels {
            let d = ar * i32::from(p[0]) + ag * i32::from(p[1]) + ab * i32::from(p[2]);
            let mut s = 0usize;
            for (i, &t) in thresh.iter().enumerate().rev() {
                if d >= t {
                    s = i + 1;
                    break;
                }
            }
            let l2 = i32::from(p[0]) * 109 + i32::from(p[1]) * 366 + i32::from(p[2]) * 37;
            let cr2 = (i32::from(p[0]) << 9) - l2;
            let cb2 = (i32::from(p[2]) << 9) - l2;
            let dl = (l1[s] - l2) >> 8;
            let dcr = (cr1[s] - cr2) >> 8;
            let dcb = (cb1[s] - cb2) >> 8;
            total_err += u64::from(pweights[0]) * (dl * dl) as u64 + u64::from(pweights[1]) * (dcr * dcr) as u64 + u64::from(pweights[2]) * (dcb * dcb) as u64;
            if total_err > best_err_so_far {
                break;
            }
        }
    } else {
        for p in pixels {
            let d = ar * i32::from(p[0]) + ag * i32::from(p[1]) + ab * i32::from(p[2]);
            let mut s = 0usize;
            for (i, &t) in thresh.iter().enumerate().rev() {
                if d >= t {
                    s = i + 1;
                    break;
                }
            }
            let e = &weighted[s];
            let dr = i32::from(e[0]) - i32::from(p[0]);
            let dg = i32::from(e[1]) - i32::from(p[1]);
            let db = i32::from(e[2]) - i32::from(p[2]);
            total_err += u64::from(pweights[0]) * (dr * dr) as u64 + u64::from(pweights[1]) * (dg * dg) as u64 + u64::from(pweights[2]) * (db * db) as u64;
            if total_err > best_err_so_far {
                break;
            }
        }
    }
    total_err
}

fn color_cell_compression_est_mode7(pixels: &[Color], perceptual: bool, pweights: &[u32; 4], best_err_so_far: u64) -> u64 {
    let (mut lo, mut hi) = ([255u32; 4], [0u32; 4]);
    for p in pixels {
        for c in 0..4 {
            lo[c] = lo[c].min(u32::from(p[c]));
            hi[c] = hi[c].max(u32::from(p[c]));
        }
    }
    let low_color: Color = [lo[0] as u8, lo[1] as u8, lo[2] as u8, lo[3] as u8];
    let high_color: Color = [hi[0] as u8, hi[1] as u8, hi[2] as u8, hi[3] as u8];

    let n = 4usize;
    let mut weighted = [[0u8; 4]; 4];
    weighted[0] = low_color;
    weighted[n - 1] = high_color;
    for i in 1..n - 1 {
        let w = WEIGHTS2[i];
        for c in 0..4 {
            weighted[i][c] = ((u32::from(low_color[c]) * (64 - w) + u32::from(high_color[c]) * w + 32) >> 6) as u8;
        }
    }

    let a: [i32; 4] = [
        i32::from(high_color[0]) - i32::from(low_color[0]),
        i32::from(high_color[1]) - i32::from(low_color[1]),
        i32::from(high_color[2]) - i32::from(low_color[2]),
        i32::from(high_color[3]) - i32::from(low_color[3]),
    ];
    let dots: [i32; 4] = std::array::from_fn(|i| {
        i32::from(weighted[i][0]) * a[0] + i32::from(weighted[i][1]) * a[1] + i32::from(weighted[i][2]) * a[2] + i32::from(weighted[i][3]) * a[3]
    });
    let thresh: [i32; 3] = std::array::from_fn(|i| (dots[i] + dots[i + 1] + 1) >> 1);

    let mut total_err = 0u64;
    if perceptual {
        let (mut l1, mut cr1, mut cb1) = ([0i32; 4], [0i32; 4], [0i32; 4]);
        for (j, e) in weighted.iter().enumerate() {
            l1[j] = i32::from(e[0]) * 109 + i32::from(e[1]) * 366 + i32::from(e[2]) * 37;
            cr1[j] = (i32::from(e[0]) << 9) - l1[j];
            cb1[j] = (i32::from(e[2]) << 9) - l1[j];
        }
        for p in pixels {
            let d = a[0] * i32::from(p[0]) + a[1] * i32::from(p[1]) + a[2] * i32::from(p[2]) + a[3] * i32::from(p[3]);
            let mut s = 0usize;
            for (i, &t) in thresh.iter().enumerate().rev() {
                if d >= t {
                    s = i + 1;
                    break;
                }
            }
            let l2 = i32::from(p[0]) * 109 + i32::from(p[1]) * 366 + i32::from(p[2]) * 37;
            let cr2 = (i32::from(p[0]) << 9) - l2;
            let cb2 = (i32::from(p[2]) << 9) - l2;
            let dl = (l1[s] - l2) >> 8;
            let dcr = (cr1[s] - cr2) >> 8;
            let dcb = (cb1[s] - cb2) >> 8;
            let dca = i32::from(p[3]) - i32::from(weighted[s][3]);
            total_err += u64::from(pweights[0]) * (dl * dl) as u64
                + u64::from(pweights[1]) * (dcr * dcr) as u64
                + u64::from(pweights[2]) * (dcb * dcb) as u64
                + u64::from(pweights[3]) * (dca * dca) as u64;
            if total_err > best_err_so_far {
                break;
            }
        }
    } else {
        for p in pixels {
            let d = a[0] * i32::from(p[0]) + a[1] * i32::from(p[1]) + a[2] * i32::from(p[2]) + a[3] * i32::from(p[3]);
            let mut s = 0usize;
            for (i, &t) in thresh.iter().enumerate().rev() {
                if d >= t {
                    s = i + 1;
                    break;
                }
            }
            let e = &weighted[s];
            let dr = i32::from(e[0]) - i32::from(p[0]);
            let dg = i32::from(e[1]) - i32::from(p[1]);
            let db = i32::from(e[2]) - i32::from(p[2]);
            let da = i32::from(e[3]) - i32::from(p[3]);
            total_err += u64::from(pweights[0]) * (dr * dr) as u64
                + u64::from(pweights[1]) * (dg * dg) as u64
                + u64::from(pweights[2]) * (db * db) as u64
                + u64::from(pweights[3]) * (da * da) as u64;
            if total_err > best_err_so_far {
                break;
            }
        }
    }
    total_err
}

// Bitmasks of the "key" partitions that must rank best before a partition is worth
// evaluating: the first 14 (sorted by usefulness) set the key, the rest follow it.
const PARTITION_PREDICTORS: [u32; 35] = [
    u32::MAX,
    u32::MAX,
    u32::MAX,
    u32::MAX,
    u32::MAX,
    (1 << 1) | (1 << 2) | (1 << 8),
    (1 << 1) | (1 << 3) | (1 << 7),
    u32::MAX,
    u32::MAX,
    (1 << 2) | (1 << 8) | (1 << 16),
    (1 << 7) | (1 << 3) | (1 << 15),
    u32::MAX,
    (1 << 8) | (1 << 14) | (1 << 16),
    (1 << 7) | (1 << 14) | (1 << 15),
    u32::MAX,
    u32::MAX,
    u32::MAX,
    u32::MAX,
    (1 << 14) | (1 << 15),
    (1 << 16) | (1 << 22) | (1 << 14),
    (1 << 17) | (1 << 24) | (1 << 14),
    (1 << 2) | (1 << 14) | (1 << 15) | (1 << 1),
    u32::MAX,
    (1 << 1) | (1 << 3) | (1 << 14) | (1 << 16) | (1 << 22),
    u32::MAX,
    (1 << 1) | (1 << 2) | (1 << 15) | (1 << 17) | (1 << 24),
    (1 << 1) | (1 << 3) | (1 << 22),
    u32::MAX,
    u32::MAX,
    u32::MAX,
    (1 << 14) | (1 << 15) | (1 << 16) | (1 << 17),
    u32::MAX,
    u32::MAX,
    (1 << 1) | (1 << 2) | (1 << 3) | (1 << 27) | (1 << 4) | (1 << 24),
    (1 << 14) | (1 << 15) | (1 << 16) | (1 << 11) | (1 << 17) | (1 << 27),
];

// Estimate the partition used by modes 1/7 by scanning each and approximating its error.
fn estimate_partition(pixels: &[Color], comp: &CompParams, pweights: &[u32; 4], mode: u32) -> u32 {
    const SORTED_PARTITION_ORDER: [u8; 64] = [
        0, 13, 1, 2, 15, 14, 10, 16, 3, 23, 26, 6, 7, 21, 19, 29, 8, 4, 9, 20, 5, 31, 22, 17, 18, 11, 12, 30, 24, 25, 28, 27, 32, 33, 34, 45, 46, 51, 49, 50,
        48, 38, 39, 37, 53, 52, 54, 36, 57, 58, 55, 41, 40, 42, 43, 59, 44, 56, 47, 35, 60, 63, 62, 61,
    ];
    let total_partitions = comp.max_partitions.min(64);
    if total_partitions <= 1 {
        return 0;
    }
    debug_assert_eq!(SORTED_PARTITION_ORDER[34], 34);

    let (mut best_err, mut best_partition, mut best_key_partition) = (u64::MAX, 0u32, 0usize);
    for (partition_iter, &order) in SORTED_PARTITION_ORDER.iter().enumerate().take(total_partitions as usize) {
        if best_err == 0 {
            break;
        }
        let partition = order as usize;
        let table = &P2[partition * 16..partition * 16 + 16];

        if comp.mode17_partition_estimation_filterbank && (14..=34).contains(&partition_iter) {
            let key_bitmask = 1u32 << (best_key_partition + 1);
            if PARTITION_PREDICTORS[partition] & key_bitmask == 0 {
                if partition_iter == 34 {
                    break;
                }
                continue;
            }
        }

        let mut subset_colors = [[[0u8; 4]; 16]; 2];
        let mut counts = [0usize; 2];
        for (i, &s) in table.iter().enumerate() {
            subset_colors[s as usize][counts[s as usize]] = pixels[i];
            counts[s as usize] += 1;
        }

        let mut total_subset_err = 0u64;
        for subset in 0..2 {
            if total_subset_err >= best_err {
                break;
            }
            let subset_pixels = &subset_colors[subset][..counts[subset]];
            total_subset_err += if mode == 7 {
                color_cell_compression_est_mode7(subset_pixels, comp.perceptual, pweights, best_err)
            } else {
                color_cell_compression_est_mode1(subset_pixels, comp.perceptual, pweights, best_err)
            };
        }

        if partition < 16 {
            total_subset_err = (total_subset_err as f64 * comp.low_frequency_partition_weight as f64 + 0.5) as u64;
        }
        if total_subset_err < best_err {
            best_err = total_subset_err;
            best_partition = partition as u32;
        }

        // The checkerboard pattern (34) gates the rest statistically.
        if partition == 34 && best_partition != 34 {
            break;
        }
        if partition_iter == 13 {
            best_key_partition = best_partition as usize;
        }
    }
    best_partition
}

fn set_block_bits(bytes: &mut [u8; 16], mut val: u32, mut num_bits: u32, ofs: &mut u32) {
    debug_assert!(num_bits <= 32 && (num_bits == 64 || u64::from(val) < (1u64 << num_bits)));
    while num_bits > 0 {
        let n = (8 - (*ofs & 7)).min(num_bits);
        bytes[(*ofs >> 3) as usize] |= (val << (*ofs & 7)) as u8;
        val >>= n;
        num_bits -= n;
        *ofs += n;
    }
    debug_assert!(*ofs <= 128);
}

fn encode_bc7_block(results: &OptResults) -> [u8; 16] {
    let mode = results.mode as usize;
    let total_subsets = N_SUBSETS[mode] as usize;
    let total_partitions = 1u32 << PARTITION_BITS[mode];

    let partition: &[u8] = match total_subsets {
        1 => &PARTITION1,
        2 => &P2[results.partition as usize * 16..results.partition as usize * 16 + 16],
        _ => &P3[results.partition as usize * 16..results.partition as usize * 16 + 16],
    };
    debug_assert!(results.index_selector <= 1 && results.rotation <= 3);

    let mut color_selectors = results.selectors;
    let mut alpha_selectors = results.alpha_selectors;
    let mut low = results.low;
    let mut high = results.high;
    let mut pbits = results.pbits;
    let mut anchor = [u8::MAX; 3];

    for k in 0..total_subsets {
        let anchor_index = if k == 0 {
            0
        } else if total_subsets == 3 && k == 1 {
            ANCHOR3_SECOND[results.partition as usize]
        } else if total_subsets == 3 && k == 2 {
            ANCHOR3_THIRD[results.partition as usize]
        } else {
            ANCHOR2[results.partition as usize]
        };
        anchor[k] = anchor_index;

        let color_index_bits = color_index_size(mode, results.index_selector);
        let num_color_indices = 1u8 << color_index_bits;
        if color_selectors[anchor_index as usize] & (num_color_indices >> 1) != 0 {
            for (i, s) in color_selectors.iter_mut().enumerate() {
                if partition[i] == k as u8 {
                    *s = (num_color_indices - 1) - *s;
                }
            }
            if mode_has_separate_alpha_selectors(mode) {
                for q in 0..3 {
                    std::mem::swap(&mut low[k][q], &mut high[k][q]);
                }
            } else {
                std::mem::swap(&mut low[k], &mut high[k]);
            }
            if MODE_HAS_SHARED_P_BITS[mode] == 0 {
                pbits[k].swap(0, 1);
            }
        }

        if mode_has_separate_alpha_selectors(mode) {
            let alpha_index_bits = alpha_index_size(mode, results.index_selector);
            let num_alpha_indices = 1u8 << alpha_index_bits;
            if alpha_selectors[anchor_index as usize] & (num_alpha_indices >> 1) != 0 {
                for (i, s) in alpha_selectors.iter_mut().enumerate() {
                    if partition[i] == k as u8 {
                        *s = (num_alpha_indices - 1) - *s;
                    }
                }
                std::mem::swap(&mut low[k][3], &mut high[k][3]);
            }
        }
    }

    let mut bytes = [0u8; 16];
    let mut ofs = 0u32;
    set_block_bits(&mut bytes, 1 << mode, mode as u32 + 1, &mut ofs);
    if mode == 4 || mode == 5 {
        set_block_bits(&mut bytes, results.rotation, 2, &mut ofs);
    }
    if mode == 4 {
        set_block_bits(&mut bytes, results.index_selector, 1, &mut ofs);
    }
    if total_partitions > 1 {
        set_block_bits(&mut bytes, results.partition, if total_partitions == 64 { 6 } else { 4 }, &mut ofs);
    }

    let total_comps = if mode >= 4 { 4 } else { 3 };
    for comp in 0..total_comps {
        for subset in 0..total_subsets {
            let bits = u32::from(if comp == 3 { ALPHA_PRECISION[mode] } else { COLOR_PRECISION[mode] });
            set_block_bits(&mut bytes, u32::from(low[subset][comp]), bits, &mut ofs);
            set_block_bits(&mut bytes, u32::from(high[subset][comp]), bits, &mut ofs);
        }
    }

    if MODE_HAS_P_BITS[mode] != 0 {
        for pb in pbits.iter().take(total_subsets) {
            set_block_bits(&mut bytes, pb[0], 1, &mut ofs);
            if MODE_HAS_SHARED_P_BITS[mode] == 0 {
                set_block_bits(&mut bytes, pb[1], 1, &mut ofs);
            }
        }
    }

    for idx in 0..16usize {
        let mut n = if results.index_selector != 0 { alpha_index_size(mode, results.index_selector) } else { color_index_size(mode, results.index_selector) };
        if anchor.iter().any(|&a| a as usize == idx) {
            n -= 1;
        }
        let v = if results.index_selector != 0 { alpha_selectors[idx] } else { color_selectors[idx] };
        set_block_bits(&mut bytes, u32::from(v), n, &mut ofs);
    }

    if mode_has_separate_alpha_selectors(mode) {
        for idx in 0..16usize {
            let mut n =
                if results.index_selector != 0 { color_index_size(mode, results.index_selector) } else { alpha_index_size(mode, results.index_selector) };
            if anchor.iter().any(|&a| a as usize == idx) {
                n -= 1;
            }
            let v = if results.index_selector != 0 { color_selectors[idx] } else { alpha_selectors[idx] };
            set_block_bits(&mut bytes, u32::from(v), n, &mut ofs);
        }
    }
    debug_assert_eq!(ofs, 128);
    bytes
}

fn handle_alpha_block_mode5(
    pixels: &[Color],
    comp: &CompParams,
    params: &mut CellParams,
    mut lo_a: u32,
    mut hi_a: u32,
    opt5: &mut OptResults,
    mode5_err: &mut u64,
) {
    params.selector_weights = &WEIGHTS2;
    params.selector_weightsx = &WEIGHTS2X;
    params.num_selector_weights = 4;
    params.comp_bits = 7;
    params.has_pbits = false;
    params.endpoints_share_pbit = false;
    params.has_alpha = false;
    params.perceptual = comp.perceptual;

    let mut results5 = CellResults::new();
    *mode5_err = color_cell_compression(5, pixels, params, &mut results5, comp);
    opt5.selectors = results5.selectors;
    opt5.low[0] = results5.low_endpoint;
    opt5.high[0] = results5.high_endpoint;

    if lo_a == hi_a {
        opt5.low[0][3] = lo_a as u8;
        opt5.high[0][3] = hi_a as u8;
        opt5.alpha_selectors = [0; 16];
        return;
    }

    let mut mode5_alpha_err = u64::MAX;
    let total_passes = if comp.uber_level >= 1 { 3 } else { 2 };
    for pass in 0..total_passes {
        let mut vals = [0i32; 4];
        vals[0] = lo_a as i32;
        vals[3] = hi_a as i32;
        vals[1] = (vals[0] * (64 - 21) + vals[3] * 21 + 32) >> 6;
        vals[2] = (vals[0] * (64 - 43) + vals[3] * 43 + 32) >> 6;

        let mut trial_selectors = [0u8; 16];
        let mut trial_err = 0u64;
        for (i, s) in trial_selectors.iter_mut().enumerate() {
            let a = i32::from(pixels[i][3]);
            let (mut be, mut bs) = ((a - vals[0]).abs(), 0usize);
            for (j, &vj) in vals.iter().enumerate().skip(1) {
                let e = (a - vj).abs();
                if e < be {
                    be = e;
                    bs = j;
                }
            }
            *s = bs as u8;
            trial_err += (be * be) as u64 * u64::from(params.weights[3]);
        }

        if trial_err < mode5_alpha_err {
            mode5_alpha_err = trial_err;
            opt5.low[0][3] = lo_a as u8;
            opt5.high[0][3] = hi_a as u8;
            opt5.alpha_selectors = trial_selectors;
        }

        if pass != total_passes - 1 {
            let (xl, xh) = compute_least_squares_endpoints_a(pixels, &trial_selectors, &WEIGHTS2X);
            let mut new_lo = clampi((xl + 0.5).floor() as i32, 0, 255) as u32;
            let mut new_hi = clampi((xh + 0.5).floor() as i32, 0, 255) as u32;
            if new_lo > new_hi {
                std::mem::swap(&mut new_lo, &mut new_hi);
            }
            if new_lo == lo_a && new_hi == hi_a {
                break;
            }
            lo_a = new_lo;
            hi_a = new_hi;
        }
    }
    *mode5_err += mode5_alpha_err;
}

fn handle_alpha_block(pixels: &[Color], comp: &CompParams, params: &mut CellParams) -> [u8; 16] {
    debug_assert!(comp.mode_mask & (1 << 6 | 1 << 5 | 1 << 7) != 0);

    params.selector_weights = &WEIGHTS4;
    params.selector_weightsx = &WEIGHTS4X;
    params.num_selector_weights = 16;
    params.comp_bits = 7;
    params.has_pbits = true;
    params.endpoints_share_pbit = false;
    params.has_alpha = true;
    params.perceptual = comp.perceptual;

    let mut best_err = u64::MAX;
    let mut best_mode = 0u32;
    let mut opt6 = OptResults::new();
    let mut opt5 = OptResults::new();
    let mut opt7 = OptResults::new();

    if comp.mode_mask & (1 << 6) != 0 {
        let mut results6 = CellResults::new();
        best_err = (color_cell_compression(6, pixels, params, &mut results6, comp) as f64 * comp.mode6_error_weight as f64 + 0.5) as u64;
        best_mode = 6;
        opt6.mode = 6;
        opt6.selectors = results6.selectors;
        opt6.low[0] = results6.low_endpoint;
        opt6.high[0] = results6.high_endpoint;
        opt6.pbits[0] = results6.pbits;
    }

    if best_err > 0 && comp.mode_mask & (1 << 5) != 0 {
        let (mut lo_a, mut hi_a) = (255u32, 0u32);
        for p in pixels {
            lo_a = lo_a.min(u32::from(p[3]));
            hi_a = hi_a.max(u32::from(p[3]));
        }
        let mut mode5_err = 0u64;
        handle_alpha_block_mode5(pixels, comp, params, lo_a, hi_a, &mut opt5, &mut mode5_err);
        let mode5_err = (mode5_err as f64 * comp.mode5_error_weight as f64 + 0.5) as u64;
        if mode5_err < best_err {
            best_err = mode5_err;
            best_mode = 5;
        }
    }

    if best_err > 0 && comp.mode_mask & (1 << 7) != 0 {
        let trial_partition = estimate_partition(pixels, comp, &params.weights, 7);
        let table = &P2[trial_partition as usize * 16..trial_partition as usize * 16 + 16];
        let mut subset_colors = [[[0u8; 4]; 16]; 2];
        let mut subset_counts = [0u32; 2];
        let mut subset_pixel_index = [[0u8; 16]; 2];
        let mut subset_results = [CellResults::new(), CellResults::new()];

        params.selector_weights = &WEIGHTS2;
        params.selector_weightsx = &WEIGHTS2X;
        params.num_selector_weights = 4;
        params.comp_bits = 5;
        params.has_pbits = true;
        params.endpoints_share_pbit = false;
        params.has_alpha = true;

        for (idx, &s) in table.iter().enumerate() {
            let p = s as usize;
            subset_colors[p][subset_counts[p] as usize] = pixels[idx];
            subset_pixel_index[p][subset_counts[p] as usize] = idx as u8;
            subset_counts[p] += 1;
        }

        let mut trial_err = 0u64;
        for subset in 0..2 {
            let err = color_cell_compression(7, &subset_colors[subset][..subset_counts[subset] as usize], params, &mut subset_results[subset], comp);
            trial_err += err;
            if (trial_err as f64 * comp.mode7_error_weight as f64 + 0.5) as u64 > best_err {
                break;
            }
        }

        let mode7_trial_err = (trial_err as f64 * comp.mode7_error_weight as f64 + 0.5) as u64;
        if mode7_trial_err < best_err {
            best_mode = 7;
            opt7.mode = 7;
            opt7.partition = trial_partition;
            opt7.index_selector = 0;
            opt7.rotation = 0;
            for subset in 0..2 {
                for (&pix_i, &sel) in subset_pixel_index[subset].iter().zip(subset_results[subset].selectors.iter()).take(subset_counts[subset] as usize) {
                    opt7.selectors[pix_i as usize] = sel;
                }
                opt7.low[subset] = subset_results[subset].low_endpoint;
                opt7.high[subset] = subset_results[subset].high_endpoint;
                opt7.pbits[subset] = subset_results[subset].pbits;
            }
        }
    }

    match best_mode {
        7 => encode_bc7_block(&opt7),
        5 => {
            opt5.mode = 5;
            opt5.partition = 0;
            opt5.rotation = 0;
            opt5.index_selector = 0;
            encode_bc7_block(&opt5)
        }
        _ => encode_bc7_block(&opt6),
    }
}

fn handle_opaque_block(pixels: &[Color], comp: &CompParams, params: &mut CellParams) -> [u8; 16] {
    debug_assert!(comp.mode_mask & (1 << 6 | 1 << 1) != 0);

    params.perceptual = comp.perceptual;
    params.has_alpha = false;

    let mut opt = OptResults::new();
    let mut best_err = u64::MAX;

    if comp.mode_mask & (1 << 6) != 0 {
        params.selector_weights = &WEIGHTS4;
        params.selector_weightsx = &WEIGHTS4X;
        params.num_selector_weights = 16;
        params.comp_bits = 7;
        params.has_pbits = true;
        params.endpoints_share_pbit = false;

        let mut results6 = CellResults::new();
        best_err = (color_cell_compression(6, pixels, params, &mut results6, comp) as f64 * comp.mode6_error_weight as f64 + 0.5) as u64;
        opt.mode = 6;
        opt.selectors = results6.selectors;
        opt.low[0] = results6.low_endpoint;
        opt.high[0] = results6.high_endpoint;
        opt.pbits[0] = results6.pbits;
    }

    if best_err > 0 && comp.max_partitions > 0 && comp.mode_mask & (1 << 1) != 0 {
        let trial_partition = estimate_partition(pixels, comp, &params.weights, 1);
        let table = &P2[trial_partition as usize * 16..trial_partition as usize * 16 + 16];
        let mut subset_colors = [[[0u8; 4]; 16]; 2];
        let mut subset_counts = [0u32; 2];
        let mut subset_pixel_index = [[0u8; 16]; 2];
        let mut subset_results = [CellResults::new(), CellResults::new()];

        params.selector_weights = &WEIGHTS3;
        params.selector_weightsx = &WEIGHTS3X;
        params.num_selector_weights = 8;
        params.comp_bits = 6;
        params.has_pbits = true;
        params.endpoints_share_pbit = true;

        for (idx, &s) in table.iter().enumerate() {
            let p = s as usize;
            subset_colors[p][subset_counts[p] as usize] = pixels[idx];
            subset_pixel_index[p][subset_counts[p] as usize] = idx as u8;
            subset_counts[p] += 1;
        }

        let mut trial_err = 0u64;
        for subset in 0..2 {
            let err = color_cell_compression(1, &subset_colors[subset][..subset_counts[subset] as usize], params, &mut subset_results[subset], comp);
            trial_err += err;
            if (trial_err as f64 * comp.mode1_error_weight as f64 + 0.5) as u64 > best_err {
                break;
            }
        }

        let mode1_trial_err = (trial_err as f64 * comp.mode1_error_weight as f64 + 0.5) as u64;
        if mode1_trial_err < best_err {
            opt.mode = 1;
            opt.partition = trial_partition;
            for subset in 0..2 {
                for (&pix_i, &sel) in subset_pixel_index[subset].iter().zip(subset_results[subset].selectors.iter()).take(subset_counts[subset] as usize) {
                    opt.selectors[pix_i as usize] = sel;
                }
                opt.low[subset] = subset_results[subset].low_endpoint;
                opt.high[subset] = subset_results[subset].high_endpoint;
                opt.pbits[subset][0] = subset_results[subset].pbits[0];
            }
        }
    }

    encode_bc7_block(&opt)
}

/// One 4×4 RGBA block to a 16-byte BC7 block: modes 1/6 for opaque blocks, 5/6/7 for
/// blocks with alpha. `quality` is the encoder's search effort: deeper `uber` passes and
/// refinements trade speed for quality (the bc7enc effort levels).
pub(crate) fn encode_bc7(block: &Block, quality: DdsQuality) -> [u8; 16] {
    let comp = match quality {
        DdsQuality::Fastest => CompParams { uber_level: 0, try_least_squares: false, mode17_partition_estimation_filterbank: true, ..Default::default() },
        DdsQuality::Normal => CompParams { uber_level: 2, mode17_partition_estimation_filterbank: true, ..Default::default() },
        DdsQuality::Production => CompParams { uber_level: 3, ..Default::default() },
        DdsQuality::Highest => CompParams::default(),
    };
    let mut params = CellParams {
        num_selector_weights: 0,
        selector_weights: &[],
        selector_weightsx: &[],
        comp_bits: 0,
        weights: comp.weights,
        has_alpha: false,
        has_pbits: false,
        endpoints_share_pbit: false,
        perceptual: comp.perceptual,
    };
    if comp.perceptual {
        // ITU-R BT.709 luma/chroma weighting.
        let pr = (0.5 / (1.0 - 0.2126)) * (0.5 / (1.0 - 0.2126));
        let pb = (0.5 / (1.0 - 0.0722)) * (0.5 / (1.0 - 0.0722));
        params.weights = [
            (comp.weights[0] as f32 * 4.0) as u32,
            (comp.weights[1] as f32 * 4.0 * pr) as u32,
            (comp.weights[2] as f32 * 4.0 * pb) as u32,
            comp.weights[3] * 4,
        ];
    }

    let pixels: &[Color] = block;
    if comp.force_alpha || pixels.iter().any(|p| p[3] < 255) {
        return handle_alpha_block(pixels, &comp, &mut params);
    }
    handle_opaque_block(pixels, &comp, &mut params)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codecs::dds_bc7::decode_bc7;

    include!("dds_bc7_golden.inc");

    /// The Rust port must produce bit-identical blocks to the C bc7enc it follows, and
    /// the decoder must reproduce the C decoder's output exactly.
    #[test]
    fn matches_the_reference_encoder_and_decoder_byte_for_byte() {
        for (i, (src, want_block, want_rgba)) in VECTORS.iter().enumerate() {
            let mut block: Block = [[0; 4]; 16];
            block.copy_from_slice(src.as_chunks::<4>().0);
            // The golden vectors were captured at the finest settings.
            let encoded = encode_bc7(&block, DdsQuality::Highest);
            assert_eq!(&encoded, want_block, "vector {i}: encoder output differs");
            let decoded = decode_bc7(want_block);
            let decoded_flat: [u8; 64] = std::array::from_fn(|i| decoded[i / 4][i % 4]);
            assert_eq!(&decoded_flat[..], &want_rgba[..], "vector {i}: decoder output differs");
        }
    }
}
