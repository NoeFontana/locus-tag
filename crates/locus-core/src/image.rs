//! Stride-aware image view for zero-copy ingestion.
#![allow(clippy::inline_always)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::missing_panics_doc)]
#![allow(unsafe_code)]

use std::sync::LazyLock;

use rayon::prelude::*;

use crate::config::RescueInterpolation;

/// A view into an image buffer with explicit stride support.
/// This allows handling NumPy arrays with padding or non-standard layouts.
#[derive(Copy, Clone)]
pub struct ImageView<'a> {
    /// The raw image data slice.
    pub data: &'a [u8],
    /// The width of the image in pixels.
    pub width: usize,
    /// The height of the image in pixels.
    pub height: usize,
    /// The stride (bytes per row) of the image.
    pub stride: usize,
}

impl<'a> ImageView<'a> {
    /// Create a new ImageView after validating that the buffer size matches the dimensions and stride.
    pub fn new(data: &'a [u8], width: usize, height: usize, stride: usize) -> Result<Self, String> {
        if stride < width {
            return Err(format!(
                "Stride ({stride}) cannot be less than width ({width})"
            ));
        }
        let required_size = if height > 0 {
            (height - 1) * stride + width
        } else {
            0
        };
        if data.len() < required_size {
            return Err(format!(
                "Buffer size ({}) is too small for {}x{} image with stride {} (required: {})",
                data.len(),
                width,
                height,
                stride,
                required_size
            ));
        }
        Ok(Self {
            data,
            width,
            height,
            stride,
        })
    }

    /// Returns true if the image buffer has sufficient padding for safe SIMD gather operations.
    ///
    /// Some SIMD kernels (e.g. AVX2 gather) may perform 32-bit loads on 8-bit data,
    /// which can read up to 3 bytes past the end of the logical buffer.
    #[must_use]
    pub fn has_simd_padding(&self) -> bool {
        let required_size = if self.height > 0 {
            (self.height - 1) * self.stride + self.width
        } else {
            0
        };
        self.data.len() >= required_size + 3
    }

    /// Safe accessor for a specific row.
    #[inline(always)]
    #[must_use]
    pub fn get_row(&self, y: usize) -> &[u8] {
        assert!(y < self.height, "Row index {y} out of bounds");
        let start = y * self.stride;
        &self.data[start..start + self.width]
    }

    /// Get a pixel value at (x, y) with boundary clamping.
    #[must_use]
    pub fn get_pixel(&self, x: usize, y: usize) -> u8 {
        let x = x.min(self.width - 1);
        let y = y.min(self.height - 1);
        // SAFETY: clamping ensures bounds
        unsafe { *self.data.get_unchecked(y * self.stride + x) }
    }

    /// Get a pixel value at (x, y) without bounds checking.
    ///
    /// # Safety
    /// Caller must ensure `x < width` and `y < height`.
    #[inline(always)]
    #[must_use]
    pub unsafe fn get_pixel_unchecked(&self, x: usize, y: usize) -> u8 {
        // debug_assert ensures we catch violations in debug mode
        debug_assert!(x < self.width, "x {} out of bounds {}", x, self.width);
        debug_assert!(y < self.height, "y {} out of bounds {}", y, self.height);
        // SAFETY: Caller guarantees that (x, y) are within the image dimensions.
        unsafe { *self.data.get_unchecked(y * self.stride + x) }
    }

    /// Sample pixel value with bilinear interpolation at sub-pixel coordinates.
    #[must_use]
    pub fn sample_bilinear(&self, x: f64, y: f64) -> f64 {
        let x = x - 0.5;
        let y = y - 0.5;

        if x < 0.0 || x >= (self.width - 1) as f64 || y < 0.0 || y >= (self.height - 1) as f64 {
            return f64::from(
                self.get_pixel(x.round().max(0.0) as usize, y.round().max(0.0) as usize),
            );
        }

        let x0 = x.floor() as usize;
        let y0 = y.floor() as usize;
        let x1 = x0 + 1;
        let y1 = y0 + 1;

        let dx = x - x0 as f64;
        let dy = y - y0 as f64;

        let v00 = f64::from(self.get_pixel(x0, y0));
        let v10 = f64::from(self.get_pixel(x1, y0));
        let v01 = f64::from(self.get_pixel(x0, y1));
        let v11 = f64::from(self.get_pixel(x1, y1));

        let v0 = v00 * (1.0 - dx) + v10 * dx;
        let v1 = v01 * (1.0 - dx) + v11 * dx;

        v0 * (1.0 - dy) + v1 * dy
    }

    /// Sample pixel value with bilinear interpolation at sub-pixel coordinates without bounds checking.
    ///
    /// # Safety
    /// Caller must ensure `0.0 <= x <= width - 1.001` and `0.0 <= y <= height - 1.001`
    /// such that floor(x), floor(x)+1, floor(y), floor(y)+1 are all valid indices.
    #[inline(always)]
    #[must_use]
    pub unsafe fn sample_bilinear_unchecked(&self, x: f64, y: f64) -> f64 {
        let x = x - 0.5;
        let y = y - 0.5;
        let x0 = x as usize; // Truncate is effectively floor for positive numbers
        let y0 = y as usize;
        let x1 = x0 + 1;
        let y1 = y0 + 1;

        debug_assert!(x1 < self.width, "x1 {} out of bounds {}", x1, self.width);
        debug_assert!(y1 < self.height, "y1 {} out of bounds {}", y1, self.height);

        let dx = x - x0 as f64;
        let dy = y - y0 as f64;

        // Use unchecked pixel access
        // We know strides and offsets are valid because of the assertions (in debug) and caller contract (in release)
        // SAFETY: Caller guarantees that floor(x), floor(x)+1, floor(y), floor(y)+1 are within bounds.
        let row0 = unsafe { self.get_row_unchecked(y0) };
        // SAFETY: Same as above.
        let row1 = unsafe { self.get_row_unchecked(y1) };

        // We can access x0/x1 directly from the row slice
        // SAFETY: x0 and x1 are within bounds guaranteed by the caller.
        unsafe {
            let v00 = f64::from(*row0.get_unchecked(x0));
            let v10 = f64::from(*row0.get_unchecked(x1));
            let v01 = f64::from(*row1.get_unchecked(x0));
            let v11 = f64::from(*row1.get_unchecked(x1));

            let v0 = v00 * (1.0 - dx) + v10 * dx;
            let v1 = v01 * (1.0 - dx) + v11 * dx;

            v0 * (1.0 - dy) + v1 * dy
        }
    }

    /// Compute the gradient [gx, gy] at sub-pixel coordinates using bilinear interpolation.
    #[must_use]
    pub fn sample_gradient_bilinear(&self, x: f64, y: f64) -> [f64; 2] {
        let x = x - 0.5;
        let y = y - 0.5;

        // Optimization: Sample [gx, gy] directly using a 3x3 or 4x4 neighborhood
        // instead of 4 separate bilinear samples.
        // For a high-quality sub-pixel gradient, we sample the 4 nearest integer locations
        // and interpolate their finite-difference gradients.

        if x < 1.0 || x >= (self.width - 2) as f64 || y < 1.0 || y >= (self.height - 2) as f64 {
            let gx = (self.sample_bilinear(x + 1.0, y) - self.sample_bilinear(x - 1.0, y)) * 0.5;
            let gy = (self.sample_bilinear(x, y + 1.0) - self.sample_bilinear(x, y - 1.0)) * 0.5;
            return [gx, gy];
        }

        let x0 = x.floor() as usize;
        let y0 = y.floor() as usize;
        let dx = x - x0 as f64;
        let dy = y - y0 as f64;

        // Fetch 4x4 neighborhood to compute central differences at 4 grid points
        // (x0, y0), (x0+1, y0), (x0, y0+1), (x0+1, y0+1)
        // Indices needed: x0-1..x0+2, y0-1..y0+2
        let mut g00 = [0.0, 0.0];
        let mut g10 = [0.0, 0.0];
        let mut g01 = [0.0, 0.0];
        let mut g11 = [0.0, 0.0];

        // SAFETY: Bounds are checked above (x >= 1.0, y >= 1.0, etc.)
        unsafe {
            for j in 0..2 {
                for i in 0..2 {
                    let cx = x0 + i;
                    let cy = y0 + j;

                    let gx = (f64::from(self.get_pixel_unchecked(cx + 1, cy))
                        - f64::from(self.get_pixel_unchecked(cx - 1, cy)))
                        * 0.5;
                    let gy = (f64::from(self.get_pixel_unchecked(cx, cy + 1))
                        - f64::from(self.get_pixel_unchecked(cx, cy - 1)))
                        * 0.5;

                    match (i, j) {
                        (0, 0) => g00 = [gx, gy],
                        (1, 0) => g10 = [gx, gy],
                        (0, 1) => g01 = [gx, gy],
                        (1, 1) => g11 = [gx, gy],
                        _ => unreachable!(),
                    }
                }
            }
        }

        let gx = (g00[0] * (1.0 - dx) + g10[0] * dx) * (1.0 - dy)
            + (g01[0] * (1.0 - dx) + g11[0] * dx) * dy;
        let gy = (g00[1] * (1.0 - dx) + g10[1] * dx) * (1.0 - dy)
            + (g01[1] * (1.0 - dx) + g11[1] * dx) * dy;

        [gx, gy]
    }

    /// Unsafe accessor for a specific row.
    #[inline(always)]
    pub(crate) unsafe fn get_row_unchecked(&self, y: usize) -> &[u8] {
        let start = y * self.stride;
        // SAFETY: Caller guarantees y < height. Width and stride are invariants.
        unsafe { &self.data.get_unchecked(start..start + self.width) }
    }

    /// Create a decimated copy of the image by subsampling every `factor` pixels.
    ///
    /// The `output` buffer must have size at least `(width/factor) * (height/factor)`.
    pub fn decimate_to<'b>(
        &self,
        factor: usize,
        output: &'b mut [u8],
    ) -> Result<ImageView<'b>, String> {
        let factor = factor.max(1);
        if factor == 1 {
            let len = self.data.len();
            if output.len() < len {
                return Err(format!(
                    "Output buffer too small: {} < {}",
                    output.len(),
                    len
                ));
            }
            output[..len].copy_from_slice(self.data);
            return ImageView::new(&output[..len], self.width, self.height, self.width);
        }

        let new_w = self.width / factor;
        let new_h = self.height / factor;

        if output.len() < new_w * new_h {
            return Err(format!(
                "Output buffer too small for decimation: {} < {}",
                output.len(),
                new_w * new_h
            ));
        }

        output
            .par_chunks_exact_mut(new_w)
            .enumerate()
            .take(new_h)
            .for_each(|(y, out_row)| {
                let src_y = y * factor;
                let src_row = self.get_row(src_y);
                for x in 0..new_w {
                    out_row[x] = src_row[x * factor];
                }
            });

        ImageView::new(&output[..new_w * new_h], new_w, new_h, new_w)
    }

    /// Create an upscaled copy of the image using bilinear interpolation.
    ///
    /// The `output` buffer must have size at least `(width*factor) * (height*factor)`.
    pub fn upscale_to<'b>(
        &self,
        factor: usize,
        output: &'b mut [u8],
    ) -> Result<ImageView<'b>, String> {
        let factor = factor.max(1);
        if factor == 1 {
            let len = self.data.len();
            if output.len() < len {
                return Err(format!(
                    "Output buffer too small: {} < {}",
                    output.len(),
                    len
                ));
            }
            output[..len].copy_from_slice(self.data);
            return ImageView::new(&output[..len], self.width, self.height, self.width);
        }

        let new_w = self.width * factor;
        let new_h = self.height * factor;

        if output.len() < new_w * new_h {
            return Err(format!(
                "Output buffer too small for upscaling: {} < {}",
                output.len(),
                new_w * new_h
            ));
        }

        let scale = 1.0 / factor as f64;

        output
            .par_chunks_exact_mut(new_w)
            .enumerate()
            .take(new_h)
            .for_each(|(y, out_row)| {
                let src_y = y as f64 * scale;
                for (x, val) in out_row.iter_mut().enumerate() {
                    let src_x = x as f64 * scale;
                    // We can use unchecked version for speed if we are confident,
                    // but sample_bilinear handles bounds checks.
                    // Given we are inside image bounds, it should be fine.
                    // To maximize perf we might want a localized optimized loop here,
                    // but for now reusing sample_bilinear is safe and clean.
                    *val = self.sample_bilinear(src_x, src_y) as u8;
                }
            });

        ImageView::new(&output[..new_w * new_h], new_w, new_h, new_w)
    }
}

// ---------------------------------------------------------------------------
// ROI upscaling for super-resolution rescue decode.
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
struct LanczosPhase {
    tap_off: [i32; 6],
    weights: [f32; 6],
}

fn lanczos3(x: f32) -> f32 {
    // Sampling at a tap center is a delta: avoids 0/0 in the sinc product.
    if x == 0.0 {
        return 1.0;
    }
    if x.abs() >= 3.0 {
        return 0.0;
    }
    let pi_x = std::f32::consts::PI * x;
    let pi_x_third = pi_x / 3.0;
    (pi_x.sin() / pi_x) * (pi_x_third.sin() / pi_x_third)
}

fn build_lanczos_phase(phase: usize, factor: usize) -> LanczosPhase {
    // Output pixel at phase `r` with q=0 maps to source position
    // Sx = (r + 0.5)/F - 0.5. `floor(Sx) + [-2..=3]` covers the 6-tap support.
    let sx = (phase as f32 + 0.5) / factor as f32 - 0.5;
    let int_base = sx.floor() as i32;
    let frac = sx - int_base as f32;
    let mut phase_tbl = LanczosPhase {
        tap_off: [0; 6],
        weights: [0.0; 6],
    };
    let mut sum = 0.0f32;
    for (k, off) in (-2..=3i32).enumerate() {
        phase_tbl.tap_off[k] = int_base + off;
        // distance from Sx to the tap at integer (int_base + off): (off - frac)
        let w = lanczos3(off as f32 - frac);
        phase_tbl.weights[k] = w;
        sum += w;
    }
    // Normalize so a constant-valued input reproduces exactly.
    for w in &mut phase_tbl.weights {
        *w /= sum;
    }
    phase_tbl
}

static LANCZOS3_PHASES_2X: LazyLock<[LanczosPhase; 2]> =
    LazyLock::new(|| [build_lanczos_phase(0, 2), build_lanczos_phase(1, 2)]);

static LANCZOS3_PHASES_4X: LazyLock<[LanczosPhase; 4]> = LazyLock::new(|| {
    [
        build_lanczos_phase(0, 4),
        build_lanczos_phase(1, 4),
        build_lanczos_phase(2, 4),
        build_lanczos_phase(3, 4),
    ]
});

/// Upscale a sub-rectangle of `src` into a caller-provided output buffer.
///
/// Used by the ROI super-resolution rescue stage to re-sample a
/// `FailedDecode` candidate at higher effective resolution.
///
/// - `bbox = (x, y, w, h)` is the source-image sub-rectangle.
/// - `factor` must be `2` or `4`.
/// - `out` must hold at least `w*factor * h*factor` bytes; the returned view
///   owns the leading slice.
/// - `scratch` is used as the horizontal-pass buffer when `kernel ==
///   Lanczos3` (size `w*factor * h`). Pass an empty slice for `Bilinear`.
pub fn upscale_roi_to_buf<'b>(
    src: &ImageView<'_>,
    bbox: (usize, usize, usize, usize),
    out: &'b mut [u8],
    factor: u8,
    scratch: &mut [u8],
    kernel: RescueInterpolation,
) -> Result<ImageView<'b>, String> {
    let (bx, by, bw, bh) = bbox;
    if factor != 2 && factor != 4 {
        return Err(format!("upscale factor must be 2 or 4, got {factor}"));
    }
    if bw == 0 || bh == 0 {
        return Err("ROI bbox has zero dimension".into());
    }
    if bx.saturating_add(bw) > src.width || by.saturating_add(bh) > src.height {
        return Err(format!(
            "ROI bbox ({bx},{by},{bw},{bh}) out of bounds for {}x{} source",
            src.width, src.height
        ));
    }
    let f = factor as usize;
    let ow = bw * f;
    let oh = bh * f;
    let required_out = ow * oh;
    if out.len() < required_out {
        return Err(format!(
            "output buffer too small: {} < {}",
            out.len(),
            required_out
        ));
    }
    match kernel {
        RescueInterpolation::Bilinear => {
            upscale_roi_bilinear(src, bbox, &mut out[..required_out], factor);
        }
        RescueInterpolation::Lanczos3 => {
            let required_scratch = ow * bh;
            if scratch.len() < required_scratch {
                return Err(format!(
                    "scratch buffer too small for Lanczos3: {} < {}",
                    scratch.len(),
                    required_scratch
                ));
            }
            upscale_roi_lanczos3(
                src,
                bbox,
                &mut out[..required_out],
                &mut scratch[..required_scratch],
                factor,
            );
        }
    }
    ImageView::new(&out[..required_out], ow, oh, ow)
}

fn upscale_roi_bilinear(
    src: &ImageView<'_>,
    bbox: (usize, usize, usize, usize),
    out: &mut [u8],
    factor: u8,
) {
    let (bx, by, bw, bh) = bbox;
    let f = factor as usize;
    let ow = bw * f;
    let oh = bh * f;
    let inv_f = 1.0 / f64::from(factor);
    let half_src_px = 0.5 * inv_f;
    for oy in 0..oh {
        let sy = by as f64 + (oy as f64) * inv_f + half_src_px;
        let row = &mut out[oy * ow..(oy + 1) * ow];
        for (ox, dst) in row.iter_mut().enumerate() {
            let sx = bx as f64 + (ox as f64) * inv_f + half_src_px;
            let v = src.sample_bilinear(sx, sy);
            *dst = v.clamp(0.0, 255.0).round() as u8;
        }
    }
}

// ROI dims are bounded by `max_roi_side_px: u16` (128 default, 65535 max) and
// the upscale factor is 2 or 4 — i32 holds all representable values.
#[allow(
    clippy::cast_possible_wrap,
    clippy::cast_possible_truncation,
    clippy::needless_range_loop
)]
fn upscale_roi_lanczos3(
    src: &ImageView<'_>,
    bbox: (usize, usize, usize, usize),
    out: &mut [u8],
    scratch: &mut [u8],
    factor: u8,
) {
    let (bx, by, bw, bh) = bbox;
    let f = factor as usize;
    let ow = bw * f;
    let oh = bh * f;
    let phases: &[LanczosPhase] = match factor {
        2 => LANCZOS3_PHASES_2X.as_slice(),
        4 => LANCZOS3_PHASES_4X.as_slice(),
        _ => unreachable!(),
    };

    // Horizontal pass: for each source row in the ROI, produce `ow` filtered
    // columns in `scratch`. Tap indices are clamped to the ROI extent so the
    // caller's bbox padding (if any) is consumed but edge samples stay safe
    // without relying on OOB pixel access.
    let max_col = bw as i32 - 1;
    for sy_local in 0..bh {
        let src_row = src.get_row(by + sy_local);
        let dst_row = &mut scratch[sy_local * ow..(sy_local + 1) * ow];
        for (ox, dst) in dst_row.iter_mut().enumerate() {
            let phase = ox % f;
            let q = (ox / f) as i32;
            let p = &phases[phase];
            let mut acc = 0.0f32;
            for k in 0..6 {
                let idx = (q + p.tap_off[k]).clamp(0, max_col) as usize;
                acc += p.weights[k] * f32::from(src_row[bx + idx]);
            }
            *dst = acc.clamp(0.0, 255.0).round() as u8;
        }
    }

    // Vertical pass: for each output row, gather 6 source rows from scratch
    // and accumulate.
    let max_row = bh as i32 - 1;
    for oy in 0..oh {
        let phase = oy % f;
        let q = (oy / f) as i32;
        let p = &phases[phase];
        let mut rows: [&[u8]; 6] = [&[]; 6];
        for (k, row_slot) in rows.iter_mut().enumerate() {
            let sy = (q + p.tap_off[k]).clamp(0, max_row) as usize;
            *row_slot = &scratch[sy * ow..(sy + 1) * ow];
        }
        let dst_row = &mut out[oy * ow..(oy + 1) * ow];
        for (x, dst) in dst_row.iter_mut().enumerate() {
            let mut acc = 0.0f32;
            for k in 0..6 {
                acc += p.weights[k] * f32::from(rows[k][x]);
            }
            *dst = acc.clamp(0.0, 255.0).round() as u8;
        }
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn test_image_view_stride() {
        let data = vec![
            1, 2, 3, 0, // row 0 + padding
            4, 5, 6, 0, // row 1 + padding
        ];
        let view = ImageView::new(&data, 3, 2, 4).expect("Valid image creation");
        assert_eq!(view.get_row(0), &[1, 2, 3]);
        assert_eq!(view.get_row(1), &[4, 5, 6]);
        assert_eq!(view.get_pixel(1, 1), 5);
    }

    #[test]
    fn test_invalid_buffer_size() {
        let data = vec![1, 2, 3];
        let result = ImageView::new(&data, 2, 2, 2);
        assert!(result.is_err());
    }

    proptest! {
        #[test]
        fn prop_image_view_creation(
            width in 0..1000usize,
            height in 0..1000usize,
            stride_extra in 0..100usize,
            has_enough_data in prop::bool::ANY
        ) {
            let stride = width + stride_extra;
            let required_size = if height > 0 {
                (height - 1) * stride + width
            } else {
                0
            };

            let data_len = if has_enough_data {
                required_size
            } else {
                required_size.saturating_sub(1)
            };

            let data = vec![0u8; data_len];
            let result = ImageView::new(&data, width, height, stride);

            if height > 0 && required_size > 0 && !has_enough_data {
                assert!(result.is_err());
            } else {
                assert!(result.is_ok());
            }
        }

        #[test]
        fn prop_get_pixel_clamping(
            width in 1..100usize,
            height in 1..100usize,
            x in 0..200usize,
            y in 0..200usize
        ) {
            let data = vec![0u8; height * width];
            let view = ImageView::new(&data, width, height, width).expect("valid creation");
            let p = view.get_pixel(x, y);
            // Clamping should prevent panic
            assert_eq!(p, 0);
        }

        #[test]
        fn prop_sample_bilinear_invariants(
            width in 2..20usize,
            height in 2..20usize,
            data in prop::collection::vec(0..=255u8, 20*20),
            x in 0.0..20.0f64,
            y in 0.0..20.0f64
        ) {
            let real_width = width.min(20);
            let real_height = height.min(20);
            let slice = &data[..real_width * real_height];
            let view = ImageView::new(slice, real_width, real_height, real_width).expect("valid creation");

            let x = x % real_width as f64;
            let y = y % real_height as f64;

            let val = view.sample_bilinear(x, y);

            // Result should be within [0, 255]
            assert!((0.0..=255.0).contains(&val));

            // If inside 2x2 neighborhood, val should be within min/max of those 4 pixels
            // The pixels are centered at i + 0.5, so sample (x, y) is between
            // floor(x-0.5) and floor(x-0.5)+1.
            let x0 = (x - 0.5).max(0.0).floor() as usize;
            let y0 = (y - 0.5).max(0.0).floor() as usize;
            let x1 = x0 + 1;
            let y1 = y0 + 1;

            if x1 < real_width && y1 < real_height {
                let v00 = view.get_pixel(x0, y0);
                let v10 = view.get_pixel(x1, y0);
                let v01 = view.get_pixel(x0, y1);
                let v11 = view.get_pixel(x1, y1);

                let min = f64::from(v00.min(v10).min(v01).min(v11));
                let max = f64::from(v00.max(v10).max(v01).max(v11));

                assert!(val >= min - 1e-9 && val <= max + 1e-9, "Value {val} not in [{min}, {max}] for x={x}, y={y}");
            }
        }
    }

    // -----------------------------------------------------------------------
    // ROI upscaling tests.
    // -----------------------------------------------------------------------

    fn make_constant_view(w: usize, h: usize, v: u8) -> (Vec<u8>, usize, usize) {
        (vec![v; w * h], w, h)
    }

    #[test]
    fn test_lanczos_phase_weights_sum_to_unity() {
        // Normalization is the "constant signal reproduces" invariant.
        for p in &*LANCZOS3_PHASES_2X {
            let s: f32 = p.weights.iter().sum();
            assert!((s - 1.0).abs() < 1e-5, "2x phase weights sum {s} != 1");
        }
        for p in &*LANCZOS3_PHASES_4X {
            let s: f32 = p.weights.iter().sum();
            assert!((s - 1.0).abs() < 1e-5, "4x phase weights sum {s} != 1");
        }
    }

    #[test]
    fn test_upscale_roi_constant_preserved_bilinear() {
        let (data, w, h) = make_constant_view(8, 8, 123);
        let view = ImageView::new(&data, w, h, w).expect("view");
        let mut out = vec![0u8; 8 * 8 * 4];
        let mut scratch: [u8; 0] = [];
        let up = upscale_roi_to_buf(
            &view,
            (0, 0, 8, 8),
            &mut out,
            2,
            &mut scratch,
            RescueInterpolation::Bilinear,
        )
        .expect("upscale ok");
        assert_eq!(up.width, 16);
        assert_eq!(up.height, 16);
        assert!(up.data.iter().all(|&p| p == 123), "constant not preserved");
    }

    #[test]
    fn test_upscale_roi_constant_preserved_lanczos() {
        let (data, w, h) = make_constant_view(16, 16, 42);
        let view = ImageView::new(&data, w, h, w).expect("view");
        let (ow, oh) = (16 * 2, 16 * 2);
        let mut out = vec![0u8; ow * oh];
        let mut scratch = vec![0u8; ow * 16];
        let up = upscale_roi_to_buf(
            &view,
            (0, 0, 16, 16),
            &mut out,
            2,
            &mut scratch,
            RescueInterpolation::Lanczos3,
        )
        .expect("upscale ok");
        // A constant signal reproduces because phase weights sum to 1.
        assert!(up.data.iter().all(|&p| p == 42), "Lanczos constant: leaked edge");
    }

    #[test]
    fn test_upscale_roi_bbox_offset_lanczos() {
        // Checker-ish pattern, rescue a sub-window with 4x factor.
        let w = 12;
        let h = 12;
        let mut data = vec![0u8; w * h];
        for y in 0..h {
            for x in 0..w {
                data[y * w + x] = if ((x + y) & 1) == 0 { 200 } else { 40 };
            }
        }
        let view = ImageView::new(&data, w, h, w).expect("view");
        let (bx, by, bw, bh) = (2, 2, 6, 6);
        let f: u8 = 4;
        let ow = bw * f as usize;
        let oh = bh * f as usize;
        let mut out = vec![0u8; ow * oh];
        let mut scratch = vec![0u8; ow * bh];
        let up = upscale_roi_to_buf(
            &view,
            (bx, by, bw, bh),
            &mut out,
            f,
            &mut scratch,
            RescueInterpolation::Lanczos3,
        )
        .expect("lanczos upscale");
        assert_eq!(up.width, ow);
        assert_eq!(up.height, oh);
        // Sub-window average ≈ (200 + 40) / 2 = 120 for a balanced checker.
        let sum: u64 = up.data.iter().map(|&b| u64::from(b)).sum();
        let avg = sum as f64 / (ow * oh) as f64;
        assert!(
            (avg - 120.0).abs() < 10.0,
            "Lanczos avg {avg} far from checker mean 120"
        );
    }

    #[test]
    fn test_upscale_roi_invalid_factor() {
        let (data, w, h) = make_constant_view(4, 4, 0);
        let view = ImageView::new(&data, w, h, w).expect("view");
        let mut out = [0u8; 256];
        let mut scratch: [u8; 0] = [];
        let err = upscale_roi_to_buf(
            &view,
            (0, 0, 4, 4),
            &mut out,
            3,
            &mut scratch,
            RescueInterpolation::Bilinear,
        );
        assert!(err.is_err());
    }

    #[test]
    fn test_upscale_roi_bbox_oob() {
        let (data, w, h) = make_constant_view(4, 4, 0);
        let view = ImageView::new(&data, w, h, w).expect("view");
        let mut out = [0u8; 1024];
        let mut scratch: [u8; 0] = [];
        let err = upscale_roi_to_buf(
            &view,
            (2, 2, 4, 4), // 2 + 4 > 4
            &mut out,
            2,
            &mut scratch,
            RescueInterpolation::Bilinear,
        );
        assert!(err.is_err());
    }
}
