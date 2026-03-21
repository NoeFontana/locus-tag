/// A simple line model for synthetic edge generation.
/// Defined by a*x + b*y + c = 0, where (a,b) is the unit normal.
#[derive(Clone, Copy, Debug)]
pub struct Line {
    /// Normal component X
    pub a: f64,
    /// Normal component Y
    pub b: f64,
    /// Distance constant
    pub c: f64,
}

impl Line {
    /// Create a line from two points, assuming CW winding.
    /// The normal (a, b) will point "outward" (to the right of the direction p1->p2).
    #[must_use]
    pub fn from_points_cw(p1: [f64; 2], p2: [f64; 2]) -> Self {
        let dx = p2[0] - p1[0];
        let dy = p2[1] - p1[1];
        let len = (dx * dx + dy * dy).sqrt();

        // CW Outward normal: (dy/len, -dx/len)
        let nx = dy / len;
        let ny = -dx / len;
        let c = -(nx * p1[0] + ny * p1[1]);

        Self { a: nx, b: ny, c }
    }

    /// Calculate signed distance from a point to the line.
    #[must_use]
    pub fn signed_distance(&self, p: [f64; 2]) -> f64 {
        self.a * p[0] + self.b * p[1] + self.c
    }
}

/// Utility for rendering sub-pixel accurate synthetic edges.
#[derive(Clone, Copy, Debug)]
pub struct SubpixelEdgeRenderer {
    /// Image width
    pub width: usize,
    /// Image height
    pub height: usize,
    /// Dark side intensity
    pub dark_intensity: f64,
    /// Light side intensity
    pub light_intensity: f64,
    /// Blur sigma
    pub sigma: f64,
}

impl SubpixelEdgeRenderer {
    /// Create a new renderer with default parameters.
    #[must_use]
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            dark_intensity: 50.0,
            light_intensity: 200.0,
            sigma: 0.8,
        }
    }

    /// Set intensities A and B.
    #[must_use]
    pub fn with_intensities(mut self, dark: f64, light: f64) -> Self {
        self.dark_intensity = dark;
        self.light_intensity = light;
        self
    }

    /// Set the blur sigma.
    #[must_use]
    pub fn with_sigma(mut self, sigma: f64) -> Self {
        self.sigma = sigma;
        self
    }

    /// Render a floating-point image of a single edge.
    /// Strictly evaluates the ERF model at pixel centers (x+0.5, y+0.5).
    #[must_use]
    pub fn render_edge(&self, line: &Line) -> Vec<f64> {
        let mut data = vec![0.0; self.width * self.height];
        let a = self.dark_intensity;
        let b = self.light_intensity;

        // I(d) = (A+B)/2 + (B-A)/2 * erf(d / sigma)
        let s = self.sigma;

        for y in 0..self.height {
            let row_off = y * self.width;
            for x in 0..self.width {
                // Foundation Principle 1: Pixel center is at (x+0.5, y+0.5)
                let px = x as f64 + 0.5;
                let py = y as f64 + 0.5;

                let d = line.signed_distance([px, py]);

                let val =
                    f64::midpoint(a, b) + (b - a) / 2.0 * crate::simd::math::erf_approx(d / s);
                data[row_off + x] = val;
            }
        }
        data
    }

    /// Render a u8 image (clamped).
    #[must_use]
    #[allow(clippy::cast_sign_loss)]
    pub fn render_edge_u8(&self, line: &Line) -> Vec<u8> {
        self.render_edge(line)
            .into_iter()
            .map(|v| v.clamp(0.0, 255.0) as u8)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Point;
    use crate::image::ImageView;
    use crate::quad::refine_edge_intensity;
    use bumpalo::Bump;

    #[test]
    fn test_line_distance() {
        // Vertical line at x=10, CW points (10,0) to (10,100)
        // Outward normal points to x > 10.
        let line = Line::from_points_cw([10.0, 0.0], [10.0, 100.0]);

        assert!(line.signed_distance([11.0, 50.0]) > 0.0);
        assert!(line.signed_distance([9.0, 50.0]) < 0.0);
        assert!((line.signed_distance([10.0, 50.0])).abs() < 1e-12);
    }

    #[test]
    fn test_renderer_center_rule() {
        let renderer = SubpixelEdgeRenderer::new(10, 10)
            .with_intensities(0.0, 200.0)
            .with_sigma(0.1); // Sharp edge

        // Vertical edge at exactly x=5.0
        let line = Line::from_points_cw([5.0, 0.0], [5.0, 10.0]);
        let data = renderer.render_edge(&line);

        // Pixel x=4 center is 4.5 -> d = -0.5 -> Dark
        // Pixel x=5 center is 5.5 -> d = 0.5 -> Light
        assert!(data[4] < 50.0);
        assert!(data[5] > 150.0);
    }

    #[test]
    fn test_renderer_meta_hand_calculated() {
        let sigma = 0.8;
        let renderer = SubpixelEdgeRenderer::new(3, 1)
            .with_intensities(0.0, 255.0)
            .with_sigma(sigma);

        // Vertical edge at x=1.5
        let line = Line::from_points_cw([1.5, 0.0], [1.5, 1.0]);
        let data = renderer.render_edge(&line);

        let d = -1.0;
        let expected_erf = libm::erf(d / sigma);
        let expected_val = 127.5 + 127.5 * expected_erf;

        // x=0 center is 0.5, d = -1.0
        // Error of erf_approx is 1.5e-7. Scaled by 127.5, max error is ~1.9e-5.
        assert!((data[0] - expected_val).abs() < 2e-5);

        // x=1 center is 1.5, d = 0.0
        // I = 127.5
        assert!((data[1] - 127.5).abs() < 1e-7);

        // x=2 center is 2.5, d = 1.0
        let expected_val_high = 127.5 + 127.5 * libm::erf(1.0 / sigma);
        assert!((data[2] - expected_val_high).abs() < 2e-5);
    }

    #[test]
    fn test_edge_recovery_axis_aligned() {
        let width = 100;
        let height = 100;
        let sigma = 0.6;
        let renderer = SubpixelEdgeRenderer::new(width, height)
            .with_intensities(0.0, 255.0)
            .with_sigma(sigma);

        // Sweeps: X = 50.0, 50.25, 50.5, 50.75
        for x_gt in [50.0, 50.25, 50.5, 50.75] {
            // Vertical edge at x_gt, CW (x_gt, 10) to (x_gt, 90)
            // Outward normal points to light side B (x > x_gt)
            let line_gt = Line::from_points_cw([x_gt, 10.0], [x_gt, 90.0]);
            let data = renderer.render_edge_u8(&line_gt);
            let img = ImageView::new(&data, width, height, width).expect("invalid image view");
            let arena = Bump::new();

            // Initial guess: exactly at integer x=50.0
            let p1 = Point { x: 50.0, y: 10.0 };
            let p2 = Point { x: 50.0, y: 90.0 };

            let result = refine_edge_intensity(&arena, &img, p1, p2, sigma, 1);
            assert!(
                result.is_some(),
                "refine_edge_intensity failed for x_gt={x_gt}"
            );

            if let Some((nx, _ny, d)) = result {
                // Recovered edge: nx*x + ny*y + d = 0
                // For vertical edge, nx should be 1.0
                assert!((nx - 1.0).abs() < 1e-7);
                let x_recovered = -d;
                let error = (x_recovered - x_gt).abs();
                println!("x_gt={x_gt}, recovered={x_recovered}, error={error}");

                // Axis-aligned edges have higher quantization noise
                assert!(error < 0.02, "Error {error} too high for x_gt={x_gt}");
            }
        }
    }

    #[test]
    fn test_edge_recovery_arbitrary_angle() {
        let width = 120;
        let height = 120;
        let sigma = 0.6;
        let renderer = SubpixelEdgeRenderer::new(width, height)
            .with_intensities(20.0, 240.0)
            .with_sigma(sigma);

        // Test angles from 0 to 45 degrees
        for &angle_deg in &[5.0, 15.0, 30.0, 45.0] {
            let angle = f64::to_radians(angle_deg);
            let cos_a = angle.cos();
            let sin_a = angle.sin();

            // Line passing through (60, 60) with angle
            // p1 = center - 40*dir, p2 = center + 40*dir
            let p1_x = 60.0 - 40.0 * sin_a;
            let p1_y = 60.0 + 40.0 * cos_a;
            let p2_x = 60.0 + 40.0 * sin_a;
            let p2_y = 60.0 - 40.0 * cos_a;

            let line_gt = Line::from_points_cw([p1_x, p1_y], [p2_x, p2_y]);
            let data = renderer.render_edge_u8(&line_gt);
            let img = ImageView::new(&data, width, height, width).expect("invalid image view");
            let arena = Bump::new();

            let p1 = Point { x: p1_x, y: p1_y };
            let p2 = Point { x: p2_x, y: p2_y };

            let result = refine_edge_intensity(&arena, &img, p1, p2, sigma, 1);
            assert!(
                result.is_some(),
                "refine_edge_intensity failed for angle {angle_deg}"
            );

            if let Some((nx, ny, d)) = result {
                // Ground truth line is nx_gt*x + ny_gt*y + d_gt = 0
                // Our recovered line parameters are (nx, ny, d)
                let error_n = (nx - line_gt.a).abs() + (ny - line_gt.b).abs();
                let error_d = (d - line_gt.c).abs();

                println!("Angle {angle_deg}deg: error_n={error_n:.6}, error_d={error_d:.6}");

                // Angle recovery is very accurate
                assert!(error_n < 0.001);
                assert!(error_d < 0.05);
            }
        }
    }

    #[test]
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    fn test_edge_recovery_scale_invariance() {
        let sigma = 0.6;
        let mut renderer = SubpixelEdgeRenderer::new(1, 1).with_sigma(sigma);

        for len in [10.0, 30.0, 80.0, 150.0] {
            let width = (len + 20.0) as usize;
            let height = (len + 20.0) as usize;
            let center = f64::midpoint(len, 20.0);

            let p1_x = center;
            let p1_y = center - len / 2.0;
            let p2_x = center;
            let p2_y = center + len / 2.0;

            let line_gt = Line::from_points_cw([p1_x, p1_y], [p2_x, p2_y]);
            renderer.width = width;
            renderer.height = height;
            let data = renderer
                .with_intensities(50.0, 200.0)
                .render_edge_u8(&line_gt);
            let img = ImageView::new(&data, width, height, width).expect("invalid image view");
            let arena = Bump::new();

            let p1 = Point { x: p1_x, y: p1_y };
            let p2 = Point { x: p2_x, y: p2_y };

            let result = refine_edge_intensity(&arena, &img, p1, p2, sigma, 1);
            assert!(
                result.is_some(),
                "refine_edge_intensity failed for length {len}"
            );

            if let Some((_nx, _ny, d)) = result {
                let x_recovered = -d;
                let error = (x_recovered - p1_x).abs();
                println!("Length {len}: error={error:.6}");
                assert!(error < 0.01);
            }
        }
    }

    #[test]
    fn test_edge_recovery_decimation_mapping() {
        let canvas_size = 200;
        let sigma = 0.8;
        let x_gt = 100.4;

        let renderer = SubpixelEdgeRenderer::new(canvas_size, canvas_size).with_sigma(sigma);
        let line_gt = Line::from_points_cw([x_gt, 0.0], [x_gt, 200.0]);
        let data_full = renderer.render_edge_u8(&line_gt);
        let img_full = ImageView::new(&data_full, canvas_size, canvas_size, canvas_size)
            .expect("invalid image view");

        // Test with decimation=2
        let decimation = 2;
        let mut data_dec = vec![0u8; (canvas_size / decimation) * (canvas_size / decimation)];
        let img_dec = img_full
            .decimate_to(decimation, &mut data_dec)
            .expect("decimation failed");

        let arena = Bump::new();

        // Initial guess in decimated coordinates: roughly at x=50
        // (maps to 100.0 in full-res)
        let p1_dec = Point { x: 50.0, y: 0.0 };
        let p2_dec = Point { x: 50.0, y: 100.0 };

        // Refine on the decimated image
        let result = refine_edge_intensity(&arena, &img_dec, p1_dec, p2_dec, sigma, decimation);
        assert!(
            result.is_some(),
            "refine_edge_intensity failed with decimation upscaling"
        );

        if let Some((_nx, _ny, d)) = result {
            // Mapping back to full res should be x_full = (x_dec - 0.5) * (decimation as f64) + 0.5
            // because SubpixelEdgeRenderer::render_edge uses subsampling (pick top-left).
            let x_dec_recovered = -d;
            let x_full_recovered = (x_dec_recovered - 0.5) * (decimation as f64) + 0.5;

            let error = (x_full_recovered - x_gt).abs();
            println!("Decimated (d=2) recovered: {x_full_recovered:.4}, error: {error:.4}");

            // Decimation reduces precision but mapping should be correct within ~0.1px
            assert!(error < 0.1, "Mapping error {error} too high");
        }
    }

    #[test]
    fn test_edge_recovery_robustness_noise() {
        use rand::prelude::*;

        let width = 60;
        let height = 60;
        let sigma = 0.6;
        let x_gt = 30.25;
        let mut rng = rand::rng();

        let renderer = SubpixelEdgeRenderer::new(width, height)
            .with_intensities(50.0, 200.0)
            .with_sigma(sigma);
        let line_gt = Line::from_points_cw([x_gt, 0.0], [x_gt, 60.0]);
        let mut data = renderer.render_edge_u8(&line_gt);

        // Add Gaussian-like noise
        #[allow(clippy::cast_sign_loss)]
        for p in &mut data {
            let noise: i16 = rng.random_range(-10..11);
            *p = (i16::from(*p) + noise).clamp(0, 255) as u8;
        }

        let img = ImageView::new(&data, width, height, width).expect("invalid image view");
        let arena = Bump::new();
        let p1 = Point { x: 30.0, y: 0.0 };
        let p2 = Point { x: 30.0, y: 60.0 };

        if let Some((_nx, _ny, d)) = refine_edge_intensity(&arena, &img, p1, p2, sigma, 1) {
            let error = (-d - x_gt).abs();
            println!("Noisy recovery error: {error:.4}");
            assert!(error < 0.05);
        }
    }

    #[test]
    fn test_edge_recovery_robustness_low_contrast() {
        let width = 60;
        let height = 60;
        let sigma = 0.6;
        let x_gt = 30.25;

        let renderer = SubpixelEdgeRenderer::new(width, height)
            .with_intensities(100.0, 130.0) // Only 30 levels of contrast
            .with_sigma(sigma);
        let line_gt = Line::from_points_cw([x_gt, 0.0], [x_gt, 60.0]);
        let data = renderer.render_edge_u8(&line_gt);

        let img = ImageView::new(&data, width, height, width).expect("invalid image view");
        let arena = Bump::new();
        let p1 = Point { x: 30.0, y: 0.0 };
        let p2 = Point { x: 30.0, y: 60.0 };

        if let Some((_nx, _ny, d)) = refine_edge_intensity(&arena, &img, p1, p2, sigma, 1) {
            let error = (-d - x_gt).abs();
            println!("Low contrast recovery error: {error:.4}");
            assert!(error < 0.05);
        }
    }

    #[test]
    fn test_edge_recovery_robustness_clipping() {
        let width = 60;
        let height = 60;
        let sigma = 0.6;
        let x_gt = 30.25;

        let renderer = SubpixelEdgeRenderer::new(width, height)
            .with_intensities(-50.0, 300.0) // Intensities outside 0-255 (will clip)
            .with_sigma(sigma);
        let line_gt = Line::from_points_cw([x_gt, 0.0], [x_gt, 60.0]);
        let data = renderer.render_edge_u8(&line_gt);

        let img = ImageView::new(&data, width, height, width).expect("invalid image view");
        let arena = Bump::new();
        let p1 = Point { x: 30.0, y: 0.0 };
        let p2 = Point { x: 30.0, y: 60.0 };

        if let Some((_nx, _ny, d)) = refine_edge_intensity(&arena, &img, p1, p2, sigma, 1) {
            let error = (-d - x_gt).abs();
            println!("Clipped recovery error: {error:.4}");
            assert!(error < 0.1);
        }
    }

    #[test]
    fn test_edge_recovery_robustness_off_edge_seed() {
        let width = 60;
        let height = 60;
        let sigma = 0.6;
        let x_gt = 30.25;

        let renderer = SubpixelEdgeRenderer::new(width, height)
            .with_intensities(50.0, 200.0)
            .with_sigma(sigma);
        let line_gt = Line::from_points_cw([x_gt, 0.0], [x_gt, 60.0]);
        let data = renderer.render_edge_u8(&line_gt);

        let img = ImageView::new(&data, width, height, width).expect("invalid image view");
        let arena = Bump::new();

        // Initial seed is 1.5 pixels away from the true edge (within 3*sigma window)
        let p1 = Point { x: 31.75, y: 0.0 };
        let p2 = Point { x: 31.75, y: 60.0 };

        if let Some((_nx, _ny, d)) = refine_edge_intensity(&arena, &img, p1, p2, sigma, 1) {
            let error = (-d - x_gt).abs();
            println!("Off-edge seed recovery error: {error:.4}");
            assert!(error < 0.1);
        }
    }
}
