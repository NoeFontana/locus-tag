use crate::quad::erf_approx;

/// A mathematical line representation: ax + by + c = 0.
/// The normal vector (a, b) points towards the light side (d > 0).
#[derive(Debug, Clone, Copy)]
pub struct Line {
    /// Coefficient 'a' in ax + by + c = 0.
    pub a: f64,
    /// Coefficient 'b' in ax + by + c = 0.
    pub b: f64,
    /// Constant 'c' in ax + by + c = 0.
    pub c: f64,
}

impl Line {
    /// Create a line from two points, assuming Clockwise (CW) winding.
    /// The outward normal (pointing to light side) will be used.
    pub fn from_points_cw(p1: [f64; 2], p2: [f64; 2]) -> Self {
        // For CW winding, the outward normal is (y2 - y1, x1 - x2)
        let a = p2[1] - p1[1];
        let b = p1[0] - p2[0];
        let c = p2[0] * p1[1] - p1[0] * p2[1];
        Self { a, b, c }
    }

    /// Calculate the signed distance from a point to the line.
    /// Positive values are on the "light" side (outward normal direction).
    pub fn signed_distance(&self, p: [f64; 2]) -> f64 {
        let norm = (self.a * self.a + self.b * self.b).sqrt();
        if norm < 1e-12 {
            return 0.0;
        }
        (self.a * p[0] + self.b * p[1] + self.c) / norm
    }
}

/// A renderer for synthetic sub-pixel edges using the ERF model.
pub struct SubpixelEdgeRenderer {
    width: usize,
    height: usize,
    /// Dark side intensity (A).
    pub dark_intensity: f64,
    /// Light side intensity (B).
    pub light_intensity: f64,
    /// Gaussian PSF standard deviation (sigma).
    pub sigma: f64,
}

impl SubpixelEdgeRenderer {
    /// Create a new renderer with default parameters.
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
    pub fn with_intensities(mut self, dark: f64, light: f64) -> Self {
        self.dark_intensity = dark;
        self.light_intensity = light;
        self
    }

    /// Set the blur sigma.
    pub fn with_sigma(mut self, sigma: f64) -> Self {
        self.sigma = sigma;
        self
    }

    /// Render a floating-point image of a single edge.
    /// Strictly evaluates the ERF model at pixel centers (x+0.5, y+0.5).
    pub fn render_edge(&self, line: &Line) -> Vec<f64> {
        let mut data = vec![0.0; self.width * self.height];
        let a = self.dark_intensity;
        let b = self.light_intensity;
        
        // Foundation Principle 2: I(d) = (A+B)/2 + (B-A)/2 * erf(d / (sigma * sqrt(2)))
        let s_sqrt2 = self.sigma * std::f64::consts::SQRT_2;

        for y in 0..self.height {
            let row_off = y * self.width;
            for x in 0..self.width {
                // Foundation Principle 1: Pixel center is at (x+0.5, y+0.5)
                let px = x as f64 + 0.5;
                let py = y as f64 + 0.5;
                
                let d = line.signed_distance([px, py]);
                
                let val = (a + b) / 2.0 + (b - a) / 2.0 * erf_approx(d / s_sqrt2);
                data[row_off + x] = val;
            }
        }
        data
    }

    /// Render a u8 image (clamped).
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
        let renderer = SubpixelEdgeRenderer::new(3, 1)
            .with_intensities(0.0, 255.0)
            .with_sigma(0.8);
            
        // Vertical edge at x=1.5
        let line = Line::from_points_cw([1.5, 0.0], [1.5, 1.0]);
        let data = renderer.render_edge(&line);
        
        let d = -1.0;
        let s_sqrt2 = 0.8 * 2.0f64.sqrt();
        let expected_erf = libm::erf(d / s_sqrt2);
        let expected_val = 127.5 + 127.5 * expected_erf;

        // x=0 center is 0.5, d = -1.0
        // Error of erf_approx is 1.5e-7. Scaled by 127.5, max error is ~1.9e-5.
        assert!((data[0] - expected_val).abs() < 2e-5);
        
        // x=1 center is 1.5, d = 0.0
        // I = 127.5
        assert!((data[1] - 127.5).abs() < 1e-7);
        
        // x=2 center is 2.5, d = 1.0
        let expected_val_high = 127.5 + 127.5 * libm::erf(1.0 / s_sqrt2);
        assert!((data[2] - expected_val_high).abs() < 2e-5);
    }

    #[test]
    fn test_edge_recovery_axis_aligned() {
        use crate::quad::{refine_edge_intensity, Point};
        use crate::image::ImageView;
        use bumpalo::Bump;

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
            let img = ImageView::new(&data, width, height, width).unwrap();
            let arena = Bump::new();

            // Initial guess: exactly at integer x=50.0
            let p1 = Point { x: 50.0, y: 10.0 };
            let p2 = Point { x: 50.0, y: 90.0 };

            if let Some((nx, _ny, d)) = refine_edge_intensity(&arena, &img, p1, p2, sigma, 1) {
                // Recovered edge: nx*x + ny*y + d = 0
                // For vertical edge, nx should be 1.0
                assert!((nx - 1.0).abs() < 1e-7);
                let x_recovered = -d;
                let error = (x_recovered - x_gt).abs();
                println!("x_gt={x_gt}, recovered={x_recovered}, error={error}");
                assert!(error < 0.001, "Error {error} too high for x_gt={x_gt}");
            } else {
                panic!("refine_edge_intensity failed for x_gt={x_gt}");
            }
        }
    }

    #[test]
    fn test_edge_recovery_arbitrary_angle() {
        use crate::quad::{refine_edge_intensity, Point};
        use crate::image::ImageView;
        use bumpalo::Bump;

        let width = 100;
        let height = 100;
        let sigma = 0.6;
        let renderer = SubpixelEdgeRenderer::new(width, height)
            .with_intensities(0.0, 255.0)
            .with_sigma(sigma);

        // Angles: 30, 45, 60 degrees
        for angle_deg in [30.0, 45.0, 60.0] {
            let angle_rad = angle_deg * std::f64::consts::PI / 180.0;
            let c = angle_rad.cos();
            let s = angle_rad.sin();

            // Edge through center (50, 50) with given angle
            // CW: P1 to P2. Vector (c, s).
            let p1_gt = [50.0 - 30.0 * c, 50.0 - 30.0 * s];
            let p2_gt = [50.0 + 30.0 * c, 50.0 + 30.0 * s];
            
            let line_gt = Line::from_points_cw(p1_gt, p2_gt);
            let data = renderer.render_edge_u8(&line_gt);
            let img = ImageView::new(&data, width, height, width).unwrap();
            let arena = Bump::new();

            // Initial guess: displaced by 0.2 pixels
            let p1_guess = Point { x: p1_gt[0] + 0.2, y: p1_gt[1] };
            let p2_guess = Point { x: p2_gt[0] + 0.2, y: p2_gt[1] };

            if let Some((nx, ny, d)) = refine_edge_intensity(&arena, &img, p1_guess, p2_guess, sigma, 1) {
                // Recovered line: nx*x + ny*y + d = 0
                // Compare with normalized GT line
                let norm_gt = (line_gt.a * line_gt.a + line_gt.b * line_gt.b).sqrt();
                let a_gt = line_gt.a / norm_gt;
                let b_gt = line_gt.b / norm_gt;
                let c_gt = line_gt.c / norm_gt;

                let n_error = ((nx - a_gt).powi(2) + (ny - b_gt).powi(2)).sqrt();
                let d_error = (d - c_gt).abs();
                
                println!("angle={angle_deg}, n_err={n_error}, d_err={d_error}");
                
                assert!(n_error < 0.001, "Normal error {n_error} too high for angle {angle_deg}");
                assert!(d_error < 0.001, "Offset error {d_error} too high for angle {angle_deg}");
            } else {
                panic!("refine_edge_intensity failed for angle {angle_deg}");
            }
        }
    }

    #[test]
    fn test_edge_recovery_scale_invariance() {
        use crate::quad::{refine_edge_intensity, Point};
        use crate::image::ImageView;
        use bumpalo::Bump;

        let sigma = 0.6;
        let intensities = (0.0, 255.0);

        // Edge lengths: 10, 50, 100
        for len in [10.0, 50.0, 100.0] {
            let width = (len + 20.0) as usize;
            let height = (len + 20.0) as usize;
            let renderer = SubpixelEdgeRenderer::new(width, height)
                .with_intensities(intensities.0, intensities.1)
                .with_sigma(sigma);

            let mid_x = width as f64 / 2.0;
            let x_gt = mid_x + 0.25;
            let line_gt = Line::from_points_cw([x_gt, 10.0], [x_gt, 10.0 + len]);
            let data = renderer.render_edge_u8(&line_gt);
            let img = ImageView::new(&data, width, height, width).unwrap();
            let arena = Bump::new();

            let p1 = Point { x: mid_x, y: 10.0 };
            let p2 = Point { x: mid_x, y: 10.0 + len };

            if let Some((_nx, _ny, d)) = refine_edge_intensity(&arena, &img, p1, p2, sigma, 1) {
                let x_recovered = -d;
                let error = (x_recovered - x_gt).abs();
                println!("len={len}, error={error}");
                // Axis-aligned u8 quantization limit is ~0.002
                assert!(error < 0.005, "Error {error} too high for length {len}");
            } else {
                panic!("refine_edge_intensity failed for length {len}");
            }
        }
    }

    #[test]
    fn test_edge_recovery_decimation_mapping() {
        use crate::quad::{refine_edge_intensity, Point};
        use crate::image::ImageView;
        use bumpalo::Bump;

        let canvas_size = 200;
        let decimation = 2;
        let sigma = 0.6;
        let intensities = (0.0, 255.0);

        let renderer = SubpixelEdgeRenderer::new(canvas_size, canvas_size)
            .with_intensities(intensities.0, intensities.1)
            .with_sigma(sigma);

        // Vertical edge at x_gt = 100.25 (full res)
        let x_gt = 100.25;
        let line_gt = Line::from_points_cw([x_gt, 10.0], [x_gt, 190.0]);
        let data_full = renderer.render_edge_u8(&line_gt);
        let img_full = ImageView::new(&data_full, canvas_size, canvas_size, canvas_size).unwrap();

        // In a decimated image (K=2), the edge would be at:
        // x_gt = (x_dec - 0.5) * 2.0 + 0.5
        // 100.25 - 0.5 = (x_dec - 0.5) * 2.0
        // 99.75 / 2.0 = x_dec - 0.5
        // 49.875 + 0.5 = x_dec = 50.375
        let x_dec_gt = (x_gt - 0.5) / decimation as f64 + 0.5;
        
        // Initial guess in decimated space (e.g. integer 50)
        let x_dec_guess = 50.0;
        
        // Foundation Principle 3: Correct Upscale Mapping
        let x_upscaled = (x_dec_guess - 0.5) * decimation as f64 + 0.5;
        
        let p1 = Point { x: x_upscaled, y: 10.0 };
        let p2 = Point { x: x_upscaled, y: 190.0 };
        
        let arena = Bump::new();
        if let Some((_nx, _ny, d)) = refine_edge_intensity(&arena, &img_full, p1, p2, sigma, decimation) {
            let x_recovered = -d;
            let error = (x_recovered - x_gt).abs();
            println!("x_gt={x_gt}, x_dec_gt={x_dec_gt}, recovered={x_recovered}, error={error}");
            assert!(error < 0.001, "Error {error} too high with decimation upscaling");
        } else {
            panic!("refine_edge_intensity failed with decimation upscaling");
        }
    }

    #[test]
    fn test_edge_recovery_robustness_low_contrast() {
        use crate::quad::{refine_edge_intensity, Point};
        use crate::image::ImageView;
        use bumpalo::Bump;

        let width = 100;
        let height = 100;
        let sigma = 0.6;
        // Low contrast: 5 grayscale values delta
        let intensities = (120.0, 125.0);

        let renderer = SubpixelEdgeRenderer::new(width, height)
            .with_intensities(intensities.0, intensities.1)
            .with_sigma(sigma);

        let x_gt = 50.25;
        let line_gt = Line::from_points_cw([x_gt, 10.0], [x_gt, 90.0]);
        let data = renderer.render_edge_u8(&line_gt);
        let img = ImageView::new(&data, width, height, width).unwrap();
        let arena = Bump::new();

        let p1 = Point { x: 50.0, y: 10.0 };
        let p2 = Point { x: 50.0, y: 90.0 };

        // Should either converge or return Some(initial) / None
        // But must not panic.
        let result = refine_edge_intensity(&arena, &img, p1, p2, sigma, 1);
        assert!(result.is_some());
    }

    #[test]
    fn test_edge_recovery_robustness_noise() {
        use crate::quad::{refine_edge_intensity, Point};
        use crate::image::ImageView;
        use bumpalo::Bump;
        use rand::RngExt;

        let width = 100;
        let height = 100;
        let sigma = 0.8;
        let intensities = (50.0, 200.0);

        let renderer = SubpixelEdgeRenderer::new(width, height)
            .with_intensities(intensities.0, intensities.1)
            .with_sigma(sigma);

        let x_gt = 50.25;
        let line_gt = Line::from_points_cw([x_gt, 10.0], [x_gt, 90.0]);
        let mut data = renderer.render_edge_u8(&line_gt);
        
        // Add high Gaussian noise
        let mut rng = rand::rng();
        for p in &mut data {
            let noise: i16 = rng.random_range(-20..20);
            *p = (*p as i16 + noise).clamp(0, 255) as u8;
        }
        
        let img = ImageView::new(&data, width, height, width).unwrap();
        let arena = Bump::new();

        let p1 = Point { x: 50.0, y: 10.0 };
        let p2 = Point { x: 50.0, y: 90.0 };

        let result = refine_edge_intensity(&arena, &img, p1, p2, sigma, 1);
        assert!(result.is_some());
        if let Some((_nx, _ny, d)) = result {
            let x_recovered = -d;
            let error = (x_recovered - x_gt).abs();
            println!("Noise test error: {error}");
            // Degrades linearly, should still be somewhat reasonable (< 0.1)
            assert!(error < 0.1, "Error {error} too high with noise");
        }
    }

    #[test]
    fn test_edge_recovery_robustness_clipping() {
        use crate::quad::{refine_edge_intensity, Point};
        use crate::image::ImageView;
        use bumpalo::Bump;

        let width = 100;
        let height = 100;
        let sigma = 0.6;
        let renderer = SubpixelEdgeRenderer::new(width, height);

        // Edge right on the boundary
        let x_gt = 1.0; 
        let line_gt = Line::from_points_cw([x_gt, 0.0], [x_gt, 100.0]);
        let data = renderer.render_edge_u8(&line_gt);
        let img = ImageView::new(&data, width, height, width).unwrap();
        let arena = Bump::new();

        // Guess on boundary
        let p1 = Point { x: 1.0, y: 0.0 };
        let p2 = Point { x: 1.0, y: 100.0 };

        let result = refine_edge_intensity(&arena, &img, p1, p2, sigma, 1);
        assert!(result.is_some());
    }

    #[test]
    fn test_edge_recovery_robustness_off_edge_seed() {
        use crate::quad::{refine_edge_intensity, Point};
        use crate::image::ImageView;
        use bumpalo::Bump;

        let width = 100;
        let height = 100;
        let sigma = 0.8;
        let renderer = SubpixelEdgeRenderer::new(width, height);

        let x_gt = 50.0;
        let line_gt = Line::from_points_cw([x_gt, 10.0], [x_gt, 90.0]);
        let data = renderer.render_edge_u8(&line_gt);
        let img = ImageView::new(&data, width, height, width).unwrap();
        let arena = Bump::new();

        // Displaced by 2 pixels (within capture radius)
        let p1 = Point { x: 52.0, y: 10.0 };
        let p2 = Point { x: 52.0, y: 90.0 };

        let result = refine_edge_intensity(&arena, &img, p1, p2, sigma, 1);
        assert!(result.is_some());
        if let Some((_nx, _ny, d)) = result {
            let x_recovered = -d;
            let error = (x_recovered - x_gt).abs();
            println!("Off-edge seed error: {error}");
            assert!(error < 0.01, "Failed to pull back off-edge seed");
        }
    }
}
