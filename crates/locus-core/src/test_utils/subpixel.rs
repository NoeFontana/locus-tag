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
}
