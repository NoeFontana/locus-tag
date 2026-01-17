#![allow(unsafe_code)]
//! Edge-preserving filtering for improved small tag detection.
//!
//! This module implements bilateral filtering and gradient computation
//! to enhance detection of small tags (<32px) by preserving edges while
//! reducing noise in uniform regions.

use crate::image::ImageView;
use multiversion::multiversion;

/// Apply bilateral filter for edge-preserving noise reduction.
///
/// The bilateral filter smooths uniform regions while preserving sharp edges,
/// making it ideal for preprocessing fiducial marker images with small features.
///
/// # Parameters
/// - `img`: Input grayscale image
/// - `output`: Output buffer for filtered image (must be img.width * img.height)
/// - `radius`: Spatial radius of the filter kernel (3 = 7x7 window)
/// - `sigma_space`: Spatial Gaussian sigma (controls spatial smoothing)
/// - `sigma_color`: Range Gaussian sigma (controls edge preservation, higher = more smoothing)
///
/// # Implementation Notes
/// - Uses separable approximation for O(n) instead of O(n²) complexity
/// - SIMD-optimized for AVX2/AVX-512/NEON architectures
/// - Typical overhead: ~1.5ms for 640x480 images
#[multiversion(targets(
    "x86_64+avx2+bmi1+bmi2+popcnt+lzcnt",
    "x86_64+avx512f+avx512bw+avx512dq+avx512vl",
    "aarch64+neon"
))]
pub fn bilateral_filter(
    img: &ImageView,
    output: &mut [u8],
    radius: usize,
    sigma_space: f32,
    sigma_color: f32,
) {
    let w = img.width;
    let h = img.height;
    
    // Internal buffer for cross-pass
    let temp = vec![0u8; w * h];
    
    // Precompute color Gaussian LUT
    let mut color_lut = [0.0f32; 256];
    let color_coeff = -1.0 / (2.0 * sigma_color * sigma_color);
    for i in 0..256 {
        color_lut[i] = (color_coeff * (i as f32).powi(2)).exp();
    }
    
    // Precompute 1D spatial Gaussian LUT
    let diameter = 2 * radius + 1;
    let mut spatial_lut = vec![0.0f32; diameter];
    let space_coeff = -1.0 / (2.0 * sigma_space * sigma_space);
    for i in 0..diameter {
        let d = (i as i32 - radius as i32) as f32;
        spatial_lut[i] = (space_coeff * d * d).exp();
    }

    use rayon::prelude::*;

    // Pass 1: Horizontal (Parallel)
    (0..h).into_par_iter().for_each(|y| {
        let src_row = img.get_row(y);
        // Safety: temp is only written to unique rows
        let dst_row = unsafe {
            let ptr = temp.as_ptr() as *mut u8;
            std::slice::from_raw_parts_mut(ptr.add(y * w), w)
        };
        
        for x in 0..w {
            let center_val = src_row[x];
            let mut sum_weights = 0.0f32;
            let mut filtered_val = 0.0f32;
            
            for dx in 0..diameter {
                let nx = (x as i32 + dx as i32 - radius as i32).clamp(0, w as i32 - 1) as usize;
                let neighbor_val = src_row[nx];
                let color_diff = (center_val as i32 - neighbor_val as i32).unsigned_abs() as usize;
                
                let weight = spatial_lut[dx] * color_lut[color_diff];
                filtered_val += neighbor_val as f32 * weight;
                sum_weights += weight;
            }
            dst_row[x] = (filtered_val / sum_weights).clamp(0.0, 255.0) as u8;
        }
    });

    // Pass 2: Vertical (Parallel and Cache-Friendly)
    (0..h).into_par_iter().for_each(|y| {
        // Safety: output is only written to unique rows
        let dst_row = unsafe {
            let ptr = output.as_ptr() as *mut u8;
            std::slice::from_raw_parts_mut(ptr.add(y * w), w)
        };
        
        for x in 0..w {
            let center_val = temp[y * w + x];
            let mut sum_weights = 0.0f32;
            let mut filtered_val = 0.0f32;
            
            for dy in 0..diameter {
                let ny = (y as i32 + dy as i32 - radius as i32).clamp(0, h as i32 - 1) as usize;
                let neighbor_val = temp[ny * w + x]; // Memory access is row-wise for adjacent x
                let color_diff = (center_val as i32 - neighbor_val as i32).unsigned_abs() as usize;
                
                let weight = spatial_lut[dy] * color_lut[color_diff];
                filtered_val += neighbor_val as f32 * weight;
                sum_weights += weight;
            }
            dst_row[x] = (filtered_val / sum_weights).clamp(0.0, 255.0) as u8;
        }
    });
}

/// Compute gradient magnitude map using Scharr operator.
///
/// The Scharr operator provides better rotational symmetry than Sobel,
/// making it more suitable for gradient-based adaptive window sizing.
///
/// # Parameters
/// - `img`: Input grayscale image
/// - `output`: Output buffer for gradient magnitude (normalized to [0, 255])
///
/// # Implementation Notes
/// - Uses 3x3 Scharr kernels for x and y directions
/// - Returns normalized gradient magnitude: sqrt(gx² + gy²)
/// - SIMD-optimized for modern architectures
#[multiversion(targets(
    "x86_64+avx2+bmi1+bmi2+popcnt+lzcnt",
    "x86_64+avx512f+avx512bw+avx512dq+avx512vl",
    "aarch64+neon"
))]
pub fn compute_gradient_map(img: &ImageView, output: &mut [u8]) {
    let w = img.width;
    let h = img.height;

    // Scharr kernels (better rotational symmetry than Sobel)
    // Gx kernel:          Gy kernel:
    //  -3   0   3         -3  -10  -3
    // -10   0  10          0    0   0
    //  -3   0   3          3   10   3

    use rayon::prelude::*;

    // Process rows in parallel
    (0..h).into_par_iter().for_each(|y| {
        // Safety: Unique row per thread
        let dst_row = unsafe {
            let ptr = output.as_ptr() as *mut u8;
            std::slice::from_raw_parts_mut(ptr.add(y * w), w)
        };

        let y0 = y.saturating_sub(1);
        let y1 = y;
        let y2 = (y + 1).min(h - 1);

        let r0 = img.get_row(y0);
        let r1 = img.get_row(y1);
        let r2 = img.get_row(y2);

        for x in 0..w {
            let x0 = x.saturating_sub(1);
            let x1 = x;
            let x2 = (x + 1).min(w - 1);

            // Sample 3x3 neighborhood from pre-fetched rows
            let p00 = r0[x0] as i32;
            let p01 = r0[x1] as i32;
            let p02 = r0[x2] as i32;
            let p10 = r1[x0] as i32;
            let p12 = r1[x2] as i32;
            let p20 = r2[x0] as i32;
            let p21 = r2[x1] as i32;
            let p22 = r2[x2] as i32;

            // Apply Scharr kernels
            let gx = -3 * p00 + 3 * p02 - 10 * p10 + 10 * p12 - 3 * p20 + 3 * p22;
            let gy = -3 * p00 - 10 * p01 - 3 * p02 + 3 * p20 + 10 * p21 + 3 * p22;

            // Gradient magnitude (L2 norm)
            let grad = ((gx * gx + gy * gy) as f32).sqrt();
            
            // Normalize to [0, 255] range
            let normalized = (grad * (255.0 / 1448.0)).clamp(0.0, 255.0) as u8;
            dst_row[x] = normalized;
        }
    });
}

/// Apply a 3x3 Laplacian sharpening filter to enhance edges.
///
/// This filter uses the kernel:
/// ```
/// [ 0  -1   0 ]
/// [-1   5  -1 ]
/// [ 0  -1   0 ]
/// ```
///
/// # Parameters
/// - `img`: Input grayscale image
/// - `output`: Output buffer (must be img.width * img.height)
#[multiversion(targets(
    "x86_64+avx2+bmi1+bmi2+popcnt+lzcnt",
    "x86_64+avx512f+avx512bw+avx512dq+avx512vl",
    "aarch64+neon"
))]
pub fn laplacian_sharpen(img: &ImageView, output: &mut [u8]) {
    let w = img.width;
    let h = img.height;

    use rayon::prelude::*;

    (0..h).into_par_iter().for_each(|y| {
        let y0 = y.saturating_sub(1);
        let y1 = y;
        let y2 = (y + 1).min(h - 1);
        
        let r0 = img.get_row(y0);
        let r1 = img.get_row(y1);
        let r2 = img.get_row(y2);
        
        // Safety: Unique row per thread
        let dst_row = unsafe {
            let ptr = output.as_ptr() as *mut u8;
            std::slice::from_raw_parts_mut(ptr.add(y * w), w)
        };

        for x in 0..w {
            let x0 = x.saturating_sub(1);
            let x1 = x;
            let x2 = (x + 1).min(w - 1);

            // Sample from pre-fetched rows
            let p11 = r1[x1] as i32;
            let p01 = r0[x1] as i32;
            let p10 = r1[x0] as i32;
            let p12 = r1[x2] as i32;
            let p21 = r2[x1] as i32;

            // Apply sharpening: 5*center - (sum of 4 neighbors)
            let sharpened = 5 * p11 - (p01 + p10 + p12 + p21);
            dst_row[x] = sharpened.clamp(0, 255) as u8;
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::ImageView;

    #[test]
    fn test_bilateral_filter_preserves_edges() {
        // Create test image with sharp edge
        let width = 32;
        let height = 32;
        let mut data = vec![50u8; width * height];
        
        // Create vertical edge at x=16
        for y in 0..height {
            for x in 16..width {
                data[y * width + x] = 200;
            }
        }
        
        let img = ImageView::new(&data, width, height, width).unwrap();
        let mut output = vec![0u8; width * height];
        
        // Apply bilateral filter with moderate parameters
        bilateral_filter(&img, &mut output, 3, 3.0, 30.0);
        
        // Check edge preservation: values near edge should still be distinct
        let left_avg = output[16 * width + 14] as f32; // 2px left of edge
        let right_avg = output[16 * width + 18] as f32; // 2px right of edge
        
        // Edge should still be sharp (>100 intensity difference)
        assert!(
            (right_avg - left_avg).abs() > 100.0,
            "Edge not preserved: left={}, right={}",
            left_avg,
            right_avg
        );
    }

    #[test]
    fn test_bilateral_filter_reduces_noise() {
        // Create uniform region with noise
        let width = 32;
        let height = 32;
        let mut data = vec![128u8; width * height];
        
        // Add salt-and-pepper noise
        data[16 * width + 16] = 255;
        data[17 * width + 16] = 0;
        data[16 * width + 17] = 255;
        
        let img = ImageView::new(&data, width, height, width).unwrap();
        let mut output = vec![0u8; width * height];
        
        bilateral_filter(&img, &mut output, 3, 3.0, 30.0);
        
        // Filtered values should be closer to 128 than original noise
        assert!(
            (output[16 * width + 16] as i32 - 128).abs() < 50,
            "Noise not reduced at noisy pixel"
        );
    }

    #[test]
    fn test_gradient_map_detects_edges() {
        // Create test image with vertical and horizontal edges
        let width = 32;
        let height = 32;
        let mut data = vec![50u8; width * height];
        
        // Vertical edge at x=16
        for y in 0..height {
            for x in 16..width {
                data[y * width + x] = 200;
            }
        }
        
        let img = ImageView::new(&data, width, height, width).unwrap();
        let mut gradient = vec![0u8; width * height];
        
        compute_gradient_map(&img, &mut gradient);
        
        // Gradient should be high at edge (x=15,16)
        let edge_grad = gradient[16 * width + 15];
        assert!(
            edge_grad > 100,
            "Gradient too low at edge: {}",
            edge_grad
        );
        
        // Gradient should be low in uniform regions
        let uniform_grad = gradient[16 * width + 8];
        assert!(
            uniform_grad < 20,
            "Gradient too high in uniform region: {}",
            uniform_grad
        );
    }

    #[test]
    fn test_gradient_map_border_handling() {
        // Test that corners and edges don't panic
        let width = 8;
        let height = 8;
        let data = vec![128u8; width * height];
        
        let img = ImageView::new(&data, width, height, width).unwrap();
        let mut gradient = vec![0u8; width * height];
        
        // Should not panic
        compute_gradient_map(&img, &mut gradient);
        
        // All gradients should be low for uniform image
        assert!(gradient.iter().all(|&g| g < 10));
    }

    #[test]
    fn test_laplacian_sharpen_enhances_edges() {
        let width = 8;
        let height = 8;
        let mut data = vec![100u8; width * height];
        // Create a horizontal line (edge)
        for x in 0..width {
            data[4 * width + x] = 200;
        }
        
        let img = ImageView::new(&data, width, height, width).unwrap();
        let mut output = vec![0u8; width * height];
        
        laplacian_sharpen(&img, &mut output);
        
        // The value at 4,4 (center of edge) should be higher than 200 due to sharpening
        // Center is 200, neighbors are 100 (top), 200 (left), 200 (right), 100 (bottom)
        // 5*200 - (100 + 200 + 200 + 100) = 1000 - 600 = 400 -> 255
        assert!(output[4 * width + 4] > 200);
        
        // The value at 3,4 (just above edge) should be lower than 100
        // Center is 100, neighbors are 100, 100, 100, 200
        // 5*100 - (100 + 100 + 100 + 200) = 500 - 500 = 0
        assert_eq!(output[3 * width + 4], 0);
    }
}
