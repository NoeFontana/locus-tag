#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_lossless)]
#![allow(clippy::manual_range_contains)]
#![allow(clippy::manual_midpoint)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::return_self_not_must_use)]

use crate::config::TagFamily;
use crate::dictionaries::{APRILTAG_16H5, APRILTAG_36H11, ARUCO_4X4_50, ARUCO_4X4_100};
use rand::Rng;

/// A placement of a tag in a scene.
#[derive(Debug, Clone)]
pub struct TagPlacement {
    /// The tag family.
    pub family: TagFamily,
    /// The specific tag ID.
    pub id: u32,
    /// X-coordinate of the center.
    pub center_x: f64,
    /// Y-coordinate of the center.
    pub center_y: f64,
    /// The physical size (side length) of the tag in pixels.
    pub size: f64,
    /// Rotation in radians.
    pub rotation_rad: f64,
}

/// A builder for complex multi-tag scenes.
pub struct SceneBuilder {
    width: usize,
    height: usize,
    tags: Vec<TagPlacement>,
    background_gray: u8,
    noise_sigma: f64,
    blur_sigma: f64,
}

impl SceneBuilder {
    /// Create a new scene builder with given dimensions.
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            tags: Vec::new(),
            background_gray: 255,
            noise_sigma: 0.0,
            blur_sigma: 0.0,
        }
    }

    /// Set the background gray level (0-255).
    pub fn with_background(mut self, gray: u8) -> Self {
        self.background_gray = gray;
        self
    }

    /// Set the noise standard deviation.
    pub fn with_noise(mut self, sigma: f64) -> Self {
        self.noise_sigma = sigma;
        self
    }

    /// Set the blur standard deviation.
    pub fn with_blur(mut self, sigma: f64) -> Self {
        self.blur_sigma = sigma;
        self
    }

    /// Add a tag if it doesn't overlap with existing ones.
    pub fn add_tag(&mut self, placement: TagPlacement) -> bool {
        // Stricter overlap check to ensure quiet zones are preserved
        for existing in &self.tags {
            let dx = existing.center_x - placement.center_x;
            let dy = existing.center_y - placement.center_y;
            let dist = (dx * dx + dy * dy).sqrt();
            let min_dist = (existing.size + placement.size) * 0.8;
            if dist < min_dist {
                return false;
            }
        }
        self.tags.push(placement);
        true
    }

    /// Add a random tag from a family within a size range, with overlap prevention.
    pub fn add_random_tag<R: Rng>(
        &mut self,
        rng: &mut R,
        family: TagFamily,
        size_range: (f64, f64),
    ) -> bool {
        for _ in 0..100 {
            let (min_s, max_s) = size_range;
            let size = rng.gen_range(min_s..max_s);
            let half_s = size / 2.0;
            // Ensure some margin from edges
            let margin = 10.0;
            if self.width as f64 <= size + margin * 2.0 || self.height as f64 <= size + margin * 2.0
            {
                continue;
            }

            let center_x = rng.gen_range(half_s + margin..self.width as f64 - half_s - margin);
            let center_y = rng.gen_range(half_s + margin..self.height as f64 - half_s - margin);
            let rotation_rad = rng.gen_range(0.0..2.0 * std::f64::consts::PI);

            let dict = match family {
                TagFamily::AprilTag36h11 => &*APRILTAG_36H11,
                TagFamily::AprilTag16h5 => &*APRILTAG_16H5,
                TagFamily::ArUco4x4_50 => &*ARUCO_4X4_50,
                TagFamily::ArUco4x4_100 => &*ARUCO_4X4_100,
            };

            let id = rng.gen_range(0..dict.len() as u32);

            let placement = TagPlacement {
                family,
                id,
                center_x,
                center_y,
                size,
                rotation_rad,
            };

            if self.add_tag(placement) {
                return true;
            }
        }
        false
    }

    /// Build the scene image and return the data and placements.
    pub fn build(self) -> (Vec<u8>, Vec<TagPlacement>) {
        let mut data = vec![self.background_gray; self.width * self.height];

        for tag in &self.tags {
            self.draw_tag(&mut data, tag);
        }

        if self.noise_sigma > 0.0 {
            let mut rng = rand::thread_rng();
            for p in &mut data {
                let noise = rng.gen_range(-self.noise_sigma..self.noise_sigma);
                *p = (*p as f64 + noise).clamp(0.0, 255.0) as u8;
            }
        }

        if self.blur_sigma > 0.0 {
            data = apply_box_blur(&data, self.width, self.height);
        }

        (data, self.tags)
    }

    fn draw_tag(&self, data: &mut [u8], tag: &TagPlacement) {
        let dict = match tag.family {
            TagFamily::AprilTag36h11 => &*APRILTAG_36H11,
            TagFamily::AprilTag16h5 => &*APRILTAG_16H5,
            TagFamily::ArUco4x4_50 => &*ARUCO_4X4_50,
            TagFamily::ArUco4x4_100 => &*ARUCO_4X4_100,
        };

        let dim = dict.dimension;
        let bits = dict.get_code(tag.id as u16).unwrap_or(0);
        let total_dim = dim + 2;

        let c = tag.rotation_rad.cos();
        let s = tag.rotation_rad.sin();
        let half_size = tag.size / 2.0;

        let qz_size = tag.size * 1.5;
        let qz_half = qz_size / 2.0;

        let min_x = (tag.center_x - qz_size).max(0.0) as usize;
        let max_x = (tag.center_x + qz_size).min(self.width as f64 - 1.0) as usize;
        let min_y = (tag.center_y - qz_size).max(0.0) as usize;
        let max_y = (tag.center_y + qz_size).min(self.height as f64 - 1.0) as usize;

        // Use samples = 1 for perfectly sharp corners in benchmarks/tests
        let samples = 1;

        for y in min_y..=max_y {
            for x in min_x..=max_x {
                let mut total_intensity = 0u32;
                let mut num_samples = 0u32;

                for sy in 0..samples {
                    for sx in 0..samples {
                        let px = x as f64 + (sx as f64 + 0.5) / samples as f64 - 0.5;
                        let py = y as f64 + (sy as f64 + 0.5) / samples as f64 - 0.5;

                        let dx = px - tag.center_x;
                        let dy = py - tag.center_y;

                        let lqx = (dx * c + dy * s) / qz_half;
                        let lqy = (-dx * s + dy * c) / qz_half;

                        if lqx >= -1.0 && lqx <= 1.0 && lqy >= -1.0 && lqy <= 1.0 {
                            let lx = (dx * c + dy * s) / half_size;
                            let ly = (-dx * s + dy * c) / half_size;

                            if lx >= -1.0 && lx <= 1.0 && ly >= -1.0 && ly <= 1.0 {
                                let gx = (lx + 1.0) / 2.0 * total_dim as f64;
                                let gy = (ly + 1.0) / 2.0 * total_dim as f64;
                                let igx = gx.floor() as i32;
                                let igy = gy.floor() as i32;
                                let igx = igx.clamp(0, total_dim as i32 - 1);
                                let igy = igy.clamp(0, total_dim as i32 - 1);

                                let color = if igx == 0
                                    || igx == (total_dim as i32 - 1)
                                    || igy == 0
                                    || igy == (total_dim as i32 - 1)
                                {
                                    0
                                } else {
                                    let ix = igx - 1;
                                    let iy = igy - 1;
                                    let bit_idx = iy as usize * dim + ix as usize;
                                    if (bits >> bit_idx) & 1 != 0 { 255 } else { 0 }
                                };
                                total_intensity += color;
                            } else {
                                total_intensity += 255;
                            }
                            num_samples += 1;
                        }
                    }
                }
                if num_samples > 0 {
                    let tag_avg = total_intensity as f64 / num_samples as f64;
                    let bg_ratio =
                        (samples * samples - num_samples) as f64 / (samples * samples) as f64;
                    let tag_ratio = num_samples as f64 / (samples * samples) as f64;

                    let final_color =
                        tag_avg * tag_ratio + u32::from(data[y * self.width + x]) as f64 * bg_ratio;
                    data[y * self.width + x] = final_color.clamp(0.0, 255.0) as u8;
                }
            }
        }
    }
}

fn apply_box_blur(data: &[u8], width: usize, height: usize) -> Vec<u8> {
    let mut blurred = data.to_vec();
    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let mut sum = 0u32;
            for dy in 0..3 {
                for dx in 0..3 {
                    sum += u32::from(data[(y + dy - 1) * width + (x + dx - 1)]);
                }
            }
            blurred[y * width + x] = (sum / 9) as u8;
        }
    }
    blurred
}
