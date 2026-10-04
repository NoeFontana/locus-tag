#![allow(missing_docs)]
#![allow(clippy::panic)]
#![allow(clippy::unwrap_used)]

use askama::Template;
use serde::Deserialize;
use std::env;
use std::fs;
use std::path::PathBuf;

/// Maps a codebase enum family string to the literal JSON file prefix
const FAMILY_MAPPING: &[(&str, &str, usize)] = &[
    ("AprilTag16h5", "dict_apriltag_16h5", 4),
    ("AprilTag36h11", "dict_apriltag_36h11", 6),
    ("ArUco4x4_50", "dict_4x4_50", 4),
    ("ArUco4x4_100", "dict_4x4_100", 4),
    ("ArUco6x6_250", "dict_6x6_250", 6),
    ("ArUcoMip36h12", "dict_aruco_mip_36h12", 6),
];

#[derive(Deserialize, Debug)]
struct DictionaryIR {
    payload_length: u32,
    minimum_hamming_distance: u32,
    dictionary_size: usize,
    canonical_sampling_points: Vec<[f64; 2]>,
    base_codes: Vec<String>,
}

#[derive(Debug)]
struct ComputedDictionary {
    enum_name: String,
    payload_length: u32,
    dimension: usize,
    minimum_hamming_distance: u32,
    dictionary_size: usize,
    codes: Vec<u64>,
    canonical_sampling_points: Vec<[f64; 2]>,
}

#[derive(Template)]
#[template(path = "dictionaries.rs.j2")]
struct DictionariesTemplate {
    dictionaries: Vec<ComputedDictionary>,
}

fn rotate_points_90(points: &[[f64; 2]]) -> Vec<[f64; 2]> {
    points.iter().map(|&[x, y]| [-y, x]).collect()
}

fn find_closest_point_index(rotated: &[f64; 2], original: &[[f64; 2]]) -> usize {
    let mut min_dist_sq = f64::MAX;
    let mut best_idx = 0;
    for (i, p) in original.iter().enumerate() {
        let dist_sq = (rotated[0] - p[0]).powi(2) + (rotated[1] - p[1]).powi(2);
        if dist_sq < min_dist_sq {
            min_dist_sq = dist_sq;
            best_idx = i;
        }
    }
    best_idx
}

fn compute_rotations(base_code: u64, payload_length: u32, points: &[[f64; 2]]) -> [u64; 4] {
    let mut result = [0u64; 4];

    // OpenCV ArUco dictionaries in our JSON are already in row-major order.
    // Bit i corresponds to points[i].
    // Base code is rotation 0.
    result[0] = base_code;

    // We compute 3 more rotations (90, 180, 270 degrees clockwise).
    // In OpenCV, rotate(90 deg CW) means:
    // point (x, y) moves to (-y, x).
    // So bit i at points[i] moves to dst_idx = find_closest_point_index(rotate(points[i])).

    for (r, item) in result.iter_mut().enumerate().skip(1) {
        let mut rotated_code = 0u64;
        let mut curr_points = points.to_vec();
        // Rotate the bit pattern r times 90 degrees CW.
        for _ in 0..r {
            curr_points = rotate_points_90(&curr_points);
        }

        for i in 0..payload_length {
            if (base_code & (1 << i)) != 0 {
                let dst_idx = find_closest_point_index(&curr_points[i as usize], points);
                rotated_code |= 1 << dst_idx;
            }
        }
        *item = rotated_code;
    }

    result
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo:rerun-if-changed=data/dictionaries");
    println!("cargo:rerun-if-changed=templates/dictionaries.rs.j2");

    let dict_dir = PathBuf::from("data/dictionaries");
    let mut computed_dicts = Vec::new();

    for &(enum_name, file_prefix, dim) in FAMILY_MAPPING {
        let json_path = dict_dir.join(format!("{file_prefix}.json"));
        assert!(
            json_path.exists(),
            "Required dictionary JSON not found: {}",
            json_path.display()
        );

        let content = fs::read_to_string(&json_path)?;
        let ir: DictionaryIR = serde_json::from_str(&content)?;

        let mut all_codes = Vec::with_capacity(ir.base_codes.len() * 4);
        for hex_str in &ir.base_codes {
            let base_code = u64::from_str_radix(hex_str, 16)?;

            let rots =
                compute_rotations(base_code, ir.payload_length, &ir.canonical_sampling_points);
            all_codes.extend_from_slice(&rots);
        }
        // `dictionaries::nearest_key` packs a code index into 16 bits.
        assert!(
            all_codes.len() <= 1 << 16,
            "{file_prefix}: {} rotated codes exceed the 16-bit decode index",
            all_codes.len()
        );

        computed_dicts.push(ComputedDictionary {
            enum_name: enum_name.to_string(),
            payload_length: ir.payload_length,
            dimension: dim,
            minimum_hamming_distance: ir.minimum_hamming_distance,
            dictionary_size: ir.dictionary_size,
            codes: all_codes,
            canonical_sampling_points: ir.canonical_sampling_points,
        });
    }

    let template = DictionariesTemplate {
        dictionaries: computed_dicts,
    };
    let rendered = template.render()?;

    let out_dir = env::var_os("OUT_DIR").ok_or("OUT_DIR not set")?;
    let dest_path = PathBuf::from(out_dir).join("dictionaries.rs");
    fs::write(&dest_path, rendered)?;
    Ok(())
}
