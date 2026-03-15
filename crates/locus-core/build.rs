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
    mih_chunks: usize,
    mih_buckets: usize,
    mih_bits_per_chunk: u32,
    mih_last_chunk_bits: u32,
    mih_offsets: Vec<usize>,
    mih_data: Vec<u64>,
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

// Compute MIH exactly as TagDictionary::new() used to do
fn compute_mih(
    codes: &[u64],
    payload_length: u32,
    min_hamming: u32,
) -> (usize, usize, u32, u32, Vec<usize>, Vec<u64>) {
    let chunks = ((min_hamming as usize - 1) / 2) + 1;
    #[allow(clippy::cast_sign_loss, clippy::cast_precision_loss)]
    let bits_per_chunk = (payload_length as f32 / chunks as f32).ceil() as u32;
    let last_chunk_bits = payload_length - (chunks as u32 - 1) * bits_per_chunk;

    let mut buckets = vec![0; chunks];
    for (chunk_idx, mask_bits) in (0..chunks).map(|i| {
        let bits = if i == chunks - 1 {
            last_chunk_bits
        } else {
            bits_per_chunk
        };
        (i, bits)
    }) {
        buckets[chunk_idx] = 1 << mask_bits;
    }

    let total_buckets: usize = buckets.iter().sum();
    let mut bucket_counts = vec![0; total_buckets];

    let get_bucket_global_idx = |chunk_idx: usize, val: usize| -> usize {
        let offset: usize = buckets.iter().take(chunk_idx).sum();
        offset + val
    };

    // First pass: count sizes
    for &code in codes {
        for chunk_idx in 0..chunks {
            let chunk_bits = if chunk_idx == chunks - 1 {
                last_chunk_bits
            } else {
                bits_per_chunk
            };
            let shift = chunk_idx as u32 * bits_per_chunk;
            let mask = (1 << chunk_bits) - 1;
            let val = ((code >> shift) & mask) as usize;

            let global_idx = get_bucket_global_idx(chunk_idx, val);
            bucket_counts[global_idx] += 1;
        }
    }

    // Prefix sums
    let mut mih_offsets = vec![0; total_buckets + 1];
    for i in 0..total_buckets {
        mih_offsets[i + 1] = mih_offsets[i] + bucket_counts[i];
    }

    // Second pass: fill data
    let mut mih_data = vec![0u64; mih_offsets[total_buckets]];
    let mut current_offsets = mih_offsets.clone();

    for (i, &code) in codes.iter().enumerate() {
        for chunk_idx in 0..chunks {
            let chunk_bits = if chunk_idx == chunks - 1 {
                last_chunk_bits
            } else {
                bits_per_chunk
            };
            let shift = chunk_idx as u32 * bits_per_chunk;
            let mask = (1 << chunk_bits) - 1;
            let val = ((code >> shift) & mask) as usize;

            let global_idx = get_bucket_global_idx(chunk_idx, val);
            let pos = current_offsets[global_idx];

            // Encode the index into u64. We keep the upper 32 bits for the candidate original value if needed?
            // TagDictionary stores `index as u32`. Wait, actually the original TagDictionary `mih_data` stores `u32` or `u64`?
            // If u32, let's just pack it as u64 and downstream casts. We'll generate as u32 to be safe if `TagDictionary` uses `Vec<u32>`.
            // Wait, TagDictionary has `mih_data: Vec<u32>`.
            mih_data[pos] = i as u64; // We'll render as `u32` in template
            current_offsets[global_idx] += 1;
        }
    }

    (
        chunks,
        1 << bits_per_chunk,
        bits_per_chunk,
        last_chunk_bits,
        mih_offsets,
        mih_data,
    )
}

fn main() {
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

        let content = fs::read_to_string(&json_path).expect("failed to read json");
        let ir: DictionaryIR = serde_json::from_str(&content).expect("failed to parse json");

        let mut all_codes = Vec::with_capacity(ir.base_codes.len() * 4);
        for hex_str in &ir.base_codes {
            let base_code = u64::from_str_radix(hex_str, 16).expect("invalid hex base code");

            let rots =
                compute_rotations(base_code, ir.payload_length, &ir.canonical_sampling_points);
            all_codes.extend_from_slice(&rots);
        }

        let (
            mih_chunks,
            mih_buckets,
            mih_bits_per_chunk,
            mih_last_chunk_bits,
            mih_offsets,
            mih_data,
        ) = compute_mih(&all_codes, ir.payload_length, ir.minimum_hamming_distance);

        computed_dicts.push(ComputedDictionary {
            enum_name: enum_name.to_string(),
            payload_length: ir.payload_length,
            dimension: dim,
            minimum_hamming_distance: ir.minimum_hamming_distance,
            dictionary_size: ir.dictionary_size,
            mih_chunks,
            mih_buckets,
            mih_bits_per_chunk,
            mih_last_chunk_bits,
            mih_offsets,
            mih_data,
            codes: all_codes,
            canonical_sampling_points: ir.canonical_sampling_points,
        });
    }

    let template = DictionariesTemplate {
        dictionaries: computed_dicts,
    };
    let rendered = template.render().expect("failed to render askama template");

    let out_dir = env::var_os("OUT_DIR").unwrap();
    let dest_path = PathBuf::from(out_dir).join("dictionaries.rs");
    fs::write(&dest_path, rendered).unwrap();
}
