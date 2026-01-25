#!/usr/bin/env python3
import re
from pathlib import Path

target_file = Path("scripts/generate_dictionaries.py")
content = target_file.read_text()

# 1. Inject imports
if "import apriltag_41h12_data" not in content:
    import_block = """import re
import sys
from pathlib import Path

# Add scripts directory to path to import generated data modules
sys.path.append(str(Path(__file__).parent))

try:
    import apriltag_41h12_data
except ImportError:
    print("Warning: apriltag_41h12_data.py not found. 41h12 will be skipped.")
    apriltag_41h12_data = None
"""
    content = content.replace("import re\nfrom pathlib import Path", import_block)

# 2. Inject helper functions
helper_funcs = """
def generate_grid_points(dim: int, total_width: int) -> list:
    points = []
    start_idx = (total_width - dim) / 2
    for r in range(dim):
        for c in range(dim):
            y_idx = start_idx + r
            x_idx = start_idx + c
            y = (y_idx + 0.5 - total_width / 2.0) * (2.0 / total_width)
            x = (x_idx + 0.5 - total_width / 2.0) * (2.0 / total_width)
            points.append((x, y))
    return points

def generate_sparse_points(bit_order: list, total_width: int) -> list:
    if not bit_order: return []
    xs = [p[0] for p in bit_order]
    ys = [p[1] for p in bit_order]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    center_x = (min_x + max_x) / 2.0
    center_y = (min_y + max_y) / 2.0
    scale = 2.0 / total_width
    points = []
    for x_idx, y_idx in bit_order:
        x = (x_idx - center_x) * scale
        y = (y_idx - center_y) * scale
        points.append((x, y))
    return points

def generate_point_array(name: str, points: list) -> str:
    lines = [f"#[rustfmt::skip]"]
    lines.append(f"pub const {name}: [(f64, f64); {len(points)}] = [")
    for i in range(0, len(points), 2):
        chunk = points[i : i + 2]
        strs = [f"({p[0]:.6}, {p[1]:.6})" for p in chunk]
        lines.append(f"    {', '.join(strs)},")
    lines.append("];")
    return "\\n".join(lines)
"""

if "def generate_grid_points" not in content:
    # Insert before generate_rust_array
    content = content.replace(
        "def generate_rust_array(name: str", helper_funcs + "\n\ndef generate_rust_array(name: str"
    )

# 3. Replace generate_dictionaries_rs function
new_func = """def generate_dictionaries_rs() -> str:
    # Pre-calculate points
    points_36h11 = generate_grid_points(6, 8)
    points_16h5 = generate_grid_points(4, 6)
    points_aruco = generate_grid_points(4, 6)
    
    parts = [
        "//! Tag family dictionaries.",
        "//!",
        "//! This module contains pre-generated code tables for AprilTag families.",
        "//! Codes are in row-major bit ordering for efficient extraction.",
        "",
        UMICH_LICENSE,
        "",
        "use std::borrow::Cow;",
        "use std::collections::HashMap;",
        "",
        "/// A tag family dictionary.",
        "#[derive(Clone, Debug)]",
        "pub struct TagDictionary {",
        "    /// Name of the tag family.",
        "    pub name: Cow<'static, str>,",
        "    /// Grid dimension (e.g., 6 for 36h11).",
        "    pub dimension: usize,",
        "    /// Minimum hamming distance of the family.",
        "    pub hamming_distance: usize,",
        "    /// Pre-computed sample points in canonical tag coordinates [-1, 1].",
        "    pub sample_points: Cow<'static, [(f64, f64)]>,",
        "    /// Raw code table.",
        "    codes: Cow<'static, [u64]>,",
        "    /// Lookup table for O(1) exact matching.",
        "    code_to_id: HashMap<u64, u16>,",
        "}",
        "",
        "impl TagDictionary {",
        "    /// Create a new dictionary from static code table.",
        "    #[must_use]",
        "    pub fn new(",
        "        name: &'static str,",
        "        dimension: usize,",
        "        hamming_distance: usize,",
        "        codes: &'static [u64],",
        "        sample_points: &'static [(f64, f64)],",
        "    ) -> Self {",
        "        let mut code_to_id = HashMap::with_capacity(codes.len());",
        "        for (id, &code) in codes.iter().enumerate() {",
        "            code_to_id.insert(code, id as u16);",
        "        }",
        "        Self {",
        "            name: Cow::Borrowed(name),",
        "            dimension,",
        "            hamming_distance,",
        "            sample_points: Cow::Borrowed(sample_points),",
        "            codes: Cow::Borrowed(codes),",
        "            code_to_id,",
        "        }",
        "    }",
        "",
        "    /// Create a new custom dictionary from a vector of codes.",
        "    #[must_use]",
        "    pub fn new_custom(",
        "        name: String,",
        "        dimension: usize,",
        "        hamming_distance: usize,",
        "        codes: Vec<u64>,",
        "        sample_points: Vec<(f64, f64)>,",
        "    ) -> Self {",
        "        let mut code_to_id = HashMap::with_capacity(codes.len());",
        "        for (id, &code) in codes.iter().enumerate() {",
        "            code_to_id.insert(code, id as u16);",
        "        }",
        "        Self {",
        "            name: Cow::Owned(name),",
        "            dimension,",
        "            hamming_distance,",
        "            sample_points: Cow::Owned(sample_points),",
        "            codes: Cow::Owned(codes),",
        "            code_to_id,",
        "        }",
        "    }",
        "",
        "    /// Get number of codes in dictionary.",
        "    #[must_use]",
        "    pub fn len(&self) -> usize {",
        "        self.codes.len()",
        "    }",
        "",
        "    /// Check if dictionary is empty.",
        "    #[must_use]",
        "    pub fn is_empty(&self) -> bool {",
        "        self.codes.is_empty()",
        "    }",
        "",
        "    /// Get the raw code for a given ID.",
        "    #[must_use]",
        "    pub fn get_code(&self, id: u16) -> Option<u64> {",
        "        self.codes.get(id as usize).copied()",
        "    }",
        "",
        "    /// Decode bits, trying all 4 rotations.",
        "    /// Returns (id, hamming_distance) if found within tolerance.",
        "    #[must_use]",
        "    pub fn decode(&self, bits: u64, max_hamming: u32) -> Option<(u16, u32)> {",
        "        let mask = if self.dimension * self.dimension <= 64 {",
        "            (1u64 << (self.dimension * self.dimension)) - 1",
        "        } else {",
        "            u64::MAX",
        "        };",
        "        let bits = bits & mask;",
        "",
        "        // Try exact match first (covers ~60% of clean reads)",
        "        let mut rbits = bits;",
        "        for _ in 0..4 {",
        "            if let Some(&id) = self.code_to_id.get(&rbits) {",
        "                return Some((id, 0));",
        "            }",
        "            if self.sample_points.len() == self.dimension * self.dimension {",
        "                 rbits = rotate90(rbits, self.dimension);",
        "            } else {",
        "                 break;",
        "            }",
        "        }",
        "",
        "        if max_hamming > 0 {",
        "            let mut best: Option<(u16, u32)> = None;",
        "            for (id, &code) in self.codes.iter().enumerate() {",
        "                let mut rbits = bits;",
        "                for _ in 0..4 {",
        "                    let hamming = (rbits ^ code).count_ones();",
        "                    if hamming <= max_hamming {",
        "                        if best.map_or(true, |(_, h)| hamming < h) {",
        "                            best = Some((id as u16, hamming));",
        "                        }",
        "                    }",
        "                    if self.sample_points.len() == self.dimension * self.dimension {",
        "                        rbits = rotate90(rbits, self.dimension);",
        "                    } else {",
        "                        break;",
        "                    }",
        "                }",
        "            }",
        "            return best;",
        "        }",
        "        None",
        "    }",
        "}",
        "",
        "/// Rotates a square bit pattern 90 degrees clockwise.",
        "#[must_use]",
        "pub fn rotate90(bits: u64, dim: usize) -> u64 {",
        "    let mut res = 0u64;",
        "    for y in 0..dim {",
        "        for x in 0..dim {",
        "            if (bits >> (y * dim + x)) & 1 != 0 {",
        "                let nx = dim - 1 - y;",
        "                let ny = x;",
        "                res |= 1 << (ny * dim + nx);",
        "            }",
        "        }",
        "    }",
        "    res",
        "}",
        "",
        "// ============================================================================",
        "// AprilTag 36h11 (587 codes)",
        "// ============================================================================",
        "",
        "#[rustfmt::skip]",
        generate_point_array("APRILTAG_36H11_POINTS", points_36h11),
        "",
        "/// AprilTag 36h11 code table (587 entries, row-major bit ordering).",
        "#[rustfmt::skip]",
        generate_rust_array(
            "APRILTAG_36H11_CODES", APRILTAG_36H11_CODES_SPIRAL, 6, UMICH_36H11_BIT_ORDER
        ),
        "",
        "/// AprilTag 36h11 dictionary singleton.",
        "pub static APRILTAG_36H11: std::sync::LazyLock<TagDictionary> = std::sync::LazyLock::new(|| {",
        '    TagDictionary::new("36h11", 6, 11, &APRILTAG_36H11_CODES, &APRILTAG_36H11_POINTS)',
        "});",
        "",
        "// ============================================================================",
        "// AprilTag 16h5 (30 codes)",
        "// ============================================================================",
        "",
        "#[rustfmt::skip]",
        generate_point_array("APRILTAG_16H5_POINTS", points_16h5),
        "",
        "/// AprilTag 16h5 code table (30 entries, row-major bit ordering).",
        "#[rustfmt::skip]",
        generate_rust_array(
            "APRILTAG_16H5_CODES", APRILTAG_16H5_CODES_SPIRAL, 4, UMICH_16H5_BIT_ORDER
        ),
        "",
        "/// AprilTag 16h5 dictionary singleton.",
        "pub static APRILTAG_16H5: std::sync::LazyLock<TagDictionary> = std::sync::LazyLock::new(|| {",
        '    TagDictionary::new("16h5", 4, 5, &APRILTAG_16H5_CODES, &APRILTAG_16H5_POINTS)',
        "});",
        "",
        "// ============================================================================",
        "// ArUco 4x4_50 (50 codes)",
        "// ============================================================================",
        "",
        "#[rustfmt::skip]",
        generate_point_array("ARUCO_4X4_POINTS", points_aruco),
        "",
        "/// ArUco 4x4_50 code table (50 entries, row-major bit ordering).",
        "#[rustfmt::skip]",
        generate_rust_array_raw("ARUCO_4X4_50_CODES", ARUCO_4X4_50_CODES),
        "",
        "/// ArUco 4x4_50 dictionary singleton.",
        "pub static ARUCO_4X4_50: std::sync::LazyLock<TagDictionary> = std::sync::LazyLock::new(|| {",
        '    TagDictionary::new("4X4_50", 4, 1, &ARUCO_4X4_50_CODES, &ARUCO_4X4_POINTS)',
        "});",
        "",
        "// ============================================================================",
        "// ArUco 4x4_100 (100 codes)",
        "// ============================================================================",
        "",
        "/// ArUco 4x4_100 code table (100 entries, row-major bit ordering).",
        "#[rustfmt::skip]",
        generate_rust_array_raw("ARUCO_4X4_100_CODES", ARUCO_4X4_100_CODES),
        "",
        "/// ArUco 4x4_100 dictionary singleton.",
        "pub static ARUCO_4X4_100: std::sync::LazyLock<TagDictionary> = std::sync::LazyLock::new(|| {",
        '    TagDictionary::new("4X4_100", 4, 1, &ARUCO_4X4_100_CODES, &ARUCO_4X4_POINTS)',
        "});",
    ]
    
    if apriltag_41h12_data:
        points_41h12 = generate_sparse_points(apriltag_41h12_data.UMICH_41H12_BIT_ORDER, 9)
        parts.extend([
            "",
            "// ============================================================================",
            "// AprilTag 41h12",
            "// ============================================================================",
            "",
            "#[rustfmt::skip]",
            generate_point_array("APRILTAG_41H12_POINTS", points_41h12),
            "",
            "/// AprilTag 41h12 code table.",
            "#[rustfmt::skip]",
            generate_rust_array_raw("APRILTAG_41H12_CODES", apriltag_41h12_data.APRILTAG_41H12_CODES_SPIRAL),
            "",
            "/// AprilTag 41h12 dictionary singleton.",
            "pub static APRILTAG_41H12: std::sync::LazyLock<TagDictionary> = std::sync::LazyLock::new(|| {",
            '    TagDictionary::new("41h12", 9, 12, &APRILTAG_41H12_CODES, &APRILTAG_41H12_POINTS)',
            "});",
        ])

    parts.append("")
    return "\\n".join(parts)
"""

# Regex to find definition of generate_dictionaries_rs and replace until main
content = re.sub(
    r"def generate_dictionaries_rs\(\) -> str:.*?(?=def main\(\):)",
    new_func,
    content,
    flags=re.DOTALL,
)

target_file.write_text(content)
print(f"Patched {target_file}")
