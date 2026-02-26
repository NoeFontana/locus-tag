#![allow(clippy::unreadable_literal)]
//! Tag family dictionaries.
//!
//! This module contains pre-generated code tables for AprilTag and ArUco families.
//!
//! # Bit Ordering Convention
//!
//! All codes use **row-major bit ordering** compatible with OpenCV's `cv2.aruco` module:
//! - Bit 0 = top-left of the data grid
//! - Bit N = (row * dim + col) for row-major traversal
//! - Rotation detection handles all 4 orientations automatically
//!
//! # Source
//!
//! Codes are extracted directly from OpenCV's predefined dictionaries to ensure
//! compatibility with `cv2.aruco.generateImageMarker()`.
//!
use std::borrow::Cow;
use std::collections::HashMap;

/// A tag family dictionary.
#[derive(Clone, Debug)]
pub struct TagDictionary {
    /// Name of the tag family.
    pub name: Cow<'static, str>,
    /// Grid dimension (e.g., 6 for 36h11).
    pub dimension: usize,
    /// Minimum hamming distance of the family.
    pub hamming_distance: usize,
    /// Pre-computed sample points in canonical tag coordinates [-1, 1].
    pub sample_points: Cow<'static, [(f64, f64)]>,
    /// Raw code table.
    pub codes: Cow<'static, [u64]>,
    /// Lookup table for O(1) exact matching. Maps bits to (ID, rotation_count).
    code_to_id: HashMap<u64, (u16, u8)>,
    /// All 4 rotated versions of all codes (for Hamming search). Stores (bits, ID, rotation_count).
    rotated_codes: Vec<(u64, u16, u8)>,
}

impl TagDictionary {
    /// Create a new dictionary from static code table.
    #[must_use]
    pub fn new(
        name: &'static str,
        dimension: usize,
        hamming_distance: usize,
        codes: &'static [u64],
        sample_points: &'static [(f64, f64)],
    ) -> Self {
        let mut code_to_id = HashMap::with_capacity(codes.len() * 4);
        let mut rotated_codes = Vec::with_capacity(codes.len() * 4);
        for (id, &code) in codes.iter().enumerate() {
            let mut r = code;
            for rot in 0u8..4 {
                code_to_id.insert(r, (id as u16, rot));
                rotated_codes.push((r, id as u16, rot));
                r = rotate90(r, dimension);
            }
        }
        Self {
            name: Cow::Borrowed(name),
            dimension,
            hamming_distance,
            sample_points: Cow::Borrowed(sample_points),
            codes: Cow::Borrowed(codes),
            code_to_id,
            rotated_codes,
        }
    }

    /// Create a new custom dictionary from a vector of codes.
    #[must_use]
    pub fn new_custom(
        name: String,
        dimension: usize,
        hamming_distance: usize,
        codes: Vec<u64>,
        sample_points: Vec<(f64, f64)>,
    ) -> Self {
        let mut code_to_id = HashMap::with_capacity(codes.len() * 4);
        let mut rotated_codes = Vec::with_capacity(codes.len() * 4);
        for (id, &code) in codes.iter().enumerate() {
            let mut r = code;
            for rot in 0u8..4 {
                code_to_id.insert(r, (id as u16, rot));
                rotated_codes.push((r, id as u16, rot));
                r = rotate90(r, dimension);
            }
        }
        Self {
            name: Cow::Owned(name),
            dimension,
            hamming_distance,
            sample_points: Cow::Owned(sample_points),
            codes: Cow::Owned(codes),
            code_to_id,
            rotated_codes,
        }
    }

    /// Get number of codes in dictionary.
    #[must_use]
    pub fn len(&self) -> usize {
        self.codes.len()
    }

    /// Check if dictionary is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.codes.is_empty()
    }

    /// Get the raw code for a given ID.
    #[must_use]
    pub fn get_code(&self, id: u16) -> Option<u64> {
        self.codes.get(id as usize).copied()
    }

    /// Get all rotated versions of all codes.
    #[must_use]
    pub fn rotated_codes(&self) -> &[(u64, u16, u8)] {
        &self.rotated_codes
    }

    /// Decode bits, trying all 4 rotations.
    /// Returns (id, hamming_distance, rotation_count) if found within tolerance.
    /// rotation_count is 0-3, representing 90-degree CW increments.
    #[must_use]
    pub fn decode(&self, bits: u64, max_hamming: u32) -> Option<(u16, u32, u8)> {
        let mask = if self.dimension * self.dimension <= 64 {
            (1u64 << (self.dimension * self.dimension)) - 1
        } else {
            u64::MAX
        };
        let bits = bits & mask;

        // Try exact match first (covers ~60% of clean reads) - now O(1) with all rotations in map
        if let Some(&(id, rot)) = self.code_to_id.get(&bits) {
            return Some((id, 0, rot));
        }

        if max_hamming > 0 {
            let mut best: Option<(u16, u32, u8)> = None;
            for &(code_rot, id, rot) in &self.rotated_codes {
                let hamming = (bits ^ code_rot).count_ones();
                if hamming <= max_hamming && best.as_ref().is_none_or(|&(_, h, _)| hamming < h) {
                    best = Some((id, hamming, rot));
                    if hamming == 0 {
                        break;
                    }
                }
            }
            return best;
        }
        None
    }
}

/// Rotates a square bit pattern 90 degrees clockwise.
#[must_use]
pub fn rotate90(bits: u64, dim: usize) -> u64 {
    let mut res = 0u64;
    for y in 0..dim {
        for x in 0..dim {
            if (bits >> (y * dim + x)) & 1 != 0 {
                let nx = dim - 1 - y;
                let ny = x;
                res |= 1 << (ny * dim + nx);
            }
        }
    }
    res
}

// ============================================================================
// AprilTag 36h11 (587 codes)
// ============================================================================

/// Sample points for AprilTag 36h11 in canonical coordinates [-1, 1].
/// Row-major order: bit 0 is top-left, scanning left-to-right, top-to-bottom.
#[rustfmt::skip]
pub static APRILTAG_36H11_POINTS: [(f64, f64); 36] = [
    (-0.625000, -0.625000), (-0.375000, -0.625000),
    (-0.125000, -0.625000), (0.125000, -0.625000),
    (0.375000, -0.625000), (0.625000, -0.625000),
    (-0.625000, -0.375000), (-0.375000, -0.375000),
    (-0.125000, -0.375000), (0.125000, -0.375000),
    (0.375000, -0.375000), (0.625000, -0.375000),
    (-0.625000, -0.125000), (-0.375000, -0.125000),
    (-0.125000, -0.125000), (0.125000, -0.125000),
    (0.375000, -0.125000), (0.625000, -0.125000),
    (-0.625000, 0.125000), (-0.375000, 0.125000),
    (-0.125000, 0.125000), (0.125000, 0.125000),
    (0.375000, 0.125000), (0.625000, 0.125000),
    (-0.625000, 0.375000), (-0.375000, 0.375000),
    (-0.125000, 0.375000), (0.125000, 0.375000),
    (0.375000, 0.375000), (0.625000, 0.375000),
    (-0.625000, 0.625000), (-0.375000, 0.625000),
    (-0.125000, 0.625000), (0.125000, 0.625000),
    (0.375000, 0.625000), (0.625000, 0.625000),
];

/// AprilTag 36h11 code table (587 codes, row-major from OpenCV).
#[rustfmt::skip]
#[allow(clippy::unreadable_literal)]
pub static APRILTAG_36H11_CODES: [u64; 587] = [
    0x000d5d628584, 0x000d97f18b49, 0x000dd280910e, 0x000e479e9c98,
    0x000ebcbca822, 0x000f31dab3ac, 0x000056a5d085, 0x00010652e1d4,
    0x00022b1dfead, 0x000265ad0472, 0x00034fe91b86, 0x0003ff962cd5,
    0x00043a25329a, 0x000474b4385f, 0x0004e9d243e9, 0x0005246149ae,
    0x0005997f5538, 0x000683bb6c4c, 0x0006be4a7211, 0x0007e3158eea,
    0x00081da494af, 0x000858339a74, 0x0008cd51a5fe, 0x0009f21cc2d7,
    0x000a2cabc89c, 0x000adc58d9eb, 0x000b16e7dfb0, 0x000b8c05eb3a,
    0x000d25ef139d, 0x000d607e1962, 0x000e4aba3076, 0x0002dde6a3da,
    0x00043d40c678, 0x0005620be351, 0x00064c47fa65, 0x000686d7002a,
    0x0006c16605ef, 0x0006fbf50bb4, 0x0008d06d39dc, 0x0009f53856b5,
    0x000adf746dc9, 0x000bc9b084dd, 0x000d290aa77b, 0x000d9e28b305,
    0x000e4dd5c454, 0x000fad2fe6f2, 0x000181a8151a, 0x00026be42c2e,
    0x0002e10237b8, 0x000405cd5491, 0x0007742eab1c, 0x00085e6ac230,
    0x0008d388cdba, 0x0009f853ea93, 0x000c41ea2445, 0x000cf1973594,
    0x00014a34a333, 0x00031eacd15b, 0x0006c79d2dab, 0x00073cbb3935,
    0x00089c155bd3, 0x0008d6a46198, 0x00091133675d, 0x000a708d89fb,
    0x000ae5ab9585, 0x000b9558a6d4, 0x000b98743ab2, 0x000d6cec68da,
    0x0001506bcaef, 0x0004becd217a, 0x0004f95c273f, 0x000658b649dd,
    0x000a76c4b1b7, 0x000ecf621f56, 0x0001c8a56a57, 0x0003628e92ba,
    0x00053706c0e2, 0x0005e6b3d231, 0x0007809cfa94, 0x000e97eead6f,
    0x0005af40604a, 0x0007492988ad, 0x000ed5994712, 0x0005eceaf9ed,
    0x0007c1632815, 0x000c1a0095b4, 0x000e9e25d52b, 0x0003a6705419,
    0x000a8333012f, 0x0004ce5704d0, 0x000508e60a95, 0x000877476120,
    0x000a864e950d, 0x000ea45cfce7, 0x00019da047e8, 0x00024d4d5937,
    0x0006e079cc9b, 0x00099f2e11d7, 0x00033aa50429, 0x000499ff26c7,
    0x00050f1d3251, 0x00066e7754ef, 0x00096ad633ce, 0x0009a5653993,
    0x000aca30566c, 0x000c298a790a, 0x0008be44b65d, 0x000dc68f354b,
    0x00016f7f919b, 0x0004dde0e826, 0x000d548cbd9f, 0x000e0439ceee,
    0x000fd8b1fd16, 0x00076521bb7b, 0x000d92375742, 0x000cab16d40c,
    0x000730c9dd72, 0x000ad9ba39c2, 0x000b14493f87, 0x00052b15651f,
    0x000185409cad, 0x00077ae2c68d, 0x00094f5af4b5, 0x0000a13bad55,
    0x00061ea437cd, 0x000a022399e2, 0x000203b163d1, 0x0007bba8f40e,
    0x00095bc9442d, 0x00041c0b5358, 0x0008e9c6cc81, 0x0000eb549670,
    0x0009da3a0b51, 0x000d832a67a1, 0x000dcd4350bc, 0x0004aa05fdd2,
    0x00060c7bb44e, 0x0004b358b96c, 0x000067299b45, 0x000b9c89b5fa,
    0x0006975acaea, 0x00062b8f7afa, 0x00033567c3d7, 0x000bac139950,
    0x000a5927c62a, 0x0005c916e6a4, 0x000260ecb7d5, 0x00029b7bbd9a,
    0x000903205f26, 0x000ae72270a4, 0x0003d2ec51a7, 0x00082ea55324,
    0x00011a6f3427, 0x0001ca1c4576, 0x000a40c81aef, 0x000bddccd730,
    0x0000e617561e, 0x000969317b0f, 0x00067f781364, 0x000610912f96,
    0x000b2549fdfc, 0x00006e5aaa6b, 0x000b6c475339, 0x000c56836a4d,
    0x000844e351eb, 0x0004647f83b4, 0x0000908a04f5, 0x0007f51034c9,
    0x000aee537fca, 0x0005e92494ba, 0x000d445808f4, 0x00028d68b563,
    0x00004d25374b, 0x0002bc065f65, 0x00096dc3ea0c, 0x0004b2ade817,
    0x00007c3fd502, 0x000e768b5caf, 0x00017605cf6c, 0x000182741ee4,
    0x00062846097c, 0x00072b5ebf80, 0x000263da6e13, 0x000fa841bcb5,
    0x0007e45e8c69, 0x000653c81fa0, 0x0007443b5e70, 0x0000a5234afd,
    0x00074756f24e, 0x000157ebf02a, 0x00082ef46939, 0x00080d420264,
    0x0002aeed3e98, 0x000b0a1dd4f8, 0x000b5436be13, 0x0007b7b4b13b,
    0x0001ce80d6d3, 0x00016c08427d, 0x000ee54462dd, 0x0001f7644cce,
    0x0009c7b5cc92, 0x000e369138f8, 0x0005d5a66e91, 0x000485d62f49,
    0x000e6e819e94, 0x000b1f340eb5, 0x00009d198ce2, 0x000d60717437,
    0x0000196b856c, 0x000f0a6173a5, 0x00012c0e1ec6, 0x00062b82d5cf,
    0x000ad154c067, 0x000ce3778832, 0x0006b0a7b864, 0x0004c7686694,
    0x0005058ff3ec, 0x000d5e21ea23, 0x0009ff4a76ee, 0x0009dd981019,
    0x0001bad4d30a, 0x000c601896d1, 0x000973439b48, 0x0001ce7431a8,
    0x00057a8021d6, 0x000f9dba96e6, 0x00083a2e4e7c, 0x0008ea585380,
    0x000af6c0e744, 0x000875b73bab, 0x000da34ca901, 0x0002ab9727ef,
    0x000d39f21b9a, 0x0008a10b742f, 0x0005f8952dba, 0x000f8da71ab0,
    0x000c25f9df96, 0x00006f8a5d94, 0x000e42e63e1a, 0x000b78409d1b,
    0x000792229add, 0x0005acf8c455, 0x0002fc29a9b0, 0x000ea486237b,
    0x000b0c9685a0, 0x0001ad748a47, 0x00003b4712d5, 0x000f29216d30,
    0x0008dad65e49, 0x0000a2cf09dd, 0x0000b5f174c6, 0x000e54f57743,
    0x000b9cf54d78, 0x0004a312a88a, 0x00027babc962, 0x000b86897111,
    0x000f2ff6c116, 0x00082274bd8a, 0x00097023505e, 0x00052d46edd1,
    0x000585c1f538, 0x000bddd00e43, 0x0005590b74df, 0x000729404a1f,
    0x00065320855e, 0x000d3d4b6956, 0x0007ae374f14, 0x0002d7a60e06,
    0x000315cd9b5e, 0x000fd36b4eac, 0x000f1df7642b, 0x00055db27726,
    0x0008f15ebc19, 0x000992f8c531, 0x00062dea2a40, 0x000928275cab,
    0x00069c263cb9, 0x000a774cca9e, 0x000266b2110e, 0x0001b14acbb8,
    0x000624b8a71b, 0x0001c539406b, 0x0003086d529b, 0x0000111dd66e,
    0x00098cd630bf, 0x0008b9d1ffdc, 0x00072b2f61e7, 0x0009ed9d672b,
    0x00096cdd15f3, 0x0006366c2504, 0x0006ca9df73a, 0x000a066d60f0,
    0x000e7a4b8add, 0x0008264647ef, 0x000aa195bf81, 0x0009a3db8244,
    0x000014d2df6a, 0x0000b63265b7, 0x0002f010de73, 0x00097e774986,
    0x000248affc29, 0x000fb57dcd11, 0x0000b1a7e4d9, 0x0004bfa2d07d,
    0x00054e5cdf96, 0x0004c15c1c86, 0x000cd9c61166, 0x000499380b2a,
    0x000540308d09, 0x0008b63fe66f, 0x000c81aeb35e, 0x00086fe0bd5c,
    0x000ce2480c2a, 0x0001ab29ee60, 0x0008048daa15, 0x000dbfeb2d39,
    0x000567c9858c, 0x0002b6edc5bc, 0x0002078fca82, 0x000adacc22aa,
    0x000b92486f49, 0x00051fac5964, 0x000691ee6420, 0x000f63b3e129,
    0x00039be7e572, 0x000da2ce6c74, 0x00020cf17a5c, 0x000ee55f9b6e,
    0x000fb8572726, 0x000b2c2de548, 0x000caa9bce92, 0x000ae9182db3,
    0x00074b6e5bd1, 0x000137b252af, 0x00051f686881, 0x000d672f6c02,
    0x000654146ce4, 0x000f944bc825, 0x000e8327f809, 0x00076a73fd59,
    0x000f79da4cb4, 0x000956f8099b, 0x0007b5f2655c, 0x000d06b114a6,
    0x000d0697ca50, 0x00027c390797, 0x000bc61ed9b2, 0x000cc12dd19b,
    0x000eb7818d2c, 0x000092fcecda, 0x00089ded4ea1, 0x000256a0ba34,
    0x000b6948e627, 0x0001ef6b1054, 0x0008639294a2, 0x000eda3780a4,
    0x00039ee2af1d, 0x000cd257edc5, 0x0002d9d6bc22, 0x000121d3b47d,
    0x00037e23f8ad, 0x000119f31cf6, 0x0002c97f4f09, 0x000d502abfe0,
    0x00010bc3ca77, 0x00053d7190ef, 0x00090c3e62a6, 0x0007e9ebf675,
    0x000979ce23d1, 0x00027f0c98e9, 0x000eafb4ae59, 0x0007ca7fe2bd,
    0x0001490ca8f6, 0x0009123387ba, 0x000b3bc73888, 0x0003ea87e325,
    0x0004888964aa, 0x000a0188a6b9, 0x000cd383c666, 0x00040029a3fd,
    0x000e1c00ac5c, 0x00039e6f2b6e, 0x000de664f622, 0x000e979a75e8,
    0x0007c6b4c86c, 0x000fd492e071, 0x0008fbb35118, 0x00040b4a09b7,
    0x000af80bd6da, 0x00070e0b2521, 0x0002f5c54d93, 0x0003f4a118d5,
    0x00009c1897b9, 0x000079776eac, 0x000084b00b17, 0x0003a95ad90e,
    0x00028c544095, 0x00039d457c05, 0x0007a3791a78, 0x000bb770e22e,
    0x0009a822bd6c, 0x00068a4b1fed, 0x000a5fd27b3b, 0x0000c3995b79,
    0x000d1519dff1, 0x0008e7eee359, 0x000cd3ca50b1, 0x000b73b8b793,
    0x00057aca1c43, 0x000ec2655277, 0x000785a2c1b3, 0x00075a07985a,
    0x000a4b01eb69, 0x000a18a11347, 0x000db1f28ca3, 0x000877ec3e25,
    0x00031f6341b8, 0x0001363a3a4c, 0x000075d8b9ba, 0x0007ae0792a9,
    0x000a83a21651, 0x0007f08f9fb5, 0x0000d0cf73a9, 0x000b04dcc98e,
    0x000f65c7b0f8, 0x00065ddaf69a, 0x0002cf9b86b3, 0x00014cb51e25,
    0x000f48027b5b, 0x0000ec26ea8b, 0x00044bafd45c, 0x000b12c7c0c4,
    0x000959fd9d82, 0x000c77c9725a, 0x00048a22d462, 0x0008398e8072,
    0x000ec89b05ce, 0x000bb682d4c9, 0x000e5a86d2ff, 0x000358f01134,
    0x0008556ddcf6, 0x00067584b6e2, 0x00011609439f, 0x00008488816e,
    0x000aaf1a2c46, 0x000f879898cf, 0x0008bbe5e2f7, 0x000101eee363,
    0x000690f69377, 0x000f5bd93cd9, 0x000cea4c2bf6, 0x0009550be706,
    0x0002c5b38a60, 0x000e72033547, 0x0004458b0629, 0x000ee8d9ed41,
    0x000d2f918d72, 0x00078dc39fd3, 0x0008212636f6, 0x0007450a72a7,
    0x000c4f0cf4c6, 0x000367bcddcd, 0x000c1caf8cc6, 0x000a7f5b853d,
    0x0009d536818b, 0x000535e021b0, 0x000a7eb8729e, 0x000422a67b49,
    0x000929e928a6, 0x00048e8aefcc, 0x000a9897393c, 0x0005eb81d37e,
    0x0001e80287b7, 0x00034770d903, 0x0002eef86728, 0x00059266ccb6,
    0x0000110bba61, 0x0001dfd284ef, 0x000447439d1b, 0x000fece0e599,
    0x0009309f3703, 0x00080764d1dd, 0x000353f1e6a0, 0x0002c1c12dcc,
    0x000c1d21b9d7, 0x000457ee453e, 0x000d66faf540, 0x00044831e652,
    0x000cfd49a848, 0x0009312d4133, 0x0003f097d3ee, 0x0008c9ebef7a,
    0x000a99e29e88, 0x0000e9fab22c, 0x0004e748f4fb, 0x000ecdee4288,
    0x000abce5f1d0, 0x000c42f6876c, 0x0007ed402ea0, 0x000e5c4242c3,
    0x000d5b2c31ae, 0x000286863be6, 0x000160444d94, 0x0005f0f5808e,
    0x000ae3d44b2a, 0x0009f5c5d109, 0x0008ad9316d7, 0x0003422ba064,
    0x0002fed11d56, 0x000bea6e3e04, 0x00004b029eec, 0x0006deed7435,
    0x0003718ce17c, 0x00055857f5e2, 0x0002edac7b62, 0x000085d6c512,
    0x000d6ca88e0f, 0x0002b7e1fc69, 0x000a699d5c1b, 0x000f05ad74de,
    0x0004cf5fb56d, 0x0005725e07e1, 0x00072f18a2de, 0x0001cec52609,
    0x00048534243c, 0x0002523a4d69, 0x00035c1b80d1, 0x000a4d7338a7,
    0x0000db1af012, 0x000e61a9475d, 0x00005df03f91, 0x00097ae260bb,
    0x00032d627fef, 0x000b640f73c2, 0x00045a1ac9c6, 0x0006a2202de1,
    0x00057d3e25f2, 0x0005aa9f986e, 0x0000cc859d8a, 0x000e3ec6cca8,
    0x00054e95e1ae, 0x000446887b06, 0x0007516732be, 0x0003817ac8f5,
    0x0003e26d938c, 0x000aa81bc235, 0x000df387ca1b, 0x0000f3a3b3f2,
    0x000b4bf69677, 0x000ae21868ed, 0x00081e1d2d9d, 0x000a0a9ea14c,
    0x0008eee297a9, 0x0004740c0559, 0x000e8b141837, 0x000ac69e0a3d,
    0x0009ed83a1e1, 0x0005edb55ecb, 0x00007340fe81, 0x00050dfbc6bf,
    0x0004f583508a, 0x000cb1fb78bc, 0x0004025ced2f, 0x00039791ebec,
    0x00053ee388f1, 0x0007d6c0bd23, 0x00093a995fbe, 0x0008a41728de,
    0x0002fe70e053, 0x000ab3db443a, 0x0001364edb05, 0x00047b6eeed6,
    0x00012e71af01, 0x00052ff83587, 0x0003a1575dd8, 0x0003feaa3564,
    0x000eacf78ba7, 0x0000872b94f8, 0x000da8ddf9a2, 0x0009aa920d2b,
    0x0001f350ed36, 0x00018a5e861f, 0x0002c35b89c3, 0x0003347ac48a,
    0x0007f23e022e, 0x0002459068fb, 0x000e83be4b73,
];

/// AprilTag 36h11 dictionary singleton.
pub static APRILTAG_36H11: std::sync::LazyLock<TagDictionary> = std::sync::LazyLock::new(|| {
    TagDictionary::new(
        "36h11",
        6,
        11,
        &APRILTAG_36H11_CODES,
        &APRILTAG_36H11_POINTS,
    )
});

// ArUco 36h11
pub static ARUCO_36H11_CODES: [u64; 587] = [
    0x000d_5d62_8584, 0x000d_97f1_8b49, 0x000d_d280_910e, 0x000e_479e_9c98,
    0x000e_bcbc_a822, 0x000f_31da_b3ac, 0x0000_56a5_d085, 0x0001_0652_e1d4,
    0x0002_2b1d_fead, 0x0002_65ad_0472, 0x0003_4fe9_1b86, 0x0003_ff96_2cd5,
    0x0004_3a25_329a, 0x0004_74b4_385f, 0x0004_e9d2_43e9, 0x0005_2461_49ae,
    0x0005_997f_5538, 0x0006_83bb_6c4c, 0x0006_be4a_7211, 0x0007_e315_8eea,
    0x0008_1da4_94af, 0x0008_5833_9a74, 0x0008_cd51_a5fe, 0x0009_f21c_c2d7,
    0x000a_2cab_c89c, 0x000a_dc58_d9eb, 0x000b_16e7_dfb0, 0x000b_8c05_eb3a,
    0x000d_25ef_139d, 0x000d_607e_1962, 0x000e_4aba_3076, 0x0002_dde6_a3da,
    0x0004_3d40_c678, 0x0005_620b_e351, 0x0006_4c47_fa65, 0x0006_86d7_002a,
    0x0006_c166_05ef, 0x0006_fbf5_0bb4, 0x0008_d06d_39dc, 0x0009_f538_56b5,
    0x000a_df74_6dc9, 0x000b_c9b0_84dd, 0x000d_290a_a77b, 0x000d_9e28_b305,
    0x000e_4dd5_c454, 0x000f_ad2f_e6f2, 0x0001_81a8_151a, 0x0002_6be4_2c2e,
    0x0002_e102_37b8, 0x0004_05cd_5491, 0x0007_742e_ab1c, 0x0008_5e6a_c230,
    0x0008_d388_cdba, 0x0009_f853_ea93, 0x000c_41ea_2445, 0x000c_f197_3594,
    0x0001_4a34_a333, 0x0003_1eac_d15b, 0x0006_c79d_2dab, 0x0007_3cbb_3935,
    0x0008_9c15_5bd3, 0x0008_d6a4_6198, 0x0009_1133_675d, 0x000a_708d_89fb,
    0x000a_e5ab_9585, 0x000b_9558_a6d4, 0x000b_9874_3ab2, 0x000d_6cec_68da,
    0x0001_506b_caef, 0x0004_becd_217a, 0x0004_f95c_273f, 0x0006_58b6_49dd,
    0x000a_76c4_b1b7, 0x000e_cf62_1f56, 0x0001_c8a5_6a57, 0x0003_628e_92ba,
    0x0005_3706_c0e2, 0x0005_e6b3_d231, 0x0007_809c_fa94, 0x000e_97ee_ad6f,
    0x0005_af40_604a, 0x0007_4929_88ad, 0x000e_d599_4712, 0x0005_ecea_f9ed,
    0x0007_c163_2815, 0x000c_1a00_95b4, 0x000e_9e25_d52b, 0x0003_a670_5419,
    0x000a_8333_012f, 0x0004_ce57_04d0, 0x0005_08e6_0a95, 0x0008_7747_6120,
    0x000a_864e_950d, 0x000e_a45c_fce7, 0x0001_9da0_47e8, 0x0002_4d4d_5937,
    0x0006_e079_cc9b, 0x0009_9f2e_11d7, 0x0003_3aa5_0429, 0x0004_99ff_26c7,
    0x0005_0f1d_3251, 0x0006_6e77_54ef, 0x0009_6ad6_33ce, 0x0009_a565_3993,
    0x000a_ca30_566c, 0x000c_298a_790a, 0x0008_be44_b65d, 0x000d_c68f_354b,
    0x0001_6f7f_919b, 0x0004_dde0_e826, 0x000d_548c_bd9f, 0x000e_0439_ceee,
    0x000f_d8b1_fd16, 0x0007_6521_bb7b, 0x000d_9237_5742, 0x000c_ab16_d40c,
    0x0007_30c9_dd72, 0x000a_d9ba_39c2, 0x000b_1449_3f87, 0x0005_2b15_651f,
    0x0001_8540_9cad, 0x0007_7ae2_c68d, 0x0009_4f5a_f4b5, 0x0000_a13b_ad55,
    0x0006_1ea4_37cd, 0x000a_0223_99e2, 0x0002_03b1_63d1, 0x0007_bba8_f40e,
    0x0009_5bc9_442d, 0x0004_1c0b_5358, 0x0008_e9c6_cc81, 0x0000_eb54_9670,
    0x0009_da3a_0b51, 0x000d_832a_67a1, 0x000d_cd43_50bc, 0x0004_aa05_fdd2,
    0x0006_0c7b_b44e, 0x0004_b358_b96c, 0x0000_6729_9b45, 0x000b_9c89_b5fa,
    0x0006_975a_caea, 0x0006_2b8f_7afa, 0x0003_3567_c3d7, 0x000b_ac13_9950,
    0x000a_5927_c62a, 0x0005_c916_e6a4, 0x0002_60ec_b7d5, 0x0002_9b7b_bd9a,
    0x0009_0320_5f26, 0x000a_e722_70a4, 0x0003_d2ec_51a7, 0x0008_2ea5_5324,
    0x0001_1a6f_3427, 0x0001_ca1c_4576, 0x000a_40c8_1aef, 0x000b_ddcc_d730,
    0x0000_e617_561e, 0x0009_6931_7b0f, 0x0006_7f78_1364, 0x0006_1091_2f96,
    0x000b_2549_fdfc, 0x0000_6e5a_aa6b, 0x000b_6c47_5339, 0x000c_5683_6a4d,
    0x0008_44e3_51eb, 0x0004_647f_83b4, 0x0000_908a_04f5, 0x0007_f510_34c9,
    0x000a_ee53_7fca, 0x0005_e924_94ba, 0x000d_4458_08f4, 0x0002_8d68_b563,
    0x0000_4d25_374b, 0x0002_bc06_5f65, 0x0009_6dc3_ea0c, 0x0004_b2ad_e817,
    0x0000_7c3f_d502, 0x000e_768b_5caf, 0x0001_7605_cf6c, 0x0001_8274_1ee4,
    0x0006_2846_097c, 0x0007_2b5e_bf80, 0x0002_63da_6e13, 0x000f_a841_bcb5,
    0x0007_e45e_8c69, 0x0006_53c8_1fa0, 0x0007_443b_5e70, 0x0000_a523_4afd,
    0x0007_4756_f24e, 0x0001_57eb_f02a, 0x0008_2ef4_6939, 0x0008_0d42_0264,
    0x0002_aeed_3e98, 0x000b_0a1d_d4f8, 0x000b_5436_be13, 0x0007_b7b4_b13b,
    0x0001_ce80_d6d3, 0x0001_6c08_427d, 0x000e_e544_62dd, 0x0001_f764_4cce,
    0x0009_c7b5_cc92, 0x000e_3691_38f8, 0x0005_d5a6_6e91, 0x0004_85d6_2f49,
    0x000e_6e81_9e94, 0x000b_1f34_0eb5, 0x0000_9d19_8ce2, 0x000d_6071_7437,
    0x0000_196b_856c, 0x000f_0a61_73a5, 0x0001_2c0e_1ec6, 0x0006_2b82_d5cf,
    0x000a_d154_c067, 0x000c_e377_8832, 0x0006_b0a7_b864, 0x0004_c768_6694,
    0x0005_058f_f3ec, 0x000d_5e21_ea23, 0x0009_ff4a_76ee, 0x0009_dd98_1019,
    0x0001_bad4_d30a, 0x000c_6018_96d1, 0x0009_7343_9b48, 0x0001_ce74_31a8,
    0x0005_7a80_21d6, 0x000f_9dba_96e6, 0x0008_3a2e_4e7c, 0x0008_ea58_5380,
    0x000a_f6c0_e744, 0x0008_75b7_3bab, 0x000d_a34c_a901, 0x0002_ab97_27ef,
    0x000d_39f2_1b9a, 0x0008_a10b_742f, 0x0005_f895_2dba, 0x000f_8da7_1ab0,
    0x000c_25f9_df96, 0x0000_6f8a_5d94, 0x000e_42e6_3e1a, 0x000b_7840_9d1b,
    0x0007_9222_9add, 0x0005_acf8_c455, 0x0002_fc29_a9b0, 0x000e_a486_237b,
    0x000b_0c96_85a0, 0x0001_ad74_8a47, 0x0000_3b47_12d5, 0x000f_2921_6d30,
    0x0008_dad6_5e49, 0x0000_a2cf_09dd, 0x0000_b5f1_74c6, 0x000e_54f5_7743,
    0x000b_9cf5_4d78, 0x0004_a312_a88a, 0x0002_7bab_c962, 0x000b_8689_7111,
    0x000f_2ff6_c116, 0x0008_2274_bd8a, 0x0009_7023_505e, 0x0005_2d46_edd1,
    0x0005_85c1_f538, 0x000b_ddd0_0e43, 0x0005_590b_74df, 0x0007_2940_4a1f,
    0x0006_5320_855e, 0x000d_3d4b_6956, 0x0007_ae37_4f14, 0x0002_d7a6_0e06,
    0x0003_15cd_9b5e, 0x000f_d36b_4eac, 0x000f_1df7_642b, 0x0005_5db2_7726,
    0x0008_f15e_bc19, 0x0009_92f8_c531, 0x0006_2dea_2a40, 0x0009_2827_5cab,
    0x0006_9c26_3cb9, 0x000a_774c_ca9e, 0x0002_66b2_110e, 0x0001_b14a_cbb8,
    0x0006_24b8_a71b, 0x0001_c539_406b, 0x0003_086d_529b, 0x0000_111d_d66e,
    0x0009_8cd6_30bf, 0x0008_b9d1_ffdc, 0x0007_2b2f_61e7, 0x0009_ed9d_672b,
    0x0009_6cdd_15f3, 0x0006_366c_2504, 0x0006_ca9d_f73a, 0x000a_066d_60f0,
    0x000e_7a4b_8add, 0x0008_2646_47ef, 0x000a_a195_bf81, 0x0009_a3db_8244,
    0x0000_14d2_df6a, 0x0000_b632_65b7, 0x0002_f010_de73, 0x0009_7e77_4986,
    0x0002_48af_fc29, 0x000f_b57d_cd11, 0x0000_b1a7_e4d9, 0x0004_bfa2_d07d,
    0x0005_4e5c_df96, 0x0004_c15c_1c86, 0x000c_d9c6_1166, 0x0004_9938_0b2a,
    0x0005_4030_8d09, 0x0008_b63f_e66f, 0x000c_81ae_b35e, 0x0008_6fe0_bd5c,
    0x000c_e248_0c2a, 0x0001_ab29_ee60, 0x0008_048d_aa15, 0x000d_bfeb_2d39,
    0x0005_67c9_858c, 0x0002_b6ed_c5bc, 0x0002_078f_ca82, 0x000a_dacc_22aa,
    0x000b_9248_6f49, 0x0005_1fac_5964, 0x0006_91ee_6420, 0x000f_63b3_e129,
    0x0003_9be7_e572, 0x000d_a2ce_6c74, 0x0002_0cf1_7a5c, 0x000e_e55f_9b6e,
    0x000f_b857_2726, 0x000b_2c2d_e548, 0x000c_aa9b_ce92, 0x000a_e918_2db3,
    0x0007_4b6e_5bd1, 0x0001_37b2_52af, 0x0005_1f68_6881, 0x000d_672f_6c02,
    0x0006_5414_6ce4, 0x000f_944b_c825, 0x000e_8327_f809, 0x0007_6a73_fd59,
    0x000f_79da_4cb4, 0x0009_56f8_099b, 0x0007_b5f2_655c, 0x000d_06b1_14a6,
    0x000d_0697_ca50, 0x0002_7c39_0797, 0x000b_c61e_d9b2, 0x000c_c12d_d19b,
    0x000e_b781_8d2c, 0x0000_92fc_ecda, 0x0008_9ded_4ea1, 0x0002_56a0_ba34,
    0x000b_6948_e627, 0x0001_ef6b_1054, 0x0008_6392_94a2, 0x000e_da37_80a4,
    0x0003_9ee2_af1d, 0x000c_d257_edc5, 0x0002_d9d6_bc22, 0x0001_21d3_b47d,
    0x0003_7e23_f8ad, 0x0001_19f3_1cf6, 0x0002_c97f_4f09, 0x000d_502a_bfe0,
    0x0001_0bc3_ca77, 0x0005_3d71_90ef, 0x0009_0c3e_62a6, 0x0007_e9eb_f675,
    0x0009_79ce_23d1, 0x0002_7f0c_98e9, 0x000e_afb4_ae59, 0x0007_ca7f_e2bd,
    0x0001_490c_a8f6, 0x0009_1233_87ba, 0x000b_3bc7_3888, 0x0003_ea87_e325,
    0x0004_8889_64aa, 0x000a_0188_a6b9, 0x000c_d383_c666, 0x0004_0029_a3fd,
    0x000e_1c00_ac5c, 0x0003_9e6f_2b6e, 0x000d_e664_f622, 0x000e_979a_75e8,
    0x0007_c6b4_c86c, 0x000f_d492_e071, 0x0008_fbb3_5118, 0x0004_0b4a_09b7,
    0x000a_f80b_d6da, 0x0007_0e0b_2521, 0x0002_f5c5_4d93, 0x0003_f4a1_18d5,
    0x0000_9c18_97b9, 0x0000_7977_6eac, 0x0000_84b0_0b17, 0x0003_a95a_d90e,
    0x0002_8c54_4095, 0x0003_9d45_7c05, 0x0007_a379_1a78, 0x000b_b770_e22e,
    0x0009_a822_bd6c, 0x0006_8a4b_1fed, 0x000a_5fd2_7b3b, 0x0000_c399_5b79,
    0x000d_1519_dff1, 0x0008_e7ee_e359, 0x000c_d3ca_50b1, 0x000b_73b8_b793,
    0x0005_7aca_1c43, 0x000e_c265_5277, 0x0007_85a2_c1b3, 0x0007_5a07_985a,
    0x000a_4b01_eb69, 0x000a_18a1_1347, 0x000d_b1f2_8ca3, 0x0008_77ec_3e25,
    0x0003_1f63_41b8, 0x0001_363a_3a4c, 0x0000_75d8_b9ba, 0x0007_ae07_92a9,
    0x000a_83a2_1651, 0x0007_f08f_9fb5, 0x0000_d0cf_73a9, 0x000b_04dc_c98e,
    0x000f_65c7_b0f8, 0x0006_5dda_f69a, 0x0002_cf9b_86b3, 0x0001_4cb5_1e25,
    0x000f_4802_7b5b, 0x0000_ec26_ea8b, 0x0004_4baf_d45c, 0x000b_12c7_c0c4,
    0x0009_59fd_9d82, 0x000c_77c9_725a, 0x0004_8a22_d462, 0x0008_398e_8072,
    0x000e_c89b_05ce, 0x000b_b682_d4c9, 0x000e_5a86_d2ff, 0x0003_58f0_1134,
    0x0008_556d_dcf6, 0x0006_7584_b6e2, 0x0001_1609_439f, 0x0000_8488_816e,
    0x000a_af1a_2c46, 0x000f_8798_98cf, 0x0008_bbe5_e2f7, 0x0001_01ee_e363,
    0x0006_90f6_9377, 0x000f_5bd9_3cd9, 0x000c_ea4c_2bf6, 0x0009_550b_e706,
    0x0002_c5b3_8a60, 0x000e_7203_3547, 0x0004_458b_0629, 0x000e_e8d9_ed41,
    0x000d_2f91_8d72, 0x0007_8dc3_9fd3, 0x0008_2126_36f6, 0x0007_450a_72a7,
    0x000c_4f0c_f4c6, 0x0003_67bc_ddcd, 0x000c_1caf_8cc6, 0x000a_7f5b_853d,
    0x0009_d536_818b, 0x0005_35e0_21b0, 0x000a_7eb8_729e, 0x0004_22a6_7b49,
    0x0009_29e9_28a6, 0x0004_8e8a_efcc, 0x000a_9897_393c, 0x0005_eb81_d37e,
    0x0001_e802_87b7, 0x0003_4770_d903, 0x0002_eef8_6728, 0x0005_9266_ccb6,
    0x0000_110b_ba61, 0x0001_dfd2_84ef, 0x0004_4743_9d1b, 0x000f_ece0_e599,
    0x0009_309f_3703, 0x0008_0764_d1dd, 0x0003_53f1_e6a0, 0x0002_c1c1_2dcc,
    0x000c_1d21_b9d7, 0x0004_57ee_453e, 0x000d_66fa_f540, 0x0004_4831_e652,
    0x000c_fd49_a848, 0x0009_312d_4133, 0x0003_f097_d3ee, 0x0008_c9eb_ef7a,
    0x000a_99e2_9e88, 0x0000_e9fa_b22c, 0x0004_e748_f4fb, 0x000e_cdee_4288,
    0x000a_bce5_f1d0, 0x000c_42f6_876c, 0x0007_ed40_2ea0, 0x000e_5c42_42c3,
    0x000d_5b2c_31ae, 0x0002_8686_3be6, 0x0001_6044_4d94, 0x0005_f0f5_808e,
    0x000a_e3d4_4b2a, 0x0009_f5c5_d109, 0x0008_ad93_16d7, 0x0003_422b_a064,
    0x0002_fed1_1d56, 0x000b_ea6e_3e04, 0x0000_4b02_9eec, 0x0006_deed_7435,
    0x0003_718c_e17c, 0x0005_5857_f5e2, 0x0002_edac_7b62, 0x0000_85d6_c512,
    0x000d_6ca8_8e0f, 0x0002_b7e1_fc69, 0x000a_699d_5c1b, 0x000f_05ad_74de,
    0x0004_cf5f_b56d, 0x0005_725e_07e1, 0x0007_2f18_a2de, 0x0001_cec5_2609,
    0x0004_8534_243c, 0x0002_523a_4d69, 0x0003_5c1b_80d1, 0x000a_4d73_38a7,
    0x0000_db1a_f012, 0x000e_61a9_475d, 0x0000_5df0_3f91, 0x0009_7ae2_60bb,
    0x0003_2d62_7fef, 0x000b_640f_73c2, 0x0004_5a1a_c9c6, 0x0006_a220_2de1,
    0x0005_7d3e_25f2, 0x0005_aa9f_986e, 0x0000_cc85_9d8a, 0x000e_3ec6_cca8,
    0x0005_4e95_e1ae, 0x0004_4688_7b06, 0x0007_5167_32be, 0x0003_817a_c8f5,
    0x0003_e26d_938c, 0x000a_a81b_c235, 0x000d_f387_ca1b, 0x0000_f3a3_b3f2,
    0x000b_4bf6_9677, 0x000a_e218_68ed, 0x0008_1e1d_2d9d, 0x000a_0a9e_a14c,
    0x0008_eee2_97a9, 0x0004_740c_0559, 0x000e_8b14_1837, 0x000a_c69e_0a3d,
    0x0009_ed83_a1e1, 0x0005_edb5_5ecb, 0x0000_7340_fe81, 0x0005_0dfb_c6bf,
    0x0004_f583_508a, 0x000c_b1fb_78bc, 0x0004_025c_ed2f, 0x0003_9791_ebec,
    0x0005_3ee3_88f1, 0x0007_d6c0_bd23, 0x0009_3a99_5fbe, 0x0008_a417_28de,
    0x0002_fe70_e053, 0x000a_b3db_443a, 0x0001_364e_db05, 0x0004_7b6e_eed6,
    0x0001_2e71_af01, 0x0005_2ff8_3587, 0x0003_a157_5dd8, 0x0003_feaa_3564,
    0x000e_acf7_8ba7, 0x0000_872b_94f8, 0x000d_a8dd_f9a2, 0x0009_aa92_0d2b,
    0x0001_f350_ed36, 0x0001_8a5e_861f, 0x0002_c35b_89c3, 0x0003_347a_c48a,
    0x0007_f23e_022e, 0x0002_4590_68fb, 0x000e_83be_4b73,
];

/// ArUco 36h11 dictionary singleton.
pub static ARUCO_36H11: std::sync::LazyLock<TagDictionary> = std::sync::LazyLock::new(|| {
    TagDictionary::new("Aruco36h11", 6, 11, &ARUCO_36H11_CODES, &APRILTAG_36H11_POINTS)
});


// ============================================================================
// AprilTag 16h5 (30 codes)
// ============================================================================

/// Sample points for AprilTag 16h5 in canonical coordinates [-1, 1].
#[rustfmt::skip]
pub static APRILTAG_16H5_POINTS: [(f64, f64); 16] = [
    (-0.500000, -0.500000), (-0.166667, -0.500000),
    (0.166667, -0.500000), (0.500000, -0.500000),
    (-0.500000, -0.166667), (-0.166667, -0.166667),
    (0.166667, -0.166667), (0.500000, -0.166667),
    (-0.500000, 0.166667), (-0.166667, 0.166667),
    (0.166667, 0.166667), (0.500000, 0.166667),
    (-0.500000, 0.500000), (-0.166667, 0.500000),
    (0.166667, 0.500000), (0.500000, 0.500000),
];

/// AprilTag 16h5 code table (30 codes, row-major from OpenCV).
#[rustfmt::skip]
#[allow(clippy::unreadable_literal)]
pub static APRILTAG_16H5_CODES: [u64; 30] = [
    0x00000000231b, 0x000000002ea5, 0x00000000346a, 0x0000000045b9,
    0x0000000079a6, 0x000000007f6b, 0x00000000b358, 0x00000000e745,
    0x00000000fe59, 0x00000000156d, 0x00000000380b, 0x00000000f0ab,
    0x000000000d84, 0x000000004736, 0x000000008c72, 0x00000000af10,
    0x00000000093c, 0x0000000093b4, 0x00000000a503, 0x00000000468f,
    0x00000000e137, 0x000000005795, 0x00000000df42, 0x000000001c1d,
    0x00000000e9dc, 0x0000000073ad, 0x00000000ad5f, 0x00000000d530,
    0x0000000007ca, 0x00000000af2e,
];

/// AprilTag 16h5 dictionary singleton.
pub static APRILTAG_16H5: std::sync::LazyLock<TagDictionary> = std::sync::LazyLock::new(|| {
    TagDictionary::new("16h5", 4, 5, &APRILTAG_16H5_CODES, &APRILTAG_16H5_POINTS)
});

// ArUco 16h5
pub static ARUCO_16H5_CODES: [u64; 30] = [
    0x0000_0000_231b, 0x0000_0000_2ea5, 0x0000_0000_346a, 0x0000_0000_45b9,
    0x0000_0000_79a6, 0x0000_0000_7f6b, 0x0000_0000_b358, 0x0000_0000_e745,
    0x0000_0000_fe59, 0x0000_0000_156d, 0x0000_0000_380b, 0x0000_0000_f0ab,
    0x0000_0000_0d84, 0x0000_0000_4736, 0x0000_0000_8c72, 0x0000_0000_af10,
    0x0000_0000_093c, 0x0000_0000_93b4, 0x0000_0000_a503, 0x0000_0000_468f,
    0x0000_0000_e137, 0x0000_0000_5795, 0x0000_0000_df42, 0x0000_0000_1c1d,
    0x0000_0000_e9dc, 0x0000_0000_73ad, 0x0000_0000_ad5f, 0x0000_0000_d530,
    0x0000_0000_07ca, 0x0000_0000_af2e,
];

/// ArUco 16h5 dictionary singleton.
pub static ARUCO_16H5: std::sync::LazyLock<TagDictionary> = std::sync::LazyLock::new(|| {
    TagDictionary::new("Aruco16h5", 4, 5, &ARUCO_16H5_CODES, &APRILTAG_16H5_POINTS)
});


// ============================================================================
// ArUco 4x4_50 (50 codes)
// ============================================================================

/// Sample points for ArUco 4x4 in canonical coordinates [-1, 1].
#[rustfmt::skip]
pub static ARUCO_4X4_POINTS: [(f64, f64); 16] = [
    (-0.500000, -0.500000), (-0.166667, -0.500000),
    (0.166667, -0.500000), (0.500000, -0.500000),
    (-0.500000, -0.166667), (-0.166667, -0.166667),
    (0.166667, -0.166667), (0.500000, -0.166667),
    (-0.500000, 0.166667), (-0.166667, 0.166667),
    (0.166667, 0.166667), (0.500000, 0.166667),
    (-0.500000, 0.500000), (-0.166667, 0.500000),
    (0.166667, 0.500000), (0.500000, 0.500000),
];

/// ArUco 4x4_50 code table (50 codes, row-major from OpenCV).
#[rustfmt::skip]
#[allow(clippy::unreadable_literal)]
pub static ARUCO_4X4_50_CODES: [u64; 50] = [
    0x000000004cad, 0x0000000059f0, 0x00000000b4cc, 0x000000006299,
    0x00000000792a, 0x00000000b39e, 0x000000007479, 0x000000004f23,
    0x000000005b7f, 0x000000006af3, 0x00000000899f, 0x00000000e588,
    0x00000000ed70, 0x00000000f054, 0x000000008d24, 0x000000007c64,
    0x00000000a662, 0x000000000066, 0x000000007a36, 0x00000000f56e,
    0x00000000d161, 0x00000000d40d, 0x00000000ab33, 0x0000000041bb,
    0x00000000e27f, 0x000000008e29, 0x000000002735, 0x000000002aa5,
    0x00000000c484, 0x00000000f62c, 0x00000000a822, 0x000000004dea,
    0x00000000f379, 0x00000000d30f, 0x000000007510, 0x000000009490,
    0x00000000ae18, 0x00000000ff20, 0x000000006fb0, 0x000000005a38,
    0x0000000018e8, 0x000000001454, 0x00000000314c, 0x000000004d1c,
    0x000000001724, 0x00000000d774, 0x0000000026d2, 0x00000000fcb4,
    0x00000000740a, 0x00000000c80a,
];

/// ArUco 4x4_50 dictionary singleton.
pub static ARUCO_4X4_50: std::sync::LazyLock<TagDictionary> = std::sync::LazyLock::new(|| {
    TagDictionary::new("4X4_50", 4, 1, &ARUCO_4X4_50_CODES, &ARUCO_4X4_POINTS)
});

// ============================================================================
// ArUco 4x4_100 (100 codes)
// ============================================================================

/// ArUco 4x4_100 code table (100 codes, row-major from OpenCV).
#[rustfmt::skip]
#[allow(clippy::unreadable_literal)]
pub static ARUCO_4X4_100_CODES: [u64; 100] = [
    0x000000004cad, 0x0000000059f0, 0x00000000b4cc, 0x000000006299,
    0x00000000792a, 0x00000000b39e, 0x000000007479, 0x000000004f23,
    0x000000005b7f, 0x000000006af3, 0x00000000899f, 0x00000000e588,
    0x00000000ed70, 0x00000000f054, 0x000000008d24, 0x000000007c64,
    0x00000000a662, 0x000000000066, 0x000000007a36, 0x00000000f56e,
    0x00000000d161, 0x00000000d40d, 0x00000000ab33, 0x0000000041bb,
    0x00000000e27f, 0x000000008e29, 0x000000002735, 0x000000002aa5,
    0x00000000c484, 0x00000000f62c, 0x00000000a822, 0x000000004dea,
    0x00000000f379, 0x00000000d30f, 0x000000007510, 0x000000009490,
    0x00000000ae18, 0x00000000ff20, 0x000000006fb0, 0x000000005a38,
    0x0000000018e8, 0x000000001454, 0x00000000314c, 0x000000004d1c,
    0x000000001724, 0x00000000d774, 0x00000000fcb4, 0x0000000026d2,
    0x00000000740a, 0x00000000c80a, 0x00000000298a, 0x0000000016aa,
    0x0000000082ba, 0x00000000e9fa, 0x000000008016, 0x00000000e616,
    0x000000002486, 0x000000009786, 0x0000000048d6, 0x00000000a7f6,
    0x00000000fbe6, 0x00000000d87e, 0x000000000501, 0x0000000022c1,
    0x0000000045d1, 0x000000005ec9, 0x000000003621, 0x0000000054a1,
    0x0000000039a1, 0x000000009139, 0x0000000085f9, 0x000000003edd,
    0x00000000203d, 0x00000000da6d, 0x0000000013fd, 0x00000000d5ed,
    0x00000000f853, 0x000000004693, 0x000000001a9b, 0x00000000abcb,
    0x000000001933, 0x0000000005e3, 0x00000000eca3, 0x00000000ba97,
    0x00000000a49f, 0x00000000dddf, 0x000000005477, 0x00000000b2ef,
    0x00000000aeac, 0x00000000b551, 0x00000000e86e, 0x00000000f350,
    0x00000000d260, 0x0000000083b4, 0x000000001b92, 0x000000002fc2,
    0x000000006cf2, 0x00000000cbf2, 0x000000002796, 0x00000000e30e,
];

/// ArUco 4x4_100 dictionary singleton.
pub static ARUCO_4X4_100: std::sync::LazyLock<TagDictionary> = std::sync::LazyLock::new(|| {
    TagDictionary::new("4X4_100", 4, 1, &ARUCO_4X4_100_CODES, &ARUCO_4X4_POINTS)
});

// AprilTag 41h12
pub static APRILTAG_41H12_POINTS: [(f64, f64); 41] = [
    (-0.888889, -0.888889), (-0.666667, -0.888889),
    (-0.444444, -0.888889), (-0.222222, -0.888889),
    (0.0, -0.888889), (0.222222, -0.888889),
    (0.444444, -0.888889), (0.666667, -0.888889),
    (-0.222222, -0.222222), (0.0, -0.222222),
    (0.888889, -0.888889), (0.888889, -0.666667),
    (0.888889, -0.444444), (0.888889, -0.222222),
    (0.888889, 0.0), (0.888889, 0.222222),
    (0.888889, 0.444444), (0.888889, 0.666667),
    (0.222222, -0.222222), (0.222222, 0.0),
    (0.888889, 0.888889), (0.666667, 0.888889),
    (0.444444, 0.888889), (0.222222, 0.888889),
    (0.0, 0.888889), (-0.222222, 0.888889),
    (-0.444444, 0.888889), (-0.666667, 0.888889),
    (0.222222, 0.222222), (0.0, 0.222222),
    (-0.888889, 0.888889), (-0.888889, 0.666667),
    (-0.888889, 0.444444), (-0.888889, 0.222222),
    (-0.888889, 0.0), (-0.888889, -0.222222),
    (-0.888889, -0.444444), (-0.888889, -0.666667),
    (-0.222222, 0.222222), (-0.222222, 0.0),
    (0.0, 0.0),
];
pub static APRILTAG_41H12_CODES: [u64; 2115] = [
    0x01bd_8a64_ad10, 0x01bd_c4f3_b2d5, 0x01bd_ff82_b89a, 0x01be_3a11_be5f,
    0x01be_74a0_c424, 0x01be_af2f_c9e9, 0x01be_e9be_cfae, 0x01bf_244d_d573,
    0x01bf_5edc_db38, 0x01bf_996b_e0fd, 0x01bf_d3fa_e6c2, 0x01c0_0e89_ec87,
    0x01c0_4918_f24c, 0x01c0_be36_fdd6, 0x01c0_f8c6_039b, 0x01c1_3355_0960,
    0x01c1_6de4_0f25, 0x01c2_1d91_2074, 0x01c2_5820_2639, 0x01c2_92af_2bfe,
    0x01c3_07cd_3788, 0x01c3_7ceb_4312, 0x01c3_b77a_48d7, 0x01c3_f209_4e9c,
    0x01c4_2c98_5461, 0x01c4_a1b6_5feb, 0x01c4_dc45_65b0, 0x01c5_16d4_6b75,
    0x01c5_5163_713a, 0x01c6_3b9f_884e, 0x01c6_762e_8e13, 0x01c6_eb4c_999d,
    0x01c7_25db_9f62, 0x01c7_606a_a527, 0x01c7_d588_b0b1, 0x01c8_1017_b676,
    0x01c8_4aa6_bc3b, 0x01c8_8535_c200, 0x01c8_bfc4_c7c5, 0x01c9_6f71_d914,
    0x01c9_e48f_e49e, 0x01ca_1f1e_ea63, 0x01ca_59ad_f028, 0x01cb_095b_0177,
    0x01cb_43ea_073c, 0x01cb_b908_12c6, 0x01cc_68b5_2415, 0x01cc_a344_29da,
    0x01cc_ddd3_2f9f, 0x01cd_1862_3564, 0x01cd_8d80_40ee, 0x01cd_c80f_46b3,
    0x01ce_029e_4c78, 0x01ce_77bc_5802, 0x01ce_b24b_5dc7, 0x01ce_ecda_638c,
    0x01cf_2769_6951, 0x01cf_9c87_74db, 0x01d0_86c3_8bef, 0x01d1_3670_9d3e,
    0x01d1_ab8e_a8c8, 0x01d2_5b3b_ba17, 0x01d2_95ca_bfdc, 0x01d3_0ae8_cb66,
    0x01d3_f524_e27a, 0x01d4_2fb3_e83f, 0x01d4_6a42_ee04, 0x01d4_a4d1_f3c9,
    0x01d5_19ef_ff53, 0x01d5_547f_0518, 0x01d5_c99d_10a2, 0x01d6_042c_1667,
    0x01d6_794a_21f1, 0x01d6_ee68_2d7b, 0x01d7_6386_3905, 0x01d7_d8a4_448f,
    0x01d8_1333_4a54, 0x01d8_4dc2_5019, 0x01d8_8851_55de, 0x01d8_c2e0_5ba3,
    0x01d8_fd6f_6168, 0x01d9_e7ab_787c, 0x01da_5cc9_8406, 0x01da_9758_89cb,
    0x01db_0c76_9555, 0x01db_4705_9b1a, 0x01db_bc23_a6a4, 0x01db_f6b2_ac69,
    0x01dc_a65f_bdb8, 0x01dc_e0ee_c37d, 0x01dd_cb2a_da91, 0x01df_6514_02f4,
    0x01df_9fa3_08b9, 0x01e0_14c1_1443, 0x01e2_23c8_4830, 0x01e2_5e57_4df5,
    0x01e3_8322_6ace, 0x01e3_bdb1_7093, 0x01e4_a7ed_87a7, 0x01e4_e27c_8d6c,
    0x01e5_1d0b_9331, 0x01e5_9229_9ebb, 0x01e6_0747_aa45, 0x01e6_7c65_b5cf,
    0x01e6_b6f4_bb94, 0x01e7_66a1_cce3, 0x01e9_008a_f546, 0x01e9_eac7_0c5a,
    0x01ea_2556_121f, 0x01ea_d503_236e, 0x01eb_84b0_34bd, 0x01ed_ce46_6e6f,
    0x01ee_b882_8583, 0x01ee_f311_8b48, 0x01ef_2da0_910d, 0x01f0_526b_ade6,
    0x01f1_0218_bf35, 0x01f1_b1c5_d084, 0x01f2_26e3_dc0e, 0x01f3_4bae_f8e7,
    0x01f3_c0cd_0471, 0x01f4_35eb_0ffb, 0x01f5_5ab6_2cd4, 0x01f5_cfd4_385e,
    0x01f6_7f81_49ad, 0x01f7_dedb_6c4b, 0x01f8_8e88_7d9a, 0x01f8_c917_835f,
    0x01f9_03a6_8924, 0x01f9_b353_9a73, 0x01fa_9d8f_b187, 0x01fd_21b4_f0fe,
    0x01fd_5c43_f6c3, 0x01fe_0bf1_0812, 0x0000_5587_41c4, 0x0000_caa5_4d4e,
    0x0001_3fc3_58d8, 0x0002_d9ac_813b, 0x0003_4eca_8cc5, 0x0003_8959_928a,
    0x0004_3906_a3d9, 0x0004_ae24_af63, 0x0004_e8b3_b528, 0x0005_2342_baed,
    0x0008_5715_0bb3, 0x0009_b66f_2e51, 0x000a_db3a_4b2a, 0x000b_8ae7_5c79,
    0x000e_499b_a1b5, 0x000f_6e66_be8e, 0x0010_cdc0_e12c, 0x0011_42de_ecb6,
    0x0013_51e6_20a3, 0x0014_3c22_37b7, 0x0017_3565_82b8, 0x0017_aa83_8e42,
    0x0018_1fa1_99cc, 0x0018_cf4e_ab1b, 0x0019_446c_b6a5, 0x001a_a3c6_d943,
    0x001b_8e02_f057, 0x001b_c891_f61c, 0x001e_4cb7_3593, 0x0022_3036_97a8,
    0x0023_1a72_aebc, 0x0023_ca1f_c00b, 0x0026_88d4_0547, 0x0026_c363_0b0c,
    0x0027_3881_1696, 0x0027_e82e_27e5, 0x0029_0cf9_44be, 0x002b_568f_7e70,
    0x002b_911e_8435, 0x002c_063c_8fbf, 0x002c_b5e9_a10e, 0x002d_2b07_ac98,
    0x002e_1543_c3ac, 0x002e_4fd2_c971, 0x002f_3a0e_e085, 0x0031_f8c3_25c1,
    0x0032_6de1_314b, 0x0035_a1b3_8211, 0x0036_5160_9360, 0x0036_8bef_9925,
    0x0037_3b9c_aa74, 0x0037_b0ba_b5fe, 0x003b_1f1c_0c89, 0x003b_59ab_124e,
    0x003b_cec9_1dd8, 0x003c_0958_239d, 0x003d_68b2_463b, 0x003e_185f_578a,
    0x003e_52ee_5d4f, 0x0042_366d_bf64, 0x0043_20a9_d678, 0x0043_5b38_dc3d,
    0x0043_95c7_e202, 0x0044_4574_f351, 0x0045_2fb1_0a65, 0x0045_a4cf_15ef,
    0x004a_ad19_94dd, 0x004e_5609_f12d, 0x0050_6511_251a, 0x0052_e936_6491,
    0x0053_5e54_701b, 0x0053_d372_7ba5, 0x0054_831f_8cf4, 0x0056_5797_bb1c,
    0x005b_d500_4594, 0x005c_0f8f_4b59, 0x005d_e407_7981, 0x005f_08d2_965a,
    0x0060_2d9d_b333, 0x0064_c0ca_2697, 0x0065_35e8_3221, 0x0065_e595_4370,
    0x0066_5ab3_4efa, 0x0066_cfd1_5a84, 0x0067_ba0d_7198, 0x0068_2f2b_7d22,
    0x0069_53f6_99fb, 0x0069_c914_a585, 0x006a_03a3_ab4a, 0x006b_286e_c823,
    0x006c_87c8_eac1, 0x006d_7205_01d5, 0x006f_810c_35c2, 0x006f_bb9b_3b87,
    0x0071_caa2_6f74, 0x0073_648b_97d7, 0x0074_1438_a926, 0x0075_7392_cbc4,
    0x007d_ea3e_a13d, 0x007e_d47a_b851, 0x0082_7d6b_14a1, 0x0082_f289_202b,
    0x0083_2d18_25f0, 0x0083_67a7_2bb5, 0x0083_dcc5_373f, 0x0088_6ff1_aaa3,
    0x008a_09da_d306, 0x008a_4469_d8cb, 0x008c_c88f_1842, 0x008d_3dad_23cc,
    0x008d_ed5a_351b, 0x008f_8743_5d7e, 0x0090_717f_7492, 0x0091_5bbb_8ba6,
    0x0092_f5a4_b409, 0x0097_4e42_21a8, 0x0099_97d8_5b5a, 0x009a_4785_6ca9,
    0x009e_2b04_cebe, 0x00a0_e9b9_13fa, 0x00a3_6dde_5371, 0x00a8_b0b7_d824,
    0x00a8_eb46_dde9, 0x00aa_1011_fac2, 0x00ac_59a8_3474, 0x00af_c809_8aff,
    0x00b2_4c2e_ca76, 0x00b2_fbdb_dbc5, 0x00b3_e617_f2d9, 0x00b5_4572_1577,
    0x00ba_132d_8ea0, 0x00c0_05b4_24a2, 0x00c3_e933_86b7, 0x00c5_488d_a955,
    0x00c5_f83a_baa4, 0x00c9_669c_112f, 0x00cd_bf39_7ece, 0x00ce_3457_8a58,
    0x00ce_a975_95e2, 0x00d4_26de_205a, 0x00db_eddc_e484, 0x00dd_c255_12ac,
    0x00e7_5dcc_04fe, 0x00eb_b669_729d, 0x00ef_99e8_d4b2, 0x00f0_f942_f750,
    0x00f3_42d9_3102, 0x00f6_ebc9_8d52, 0x00fa_1f9b_de18, 0x00fb_09d7_f52c,
    0x00fb_f414_0c40, 0x00fe_ed57_5741, 0x0103_45f4_c4e0, 0x0105_1a6c_f308,
    0x0108_4e3f_43ce, 0x0109_387b_5ae2, 0x0109_e828_6c31, 0x010a_5d46_77bb,
    0x010b_0cf3_890a, 0x010c_31be_a5e3, 0x010d_5689_c2bc, 0x010e_0636_d40b,
    0x0112_9963_476f, 0x0113_839f_5e83, 0x011a_25d3_05d4, 0x011b_4a9e_22ad,
    0x011b_bfbc_2e37, 0x011d_cec3_6224, 0x011e_f38e_7efd, 0x011f_68ac_8a87,
    0x011f_ddca_9611, 0x0120_1859_9bd6, 0x0120_52e8_a19b, 0x012a_9e0c_a53c,
    0x0130_5604_3579, 0x0135_23bf_aea2, 0x0140_ce3d_d4e1, 0x0142_dd45_08ce,
    0x014a_2f25_c16e, 0x014a_ded2_d2bd, 0x014e_12a5_2383, 0x0152_a5d1_96e7,
    0x0153_557e_a836, 0x0154_052b_b985, 0x0154_b4d8_cad4, 0x0154_ef67_d099,
    0x0157_ae1c_15d5, 0x0157_e8ab_1b9a, 0x015b_1c7d_6c60, 0x0162_e37c_308a,
    0x0169_c03e_dda0, 0x016e_c889_5c8e, 0x016f_3da7_6818, 0x0171_121f_9640,
    0x0182_af24_5281, 0x018a_eb41_2235, 0x018b_9aee_3384, 0x0192_0292_d510,
    0x0195_e612_3725, 0x019a_b3cd_b04e, 0x019b_d898_cd27, 0x019d_37f2_efc5,
    0x01a2_b55b_7a3d, 0x01a6_d369_e217, 0x01ad_eabb_94f2, 0x01b3_dd42_2af4,
    0x01ba_7f75_d245, 0x01bb_2f22_e394, 0x01c1_5c38_7f5b, 0x01c6_d9a1_09d3,
    0x01cc_9198_9a10, 0x01d8_3c16_c04f, 0x01e1_27e0_a152, 0x01e4_2123_ec53,
    0x01e6_dfd8_318f, 0x01e8_79c1_59f2, 0x01f7_5811_d0f7, 0x01f8_7cdc_edd0,
    0x01f8_f1fa_f95a, 0x0001_68a6_ced3, 0x000a_19e1_aa11, 0x000b_3eac_c6ea,
    0x000e_37f0_11eb, 0x0010_8186_4b9d, 0x0018_8314_158c, 0x001f_5fd6_c2a2,
    0x0021_6edd_f68f, 0x0038_fe69_48d2, 0x003f_a09c_f023, 0x0041_ea33_29d5,
    0x004c_aa75_3900, 0x004d_cf40_55d9, 0x0052_d78a_d4c7, 0x0056_807b_3117,
    0x0058_ca11_6ac9, 0x005e_bc98_00cb, 0x0062_6588_5d1b, 0x0070_942b_c2d1,
    0x0073_8d6f_0dd2, 0x007a_df4f_c672, 0x0085_da20_db62, 0x008a_6d4d_4ec6,
    0x008a_e26b_5a50, 0x008b_9218_6b9f, 0x008e_c5ea_bc65, 0x0096_525a_7aca,
    0x00a0_d80d_8430, 0x00a5_a5c8_fd59, 0x00a9_142a_53e4, 0x00b4_8419_745e,
    0x00b8_6798_d673, 0x00c6_963c_3c29, 0x00ca_3f2c_9879, 0x00f0_e797_6786,
    0x00f4_1b69_b84c, 0x00fe_668d_bbed, 0x00ff_163a_cd3c, 0x0101_5fd1_06ee,
    0x010f_c903_7269, 0x011e_6cc4_e3a9, 0x0136_716e_4176, 0x013f_97c7_283e,
    0x0142_567b_6d7a, 0x0148_f8af_14cb, 0x014a_9298_3d2e, 0x014c_a19f_711b,
    0x014d_16bd_7ca5, 0x0157_d6ff_8bd0, 0x015e_ee51_3eab, 0x016c_a7d6_98d7,
    0x0174_a964_62c6, 0x0175_93a0_79da, 0x0177_a2a7_adc7, 0x0180_8e71_8eca,
    0x0181_038f_9a54, 0x0190_918d_22a8, 0x019e_1083_770f, 0x01ac_3f26_dcc5,
    0x01b4_7b43_ac79, 0x01b8_d3e1_1a18, 0x01bb_cd24_6519, 0x01ef_b9f6_82c8,
    0x0005_af98_aca8, 0x0026_a00b_eb78, 0x0058_08b8_c9b0, 0x0065_4d20_1852,
    0x007c_2cfe_5946, 0x008d_1a56_0438, 0x00a4_e470_5c40, 0x00a8_1842_ad06,
    0x00bc_e919_ba0d, 0x00c5_2536_89c1, 0x00d4_edc3_17da, 0x00ea_a8d6_3bf5,
    0x010f_b757_e29f, 0x0119_c7ec_e07b, 0x0137_bf1c_d44a, 0x0141_cfb1_d226,
    0x015c_5880_6f6a, 0x0165_444a_506d, 0x016b_abee_f1f9, 0x016d_0b49_1497,
    0x0187_5988_ac16, 0x0190_f4ff_9e68, 0x0198_f68d_6857, 0x01a6_0065_b134,
    0x01ad_17b7_640f, 0x01af_26be_97fc, 0x01b2_2001_e2fd, 0x01b4_6998_1caf,
    0x01b8_4d17_7ec4, 0x01b9_ac71_a162, 0x01ba_5c1e_b2b1, 0x01ce_42b9_a8a4,
    0x01d1_3bfc_f3a5, 0x01da_d773_e5f7, 0x01dc_abec_141f, 0x01e0_1a4d_6aaa,
    0x01e1_0489_81be, 0x01e3_88ae_c135, 0x01e9_7b35_5737, 0x01fe_4c0c_643e,
    0x0015_db97_b681, 0x0042_eba7_271a, 0x0044_4b01_49b8, 0x004b_d771_081d,
    0x007f_c443_25cc, 0x0099_2846_a637, 0x009b_71dc_dfe9, 0x00b7_1f76_9a06,
    0x00c2_54d6_b4bb, 0x00e5_c96f_3302, 0x00f8_c5ce_11e1, 0x00fe_08a7_9694,
    0x0109_0378_ab84, 0x0109_3e07_b149, 0x010b_c22c_f0c0, 0x010e_f5ff_4186,
    0x011b_502a_7914, 0x0126_858a_93c9, 0x012b_5346_0cf2, 0x0142_a842_5970,
    0x015f_7aa7_3066, 0x01a0_e66f_a27c, 0x01d5_bd7d_d73f, 0x01dd_f99a_a6f3,
    0x0010_c1a1_a7c9, 0x001b_81e3_b6f4, 0x0021_e988_5880, 0x0025_9278_b4d0,
    0x003a_d86d_cd61, 0x003e_f67c_353b, 0x0099_169b_166d, 0x00a2_eca1_0e84,
    0x00b4_fec3_d64f, 0x00b7_0dcb_0a3c, 0x00b8_3296_2715, 0x00cb_dea2_1743,
    0x00ce_d7e5_6244, 0x00de_65e2_ea98, 0x00f0_02e7_a6d9, 0x0125_89a2_eceb,
    0x0149_e877_8246, 0x0167_2ffa_64c6, 0x01be_56d5_faf7, 0x01ec_8bb0_8869,
    0x01f5_777a_696c, 0x0017_c747_cada, 0x0045_4c75_46fd, 0x007d_91e4_d24b,
    0x00bb_1a2d_e24c, 0x0107_0ba9_5dc8, 0x0127_4c6f_8b49, 0x0129_20e7_b971,
    0x014a_4be9_fe06, 0x0150_03e1_8e43, 0x017f_22f8_32c9, 0x01f3_cbe5_b13f,
    0x01fd_2ccd_9dcc, 0x001c_bde6_b9fe, 0x001f_f1b9_0ac4, 0x002c_fb91_53a1,
    0x0030_2f63_a467, 0x004e_6122_9dfb, 0x00a0_0a95_a9b4, 0x00b2_1cb8_717f,
    0x00dd_584f_b3f0, 0x0126_15f8_dea6, 0x012f_76e0_cb33, 0x013f_b48b_64d6,
    0x016e_5e83_fdd2, 0x0170_a81a_3784, 0x0195_06ee_ccdf, 0x01aa_4ce3_e570,
    0x0058_257d_0648, 0x0064_0a8a_324c, 0x007e_9358_cf90, 0x008b_2813_0ce3,
    0x00c4_ccdc_bacf, 0x00c7_1672_f481, 0x00d4_5ada_4323, 0x0118_1038_eeeb,
    0x014d_d183_3ac2, 0x0163_c725_64a2, 0x017e_c512_0d70, 0x002d_fd05_50e6,
    0x0048_4b44_e865, 0x0079_b3f1_c69d, 0x00c6_551a_5368, 0x00ee_976e_4ad8,
    0x011f_506e_17c1, 0x0156_3683_8071, 0x017e_edf5_836b, 0x01a5_5bd1_4cb3,
    0x01f8_d9bc_8694, 0x0033_a351_5159, 0x0058_0225_e6b4, 0x0063_7215_072e,
    0x00c0_c606_3926, 0x0156_5f66_f66c, 0x015d_b147_af0c, 0x018c_d05e_5392,
    0x01b2_c91c_1150, 0x008c_c788_8bad, 0x0136_8213_44ab, 0x0163_1d04_a9ba,
    0x01b9_1f15_2312, 0x01cd_403f_1eca, 0x01d1_d36b_922e, 0x01f1_29f5_a89b,
    0x002b_b8fb_6d9b, 0x006b_c569_bd13, 0x0113_ab7c_47e9, 0x0116_6a30_8d25,
    0x0131_a2ac_3bb8, 0x0170_c4de_741c, 0x01cf_7829_c8b2, 0x01e3_9953_c46a,
    0x000d_b01f_ea02, 0x0082_ce2b_7402, 0x00aa_60d2_5a23, 0x00b6_45df_8627,
    0x00b6_806e_8bec, 0x00cd_d56a_d86a, 0x0129_54e3_dc3a, 0x013f_bfa4_11a4,
    0x0182_c555_ac1d, 0x019e_72ef_663a, 0x01a4_a005_0201, 0x01af_25b8_0b67,
    0x01b5_52cd_a72e, 0x01bb_0ac5_376b, 0x01d4_3439_b211, 0x000e_88b0_714c,
    0x0027_ecb3_f1b7, 0x005c_c3c2_267a, 0x00bb_ec2b_869a, 0x00fa_993f_b374,
    0x0111_ee3b_fff2, 0x016c_0e5a_e124, 0x018e_2399_3ccd, 0x0017_d7ec_ce0f,
    0x0072_a7b8_c090, 0x007c_f2dc_c431, 0x00a9_533f_237b, 0x00e0_3954_8c2b,
    0x0101_d974_dc4a, 0x0143_f4ea_5faf, 0x0147_634b_b63a, 0x015f_2d66_0e42,
    0x0163_8603_7be1, 0x0180_cd86_5e61, 0x01b2_e5e0_4de8, 0x01b4_ba58_7c10,
    0x0021_2729_2ad2, 0x0071_abd1_19b2, 0x0072_20ef_253c, 0x008d_93f9_d994,
    0x00c5_9eda_5f1d, 0x00d7_b0fd_26e8, 0x017d_fd26_895b, 0x01c1_77f6_2f5e,
    0x01c8_54b8_dc74, 0x01f6_8993_69e6, 0x0103_5006_e519, 0x012b_925a_dc89,
    0x015b_9bad_9823, 0x01a9_2712_3c02, 0x0015_5953_e4ff, 0x006a_e646_52cd,
    0x00d4_cef1_c218, 0x00e3_ad42_391d, 0x015c_39af_19a8, 0x0192_aaa6_76ce,
    0x01bb_6218_79c8, 0x00a5_2911_823e, 0x00ac_05d4_2f54, 0x00dd_a910_1351,
    0x0184_2fc8_7b89, 0x001f_4691_c347, 0x00bc_e180_4a7c, 0x010c_06ce_16be,
    0x014b_638f_54e7, 0x01f6_b803_3648, 0x000f_31ca_9f9f, 0x002c_042f_7695,
    0x00f4_6597_34b1, 0x00ff_25d9_43dc, 0x01ad_e8ae_7bc8, 0x01c8_36ee_1347,
    0x002a_1e0b_b8a3, 0x003e_b453_bfe5, 0x00fd_b4d3_9174, 0x0184_3554_d1f0,
    0x01c6_8b59_5b1a, 0x0051_d996_14bf, 0x018d_bf20_3478, 0x01f7_6d3c_9dfe,
    0x007f_1288_fb53, 0x00aa_1391_37ff, 0x00b5_be0f_5e3e, 0x015c_7f56_cc3b,
    0x01a1_1ef1_8f17, 0x01cf_c8ea_2813, 0x00eb_a83d_2010, 0x0181_f14a_eea5,
    0x0000_aacd_6af7, 0x004b_ec9b_d524, 0x0054_d865_b627, 0x0175_8574_274d,
    0x006e_654c_ac8d, 0x0115_6123_204f, 0x0122_a58a_6ef1, 0x01d0_7e23_8fc9,
    0x01f3_b82d_084b, 0x007f_0669_c1f0, 0x009a_3ee5_7083, 0x011c_dbe7_4eea,
    0x0171_7e9d_a5a4, 0x0174_b26f_f66a, 0x0015_8130_ce65, 0x00ac_ef09_b9d3,
    0x0178_f961_d43f, 0x01a6_0971_44d8, 0x01d1_4508_8749, 0x001b_620b_d49d,
    0x016d_77c7_23fb, 0x00f0_4682_4042, 0x013d_5cc8_d897, 0x006d_5d45_cc4c,
    0x0140_7eef_9993, 0x0165_52e2_3a78, 0x01d1_8523_e375, 0x000d_38f4_c54e,
    0x0199_6897_ce22, 0x019b_db11_7dcf, 0x01d1_9c5b_c9a6, 0x01d6_df35_4e59,
    0x0078_9832_3d68, 0x0105_45c9_19ab, 0x0159_38d2_5f16, 0x0194_b214_3b2a,
    0x0038_7a18_5e26, 0x00d4_408e_b733, 0x011e_d2b0_1011, 0x0143_6c13_ab31,
    0x0168_4006_4c16, 0x01d8_1b38_5163, 0x0097_cb65_3441, 0x01c7_9153_2231,
    0x01e8_f6e4_6c8b, 0x0076_5428_5a1d, 0x00fe_6e92_c2fc, 0x019d_ddf9_7859,
    0x008d_225b_0b47, 0x0106_23e5_f75c, 0x01bd_d285_104b, 0x004f_fd84_7706,
    0x01bc_617f_5de3, 0x004e_1760_b914, 0x0157_aa01_e381, 0x00a9_fa4c_38a4,
    0x0106_17c6_bdf9, 0x0108_26cd_f1e6, 0x0138_6aaf_b345, 0x00f3_329f_c54b,
    0x01ca_ace7_0031, 0x0003_2ce5_9144, 0x000f_11f2_bd48, 0x0095_57e4_f7ff,
    0x0106_cd00_25af, 0x007f_8b26_441a, 0x00a4_99a7_eac4, 0x0151_b0e8_6a83,
    0x0035_4acb_d732, 0x00a7_aa23_1bf6, 0x0037_4827_7b55, 0x005c_cbc7_2d89,
    0x0139_c376_f2e7, 0x0008_b566_c88a, 0x00ba_36f0_45b2, 0x017b_e478_ccb3,
    0x01f7_930c_6e3a, 0x005b_4ea2_41be, 0x002a_be85_ead0, 0x010a_af78_fb2f,
    0x0079_227b_15f9, 0x015e_1bb8_a546, 0x0194_c73f_0831, 0x01e1_6867_94fc,
    0x01e2_52a3_ac10, 0x00a8_6a75_307a, 0x0170_1c2f_dd47, 0x0172_8ea9_8cf4,
    0x01ac_a891_466a, 0x0107_e4a5_7c90, 0x0140_9f33_1368, 0x0196_66b4_86fb,
    0x0042_3046_73e6, 0x0187_3c29_7a67, 0x0076_45fc_0790, 0x0148_084b_b239,
    0x0160_bca2_2155, 0x002d_0189_4186, 0x009c_2d0e_3584, 0x0152_b6e2_319a,
    0x01f8_18cf_7cf9, 0x0008_9109_1c61, 0x0020_95b2_7a2e, 0x0049_1295_7763,
    0x00a0_27c5_7dca, 0x01d6_8fe7_130b, 0x0047_f356_b0f1, 0x0176_f7eb_fdc8,
    0x01a8_d5b6_e78a, 0x00fe_0d98_f7e4, 0x006c_0b7d_0724, 0x01b6_f83b_13dd,
    0x019b_38f5_c9f6, 0x01d8_86af_d432, 0x01f9_3c94_0d3d, 0x001f_e4fe_dc4a,
    0x006e_cfbd_a2c7, 0x019f_4558_a206, 0x0004_d566_a3b2, 0x0107_1627_157f,
    0x01e8_1a39_b2ed, 0x010f_0b95_a60b, 0x013d_b58e_3f07, 0x01c3_fb80_79be,
    0x0129_4829_adc0, 0x00fe_3575_e14a, 0x018a_6dee_b203, 0x0095_fdeb_8093,
    0x019d_7566_3db0, 0x0005_3d5e_e944, 0x018f_236b_b866, 0x01fb_1b1e_5b9e,
    0x008f_7e08_6c41, 0x00a7_82b1_ca0e, 0x009d_9b00_422d, 0x00bf_eacd_a39b,
    0x0031_cc31_14f0, 0x00f9_b87a_c782, 0x01c8_0c69_1ba0, 0x0006_7eee_42b5,
    0x00d2_a07e_4352, 0x0134_c22a_ee73, 0x013b_d97c_a14e, 0x006f_bd78_f718,
    0x00e6_004f_9df1, 0x018c_755c_765f, 0x010f_4417_92a6, 0x0163_5971_6b10,
    0x01f2_dcf4_72c0, 0x002c_0ca0_1522, 0x0199_f8d8_9498, 0x0168_43f1_20d1,
    0x0075_ba11_ad53, 0x01df_24c9_492f, 0x0019_ee5e_13f4, 0x0056_6913_ed4d,
    0x0064_c09a_c8fe, 0x01be_511a_46f7, 0x00be_4844_7f12, 0x0106_1bb1_92b4,
    0x00b0_b7a2_9ae1, 0x0147_3b3f_6f3b, 0x0016_044b_cee3, 0x01c0_65ad_d14b,
    0x01d0_68c9_6529, 0x008a_d61c_c354, 0x0045_b544_bb8b, 0x011c_9716_cb53,
    0x004a_136e_7f91, 0x0104_a9a5_53b7, 0x0049_f017_5ffd, 0x0011_4cc1_af56,
    0x0103_b34a_0340, 0x015e_d4dc_e1b7, 0x014f_86fa_b58f, 0x01e6_f4d3_a0fd,
    0x0010_9681_bb0b, 0x014a_95e8_1cd2, 0x0132_0a75_23b1, 0x0030_6d42_89d0,
    0x00d4_2927_7369, 0x0015_ba89_e9fc, 0x0190_0679_f767, 0x0185_1135_38de,
    0x01b6_b471_1cdb, 0x01f6_fb6e_7218, 0x0122_1c84_5cda, 0x01dd_e29e_faa7,
    0x0036_33d2_0418, 0x01dc_f1d0_0097, 0x00ee_fb1d_007d, 0x0077_bc5e_b2c6,
    0x00c1_5298_64c6, 0x0026_f108_84be, 0x00b1_8f98_2d14, 0x005d_7937_c815,
    0x002b_89c1_4e89, 0x000d_80e5_caf0, 0x0163_101b_1da7, 0x018c_3cab_2c2b,
    0x00ee_e1d8_0122, 0x01ea_45d5_c5d9, 0x00cf_dd14_d6ab, 0x0090_a937_0e7d,
    0x00c8_dcfb_0a01, 0x0102_4735_b228, 0x01c2_cff3_1c50, 0x0115_6058_cd9f,
    0x00dc_5990_a138, 0x01e2_1fea_4fc1, 0x008b_5937_c3d2, 0x0145_b4df_9233,
    0x014f_cb00_e676, 0x009d_4170_890d, 0x006a_2d2e_f2a8, 0x003e_5922_8519,
    0x004a_1ad8_9189, 0x0062_abd7_e111, 0x019c_3bac_8db5, 0x018a_bb6c_0e0c,
    0x01d9_59f0_3efa, 0x00a3_cfeb_876a, 0x01e8_b877_6e57, 0x0178_7f5f_43b1,
    0x01ec_5ad4_e7ab, 0x00c2_b051_05b8, 0x0096_a1b5_9264, 0x0129_9926_4706,
    0x0081_ca4b_a261, 0x0091_34f2_0b21, 0x0179_d71f_f6be, 0x0124_3881_f926,
    0x00c8_f924_5182, 0x00cb_a62d_06f4, 0x0017_6b7b_9af7, 0x0110_8b6f_7c63,
    0x0188_0a49_2646, 0x01e3_fee0_35a0, 0x0166_8160_bc58, 0x0027_3301_9c7b,
    0x00b9_de37_bb8e, 0x012e_d8ec_25fa, 0x004f_62a3_778c, 0x0173_6048_7610,
    0x014e_33fc_0639, 0x00a0_a10a_97f4, 0x0176_3aba_6b4f, 0x01dc_7a75_7e4a,
    0x017a_bc3b_4ee9, 0x01ea_aea5_3a67, 0x0145_007d_5979, 0x01dc_e374_5071,
    0x01f1_b44b_5d78, 0x003b_7368_8573, 0x005c_d8f9_cfcd, 0x0178_5a66_a271,
    0x008f_0e17_fbec, 0x00f9_488a_572d, 0x00e8_60bf_02a2, 0x01fd_d86d_5913,
    0x006a_3eab_24d9, 0x01f4_5f46_f9c0, 0x001f_1414_a0dd, 0x0092_2e31_a3be,
    0x0106_c46d_05d5, 0x0079_a744_746f, 0x0005_ffca_f33e, 0x00b4_4162_e63d,
    0x00ea_f275_9f8f, 0x006a_28e5_62bb, 0x0174_6a2d_11e2, 0x00ed_ae16_3f0c,
    0x0031_890e_ef51, 0x0083_7829_ad9d, 0x0082_698f_ea60, 0x00da_c5db_a09f,
    0x01b8_3dc2_1e55, 0x00fd_ad17_a096, 0x0104_2bf4_2853, 0x0037_9ad2_7293,
    0x008a_de2e_a6af, 0x0024_9254_5a51, 0x0128_bcb7_c74d, 0x016f_9336_a77c,
    0x011c_2c83_53cc, 0x0110_643a_6460, 0x01dd_0b8d_73bc, 0x0086_50fa_2130,
    0x018b_a7c2_1a96, 0x0028_c173_5cde, 0x0126_fb5d_4d02, 0x018a_939c_00f2,
    0x0037_17f3_abfa, 0x004f_797c_a28b, 0x0054_8753_77e0, 0x01dd_a46e_3658,
    0x01f4_fef6_d93d, 0x00ef_43b5_d782, 0x0154_eafb_bf5f, 0x0185_512e_13bd,
    0x0085_bd76_5762, 0x0163_8db6_a40a, 0x0070_70ee_5bd5, 0x018b_a620_98ea,
    0x01a3_00a9_3bcf, 0x018d_b9ad_96a9, 0x0009_5c21_fecd, 0x003d_f20e_4acf,
    0x007b_0539_4f46, 0x0072_b0de_0ccc, 0x00f8_c759_ee8c, 0x00a1_2fe6_16a3,
    0x01ad_5de4_66b8, 0x001b_d329_666b, 0x00cc_98e6_98e1, 0x0159_3a5e_3bc1,
    0x01b0_9bc8_d7b7, 0x00d7_d95f_6064, 0x0146_d56d_fb6b, 0x0151_ace7_f0c7,
    0x0052_35f4_7104, 0x0160_552f_2bd9, 0x00b8_5d71_1139, 0x01b9_909e_4c5e,
    0x01e9_fd63_83b8, 0x0013_aa2a_4a94, 0x01cd_28f1_9398, 0x0132_8cd2_adcb,
    0x0036_f95e_901d, 0x01ff_1761_808f, 0x01bf_5cba_1d0d, 0x0152_fb02_1b19,
    0x00f9_ecfc_3a61, 0x00ac_948d_2cb6, 0x000b_7e17_88fe, 0x0121_f739_dcb4,
    0x01c8_5a9b_2558, 0x005e_91fd_6423, 0x0198_6779_c35a, 0x0014_6247_fa70,
    0x0191_60cd_13b4, 0x00e9_90eb_e27a, 0x0121_ff3e_e3c3, 0x0103_173f_f5e4,
    0x0099_9d1f_af27, 0x01be_d034_cdb9, 0x0129_c223_7599, 0x00b7_3e6e_84ac,
    0x014f_edd6_c98b, 0x0032_fddc_f527, 0x0112_f9e8_b254, 0x01c2_b79f_0489,
    0x00c9_2da5_c461, 0x01eb_749d_5dea, 0x0146_d88e_7d76, 0x0127_75e5_2da6,
    0x019c_87d1_7e43, 0x007c_8ffc_74d3, 0x0015_f7e7_92e6, 0x0028_d0ef_5231,
    0x0074_8148_e4ec, 0x009f_d9a4_63f5, 0x0159_974a_b0d1, 0x007b_2e95_390b,
    0x01ff_14f5_ac33, 0x001f_655a_5054, 0x00ac_f593_d41a, 0x0089_8c14_02a1,
    0x0093_b3e0_e6ae, 0x018d_45df_2de5, 0x011e_e6cb_87ce, 0x0109_36d1_1081,
    0x01ca_e459_9782, 0x0188_e5a8_50b5, 0x00cd_074f_4022, 0x01df_ca5b_7190,
    0x00b6_8e62_b82b, 0x016e_8dc2_307b, 0x017c_57ec_8ddc, 0x0028_2604_4499,
    0x0106_bc22_fc2c, 0x00c1_4f10_5ed4, 0x015a_f4d3_f42a, 0x0137_e93a_480a,
    0x0184_9474_f50e, 0x016d_ba23_0a81, 0x0017_3918_3125, 0x0135_fefc_57c4,
    0x00b3_f2d6_34ea, 0x0029_710a_c92c, 0x0148_cf64_1ae9, 0x0067_08f2_4fcd,
    0x00f9_4815_f6fa, 0x0121_607f_ebda, 0x0080_5bb5_d7ec, 0x01c3_9320_b045,
    0x0062_000c_dbc8, 0x014a_c177_b4a5, 0x0006_3a51_304e, 0x0124_4362_7fa6,
    0x01ac_d66a_314c, 0x00f3_b5be_d960, 0x0151_7dc7_8a4d, 0x0168_3948_1f18,
    0x007a_a607_9abe, 0x01f6_e9c6_f5e5, 0x0094_2c5b_ae28, 0x0045_8cd0_f0a5,
    0x0143_2717_dd98, 0x00e3_9ebb_3ef5, 0x01f0_115a_b508, 0x0166_2d90_cacf,
    0x004d_1177_e1e0, 0x0092_137e_93e7, 0x00bf_5c0f_f11b, 0x01a8_1e81_568d,
    0x011a_af91_d931, 0x0189_22c9_bfba, 0x01d6_8ad7_4405, 0x00a7_8bce_4d95,
    0x010e_dea8_ed9f, 0x002f_6bdf_7c6e, 0x0146_0858_efb8, 0x0062_8b39_bfa1,
    0x01c3_215f_60ab, 0x018e_a1a4_6e45, 0x0019_bf64_425e, 0x00d3_10f8_1754,
    0x0171_0d4c_011f, 0x018a_69b6_11f9, 0x00de_bbe1_d511, 0x002f_bfef_bb73,
    0x0097_f0e7_392e, 0x00ea_0d6b_3747, 0x00ee_8545_9226, 0x019f_66c0_749f,
    0x01db_9435_2bd4, 0x0171_dd78_c628, 0x00cb_ba32_d9b0, 0x00d5_e371_e1d0,
    0x0137_c3fc_a430, 0x006a_d97a_92e9, 0x012f_79b3_81ef, 0x0059_b998_e941,
    0x00b2_e255_ed67, 0x0013_354c_8c86, 0x01aa_40ef_5488, 0x00b9_5adb_cc0d,
    0x01d4_897b_2626, 0x0116_6b9d_fc1a, 0x0178_4c28_be7a, 0x0010_15da_b617,
    0x005e_cb96_cd36, 0x0081_6005_54a2, 0x01f4_8f17_58cb, 0x0110_2cad_72f0,
    0x010f_94a3_df50, 0x00db_3840_0c7e, 0x01d2_9e72_e330, 0x0059_698d_378f,
    0x0167_078a_ad77, 0x0048_a99e_cc6a, 0x014c_574a_be4b, 0x006f_e03d_48c3,
    0x01ba_f5de_cb77, 0x012f_db9e_349b, 0x0103_664d_422f, 0x0075_9977_9f7a,
    0x01ee_922c_c3aa, 0x0162_7678_2f89, 0x0037_c979_c3bc, 0x00cd_ba2d_c35f,
    0x0052_f84e_e994, 0x01ef_e6aa_735f, 0x015a_1639_ed91, 0x009e_3a59_8da6,
    0x0019_a86c_a9a9, 0x0097_0dad_b02b, 0x00d2_d223_9539, 0x01b5_a9a7_d43a,
    0x01a2_8d71_1304, 0x0073_027d_c257, 0x0086_9722_6ebb, 0x0091_ab09_256d,
    0x00be_79f6_ad45, 0x001f_077c_5229, 0x0078_5497_0278, 0x00d3_9af3_2496,
    0x0195_054c_a9ac, 0x01c1_a949_a25f, 0x0199_8739_24c4, 0x0184_7acc_8563,
    0x004a_cd98_a710, 0x000f_0402_0319, 0x014d_e092_f710, 0x0030_5e1b_e573,
    0x015b_d8fd_c33a, 0x015d_07db_004c, 0x00ab_d7e9_1181, 0x008f_5bd0_f053,
    0x00cb_e19f_767a, 0x010b_e8ed_0709, 0x0134_b8d9_b6ae, 0x0190_e4ec_260e,
    0x003b_dcbc_a890, 0x0112_372a_27ed, 0x001f_eb58_f771, 0x01c4_cfbe_06df,
    0x00de_e5b1_7d82, 0x0187_1a77_4e8f, 0x00fd_5def_58a5, 0x012e_c074_eb5f,
    0x006b_55ac_1c67, 0x00c5_ffe4_7891, 0x0073_b6ce_69a7, 0x0066_b0e0_f585,
    0x0100_2486_23a3, 0x0016_392f_6f04, 0x007d_5074_7cb4, 0x01e1_9f28_f0ae,
    0x014f_8084_fd3b, 0x018f_1b54_7e66, 0x00d2_8e25_8b80, 0x0174_de91_1918,
    0x00cf_50ac_b1ff, 0x00c8_ee94_66da, 0x00d6_3367_4cfa, 0x0032_3c58_6885,
    0x0128_5f5a_641a, 0x01f6_d00c_f4d0, 0x0106_da5c_cee7, 0x0080_20be_acb9,
    0x01af_2bb7_7ef3, 0x0007_a2f6_9285, 0x00ea_b509_d74b, 0x0170_b347_4645,
    0x005a_4dae_3dea, 0x002c_4336_26d2, 0x019d_15ff_d4ea, 0x0114_7744_815f,
    0x015d_daf4_34a9, 0x0090_2507_6210, 0x0093_9368_b89b, 0x0185_9a3a_07e7,
    0x0159_db29_09c6, 0x016d_8f3a_0103, 0x0190_5ea3_25b2, 0x0193_81a1_15aa,
    0x01f9_ea10_4107, 0x001e_79cd_536c, 0x00a9_a3ac_60e8, 0x0070_d66c_adb1,
    0x00bc_17a0_22c7, 0x0116_da46_4f50, 0x00d8_3ca1_3b7d, 0x009d_cfec_097c,
    0x01a3_240b_f4a1, 0x0114_26b7_932e, 0x01ac_df2b_6bb1, 0x00c1_0683_210d,
    0x007d_7a73_82be, 0x005a_6740_670d, 0x015b_4c25_f379, 0x00f3_cb47_1e8b,
    0x0171_4233_b4d7, 0x00b3_28dc_549d, 0x0123_e124_ab06, 0x00c8_d2af_806c,
    0x019b_6c89_25ca, 0x0064_f327_983d, 0x00dc_5a2e_66d8, 0x01df_61d3_d025,
    0x0070_fac1_9125, 0x0066_17ff_9162, 0x01da_1d95_03ab, 0x01ee_6af2_5502,
    0x0032_9c6d_86ce, 0x0157_e5f0_38e1, 0x00cc_6dc9_7cac, 0x0024_778a_626b,
    0x0138_894b_b342, 0x007d_6f00_2e55, 0x004b_9b47_64cc, 0x00a0_cbc5_d154,
    0x012a_ff49_8e59, 0x0001_653b_5202, 0x0149_b66c_db7a, 0x004c_ffac_a152,
    0x002f_69b2_b280, 0x00ae_d4bf_fbb2, 0x01dc_f40a_92c5, 0x01d5_606c_fc4d,
    0x01d2_8844_5a1d, 0x00a9_a40a_7a93, 0x016c_34dd_7a7a, 0x00e3_0c44_e5dc,
    0x00e7_596a_eb7b, 0x00d8_ec1e_4dac, 0x019a_cb07_c581, 0x01c5_9956_a5de,
    0x005e_b32f_e0f7, 0x00d4_f39a_b374, 0x0041_c094_6c78, 0x01a8_ca1f_2d3f,
    0x001a_ebd3_c67f, 0x00ed_6dda_9095, 0x000d_4eb3_edb9, 0x0078_19d2_a6ee,
    0x0045_b8f9_98fa, 0x003c_0201_308a, 0x01e0_ffab_3f53, 0x01e6_cc32_a780,
    0x0104_5e55_0c09, 0x0083_c4a6_bfaa, 0x0022_b798_a1f7, 0x00f4_2f1f_db24,
    0x01bb_fc98_37f4, 0x00ce_03a8_c117, 0x0066_b874_6cea, 0x0053_4be8_e3d1,
    0x0037_c525_8685, 0x00ed_86de_a0ed, 0x015e_cb7c_e911, 0x0124_5834_d414,
    0x00c6_ec05_2f56, 0x009d_f26a_b706, 0x01e5_d6ee_d381, 0x0067_5416_f65f,
    0x007e_e172_ae2b, 0x0053_bff7_9a11, 0x01bf_a2b5_3c01, 0x0116_9abb_dc78,
    0x0112_7893_424a, 0x0128_bd3e_a4de, 0x0147_e8cf_6371, 0x0044_4d0b_3b3f,
    0x005d_2e9b_8c8f, 0x014e_c21f_af96, 0x002a_021b_8364, 0x0188_855b_f812,
    0x0168_438f_3dfc, 0x01b3_a7ea_750d, 0x01ff_1a13_4379, 0x010c_8c19_9da7,
    0x00ec_9a72_4ddb, 0x0011_f6a7_1c4d, 0x00ad_5318_169e, 0x01b1_5646_6b97,
    0x0184_6159_b5ea, 0x006d_0af1_b37f, 0x00b7_d565_9b5f, 0x0163_fadd_70c5,
    0x0128_7c7a_d5c8, 0x00f6_df36_dfb0, 0x0188_bbbf_3a19, 0x0141_6261_9351,
    0x0186_af9c_4e52, 0x0009_58ca_4268, 0x0080_44bb_1794, 0x0071_113e_2ff4,
    0x01fc_6621_1707, 0x0182_7c74_0954, 0x01ff_95d0_6cc4, 0x01b5_4ff6_85c1,
    0x006d_88de_7741, 0x01c9_27e3_eca8, 0x0042_c6d5_652b, 0x01c2_8d92_cb19,
    0x01b5_779b_48d9, 0x0134_4e24_a9ce, 0x0031_b6b8_c707, 0x011c_4a8e_b4e2,
    0x0199_5b6d_9d79, 0x0119_3ac8_315f, 0x017f_611b_c3b2, 0x0099_02c8_5dc7,
    0x010c_a172_8539, 0x0009_a5bc_f7b6, 0x00c8_75bf_e3b9, 0x01c8_0ee1_752e,
    0x000b_61b1_b07e, 0x01e4_f755_2473, 0x0141_ae81_2d3a, 0x0099_2bdf_44f2,
    0x0177_cb82_03f3, 0x000c_c54c_9a54, 0x012b_5f0a_47a0, 0x0117_884a_0232,
    0x00bf_7d50_d7b6, 0x004c_6c3f_6879, 0x0037_a66b_633f, 0x010f_d1a2_75ee,
    0x011e_5790_554c, 0x015c_168a_72a3, 0x013c_7537_eaba, 0x00ca_23dd_9aea,
    0x012d_bb51_fc2a, 0x014a_e352_eed0, 0x0020_e166_257f, 0x011c_e40d_3d1e,
    0x010f_4929_db1b, 0x019f_d90a_2f45, 0x004e_e5bb_04b0, 0x0060_b378_e062,
    0x01c8_6c08_e115, 0x0008_49e4_e2de, 0x0116_1e70_e4cf, 0x0087_d520_00e4,
    0x01c0_acc1_58ab, 0x0136_f7d2_d252, 0x0158_c0d6_986c, 0x0117_4bd3_3519,
    0x005a_7231_8656, 0x00d6_e9d0_f11f, 0x0188_b600_5e76, 0x00ec_bb0c_a621,
    0x0113_1eb3_cde3, 0x01de_6369_b749, 0x0153_d9cc_b5ac, 0x0060_22a8_be13,
    0x0036_8ec8_3b60, 0x0134_5506_440d, 0x00ae_9863_319e, 0x00d9_6362_3257,
    0x0097_fe23_da88, 0x01cb_2340_3fe1, 0x0190_a0c5_4bc2, 0x0058_5f24_8208,
    0x01be_4b97_4e1f, 0x0057_1a78_ba23, 0x0113_f244_d483, 0x00f8_780c_4818,
    0x00db_51a8_2200, 0x016c_49ec_533a, 0x0158_5317_f190, 0x01fc_7010_5ed9,
    0x01ae_4053_905e, 0x01dd_a20b_1e04, 0x014e_0656_7fda, 0x01fc_f73c_c8f6,
    0x00ab_374c_f2e1, 0x0075_2720_033a, 0x004a_6432_e5dc, 0x01c0_f08c_dd28,
    0x005e_b870_43f7, 0x01ec_3806_7d90, 0x01fe_f4cf_5059, 0x0170_0dac_4882,
    0x000c_dd79_2496, 0x0019_94ef_8c66, 0x0166_9a97_8fac, 0x0093_695c_2811,
    0x008c_8465_1653, 0x0130_5144_f59e, 0x01e5_597b_1863, 0x0154_d261_5543,
    0x01e5_a8c0_8afc, 0x0084_1f1b_18ec, 0x0147_6b7e_29a0, 0x0107_31c0_52f3,
    0x0087_4410_7d0d, 0x00c6_1901_934d, 0x00ec_347b_7b88, 0x00e7_476d_701e,
    0x0006_e867_aafb, 0x00e1_d030_44bb, 0x0065_8319_e530, 0x0143_3637_3a0e,
    0x016c_b93a_933e, 0x01b8_b6d9_5bb4, 0x0004_a541_3171, 0x0010_ff86_2197,
    0x0111_5f32_02f1, 0x01d6_f7ed_1430, 0x01b3_73d7_ff0f, 0x0025_cfa7_3de1,
    0x00cc_6b38_943a, 0x0075_9f92_4c6c, 0x0148_6220_0a2c, 0x01b7_4c7a_4cb4,
    0x012d_44b5_851b, 0x00ca_97f5_aeb9, 0x004d_3670_1f59, 0x01de_88f2_493c,
    0x0042_0aba_a0f6, 0x00d8_92ba_bdd5, 0x0062_fe9b_8b99, 0x006a_1c54_d76e,
    0x00b8_7dde_2e3b, 0x00c7_7657_afe3, 0x006b_72f0_9030, 0x0184_b68b_baea,
    0x0022_a6cd_479e, 0x0016_6f0c_5ba7, 0x0180_9f47_bad3, 0x018e_d0cf_68af,
    0x01ae_1368_afda, 0x00ca_2d64_6634, 0x0132_0f49_e286, 0x00d5_6c03_85bd,
    0x006b_6f80_ffc8, 0x00b0_615b_2264, 0x0125_287f_0185, 0x0011_0b2b_ee8f,
    0x01e9_004a_8e70, 0x015d_de15_e068, 0x0104_3869_3ad9, 0x015b_aa26_fb1e,
    0x0183_bfc6_6070, 0x0080_4a41_b38f, 0x005d_bd4a_1a67, 0x00b1_0be9_0245,
    0x0050_b250_494f, 0x001d_ac32_3cc2, 0x0175_a189_df75, 0x012d_df98_fe04,
    0x0108_62cc_7c48, 0x001d_c90b_7cd4, 0x011d_bf85_1ad7, 0x01a8_7ae5_dc11,
    0x017f_7fa9_e215, 0x0130_38ea_d8fe, 0x0118_1bd7_be69, 0x0119_086a_04d8,
    0x0158_9d56_9b98, 0x01e9_b004_24f1, 0x0080_ff18_96e9, 0x0090_d4c2_2245,
    0x001e_af96_72f6, 0x004a_fcdf_31fa, 0x0145_d412_a4c3, 0x00ab_1978_5bb1,
    0x01d9_6d08_2bbe, 0x0142_86dc_e6ec, 0x0160_fd92_f8fb, 0x012f_dcb8_b2af,
    0x01a0_1a34_25da, 0x0133_1628_49bf, 0x017f_312a_577b, 0x00a2_8b92_bb72,
    0x018e_5fbd_6372, 0x01df_71ce_acee, 0x012c_5bdf_9515, 0x00e6_c1ac_ce21,
    0x009d_2569_9cd2, 0x0152_70b9_1c81, 0x0126_e0bb_0707, 0x00fd_fae6_13f7,
    0x016e_8679_d842, 0x01d2_c9b4_89ad, 0x01ab_d33e_45cc, 0x0192_f2c1_5d5d,
    0x0171_3ce8_276c, 0x010e_5651_6b04, 0x0137_39a4_e4b7, 0x0076_c27d_6558,
    0x01ee_e4c4_1ee2, 0x0064_d3c6_c09b, 0x015d_2944_dbe0, 0x00c6_1d6b_a698,
    0x0180_dffc_6541, 0x0150_61fb_4932, 0x00df_6e88_5666, 0x0127_eec2_46c8,
    0x019b_51fd_70c4, 0x0118_7e4d_f717, 0x006a_8b0a_8f1d, 0x0119_4239_0170,
    0x00a4_2196_5912, 0x01f7_26d5_1351, 0x0031_7c4d_aa03, 0x01ac_dc1e_ce78,
    0x003f_d243_f3eb, 0x00fd_c16c_4efb, 0x002f_1237_1b10, 0x0039_b6e2_0f1c,
    0x0169_ab73_3a9e, 0x00fb_398f_0a72, 0x0027_61cb_ddc4, 0x00d6_a06b_bcce,
    0x0056_f492_7d58, 0x00c5_a437_2537, 0x01b9_0381_e8d6, 0x0091_b9fa_e2d8,
    0x016e_e910_633b, 0x00a7_bc7d_d016, 0x00e7_25ab_716a, 0x0182_27cf_3c37,
    0x00db_a6f9_1ce3, 0x01fe_dfb8_bf3e, 0x015d_e029_22d1, 0x00d1_1529_aaa3,
    0x01a1_62d5_f7f1, 0x004b_0b4d_9d3e, 0x006f_5039_7572, 0x0134_8ebc_fc2e,
    0x009d_f431_57d4, 0x0199_9827_f76a, 0x0146_7f25_70f3, 0x01eb_a5df_f8c1,
    0x010e_e96b_1e22, 0x0107_be5e_ddbe, 0x0141_1df4_97c3, 0x00d2_26a1_66c0,
    0x00f3_d092_e815, 0x0058_e2ed_63ce, 0x013b_d7ba_8df6, 0x0129_8e4f_35ce,
    0x011b_e984_5c0d, 0x00b5_3afc_8ffa, 0x0067_af68_8e82, 0x010a_41cd_33d7,
    0x00f6_838b_aeab, 0x0007_77ac_0858, 0x01df_cd3f_233b, 0x0034_944e_7100,
    0x0124_87a3_b270, 0x00ec_12b9_190d, 0x01bb_c01a_91c7, 0x0065_6b8c_243b,
    0x004d_bb62_b088, 0x013a_b915_c3fa, 0x01f8_a3d6_2167, 0x0000_9a46_896a,
    0x00fb_5f4d_2fea, 0x0111_6bc7_e342, 0x0183_e9fd_5a14, 0x01d0_c98a_1c5e,
    0x015d_4596_76d3, 0x00f7_60f6_f9a3, 0x012c_66e4_0456, 0x0173_ddc7_71b1,
    0x0007_db0a_1578, 0x01f3_ce6f_5029, 0x0077_d9f5_679b, 0x001a_c5f8_5b64,
    0x0004_c672_cd5d, 0x0156_8065_5ee8, 0x0186_876a_1255, 0x0077_4469_842e,
    0x0132_71af_9a24, 0x006e_89c9_fb2c, 0x006c_1d9e_2be1, 0x00f0_741a_cf5a,
    0x002a_a8d8_8fee, 0x0003_8b16_ed82, 0x00e9_cba3_9dd7, 0x0091_2e60_05cb,
    0x0125_1d66_08a9, 0x00e0_b6ff_59fb, 0x00e0_8ea9_5b44, 0x0145_73ad_e1a2,
    0x015b_7c8a_4880, 0x0030_578d_9c9a, 0x01fd_9016_e391, 0x00a0_f621_9300,
    0x00d6_556f_9ac1, 0x001c_1012_de94, 0x00b9_7189_3af5, 0x011c_a283_b57b,
    0x010c_64cb_9e05, 0x01aa_714b_75b4, 0x014a_ca14_1713, 0x0125_c7fa_bffd,
    0x00a9_8163_2be8, 0x0113_c56a_7e7a, 0x00d2_ba19_57ef, 0x0160_d918_b563,
    0x00fb_8ce1_8705, 0x0071_27d3_be2c, 0x012a_3258_12ac, 0x00a0_ae4e_4033,
    0x007a_50e3_6769, 0x000c_0f50_d2c7, 0x0008_90ba_241c, 0x010b_3a09_1b74,
    0x001d_fa79_7966, 0x0195_2caa_9bad, 0x009d_8252_84d7, 0x004b_a943_7737,
    0x01f3_3933_c933, 0x006d_53c8_10e2, 0x00e4_a1f0_20fb, 0x0152_a41f_c39d,
    0x01e6_f1b3_bd37, 0x0091_ea70_722f, 0x0028_924a_2a6d, 0x01ec_4fbf_a9ab,
    0x00a9_89d4_9062, 0x0079_387b_2a25, 0x0054_b2e5_3b73, 0x014d_c386_8cf9,
    0x014b_9d29_0525, 0x00b7_77ff_6c35, 0x0097_7474_6651, 0x014c_55cb_6390,
    0x0057_1a46_88bf, 0x0031_2aa0_996a, 0x0153_2efa_1a5c, 0x00df_ec4a_dedf,
    0x0096_8187_0c1a, 0x00c3_0efd_0321, 0x001e_872e_ec40, 0x0010_aee5_189e,
    0x01b5_4c11_e3b0, 0x0105_4076_b18b, 0x0120_09ba_b10f, 0x01f5_de64_8026,
    0x01fe_c2d0_89f6, 0x01ae_79a3_d351, 0x0160_5584_6033, 0x014a_18cf_43cd,
    0x002b_4868_b059, 0x00f3_9d11_8add, 0x0184_fa8a_563b, 0x005d_cd75_adef,
    0x01f8_67bc_a4ca, 0x0072_2f82_30e0, 0x01c7_739a_0b92, 0x014d_27af_cac6,
    0x01ff_4483_a337, 0x01ee_3e7e_c65e, 0x015b_df9c_f572, 0x00ac_55b1_83d8,
    0x01bd_7ba5_eb7e, 0x013d_6f28_d36d, 0x0094_483a_59c7, 0x0197_1a68_11fc,
    0x01d0_1f6d_c8e5, 0x01d7_041a_8164, 0x00cd_4bb6_6307, 0x003c_c8ba_e5d2,
    0x0040_a28f_0a0e, 0x0028_ff94_2508, 0x010f_7db1_4df0, 0x0097_0a05_aa25,
    0x004a_714b_291e, 0x00c7_01e4_e2e9, 0x000c_3311_c809, 0x0085_c5af_52eb,
    0x00d2_a0dd_cc3b, 0x01b8_6f41_0788, 0x01ed_e348_4ca6, 0x0040_7409_1b8a,
    0x00cd_ace8_c340, 0x0096_4a81_7ec7, 0x0053_c4b1_e08a, 0x0071_f3cf_affd,
    0x01c8_47b0_f581, 0x01f9_5bfc_003b, 0x001c_8666_6096, 0x0044_8e6a_5caf,
    0x017d_9562_fe30, 0x014b_11e2_c939, 0x0146_ba94_fe60, 0x0139_17b4_bc7c,
    0x01bf_ab59_51e7, 0x0192_79b2_0f95, 0x0021_6bd3_6a1b, 0x0178_88e3_9bb5,
    0x00ae_8079_2a03, 0x0082_98ab_4acb, 0x010d_fed4_9850, 0x00cb_166c_8a85,
    0x0101_beba_4048, 0x0085_5cff_4fc0, 0x0184_9168_c1c5, 0x010a_38f6_2987,
    0x0049_cf1f_189b, 0x016a_2876_af49, 0x0087_fa6e_2a56, 0x0062_c7b0_15c2,
    0x00d5_c29b_c4a1, 0x0062_0917_56ed, 0x0051_8e01_968d, 0x003d_09d8_3c3a,
    0x0054_f111_c2e2, 0x0155_5bb7_41cd, 0x009d_bccd_3073, 0x01ec_92ec_e825,
    0x0001_1299_8bc8, 0x00e0_2379_d7c8, 0x0149_f696_fd0f, 0x01b7_137d_7768,
    0x01a3_b030_519c, 0x01e1_d806_bfcd, 0x000b_c65b_06e8, 0x0198_3059_408a,
    0x01e9_fe1f_eca9, 0x0013_2c0a_95aa, 0x00a6_dc51_19e7, 0x00fd_58b2_db10,
    0x0137_e20c_98e5, 0x0096_8688_531c, 0x00a8_1dc8_92a8, 0x00f8_61cd_9997,
    0x01b0_d2b1_2efb, 0x0000_bea3_4e27, 0x00db_665c_24f9, 0x00e9_0e72_2940,
    0x0010_6d98_9a38, 0x002e_84aa_c68e, 0x00f3_e4a7_0319, 0x01d9_d002_369b,
    0x001c_31c3_1105, 0x0091_514c_5856, 0x01c4_9b1c_053b, 0x01eb_ef82_c4a5,
    0x01ec_8692_0675, 0x0125_3eb1_b8cf, 0x019c_293a_fdd8, 0x01e6_a30f_c496,
    0x001c_f754_69e1, 0x0121_6b1b_956c, 0x006d_c53c_f47c, 0x016b_338c_b908,
    0x007b_c898_31bb, 0x012b_fcea_ee52, 0x0060_2827_afa6, 0x0129_bef4_e086,
    0x019a_de8a_837b, 0x008f_365a_39dc, 0x0133_31a3_ed5d, 0x0167_bf56_7e12,
    0x0181_7eb5_403d, 0x0116_a0ac_445b, 0x0151_f5a1_ae96, 0x0035_5fba_fed3,
    0x0172_1a85_6f18, 0x0039_8c5f_b690, 0x018f_de25_2209, 0x009b_c1ef_fd8d,
    0x015e_0583_68cb, 0x01ef_19f8_672d, 0x00ef_eb28_bcb5, 0x0085_97fe_8243,
    0x01e3_558b_7241, 0x01f0_6d2b_5a4d, 0x0067_5316_6ca7, 0x019d_0525_2f87,
    0x007e_ce87_19c7, 0x00ab_ada7_be10, 0x003b_7bae_0915, 0x00b9_20db_fa1e,
    0x0077_c3d7_50ef, 0x00c8_a77b_91e5, 0x013c_1251_1766, 0x01bd_655e_970d,
    0x00d5_f0a1_0e67, 0x005c_151f_0386, 0x00f3_6c12_3216, 0x0073_3590_bd56,
    0x0013_4800_baf0, 0x0166_cf1d_e51f, 0x0117_dd06_e881, 0x0129_a092_1efb,
    0x00a6_ff0c_a5c4, 0x008e_a200_b050, 0x0034_f938_7b94, 0x01d3_723a_d2de,
    0x018e_5af2_b62e, 0x010b_d979_c6df, 0x014d_3396_07a4, 0x01b8_8544_514a,
    0x003a_4450_d2d8, 0x01a6_1117_a18f, 0x01e2_1725_e539, 0x0123_cf73_64fd,
    0x0179_908b_5b01, 0x0069_ec91_a349, 0x0094_5679_ec02, 0x00e6_a4ca_1b53,
    0x007e_2c61_2da0, 0x01f5_68f3_2763, 0x01b5_64f9_8a0f, 0x0020_e11f_4d89,
    0x0090_c3ea_1433, 0x014c_7e71_acd3, 0x00dc_4e4b_061f, 0x01c8_1388_6671,
    0x0099_8741_f58b, 0x00fe_30e8_0a8f, 0x0060_7de6_a503, 0x0148_4657_259d,
    0x00e5_3c33_5255, 0x0174_d9bd_2dc8, 0x007d_39cf_a317, 0x01c0_992c_6850,
    0x0112_351c_0a60, 0x01e0_d5e6_cd44, 0x0066_7998_400c, 0x0042_dfae_f259,
    0x019a_bbd3_ed08, 0x014a_d9c7_ab72, 0x0056_389f_f426, 0x0194_27c1_236c,
    0x01a4_063a_9d61, 0x00af_87f9_7b4d, 0x0166_6612_b513, 0x01f2_3ac5_39e4,
    0x01e8_73a1_cf84, 0x0170_1f1a_eddc, 0x018b_22a9_8565, 0x01dd_c760_ab43,
    0x0022_da16_9d55, 0x00f1_4168_93de, 0x00ee_14c0_e6db, 0x0192_52aa_e1f5,
    0x000b_4fb6_141d, 0x00c2_b99d_a72c, 0x019e_c587_f23a, 0x01f4_4dc5_cd77,
    0x0101_573c_5f0c, 0x00c8_590a_c48f, 0x01f1_d4b1_d98d, 0x012a_e39c_0c1e,
    0x016e_4699_6bb3, 0x00d9_0707_1ae1, 0x00ee_8924_7a5b, 0x0127_53bf_dbce,
    0x0079_06c2_f180, 0x0066_32db_9a04, 0x006d_d60d_443f, 0x006b_8593_27f2,
    0x018f_08bb_0ad5, 0x0100_dd9d_760a, 0x001e_8c64_4d0d, 0x01e8_edb7_40dc,
    0x016b_a7ff_a32e, 0x00bf_bda5_d84c, 0x017f_df13_7ad8, 0x0077_7ffe_01db,
    0x013b_9a93_d0a6, 0x0032_bde7_48a4, 0x0104_f87c_4ba8, 0x0110_2e54_38a0,
    0x0179_0599_f8b8, 0x006e_0e97_a55c, 0x0098_0d89_8e6b, 0x0140_3582_5c86,
    0x0199_ded7_a4bf, 0x0138_ab00_50f4, 0x00f8_c194_0dda, 0x0129_ad4d_2a65,
    0x0076_a9ac_a541, 0x0133_2c5d_323c, 0x01db_a511_3a26, 0x01c6_6d9f_5293,
    0x017e_232d_0d9a, 0x0099_7691_5862, 0x0191_3b2a_7244, 0x00b2_ba53_4e99,
    0x0160_d1bf_0cf7, 0x000e_b992_2d30, 0x002e_3f78_8035, 0x01d1_58a6_dfbe,
    0x014e_9413_f1fe, 0x0132_d8c1_9753, 0x014b_4f73_c0c3, 0x001f_33b2_8cfd,
    0x0072_a8fa_d428, 0x0151_4364_455c, 0x002d_98ab_517f, 0x0167_cedc_2d26,
    0x01bb_3c03_9f02, 0x0144_f9f0_6518, 0x0086_d324_dd1c, 0x012c_4eb2_6f57,
    0x0078_ea84_3219, 0x006a_55f7_e6b6, 0x006d_7b34_c6e0, 0x0101_6cfe_b2ec,
    0x0042_1567_6871, 0x0156_b12b_4c31, 0x0058_6fdb_e5d9, 0x0030_212d_8b7b,
    0x01cc_7382_e8cd, 0x006c_a5a2_d1ec, 0x0087_5c7a_96f2, 0x0171_0b90_c031,
    0x0010_d1d8_c47a, 0x01bf_01e5_1bb9, 0x00c1_793e_8778, 0x01b3_26c2_d7e2,
    0x00c5_310a_5aab, 0x011c_d7b3_c8e5, 0x018d_5912_ebaa, 0x0174_50d7_23c4,
    0x00ab_2fa3_4dcd, 0x012c_7246_a7bc, 0x01b3_6529_6a34, 0x00c4_7f95_4743,
    0x0050_467b_a354, 0x002c_c0b8_16e1, 0x0175_98c4_f17d, 0x01af_4283_647b,
    0x005a_5d53_79ec, 0x01aa_ecdd_c00d, 0x0132_5e99_894f, 0x0039_35af_6226,
    0x0184_50e1_eebe, 0x009a_dfe3_a3d0, 0x0079_383c_9291, 0x0022_1210_2d63,
    0x0153_d2be_7427, 0x00fb_53f1_bb55, 0x0091_8bed_2982, 0x010f_53f0_d799,
    0x0052_6ff4_d1b3, 0x00bc_9063_5acd, 0x0145_3b55_ba1d, 0x0107_9d69_bec4,
    0x01af_3934_bd83, 0x00b2_9697_fc3f, 0x0108_533d_e8dd, 0x0129_68f6_7594,
    0x011c_be9c_b3ed, 0x000b_ac4e_20c3, 0x0134_251a_ba54, 0x014e_aaa2_0df5,
    0x0152_8aec_9718, 0x0170_983c_c1ec, 0x002f_097e_68c8, 0x005f_746b_9d5c,
    0x01ca_56b3_10f3, 0x0039_7253_0948, 0x009b_a80f_3bce, 0x016c_5f86_f30d,
    0x0119_664a_bd83, 0x016f_dda2_5a23, 0x0176_31d1_a1d3, 0x003d_72d2_b20d,
    0x011f_52f5_697b, 0x001d_6b6d_864f, 0x0043_d642_2f0a, 0x016e_f1aa_2e0c,
    0x00e1_a379_0b11, 0x00f1_667a_531d, 0x011c_c7bb_bcda, 0x01ac_e1b0_3f19,
    0x018b_d83b_1d52, 0x008a_5de6_8db4, 0x018f_bf08_cfef, 0x0082_0196_39cc,
    0x00ab_6b9a_8d3b, 0x01fc_4a83_8fba, 0x0104_f0d6_7db1, 0x01e4_bd14_4895,
    0x0104_93fa_ac76, 0x006f_1ac4_5d00, 0x001b_845a_c503, 0x01a0_71ba_aaef,
    0x017c_e582_2109, 0x01a7_b999_2deb, 0x0164_1ad2_afd1, 0x0198_3f74_b6ee,
    0x011d_d449_1184, 0x0159_7809_c70c, 0x01c9_ee20_7a02, 0x01fb_967d_bcce,
    0x0013_dd5d_0740, 0x01f0_4a7c_f90a, 0x00bd_a526_f2c0, 0x000a_290a_fad5,
    0x01f7_4ef5_e0ed, 0x00ad_155b_c8de, 0x0008_a74e_0d64, 0x01e2_6d6d_2095,
    0x01ef_e9d3_f3fe, 0x00b7_bb79_0080, 0x00e8_3eb7_940b, 0x0019_5a67_21de,
    0x0169_7162_ecc6, 0x01a8_2242_4ecc, 0x00bb_9409_ef14, 0x0088_dc8a_f7de,
    0x001a_109f_bdf8, 0x0126_3a62_24e1, 0x0197_6245_1a46, 0x0123_f808_7a5d,
    0x01ee_61da_2ff8, 0x0105_0f8b_67b3, 0x005b_d05e_0ad3, 0x017f_3054_510b,
    0x01f0_a65c_be49, 0x00c3_8f39_ecfb, 0x01b1_86ad_f1ca, 0x0003_c9d6_cb25,
    0x00b0_d333_43b8, 0x008c_b15b_5077, 0x0083_9608_95dd, 0x00a4_f544_f26c,
    0x0146_37fb_a5b5, 0x00c8_a7c5_184d, 0x0109_ef50_3b91, 0x0110_cc27_f58c,
    0x01dd_15e9_f124, 0x01ed_db2c_a0d9, 0x00aa_fbfc_ea5b, 0x003f_14f9_3014,
    0x01c8_249b_99d9, 0x007c_3667_f5b3, 0x01a6_2b64_0f70, 0x01ee_c6a2_2d2a,
    0x0085_5be9_3f97, 0x00c7_af09_1027, 0x010a_7ffe_2790, 0x0104_2e09_ea4c,
    0x0076_db25_a8e1, 0x0165_4d70_6d3f, 0x0026_0b3a_5e33, 0x0088_f941_9e9e,
    0x01d6_8917_df54, 0x0118_8549_820d, 0x0158_d481_69b9, 0x019e_93fc_dd73,
    0x016b_91a4_7690, 0x01fb_a468_bd65, 0x00bb_64a6_d60f, 0x0157_432e_4097,
    0x0088_a4f7_0f06, 0x00f0_91d8_3cf0, 0x01c4_a6fb_c192, 0x00ad_a88c_3e7d,
    0x00e5_31e1_41b5, 0x0123_7b66_171c, 0x0127_49c7_c2f4, 0x00e2_cee2_69f9,
    0x0026_0340_689f, 0x0195_0050_4a28, 0x0144_3209_9794, 0x0047_cbdd_04d3,
    0x001a_f023_fdbe, 0x01ee_e99e_354e, 0x0012_649e_3aff, 0x01a7_f7a4_9cbf,
    0x00e4_3793_e465, 0x0062_09ca_0402, 0x01ea_dfe9_2cff, 0x003d_acce_d0bc,
    0x0097_a157_58c0, 0x01fc_d58a_eaae, 0x00ff_774c_923f, 0x007a_5654_482c,
    0x01b4_9202_8846, 0x015c_12bf_403f, 0x01e7_5752_19b9, 0x0034_2bb3_1b5c,
    0x01a8_052d_b35e, 0x00e8_274a_d0db, 0x000f_1a93_7c8a, 0x0094_b5e4_8ed7,
    0x0035_2cb6_ab16, 0x015c_ff6a_6f12, 0x01ee_4015_5a64,
];

/// AprilTag 41h12 dictionary singleton.
pub static APRILTAG_41H12: std::sync::LazyLock<TagDictionary> = std::sync::LazyLock::new(|| {
    TagDictionary::new("41h12", 9, 12, &APRILTAG_41H12_CODES, &APRILTAG_41H12_POINTS)
});

