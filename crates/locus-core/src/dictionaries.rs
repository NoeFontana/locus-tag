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
            for rot in 0..4 {
                code_to_id.insert(r, (id as u16, rot as u8));
                rotated_codes.push((r, id as u16, rot as u8));
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
            for rot in 0..4 {
                code_to_id.insert(r, (id as u16, rot as u8));
                rotated_codes.push((r, id as u16, rot as u8));
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
