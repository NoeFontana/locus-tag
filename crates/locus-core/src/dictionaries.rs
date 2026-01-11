//! Tag family dictionaries.
//!
//! This module contains pre-generated code tables for AprilTag families.
//! Codes are in row-major bit ordering for efficient extraction.

// Copyright (C) 2013-2016, The Regents of The University of Michigan.
// All rights reserved.
// This software was developed in the APRIL Robotics Lab under the
// direction of Edwin Olson, ebolson@umich.edu. This software may be
// available under alternative licensing terms; contact the address above.
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
// ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


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
    codes: Cow<'static, [u64]>,
    /// Lookup table for O(1) exact matching.
    code_to_id: HashMap<u64, u16>,
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
        let mut code_to_id = HashMap::with_capacity(codes.len());
        for (id, &code) in codes.iter().enumerate() {
            code_to_id.insert(code, id as u16);
        }
        Self {
            name: Cow::Borrowed(name),
            dimension,
            hamming_distance,
            sample_points: Cow::Borrowed(sample_points),
            codes: Cow::Borrowed(codes),
            code_to_id,
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
        let mut code_to_id = HashMap::with_capacity(codes.len());
        for (id, &code) in codes.iter().enumerate() {
            code_to_id.insert(code, id as u16);
        }
        Self {
            name: Cow::Owned(name),
            dimension,
            hamming_distance,
            sample_points: Cow::Owned(sample_points),
            codes: Cow::Owned(codes),
            code_to_id,
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
    /// Returns (id, hamming_distance) if found within tolerance.
    #[must_use]
    pub fn decode(&self, bits: u64, max_hamming: u32) -> Option<(u16, u32)> {
        let mask = if self.dimension * self.dimension <= 64 {
            (1u64 << (self.dimension * self.dimension)) - 1
        } else {
            u64::MAX
        };
        let bits = bits & mask;

        // Try exact match first (covers ~60% of clean reads)
        let mut rbits = bits;
        for _ in 0..4 {
            if let Some(&id) = self.code_to_id.get(&rbits) {
                return Some((id, 0));
            }
            if self.sample_points.len() == self.dimension * self.dimension {
                 rbits = rotate90(rbits, self.dimension);
            } else {
                 break;
            }
        }

        if max_hamming > 0 {
            let mut best: Option<(u16, u32)> = None;
            for (id, &code) in self.codes.iter().enumerate() {
                let mut rbits = bits;
                for _ in 0..4 {
                    let hamming = (rbits ^ code).count_ones();
                    if hamming <= max_hamming {
                        if best.map_or(true, |(_, h)| hamming < h) {
                            best = Some((id as u16, hamming));
                        }
                    }
                    if self.sample_points.len() == self.dimension * self.dimension {
                        rbits = rotate90(rbits, self.dimension);
                    } else {
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

#[rustfmt::skip]
/// Sample points for APRILTAG_36H11_POINTS (canonical coordinates).
#[rustfmt::skip]
pub const APRILTAG_36H11_POINTS: [(f64, f64); 36] = [
    (-0.625, -0.625), (-0.375, -0.625),
    (-0.125, -0.625), (0.125, -0.625),
    (0.375, -0.625), (0.625, -0.625),
    (-0.625, -0.375), (-0.375, -0.375),
    (-0.125, -0.375), (0.125, -0.375),
    (0.375, -0.375), (0.625, -0.375),
    (-0.625, -0.125), (-0.375, -0.125),
    (-0.125, -0.125), (0.125, -0.125),
    (0.375, -0.125), (0.625, -0.125),
    (-0.625, 0.125), (-0.375, 0.125),
    (-0.125, 0.125), (0.125, 0.125),
    (0.375, 0.125), (0.625, 0.125),
    (-0.625, 0.375), (-0.375, 0.375),
    (-0.125, 0.375), (0.125, 0.375),
    (0.375, 0.375), (0.625, 0.375),
    (-0.625, 0.625), (-0.375, 0.625),
    (-0.125, 0.625), (0.125, 0.625),
    (0.375, 0.625), (0.625, 0.625),
];

/// AprilTag 36h11 code table (587 entries, row-major bit ordering).
#[rustfmt::skip]
pub const APRILTAG_36H11_CODES: [u64; 587] = [
    0x000047b7310b, 0x0009c712bec7, 0x0001127334c1, 0x000b3db82789,
    0x000e495c72d1, 0x000e169b7d93, 0x000159a190a5, 0x000da3830123,
    0x000f1c8dce3d, 0x0002ed68409c, 0x000357ef0a86, 0x000dafad93d8,
    0x000578c43c14, 0x000cf961b690, 0x000884a6edf2, 0x000c43c36636,
    0x000a7e06756e, 0x000fc40927ec, 0x0007310cb972, 0x00098ee86e5d,
    0x00005dd5d489, 0x0008f0355b05, 0x000ca5f7444f, 0x000baaf19871,
    0x0002619d07b5, 0x000a91fed663, 0x00017b9a5baf, 0x00042b5e4e65,
    0x00027f93ad96, 0x000ada726312, 0x000ff07d6180, 0x0004edee1dc3,
    0x0000a5047c3b, 0x0007a222a935, 0x0004992deb27, 0x0009094865c6,
    0x0000ccebe54a, 0x00096caf7ad6, 0x0006f8b31646, 0x000a77b1d878,
    0x000dcdbe966a, 0x0008e6bd84c9, 0x0006a656ed19, 0x00075317b841,
    0x0008ad3d20af, 0x0006efdc697d, 0x0002764204c8, 0x00054c6d469a,
    0x000434aa4d58, 0x00023d80a0ae, 0x00066b2b3f11, 0x000361345923,
    0x000324f256e9, 0x000c22f49b77, 0x0006c431a18a, 0x000c3cb331dc,
    0x000d6a66c801, 0x0003fb4e94a1, 0x000f0deae6cc, 0x000e730ff394,
    0x0008b9d69a64, 0x000569b214e0, 0x000ce6139d2c, 0x0002a8fad695,
    0x000255bb81dd, 0x000ea799184b, 0x000c7adc5a42, 0x0006ebf426b2,
    0x0002c2e1df27, 0x0007a94674d6, 0x000e2c67fc5a, 0x0008e8afb7a0,
    0x000539fbd093, 0x0006463f9f4a, 0x0004ea658ae4, 0x00033ae84d91,
    0x00018fc07131, 0x00097320e9f5, 0x000e3a892ae1, 0x0007cd5bf7cb,
    0x000587442472, 0x000246ade605, 0x000a257a38ec, 0x0006d3a7e7f3,
    0x00046629a346, 0x000130977009, 0x0001595ef46d, 0x00097210847a,
    0x0009445bc544, 0x0009a9a4214e, 0x00006a85ab82, 0x00050d325136,
    0x0003191b854b, 0x000e99d9e27b, 0x0000c7865ce8, 0x00023d6fc226,
    0x000a60e8a67f, 0x0003ffd79140, 0x00014a0cd49c, 0x000eccc5b9ce,
    0x000fbf04a804, 0x0009d9ede53e, 0x000d9af70d92, 0x00047fd28256,
    0x0009d03d4c68, 0x0006145627b0, 0x0005b9159c5b, 0x00079b72a5cc,
    0x000b7fe68517, 0x0004456572e3, 0x00063bf3b689, 0x000ac1d96e2d,
    0x000c727f32ed, 0x0004f76aee15, 0x0009da52396c, 0x00091c152579,
    0x0002b04262bf, 0x000ed4fe13c0, 0x000613db9a0e, 0x000d2e47a43c,
    0x00001781c64b, 0x000142adbdbb, 0x000f37b5c12b, 0x000ee403835d,
    0x0005d98fbc88, 0x0001d0da4305, 0x000de48a88a4, 0x0007564d34f9,
    0x00030635d4ae, 0x0002b20a2d24, 0x00000cb483fb, 0x0009bc24485b,
    0x000be2369b40, 0x00074692e968, 0x000037b56566, 0x0005b8c6227d,
    0x000ed14d250f, 0x000f94037653, 0x0003d5238a15, 0x0006b3de54cd,
    0x000b85c87f63, 0x0007bccc6fb4, 0x0000efcb9937, 0x0008b31e0355,
    0x00004c7c5d2d, 0x000c0ea56969, 0x0006f8ab889b, 0x000f74ce174f,
    0x000156534a28, 0x000555b94170, 0x00035aebd0e2, 0x0001591748b4,
    0x00075a45d10e, 0x000baa674068, 0x000290f9ce82, 0x00023f3e58eb,
    0x000939610d7c, 0x000c56778e34, 0x000bd52f7812, 0x000c20cb3a8c,
    0x0006b79b463f, 0x000f8164cf13, 0x00003b3ecd36, 0x00058131bfa4,
    0x0000d1f2c5a6, 0x000a69a36917, 0x0002a081d1c8, 0x000c97a8b458,
    0x000d91fe0f7e, 0x00007ee46459, 0x000aa3b16202, 0x0006d54ec04b,
    0x0004dd668c0c, 0x0000990fdb78, 0x000407350fb7, 0x00076841b2f5,
    0x000a5966113d, 0x000311f9f7bc, 0x00018b235e3d, 0x0009da814a4a,
    0x0000a80f6712, 0x000f1e8e2b1b, 0x000f24688bba, 0x0004329de25f,
    0x000a8b28e75b, 0x000314aa7a8a, 0x000af3286b2c, 0x0000e581cf74,
    0x000d9f692d23, 0x0007576055a7, 0x000d6916c6b2, 0x000085154902,
    0x0007798c0ede, 0x000bba9c442d, 0x000c7b789b09, 0x000d7f4af4d1,
    0x0001b3e488e9, 0x0002a325cc30, 0x0004adb9ac72, 0x0001cfe1167a,
    0x00096ff002ed, 0x000db1987694, 0x00046fa0bbe8, 0x000c8d02afca,
    0x000131bd2a9d, 0x00096f9dda08, 0x000a85c4524d, 0x000c7271e03e,
    0x0002c407550f, 0x0005529fe826, 0x00029bc50b18, 0x000194cfa5b9,
    0x00088c79d063, 0x00096c706357, 0x0004d80973d5, 0x000765a1286a,
    0x00069f836da5, 0x00054374fa25, 0x000797f55d7a, 0x000a373494c0,
    0x00091a461cf3, 0x000ab0b0a819, 0x000196321f17, 0x000d5ba64442,
    0x0005a2e73090, 0x000ad7dd79c9, 0x0003e8155f38, 0x000b10b60872,
    0x0005813b18fb, 0x000c5df2df94, 0x00070e12a253, 0x000d8ccfcddc,
    0x000876d63f92, 0x00061451c57c, 0x000c2ae676dc, 0x00007f9c6bc4,
    0x000a75d32abf, 0x000335a703b8, 0x000578782f8a, 0x0000327e961b,
    0x0001f289bf41, 0x000ae305a0fb, 0x000661ae5255, 0x0004a95aedd0,
    0x00080b9e4189, 0x0008cf458a53, 0x0001bc859916, 0x0004661e623c,
    0x000998349fea, 0x0003a88387d6, 0x000cd5c110fe, 0x000cd97ab8ae,
    0x0008eb1e56ee, 0x000d04c02751, 0x0003c46e53b5, 0x0007331a80e4,
    0x00096f5f21b3, 0x000d58d2061b, 0x0000f2711534, 0x0004af86a33b,
    0x0004370264ef, 0x0008877c9aca, 0x0006b6e5b52c, 0x0000264dae32,
    0x0001e46b3409, 0x0006a7573336, 0x00096b0f2b7c, 0x00014d691bc8,
    0x0002bf4b1e87, 0x000346b97f6e, 0x000c4f5cf5ae, 0x000c576779a8,
    0x000e3c30975b, 0x000b6212d0eb, 0x0006c50c2b92, 0x00005ad4c73c,
    0x0004798cf748, 0x00032df91e33, 0x0009516b0590, 0x000226825f73,
    0x000e614aac99, 0x000ac760c464, 0x00027acc8c26, 0x000a9c415c2d,
    0x000c3bd5c5c2, 0x000cb4971eff, 0x0007cecfe134, 0x000e0f76ccfc,
    0x000abbf6c09e, 0x0007490b301a, 0x000f386e6ced, 0x0007e9984026,
    0x0003a0bdbf17, 0x000189d3cd3a, 0x000c1c9a8add, 0x000b861109d7,
    0x000891425fab, 0x000d61c3d178, 0x0008b068da79, 0x00094bf71336,
    0x0006582cc7ad, 0x000a6f1ab27f, 0x0004ec8095fd, 0x0001f505f5f1,
    0x000b3be72a2b, 0x000a1ce1224a, 0x00009c7771c2, 0x000a44467e40,
    0x00084222a609, 0x000fc951dd7d, 0x0006fc532dc1, 0x0005f537069b,
    0x00030070665a, 0x0007c6044a7d, 0x000629118a85, 0x00076716f7de,
    0x000307a3249f, 0x0003698b54ff, 0x00030dc80ba5, 0x000708fc5cc2,
    0x0007821a9e6a, 0x0003df0772a0, 0x00064c0871ea, 0x000d463ae5b5,
    0x0005ee4e51ef, 0x0007aa1163fa, 0x000cf10d0ea6, 0x000a9d7b6f57,
    0x000c0a5f795e, 0x0006cb1e043d, 0x000b20d42bfd, 0x000e24fec258,
    0x0003feaeab22, 0x000957c1ddb0, 0x00074784b222, 0x00074f70233c,
    0x000c89a97228, 0x00020319f367, 0x00055c18a765, 0x000df22ea73f,
    0x000a26bd73ba, 0x000b63f29682, 0x000ce70b35fa, 0x000953d1608c,
    0x0009ab102ba5, 0x000a61ef981c, 0x000b3bfa4361, 0x00027cf2a465,
    0x0001051b76dd, 0x000fe8c016eb, 0x00024d94daee, 0x000571295a81,
    0x0006067dc83b, 0x0003f7250156, 0x000914f04199, 0x000948bd7145,
    0x0005630f9fcb, 0x000d88b3b36f, 0x000c1c6c53cb, 0x000cb601c59f,
    0x000553add735, 0x0008f6c5538e, 0x000a4c2e8f6c, 0x0006d2b27b09,
    0x0001a645cba7, 0x0008d7c5f417, 0x000e4bd54920, 0x0006f62de9ff,
    0x0006aeb69992, 0x00039dacd611, 0x000ded1caed9, 0x000f6aaded67,
    0x0006aee54201, 0x000962d25d0d, 0x00051e9c1796, 0x00050a2fc9f5,
    0x000600c464ec, 0x00062498cc89, 0x0001847179ed, 0x0006e083ec05,
    0x0004a11d3609, 0x0007cb4f5f46, 0x00055b70687b, 0x000f959a75e8,
    0x0009cb2966e1, 0x000ca338f1e1, 0x0009743615f4, 0x000324c7e302,
    0x0002b0fc1d7d, 0x0007030ee10c, 0x00002dea92fe, 0x0000f3a992d4,
    0x000a3186dc49, 0x000c4ca55f3e, 0x000861438ac0, 0x000a164f0773,
    0x0008298d8062, 0x00041f0d926e, 0x000bf6086e56, 0x000d47595c73,
    0x0004d2174759, 0x0003908fef4e, 0x000d357edfa2, 0x000bb422cee4,
    0x000ab792fa2d, 0x0007ed328df3, 0x000334b0f1e2, 0x000f76fa9899,
    0x00039264b39a, 0x0001f879e866, 0x000067cae1e1, 0x0001ba6c3705,
    0x0005843ece25, 0x0000d05f9884, 0x000846d0f3db, 0x00075d31da9a,
    0x0001678e5526, 0x000fd3011f10, 0x000e35e25693, 0x00011b8ced55,
    0x0001f41889c8, 0x00023aabfbdd, 0x000618a2dde6, 0x000a0bdb06a3,
    0x0004bfb86597, 0x000e35ec3dab, 0x000b25ecc9cd, 0x00085b25ca8c,
    0x0004b27eaf20, 0x000449e48f71, 0x0003fc2525ad, 0x00018a9911a7,
    0x000a5ef6128f, 0x0007b5703cb6, 0x0001d0446169, 0x0002ac545191,
    0x000a80ff25cc, 0x0001939895f9, 0x0001b8fdfda1, 0x0008722f5082,
    0x0002fdf1522f, 0x00049de87899, 0x000323c39c24, 0x0002814344c1,
    0x000f855d0358, 0x000b97d9a6c1, 0x0005ecd5d8f7, 0x0006ce42c9a3,
    0x0008f84bf9c3, 0x000fb6bcb68e, 0x0007a8f76a52, 0x00060773192d,
    0x0008c5284bc5, 0x0005907bb11c, 0x00020520ed8c, 0x000e803ea2ff,
    0x0009a756629d, 0x0000b7ceabcf, 0x0004fcd14918, 0x000617e9e920,
    0x00079df52029, 0x000bdfab86b9, 0x0002c9d5338d, 0x000b253fd51f,
    0x00084ff29541, 0x000467827092, 0x000f71fd1cb0, 0x0005d802afb0,
    0x000646d54296, 0x000781872fe9, 0x000c381f57c4, 0x0001b6676cf5,
    0x000022e7c959, 0x0009576a8223, 0x000f412e4cfa, 0x00016ac1736b,
    0x00069400db05, 0x000987e5d5cb, 0x00013562a70f, 0x000463bea4fb,
    0x000e1a52999c, 0x0001fd938423, 0x000d46a858af, 0x000484ab06ce,
    0x0004f5d7b205, 0x00036d6375aa, 0x000fd33221bb, 0x000ce064282d,
    0x000685343657, 0x00026e52d034, 0x00089aeb5df5, 0x0006e4764fef,
    0x0000549c1fcb, 0x000e54254dd3, 0x0007b5e0e47b, 0x00024dbc2de2,
    0x0004f99e10f7, 0x0009c8336d8b, 0x000407ac6a5a, 0x000081fcb922,
    0x00075ef77400, 0x000599cb4bc0, 0x00002aa3023a, 0x00084ae134d7,
    0x00090c7a4ef2, 0x00001f3294f7, 0x0008b5d589dc, 0x0007c2294105,
    0x0009b16f12de, 0x00075a3d0b5a, 0x000194a54f09, 0x0007792df0ee,
    0x0006ae2b54b1, 0x000c9ae6712f, 0x0006dd6e4af0, 0x00082d4201eb,
    0x00024375ae99, 0x0005d508d6ff, 0x000a3c7c86bc, 0x0006ffd924ac,
    0x000f9d27e54f, 0x000b8aa2f91a, 0x000fa7cd2c11, 0x00050b248cce,
    0x000c6d016448, 0x000bc02ad728, 0x000aa3ac9105, 0x000c55fdc306,
    0x000f34641161, 0x0002e43bacbc, 0x000c75a69a8a, 0x000562f4d5b2,
    0x000f72ef4fda, 0x00069bfa0934, 0x000b80e73321, 0x0005c08ae258,
    0x000eefe67118, 0x000b9a4567d5, 0x000019e606cd, 0x0001099c77bb,
    0x000d0be764a5, 0x000711632aa0, 0x00047ee97d06, 0x000ae689c363,
    0x00035aab0c57, 0x000a201dc975, 0x00012e70bff5, 0x0005f4e259d5,
    0x0009fe7dc98b, 0x000f80b9c670, 0x000f2997960c, 0x000f881f0581,
    0x000151b6cddb, 0x0002a922b418, 0x00093c5de240, 0x000b2939cfc0,
    0x000487b6c1d5, 0x0008dfe4aefc, 0x000514a09a3b, 0x000a67c5edaf,
    0x000015e035f4, 0x000e749177f6, 0x000f0843e62b, 0x000d878b5ee5,
    0x0001e384f397, 0x0005136af2cb, 0x000b32d75ebc, 0x000ca9d10754,
    0x000de16c9073, 0x000b245855fe, 0x00031b039b33, 0x0007ece53b3b,
    0x000d43068a1f, 0x000f57c7a09a, 0x0008be8a077e, 0x0007d32f51d8,
    0x000849dfebd7, 0x0003f580454d, 0x000e1ad662f7, 0x00090256c7d8,
    0x000d2663527b, 0x000b2a458d4b, 0x000b84ea8347, 0x000a43c8153b,
    0x000b4a697d50, 0x000ca5e8c6a0, 0x000bec5aebe0,
];

/// AprilTag 36h11 dictionary singleton.
pub static APRILTAG_36H11: std::sync::LazyLock<TagDictionary> = std::sync::LazyLock::new(|| {
    TagDictionary::new("36h11", 6, 11, &APRILTAG_36H11_CODES, &APRILTAG_36H11_POINTS)
});

// ============================================================================
// AprilTag 16h5 (30 codes)
// ============================================================================

#[rustfmt::skip]
/// Sample points for APRILTAG_16H5_POINTS (canonical coordinates).
#[rustfmt::skip]
pub const APRILTAG_16H5_POINTS: [(f64, f64); 16] = [
    (-0.5, -0.5), (-0.166667, -0.5),
    (0.166667, -0.5), (0.5, -0.5),
    (-0.5, -0.166667), (-0.166667, -0.166667),
    (0.166667, -0.166667), (0.5, -0.166667),
    (-0.5, 0.166667), (-0.166667, 0.166667),
    (0.166667, 0.166667), (0.5, 0.166667),
    (-0.5, 0.5), (-0.166667, 0.5),
    (0.166667, 0.5), (0.5, 0.5),
];

/// AprilTag 16h5 code table (30 entries, row-major bit ordering).
#[rustfmt::skip]
pub const APRILTAG_16H5_CODES: [u64; 30] = [
    0x00000000e960, 0x0000000091ce, 0x000000001d29, 0x00000000707c,
    0x000000002d9e, 0x00000000bd7b, 0x00000000e721, 0x00000000b3d1,
    0x00000000d773, 0x0000000034e9, 0x000000000d62, 0x000000000f7c,
    0x000000003086, 0x00000000f898, 0x000000005a0b, 0x00000000f302,
    0x0000000060aa, 0x00000000e68c, 0x000000003b40, 0x0000000098f4,
    0x000000006bd8, 0x00000000f4d4, 0x00000000be13, 0x0000000054e2,
    0x0000000063b7, 0x00000000a5fc, 0x000000007be3, 0x000000007618,
    0x00000000b825, 0x00000000bbaa,
];

/// AprilTag 16h5 dictionary singleton.
pub static APRILTAG_16H5: std::sync::LazyLock<TagDictionary> = std::sync::LazyLock::new(|| {
    TagDictionary::new("16h5", 4, 5, &APRILTAG_16H5_CODES, &APRILTAG_16H5_POINTS)
});

// ============================================================================
// ArUco 4x4_50 (50 codes)
// ============================================================================

#[rustfmt::skip]
/// Sample points for ARUCO_4X4_POINTS (canonical coordinates).
#[rustfmt::skip]
pub const ARUCO_4X4_POINTS: [(f64, f64); 16] = [
    (-0.5, -0.5), (-0.166667, -0.5),
    (0.166667, -0.5), (0.5, -0.5),
    (-0.5, -0.166667), (-0.166667, -0.166667),
    (0.166667, -0.166667), (0.5, -0.166667),
    (-0.5, 0.166667), (-0.166667, 0.166667),
    (0.166667, 0.166667), (0.5, 0.166667),
    (-0.5, 0.5), (-0.166667, 0.5),
    (0.166667, 0.5), (0.5, 0.5),
];

/// ArUco 4x4_50 code table (50 entries, row-major bit ordering).
#[rustfmt::skip]
pub const ARUCO_4X4_50_CODES: [u64; 50] = [
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
    0x00000000740a, 0x00000000c80a,
];

/// ArUco 4x4_50 dictionary singleton.
pub static ARUCO_4X4_50: std::sync::LazyLock<TagDictionary> = std::sync::LazyLock::new(|| {
    TagDictionary::new("4X4_50", 4, 1, &ARUCO_4X4_50_CODES, &ARUCO_4X4_POINTS)
});

// ============================================================================
// ArUco 4x4_100 (100 codes)
// ============================================================================

/// ArUco 4x4_100 code table (100 entries, row-major bit ordering).
#[rustfmt::skip]
pub const ARUCO_4X4_100_CODES: [u64; 100] = [
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

// ============================================================================
// AprilTag 41h12
// ============================================================================

#[rustfmt::skip]
/// Sample points for APRILTAG_41H12_POINTS (canonical coordinates).
#[rustfmt::skip]
pub const APRILTAG_41H12_POINTS: [(f64, f64); 41] = [
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

/// AprilTag 41h12 code table.
#[rustfmt::skip]
pub const APRILTAG_41H12_CODES: [u64; 2115] = [
    0x01bd8a64ad10, 0x01bdc4f3b2d5, 0x01bdff82b89a, 0x01be3a11be5f,
    0x01be74a0c424, 0x01beaf2fc9e9, 0x01bee9becfae, 0x01bf244dd573,
    0x01bf5edcdb38, 0x01bf996be0fd, 0x01bfd3fae6c2, 0x01c00e89ec87,
    0x01c04918f24c, 0x01c0be36fdd6, 0x01c0f8c6039b, 0x01c133550960,
    0x01c16de40f25, 0x01c21d912074, 0x01c258202639, 0x01c292af2bfe,
    0x01c307cd3788, 0x01c37ceb4312, 0x01c3b77a48d7, 0x01c3f2094e9c,
    0x01c42c985461, 0x01c4a1b65feb, 0x01c4dc4565b0, 0x01c516d46b75,
    0x01c55163713a, 0x01c63b9f884e, 0x01c6762e8e13, 0x01c6eb4c999d,
    0x01c725db9f62, 0x01c7606aa527, 0x01c7d588b0b1, 0x01c81017b676,
    0x01c84aa6bc3b, 0x01c88535c200, 0x01c8bfc4c7c5, 0x01c96f71d914,
    0x01c9e48fe49e, 0x01ca1f1eea63, 0x01ca59adf028, 0x01cb095b0177,
    0x01cb43ea073c, 0x01cbb90812c6, 0x01cc68b52415, 0x01cca34429da,
    0x01ccddd32f9f, 0x01cd18623564, 0x01cd8d8040ee, 0x01cdc80f46b3,
    0x01ce029e4c78, 0x01ce77bc5802, 0x01ceb24b5dc7, 0x01ceecda638c,
    0x01cf27696951, 0x01cf9c8774db, 0x01d086c38bef, 0x01d136709d3e,
    0x01d1ab8ea8c8, 0x01d25b3bba17, 0x01d295cabfdc, 0x01d30ae8cb66,
    0x01d3f524e27a, 0x01d42fb3e83f, 0x01d46a42ee04, 0x01d4a4d1f3c9,
    0x01d519efff53, 0x01d5547f0518, 0x01d5c99d10a2, 0x01d6042c1667,
    0x01d6794a21f1, 0x01d6ee682d7b, 0x01d763863905, 0x01d7d8a4448f,
    0x01d813334a54, 0x01d84dc25019, 0x01d8885155de, 0x01d8c2e05ba3,
    0x01d8fd6f6168, 0x01d9e7ab787c, 0x01da5cc98406, 0x01da975889cb,
    0x01db0c769555, 0x01db47059b1a, 0x01dbbc23a6a4, 0x01dbf6b2ac69,
    0x01dca65fbdb8, 0x01dce0eec37d, 0x01ddcb2ada91, 0x01df651402f4,
    0x01df9fa308b9, 0x01e014c11443, 0x01e223c84830, 0x01e25e574df5,
    0x01e383226ace, 0x01e3bdb17093, 0x01e4a7ed87a7, 0x01e4e27c8d6c,
    0x01e51d0b9331, 0x01e592299ebb, 0x01e60747aa45, 0x01e67c65b5cf,
    0x01e6b6f4bb94, 0x01e766a1cce3, 0x01e9008af546, 0x01e9eac70c5a,
    0x01ea2556121f, 0x01ead503236e, 0x01eb84b034bd, 0x01edce466e6f,
    0x01eeb8828583, 0x01eef3118b48, 0x01ef2da0910d, 0x01f0526bade6,
    0x01f10218bf35, 0x01f1b1c5d084, 0x01f226e3dc0e, 0x01f34baef8e7,
    0x01f3c0cd0471, 0x01f435eb0ffb, 0x01f55ab62cd4, 0x01f5cfd4385e,
    0x01f67f8149ad, 0x01f7dedb6c4b, 0x01f88e887d9a, 0x01f8c917835f,
    0x01f903a68924, 0x01f9b3539a73, 0x01fa9d8fb187, 0x01fd21b4f0fe,
    0x01fd5c43f6c3, 0x01fe0bf10812, 0x0000558741c4, 0x0000caa54d4e,
    0x00013fc358d8, 0x0002d9ac813b, 0x00034eca8cc5, 0x00038959928a,
    0x00043906a3d9, 0x0004ae24af63, 0x0004e8b3b528, 0x00052342baed,
    0x000857150bb3, 0x0009b66f2e51, 0x000adb3a4b2a, 0x000b8ae75c79,
    0x000e499ba1b5, 0x000f6e66be8e, 0x0010cdc0e12c, 0x001142deecb6,
    0x001351e620a3, 0x00143c2237b7, 0x0017356582b8, 0x0017aa838e42,
    0x00181fa199cc, 0x0018cf4eab1b, 0x0019446cb6a5, 0x001aa3c6d943,
    0x001b8e02f057, 0x001bc891f61c, 0x001e4cb73593, 0x0022303697a8,
    0x00231a72aebc, 0x0023ca1fc00b, 0x002688d40547, 0x0026c3630b0c,
    0x002738811696, 0x0027e82e27e5, 0x00290cf944be, 0x002b568f7e70,
    0x002b911e8435, 0x002c063c8fbf, 0x002cb5e9a10e, 0x002d2b07ac98,
    0x002e1543c3ac, 0x002e4fd2c971, 0x002f3a0ee085, 0x0031f8c325c1,
    0x00326de1314b, 0x0035a1b38211, 0x003651609360, 0x00368bef9925,
    0x00373b9caa74, 0x0037b0bab5fe, 0x003b1f1c0c89, 0x003b59ab124e,
    0x003bcec91dd8, 0x003c0958239d, 0x003d68b2463b, 0x003e185f578a,
    0x003e52ee5d4f, 0x0042366dbf64, 0x004320a9d678, 0x00435b38dc3d,
    0x004395c7e202, 0x00444574f351, 0x00452fb10a65, 0x0045a4cf15ef,
    0x004aad1994dd, 0x004e5609f12d, 0x00506511251a, 0x0052e9366491,
    0x00535e54701b, 0x0053d3727ba5, 0x0054831f8cf4, 0x00565797bb1c,
    0x005bd5004594, 0x005c0f8f4b59, 0x005de4077981, 0x005f08d2965a,
    0x00602d9db333, 0x0064c0ca2697, 0x006535e83221, 0x0065e5954370,
    0x00665ab34efa, 0x0066cfd15a84, 0x0067ba0d7198, 0x00682f2b7d22,
    0x006953f699fb, 0x0069c914a585, 0x006a03a3ab4a, 0x006b286ec823,
    0x006c87c8eac1, 0x006d720501d5, 0x006f810c35c2, 0x006fbb9b3b87,
    0x0071caa26f74, 0x0073648b97d7, 0x00741438a926, 0x00757392cbc4,
    0x007dea3ea13d, 0x007ed47ab851, 0x00827d6b14a1, 0x0082f289202b,
    0x00832d1825f0, 0x008367a72bb5, 0x0083dcc5373f, 0x00886ff1aaa3,
    0x008a09dad306, 0x008a4469d8cb, 0x008cc88f1842, 0x008d3dad23cc,
    0x008ded5a351b, 0x008f87435d7e, 0x0090717f7492, 0x00915bbb8ba6,
    0x0092f5a4b409, 0x00974e4221a8, 0x009997d85b5a, 0x009a47856ca9,
    0x009e2b04cebe, 0x00a0e9b913fa, 0x00a36dde5371, 0x00a8b0b7d824,
    0x00a8eb46dde9, 0x00aa1011fac2, 0x00ac59a83474, 0x00afc8098aff,
    0x00b24c2eca76, 0x00b2fbdbdbc5, 0x00b3e617f2d9, 0x00b545721577,
    0x00ba132d8ea0, 0x00c005b424a2, 0x00c3e93386b7, 0x00c5488da955,
    0x00c5f83abaa4, 0x00c9669c112f, 0x00cdbf397ece, 0x00ce34578a58,
    0x00cea97595e2, 0x00d426de205a, 0x00dbeddce484, 0x00ddc25512ac,
    0x00e75dcc04fe, 0x00ebb669729d, 0x00ef99e8d4b2, 0x00f0f942f750,
    0x00f342d93102, 0x00f6ebc98d52, 0x00fa1f9bde18, 0x00fb09d7f52c,
    0x00fbf4140c40, 0x00feed575741, 0x010345f4c4e0, 0x01051a6cf308,
    0x01084e3f43ce, 0x0109387b5ae2, 0x0109e8286c31, 0x010a5d4677bb,
    0x010b0cf3890a, 0x010c31bea5e3, 0x010d5689c2bc, 0x010e0636d40b,
    0x01129963476f, 0x0113839f5e83, 0x011a25d305d4, 0x011b4a9e22ad,
    0x011bbfbc2e37, 0x011dcec36224, 0x011ef38e7efd, 0x011f68ac8a87,
    0x011fddca9611, 0x012018599bd6, 0x012052e8a19b, 0x012a9e0ca53c,
    0x013056043579, 0x013523bfaea2, 0x0140ce3dd4e1, 0x0142dd4508ce,
    0x014a2f25c16e, 0x014aded2d2bd, 0x014e12a52383, 0x0152a5d196e7,
    0x0153557ea836, 0x0154052bb985, 0x0154b4d8cad4, 0x0154ef67d099,
    0x0157ae1c15d5, 0x0157e8ab1b9a, 0x015b1c7d6c60, 0x0162e37c308a,
    0x0169c03edda0, 0x016ec8895c8e, 0x016f3da76818, 0x0171121f9640,
    0x0182af245281, 0x018aeb412235, 0x018b9aee3384, 0x01920292d510,
    0x0195e6123725, 0x019ab3cdb04e, 0x019bd898cd27, 0x019d37f2efc5,
    0x01a2b55b7a3d, 0x01a6d369e217, 0x01adeabb94f2, 0x01b3dd422af4,
    0x01ba7f75d245, 0x01bb2f22e394, 0x01c15c387f5b, 0x01c6d9a109d3,
    0x01cc91989a10, 0x01d83c16c04f, 0x01e127e0a152, 0x01e42123ec53,
    0x01e6dfd8318f, 0x01e879c159f2, 0x01f75811d0f7, 0x01f87cdcedd0,
    0x01f8f1faf95a, 0x000168a6ced3, 0x000a19e1aa11, 0x000b3eacc6ea,
    0x000e37f011eb, 0x001081864b9d, 0x00188314158c, 0x001f5fd6c2a2,
    0x00216eddf68f, 0x0038fe6948d2, 0x003fa09cf023, 0x0041ea3329d5,
    0x004caa753900, 0x004dcf4055d9, 0x0052d78ad4c7, 0x0056807b3117,
    0x0058ca116ac9, 0x005ebc9800cb, 0x006265885d1b, 0x0070942bc2d1,
    0x00738d6f0dd2, 0x007adf4fc672, 0x0085da20db62, 0x008a6d4d4ec6,
    0x008ae26b5a50, 0x008b92186b9f, 0x008ec5eabc65, 0x0096525a7aca,
    0x00a0d80d8430, 0x00a5a5c8fd59, 0x00a9142a53e4, 0x00b48419745e,
    0x00b86798d673, 0x00c6963c3c29, 0x00ca3f2c9879, 0x00f0e7976786,
    0x00f41b69b84c, 0x00fe668dbbed, 0x00ff163acd3c, 0x01015fd106ee,
    0x010fc9037269, 0x011e6cc4e3a9, 0x0136716e4176, 0x013f97c7283e,
    0x0142567b6d7a, 0x0148f8af14cb, 0x014a92983d2e, 0x014ca19f711b,
    0x014d16bd7ca5, 0x0157d6ff8bd0, 0x015eee513eab, 0x016ca7d698d7,
    0x0174a96462c6, 0x017593a079da, 0x0177a2a7adc7, 0x01808e718eca,
    0x0181038f9a54, 0x0190918d22a8, 0x019e1083770f, 0x01ac3f26dcc5,
    0x01b47b43ac79, 0x01b8d3e11a18, 0x01bbcd246519, 0x01efb9f682c8,
    0x0005af98aca8, 0x0026a00beb78, 0x005808b8c9b0, 0x00654d201852,
    0x007c2cfe5946, 0x008d1a560438, 0x00a4e4705c40, 0x00a81842ad06,
    0x00bce919ba0d, 0x00c5253689c1, 0x00d4edc317da, 0x00eaa8d63bf5,
    0x010fb757e29f, 0x0119c7ece07b, 0x0137bf1cd44a, 0x0141cfb1d226,
    0x015c58806f6a, 0x0165444a506d, 0x016babeef1f9, 0x016d0b491497,
    0x01875988ac16, 0x0190f4ff9e68, 0x0198f68d6857, 0x01a60065b134,
    0x01ad17b7640f, 0x01af26be97fc, 0x01b22001e2fd, 0x01b469981caf,
    0x01b84d177ec4, 0x01b9ac71a162, 0x01ba5c1eb2b1, 0x01ce42b9a8a4,
    0x01d13bfcf3a5, 0x01dad773e5f7, 0x01dcabec141f, 0x01e01a4d6aaa,
    0x01e1048981be, 0x01e388aec135, 0x01e97b355737, 0x01fe4c0c643e,
    0x0015db97b681, 0x0042eba7271a, 0x00444b0149b8, 0x004bd771081d,
    0x007fc44325cc, 0x00992846a637, 0x009b71dcdfe9, 0x00b71f769a06,
    0x00c254d6b4bb, 0x00e5c96f3302, 0x00f8c5ce11e1, 0x00fe08a79694,
    0x01090378ab84, 0x01093e07b149, 0x010bc22cf0c0, 0x010ef5ff4186,
    0x011b502a7914, 0x0126858a93c9, 0x012b53460cf2, 0x0142a8425970,
    0x015f7aa73066, 0x01a0e66fa27c, 0x01d5bd7dd73f, 0x01ddf99aa6f3,
    0x0010c1a1a7c9, 0x001b81e3b6f4, 0x0021e9885880, 0x00259278b4d0,
    0x003ad86dcd61, 0x003ef67c353b, 0x0099169b166d, 0x00a2eca10e84,
    0x00b4fec3d64f, 0x00b70dcb0a3c, 0x00b832962715, 0x00cbdea21743,
    0x00ced7e56244, 0x00de65e2ea98, 0x00f002e7a6d9, 0x012589a2eceb,
    0x0149e8778246, 0x01672ffa64c6, 0x01be56d5faf7, 0x01ec8bb08869,
    0x01f5777a696c, 0x0017c747cada, 0x00454c7546fd, 0x007d91e4d24b,
    0x00bb1a2de24c, 0x01070ba95dc8, 0x01274c6f8b49, 0x012920e7b971,
    0x014a4be9fe06, 0x015003e18e43, 0x017f22f832c9, 0x01f3cbe5b13f,
    0x01fd2ccd9dcc, 0x001cbde6b9fe, 0x001ff1b90ac4, 0x002cfb9153a1,
    0x00302f63a467, 0x004e61229dfb, 0x00a00a95a9b4, 0x00b21cb8717f,
    0x00dd584fb3f0, 0x012615f8dea6, 0x012f76e0cb33, 0x013fb48b64d6,
    0x016e5e83fdd2, 0x0170a81a3784, 0x019506eeccdf, 0x01aa4ce3e570,
    0x0058257d0648, 0x00640a8a324c, 0x007e9358cf90, 0x008b28130ce3,
    0x00c4ccdcbacf, 0x00c71672f481, 0x00d45ada4323, 0x01181038eeeb,
    0x014dd1833ac2, 0x0163c72564a2, 0x017ec5120d70, 0x002dfd0550e6,
    0x00484b44e865, 0x0079b3f1c69d, 0x00c6551a5368, 0x00ee976e4ad8,
    0x011f506e17c1, 0x015636838071, 0x017eedf5836b, 0x01a55bd14cb3,
    0x01f8d9bc8694, 0x0033a3515159, 0x00580225e6b4, 0x00637215072e,
    0x00c0c6063926, 0x01565f66f66c, 0x015db147af0c, 0x018cd05e5392,
    0x01b2c91c1150, 0x008cc7888bad, 0x0136821344ab, 0x01631d04a9ba,
    0x01b91f152312, 0x01cd403f1eca, 0x01d1d36b922e, 0x01f129f5a89b,
    0x002bb8fb6d9b, 0x006bc569bd13, 0x0113ab7c47e9, 0x01166a308d25,
    0x0131a2ac3bb8, 0x0170c4de741c, 0x01cf7829c8b2, 0x01e39953c46a,
    0x000db01fea02, 0x0082ce2b7402, 0x00aa60d25a23, 0x00b645df8627,
    0x00b6806e8bec, 0x00cdd56ad86a, 0x012954e3dc3a, 0x013fbfa411a4,
    0x0182c555ac1d, 0x019e72ef663a, 0x01a4a0050201, 0x01af25b80b67,
    0x01b552cda72e, 0x01bb0ac5376b, 0x01d43439b211, 0x000e88b0714c,
    0x0027ecb3f1b7, 0x005cc3c2267a, 0x00bbec2b869a, 0x00fa993fb374,
    0x0111ee3bfff2, 0x016c0e5ae124, 0x018e23993ccd, 0x0017d7ecce0f,
    0x0072a7b8c090, 0x007cf2dcc431, 0x00a9533f237b, 0x00e039548c2b,
    0x0101d974dc4a, 0x0143f4ea5faf, 0x0147634bb63a, 0x015f2d660e42,
    0x016386037be1, 0x0180cd865e61, 0x01b2e5e04de8, 0x01b4ba587c10,
    0x002127292ad2, 0x0071abd119b2, 0x007220ef253c, 0x008d93f9d994,
    0x00c59eda5f1d, 0x00d7b0fd26e8, 0x017dfd26895b, 0x01c177f62f5e,
    0x01c854b8dc74, 0x01f6899369e6, 0x01035006e519, 0x012b925adc89,
    0x015b9bad9823, 0x01a927123c02, 0x00155953e4ff, 0x006ae64652cd,
    0x00d4cef1c218, 0x00e3ad42391d, 0x015c39af19a8, 0x0192aaa676ce,
    0x01bb621879c8, 0x00a52911823e, 0x00ac05d42f54, 0x00dda9101351,
    0x01842fc87b89, 0x001f4691c347, 0x00bce1804a7c, 0x010c06ce16be,
    0x014b638f54e7, 0x01f6b8033648, 0x000f31ca9f9f, 0x002c042f7695,
    0x00f4659734b1, 0x00ff25d943dc, 0x01ade8ae7bc8, 0x01c836ee1347,
    0x002a1e0bb8a3, 0x003eb453bfe5, 0x00fdb4d39174, 0x01843554d1f0,
    0x01c68b595b1a, 0x0051d99614bf, 0x018dbf203478, 0x01f76d3c9dfe,
    0x007f1288fb53, 0x00aa139137ff, 0x00b5be0f5e3e, 0x015c7f56cc3b,
    0x01a11ef18f17, 0x01cfc8ea2813, 0x00eba83d2010, 0x0181f14aeea5,
    0x0000aacd6af7, 0x004bec9bd524, 0x0054d865b627, 0x01758574274d,
    0x006e654cac8d, 0x01156123204f, 0x0122a58a6ef1, 0x01d07e238fc9,
    0x01f3b82d084b, 0x007f0669c1f0, 0x009a3ee57083, 0x011cdbe74eea,
    0x01717e9da5a4, 0x0174b26ff66a, 0x00158130ce65, 0x00acef09b9d3,
    0x0178f961d43f, 0x01a6097144d8, 0x01d145088749, 0x001b620bd49d,
    0x016d77c723fb, 0x00f046824042, 0x013d5cc8d897, 0x006d5d45cc4c,
    0x01407eef9993, 0x016552e23a78, 0x01d18523e375, 0x000d38f4c54e,
    0x01996897ce22, 0x019bdb117dcf, 0x01d19c5bc9a6, 0x01d6df354e59,
    0x007898323d68, 0x010545c919ab, 0x015938d25f16, 0x0194b2143b2a,
    0x00387a185e26, 0x00d4408eb733, 0x011ed2b01011, 0x01436c13ab31,
    0x016840064c16, 0x01d81b385163, 0x0097cb653441, 0x01c791532231,
    0x01e8f6e46c8b, 0x007654285a1d, 0x00fe6e92c2fc, 0x019dddf97859,
    0x008d225b0b47, 0x010623e5f75c, 0x01bdd285104b, 0x004ffd847706,
    0x01bc617f5de3, 0x004e1760b914, 0x0157aa01e381, 0x00a9fa4c38a4,
    0x010617c6bdf9, 0x010826cdf1e6, 0x01386aafb345, 0x00f3329fc54b,
    0x01caace70031, 0x00032ce59144, 0x000f11f2bd48, 0x009557e4f7ff,
    0x0106cd0025af, 0x007f8b26441a, 0x00a499a7eac4, 0x0151b0e86a83,
    0x00354acbd732, 0x00a7aa231bf6, 0x003748277b55, 0x005ccbc72d89,
    0x0139c376f2e7, 0x0008b566c88a, 0x00ba36f045b2, 0x017be478ccb3,
    0x01f7930c6e3a, 0x005b4ea241be, 0x002abe85ead0, 0x010aaf78fb2f,
    0x0079227b15f9, 0x015e1bb8a546, 0x0194c73f0831, 0x01e1686794fc,
    0x01e252a3ac10, 0x00a86a75307a, 0x01701c2fdd47, 0x01728ea98cf4,
    0x01aca891466a, 0x0107e4a57c90, 0x01409f331368, 0x019666b486fb,
    0x0042304673e6, 0x01873c297a67, 0x007645fc0790, 0x0148084bb239,
    0x0160bca22155, 0x002d01894186, 0x009c2d0e3584, 0x0152b6e2319a,
    0x01f818cf7cf9, 0x000891091c61, 0x002095b27a2e, 0x004912957763,
    0x00a027c57dca, 0x01d68fe7130b, 0x0047f356b0f1, 0x0176f7ebfdc8,
    0x01a8d5b6e78a, 0x00fe0d98f7e4, 0x006c0b7d0724, 0x01b6f83b13dd,
    0x019b38f5c9f6, 0x01d886afd432, 0x01f93c940d3d, 0x001fe4fedc4a,
    0x006ecfbda2c7, 0x019f4558a206, 0x0004d566a3b2, 0x01071627157f,
    0x01e81a39b2ed, 0x010f0b95a60b, 0x013db58e3f07, 0x01c3fb8079be,
    0x01294829adc0, 0x00fe3575e14a, 0x018a6deeb203, 0x0095fdeb8093,
    0x019d75663db0, 0x00053d5ee944, 0x018f236bb866, 0x01fb1b1e5b9e,
    0x008f7e086c41, 0x00a782b1ca0e, 0x009d9b00422d, 0x00bfeacda39b,
    0x0031cc3114f0, 0x00f9b87ac782, 0x01c80c691ba0, 0x00067eee42b5,
    0x00d2a07e4352, 0x0134c22aee73, 0x013bd97ca14e, 0x006fbd78f718,
    0x00e6004f9df1, 0x018c755c765f, 0x010f441792a6, 0x016359716b10,
    0x01f2dcf472c0, 0x002c0ca01522, 0x0199f8d89498, 0x016843f120d1,
    0x0075ba11ad53, 0x01df24c9492f, 0x0019ee5e13f4, 0x00566913ed4d,
    0x0064c09ac8fe, 0x01be511a46f7, 0x00be48447f12, 0x01061bb192b4,
    0x00b0b7a29ae1, 0x01473b3f6f3b, 0x0016044bcee3, 0x01c065add14b,
    0x01d068c96529, 0x008ad61cc354, 0x0045b544bb8b, 0x011c9716cb53,
    0x004a136e7f91, 0x0104a9a553b7, 0x0049f0175ffd, 0x00114cc1af56,
    0x0103b34a0340, 0x015ed4dce1b7, 0x014f86fab58f, 0x01e6f4d3a0fd,
    0x00109681bb0b, 0x014a95e81cd2, 0x01320a7523b1, 0x00306d4289d0,
    0x00d429277369, 0x0015ba89e9fc, 0x01900679f767, 0x0185113538de,
    0x01b6b4711cdb, 0x01f6fb6e7218, 0x01221c845cda, 0x01dde29efaa7,
    0x003633d20418, 0x01dcf1d00097, 0x00eefb1d007d, 0x0077bc5eb2c6,
    0x00c1529864c6, 0x0026f10884be, 0x00b18f982d14, 0x005d7937c815,
    0x002b89c14e89, 0x000d80e5caf0, 0x0163101b1da7, 0x018c3cab2c2b,
    0x00eee1d80122, 0x01ea45d5c5d9, 0x00cfdd14d6ab, 0x0090a9370e7d,
    0x00c8dcfb0a01, 0x01024735b228, 0x01c2cff31c50, 0x01156058cd9f,
    0x00dc5990a138, 0x01e21fea4fc1, 0x008b5937c3d2, 0x0145b4df9233,
    0x014fcb00e676, 0x009d4170890d, 0x006a2d2ef2a8, 0x003e59228519,
    0x004a1ad89189, 0x0062abd7e111, 0x019c3bac8db5, 0x018abb6c0e0c,
    0x01d959f03efa, 0x00a3cfeb876a, 0x01e8b8776e57, 0x01787f5f43b1,
    0x01ec5ad4e7ab, 0x00c2b05105b8, 0x0096a1b59264, 0x012999264706,
    0x0081ca4ba261, 0x009134f20b21, 0x0179d71ff6be, 0x01243881f926,
    0x00c8f9245182, 0x00cba62d06f4, 0x00176b7b9af7, 0x01108b6f7c63,
    0x01880a492646, 0x01e3fee035a0, 0x01668160bc58, 0x002733019c7b,
    0x00b9de37bb8e, 0x012ed8ec25fa, 0x004f62a3778c, 0x017360487610,
    0x014e33fc0639, 0x00a0a10a97f4, 0x01763aba6b4f, 0x01dc7a757e4a,
    0x017abc3b4ee9, 0x01eaaea53a67, 0x0145007d5979, 0x01dce3745071,
    0x01f1b44b5d78, 0x003b73688573, 0x005cd8f9cfcd, 0x01785a66a271,
    0x008f0e17fbec, 0x00f9488a572d, 0x00e860bf02a2, 0x01fdd86d5913,
    0x006a3eab24d9, 0x01f45f46f9c0, 0x001f1414a0dd, 0x00922e31a3be,
    0x0106c46d05d5, 0x0079a744746f, 0x0005ffcaf33e, 0x00b44162e63d,
    0x00eaf2759f8f, 0x006a28e562bb, 0x01746a2d11e2, 0x00edae163f0c,
    0x0031890eef51, 0x00837829ad9d, 0x0082698fea60, 0x00dac5dba09f,
    0x01b83dc21e55, 0x00fdad17a096, 0x01042bf42853, 0x00379ad27293,
    0x008ade2ea6af, 0x002492545a51, 0x0128bcb7c74d, 0x016f9336a77c,
    0x011c2c8353cc, 0x0110643a6460, 0x01dd0b8d73bc, 0x008650fa2130,
    0x018ba7c21a96, 0x0028c1735cde, 0x0126fb5d4d02, 0x018a939c00f2,
    0x003717f3abfa, 0x004f797ca28b, 0x0054875377e0, 0x01dda46e3658,
    0x01f4fef6d93d, 0x00ef43b5d782, 0x0154eafbbf5f, 0x0185512e13bd,
    0x0085bd765762, 0x01638db6a40a, 0x007070ee5bd5, 0x018ba62098ea,
    0x01a300a93bcf, 0x018db9ad96a9, 0x00095c21fecd, 0x003df20e4acf,
    0x007b05394f46, 0x0072b0de0ccc, 0x00f8c759ee8c, 0x00a12fe616a3,
    0x01ad5de466b8, 0x001bd329666b, 0x00cc98e698e1, 0x01593a5e3bc1,
    0x01b09bc8d7b7, 0x00d7d95f6064, 0x0146d56dfb6b, 0x0151ace7f0c7,
    0x005235f47104, 0x0160552f2bd9, 0x00b85d711139, 0x01b9909e4c5e,
    0x01e9fd6383b8, 0x0013aa2a4a94, 0x01cd28f19398, 0x01328cd2adcb,
    0x0036f95e901d, 0x01ff1761808f, 0x01bf5cba1d0d, 0x0152fb021b19,
    0x00f9ecfc3a61, 0x00ac948d2cb6, 0x000b7e1788fe, 0x0121f739dcb4,
    0x01c85a9b2558, 0x005e91fd6423, 0x01986779c35a, 0x00146247fa70,
    0x019160cd13b4, 0x00e990ebe27a, 0x0121ff3ee3c3, 0x0103173ff5e4,
    0x00999d1faf27, 0x01bed034cdb9, 0x0129c2237599, 0x00b73e6e84ac,
    0x014fedd6c98b, 0x0032fddcf527, 0x0112f9e8b254, 0x01c2b79f0489,
    0x00c92da5c461, 0x01eb749d5dea, 0x0146d88e7d76, 0x012775e52da6,
    0x019c87d17e43, 0x007c8ffc74d3, 0x0015f7e792e6, 0x0028d0ef5231,
    0x00748148e4ec, 0x009fd9a463f5, 0x0159974ab0d1, 0x007b2e95390b,
    0x01ff14f5ac33, 0x001f655a5054, 0x00acf593d41a, 0x00898c1402a1,
    0x0093b3e0e6ae, 0x018d45df2de5, 0x011ee6cb87ce, 0x010936d11081,
    0x01cae4599782, 0x0188e5a850b5, 0x00cd074f4022, 0x01dfca5b7190,
    0x00b68e62b82b, 0x016e8dc2307b, 0x017c57ec8ddc, 0x002826044499,
    0x0106bc22fc2c, 0x00c14f105ed4, 0x015af4d3f42a, 0x0137e93a480a,
    0x01849474f50e, 0x016dba230a81, 0x001739183125, 0x0135fefc57c4,
    0x00b3f2d634ea, 0x0029710ac92c, 0x0148cf641ae9, 0x006708f24fcd,
    0x00f94815f6fa, 0x0121607febda, 0x00805bb5d7ec, 0x01c39320b045,
    0x0062000cdbc8, 0x014ac177b4a5, 0x00063a51304e, 0x012443627fa6,
    0x01acd66a314c, 0x00f3b5bed960, 0x01517dc78a4d, 0x016839481f18,
    0x007aa6079abe, 0x01f6e9c6f5e5, 0x00942c5bae28, 0x00458cd0f0a5,
    0x01432717dd98, 0x00e39ebb3ef5, 0x01f0115ab508, 0x01662d90cacf,
    0x004d1177e1e0, 0x0092137e93e7, 0x00bf5c0ff11b, 0x01a81e81568d,
    0x011aaf91d931, 0x018922c9bfba, 0x01d68ad74405, 0x00a78bce4d95,
    0x010edea8ed9f, 0x002f6bdf7c6e, 0x01460858efb8, 0x00628b39bfa1,
    0x01c3215f60ab, 0x018ea1a46e45, 0x0019bf64425e, 0x00d310f81754,
    0x01710d4c011f, 0x018a69b611f9, 0x00debbe1d511, 0x002fbfefbb73,
    0x0097f0e7392e, 0x00ea0d6b3747, 0x00ee85459226, 0x019f66c0749f,
    0x01db94352bd4, 0x0171dd78c628, 0x00cbba32d9b0, 0x00d5e371e1d0,
    0x0137c3fca430, 0x006ad97a92e9, 0x012f79b381ef, 0x0059b998e941,
    0x00b2e255ed67, 0x0013354c8c86, 0x01aa40ef5488, 0x00b95adbcc0d,
    0x01d4897b2626, 0x01166b9dfc1a, 0x01784c28be7a, 0x001015dab617,
    0x005ecb96cd36, 0x0081600554a2, 0x01f48f1758cb, 0x01102cad72f0,
    0x010f94a3df50, 0x00db38400c7e, 0x01d29e72e330, 0x0059698d378f,
    0x0167078aad77, 0x0048a99ecc6a, 0x014c574abe4b, 0x006fe03d48c3,
    0x01baf5decb77, 0x012fdb9e349b, 0x0103664d422f, 0x007599779f7a,
    0x01ee922cc3aa, 0x016276782f89, 0x0037c979c3bc, 0x00cdba2dc35f,
    0x0052f84ee994, 0x01efe6aa735f, 0x015a1639ed91, 0x009e3a598da6,
    0x0019a86ca9a9, 0x00970dadb02b, 0x00d2d2239539, 0x01b5a9a7d43a,
    0x01a28d711304, 0x0073027dc257, 0x008697226ebb, 0x0091ab09256d,
    0x00be79f6ad45, 0x001f077c5229, 0x007854970278, 0x00d39af32496,
    0x0195054ca9ac, 0x01c1a949a25f, 0x0199873924c4, 0x01847acc8563,
    0x004acd98a710, 0x000f04020319, 0x014de092f710, 0x00305e1be573,
    0x015bd8fdc33a, 0x015d07db004c, 0x00abd7e91181, 0x008f5bd0f053,
    0x00cbe19f767a, 0x010be8ed0709, 0x0134b8d9b6ae, 0x0190e4ec260e,
    0x003bdcbca890, 0x0112372a27ed, 0x001feb58f771, 0x01c4cfbe06df,
    0x00dee5b17d82, 0x01871a774e8f, 0x00fd5def58a5, 0x012ec074eb5f,
    0x006b55ac1c67, 0x00c5ffe47891, 0x0073b6ce69a7, 0x0066b0e0f585,
    0x0100248623a3, 0x0016392f6f04, 0x007d50747cb4, 0x01e19f28f0ae,
    0x014f8084fd3b, 0x018f1b547e66, 0x00d28e258b80, 0x0174de911918,
    0x00cf50acb1ff, 0x00c8ee9466da, 0x00d633674cfa, 0x00323c586885,
    0x01285f5a641a, 0x01f6d00cf4d0, 0x0106da5ccee7, 0x008020beacb9,
    0x01af2bb77ef3, 0x0007a2f69285, 0x00eab509d74b, 0x0170b3474645,
    0x005a4dae3dea, 0x002c433626d2, 0x019d15ffd4ea, 0x01147744815f,
    0x015ddaf434a9, 0x009025076210, 0x00939368b89b, 0x01859a3a07e7,
    0x0159db2909c6, 0x016d8f3a0103, 0x01905ea325b2, 0x019381a115aa,
    0x01f9ea104107, 0x001e79cd536c, 0x00a9a3ac60e8, 0x0070d66cadb1,
    0x00bc17a022c7, 0x0116da464f50, 0x00d83ca13b7d, 0x009dcfec097c,
    0x01a3240bf4a1, 0x011426b7932e, 0x01acdf2b6bb1, 0x00c10683210d,
    0x007d7a7382be, 0x005a6740670d, 0x015b4c25f379, 0x00f3cb471e8b,
    0x01714233b4d7, 0x00b328dc549d, 0x0123e124ab06, 0x00c8d2af806c,
    0x019b6c8925ca, 0x0064f327983d, 0x00dc5a2e66d8, 0x01df61d3d025,
    0x0070fac19125, 0x006617ff9162, 0x01da1d9503ab, 0x01ee6af25502,
    0x00329c6d86ce, 0x0157e5f038e1, 0x00cc6dc97cac, 0x0024778a626b,
    0x0138894bb342, 0x007d6f002e55, 0x004b9b4764cc, 0x00a0cbc5d154,
    0x012aff498e59, 0x0001653b5202, 0x0149b66cdb7a, 0x004cffaca152,
    0x002f69b2b280, 0x00aed4bffbb2, 0x01dcf40a92c5, 0x01d5606cfc4d,
    0x01d288445a1d, 0x00a9a40a7a93, 0x016c34dd7a7a, 0x00e30c44e5dc,
    0x00e7596aeb7b, 0x00d8ec1e4dac, 0x019acb07c581, 0x01c59956a5de,
    0x005eb32fe0f7, 0x00d4f39ab374, 0x0041c0946c78, 0x01a8ca1f2d3f,
    0x001aebd3c67f, 0x00ed6dda9095, 0x000d4eb3edb9, 0x007819d2a6ee,
    0x0045b8f998fa, 0x003c0201308a, 0x01e0ffab3f53, 0x01e6cc32a780,
    0x01045e550c09, 0x0083c4a6bfaa, 0x0022b798a1f7, 0x00f42f1fdb24,
    0x01bbfc9837f4, 0x00ce03a8c117, 0x0066b8746cea, 0x00534be8e3d1,
    0x0037c5258685, 0x00ed86dea0ed, 0x015ecb7ce911, 0x01245834d414,
    0x00c6ec052f56, 0x009df26ab706, 0x01e5d6eed381, 0x00675416f65f,
    0x007ee172ae2b, 0x0053bff79a11, 0x01bfa2b53c01, 0x01169abbdc78,
    0x01127893424a, 0x0128bd3ea4de, 0x0147e8cf6371, 0x00444d0b3b3f,
    0x005d2e9b8c8f, 0x014ec21faf96, 0x002a021b8364, 0x0188855bf812,
    0x0168438f3dfc, 0x01b3a7ea750d, 0x01ff1a134379, 0x010c8c199da7,
    0x00ec9a724ddb, 0x0011f6a71c4d, 0x00ad5318169e, 0x01b156466b97,
    0x01846159b5ea, 0x006d0af1b37f, 0x00b7d5659b5f, 0x0163fadd70c5,
    0x01287c7ad5c8, 0x00f6df36dfb0, 0x0188bbbf3a19, 0x014162619351,
    0x0186af9c4e52, 0x000958ca4268, 0x008044bb1794, 0x0071113e2ff4,
    0x01fc66211707, 0x01827c740954, 0x01ff95d06cc4, 0x01b54ff685c1,
    0x006d88de7741, 0x01c927e3eca8, 0x0042c6d5652b, 0x01c28d92cb19,
    0x01b5779b48d9, 0x01344e24a9ce, 0x0031b6b8c707, 0x011c4a8eb4e2,
    0x01995b6d9d79, 0x01193ac8315f, 0x017f611bc3b2, 0x009902c85dc7,
    0x010ca1728539, 0x0009a5bcf7b6, 0x00c875bfe3b9, 0x01c80ee1752e,
    0x000b61b1b07e, 0x01e4f7552473, 0x0141ae812d3a, 0x00992bdf44f2,
    0x0177cb8203f3, 0x000cc54c9a54, 0x012b5f0a47a0, 0x0117884a0232,
    0x00bf7d50d7b6, 0x004c6c3f6879, 0x0037a66b633f, 0x010fd1a275ee,
    0x011e5790554c, 0x015c168a72a3, 0x013c7537eaba, 0x00ca23dd9aea,
    0x012dbb51fc2a, 0x014ae352eed0, 0x0020e166257f, 0x011ce40d3d1e,
    0x010f4929db1b, 0x019fd90a2f45, 0x004ee5bb04b0, 0x0060b378e062,
    0x01c86c08e115, 0x000849e4e2de, 0x01161e70e4cf, 0x0087d52000e4,
    0x01c0acc158ab, 0x0136f7d2d252, 0x0158c0d6986c, 0x01174bd33519,
    0x005a72318656, 0x00d6e9d0f11f, 0x0188b6005e76, 0x00ecbb0ca621,
    0x01131eb3cde3, 0x01de6369b749, 0x0153d9ccb5ac, 0x006022a8be13,
    0x00368ec83b60, 0x01345506440d, 0x00ae9863319e, 0x00d963623257,
    0x0097fe23da88, 0x01cb23403fe1, 0x0190a0c54bc2, 0x00585f248208,
    0x01be4b974e1f, 0x00571a78ba23, 0x0113f244d483, 0x00f8780c4818,
    0x00db51a82200, 0x016c49ec533a, 0x01585317f190, 0x01fc70105ed9,
    0x01ae4053905e, 0x01dda20b1e04, 0x014e06567fda, 0x01fcf73cc8f6,
    0x00ab374cf2e1, 0x00752720033a, 0x004a6432e5dc, 0x01c0f08cdd28,
    0x005eb87043f7, 0x01ec38067d90, 0x01fef4cf5059, 0x01700dac4882,
    0x000cdd792496, 0x001994ef8c66, 0x01669a978fac, 0x0093695c2811,
    0x008c84651653, 0x01305144f59e, 0x01e5597b1863, 0x0154d2615543,
    0x01e5a8c08afc, 0x00841f1b18ec, 0x01476b7e29a0, 0x010731c052f3,
    0x008744107d0d, 0x00c61901934d, 0x00ec347b7b88, 0x00e7476d701e,
    0x0006e867aafb, 0x00e1d03044bb, 0x00658319e530, 0x014336373a0e,
    0x016cb93a933e, 0x01b8b6d95bb4, 0x0004a5413171, 0x0010ff862197,
    0x01115f3202f1, 0x01d6f7ed1430, 0x01b373d7ff0f, 0x0025cfa73de1,
    0x00cc6b38943a, 0x00759f924c6c, 0x014862200a2c, 0x01b74c7a4cb4,
    0x012d44b5851b, 0x00ca97f5aeb9, 0x004d36701f59, 0x01de88f2493c,
    0x00420abaa0f6, 0x00d892babdd5, 0x0062fe9b8b99, 0x006a1c54d76e,
    0x00b87dde2e3b, 0x00c77657afe3, 0x006b72f09030, 0x0184b68bbaea,
    0x0022a6cd479e, 0x00166f0c5ba7, 0x01809f47bad3, 0x018ed0cf68af,
    0x01ae1368afda, 0x00ca2d646634, 0x01320f49e286, 0x00d56c0385bd,
    0x006b6f80ffc8, 0x00b0615b2264, 0x0125287f0185, 0x00110b2bee8f,
    0x01e9004a8e70, 0x015dde15e068, 0x010438693ad9, 0x015baa26fb1e,
    0x0183bfc66070, 0x00804a41b38f, 0x005dbd4a1a67, 0x00b10be90245,
    0x0050b250494f, 0x001dac323cc2, 0x0175a189df75, 0x012ddf98fe04,
    0x010862cc7c48, 0x001dc90b7cd4, 0x011dbf851ad7, 0x01a87ae5dc11,
    0x017f7fa9e215, 0x013038ead8fe, 0x01181bd7be69, 0x0119086a04d8,
    0x01589d569b98, 0x01e9b00424f1, 0x0080ff1896e9, 0x0090d4c22245,
    0x001eaf9672f6, 0x004afcdf31fa, 0x0145d412a4c3, 0x00ab19785bb1,
    0x01d96d082bbe, 0x014286dce6ec, 0x0160fd92f8fb, 0x012fdcb8b2af,
    0x01a01a3425da, 0x0133162849bf, 0x017f312a577b, 0x00a28b92bb72,
    0x018e5fbd6372, 0x01df71ceacee, 0x012c5bdf9515, 0x00e6c1acce21,
    0x009d25699cd2, 0x015270b91c81, 0x0126e0bb0707, 0x00fdfae613f7,
    0x016e8679d842, 0x01d2c9b489ad, 0x01abd33e45cc, 0x0192f2c15d5d,
    0x01713ce8276c, 0x010e56516b04, 0x013739a4e4b7, 0x0076c27d6558,
    0x01eee4c41ee2, 0x0064d3c6c09b, 0x015d2944dbe0, 0x00c61d6ba698,
    0x0180dffc6541, 0x015061fb4932, 0x00df6e885666, 0x0127eec246c8,
    0x019b51fd70c4, 0x01187e4df717, 0x006a8b0a8f1d, 0x011942390170,
    0x00a421965912, 0x01f726d51351, 0x00317c4daa03, 0x01acdc1ece78,
    0x003fd243f3eb, 0x00fdc16c4efb, 0x002f12371b10, 0x0039b6e20f1c,
    0x0169ab733a9e, 0x00fb398f0a72, 0x002761cbddc4, 0x00d6a06bbcce,
    0x0056f4927d58, 0x00c5a4372537, 0x01b90381e8d6, 0x0091b9fae2d8,
    0x016ee910633b, 0x00a7bc7dd016, 0x00e725ab716a, 0x018227cf3c37,
    0x00dba6f91ce3, 0x01fedfb8bf3e, 0x015de02922d1, 0x00d11529aaa3,
    0x01a162d5f7f1, 0x004b0b4d9d3e, 0x006f50397572, 0x01348ebcfc2e,
    0x009df43157d4, 0x01999827f76a, 0x01467f2570f3, 0x01eba5dff8c1,
    0x010ee96b1e22, 0x0107be5eddbe, 0x01411df497c3, 0x00d226a166c0,
    0x00f3d092e815, 0x0058e2ed63ce, 0x013bd7ba8df6, 0x01298e4f35ce,
    0x011be9845c0d, 0x00b53afc8ffa, 0x0067af688e82, 0x010a41cd33d7,
    0x00f6838baeab, 0x000777ac0858, 0x01dfcd3f233b, 0x0034944e7100,
    0x012487a3b270, 0x00ec12b9190d, 0x01bbc01a91c7, 0x00656b8c243b,
    0x004dbb62b088, 0x013ab915c3fa, 0x01f8a3d62167, 0x00009a46896a,
    0x00fb5f4d2fea, 0x01116bc7e342, 0x0183e9fd5a14, 0x01d0c98a1c5e,
    0x015d459676d3, 0x00f760f6f9a3, 0x012c66e40456, 0x0173ddc771b1,
    0x0007db0a1578, 0x01f3ce6f5029, 0x0077d9f5679b, 0x001ac5f85b64,
    0x0004c672cd5d, 0x015680655ee8, 0x0186876a1255, 0x00774469842e,
    0x013271af9a24, 0x006e89c9fb2c, 0x006c1d9e2be1, 0x00f0741acf5a,
    0x002aa8d88fee, 0x00038b16ed82, 0x00e9cba39dd7, 0x00912e6005cb,
    0x01251d6608a9, 0x00e0b6ff59fb, 0x00e08ea95b44, 0x014573ade1a2,
    0x015b7c8a4880, 0x0030578d9c9a, 0x01fd9016e391, 0x00a0f6219300,
    0x00d6556f9ac1, 0x001c1012de94, 0x00b971893af5, 0x011ca283b57b,
    0x010c64cb9e05, 0x01aa714b75b4, 0x014aca141713, 0x0125c7fabffd,
    0x00a981632be8, 0x0113c56a7e7a, 0x00d2ba1957ef, 0x0160d918b563,
    0x00fb8ce18705, 0x007127d3be2c, 0x012a325812ac, 0x00a0ae4e4033,
    0x007a50e36769, 0x000c0f50d2c7, 0x000890ba241c, 0x010b3a091b74,
    0x001dfa797966, 0x01952caa9bad, 0x009d825284d7, 0x004ba9437737,
    0x01f33933c933, 0x006d53c810e2, 0x00e4a1f020fb, 0x0152a41fc39d,
    0x01e6f1b3bd37, 0x0091ea70722f, 0x0028924a2a6d, 0x01ec4fbfa9ab,
    0x00a989d49062, 0x0079387b2a25, 0x0054b2e53b73, 0x014dc3868cf9,
    0x014b9d290525, 0x00b777ff6c35, 0x009774746651, 0x014c55cb6390,
    0x00571a4688bf, 0x00312aa0996a, 0x01532efa1a5c, 0x00dfec4adedf,
    0x009681870c1a, 0x00c30efd0321, 0x001e872eec40, 0x0010aee5189e,
    0x01b54c11e3b0, 0x01054076b18b, 0x012009bab10f, 0x01f5de648026,
    0x01fec2d089f6, 0x01ae79a3d351, 0x016055846033, 0x014a18cf43cd,
    0x002b4868b059, 0x00f39d118add, 0x0184fa8a563b, 0x005dcd75adef,
    0x01f867bca4ca, 0x00722f8230e0, 0x01c7739a0b92, 0x014d27afcac6,
    0x01ff4483a337, 0x01ee3e7ec65e, 0x015bdf9cf572, 0x00ac55b183d8,
    0x01bd7ba5eb7e, 0x013d6f28d36d, 0x0094483a59c7, 0x01971a6811fc,
    0x01d01f6dc8e5, 0x01d7041a8164, 0x00cd4bb66307, 0x003cc8bae5d2,
    0x0040a28f0a0e, 0x0028ff942508, 0x010f7db14df0, 0x00970a05aa25,
    0x004a714b291e, 0x00c701e4e2e9, 0x000c3311c809, 0x0085c5af52eb,
    0x00d2a0ddcc3b, 0x01b86f410788, 0x01ede3484ca6, 0x004074091b8a,
    0x00cdace8c340, 0x00964a817ec7, 0x0053c4b1e08a, 0x0071f3cfaffd,
    0x01c847b0f581, 0x01f95bfc003b, 0x001c86666096, 0x00448e6a5caf,
    0x017d9562fe30, 0x014b11e2c939, 0x0146ba94fe60, 0x013917b4bc7c,
    0x01bfab5951e7, 0x019279b20f95, 0x00216bd36a1b, 0x017888e39bb5,
    0x00ae80792a03, 0x008298ab4acb, 0x010dfed49850, 0x00cb166c8a85,
    0x0101beba4048, 0x00855cff4fc0, 0x01849168c1c5, 0x010a38f62987,
    0x0049cf1f189b, 0x016a2876af49, 0x0087fa6e2a56, 0x0062c7b015c2,
    0x00d5c29bc4a1, 0x0062091756ed, 0x00518e01968d, 0x003d09d83c3a,
    0x0054f111c2e2, 0x01555bb741cd, 0x009dbccd3073, 0x01ec92ece825,
    0x000112998bc8, 0x00e02379d7c8, 0x0149f696fd0f, 0x01b7137d7768,
    0x01a3b030519c, 0x01e1d806bfcd, 0x000bc65b06e8, 0x01983059408a,
    0x01e9fe1feca9, 0x00132c0a95aa, 0x00a6dc5119e7, 0x00fd58b2db10,
    0x0137e20c98e5, 0x00968688531c, 0x00a81dc892a8, 0x00f861cd9997,
    0x01b0d2b12efb, 0x0000bea34e27, 0x00db665c24f9, 0x00e90e722940,
    0x00106d989a38, 0x002e84aac68e, 0x00f3e4a70319, 0x01d9d002369b,
    0x001c31c31105, 0x0091514c5856, 0x01c49b1c053b, 0x01ebef82c4a5,
    0x01ec86920675, 0x01253eb1b8cf, 0x019c293afdd8, 0x01e6a30fc496,
    0x001cf75469e1, 0x01216b1b956c, 0x006dc53cf47c, 0x016b338cb908,
    0x007bc89831bb, 0x012bfceaee52, 0x00602827afa6, 0x0129bef4e086,
    0x019ade8a837b, 0x008f365a39dc, 0x013331a3ed5d, 0x0167bf567e12,
    0x01817eb5403d, 0x0116a0ac445b, 0x0151f5a1ae96, 0x00355fbafed3,
    0x01721a856f18, 0x00398c5fb690, 0x018fde252209, 0x009bc1effd8d,
    0x015e058368cb, 0x01ef19f8672d, 0x00efeb28bcb5, 0x008597fe8243,
    0x01e3558b7241, 0x01f06d2b5a4d, 0x006753166ca7, 0x019d05252f87,
    0x007ece8719c7, 0x00abada7be10, 0x003b7bae0915, 0x00b920dbfa1e,
    0x0077c3d750ef, 0x00c8a77b91e5, 0x013c12511766, 0x01bd655e970d,
    0x00d5f0a10e67, 0x005c151f0386, 0x00f36c123216, 0x00733590bd56,
    0x00134800baf0, 0x0166cf1de51f, 0x0117dd06e881, 0x0129a0921efb,
    0x00a6ff0ca5c4, 0x008ea200b050, 0x0034f9387b94, 0x01d3723ad2de,
    0x018e5af2b62e, 0x010bd979c6df, 0x014d339607a4, 0x01b88544514a,
    0x003a4450d2d8, 0x01a61117a18f, 0x01e21725e539, 0x0123cf7364fd,
    0x0179908b5b01, 0x0069ec91a349, 0x00945679ec02, 0x00e6a4ca1b53,
    0x007e2c612da0, 0x01f568f32763, 0x01b564f98a0f, 0x0020e11f4d89,
    0x0090c3ea1433, 0x014c7e71acd3, 0x00dc4e4b061f, 0x01c813886671,
    0x00998741f58b, 0x00fe30e80a8f, 0x00607de6a503, 0x01484657259d,
    0x00e53c335255, 0x0174d9bd2dc8, 0x007d39cfa317, 0x01c0992c6850,
    0x0112351c0a60, 0x01e0d5e6cd44, 0x00667998400c, 0x0042dfaef259,
    0x019abbd3ed08, 0x014ad9c7ab72, 0x0056389ff426, 0x019427c1236c,
    0x01a4063a9d61, 0x00af87f97b4d, 0x01666612b513, 0x01f23ac539e4,
    0x01e873a1cf84, 0x01701f1aeddc, 0x018b22a98565, 0x01ddc760ab43,
    0x0022da169d55, 0x00f1416893de, 0x00ee14c0e6db, 0x019252aae1f5,
    0x000b4fb6141d, 0x00c2b99da72c, 0x019ec587f23a, 0x01f44dc5cd77,
    0x0101573c5f0c, 0x00c8590ac48f, 0x01f1d4b1d98d, 0x012ae39c0c1e,
    0x016e46996bb3, 0x00d907071ae1, 0x00ee89247a5b, 0x012753bfdbce,
    0x007906c2f180, 0x006632db9a04, 0x006dd60d443f, 0x006b859327f2,
    0x018f08bb0ad5, 0x0100dd9d760a, 0x001e8c644d0d, 0x01e8edb740dc,
    0x016ba7ffa32e, 0x00bfbda5d84c, 0x017fdf137ad8, 0x00777ffe01db,
    0x013b9a93d0a6, 0x0032bde748a4, 0x0104f87c4ba8, 0x01102e5438a0,
    0x01790599f8b8, 0x006e0e97a55c, 0x00980d898e6b, 0x014035825c86,
    0x0199ded7a4bf, 0x0138ab0050f4, 0x00f8c1940dda, 0x0129ad4d2a65,
    0x0076a9aca541, 0x01332c5d323c, 0x01dba5113a26, 0x01c66d9f5293,
    0x017e232d0d9a, 0x009976915862, 0x01913b2a7244, 0x00b2ba534e99,
    0x0160d1bf0cf7, 0x000eb9922d30, 0x002e3f788035, 0x01d158a6dfbe,
    0x014e9413f1fe, 0x0132d8c19753, 0x014b4f73c0c3, 0x001f33b28cfd,
    0x0072a8fad428, 0x01514364455c, 0x002d98ab517f, 0x0167cedc2d26,
    0x01bb3c039f02, 0x0144f9f06518, 0x0086d324dd1c, 0x012c4eb26f57,
    0x0078ea843219, 0x006a55f7e6b6, 0x006d7b34c6e0, 0x01016cfeb2ec,
    0x004215676871, 0x0156b12b4c31, 0x00586fdbe5d9, 0x0030212d8b7b,
    0x01cc7382e8cd, 0x006ca5a2d1ec, 0x00875c7a96f2, 0x01710b90c031,
    0x0010d1d8c47a, 0x01bf01e51bb9, 0x00c1793e8778, 0x01b326c2d7e2,
    0x00c5310a5aab, 0x011cd7b3c8e5, 0x018d5912ebaa, 0x017450d723c4,
    0x00ab2fa34dcd, 0x012c7246a7bc, 0x01b365296a34, 0x00c47f954743,
    0x0050467ba354, 0x002cc0b816e1, 0x017598c4f17d, 0x01af4283647b,
    0x005a5d5379ec, 0x01aaecddc00d, 0x01325e99894f, 0x003935af6226,
    0x018450e1eebe, 0x009adfe3a3d0, 0x0079383c9291, 0x002212102d63,
    0x0153d2be7427, 0x00fb53f1bb55, 0x00918bed2982, 0x010f53f0d799,
    0x00526ff4d1b3, 0x00bc90635acd, 0x01453b55ba1d, 0x01079d69bec4,
    0x01af3934bd83, 0x00b29697fc3f, 0x0108533de8dd, 0x012968f67594,
    0x011cbe9cb3ed, 0x000bac4e20c3, 0x0134251aba54, 0x014eaaa20df5,
    0x01528aec9718, 0x0170983cc1ec, 0x002f097e68c8, 0x005f746b9d5c,
    0x01ca56b310f3, 0x003972530948, 0x009ba80f3bce, 0x016c5f86f30d,
    0x0119664abd83, 0x016fdda25a23, 0x017631d1a1d3, 0x003d72d2b20d,
    0x011f52f5697b, 0x001d6b6d864f, 0x0043d6422f0a, 0x016ef1aa2e0c,
    0x00e1a3790b11, 0x00f1667a531d, 0x011cc7bbbcda, 0x01ace1b03f19,
    0x018bd83b1d52, 0x008a5de68db4, 0x018fbf08cfef, 0x0082019639cc,
    0x00ab6b9a8d3b, 0x01fc4a838fba, 0x0104f0d67db1, 0x01e4bd144895,
    0x010493faac76, 0x006f1ac45d00, 0x001b845ac503, 0x01a071baaaef,
    0x017ce5822109, 0x01a7b9992deb, 0x01641ad2afd1, 0x01983f74b6ee,
    0x011dd4491184, 0x01597809c70c, 0x01c9ee207a02, 0x01fb967dbcce,
    0x0013dd5d0740, 0x01f04a7cf90a, 0x00bda526f2c0, 0x000a290afad5,
    0x01f74ef5e0ed, 0x00ad155bc8de, 0x0008a74e0d64, 0x01e26d6d2095,
    0x01efe9d3f3fe, 0x00b7bb790080, 0x00e83eb7940b, 0x00195a6721de,
    0x01697162ecc6, 0x01a822424ecc, 0x00bb9409ef14, 0x0088dc8af7de,
    0x001a109fbdf8, 0x01263a6224e1, 0x019762451a46, 0x0123f8087a5d,
    0x01ee61da2ff8, 0x01050f8b67b3, 0x005bd05e0ad3, 0x017f3054510b,
    0x01f0a65cbe49, 0x00c38f39ecfb, 0x01b186adf1ca, 0x0003c9d6cb25,
    0x00b0d33343b8, 0x008cb15b5077, 0x0083960895dd, 0x00a4f544f26c,
    0x014637fba5b5, 0x00c8a7c5184d, 0x0109ef503b91, 0x0110cc27f58c,
    0x01dd15e9f124, 0x01eddb2ca0d9, 0x00aafbfcea5b, 0x003f14f93014,
    0x01c8249b99d9, 0x007c3667f5b3, 0x01a62b640f70, 0x01eec6a22d2a,
    0x00855be93f97, 0x00c7af091027, 0x010a7ffe2790, 0x01042e09ea4c,
    0x0076db25a8e1, 0x01654d706d3f, 0x00260b3a5e33, 0x0088f9419e9e,
    0x01d68917df54, 0x01188549820d, 0x0158d48169b9, 0x019e93fcdd73,
    0x016b91a47690, 0x01fba468bd65, 0x00bb64a6d60f, 0x0157432e4097,
    0x0088a4f70f06, 0x00f091d83cf0, 0x01c4a6fbc192, 0x00ada88c3e7d,
    0x00e531e141b5, 0x01237b66171c, 0x012749c7c2f4, 0x00e2cee269f9,
    0x00260340689f, 0x019500504a28, 0x014432099794, 0x0047cbdd04d3,
    0x001af023fdbe, 0x01eee99e354e, 0x0012649e3aff, 0x01a7f7a49cbf,
    0x00e43793e465, 0x006209ca0402, 0x01eadfe92cff, 0x003dacced0bc,
    0x0097a15758c0, 0x01fcd58aeaae, 0x00ff774c923f, 0x007a5654482c,
    0x01b492028846, 0x015c12bf403f, 0x01e7575219b9, 0x00342bb31b5c,
    0x01a8052db35e, 0x00e8274ad0db, 0x000f1a937c8a, 0x0094b5e48ed7,
    0x00352cb6ab16, 0x015cff6a6f12, 0x01ee40155a64,
];

/// AprilTag 41h12 dictionary singleton.
pub static APRILTAG_41H12: std::sync::LazyLock<TagDictionary> = std::sync::LazyLock::new(|| {
    TagDictionary::new("41h12", 9, 12, &APRILTAG_41H12_CODES, &APRILTAG_41H12_POINTS)
});
