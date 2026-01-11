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


use std::collections::HashMap;

/// A tag family dictionary.
pub struct TagDictionary {
    /// Name of the tag family.
    pub name: &'static str,
    /// Grid dimension (e.g., 6 for 36h11).
    pub dimension: usize,
    /// Minimum hamming distance of the family.
    pub hamming_distance: usize,
    /// Raw code table.
    codes: &'static [u64],
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
    ) -> Self {
        let mut code_to_id = HashMap::with_capacity(codes.len());
        for (id, &code) in codes.iter().enumerate() {
            code_to_id.insert(code, id as u16);
        }
        Self { name, dimension, hamming_distance, codes, code_to_id }
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
        let mask = (1u64 << (self.dimension * self.dimension)) - 1;
        let bits = bits & mask;

        // Try exact match first (covers ~60% of clean reads)
        let mut rbits = bits;
        for _ in 0..4 {
            if let Some(&id) = self.code_to_id.get(&rbits) {
                return Some((id, 0));
            }
            rbits = rotate90(rbits, self.dimension);
        }

        // Fall back to hamming search
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
                    rbits = rotate90(rbits, self.dimension);
                }
            }
            return best;
        }

        None
    }
}

/// Rotate a bit pattern 90 degrees clockwise.
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
    TagDictionary::new("36h11", 6, 11, &APRILTAG_36H11_CODES)
});

// ============================================================================
// AprilTag 16h5 (30 codes)
// ============================================================================

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
    TagDictionary::new("16h5", 4, 5, &APRILTAG_16H5_CODES)
});

// ============================================================================
// ArUco 4x4_50 (50 codes)
// ============================================================================

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
    TagDictionary::new("4X4_50", 4, 1, &ARUCO_4X4_50_CODES)
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
    TagDictionary::new("4X4_100", 4, 1, &ARUCO_4X4_100_CODES)
});

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_36h11_dict_size() {
        assert_eq!(APRILTAG_36H11.len(), 587);
    }

    #[test]
    fn test_16h5_dict_size() {
        assert_eq!(APRILTAG_16H5.len(), 30);
    }

    #[test]
    fn test_exact_decode() {
        let code0 = APRILTAG_36H11.get_code(0).unwrap();
        let result = APRILTAG_36H11.decode(code0, 0);
        assert_eq!(result, Some((0, 0)));
    }

    #[test]
    fn test_rotated_decode() {
        let code = APRILTAG_36H11.get_code(42).unwrap();
        let rotated = rotate90(code, 6);
        let result = APRILTAG_36H11.decode(rotated, 0);
        assert_eq!(result, Some((42, 0)));
    }

    #[test]
    fn test_hamming_tolerance() {
        let code = APRILTAG_36H11.get_code(100).unwrap();
        // Flip 2 bits
        let noisy = code ^ 0b11;
        let result = APRILTAG_36H11.decode(noisy, 2);
        assert_eq!(result, Some((100, 2)));
    }
}
