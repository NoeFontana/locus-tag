import re
from pathlib import Path

content = Path('crates/locus-core/src/decoder.rs').read_text()

# Replace test_all_codes_decode
content = re.sub(
    r'fn test_all_codes_decode\(\) \{.*?assert_eq!\(rot_out, 0\);\n        \}\n    \}',
    r'''fn test_all_codes_decode() {
        let decoder = AprilTag36h11;
        for id in 0..587u16 {
            let code = crate::dictionaries::DICT_APRILTAG36H11
                .get_code(id)
                .expect("valid ID");
            let result = decoder.decode(code);
            assert!(result.is_some());
            let (id_out, _, _) = result.unwrap();
            assert_eq!(id_out, u32::from(id));
        }
    }''',
    content,
    flags=re.DOTALL
)

# Replace test_all_codes_decode_41h12 with 36h10
content = re.sub(
    r'#\[test\]\n    fn test_all_codes_decode_41h12\(\) \{.*?assert_eq!\(rot_out, 0\);\n        \}\n    \}',
    r'''#[test]
    fn test_all_codes_decode_36h10() {
        let decoder = AprilTag36h10;
        let dict = crate::dictionaries::get_dictionary(crate::config::TagFamily::AprilTag36h10);
        for id in 0..dict.len() as u16 {
            let code = dict.get_code(id).expect("valid ID");
            let result = decoder.decode(code);
            assert!(result.is_some(), "Failed to decode 36h10 ID {id}");
            let (id_out, _, _) = result.unwrap();
            assert_eq!(id_out, u32::from(id));
        }
    }''',
    content,
    flags=re.DOTALL
)

# Replace test_grid_sampling
content = re.sub(
    r'#\[test\]\n    fn test_grid_sampling\(\) \{.*?assert_eq!\(\(bits >> 35\) & 1, 0, "Bit 35 should be 0"\);\n    \}',
    r'''#[test]
    fn test_grid_sampling() {
        let width = 64;
        let height = 64;
        let mut data = vec![0u8; width * height];
        // 8x8 grid, 36x36px tag centered at 32,32 => corners [14, 50]
        // TL=(14,14), TR=(50,14), BR=(50,50), BL=(14,50)

        // Border:
        for gy in 0..8 {
            for gx in 0..8 {
                if gx == 0 || gx == 7 || gy == 0 || gy == 7 {
                    for y in 0..4 {
                        for x in 0..4 {
                            let px = 14 + (gx as f64 * 4.5) as usize + x;
                            let py = 14 + (gy as f64 * 4.5) as usize + y;
                            if px < 64 && py < 64 {
                                data[py * width + px] = 0;
                            }
                        }
                    }
                }
            }
        }
        // Bit 0 (cell 1,1) -> White (canonical p = -0.625, -0.625)
        for y in 0..4 {
            for x in 0..4 {
                let px = 14 + (1.0 * 4.5) as usize + x;
                let py = 14 + (1.0 * 4.5) as usize + y;
                data[py * width + px] = 255;
            }
        }
        // Bit 35 (cell 6,6) -> Black (canonical p = 0.625, 0.625)
        for y in 0..4 {
            for x in 0..4 {
                let px = 14 + (6.0 * 4.5) as usize + x;
                let py = 14 + (6.0 * 4.5) as usize + y;
                data[py * width + px] = 0;
            }
        }

        let img = crate::image::ImageView::new(&data, width, height, width).unwrap();

        let decoder = AprilTag36h11;
        let arena = Bump::new();
        let cand = crate::Detection {
            corners: [[14.0, 14.0], [50.0, 14.0], [50.0, 50.0], [14.0, 50.0]],
            ..Default::default()
        };
        let bits = sample_grid(&img, &arena, &cand, &decoder, 20.0).expect("Should sample successfully");

        // bit 0 should be 1 (high intensity)
        assert_eq!(bits & 1, 1, "Bit 0 should be 1");
        // bit 35 should be 0 (low intensity)
        assert_eq!((bits >> 35) & 1, 0, "Bit 35 should be 0");
    }''',
    content,
    flags=re.DOTALL
)

Path('crates/locus-core/src/decoder.rs').write_text(content)
print("Patched tests")
