use locus_core::segmentation::label_components_with_stats;
use bumpalo::Bump;

#[test]
fn test_segmentation_correctness_large_image() {
    let arena = Bump::new();
    let width = 3840; // 4K width
    let height = 2160; // 4K height
    let mut binary = vec![255u8; width * height];
    
    // Create multiple separate components
    // 1. A square at the top left
    for y in 100..200 {
        for x in 100..200 {
            binary[y * width + x] = 0;
        }
    }
    
    // 2. A long horizontal strip in the middle
    for x in 500..3000 {
        binary[1000 * width + x] = 0;
        binary[1001 * width + x] = 0;
    }
    
    // 3. Horizontal stripes at the bottom
    for y in 1800..2000 {
        if y % 4 == 0 {
            for x in 1800..2000 {
                binary[y * width + x] = 0;
            }
        }
    }
    
    // 4. Noise (avoiding the square area)
    for y in 0..height {
        if y % 10 == 0 {
            for x in 0..width {
                if (x < 100 || x >= 200 || y < 100 || y >= 200) && (x + y) % 31 == 0 {
                    binary[y * width + x] = 0;
                }
            }
        }
    }

    // Run segmentation
    let start = std::time::Instant::now();
    let result = label_components_with_stats(&arena, &binary, width, height, true);
    let duration = start.elapsed();
    
    // Basic verification of component counts
    assert!(result.component_stats.len() > 1000);
    
    println!("Found {} components on 4K image in {:?}", result.component_stats.len(), duration);
}
