
import numpy as np
import locus
import logging
import sys

# Configure logging to see Rust's tracing output if it's bridged (tracing-subscriber often prints to stderr by default)
# Note: locus-py doesn't automatically install a subscriber, so we might not see it unless we init it in Rust or use RUST_LOG env var.
# But we can at least verify it doesn't crash.

def verify_non_contiguous():
    print("Creating synthetic image...")
    # 100x100
    img = np.zeros((100, 100), dtype=np.uint8)
    
    # Draw a simple quad
    img[20:80, 20:80] = 255
    img[30:70, 30:70] = 0
    
    print("Testing contiguous detection...")
    dets = locus.detect_tags(img)
    print(f"Contiguous detection found: {len(dets)} (Expected 0 or 1 depending on noise/pattern)")

    # Create non-contiguous slice
    # Resample columns by 2
    # Stride will be 2 * 1 = 2 bytes for inner dimension? No, stride_x will be 2.
    print("Testing non-contiguous detection (slice [:, ::2])...")
    img_sliced = img[:, ::2]
    
    if img_sliced.flags['C_CONTIGUOUS']:
        print("Error: Sliced image IS contiguous? That's unexpected.")
        print(f"Flags: {img_sliced.flags}")
        sys.exit(1)
        
    print(f"Sliced shape: {img_sliced.shape}")
    print(f"Sliced strides: {img_sliced.strides}")
    
    try:
        dets_sliced = locus.detect_tags(img_sliced)
        print(f"Non-contiguous detection found: {len(dets_sliced)}")
        print("SUCCESS: Non-contiguous array accepted without panic.")
    except Exception as e:
        print(f"FAILURE: Exception occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    verify_non_contiguous()
