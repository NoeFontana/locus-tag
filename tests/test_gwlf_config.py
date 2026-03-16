import numpy as np
import locus

def test_gwlf_config_option():
    # Test creation via Detector class
    detector = locus.Detector(refinement_mode=locus.CornerRefinementMode.Gwlf)
    config = detector.config()
    assert config.refinement_mode == locus.CornerRefinementMode.Gwlf

    # Test that we can also use the integer value if needed
    detector2 = locus.Detector(refinement_mode=4) # 4 is Gwlf
    config2 = detector2.config()
    assert config2.refinement_mode == locus.CornerRefinementMode.Gwlf

def test_gwlf_telemetry():
    detector = locus.Detector(refinement_mode=locus.CornerRefinementMode.Gwlf)
    
    # Create a synthetic image with a square
    img = np.zeros((100, 100), dtype=np.uint8)
    img[20:60, 20:60] = 255
    
    # Detect with telemetry
    batch = detector.detect(img, debug_telemetry=True)
    
    if batch.telemetry:
        assert hasattr(batch.telemetry, "gwlf_fallback_count")
        assert hasattr(batch.telemetry, "gwlf_avg_delta")
        print(f"GWLF Fallback: {batch.telemetry.gwlf_fallback_count}")
        print(f"GWLF Avg Delta: {batch.telemetry.gwlf_avg_delta}")

if __name__ == "__main__":
    test_gwlf_config_option()
    test_gwlf_telemetry()
    print("Python GWLF tests passed!")
