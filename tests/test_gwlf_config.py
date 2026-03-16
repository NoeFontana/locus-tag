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

if __name__ == "__main__":
    test_gwlf_config_option()
    print("Python GWLF config test passed!")
