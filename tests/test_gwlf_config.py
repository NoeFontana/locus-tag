import locus
import numpy as np

def test_gwlf_config():
    config = locus.create_detector(refinement_mode=locus.CornerRefinementMode.Gwlf).config()
    assert config.refinement_mode == locus.CornerRefinementMode.Gwlf
    print("Python GWLF config test passed!")

if __name__ == "__main__":
    test_gwlf_config()
