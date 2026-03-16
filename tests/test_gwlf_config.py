import locus


def test_gwlf_config():
    detector = locus.Detector(refinement_mode=locus.CornerRefinementMode.Gwlf)
    config = detector.config()
    assert config.refinement_mode == locus.CornerRefinementMode.Gwlf
    # Check that gwlf_transversal_alpha is exposed and has correct default
    assert config.gwlf_transversal_alpha == 0.01
    print("Python GWLF config test passed!")


if __name__ == "__main__":
    test_gwlf_config()
