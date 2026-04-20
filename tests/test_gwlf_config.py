import locus


def test_gwlf_config():
    cfg = locus.DetectorConfig.from_profile("standard")
    cfg_dict = cfg.model_dump()
    cfg_dict["decoder"]["refinement_mode"] = "Gwlf"
    custom = locus.DetectorConfig.model_validate(cfg_dict)

    detector = locus.Detector(config=custom)
    out = detector.config()
    assert out.decoder.refinement_mode == locus.CornerRefinementMode.Gwlf
    assert out.decoder.gwlf_transversal_alpha == 0.01


if __name__ == "__main__":
    test_gwlf_config()
