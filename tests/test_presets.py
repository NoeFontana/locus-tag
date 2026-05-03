import locus
import pytest


def test_profile_initialization():
    """`Detector(profile=...)` loads shipped JSON profiles with the expected settings."""
    d1 = locus.Detector(profile="high_accuracy")
    assert d1.config().quad.extraction_mode == locus.QuadExtractionMode.EdLines

    d3 = locus.Detector(profile="grid")
    assert d3.config().segmentation.connectivity == locus.SegmentationConnectivity.Four

    d4 = locus.Detector(profile="standard")
    assert d4.config().decoder.refinement_mode == locus.CornerRefinementMode.Erf


def test_profile_and_config_mutually_exclusive():
    cfg = locus.DetectorConfig.from_profile("standard")
    with pytest.raises(ValueError, match="Pass either"):
        locus.Detector(profile="standard", config=cfg)


def test_unknown_profile_rejected():
    with pytest.raises(Exception):  # noqa: B017 — Pydantic ValidationError / FileNotFoundError path
        locus.Detector(profile="does_not_exist")  # pyright: ignore[reportArgumentType]


def test_config_overrides_via_pydantic():
    """Custom `DetectorConfig` overrides cleanly propagate to the built detector."""
    cfg = locus.DetectorConfig.from_profile("high_accuracy")
    cfg_dict = cfg.model_dump()
    cfg_dict["quad"]["extraction_mode"] = "ContourRdp"
    overridden = locus.DetectorConfig.model_validate(cfg_dict)
    d = locus.Detector(config=overridden)
    assert d.config().quad.extraction_mode == locus.QuadExtractionMode.ContourRdp
    assert d.config().threshold.enable_sharpening is False  # high_accuracy default
