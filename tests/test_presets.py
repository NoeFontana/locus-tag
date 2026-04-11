import locus
import pytest


def test_preset_initialization():
    """Test that Detector can be initialized with presets."""
    d1 = locus.Detector(preset=locus.DetectorPreset.Metrology)
    assert d1.config().quad_extraction_mode == locus.QuadExtractionMode.EdLines

    d2 = locus.Detector(preset=locus.DetectorPreset.PureTags)
    assert d2.config().decode_mode == locus.DecodeMode.Soft

    d3 = locus.Detector(preset=locus.DetectorPreset.Checkerboard)
    assert d3.config().segmentation_connectivity == locus.SegmentationConnectivity.Four

    d4 = locus.Detector(preset=locus.DetectorPreset.Production)
    assert d4.config().refinement_mode == locus.CornerRefinementMode.Erf

    d5 = locus.Detector(preset=locus.DetectorPreset.Fast)
    assert d5.config().threshold_tile_size == 8


def test_preset_warnings():
    """Test that conflicting kwargs trigger the appropriate error or warning."""
    # EdLines + Soft is now a hard validation error.
    with pytest.raises(ValueError, match="EdLines.*Soft"):
        locus.Detector(preset=locus.DetectorPreset.Metrology, decode_mode=locus.DecodeMode.Soft)

    with pytest.warns(UserWarning, match="Checkerboard preset relies on 4-connectivity"):
        locus.Detector(
            preset=locus.DetectorPreset.Checkerboard,
            segmentation_connectivity=locus.SegmentationConnectivity.Eight,
        )


def test_override_preset():
    """Verify that explicit arguments override preset defaults correctly."""
    # Metrology defaults to EdLines
    d1 = locus.Detector(preset=locus.DetectorPreset.Metrology)
    assert d1.config().quad_extraction_mode == locus.QuadExtractionMode.EdLines

    # Override with ContourRdp
    d2 = locus.Detector(
        preset=locus.DetectorPreset.Metrology,
        quad_extraction_mode=locus.QuadExtractionMode.ContourRdp,
    )
    assert d2.config().quad_extraction_mode == locus.QuadExtractionMode.ContourRdp

    # Check that unrelated preset settings are preserved
    assert d2.config().enable_sharpening is False  # Metrology default
