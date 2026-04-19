import locus
import pytest


def test_pinhole_always_available():
    assert locus.DistortionModel.Pinhole is not None
    k = locus.CameraIntrinsics(
        fx=800.0,
        fy=800.0,
        cx=400.0,
        cy=300.0,
        distortion_model=locus.DistortionModel.Pinhole,
    )
    assert k.distortion_model == locus.DistortionModel.Pinhole


@pytest.mark.skipif(locus.HAS_NON_RECTIFIED, reason="lean-build guardrail")
@pytest.mark.parametrize("variant", ["BrownConrady", "KannalaBrandt"])
def test_stripped_variants_raise_locus_feature_error(variant):
    with pytest.raises(locus.LocusFeatureError, match="non_rectified"):
        getattr(locus.DistortionModel, variant)


@pytest.mark.skipif(locus.HAS_NON_RECTIFIED, reason="lean-build guardrail")
def test_unknown_attr_still_raises_attribute_error():
    with pytest.raises(AttributeError):
        _ = locus.DistortionModel.NotAVariant
