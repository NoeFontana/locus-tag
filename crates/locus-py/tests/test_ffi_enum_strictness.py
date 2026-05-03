"""Raw integers must be rejected where typed PyO3 enums are expected."""

from __future__ import annotations

import locus
import pytest
from locus.locus import PyDetectorConfig


def _base_kwargs() -> dict:
    cfg = locus.DetectorConfig.from_profile("standard")._to_ffi_config()
    return {name: getattr(cfg, name) for name in dir(cfg) if not name.startswith("_")}


@pytest.mark.parametrize(
    "field",
    [
        "quad_extraction_mode",
        "refinement_mode",
        "segmentation_connectivity",
    ],
)
def test_raw_int_rejected_for_enum_fields(field: str) -> None:
    kwargs = _base_kwargs()
    kwargs[field] = 0
    with pytest.raises((TypeError, ValueError)):
        PyDetectorConfig(**kwargs)


def test_typed_enum_accepted() -> None:
    kwargs = _base_kwargs()
    cfg = PyDetectorConfig(**kwargs)
    det = locus._create_detector_from_config(
        config=cfg,
        decimation=None,
        threads=None,
        families=[int(locus.TagFamily.AprilTag36h11)],
    )
    assert det is not None
