"""Raw integers must be rejected where typed PyO3 enums are expected."""

from __future__ import annotations

import pytest

import locus
from locus import _create_detector_from_config


def _base_flat() -> dict:
    return locus.DetectorConfig.from_profile("standard")._to_flat_ffi_dict()


@pytest.mark.parametrize(
    "field",
    [
        "quad_extraction_mode",
        "refinement_mode",
        "decode_mode",
        "segmentation_connectivity",
    ],
)
def test_raw_int_rejected_for_enum_fields(field: str) -> None:
    flat = _base_flat()
    flat[field] = 0
    with pytest.raises((TypeError, ValueError)):
        _create_detector_from_config(
            config=flat,
            decimation=None,
            threads=None,
            families=[int(locus.TagFamily.AprilTag36h11)],
        )


def test_typed_enum_accepted() -> None:
    flat = _base_flat()
    det = _create_detector_from_config(
        config=flat,
        decimation=None,
        threads=None,
        families=[int(locus.TagFamily.AprilTag36h11)],
    )
    assert det is not None
