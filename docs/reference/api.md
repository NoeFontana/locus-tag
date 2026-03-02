# API Reference

This page provides detailed information about the Locus Python API.

## Core Interface

The primary entry point for using Locus is the `Detector` class or the high-level convenience functions.

::: locus.Detector
    options:
        show_root_toc_entry: false
        members:
            - __init__
            - checkerboard
            - detect
            - detect_with_options
            - detect_with_stats
            - set_families

::: locus.detect_tags

::: locus.detect_tags_with_stats

## Configuration

Locus uses Pydantic for robust configuration validation.

::: locus.DetectorConfig
    options:
        heading_level: 3

::: locus.DetectOptions
    options:
        heading_level: 3

## Data Models

These classes represent the output and internal state of the detection pipeline.

::: locus.Detection
    options:
        heading_level: 3

::: locus.Pose
    options:
        heading_level: 3

::: locus.PipelineStats
    options:
        heading_level: 3

## Geometry

::: locus.CameraIntrinsics
    options:
        heading_level: 3

## Enumerations

::: locus.TagFamily
    options:
        heading_level: 3

::: locus.DecodeMode
    options:
        heading_level: 3

::: locus.PoseEstimationMode
    options:
        heading_level: 3

::: locus.CornerRefinementMode
    options:
        heading_level: 3

::: locus.SegmentationConnectivity
    options:
        heading_level: 3

## Debugging Utilities

These tools are provided for performance profiling and algorithm inspection. They are not intended for use in production pipelines.

::: locus.FullDetectionResult
    options:
        heading_level: 3

::: locus.Detector.detect_full

::: locus.Detector.extract_candidates

::: locus.dummy_detect

::: locus.debug_threshold

::: locus.debug_segmentation
