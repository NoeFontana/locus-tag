# API Reference

This page provides detailed information about the Locus Python API.

## Core Interface

The primary entry point for using Locus is the `Detector` class.

::: locus.Detector
    options:
        show_root_toc_entry: false
        members:
            - __init__
            - checkerboard
            - detect
            - detect_with_options
            - set_families

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
