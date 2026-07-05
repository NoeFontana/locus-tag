"""Parallel parameter-tuning + comparative-evaluation harness for locus-tag.

Additive, Python-side research tooling built on top of the existing bench
substrate (``LibraryWrapper`` detectors, ``Collector``/``ObservationRecord``
Tier-1 records, 5-axis stratification). Nothing here touches the Rust core or
the insta snapshot regression suite — it reuses the same wrappers and metric
conventions to sweep detector configs across cores and compare tuned Locus
against tuned competitors.

See ``docs`` / the plan for the phase breakdown. Public entry points are the
``bench sweep`` / ``bench tune`` / ``bench compare-report`` CLI subcommands.
"""

from __future__ import annotations
