"""Comparative SOTA benchmarking against pinned reference detectors.

Driven by ``cargo xtask sota`` (see ``xtask/README.md``). Every detector writes
the same JSONL schema, one object per image::

    {"image": str, "ms": float, "ids": [int], "corners": [[[x, y] x 4]], "convention": str}

``convention`` is ``"locus"`` (pixel centre at +0.5) or ``"opencv"`` (pixel centre
at integer); scorers convert to the dataset's ground-truth convention.
"""
