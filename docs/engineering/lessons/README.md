# Engineering lessons

Durable conclusions from Locus's investigation history — what we tried, what we
shipped, and what was empirically falsified. Each page distills a cluster of
point-in-time postmortems and diagnostics (now removed; recoverable from git
history) into one maintainable record, so a future engineer can find "did we
already try X, and why didn't it work?" without reading twenty dated files.

This pairs with the anti-pattern index in the agent memory: the memory is the
fast "don't do X" lookup; these pages carry the mechanism and the re-attempt
conditions.

## Index

| Lesson | Status | One-line takeaway |
| :--- | :--- | :--- |
| [Pose covariance calibration](pose-covariance.md) | CLOSED | Every diagonal/scalar rescale of `(JᵀWJ)⁻¹` is falsified; the miscalibration is off-diagonal structure. Frontier: model-edge Fisher covariance (unmerged). |
| [Pose rotation-error tail & edge refinement](rotation-tail-and-edge-refinement.md) | RESOLVED (single-frame); photometric inset SHIPPED 2026-10-04 | Corner-level levers are all trade-bound; adding information (model-edge refinement, v0.7.0) beat reshaping the 4 corners. The corner bias is photometric (blur in linear light, then the tone curve), not a PSF floor; since #434 each decoded marker calibrates its own inset from its bit edges (`decoder.corner_subpix`), and the board pose estimates one per corner estimator. |
| [EdLines quad extraction & segmentation](edlines-segmentation.md) | ACTIVE | EdLines ships in `high_accuracy`; the arc-balance selector and the Phase-5 line decoupling were both falsified (decoder-rejection invisibility; chord-cost is a real regulariser). Open defect: ~+0.5 px outward bias in linear light, masked by sRGB on render-tag. |
| [Recall, quad extraction & ICRA](recall-quad-icra.md) | ACTIVE | No static `(extraction, refinement)` pair wins both regimes; PPB adaptive routing does. Real-image recall was limited by the segmentation model: the filled-blob gates and the refine-before-decode latency are resolved in `standard` (decode-first with the dark-ring check, #433; 4-connectivity, gates off, clipped and gross corners, #436 — EuRoC recall 20.9 → 86.0 %). The min/max-*midpoint* cut was the next limit and is resolved: the tile cut moved to 179/400 (0.4475) of the neighbourhood range (#449), taking EuRoC 86.0 → 92.1 % and Liu4K 37.2 → 44.6 %. Still open: a merge-robust quad seed, and the seeded split that would let the cut erode for topology without eroding small markers out of existence (the measured 99.6 % / 74.6 % frontier). |

## Convention — how to record a new learning (keeps this from re-sprawling)

1. **Two homes only.** A durable conclusion goes into the relevant `lessons/<topic>.md`
   page as a **dated subsection** — never a new top-level `docs/engineering/*_YYYYMMDD.md`
   file. Current benchmark numbers go in `../benchmarking/` (one live snapshot per
   metric family; supersede it in place).
2. **Status banner is mandatory.** Every lesson page (and every dated subsection)
   opens with `Status: ACTIVE | CLOSED | RESOLVED | SUPERSEDED-BY <link> | FALSIFIED`.
3. **Supersede, don't accrete.** When a re-run replaces an old snapshot, overwrite or
   delete the old one — do not add a dated sibling.
4. **Distill, then delete.** Once a postmortem's conclusion is captured here, delete the
   raw file; git history is the archive. Keep only pages that a future engineer would
   actually open.
5. **CHANGELOG references are historical, not live links.** Moving or deleting a doc that
   only `CHANGELOG.md` cites is safe.
