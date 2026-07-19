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
| [Pose rotation-error tail & edge refinement](rotation-tail-and-edge-refinement.md) | RESOLVED (single-frame) | Corner-level levers are all trade-bound; adding information (model-edge refinement, v0.7.0) beat reshaping the 4 corners. |
| [EdLines quad extraction & segmentation](edlines-segmentation.md) | ACTIVE | EdLines ships in `high_accuracy`; the arc-balance selector and the Phase-5 line decoupling were both falsified (decoder-rejection invisibility; chord-cost is a real regulariser). |
| [Recall, quad extraction & ICRA](recall-quad-icra.md) | CLOSED | No static `(extraction, refinement)` pair wins both regimes; PPB adaptive routing does. `max_recall_adaptive` was folded into `high_accuracy`. |

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
