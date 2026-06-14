# Locus — Claude Code Instructions

## Fallback meta-rule (strict)
NEVER hallucinate project constraints, architecture, or engineering rules.
If a rule isn't in the files referenced below, stop and ask the engineer
rather than assume an industry-standard practice.

## Project identity
- **Name**: Locus (`locus-tag`).
- **Mission**: production-grade, memory-safe, high-performance fiducial-marker
  detector for robotics, AV, and perception engineers.
- **Single source of truth**: canonical docs live under `docs/`. The `.agent/`
  tree contains skills only.

## Always loaded (every session)
@docs/engineering/core.md

@docs/engineering/constraints.md

## Load on demand
Read the relevant file when the task touches its area.

| Topic | File |
| :--- | :--- |
| System architecture, pipeline stages, component diagrams | `docs/explanation/architecture.md` |
| `DetectionBatch` SoA contract, phase R/W privileges | `docs/engineering/detection-batch-contract.md` |
| Pre-commit / pre-PR commands, regression suites, snapshot flow | `docs/engineering/quality-gates.md` |
| Rust style | `docs/engineering/rust-style.md` |
| Python style | `docs/engineering/python-style.md` |
| General style | `docs/engineering/general-style.md` |
| Coordinates & math conventions | `docs/explanation/coordinates.md` |
| Memory model (SoA / arena / FFI) | `docs/explanation/memory_model.md` |
| Workflow (branching, testing requirements) | `docs/engineering/workflow.md` |
| Release lifecycle (cargo-release, OIDC, mike, rollback) | `docs/engineering/release-runbook.md` |

## Skills (slash commands)
| Command | Description |
| :--- | :--- |
| `/testing` | Run and evaluate the full test suite (Rust + Python) |
| `/performance_benchmark` | Run and analyse performance benchmarks |
| `/release` | Manage the end-to-end release lifecycle |

Skill definitions live in `.agent/skills/<name>/SKILL.md`.
