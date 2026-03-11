# Locus — Claude Code Instructions

> **Single source of truth:** This file delegates to the shared `.agent/` directory.
> Do NOT duplicate rules here. Update the canonical files instead.

## Meta-Rules & Context Routing

@GEMINI.md

## Engineering Rules

@.agent/rules/core.md

@.agent/rules/architecture.md

@.agent/rules/constraints.md

@.agent/rules/detection-batch-contract.md

@.agent/rules/quality-gates.md

## Skills (Slash Commands)

The following skills are available via `/skill-name`:

| Command | Description |
| :--- | :--- |
| `/testing` | Run and evaluate the full test suite (Rust + Python). |
| `/performance_benchmark` | Run and analyse performance benchmarks. |
| `/release` | Manage the end-to-end release lifecycle. |

Skill definitions live in `.agent/skills/<name>/SKILL.md`.
