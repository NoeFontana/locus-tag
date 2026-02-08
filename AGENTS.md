# Agent Guide: Locus Perception Engineer

This guide is the entry point for AI agents working on `locus-vision`.

## 1. Modular Context (Rules)
For deep context, reference the specific rule files:
- **Identity & Persona:** `@core` (Start here)
- **Architecture:** `@architecture` (Data patterns & SIMD)
- **Constraints:** `@constraints` (Strict safety & memory rules)
- **Quality Gates:** `@quality-gates` (Mandatory checks)

## 2. Workflows
- **Build & Test:** Use standard workflows (`/test`).
- **Performance:** Use `/bench` and see `@quality-gates` for targets.

## 3. Operations
- **Lint:** `/lint`
- **Format:** `/format`
- **Type Check:** `/type_check`
- **Visual Debug:** Use `rerun` (`locus.viz`).
