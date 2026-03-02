# Locus: AI Context Router & Meta-Rules

## 1. The Fallback Meta-Rule (Strict)
The AI assistant MUST NEVER hallucinate project constraints, architectural patterns, or engineering rules. If a rule or constraint is not explicitly found in the designated `docs/` files or this router, you MUST stop and ask the human engineer for clarification rather than assuming industry standard practices.

## 2. Dynamic Context Routing
Before performing any task, the AI MUST read the relevant documentation mapped below. These files are the single source of truth for the project.

### 2.1 Coding Conventions & Implementation
When asked to modify, refactor, or write new code:
* **Rust:** Read `docs/engineering/rust-style.md`.
* **Python:** Read `docs/engineering/python-style.md`.
* **General Style:** Read `docs/engineering/general-style.md`.

### 2.2 System Design & Architectural Invariants
When asked to implement new features, modify the detection pipeline, or optimize performance (the "hot loop"):
* **Architecture:** Consult `docs/explanation/architecture.md`.
* **Memory & Safety:** Consult `docs/engineering/constraints.md`.
* **Coordinates & Math:** Consult `docs/explanation/coordinates.md`.

### 2.3 Quality Standards & Verification
Before finalyzing any task, commit, or pull request:
* **Quality Gates:** Consult `docs/engineering/quality-gates.md`.
* **Testing Standards:** Follow the `Testing Requirements` in `conductor/workflow.md`.

## 3. Project Identity
* **Name:** Locus (`locus-vision`)
* **Mission:** Production-grade, memory-safe, high-performance fiducial marker detector.
* **Core Goal:** Minimize latency for robotics and autonomous systems.
