# Core Identity & Principles

## 1. Project Identity
* **Name:** Locus (`locus-vision`)
* **Mission:** Production-grade, memory-safe, state-of-the-art fiducial marker detector.
* **Target Audience:** Robotics and AV Perception Engineers.

## 2. Engineering Directives
- **Latency Obsessed:** Scrutinize every cache miss and heap allocation.
- **Safety First:** Thoroughly document every `unsafe` block with a `// SAFETY:` comment explaining why it is sound.
- **Modern Tooling:** Utilize `rerun` for visual debugging and `uv` for Python environment management.