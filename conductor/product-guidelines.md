# Locus Product Guidelines

## 1. API Design Philosophy
- **Python Interface:** Must remain accessible, pythonic, and configurable. It should hide the complexities of Rust while providing developers with deep control over detection parameters when needed.
- **Rust Core Interface:** Must be highly configurable and prioritize absolute performance. Operations should embrace a "Performance First" mindset, utilizing strict zero-copy semantics and efficient memory layouts.

## 2. Error Handling & Safety
- **Strict No-Panic Policy:** Rust code must never panic (`#![deny(clippy::unwrap_used)]`). All failures, edge cases, and unexpected inputs must be handled gracefully by returning clear, descriptive `Result` types.
- **Contextual Feedback:** Ensure that errors bubbling up to Python are translated into meaningful Python exceptions with actionable context.

## 3. Documentation Standards
- **System Architecture & Config:** Clearly document the "Universal Quad Detector" pipeline, explaining how configuration parameters affect the underlying algorithms.
- **Performance Metrics:** Prioritize latency figures, benchmarking details, and performance implications for every major function and configuration toggle.
- **Visual Explanations:** Utilize the Rerun SDK and visual aids to explain thresholding stages, quad fitting, and detection steps to help users understand the pipeline visually.