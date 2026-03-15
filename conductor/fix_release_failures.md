# Fix Release Failures (v0.2.0) - Simplified

This plan addresses release workflow failures with minimal complexity and robust pathing.

## 1. Objective
Fix crate packaging, SIMD compilation, and CI environment issues with simple, maintainable solutions.

## 2. Changes

### 2.1 Crate Packaging (`locus-core`)
- **Move Dictionaries**: Move `data/dictionaries` to `crates/locus-core/data/dictionaries`.
- **Update Metadata**: 
    - Update `crates/locus-core/Cargo.toml` to include `data/dictionaries/**/*` in the package.
    - Update `crates/locus-core/build.rs` to look for dictionaries in `data/dictionaries` relative to the crate root.
    - Update any other references (e.g., `examples/dictionary_generation/extract_opencv.py`).

### 2.2 SIMD Implementation (`locus-core`)
- **Fix `aarch64` `rcp_nr`**:
    - Add `#[allow(unsafe_code)]`.
    - Fix NEON calls to use vector-in/vector-out intrinsics (`vrecpe_f32`, `vrecps_f32`, `vmul_f32`) and extract the lane. This is more standard than trying to find scalar-equivalent intrinsics which are often not exposed the same way in Rust.
- **Fix Warnings**: 
    - Remove unnecessary `unsafe` in `segmentation.rs`.
    - Fix unused assignment in `segmentation.rs`.

### 2.3 `sdist` Build Workflow
- **Update `.github/workflows/release.yml`**: 
    - Use `uv build --sdist` instead of `maturin sdist`. This avoids `maturin-action`'s dependency on `pip`.

## 3. Implementation Steps

### Step 1: Move Dictionaries and Update Paths
- `mv data/dictionaries crates/locus-core/data/dictionaries`
- Edit `crates/locus-core/Cargo.toml`: Add `include = ["src/**/*", "build.rs", "data/dictionaries/**/*", "templates/**/*"]`.
- Edit `crates/locus-core/build.rs`: Change `../../data/dictionaries` to `data/dictionaries`.

### Step 2: Fix SIMD and Warnings
- Edit `crates/locus-core/src/simd/math.rs`: Correct the NEON implementation.
- Edit `crates/locus-core/src/segmentation.rs`: Remove `unsafe` and unused variable.

### Step 3: Update Workflow
- Edit `.github/workflows/release.yml`: Change `maturin sdist` to `uv build --sdist`.

## 4. Verification
1.  `cargo build --release` (on x86_64).
2.  `cargo publish --dry-run` in `crates/locus-core`.
3.  `uv build --sdist` locally.
