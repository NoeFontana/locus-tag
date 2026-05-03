# How-to: Add a custom fiducial dictionary

This guide walks you through registering a new tag family in Locus.

> **Heads-up: this is a build-time procedure.** The set of tag families is
> compiled into the wheel — there is no `Detector.register_family(...)` runtime
> hook. Adding a family means editing the source and rebuilding `locus-core`.
> If all you need is one of the *already-shipped* families (the
> `locus.TagFamily` enum), use it directly and skip this guide.

---

## 1. Decide which path you need

| Goal | What to do |
| --- | --- |
| Use a different ArUco preset Locus already ships (e.g. `ArUco6x6_250`). | Pass the matching `locus.TagFamily.*` enum value. No code changes. |
| Use an OpenCV ArUco preset Locus does **not** ship (e.g. `DICT_5X5_50`). | Run `extract_opencv.py` to generate the JSON IR, then register a new variant (steps 2 + 3). |
| Use a wholly custom code table you generated yourself (e.g. an STag dictionary, a private codeword set). | Author the JSON IR by hand, then register a new variant (steps 2 + 3). |

---

## 2. Provide the dictionary JSON

Locus consumes a small intermediate representation (IR). Each shipped
dictionary lives at `crates/locus-core/data/dictionaries/*.json` and looks
like this:

```json
{
  "payload_length": 16,
  "minimum_hamming_distance": 4,
  "dictionary_size": 50,
  "canonical_sampling_points": [
    [-0.5, -0.5], [-0.1667, -0.5], [0.1667, -0.5], [0.5, -0.5],
    [-0.5, -0.1667], [-0.1667, -0.1667], ...
  ],
  "base_codes": ["0xAB12C0...", "0x4D8E70...", "..."]
}
```

| Field | Meaning |
| --- | --- |
| `payload_length` | Number of payload bits (e.g. `16` for a 4×4 grid, `36` for a 6×6 grid, `49` for 7×7). |
| `minimum_hamming_distance` | Minimum Hamming distance between any two codes in the dictionary. Used by the decoder's tolerance gate. |
| `dictionary_size` | Number of distinct codes (i.e. `len(base_codes)`). |
| `canonical_sampling_points` | Bit-cell centres in tag-local coordinates `[-1, 1]`, **row-major**. The square `[-1, 1]` covers the *full* tag including its 1-cell-wide border. The list length must equal `payload_length`. |
| `base_codes` | One hex bit-string per tag, rotation 0 (the canonical orientation). The build script computes the other three rotations automatically. |

### Generating from an OpenCV preset

`examples/dictionary_generation/extract_opencv.py` does this for you for any
dictionary in `cv2.aruco`:

```bash
uv run examples/dictionary_generation/extract_opencv.py --dict DICT_5X5_50
```

The script writes the JSON to `crates/locus-core/data/dictionaries/dict_5x5_50.json`.
See the script's [README](https://github.com/NoeFontana/locus-tag/blob/main/examples/dictionary_generation/README.md)
for the full list of supported presets.

### Authoring by hand

If your codes don't come from OpenCV, hand-author a JSON file matching the
schema above. Use one of the shipped files (e.g. `dict_4x4_50.json`) as a
template, and pay particular attention to:

- **Bit ordering.** The bit at index `i` of a code corresponds to the cell at
  `canonical_sampling_points[i]`. Locus follows OpenCV's row-major convention
  (left-to-right, top-to-bottom).
- **Sampling-point space.** Coordinates are in `[-1, 1]` and include the
  border, *not* `[-0.5, 0.5]` over data bits only. For a `D × D` data grid the
  full tag is `(D+2) × (D+2)`; bit-cell centres are at
  `((x + 1.5) · 2 / (D+2) - 1, (y + 1.5) · 2 / (D+2) - 1)` for
  `x, y ∈ [0, D)`.
- **Dictionary size.** Keep this in sync with `len(base_codes)` —
  `build.rs` does not validate it for you.

---

## 3. Register the family in the source tree

Five files name the family explicitly. All five must be updated in lock-step
or the build will fail.

### 3.1 Add the `FAMILY_MAPPING` row

`crates/locus-core/build.rs`:

```rust
const FAMILY_MAPPING: &[(&str, &str, usize)] = &[
    ("AprilTag16h5",  "dict_apriltag_16h5", 4),
    ("AprilTag36h11", "dict_apriltag_36h11", 6),
    ("ArUco4x4_50",   "dict_4x4_50",  4),
    ("ArUco4x4_100",  "dict_4x4_100", 4),
    ("ArUco6x6_250",  "dict_6x6_250", 6),
    ("ArUco5x5_50",   "dict_5x5_50",  5),  // ← add me
];
```

The tuple is `(enum-variant-name, JSON-file-stem, grid-dimension)`. The build
script reads the matching JSON, computes all four rotations and the
Multi-Index Hashing tables, and codegens a `&'static DICT_<NAME>: TagDictionary`
into `OUT_DIR/dictionaries.rs`.

### 3.2 Add the enum variant

`crates/locus-core/src/config.rs` (the `TagFamily` enum, `pub enum TagFamily`):

```rust
pub enum TagFamily {
    AprilTag16h5,
    AprilTag36h11,
    ArUco4x4_50,
    ArUco4x4_100,
    ArUco6x6_250,
    ArUco5x5_50,  // ← add me
}
```

Also add the same variant to `TagFamily::all()` in the same file so iteration
helpers see it.

### 3.3 Wire the dictionary lookup

`crates/locus-core/src/dictionaries.rs`, `get_dictionary`:

```rust
pub fn get_dictionary(family: TagFamily) -> &'static TagDictionary {
    match family {
        TagFamily::AprilTag16h5  => &DICT_APRILTAG16H5,
        TagFamily::AprilTag36h11 => &DICT_APRILTAG36H11,
        TagFamily::ArUco4x4_50   => &DICT_ARUCO4X4_50,
        TagFamily::ArUco4x4_100  => &DICT_ARUCO4X4_100,
        TagFamily::ArUco6x6_250  => &DICT_ARUCO6X6_250,
        TagFamily::ArUco5x5_50   => &DICT_ARUCO5X5_50,  // ← add me
    }
}
```

### 3.4 Expose the variant to Python

`crates/locus-py/src/lib.rs` — three places, all in the same file:

1. The `#[pyclass] enum TagFamily` — add the variant.
2. The `From<TagFamily> for locus_core::TagFamily` impl — add the match arm.
3. The `tag_family_from_i32` helper used by the deserializer — add the
   integer discriminant arm.

If you forget any of these three, the wheel will compile but the family will
panic with `Invalid TagFamily value` at the FFI boundary.

### 3.5 Update the type stub

`crates/locus-py/locus/locus.pyi` — add the new value to the `TagFamily`
enum so type checkers see it.

---

## 4. Rebuild and verify

```bash
# 1. Rebuild the wheel; build.rs picks up the new JSON + mapping.
uv run maturin develop --release --manifest-path crates/locus-py/Cargo.toml

# 2. Smoke-test in Python.
uv run python -c "
import locus
print(locus.TagFamily.ArUco5x5_50)
det = locus.Detector(families=[locus.TagFamily.ArUco5x5_50])
print('OK')
"
```

If `build.rs` fails with `failed to read JSON` the most common causes are:

- The file stem in `FAMILY_MAPPING` doesn't match the filename on disk.
- `dictionary_size` disagrees with `len(base_codes)`.
- `canonical_sampling_points` length disagrees with `payload_length`.

For a real-world tested example, look at any of the
`crates/locus-core/data/dictionaries/dict_*.json` files alongside their
matching `FAMILY_MAPPING` row.

---

## 5. The `TagDecoder` trait (advanced)

Most custom dictionaries fit the JSON-IR path above. If you need bespoke
decoding logic — non-square grids, multi-stage error correction, families
that aren't a simple bitwise codeword lookup — implement
[`TagDecoder`](https://github.com/NoeFontana/locus-tag/blob/main/crates/locus-core/src/decoder.rs)
in `locus-core` and wire it through the same five files in step 3. The trait
is intentionally minimal: `name`, `dimension`, `sample_points`, `decode`,
`decode_full`, `rotated_codes`. The shipped `AprilTagDecoder` and
`ArUcoDecoder` impls in `decoder.rs` are the reference implementations.
