# OpenCV ArUco dictionary extractor

`extract_opencv.py` converts OpenCV's predefined ArUco dictionaries into the
JSON intermediate representation that `locus-core`'s `build.rs` consumes.

## When to use this

Run this script when you want Locus to recognise an OpenCV ArUco preset that
is not already shipped (the shipped set lives in
`crates/locus-core/data/dictionaries/dict_*.json`).

If you only need the families exposed via `locus.TagFamily` you do **not**
need this script.

## Dependencies

```bash
uv pip install opencv-python  # or opencv-python-headless
```

## Usage

```bash
# Extract every preset listed in STANDARD_FAMILIES at the top of the script.
uv run examples/dictionary_generation/extract_opencv.py --all

# Extract a single preset.
uv run examples/dictionary_generation/extract_opencv.py --dict DICT_5X5_50

# Override the output directory (default: crates/locus-core/data/dictionaries).
uv run examples/dictionary_generation/extract_opencv.py --all --output /tmp/dicts
```

The supported preset list is the `STANDARD_FAMILIES` constant at the top of
the script — currently `DICT_4X4_*`, `DICT_5X5_*`, `DICT_6X6_*`,
`DICT_7X7_*`, and `DICT_APRILTAG_*`.

## Next step

Generating the JSON file is only step 1 of registering a new family. To make
it usable from Python you also need to add the variant to the `TagFamily`
enum and wire the dictionary lookup. The full procedure is documented in
[How-to: Add a custom fiducial dictionary](../../docs/how-to/add_dictionary.md).
