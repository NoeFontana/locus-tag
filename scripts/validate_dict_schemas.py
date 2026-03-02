#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import jsonschema


def parse_args():
    parser = argparse.ArgumentParser(description="Validate dictionary JSON files against a schema.")
    parser.add_argument("--schema", required=True, type=Path, help="Path to JSON schema file")
    parser.add_argument("files", nargs="+", type=Path, help="JSON files to validate")
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        with open(args.schema) as f:
            schema = json.load(f)
    except Exception as e:
        print(f"Error loading schema {args.schema}: {e}")
        sys.exit(1)

    has_errors = False

    for file_path in args.files:
        try:
            with open(file_path) as f:
                data = json.load(f)
            jsonschema.validate(instance=data, schema=schema)
            # Add mathematical checks requested
            if len(data["canonical_sampling_points"]) != data["payload_length"]:
                print(
                    f"FAILED: {file_path} - length of canonical_sampling_points ({len(data['canonical_sampling_points'])}) does not match payload_length ({data['payload_length']})"
                )
                has_errors = True
            elif len(data["base_codes"]) != data["dictionary_size"]:
                print(
                    f"FAILED: {file_path} - length of base_codes ({len(data['base_codes'])}) does not match dictionary_size ({data['dictionary_size']})"
                )
                has_errors = True
            else:
                print(f"PASS: {file_path}")
        except jsonschema.exceptions.ValidationError as e:
            print(f"FAILED: {file_path} - Schema validation error: {e.message}")
            has_errors = True
        except Exception as e:
            print(f"FAILED: {file_path} - {e}")
            has_errors = True

    if has_errors:
        sys.exit(1)
    else:
        print("All dictionaries passed validation.")
        sys.exit(0)


if __name__ == "__main__":
    main()
