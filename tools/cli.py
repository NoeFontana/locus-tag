import json
import sys
from pathlib import Path
from typing import List

import typer
import jsonschema

app = typer.Typer(help="Locus Developer CLI")

@app.command()
def validate_dicts(
    schema: Path = typer.Option(..., help="Path to JSON schema file"),
    files: List[Path] = typer.Argument(..., help="JSON files to validate"),
):
    """
    Validate dictionary JSON files against a schema.
    """
    try:
        with open(schema) as f:
            schema_data = json.load(f)
    except Exception as e:
        typer.echo(f"Error loading schema {schema}: {e}", err=True)
        raise typer.Exit(code=1)

    has_errors = False

    for file_path in files:
        try:
            with open(file_path) as f:
                data = json.load(f)
            jsonschema.validate(instance=data, schema=schema_data)
            
            # Mathematical consistency checks
            if len(data["canonical_sampling_points"]) != data["payload_length"]:
                typer.echo(
                    f"FAILED: {file_path} - length of canonical_sampling_points ({len(data['canonical_sampling_points'])}) does not match payload_length ({data['payload_length']})",
                    err=True
                )
                has_errors = True
            elif len(data["base_codes"]) != data["dictionary_size"]:
                typer.echo(
                    f"FAILED: {file_path} - length of base_codes ({len(data['base_codes'])}) does not match dictionary_size ({data['dictionary_size']})",
                    err=True
                )
                has_errors = True
            else:
                typer.echo(f"PASS: {file_path}")
        except jsonschema.exceptions.ValidationError as e:
            typer.echo(f"FAILED: {file_path} - Schema validation error: {e.message}", err=True)
            has_errors = True
        except Exception as e:
            typer.echo(f"FAILED: {file_path} - {e}", err=True)
            has_errors = True

    if has_errors:
        raise typer.Exit(code=1)
    else:
        typer.echo("All dictionaries passed validation.")

if __name__ == "__main__":
    app()
