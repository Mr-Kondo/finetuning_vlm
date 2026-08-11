#!/usr/bin/env python
"""Structural validation for this repository's Jupyter notebooks.

This is a lightweight, offline structural check -- it does NOT execute
notebooks. It only verifies that each notebook file is parseable and
internally consistent enough to trust as a checked-in artifact. Full
top-to-bottom execution validation is a separate, heavier concern owned
by whichever task authors each notebook.

Checks performed per notebook:

1. The file parses as valid JSON and conforms to the nbformat schema
   (via ``nbformat.read`` + ``nbformat.validate``, with
   ``relax_add_props=True`` so a kernel-added extra field -- e.g. Google
   Colab's kernel adding a ``metadata`` key to stream outputs -- doesn't
   fail validation on its own; missing/misshapen *required* structure
   still does).
2. No code cell has non-empty ``outputs`` while its ``execution_count``
   is null/missing. A code cell with real outputs should also carry the
   execution count that produced them; outputs without an execution
   count are a sign of hand-inserted or corrupted cell state rather than
   a real kernel run.

Usage:
    .venv/bin/python scripts/validate_notebooks.py
    .venv/bin/python scripts/validate_notebooks.py notebooks/01_setup.ipynb notebooks/02_data.ipynb

With no arguments, all ``notebooks/*.ipynb`` files are discovered and
validated. Exits 0 if every discovered notebook is valid (including the
case where no notebooks exist yet), and 1 if any notebook fails
validation -- so this script can act as a CI-style gate later.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nbformat

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_NOTEBOOKS_DIR = REPO_ROOT / "notebooks"


def find_default_notebooks() -> list[Path]:
    """Return all *.ipynb files under notebooks/, sorted for stable output."""
    if not DEFAULT_NOTEBOOKS_DIR.is_dir():
        return []
    return sorted(DEFAULT_NOTEBOOKS_DIR.glob("*.ipynb"))


def find_cells_with_outputs_but_no_execution_count(notebook: nbformat.NotebookNode) -> list[int]:
    """Return the indices of code cells that have outputs but no execution count."""
    suspicious_cell_indices = []
    for index, cell in enumerate(notebook.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        has_outputs = bool(cell.get("outputs"))
        has_execution_count = cell.get("execution_count") is not None
        if has_outputs and not has_execution_count:
            suspicious_cell_indices.append(index)
    return suspicious_cell_indices


def validate_notebook(path: Path) -> list[str]:
    """Validate a single notebook, returning a list of human-readable errors.

    An empty list means the notebook is structurally valid.
    """
    errors: list[str] = []

    try:
        notebook = nbformat.read(path, as_version=4)
    except Exception as exc:  # noqa: BLE001 - report any parse failure as a validation error
        errors.append(f"failed to parse as a notebook: {exc}")
        return errors

    try:
        # relax_add_props=True: some kernels (observed: Google Colab's) add extra,
        # harmless fields to cell outputs -- e.g. a stream output's `metadata`
        # key, which strict nbformat v4 does not define for that output type.
        # Reject unknown *required* structure, but don't fail a real, executed
        # notebook over an extra field a legitimate kernel added.
        nbformat.validate(notebook, relax_add_props=True)
    except nbformat.ValidationError as exc:
        errors.append(f"failed nbformat schema validation: {exc}")

    suspicious_cell_indices = find_cells_with_outputs_but_no_execution_count(notebook)
    if suspicious_cell_indices:
        errors.append(
            "code cell(s) at index "
            f"{suspicious_cell_indices} have non-empty outputs but a null/missing "
            "execution_count (looks like hand-inserted or corrupted output state, "
            "not a real kernel run)"
        )

    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "notebooks",
        nargs="*",
        type=Path,
        help="Specific notebook paths to validate. Defaults to all notebooks/*.ipynb.",
    )
    args = parser.parse_args(argv)

    notebook_paths = args.notebooks if args.notebooks else find_default_notebooks()

    if not notebook_paths:
        print(f"No notebooks found under {DEFAULT_NOTEBOOKS_DIR}. Nothing to validate.")
        return 0

    any_failed = False
    for path in notebook_paths:
        errors = validate_notebook(path)
        if errors:
            any_failed = True
            print(f"INVALID  {path}")
            for error in errors:
                print(f"         - {error}")
        else:
            print(f"OK       {path}")

    return 1 if any_failed else 0


if __name__ == "__main__":
    sys.exit(main())
