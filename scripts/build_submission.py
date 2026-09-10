"""Build the Gradescope submission zip: every controller file plus pyproject.toml.

Runs ``scripts/export_student_controllers.py --all-controllers`` (so the
``controllers/`` package, including ``formula110-submission.json``, is at the
archive root) and then adds the project's ``pyproject.toml`` at the root.

Usage:
    uv run python scripts/build_submission.py
    uv run python scripts/build_submission.py --output artifacts/my-submission.zip
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import export_student_controllers as exporter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts" / "formula110-submission.zip"
ROOT_FILES = ("pyproject.toml",)


def build_submission(output: Path) -> Path:
    written = exporter.export_controllers((), output, all_controllers=True)
    with zipfile.ZipFile(written, "a", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name in ROOT_FILES:
            archive.write(PROJECT_ROOT / name, arcname=name)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = build_submission(args.output)
    with zipfile.ZipFile(output) as archive:
        names = archive.namelist()
    print(f"wrote {output} ({len(names)} files):")
    for name in names:
        print(f"  {name}")


if __name__ == "__main__":
    main()
