#!/usr/bin/env python3
"""Migrate one explicitly hash-bound legacy genome artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Sequence

from genetic_algorithm.genome.migration import migrate_genome_artifact_bytes


def _write_once_atomic(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite migration output: {path}")
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temp_path, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Migrate a legacy population/HOF into a semantic strategy-gene-v2 seed artifact"
        )
    )
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--source-sha256",
        required=True,
        help="Expected lowercase SHA-256 of the exact legacy source bytes",
    )
    args = parser.parse_args(argv)

    try:
        if (
            len(args.source_sha256) != 64
            or args.source_sha256.lower() != args.source_sha256
            or any(character not in "0123456789abcdef" for character in args.source_sha256)
        ):
            raise ValueError("--source-sha256 must be a lowercase SHA-256 digest")
        content = args.source.read_bytes()
        artifact, report = migrate_genome_artifact_bytes(
            content,
            expected_sha256=args.source_sha256,
        )
        encoded = (
            json.dumps(
                artifact,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        ).encode()
        _write_once_atomic(args.output, encoded)
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    print(f"Migrated: {args.output}")
    print(f"Artifact SHA-256: {hashlib.sha256(encoded).hexdigest()}")
    print(f"Migration report: {report.report_hash}")
    print(f"Entries: {report.entry_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
