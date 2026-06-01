#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from urllib.parse import unquote

CUDA_PYTHON_URL_RE = re.compile(
    r"https://github\.com/NVIDIA/cuda-python/"
    r"(?P<kind>blob|tree)/"
    r"(?P<ref>[^/\s<>`]+)/"
    r"(?P<path>[^\s<>`)]+)"
)
SOURCE_SUFFIXES = {".md", ".rst"}


def _source_files(source_dir: Path):
    for path in sorted(source_dir.rglob("*")):
        if path.suffix in SOURCE_SUFFIXES and path.is_file():
            yield path


def _normalize_url_path(url_path: str) -> str:
    path = unquote(url_path)
    for separator in ("#", "?"):
        path = path.split(separator, 1)[0]
    return path.rstrip(".")


def _is_within(path: str, root: str) -> bool:
    return path == root or path.startswith(f"{root}/")


def _display_path(path: Path, repo_root: Path) -> str:
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


def check_links(args: argparse.Namespace) -> int:
    repo_root = args.repo_root.resolve()
    source_dir = args.source_dir.resolve()
    examples_root = args.examples_root.strip("/")
    placeholder_ref = f"|{args.placeholder}|" if args.placeholder else None
    checked = 0
    failures: list[str] = []

    for source_path in _source_files(source_dir):
        text = source_path.read_text(encoding="utf-8")
        for match in CUDA_PYTHON_URL_RE.finditer(text):
            url_path = _normalize_url_path(match.group("path"))
            if not _is_within(url_path, examples_root):
                continue

            checked += 1
            ref = match.group("ref")
            kind = match.group("kind")
            location = _display_path(source_path, repo_root)
            target_path = Path(url_path)
            target = repo_root / target_path

            if target_path.is_absolute() or ".." in target_path.parts:
                failures.append(f"{location}: invalid repository path in {match.group(0)}")
                continue

            if placeholder_ref and ref == placeholder_ref:
                rendered_ref = args.expected_ref
            elif ref == args.expected_ref:
                rendered_ref = ref
            else:
                expected = placeholder_ref or args.expected_ref
                failures.append(f"{location}: {match.group(0)} uses ref {ref!r}; expected {expected!r}")
                rendered_ref = ref

            if kind == "blob" and not target.is_file():
                failures.append(
                    f"{location}: {match.group(0)} resolves to missing file "
                    f"{target.relative_to(repo_root)} at ref {rendered_ref}"
                )
            elif kind == "tree" and not target.is_dir():
                failures.append(
                    f"{location}: {match.group(0)} resolves to missing directory "
                    f"{target.relative_to(repo_root)} at ref {rendered_ref}"
                )

    if checked == 0 and not args.allow_empty:
        failures.append(f"No example links under {examples_root!r} found in {source_dir}")

    if failures:
        sys.stderr.write("Example link check failed:\n")
        for failure in failures:
            sys.stderr.write(f"  - {failure}\n")
        return 1

    sys.stdout.write(f"Checked {checked} example link(s) under {examples_root} against ref {args.expected_ref}\n")
    return 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]

    parser = argparse.ArgumentParser(description="Validate cuda-python example links without probing GitHub.")
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--examples-root", required=True)
    parser.add_argument("--expected-ref", required=True)
    parser.add_argument("--placeholder")
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument("--allow-empty", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    return check_links(args)


if __name__ == "__main__":
    raise SystemExit(main())
