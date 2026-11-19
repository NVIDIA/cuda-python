# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check release-note pairing and documentation version selectors."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
METAPACKAGE_NOTES = "cuda_python/docs/source/release"
RELEASE_NOTE = re.compile(
    r"(cuda_bindings_[^/]+|cuda_python|cuda_core|cuda_pathfinder)"
    r"/docs/source/release/([0-9]+\.[0-9]+\.[0-9]+)-notes\.rst"
)
SELECTOR_URLS = {
    "cuda_python": "https://nvidia.github.io/cuda-python/",
    "cuda_core": "https://nvidia.github.io/cuda-python/cuda-core/",
    "cuda_pathfinder": "https://nvidia.github.io/cuda-python/cuda-pathfinder/",
    "cuda_bindings": "https://nvidia.github.io/cuda-python/cuda-bindings/",
}


def _requires_pairing(version: str) -> bool:
    """Enforce pairing starting from releases 12.9.11 and 13.5.1."""
    major, minor, patch = map(int, version.split("."))
    return (major == 12 and (minor, patch) >= (9, 11)) or (major == 13 and (minor, patch) >= (5, 1)) or major >= 14


def _build_indexed_release_notes(repo_root: Path) -> dict[str, tuple[str, str]]:
    """Collect the package and version of each release note covered by the checks."""
    listing = subprocess.run(  # noqa: S603
        [  # noqa: S607
            "git",
            "ls-files",
            "-z",
            "--",
            "cuda_bindings_*/docs/source/release/*-notes.rst",
            f"{METAPACKAGE_NOTES}/*-notes.rst",
            "cuda_core/docs/source/release/*-notes.rst",
            "cuda_pathfinder/docs/source/release/*-notes.rst",
        ],
        cwd=repo_root,
        stdout=subprocess.PIPE,
        check=True,
        encoding="utf-8",
    ).stdout
    indexed_release_notes: dict[str, tuple[str, str]] = {}
    for path in listing.split("\0"):
        if match := RELEASE_NOTE.fullmatch(path):
            package, version = match.groups()
            if package == "cuda_core":
                required = tuple(map(int, version.split("."))) >= (1, 3, 2)
            elif package == "cuda_pathfinder":
                required = tuple(map(int, version.split("."))) >= (1, 9, 2)
            else:
                required = _requires_pairing(version)
            if required:
                indexed_release_notes[path] = package, version
    return indexed_release_notes


def missing_release_note_pairs(indexed_release_notes: dict[str, tuple[str, str]]) -> list[str]:
    """Return missing release-note counterparts in the Git index."""
    missing: set[str] = set()
    for package, version in indexed_release_notes.values():
        if package.startswith("cuda_bindings_"):
            counterpart = f"{METAPACKAGE_NOTES}/{version}-notes.rst"
            if counterpart not in indexed_release_notes:
                missing.add(counterpart)

    for package, version in indexed_release_notes.values():
        if package == "cuda_python":
            major = int(version.split(".", 1)[0])
            counterpart = f"cuda_bindings_{major}/docs/source/release/{version}-notes.rst"
            if counterpart not in indexed_release_notes:
                missing.add(counterpart)
    return sorted(missing)


def _bindings_selector_package(repo_root: Path) -> str:
    listing = subprocess.run(
        ["git", "ls-files", "-z", "--", "cuda_bindings_*/*"],  # noqa: S607
        cwd=repo_root,
        stdout=subprocess.PIPE,
        check=True,
        encoding="utf-8",
    ).stdout
    packages = {path.split("/", 1)[0] for path in listing.split("\0") if path}
    ordered_packages = sorted(packages, key=lambda package: int(package.removeprefix("cuda_bindings_")))
    # Expect roots for maintenance, current, and optionally a newer unreleased major.
    # Fail on an unexpected count to avoid silently choosing the wrong selector.
    assert len(ordered_packages) in (2, 3), f"Expected two or three bindings roots, found {ordered_packages}"
    return ordered_packages[1]


def version_selector_errors(repo_root: Path, indexed_release_notes: dict[str, tuple[str, str]]) -> list[str]:
    """Require matching versions and documentation URLs in the staged selectors."""
    bindings_selector = _bindings_selector_package(repo_root)
    selector_urls = {
        bindings_selector if package == "cuda_bindings" else package: url for package, url in SELECTOR_URLS.items()
    }
    selectors: dict[str, dict[str, str]] = {}
    for package in selector_urls:
        path = f"{package}/docs/nv-versions.json"
        contents = subprocess.run(  # noqa: S603
            ["git", "show", f":{path}"],  # noqa: S607
            cwd=repo_root,
            stdout=subprocess.PIPE,
            check=True,
            encoding="utf-8",
        ).stdout
        selectors[package] = {entry["version"]: entry["url"] for entry in json.loads(contents)}

    errors = []
    for note, (package, version) in sorted(indexed_release_notes.items()):
        selector_package = bindings_selector if package.startswith("cuda_bindings_") else package
        path = f"{selector_package}/docs/nv-versions.json"
        entries = selectors[selector_package]
        expected_url = f"{selector_urls[selector_package]}{version}/"
        if version not in entries:
            errors.append(f"{path}: missing version {version} (required by {note})")
        elif entries[version] != expected_url:
            errors.append(f"{path}: version {version} URL is {entries[version]!r}; expected {expected_url!r} ({note})")
    return errors


def main() -> int:
    """Report missing release-note pairs and incorrect staged version selectors."""
    indexed_release_notes = _build_indexed_release_notes(ROOT)
    errors = missing_release_note_pairs(indexed_release_notes) + version_selector_errors(ROOT, indexed_release_notes)
    if not errors:
        return 0

    print("Release-note checks failed in the Git index:", file=sys.stderr)
    for error in errors:
        print(f"  - {error}", file=sys.stderr)
    print("Add and stage matching release notes and documentation version entries.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
