# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check pixi cu13 cuda-version pins track ci/versions.yml (cuda.build.version)."""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
VERSIONS_FILE_PATH = ROOT / "ci" / "versions.yml"
PIXI_FILES = [ROOT / d / "pixi.toml" for d in ("cuda_bindings", "cuda_core")]


def main() -> int:
    """Verify cuda_bindings/cuda_core pixi cu13 pins match ci/versions.yml."""
    if not VERSIONS_FILE_PATH.is_file():
        print(f"error: {VERSIONS_FILE_PATH} not found", file=sys.stderr)
        return 2
    try:
        build_version = yaml.safe_load(VERSIONS_FILE_PATH.read_text(encoding="utf-8"))["cuda"]["build"]["version"]
    except (KeyError, TypeError):
        print(f"error: cuda.build.version not found in {VERSIONS_FILE_PATH}", file=sys.stderr)
        return 2

    major, minor, *_ = build_version.split(".")
    expected = f"{major}.{minor}.*"
    errors: list[str] = []
    for path in PIXI_FILES:
        if not path.is_file():
            print(f"error: {path} not found", file=sys.stderr)
            return 2
        with path.open("rb") as f:
            data = tomllib.load(f)
        rel = path.relative_to(ROOT)
        try:
            variants = data["workspace"]["build-variants"]["cuda-version"]
            cu13 = data["feature"]["cu13"]["dependencies"]["cuda-version"]
        except KeyError as exc:
            print(f"error: {rel} missing cuda-version key: {exc}", file=sys.stderr)
            return 2
        if expected not in variants:
            errors.append(
                f"{rel}: [workspace.build-variants] cuda-version={variants!r} "
                f"does not include {expected!r} "
                f"(from ci/versions.yml cuda.build.version={build_version!r})"
            )
        if cu13 != expected:
            errors.append(
                f"{rel}: [feature.cu13.dependencies] cuda-version={cu13!r} "
                f"!= {expected!r} "
                f"(from ci/versions.yml cuda.build.version={build_version!r})"
            )

    if errors:
        print(
            f"error: cuda_bindings/cuda_core pixi cuda-version pins out of sync with "
            f"ci/versions.yml cuda.build.version={build_version!r} "
            f"(expected pin {expected!r}):",
            file=sys.stderr,
        )
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    print(f"OK: pixi cuda-version pins match ci/versions.yml ({expected!r})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
