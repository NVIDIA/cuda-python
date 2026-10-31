# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check pixi cuda-version pins track ci/versions.yml (cuda.build.version)."""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

from .bindings_config import load_config

ROOT = Path(__file__).resolve().parents[2]
VERSIONS_FILE_PATH = ROOT / "ci" / "versions.yml"


def main() -> int:
    """Verify current bindings and cuda_core pixi pins match ci/versions.yml."""
    if not VERSIONS_FILE_PATH.is_file():
        print(f"error: {VERSIONS_FILE_PATH} not found", file=sys.stderr)
        return 2
    try:
        current = load_config(VERSIONS_FILE_PATH).package_for_release_status("current")
    except (KeyError, TypeError, ValueError) as exc:
        print(f"error: invalid CUDA toolkit pins in {VERSIONS_FILE_PATH}: {exc}", file=sys.stderr)
        return 2

    build_version = current.toolkit_version
    expected = f"{current.ctk_target}.*"
    cuda_feature = current.cuda_variant

    errors: list[str] = []
    checked: list[str] = []
    for package_root in (current.package_root, "cuda_core"):
        path = ROOT / package_root / "pixi.toml"
        if not path.is_file():
            print(f"error: {path} not found", file=sys.stderr)
            return 2
        with path.open("rb") as f:
            data = tomllib.load(f)
        rel = path.relative_to(ROOT)
        try:
            variants = data["workspace"]["build-variants"]["cuda-version"]
            cuda_pin = data["feature"][cuda_feature]["dependencies"]["cuda-version"]
        except KeyError as exc:
            print(
                f"error: {rel} missing feature {cuda_feature!r} or cuda-version key: {exc}",
                file=sys.stderr,
            )
            return 2
        if expected not in variants:
            errors.append(
                f"{rel}: workspace.build-variants.cuda-version={variants!r} "
                f"does not include {expected!r} "
                f"(from ci/versions.yml cuda.build.version={build_version!r})"
            )
        if cuda_pin != expected:
            errors.append(
                f"{rel}: feature.{cuda_feature}.dependencies.cuda-version={cuda_pin!r} "
                f"!= {expected!r} "
                f"(from ci/versions.yml cuda.build.version={build_version!r})"
            )

        checked.append(
            f"{rel} (workspace.build-variants.cuda-version={variants!r}, "
            f"feature.{cuda_feature}.dependencies.cuda-version={cuda_pin!r})"
        )

    if errors:
        print(
            f"error: {current.package_root}/cuda_core pixi cuda-version pins out of sync with "
            f"ci/versions.yml cuda.build.version={build_version!r} "
            f"(expected pin {expected!r}):",
            file=sys.stderr,
        )
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    print(
        f"OK: pixi cuda-version pins match ci/versions.yml "
        f"cuda.build.version={build_version!r} (expected pin {expected!r}):"
    )
    for item in checked:
        print(f"  - {item}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
