# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check bindings and Core Pixi toolkit pins track ci/versions.yml."""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

from .bindings_config import load_config

ROOT = Path(__file__).resolve().parents[2]
VERSIONS_FILE_PATH = ROOT / "ci" / "versions.yml"


def main() -> int:
    """Verify both bindings roots and Core use their configured toolkit line."""
    if not VERSIONS_FILE_PATH.is_file():
        print(f"error: {VERSIONS_FILE_PATH} not found", file=sys.stderr)
        return 2
    try:
        config = load_config(VERSIONS_FILE_PATH)
    except (KeyError, TypeError, ValueError) as exc:
        print(f"error: invalid CUDA toolkit pins in {VERSIONS_FILE_PATH}: {exc}", file=sys.stderr)
        return 2

    current = config.package_for_release_status("current")
    checks = [(package.package_root, package) for package in config.package_roots]
    checks.append(("cuda_core", current))
    errors: list[str] = []
    checked: list[str] = []
    for package_root, package in checks:
        allowed = {f"{package.ctk_target}.*"}
        if package.release_status == "maintenance":
            allowed.add(f"{package.cuda_major}.*")
        cuda_feature = package.cuda_variant
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
        if not allowed.intersection(variants):
            errors.append(
                f"{rel}: workspace.build-variants.cuda-version={variants!r} "
                f"does not include a pin for toolkit {package.toolkit_version!r} (allowed {sorted(allowed)!r})"
            )
        if cuda_pin not in allowed:
            errors.append(
                f"{rel}: feature.{cuda_feature}.dependencies.cuda-version={cuda_pin!r} "
                f"does not match toolkit {package.toolkit_version!r} (allowed {sorted(allowed)!r})"
            )

        checked.append(
            f"{rel} (workspace.build-variants.cuda-version={variants!r}, "
            f"feature.{cuda_feature}.dependencies.cuda-version={cuda_pin!r})"
        )

    if errors:
        print("error: Pixi cuda-version pins out of sync with ci/versions.yml:", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    print("OK: Pixi cuda-version pins match ci/versions.yml:")
    for item in checked:
        print(f"  - {item}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
