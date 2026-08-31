# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check pixi cuda-version pins track the current CUDA bindings line."""

from __future__ import annotations

import sys
from pathlib import Path

import bindings_config
import tomllib

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    """Verify cuda_bindings/cuda_core pixi pins match ci/versions.yml."""
    try:
        current_line = bindings_config.load_config().line_for_role("current")
    except bindings_config.BindingsConfigError as error:
        print(f"error: invalid CUDA bindings configuration: {error}", file=sys.stderr)
        return 2
    toolkit_version = current_line.toolkit_version
    pixi_files = [ROOT / current_line.source_dir / "pixi.toml", ROOT / "cuda_core" / "pixi.toml"]

    major, minor, *_ = toolkit_version.split(".")
    expected = f"{major}.{minor}.*"
    cuda_feature = f"cu{major}"

    errors: list[str] = []
    checked: list[str] = []
    for path in pixi_files:
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
                f"(from current CUDA bindings toolkit_version={toolkit_version!r})"
            )
        if cuda_pin != expected:
            errors.append(
                f"{rel}: feature.{cuda_feature}.dependencies.cuda-version={cuda_pin!r} "
                f"!= {expected!r} "
                f"(from current CUDA bindings toolkit_version={toolkit_version!r})"
            )

        checked.append(
            f"{rel} (workspace.build-variants.cuda-version={variants!r}, "
            f"feature.{cuda_feature}.dependencies.cuda-version={cuda_pin!r})"
        )

    if errors:
        print(
            f"error: cuda_bindings/cuda_core pixi cuda-version pins out of sync with "
            f"the current CUDA bindings toolkit_version={toolkit_version!r} "
            f"(expected pin {expected!r}):",
            file=sys.stderr,
        )
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    print(
        f"OK: pixi cuda-version pins match the current CUDA bindings "
        f"toolkit_version={toolkit_version!r} (expected pin {expected!r}):"
    )
    for item in checked:
        print(f"  - {item}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
