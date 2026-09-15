# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check pixi cuda-version pins track every registered CUDA bindings package."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import tomllib

from . import bindings_config

ROOT = Path(__file__).resolve().parents[2]


class PixiConfigError(ValueError):
    """A pixi manifest cannot be interpreted by this check."""


def _load_pixi(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise PixiConfigError(f"{path.relative_to(ROOT)} not found")
    try:
        with path.open("rb") as file:
            data = tomllib.load(file)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise PixiConfigError(f"could not read {path.relative_to(ROOT)}: {error}") from error
    return data


def _cuda_pins(data: dict[str, Any], path: Path, cuda_feature: str) -> tuple[list[str], str]:
    rel = path.relative_to(ROOT)
    try:
        variants = data["workspace"]["build-variants"]["cuda-version"]
        cuda_pin = data["feature"][cuda_feature]["dependencies"]["cuda-version"]
    except (KeyError, TypeError) as error:
        raise PixiConfigError(f"{rel} missing feature {cuda_feature!r} or cuda-version key: {error}") from error
    if not isinstance(variants, list) or not all(isinstance(item, str) for item in variants):
        raise PixiConfigError(f"{rel} workspace.build-variants.cuda-version must be a list of strings")
    if not isinstance(cuda_pin, str):
        raise PixiConfigError(f"{rel} feature.{cuda_feature}.dependencies.cuda-version must be a string")
    return variants, cuda_pin


def _pin_covers_toolkit(cuda_pin: str, toolkit_version: str) -> bool:
    if not cuda_pin.endswith(".*"):
        return False
    prefix = cuda_pin.removesuffix(".*").split(".")
    if not 1 <= len(prefix) <= 2 or not all(part.isdigit() for part in prefix):
        return False
    return toolkit_version.split(".")[: len(prefix)] == prefix


def _check_pixi_package(
    path: Path,
    data: dict[str, Any],
    package: bindings_config.BindingsPackage,
    errors: list[str],
    checked: list[str],
) -> None:
    rel = path.relative_to(ROOT)
    variants, cuda_pin = _cuda_pins(data, path, package.cuda_variant)
    context = f"registered package root {package.package_root!r} toolkit_version={package.toolkit_version!r}"
    if cuda_pin not in variants:
        errors.append(
            f"{rel}: workspace.build-variants.cuda-version={variants!r} does not include "
            f"feature.{package.cuda_variant} pin {cuda_pin!r} ({context})"
        )
    if not _pin_covers_toolkit(cuda_pin, package.toolkit_version):
        errors.append(
            f"{rel}: feature.{package.cuda_variant}.dependencies.cuda-version={cuda_pin!r} does not cover {context}"
        )
    checked.append(
        f"{rel}: {package.package_root} uses feature.{package.cuda_variant}={cuda_pin!r} "
        f"(workspace variants={variants!r})"
    )


def main() -> int:
    """Verify bindings and core pixi pins match all packages in ci/versions.yml."""
    try:
        config = bindings_config.load_config()
    except bindings_config.BindingsConfigError as error:
        print(f"error: invalid CUDA bindings configuration: {error}", file=sys.stderr)
        return 2

    errors: list[str] = []
    checked: list[str] = []
    try:
        core_path = ROOT / "cuda_core" / "pixi.toml"
        core_data = _load_pixi(core_path)
        for package in config.package_roots:
            source_path = ROOT / package.package_root / "pixi.toml"
            _check_pixi_package(source_path, _load_pixi(source_path), package, errors, checked)
            _check_pixi_package(core_path, core_data, package, errors, checked)
    except PixiConfigError as error:
        print(f"error: invalid pixi configuration: {error}", file=sys.stderr)
        return 2

    if errors:
        print(
            "error: cuda_bindings/cuda_core pixi cuda-version pins are out of sync "
            "with registered CUDA bindings package roots:",
            file=sys.stderr,
        )
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    print("OK: pixi cuda-version pins cover every registered CUDA bindings package root:")
    for item in checked:
        print(f"  - {item}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
